#include <thrust/reduce.h>

#include "modularity.cuh"

struct deg {
    unsigned int *vertndx;
    unsigned int lb_deg, ub_deg;

    deg(unsigned int *_vertndx, unsigned int _lb_deg, unsigned int _ub_deg) :
            vertndx(_vertndx),
            lb_deg(_lb_deg),
            ub_deg(_ub_deg) {}

    __device__ bool operator()(const int &v) const {
        unsigned int deg = vertndx[v + 1] - vertndx[v];
        return lb_deg < deg && deg <= ub_deg;
    }
};

template<typename T>
struct square {
    __device__ T operator()(const T &x) const {
        return x * x;
    }
};

static __global__ void sumNeighWeights(Graph graph) {
    unsigned int vertndx = blockIdx.x * blockDim.y + threadIdx.y;

    if (vertndx < graph.vert_sz) {
        unsigned int begin = graph.vertices[vertndx];
        unsigned int deg = graph.vertices[vertndx + 1] - begin;

        unsigned int e_ndx = threadIdx.x;
        float sum = 0.0;
        while (e_ndx < deg) {
            sum += graph.weights[begin + e_ndx];
            e_ndx += blockDim.x;
        }

        for (unsigned int offset = blockDim.x / 2; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(FULL_MASK, sum, offset);
        }

        if (threadIdx.x == 0) {
            graph.neigh_w[vertndx] = sum;
        }
    }
}

static __global__ void graphVertToComm(Graph graph) {
    unsigned int v_ndx = blockIdx.x * blockDim.x + threadIdx.x;
    if (v_ndx < graph.init_sz) {
        graph.init_comm[v_ndx] = graph.comm[graph.init_comm[v_ndx]];
    }
}

template<typename T>
static __device__ void initTable(T *arr, unsigned int arr_sz, T init_val) {
    unsigned int block_dim = blockDim.x * blockDim.y;
    unsigned int thrd_id = threadIdx.x * blockDim.y + threadIdx.y;
    for (unsigned int i = thrd_id; i < arr_sz; i += block_dim) {
        arr[i] = init_val;
    }
    __syncthreads();
}

template<typename T>
static __device__ void initSharedVar(T *var, T val) {
    if (threadIdx.x == 0) {
        *var = val;
    }
}

static __device__ void initHashTables(int *hash_c, float *hash_w, unsigned int hash_sz) {
    unsigned int block_dim = blockDim.x * blockDim.y;
    unsigned int thrd_id = threadIdx.x * blockDim.y + threadIdx.y;
    for (unsigned int i = thrd_id; i < hash_sz; i += block_dim) {
        hash_c[i] = HASH_INIT;
        hash_w[i] = 0;
    }
    __syncthreads();
}

static __device__ bool isLowerIdx(Graph &g, unsigned int vert, unsigned int neigh) {
    unsigned int vert_comm = g.comm[vert];
    unsigned int neigh_comm = g.comm[neigh];

    if (g.comm_sz[vert_comm] == 1 && g.comm_sz[neigh_comm] == 1) {
        return neigh_comm < vert_comm;
    }
    return true;
}

static __global__ void computeMoveGlobal(
        Graph graph,
        const unsigned int *verts,
        unsigned int vert_size,
        unsigned int prime,
        unsigned int vert_hash_sz,
        int *hash_comm,
        float *hash_weight) {
    extern __shared__ int cache_pos[];
    int *cache_comm = cache_pos + blockDim.x;
    float *cache_max = reinterpret_cast<float *>(cache_comm + blockDim.x);
    float *comm_e_w = cache_max + blockDim.x;
    initSharedVar<float>(comm_e_w, (float) 0);
    __syncthreads();

    unsigned int curr_pos;
    float max_mod_gain = (-1) * INFINITY;
    unsigned int deg = 0;
    unsigned int vertndx = blockIdx.x * blockDim.y + threadIdx.y;
    unsigned int v, e, max_e, max_comm;
    if (vertndx < vert_size) {
        int *vert_hash_comm = hash_comm + vertndx * vert_hash_sz;
        float *vert_hash_weight = hash_weight + vertndx * vert_hash_sz;
        initHashTables(vert_hash_comm, vert_hash_weight, vert_hash_sz);
        v = verts[vertndx];
        deg = graph.vertices[v + 1] - graph.vertices[v];
        unsigned int e_ndx = threadIdx.x;
        while (e_ndx < deg) {
            e = graph.edges[graph.vertices[v] + e_ndx];
            unsigned int e_comm = graph.comm[e];
            unsigned int v_comm = graph.comm[v];
            float edge_weight = graph.weights[graph.vertices[v] + e_ndx];

            unsigned int it = 0;
            do {
                curr_pos = hash(e_comm, it, prime);
                it++;
                if (vert_hash_comm[curr_pos] == e_comm) {
                    atomicAdd(vert_hash_weight + curr_pos, edge_weight);
                } else if (vert_hash_comm[curr_pos] == HASH_INIT) {
                    if (atomicCAS(vert_hash_comm + curr_pos, HASH_INIT, (int) e_comm) == HASH_INIT) {
                        atomicAdd(vert_hash_weight + curr_pos, edge_weight);
                    } else if (vert_hash_comm[curr_pos] == e_comm) {
                        atomicAdd(vert_hash_weight + curr_pos, edge_weight);
                    }
                }
            } while (vert_hash_comm[curr_pos] != e_comm);

            if (e_comm == v_comm && v != e) {
                atomicAdd(comm_e_w, edge_weight);
            }

            if (isLowerIdx(graph, v, e)) {
                float v_weights = graph.neigh_w[v];
                float mod_gain = (vert_hash_weight[curr_pos] / graph.all_w)
                                 + (v_weights
                                    * ((graph.comm_w[v_comm] - v_weights) - graph.comm_w[e_comm])
                                    / (2 * graph.all_w * graph.all_w));

                if (mod_gain > max_mod_gain) {
                    max_mod_gain = mod_gain;
                    max_e = e;
                    max_comm = e_comm;
                }
            }

            e_ndx += blockDim.x;
        }
    }
    cache_max[threadIdx.x] = max_mod_gain;
    cache_pos[threadIdx.x] = threadIdx.x;
    cache_comm[threadIdx.x] = max_comm;
    __syncthreads();

    for (unsigned int offset = blockDim.x / 2; offset > 0; offset /= 2) {
        if (threadIdx.x < offset
            && (cache_max[threadIdx.x + offset] > cache_max[threadIdx.x]
                || (cache_max[threadIdx.x + offset] == cache_max[threadIdx.x] &&
                    cache_comm[threadIdx.x + offset] < cache_comm[threadIdx.x]))) {
            cache_max[threadIdx.x] = cache_max[threadIdx.x + offset];
            cache_pos[threadIdx.x] = cache_pos[threadIdx.x + offset];
            cache_comm[threadIdx.x] = cache_comm[threadIdx.x + offset];
        }
        __syncthreads();
    }

    if (vertndx < vert_size && threadIdx.x < deg && cache_pos[0] == threadIdx.x) {
        if (max_mod_gain - (*comm_e_w / graph.all_w) > 0) {
            graph.new_comm[v] = graph.comm[e];
        } else {
            graph.new_comm[v] = graph.comm[v];
        }
    }
}

static __global__ void computeMoveBlock(
        Graph graph,
        const unsigned int *verts,
        unsigned int vert_size,
        unsigned int prime,
        unsigned int vert_hash_sz) {
    extern __shared__ float s[];
    float *hash_weight = s;
    float *vert_hash_weight = s + threadIdx.y * vert_hash_sz;
    int *hash_comm = reinterpret_cast<int *>(hash_weight + blockDim.y * vert_hash_sz);
    int *vert_hash_comm = hash_comm + threadIdx.y * vert_hash_sz;
    initHashTables(hash_comm, hash_weight, blockDim.y * vert_hash_sz);
    int *cache_pos = hash_comm + blockDim.y * vert_hash_sz;
    int *cache_comm = cache_pos + blockDim.x;
    float *cache_max = reinterpret_cast<float *>(cache_comm + blockDim.x);
    float *comm_e_w = cache_max + blockDim.x;
    initSharedVar<float>(comm_e_w, (float) 0);
    __syncthreads();

    unsigned int curr_pos;
    float max_mod_gain = (-1) * INFINITY;
    unsigned int deg = 0;
    unsigned int vertndx = blockIdx.x * blockDim.y + threadIdx.y;
    unsigned int v, e, max_e, max_comm;
    if (vertndx < vert_size) {
        v = verts[vertndx];
        deg = graph.vertices[v + 1] - graph.vertices[v];
        unsigned int e_ndx = threadIdx.x;
        while (e_ndx < deg) {
            e = graph.edges[graph.vertices[v] + e_ndx];
            unsigned int e_comm = graph.comm[e];
            unsigned int v_comm = graph.comm[v];
            float edge_weight = graph.weights[graph.vertices[v] + e_ndx];

            unsigned int it = 0;
            do {
                curr_pos = hash(e_comm, it, prime);
                it++;
                if (vert_hash_comm[curr_pos] == e_comm) {
                    atomicAdd(vert_hash_weight + curr_pos, edge_weight);
                } else if (vert_hash_comm[curr_pos] == HASH_INIT) {
                    if (atomicCAS(vert_hash_comm + curr_pos, HASH_INIT, (int) e_comm) == HASH_INIT) {
                        atomicAdd(vert_hash_weight + curr_pos, edge_weight);
                    } else if (vert_hash_comm[curr_pos] == e_comm) {
                        atomicAdd(vert_hash_weight + curr_pos, edge_weight);
                    }
                }
            } while (vert_hash_comm[curr_pos] != e_comm);

            if (e_comm == v_comm && e != v) {
                atomicAdd(comm_e_w, edge_weight);
            }

            if (isLowerIdx(graph, v, e)) {
                float v_weights = graph.neigh_w[v];
                float mod_gain = (vert_hash_weight[curr_pos] / graph.all_w)
                                 + (v_weights
                                    * ((graph.comm_w[v_comm] - v_weights) - graph.comm_w[e_comm])
                                    / (2 * graph.all_w * graph.all_w));

                if (mod_gain > max_mod_gain) {
                    max_mod_gain = mod_gain;
                    max_e = e;
                    max_comm = e_comm;
                }
            }

            e_ndx += blockDim.x;
        }
    }
    cache_max[threadIdx.x] = max_mod_gain;
    cache_pos[threadIdx.x] = threadIdx.x;
    cache_comm[threadIdx.x] = max_comm;
    __syncthreads();

    for (unsigned int offset = blockDim.x / 2; offset > 0; offset /= 2) {
        if (threadIdx.x < offset
            && (cache_max[threadIdx.x + offset] > cache_max[threadIdx.x]
                || (cache_max[threadIdx.x + offset] == cache_max[threadIdx.x] &&
                    cache_comm[threadIdx.x + offset] < cache_comm[threadIdx.x]))) {
            cache_max[threadIdx.x] = cache_max[threadIdx.x + offset];
            cache_pos[threadIdx.x] = cache_pos[threadIdx.x + offset];
            cache_comm[threadIdx.x] = cache_comm[threadIdx.x + offset];
        }
        __syncthreads();
    }

    if (vertndx < vert_size && threadIdx.x < deg && cache_pos[0] == threadIdx.x) {
        if (max_mod_gain - (*comm_e_w / graph.all_w) > 0) {
            graph.new_comm[v] = graph.comm[e];
        } else {
            graph.new_comm[v] = graph.comm[v];
        }
    }
}

static __global__ void computeMoveWarp(
        Graph graph,
        const unsigned int *verts,
        unsigned int vert_size,
        unsigned int prime,
        unsigned int vert_hash_sz) {
    extern __shared__ float s[];
    float *hash_weight = s;
    float *vert_hash_weight = s + threadIdx.y * vert_hash_sz;
    int *hash_comm = reinterpret_cast<int *>(hash_weight + blockDim.y * vert_hash_sz);
    int *vert_hash_comm = hash_comm + threadIdx.y * vert_hash_sz;
    initHashTables(hash_comm, hash_weight, blockDim.y * vert_hash_sz);
    int *block_max_pos = hash_comm + blockDim.y * vert_hash_sz;
    int *vert_max_pos = block_max_pos + threadIdx.y;
    float *comm_e_w = reinterpret_cast<float *>(block_max_pos + blockDim.y + threadIdx.y);
    initSharedVar<float>(comm_e_w, (float) 0);

    unsigned int curr_pos;
    float max_mod_gain = (-1) * INFINITY;
    unsigned int deg = 0;
    unsigned int vertndx = blockIdx.x * blockDim.y + threadIdx.y;
    unsigned int v, e, max_e, max_comm;
    if (vertndx < vert_size) {
        v = verts[vertndx];
        deg = graph.vertices[v + 1] - graph.vertices[v];
        unsigned int e_ndx = threadIdx.x;
        while (e_ndx < deg) {
            e = graph.edges[graph.vertices[v] + e_ndx];
            unsigned int e_comm = graph.comm[e];
            unsigned int v_comm = graph.comm[v];
            float edge_weight = graph.weights[graph.vertices[v] + e_ndx];

            unsigned int it = 0;
            do {
                curr_pos = hash(e_comm, it, prime);
                it++;
                if (vert_hash_comm[curr_pos] == e_comm) {
                    atomicAdd(vert_hash_weight + curr_pos, edge_weight);
                } else if (vert_hash_comm[curr_pos] == HASH_INIT) {
                    if (atomicCAS(vert_hash_comm + curr_pos, HASH_INIT, (int) e_comm) == HASH_INIT) {
                        atomicAdd(vert_hash_weight + curr_pos, edge_weight);
                    } else if (vert_hash_comm[curr_pos] == e_comm) {
                        atomicAdd(vert_hash_weight + curr_pos, edge_weight);
                    }
                }
            } while (vert_hash_comm[curr_pos] != e_comm);

            if (e_comm == v_comm && e != v) {
                atomicAdd(comm_e_w, edge_weight);
            }

            if (isLowerIdx(graph, v, e)) {
                float v_weights = graph.neigh_w[v];
                float mod_gain = (vert_hash_weight[curr_pos] / graph.all_w)
                                 + (v_weights
                                    * ((graph.comm_w[v_comm] - v_weights) - graph.comm_w[e_comm])
                                    / (2 * graph.all_w * graph.all_w));

                if (mod_gain > max_mod_gain) {
                    max_mod_gain = mod_gain;
                    max_e = e;
                    max_comm = e_comm;
                }
            }

            e_ndx += blockDim.x;
        }
    }

    float tmp_gain;
    float e_mod_gain = max_mod_gain;
    unsigned int tmp_pos, tmp_comm;
    unsigned int pos = threadIdx.x;
    for (unsigned int offset = blockDim.x / 2; offset > 0; offset /= 2) {
        tmp_gain = __shfl_down_sync(FULL_MASK, max_mod_gain, offset);
        tmp_pos = __shfl_down_sync(FULL_MASK, pos, offset);
        tmp_comm = __shfl_down_sync(FULL_MASK, max_comm, offset);
        if (tmp_gain > max_mod_gain || (tmp_gain == max_mod_gain && tmp_comm < max_comm)) {
            max_mod_gain = tmp_gain;
            pos = tmp_pos;
            max_comm = tmp_comm;
        }
    }
    if (threadIdx.x == 0) {
        *vert_max_pos = pos;
    }

    if (vertndx < vert_size && threadIdx.x < deg && *vert_max_pos == threadIdx.x) {
        if (e_mod_gain - (*comm_e_w / graph.all_w) > 0) {
            graph.new_comm[v] = graph.comm[max_e];
        } else {
            graph.new_comm[v] = graph.comm[v];
        }
    }
}

static __global__ void assignCommunity(Graph graph, const unsigned int *verts, unsigned int vert_size) {
    unsigned int v_ndx = blockIdx.x * blockDim.x + threadIdx.x;
    if (v_ndx < vert_size) {
        unsigned int v = verts[v_ndx];
        unsigned int v_new_comm = graph.new_comm[v];
        atomicSub(graph.comm_sz + graph.comm[v], 1);
        atomicAdd(graph.comm_sz + v_new_comm, 1);
        graph.comm[v] = v_new_comm;
    }
}

static __global__ void computeNewCommWeights(Graph graph) {
    unsigned int v_ndx = blockIdx.x * blockDim.x + threadIdx.x;
    if (v_ndx < graph.vert_sz) {
        atomicAdd(graph.comm_w + graph.comm[v_ndx], graph.neigh_w[v_ndx]);
    }
}

static __global__ void computeCommNeighSum(Graph graph, float *sum) {
    unsigned int v_ndx = blockIdx.x * blockDim.y + threadIdx.y;
    unsigned int e_ndx = threadIdx.x;
    float temp = 0;

    if (v_ndx < graph.vert_sz) {
        unsigned int v_e_ndx = graph.vertices[v_ndx];
        unsigned int v_comm = graph.comm[v_ndx];
        unsigned int deg = graph.vertices[v_ndx + 1] - v_e_ndx;
        unsigned int *v_neigh = graph.edges + v_e_ndx;
        float *v_weights = graph.weights + v_e_ndx;
        while (e_ndx < deg) {
            if (graph.comm[v_neigh[e_ndx]] == v_comm) {
                temp += v_weights[e_ndx];
            }
            e_ndx += blockDim.x;
        }

        for (unsigned int offset = blockDim.x / 2; offset > 0; offset /= 2) {
            temp += __shfl_down_sync(FULL_MASK, temp, offset);
        }

        if (threadIdx.x == 0) {
            sum[v_ndx] = temp;
        }
    }
}

static float computeModularity(Graph &dev_graph) {
    computeCommNeighSum<<<(dev_graph.vert_sz + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK, dim3(WARP_SZ, WARPS_PER_BLOCK)>>>(
            dev_graph, dev_graph.modularity);
    float comm_w_sum = thrust::transform_reduce(thrust::device, dev_graph.comm_w, dev_graph.comm_w + dev_graph.vert_sz,
                                                square<float>(), 0.0, thrust::plus<float>());
    float sum_edges_comm_v = thrust::reduce(thrust::device, dev_graph.modularity,
                                            dev_graph.modularity + dev_graph.vert_sz);
    return (sum_edges_comm_v / (2 * dev_graph.all_w)) - (comm_w_sum / (4 * dev_graph.all_w * dev_graph.all_w));
}

bool modularityOptimization(Graph &dev_graph, unsigned int *v_set, float threshold) {
    unsigned int buckets[] = {0, 4, 8, 16, 32, 84, 319, INT_MAX};
    unsigned int prime[] = {7, 13, 29, 53, 127, 479,
                            findNearestPrime((unsigned int) (dev_graph.max_deg * 1.5) + 1)};
    unsigned int vert_hash_sizes[] = {8, 16, 32, 64, 128, 512};
    dim3 blocks_dim[] = {
            {4,   32},
            {8,   16},
            {16,  8},
            {32,  4},
            {32,  4},
            {128, 1},
            {128, 1}};
    size_t bucket_sz = 8;
    int *hash_comm = nullptr;
    float *hash_weights = nullptr;

    sumNeighWeights<<<(dev_graph.vert_sz + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK, dim3(WARP_SZ, WARPS_PER_BLOCK)>>>(
            dev_graph);
    dev_graph.all_w = thrust::reduce(thrust::device, dev_graph.neigh_w, dev_graph.neigh_w + dev_graph.vert_sz,
                                     (float) 0) / 2;
    thrust::sequence(thrust::device, v_set, v_set + dev_graph.vert_sz, 0);
    thrust::sequence(thrust::device, dev_graph.comm, dev_graph.comm + dev_graph.vert_sz, 0);
    thrust::fill(thrust::device, dev_graph.comm_sz, dev_graph.comm_sz + dev_graph.vert_sz, 1);
    thrust::copy(thrust::device, dev_graph.neigh_w, dev_graph.neigh_w + dev_graph.vert_sz, dev_graph.comm_w);
    float act_modularity = computeModularity(dev_graph);
    float old_modularity;
    bool first_loop = true, ret = true;
    do {
        for (int i = 1; i < bucket_sz; i++) {
            unsigned int *v_set_end = thrust::partition(thrust::device, v_set, v_set + dev_graph.vert_sz,
                                                        deg(dev_graph.vertices, buckets[i - 1], buckets[i]));
            unsigned int v_set_sz = v_set_end - v_set;
            if (v_set_sz > 0) {
                unsigned int grid_sz = (v_set_sz + blocks_dim[i - 1].y - 1) / blocks_dim[i - 1].y;
                if (blocks_dim[i - 1].x <= WARP_SZ) {
                    unsigned int hash_sz = vert_hash_sizes[i - 1] * blocks_dim[i - 1].y;
                    unsigned int shmem_sz = (hash_sz + blocks_dim[i - 1].y) * (sizeof(float) + sizeof(int));
                    computeMoveWarp<<<grid_sz, blocks_dim[i - 1], shmem_sz>>>(dev_graph, v_set, v_set_sz, prime[i - 1],
                                                                              vert_hash_sizes[i - 1]);
                } else if (buckets[i] == 319) {
                    unsigned int hash_sz = vert_hash_sizes[i - 1];
                    unsigned int shmem_sz = (hash_sz + blocks_dim[i - 1].x + 1) * sizeof(float) +
                                            (hash_sz + 2 * blocks_dim[i - 1].x) * sizeof(int);
                    computeMoveBlock<<<grid_sz, blocks_dim[i - 1], shmem_sz>>>(dev_graph, v_set, v_set_sz, prime[i - 1],
                                                                               vert_hash_sizes[i - 1]);
                } else {
                    unsigned int hash_sz = prime[i - 1];
                    unsigned int shmem_sz = blocks_dim[i - 1].x * (sizeof(float) + 2 * sizeof(int)) + sizeof(float);
                    if (!hash_comm || !hash_weights) {
                        HANDLE_ERROR(cudaMalloc((void **) &hash_comm, v_set_sz * hash_sz * sizeof(int)));
                        HANDLE_ERROR(cudaMalloc((void **) &hash_weights, v_set_sz * hash_sz * sizeof(float)));
                    }
                    computeMoveGlobal<<<grid_sz, blocks_dim[i - 1], shmem_sz>>>(dev_graph, v_set, v_set_sz,
                                                                                prime[i - 1],
                                                                                hash_sz, hash_comm, hash_weights);
                }
                assignCommunity<<<(v_set_sz + THREADS_N - 1) / THREADS_N, THREADS_N>>>(
                        dev_graph, v_set, v_set_sz);
                thrust::fill(thrust::device, dev_graph.comm_w, dev_graph.comm_w + dev_graph.vert_sz, 0.0);
                computeNewCommWeights<<<(dev_graph.vert_sz + THREADS_N - 1) / THREADS_N, THREADS_N>>>(dev_graph);
            }
        }
        old_modularity = act_modularity;
        act_modularity = computeModularity(dev_graph);
        if (first_loop && act_modularity - old_modularity < threshold) {
            std::cout << old_modularity << std::endl;
            ret = false;
        }
        first_loop = false;
    } while (act_modularity - old_modularity >= threshold);

    if (hash_comm || hash_weights) {
        HANDLE_ERROR(cudaFree(hash_comm));
        HANDLE_ERROR(cudaFree(hash_weights));
    }

    if (ret) {
        graphVertToComm<<<(dev_graph.init_sz + THREADS_N - 1) / THREADS_N, THREADS_N>>>(dev_graph);
    }

    return ret;
}
