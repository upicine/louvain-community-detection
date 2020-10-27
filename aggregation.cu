#include "aggregation.cuh"

#include <thrust/fill.h>
#include <thrust/execution_policy.h>

struct comDeg {
    unsigned int *comm_deg;
    unsigned int lb_deg, ub_deg;

    comDeg(unsigned int *_comm_deg, unsigned int _lb_deg, unsigned int _ub_deg) :
            comm_deg(_comm_deg),
            lb_deg(_lb_deg),
            ub_deg(_ub_deg) {}

    __device__ bool operator()(const int &v) const {
        unsigned int deg = comm_deg[v];
        return lb_deg < deg && deg <= ub_deg;
    }
};

static __global__ void setCommData(Graph dev_graph, CommAggr aggr_data) {
    unsigned int v_ndx = blockIdx.x * blockDim.x + threadIdx.x;

    if (v_ndx < dev_graph.vert_sz) {
        atomicAdd(aggr_data.comm_deg + dev_graph.comm[v_ndx],
                  dev_graph.vertices[v_ndx + 1] - dev_graph.vertices[v_ndx]);

        if (dev_graph.comm_sz[v_ndx] == 0) {
            aggr_data.new_comm_id[v_ndx] = 0;
        } else {
            aggr_data.new_comm_id[v_ndx] = 1;
        }

        aggr_data.new_deg[v_ndx] = 0;
    }
}

static __global__ void vertByCommOrder(Graph dev_graph, CommAggr aggr_data) {
    unsigned int v_ndx = blockIdx.x * blockDim.x + threadIdx.x;

    if (v_ndx < dev_graph.vert_sz) {
        unsigned int res = atomicAdd(aggr_data.vert_start + dev_graph.comm[v_ndx], 1);
        aggr_data.verts_by_comm[res] = v_ndx;
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

static __global__ void mergeCommunityGlobal(
        Graph dev_graph,
        CommAggr aggr_data,
        const unsigned int *comm_set,
        unsigned int comm_set_sz,
        unsigned int prime,
        unsigned int comm_hash_sz,
        int *hash_comm,
        float *hash_weight) {
    extern __shared__ int cache[];

    unsigned int comm_ndx = blockIdx.x * blockDim.y + threadIdx.y;
    int *comm_hash_comm = hash_comm + comm_ndx * comm_hash_sz;
    float *comm_hash_weight = hash_weight + comm_ndx * comm_hash_sz;
    initHashTables(comm_hash_comm, comm_hash_weight, comm_hash_sz);
    if (comm_ndx < comm_set_sz) {
        unsigned int comm_id = comm_set[comm_ndx];
        unsigned int *verts;
        if (comm_id == 0) {
            verts = aggr_data.verts_by_comm;
        } else {
            verts = aggr_data.verts_by_comm + aggr_data.vert_start[comm_id - 1];
        }
        unsigned int v_ndx = 0;
        unsigned int curr_pos;
        unsigned int comm_sz = dev_graph.comm_sz[comm_id];
        unsigned int met_comm_n = 0;
        while (v_ndx < comm_sz) {
            unsigned int e_ndx = threadIdx.x;
            unsigned int v = verts[v_ndx];
            unsigned int arr_pos = dev_graph.vertices[v];
            unsigned int deg = dev_graph.vertices[v + 1] - arr_pos;
            while (e_ndx < deg) {
                unsigned int e = dev_graph.edges[arr_pos + e_ndx];
                unsigned int e_comm = dev_graph.comm[e];
                float w = dev_graph.weights[arr_pos + e_ndx];
                unsigned int it = 0;
                do {
                    curr_pos = hash(e_comm, it, prime);
                    it++;
                    if (comm_hash_comm[curr_pos] == e_comm) {
                        atomicAdd(comm_hash_weight + curr_pos, w);
                    } else if (comm_hash_comm[curr_pos] == HASH_INIT) {
                        if (atomicCAS(comm_hash_comm + curr_pos, HASH_INIT, (int) e_comm) == HASH_INIT) {
                            met_comm_n++;
                            aggr_data.vert_hash_pos[arr_pos + e_ndx] = curr_pos;
                            atomicAdd(comm_hash_weight + curr_pos, w);
                        } else if (comm_hash_comm[curr_pos] == e_comm) {
                            atomicAdd(comm_hash_weight + curr_pos, w);
                        }
                    }
                } while (comm_hash_comm[curr_pos] != e_comm);

                e_ndx += blockDim.x;
            }

            v_ndx++;
        }

        cache[threadIdx.x] = met_comm_n;
        __syncthreads();

        for (unsigned int offset = 1; offset <= blockDim.x / 2; offset *= 2) {
            int tmp;
            if (threadIdx.x >= offset) {
                tmp = cache[threadIdx.x - offset];
            }
            __syncthreads();

            if (threadIdx.x >= offset) {
                cache[threadIdx.x] += tmp;
            }
            __syncthreads();
        }

        if (threadIdx.x == blockDim.x - 1) {
            aggr_data.new_deg[aggr_data.new_comm_id[comm_id] + 1] = cache[threadIdx.x];
        }

        v_ndx = 0;
        unsigned int indx = cache[threadIdx.x] - met_comm_n;
        while (v_ndx < comm_sz) {
            unsigned int e_ndx = threadIdx.x;
            unsigned int v = verts[v_ndx];
            unsigned int arr_pos = dev_graph.vertices[v];
            unsigned int deg = dev_graph.vertices[v + 1] - arr_pos;
            unsigned int *v_hash_pos = aggr_data.vert_hash_pos + arr_pos;
            unsigned int edge_pos = aggr_data.edge_pos[comm_id];
            while (e_ndx < deg) {
                unsigned int pos = v_hash_pos[e_ndx];
                if (pos != HASH_INIT) {
                    aggr_data.new_edges[edge_pos + indx] = comm_hash_comm[pos];
                    aggr_data.new_weights[edge_pos + indx] = comm_hash_weight[pos];
                    indx++;
                }
                e_ndx += blockDim.x;
            }

            v_ndx++;
        }
    }
}

static __global__ void mergeCommunityBlock(
        Graph dev_graph,
        CommAggr aggr_data,
        const unsigned int *comm_set,
        unsigned int comm_set_sz,
        unsigned int prime,
        unsigned int comm_hash_sz) {
    extern __shared__ float s[];
    float *hash_weight = s;
    float *comm_hash_weight = hash_weight + threadIdx.y * comm_hash_sz;
    int *hash_comm = reinterpret_cast<int *>(hash_weight + blockDim.y * comm_hash_sz);
    int *comm_hash_comm = hash_comm + threadIdx.y * comm_hash_sz;
    initHashTables(hash_comm, hash_weight, blockDim.y * comm_hash_sz);
    int *cache = comm_hash_comm + blockDim.y * comm_hash_sz;

    unsigned int comm_ndx = blockIdx.x * blockDim.y + threadIdx.y;
    if (comm_ndx < comm_set_sz) {
        unsigned int comm_id = comm_set[comm_ndx];
        unsigned int *verts;
        if (comm_id == 0) {
            verts = aggr_data.verts_by_comm;
        } else {
            verts = aggr_data.verts_by_comm + aggr_data.vert_start[comm_id - 1];
        }
        unsigned int v_ndx = 0;
        unsigned int curr_pos;
        unsigned int comm_sz = dev_graph.comm_sz[comm_id];
        unsigned int met_comm_n = 0;
        while (v_ndx < comm_sz) {
            unsigned int e_ndx = threadIdx.x;
            unsigned int v = verts[v_ndx];
            unsigned int arr_pos = dev_graph.vertices[v];
            unsigned int deg = dev_graph.vertices[v + 1] - arr_pos;
            while (e_ndx < deg) {
                unsigned int e = dev_graph.edges[arr_pos + e_ndx];
                unsigned int e_comm = dev_graph.comm[e];
                float w = dev_graph.weights[arr_pos + e_ndx];
                unsigned int it = 0;
                do {
                    curr_pos = hash(e_comm, it, prime);
                    it++;
                    if (comm_hash_comm[curr_pos] == e_comm) {
                        atomicAdd(comm_hash_weight + curr_pos, w);
                    } else if (comm_hash_comm[curr_pos] == HASH_INIT) {
                        if (atomicCAS(comm_hash_comm + curr_pos, HASH_INIT, (int) e_comm) == HASH_INIT) {
                            met_comm_n++;
                            aggr_data.vert_hash_pos[arr_pos + e_ndx] = curr_pos;
                            atomicAdd(comm_hash_weight + curr_pos, w);
                        } else if (comm_hash_comm[curr_pos] == e_comm) {
                            atomicAdd(comm_hash_weight + curr_pos, w);
                        }
                    }
                } while (comm_hash_comm[curr_pos] != e_comm);

                e_ndx += blockDim.x;
            }

            v_ndx++;
        }

        cache[threadIdx.x] = met_comm_n;
        __syncthreads();

        for (unsigned int offset = 1; offset <= blockDim.x / 2; offset *= 2) {
            int tmp;
            if (threadIdx.x >= offset) {
                tmp = cache[threadIdx.x - offset];
            }
            __syncthreads();

            if (threadIdx.x >= offset) {
                cache[threadIdx.x] += tmp;
            }
            __syncthreads();
        }

        if (threadIdx.x == blockDim.x - 1) {
            aggr_data.new_deg[aggr_data.new_comm_id[comm_id] + 1] = cache[threadIdx.x];
        }

        v_ndx = 0;
        unsigned int indx = cache[threadIdx.x] - met_comm_n;
        while (v_ndx < comm_sz) {
            unsigned int e_ndx = threadIdx.x;
            unsigned int v = verts[v_ndx];
            unsigned int arr_pos = dev_graph.vertices[v];
            unsigned int deg = dev_graph.vertices[v + 1] - arr_pos;
            unsigned int *v_hash_pos = aggr_data.vert_hash_pos + arr_pos;
            unsigned int edge_pos = aggr_data.edge_pos[comm_id];
            while (e_ndx < deg) {
                unsigned int pos = v_hash_pos[e_ndx];
                if (pos != HASH_INIT) {
                    aggr_data.new_edges[edge_pos + indx] = comm_hash_comm[pos];
                    aggr_data.new_weights[edge_pos + indx] = comm_hash_weight[pos];
                    indx++;
                }
                e_ndx += blockDim.x;
            }

            v_ndx++;
        }
    }
}

static __global__ void mergeCommunityWarp(
        Graph dev_graph,
        CommAggr aggr_data,
        const unsigned int *comm_set,
        unsigned int comm_set_sz,
        unsigned int prime,
        unsigned int comm_hash_sz) {
    extern __shared__ float s[];
    float *hash_weight = s;
    float *comm_hash_weight = hash_weight + threadIdx.y * comm_hash_sz;
    int *hash_comm = reinterpret_cast<int *>(hash_weight + blockDim.y * comm_hash_sz);
    int *comm_hash_comm = hash_comm + threadIdx.y * comm_hash_sz;
    initHashTables(hash_comm, hash_weight, blockDim.y * comm_hash_sz);

    unsigned int comm_ndx = blockIdx.x * blockDim.y + threadIdx.y;
    if (comm_ndx < comm_set_sz) {
        unsigned int comm_id = comm_set[comm_ndx];
        unsigned int *verts;
        if (comm_id == 0) {
            verts = aggr_data.verts_by_comm;
        } else {
            verts = aggr_data.verts_by_comm + aggr_data.vert_start[comm_id - 1];
        }
        unsigned int v_ndx = 0;
        unsigned int curr_pos;
        unsigned int comm_sz = dev_graph.comm_sz[comm_id];
        unsigned int met_comm_n = 0;
        while (v_ndx < comm_sz) {
            unsigned int e_ndx = threadIdx.x;
            unsigned int v = verts[v_ndx];
            unsigned int arr_pos = dev_graph.vertices[v];
            unsigned int deg = dev_graph.vertices[v + 1] - arr_pos;
            while (e_ndx < deg) {
                unsigned int e = dev_graph.edges[arr_pos + e_ndx];
                unsigned int e_comm = dev_graph.comm[e];
                float w = dev_graph.weights[arr_pos + e_ndx];
                unsigned int it = 0;
                do {
                    curr_pos = hash(e_comm, it, prime);
                    it++;
                    if (comm_hash_comm[curr_pos] == e_comm) {
                        atomicAdd(comm_hash_weight + curr_pos, w);
                    } else if (comm_hash_comm[curr_pos] == HASH_INIT) {
                        if (atomicCAS(comm_hash_comm + curr_pos, HASH_INIT, (int) e_comm) == HASH_INIT) {
                            met_comm_n++;
                            aggr_data.vert_hash_pos[arr_pos + e_ndx] = curr_pos;
                            atomicAdd(comm_hash_weight + curr_pos, w);
                        } else if (comm_hash_comm[curr_pos] == e_comm) {
                            atomicAdd(comm_hash_weight + curr_pos, w);
                        }
                    }
                } while (comm_hash_comm[curr_pos] != e_comm);

                e_ndx += blockDim.x;
            }

            v_ndx++;
        }

        unsigned int partial_sum = met_comm_n;
        for (unsigned int offset = 1; offset <= blockDim.x / 2; offset *= 2) {
            unsigned int n = __shfl_up_sync(FULL_MASK, partial_sum, offset);
            if (threadIdx.x >= offset) {
                partial_sum += n;
            }
        }

        if (threadIdx.x == blockDim.x - 1) {
            aggr_data.new_deg[aggr_data.new_comm_id[comm_id] + 1] = partial_sum;
        }

        v_ndx = 0;
        unsigned int indx = partial_sum - met_comm_n;
        while (v_ndx < comm_sz) {
            unsigned int e_ndx = threadIdx.x;
            unsigned int v = verts[v_ndx];
            unsigned int arr_pos = dev_graph.vertices[v];
            unsigned int deg = dev_graph.vertices[v + 1] - arr_pos;
            unsigned int *v_hash_pos = aggr_data.vert_hash_pos + arr_pos;
            unsigned int edge_pos = aggr_data.edge_pos[comm_id];
            while (e_ndx < deg) {
                unsigned int pos = v_hash_pos[e_ndx];
                if (pos != HASH_INIT) {
                    aggr_data.new_edges[edge_pos + indx] = comm_hash_comm[pos];
                    aggr_data.new_weights[edge_pos + indx] = comm_hash_weight[pos];
                    indx++;
                }
                e_ndx += blockDim.x;
            }

            v_ndx++;
        }
    }
}

static __global__ void compressEdges(Graph dev_graph, CommAggr aggr_data) {
    unsigned int comm_ndx = blockIdx.x * blockDim.y + threadIdx.y;

    if (comm_ndx < dev_graph.vert_sz && aggr_data.comm_deg[comm_ndx] > 0) {
        unsigned int new_comm_id = aggr_data.new_comm_id[comm_ndx];
        unsigned int edge_beg = aggr_data.new_deg[new_comm_id];
        unsigned int edge_end = aggr_data.new_deg[new_comm_id + 1];
        unsigned int deg = edge_end - edge_beg;
        unsigned int e_ndx = threadIdx.x;
        unsigned int edge_pos = aggr_data.edge_pos[comm_ndx];
        if (threadIdx.x == 0) {
            dev_graph.vertices[new_comm_id + 1] = edge_end;
        }
        while (e_ndx < deg) {
            dev_graph.edges[edge_beg + e_ndx] = aggr_data.new_comm_id[aggr_data.new_edges[edge_pos + e_ndx]];
            dev_graph.weights[edge_beg + e_ndx] = aggr_data.new_weights[edge_pos + e_ndx];
            e_ndx += blockDim.x;
        }
    }
}

static __global__ void graphVertToNewComm(Graph graph, CommAggr aggr_data) {
    unsigned int v_ndx = blockIdx.x * blockDim.x + threadIdx.x;
    if (v_ndx < graph.init_sz) {
        graph.init_comm[v_ndx] = aggr_data.new_comm_id[graph.init_comm[v_ndx]];
    }
}

void aggregation(Graph &dev_graph, CommAggr &aggr_data, unsigned int *comm_set) {
    thrust::fill(thrust::device, aggr_data.comm_deg, aggr_data.comm_deg + dev_graph.vert_sz, 0);
    setCommData<<<(dev_graph.vert_sz + THREADS_N - 1) / THREADS_N, THREADS_N>>>(dev_graph, aggr_data);
    unsigned int new_vert_sz =
            thrust::reduce(thrust::device, aggr_data.new_comm_id, aggr_data.new_comm_id + dev_graph.vert_sz);
    thrust::exclusive_scan(thrust::device, aggr_data.new_comm_id, aggr_data.new_comm_id + dev_graph.vert_sz,
                           aggr_data.new_comm_id);
    thrust::exclusive_scan(thrust::device, aggr_data.comm_deg, aggr_data.comm_deg + dev_graph.vert_sz,
                           aggr_data.edge_pos);
    thrust::exclusive_scan(thrust::device, dev_graph.comm_sz, dev_graph.comm_sz + dev_graph.vert_sz,
                           aggr_data.vert_start);
    thrust::fill(thrust::device, aggr_data.vert_hash_pos, aggr_data.vert_hash_pos + dev_graph.edges_sz, HASH_INIT);
    vertByCommOrder<<<(dev_graph.vert_sz + THREADS_N - 1) / THREADS_N, THREADS_N>>>(dev_graph, aggr_data);
    thrust::sequence(thrust::device, comm_set, comm_set + dev_graph.vert_sz, 0);
    unsigned int buckets[] = {0, 127, 479, INT_MAX};
    static unsigned int primes[] = {191, 719, findNearestPrime((unsigned int) (new_vert_sz * 1.5) + 1)};
    unsigned int comm_hash_sz[] = {256, 1024};
    dim3 block_dims[] = {{32,  4},
                         {128, 1},
                         {128, 1}};
    unsigned int buckets_sz = 4;

    for (int i = 1; i < buckets_sz; i++) {
        unsigned int *comm_set_end = thrust::partition(thrust::device, comm_set, comm_set + dev_graph.vert_sz,
                                                       comDeg(aggr_data.comm_deg, buckets[i - 1], buckets[i]));
        unsigned int comm_set_sz = comm_set_end - comm_set;
        unsigned int grid_dim = (comm_set_sz + block_dims[i - 1].y - 1) / block_dims[i - 1].y;
        if (comm_set_sz > 0) {
            if (buckets[i] == 127) {
                unsigned int shmem_sz = (block_dims[i - 1].y * comm_hash_sz[i - 1]) * (sizeof(float) + sizeof(int));
                mergeCommunityWarp<<<grid_dim, block_dims[i - 1], shmem_sz>>>(
                        dev_graph,
                        aggr_data,
                        comm_set,
                        comm_set_sz, primes[i - 1],
                        comm_hash_sz[i - 1]);
            } else if (buckets[i] == 479) {
                unsigned int shmem_sz = (block_dims[i - 1].y * comm_hash_sz[i - 1]) * (sizeof(float) + sizeof(int)) +
                                        block_dims[i - 1].x * sizeof(int);
                mergeCommunityBlock<<<grid_dim, block_dims[i - 1], shmem_sz>>>(
                        dev_graph,
                        aggr_data,
                        comm_set,
                        comm_set_sz,
                        primes[i - 1],
                        comm_hash_sz[i - 1]);
            } else {
                unsigned int hash_sz = primes[2];
                unsigned int shmem_sz = block_dims[i - 1].x * sizeof(int);
                HANDLE_ERROR(cudaMalloc((void **) &aggr_data.hash_comm, comm_set_sz * hash_sz * sizeof(int)));
                HANDLE_ERROR(cudaMalloc((void **) &aggr_data.hash_weights, comm_set_sz * hash_sz * sizeof(float)));
                mergeCommunityGlobal<<<grid_dim, block_dims[i - 1], shmem_sz>>>(
                        dev_graph,
                        aggr_data,
                        comm_set,
                        comm_set_sz,
                        primes[i - 1],
                        hash_sz,
                        aggr_data.hash_comm,
                        aggr_data.hash_weights);
                HANDLE_ERROR(cudaFree(aggr_data.hash_comm));
                HANDLE_ERROR(cudaFree(aggr_data.hash_weights));
            }
        }
    }
    unsigned int new_edges_sz = thrust::reduce(thrust::device, aggr_data.new_deg, aggr_data.new_deg + new_vert_sz + 1);
    unsigned int *max_deg_iter = thrust::max_element(thrust::device, aggr_data.new_deg, aggr_data.new_deg + new_vert_sz + 1);
    HANDLE_ERROR(cudaMemcpy(&dev_graph.max_deg, max_deg_iter, sizeof(unsigned int), cudaMemcpyDeviceToHost));
    thrust::inclusive_scan(thrust::device, aggr_data.new_deg + 1, aggr_data.new_deg + new_vert_sz + 1,
            aggr_data.new_deg + 1);
    compressEdges<<<(dev_graph.vert_sz + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK, dim3(WARP_SZ, WARPS_PER_BLOCK)>>>(
            dev_graph, aggr_data);
    graphVertToNewComm<<<(dev_graph.init_sz + THREADS_N - 1) / THREADS_N, THREADS_N>>>(dev_graph, aggr_data);
    dev_graph.edges_sz = new_edges_sz;
    dev_graph.vert_sz = new_vert_sz;
}
