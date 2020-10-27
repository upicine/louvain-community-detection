#ifndef LOUVIAN_UTILS_CUH
#define LOUVIAN_UTILS_CUH

#include <iostream>
#include <iterator>

#define HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__))

#define HASH_INIT -1
#define FULL_MASK 0xffffffff
#define THREADS_N 128
#define WARP_SZ 32
#define WARPS_PER_BLOCK 4

typedef struct {
    unsigned int *comm = nullptr;
    unsigned int *comm_sz = nullptr;
    unsigned int *init_comm = nullptr;
    unsigned int *new_comm = nullptr;
    unsigned int *vertices = nullptr;
    unsigned int *edges = nullptr;
    float *comm_w = nullptr;
    float *weights = nullptr;
    float *neigh_w = nullptr;
    float *modularity = nullptr;
    size_t vert_sz;
    size_t edges_sz;
    size_t init_sz;
    float all_w;
    unsigned int max_deg;
} Graph;

typedef struct {
    unsigned int *comm_deg = nullptr;
    unsigned int *vert_start = nullptr;
    unsigned int *edge_pos = nullptr;
    unsigned int *new_comm_id = nullptr;
    unsigned int *verts_by_comm = nullptr;
    unsigned int *vert_hash_pos = nullptr;
    unsigned int *new_edges = nullptr;
    unsigned int *new_deg = nullptr;
    int *hash_comm = nullptr;
    float *hash_weights = nullptr;
    float* new_weights = nullptr;
} CommAggr;

void initCommAggr(CommAggr &data, Graph &dev_graph);

void initHostGraph(Graph &graph, unsigned int vert_n,
                   unsigned int edges_n);

void initDevGraph(Graph &graph, size_t vert_sz, size_t edges_sz);

void deleteHostGraph(Graph &graph);

void deleteAggrData(CommAggr &data);

void deleteDevGraph(Graph &graph);

void parseGraph(Graph &graph, const char *filename);

void copyToDevice(Graph &dev_graph, Graph &host_graph);

unsigned int findNearestPrime(unsigned int n);

void copyToHost(Graph &dev_graph, Graph &host_graph);

void parseArgs(int argc, char **argv, float &threshold, bool &verbose, char *&filename);

void setInitComm(Graph &dev_graph);

void printCommunities(Graph &dev_graph);

void printGraph(Graph &g);

__device__ unsigned int hash(unsigned int key, unsigned int it, unsigned int prime);

template<typename T>
void copyPrintDev(T *dev_arr, size_t size) {
    T *host = new T[size];
    HANDLE_ERROR(cudaMemcpy(host, dev_arr, size * sizeof(T), cudaMemcpyDeviceToHost));
    std::copy(host, host + size,
              std::ostream_iterator<T>(std::cout, " "));
    std::cout << std::endl;
    delete [] host;
}

template<typename T>
void printArr(T *arr, size_t size) {
    std::copy(arr, arr + size,
              std::ostream_iterator<T>(std::cout, " "));
    std::cout << std::endl;
}

static void HandleError(cudaError_t error, const char *file, int line) {
    if (error != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(error), file, line);
        exit(EXIT_FAILURE);
    }
}

#endif //LOUVIAN_UTILS_CUH
