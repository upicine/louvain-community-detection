#include "utils.cuh"
#include "modularity.cuh"
#include "aggregation.cuh"

int main(int argc, char **argv) {
    char *filename;
    unsigned int *v_set, *comm_set;
    float threshold, alg_time, copy_time;
    bool verbose = false;

    parseArgs(argc, argv, threshold, verbose, filename);

    Graph host_graph, dev_graph;
    CommAggr aggr_data;

    parseGraph(host_graph, filename);


    cudaEvent_t copy_start, copy_stop, alg_start, alg_stop;
    HANDLE_ERROR(cudaEventCreate(&copy_start));
    HANDLE_ERROR(cudaEventCreate(&copy_stop));
    HANDLE_ERROR(cudaEventRecord(copy_start, 0));

    initDevGraph(dev_graph, host_graph.vert_sz, host_graph.edges_sz);
    copyToDevice(dev_graph, host_graph);
    initCommAggr(aggr_data, dev_graph);
    setInitComm(dev_graph);

    HANDLE_ERROR(cudaMalloc((void **) &v_set, dev_graph.vert_sz * sizeof(unsigned int)));
    HANDLE_ERROR(cudaMalloc((void **) &comm_set, dev_graph.vert_sz * sizeof(unsigned int)));

    HANDLE_ERROR(cudaEventRecord(copy_stop, 0));
    HANDLE_ERROR(cudaEventSynchronize(copy_stop));
    HANDLE_ERROR(cudaEventElapsedTime(&copy_time, copy_start, copy_stop));
    HANDLE_ERROR(cudaEventDestroy(copy_start));
    HANDLE_ERROR(cudaEventDestroy(copy_stop));


    HANDLE_ERROR(cudaEventCreate(&alg_start));
    HANDLE_ERROR(cudaEventCreate(&alg_stop));
    HANDLE_ERROR(cudaEventRecord(alg_start, 0));

    for (;;) {
        if (!modularityOptimization(dev_graph, v_set, threshold)) {
            break;
        }
        aggregation(dev_graph, aggr_data, comm_set);
    }

    HANDLE_ERROR(cudaEventRecord(alg_stop, 0));
    HANDLE_ERROR(cudaEventSynchronize(alg_stop));
    HANDLE_ERROR(cudaEventElapsedTime(&alg_time, alg_start, alg_stop));
    HANDLE_ERROR(cudaEventDestroy(alg_start));
    HANDLE_ERROR(cudaEventDestroy(alg_stop));

    std::cout << alg_time << " " << alg_time + copy_time << std::endl;

    if (verbose) {
        printCommunities(dev_graph);
    }

    HANDLE_ERROR(cudaFree(v_set));
    HANDLE_ERROR(cudaFree(comm_set));

    deleteHostGraph(host_graph);
    deleteDevGraph(dev_graph);
    deleteAggrData(aggr_data);

    return 0;
}
