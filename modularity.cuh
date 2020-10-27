#ifndef LOUVIAN_MODULARITY_CUH
#define LOUVIAN_MODULARITY_CUH

#include "utils.cuh"

bool modularityOptimization(Graph &dev_graph, unsigned int *v_set, float threshold);

#endif //LOUVIAN_MODULARITY_CUH
