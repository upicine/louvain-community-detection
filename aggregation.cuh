#ifndef LOUVIAN_AGGREGATION_CUH
#define LOUVIAN_AGGREGATION_CUH

#include "utils.cuh"

void aggregation(Graph &dev_graph, CommAggr &aggr_data, unsigned int *comm_set);

#endif //LOUVIAN_AGGREGATION_CUH
