#include <cstdio>
#include <fstream>
#include <vector>
#include <iostream>
#include <iterator>
#include <sstream>
#include <unistd.h>

#include <thrust/reduce.h>

#include "utils.cuh"

void initHostGraph(Graph &graph, unsigned int vert_n,
                   unsigned int edges_n) {
    graph.comm = new unsigned int[vert_n];
    graph.new_comm = new unsigned int[vert_n];
    graph.comm_sz = nullptr;
    graph.comm_w = nullptr;
    graph.init_comm = nullptr;
    graph.vertices = new unsigned int[vert_n + 1];
    graph.edges = new unsigned int[2 * edges_n];
    graph.weights = new float[2 * edges_n];
    graph.neigh_w = new float[vert_n];
    graph.vert_sz = vert_n;
    graph.init_sz = vert_n;
    graph.edges_sz = 2 * edges_n;
    graph.max_deg = 0;
}

void deleteHostGraph(Graph &graph) {
    delete[] graph.comm;
    delete[] graph.new_comm;
    delete[] graph.comm_sz;
    delete[] graph.comm_w;
    delete[] graph.init_comm;
    delete[] graph.vertices;
    delete[] graph.edges;
    delete[] graph.weights;
    delete[] graph.neigh_w;
}

void deleteAggrData(CommAggr &data) {
    HANDLE_ERROR(cudaFree(data.comm_deg));
    HANDLE_ERROR(cudaFree(data.vert_start));
    HANDLE_ERROR(cudaFree(data.edge_pos));
    HANDLE_ERROR(cudaFree(data.new_comm_id));
    HANDLE_ERROR(cudaFree(data.verts_by_comm));
    HANDLE_ERROR(cudaFree(data.vert_hash_pos));
    HANDLE_ERROR(cudaFree(data.new_edges));
    HANDLE_ERROR(cudaFree(data.new_deg));
}

void deleteDevGraph(Graph &graph) {
    HANDLE_ERROR(cudaFree(graph.comm));
    HANDLE_ERROR(cudaFree(graph.comm_sz));
    HANDLE_ERROR(cudaFree(graph.init_comm));
    HANDLE_ERROR(cudaFree(graph.new_comm));
    HANDLE_ERROR(cudaFree(graph.vertices));
    HANDLE_ERROR(cudaFree(graph.edges));
    HANDLE_ERROR(cudaFree(graph.comm_w));
    HANDLE_ERROR(cudaFree(graph.weights));
    HANDLE_ERROR(cudaFree(graph.neigh_w));
    HANDLE_ERROR(cudaFree(graph.modularity));
}

void setInitComm(Graph &dev_graph) {
    thrust::sequence(thrust::device, dev_graph.init_comm, dev_graph.init_comm + dev_graph.init_sz, 0);
}

void initDevGraph(Graph &graph, size_t vert_sz, size_t edges_sz) {
    HANDLE_ERROR(cudaMalloc((void **) &graph.comm, vert_sz * sizeof(unsigned int)));
    HANDLE_ERROR(cudaMalloc((void **) &graph.new_comm, vert_sz * sizeof(unsigned int)));
    HANDLE_ERROR(cudaMalloc((void **) &graph.init_comm, vert_sz * sizeof(unsigned int)));
    HANDLE_ERROR(cudaMalloc((void **) &graph.comm_sz, vert_sz * sizeof(unsigned int)));
    HANDLE_ERROR(cudaMalloc((void **) &graph.vertices, (vert_sz + 1) * sizeof(unsigned int)));
    HANDLE_ERROR(cudaMalloc((void **) &graph.edges, edges_sz * sizeof(unsigned int)));
    HANDLE_ERROR(cudaMalloc((void **) &graph.weights, edges_sz * sizeof(float)));
    HANDLE_ERROR(cudaMalloc((void **) &graph.neigh_w, vert_sz * sizeof(float)));
    HANDLE_ERROR(cudaMalloc((void **) &graph.comm_w, vert_sz * sizeof(float)));
    HANDLE_ERROR(cudaMalloc((void **) &graph.modularity, vert_sz * sizeof(float)));
    graph.vert_sz = vert_sz;
    graph.init_sz = vert_sz;
    graph.edges_sz = edges_sz;
}

void initCommAggr(CommAggr &data, Graph &dev_graph) {
    HANDLE_ERROR(cudaMalloc((void **) &data.comm_deg, dev_graph.vert_sz * sizeof(unsigned int)));
    HANDLE_ERROR(cudaMalloc((void **) &data.new_comm_id, dev_graph.vert_sz * sizeof(unsigned int)));
    HANDLE_ERROR(cudaMalloc((void **) &data.verts_by_comm, dev_graph.vert_sz * sizeof(unsigned int)));
    HANDLE_ERROR(cudaMalloc((void **) &data.vert_start, dev_graph.vert_sz * sizeof(unsigned int)));
    HANDLE_ERROR(cudaMalloc((void **) &data.edge_pos, dev_graph.vert_sz * sizeof(unsigned int)));
    HANDLE_ERROR(cudaMalloc((void **) &data.new_deg, (dev_graph.vert_sz + 1) * sizeof(unsigned int)));
    HANDLE_ERROR(cudaMalloc((void **) &data.vert_hash_pos, dev_graph.edges_sz * sizeof(unsigned int)));
    HANDLE_ERROR(cudaMalloc((void **) &data.new_edges, dev_graph.edges_sz * sizeof(unsigned int)));
    HANDLE_ERROR(cudaMalloc((void **) &data.new_weights, dev_graph.edges_sz * sizeof(float)));
}

void copyToDevice(Graph &dev_graph, Graph &host_graph) {
    HANDLE_ERROR(cudaMemcpy(dev_graph.comm, host_graph.comm,
                            host_graph.vert_sz * sizeof(unsigned int), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(dev_graph.vertices, host_graph.vertices,
                            (host_graph.vert_sz + 1) * sizeof(unsigned int), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(dev_graph.edges, host_graph.edges,
                            host_graph.edges_sz * sizeof(unsigned int), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(dev_graph.weights, host_graph.weights,
                            host_graph.edges_sz * sizeof(float), cudaMemcpyHostToDevice));
    dev_graph.max_deg = host_graph.max_deg;
}

void copyToHost(Graph &dev_graph, Graph &host_graph) {
    HANDLE_ERROR(cudaMemcpy(host_graph.comm, dev_graph.comm,
                            host_graph.vert_sz * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy(host_graph.vertices, dev_graph.vertices,
                            (host_graph.vert_sz + 1) * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy(host_graph.edges, dev_graph.edges,
                            host_graph.edges_sz * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy(host_graph.weights, dev_graph.weights,
                            host_graph.edges_sz * sizeof(float), cudaMemcpyDeviceToHost));
    host_graph.edges_sz = dev_graph.edges_sz;
    host_graph.vert_sz = dev_graph.vert_sz;
}

void printGraph(Graph &g) {
    for (int i = 0; i < g.vert_sz; i++) {
        std::cout << i << ": ";
        printArr(g.edges + g.vertices[i], g.vertices[i + 1] - g.vertices[i]);
        std::cout << i << ": ";
        printArr(g.weights + g.vertices[i], g.vertices[i + 1] - g.vertices[i]);
    }
}

void parseGraphSize(std::ifstream &file, unsigned int &vert_n, unsigned int &edges_n) {
    std::string line;

    do {
        std::getline(file, line);
    } while (line[0] == '%');

    std::stringstream ss(line);
    ss >> vert_n >> vert_n >> edges_n;
}

bool isZeroStartingIndex(std::ifstream &file) {
    unsigned int vert_n, edges_n;
    bool is_zero = false;

    parseGraphSize(file, vert_n, edges_n);

    unsigned int v1, v2;
    float weight = 1.0;
    for (int i = 0; i < edges_n; i++) {
        std::string line;
        std::getline(file, line);
        std::stringstream ss(line);
        ss >> v1 >> v2 >> weight;

        if (v1 == 0 || v2 == 0) {
            is_zero = true;
            break;
        }
    }

    file.seekg(0, std::ios::beg);

    return is_zero;
}

void parseGraph(Graph &graph, const char *filename) {
    std::ifstream graph_file(filename);

    if (!graph_file.good()) {
        std::cout << "Error opening file" << std::endl;
        exit(1);
    }

    unsigned int vert_n, edges_n;
    bool is_zero_start_idx = isZeroStartingIndex(graph_file);
    parseGraphSize(graph_file, vert_n, edges_n);

    initHostGraph(graph, vert_n, edges_n);

    std::vector<std::vector<unsigned int>> edges(vert_n);
    std::vector<std::vector<float>> weights(vert_n);
    std::vector<unsigned int> vertndxs(vert_n + 1, 0);

    unsigned int v1, v2;
    float weight = 1.0;
    for (int i = 0; i < edges_n; i++) {
        std::string line;
        std::getline(graph_file, line);
        std::stringstream ss(line);
        ss >> v1 >> v2 >> weight;
        weight = std::abs(weight);

        if (!is_zero_start_idx) {
            v1--;
            v2--;
        }

        if (v1 != v2) {
            edges[v1].push_back(v2);
            edges[v2].push_back(v1);
            weights[v1].push_back(weight);
            weights[v2].push_back(weight);
            vertndxs[v1 + 1]++;
            vertndxs[v2 + 1]++;
        } else {
            edges[v1].push_back(v1);
            weights[v1].push_back(weight);
            vertndxs[v1 + 1]++;
        }
    }

    graph.max_deg = 0;
    for (int i = 1; i <= vert_n; i++) {
        graph.vertices[i] = vertndxs[i] + graph.vertices[i - 1];
        graph.max_deg = std::max(graph.max_deg, vertndxs[i]);
    }

    for (int i = 0; i < vert_n; i++) {
        auto weights_it = graph.weights + graph.vertices[i];
        auto edges_it = graph.edges + graph.vertices[i];

        std::copy(weights[i].begin(), weights[i].end(), weights_it);
        std::copy(edges[i].begin(), edges[i].end(), edges_it);
    }

    graph_file.close();
}

void parseArgs(int argc, char **argv, float &threshold, bool &verbose, char *&filename) {
    int c;
    bool fopt = false;
    bool gopt = false;
    while ((c = getopt(argc, argv, "f:g:v")) != -1) {
        switch (c) {
            case 'f':
                filename = optarg;
                fopt = true;
                break;
            case 'g':
                try {
                    threshold = std::stof(std::string(optarg));
                } catch (const std::invalid_argument &ia) {
                    std::cout << "-g option invalid argument: " << ia.what() << std::endl;
                    exit(1);
                } catch (const std::out_of_range &oor) {
                    std::cout << "-g option error out of range: " << oor.what() << std::endl;
                    exit(1);
                }
                gopt = true;
                break;
            case 'v':
                verbose = true;
                break;
            case '?':
                std::cout << "Usage: ./gpulouvain -f mtx-matrix-file -g min-gain [-v]" << std::endl;
                exit(1);
            default:
                std::cout << "?? arg_parser error ??" << std::endl;
        }
    }
    if (!fopt || !gopt) {
        std::cout << "Usage: ./gpulouvain -f mtx-matrix-file -g min-gain [-v]" << std::endl;
        exit(1);
    }
}


void printCommunities(Graph &dev_graph) {
    auto *comm = new unsigned int[dev_graph.init_sz];
    HANDLE_ERROR(
            cudaMemcpy(comm, dev_graph.init_comm, dev_graph.init_sz * sizeof(unsigned int), cudaMemcpyDeviceToHost));

    std::vector<std::vector<unsigned int>> comm_vert(dev_graph.vert_sz);
    for (int i = 0; i < dev_graph.init_sz; i++) {
        comm_vert[comm[i]].push_back(i + 1);
    }

    unsigned int comm_n = 1;
    std::cout << dev_graph.vert_sz << std::endl;
    for (int i = 0; i < dev_graph.vert_sz; i++) {
        if (!comm_vert[i].empty()) {
            std::cout << comm_n << " ";
            std::copy(comm_vert[i].begin(), comm_vert[i].end(),
                      std::ostream_iterator<unsigned int>(std::cout, " "));
            std::cout << std::endl;
            comm_n++;
        }
    }

    delete[] comm;
}

static bool isPrime(unsigned int n) {
    for (unsigned int i = 2; i * i <= n; i++) {
        if (n % i == 0) {
            return false;
        }
    }
    return true;
}

unsigned int findNearestPrime(unsigned int n) {
    while (!isPrime(n)) {
        n++;
    }
    return n;
}

__device__ unsigned int hash(unsigned int key, unsigned int it, unsigned int prime) {
    unsigned int h1 = key % prime;
    unsigned int h2 = 1 + (key % (prime - 1));
    return (h1 + it * h2) % prime;
}
