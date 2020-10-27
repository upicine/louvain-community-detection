import math
import collections
import argparse

debug = False


class Graph(object):

    def __init__(self):
        self.comm = []
        self.comm_w = []
        self.comm_sz = []
        self.vertices = []
        self.edges = []
        self.weights = []
        self.neigh_w = []
        self.new_comm = []
        self.all_w = 0
        self.vert_n = 0

    @staticmethod
    def _parse_line(line):
        edge_str = line.split()
        v1 = int(edge_str[0])
        v2 = int(edge_str[1])
        weight = float(edge_str[2]) if len(edge_str) == 3 else 1.0
        return v1, v2, weight

    def parse_graph(self, filename):
        with open(filename, 'r') as f:
            vert_n, col_n, edges_n = (int(num) for num in next(f).split())
            vert_n += 1
            self.edges = [[] for _ in range(vert_n)]
            self.weights = [[] for _ in range(vert_n)]
            self.vert_n = vert_n
            for line in f:
                v1, v2, weight = self._parse_line(line)

                if v1 != v2:
                    self.edges[v1].append(v2)
                    self.edges[v2].append(v1)
                    self.weights[v1].append(weight)
                    self.weights[v2].append(weight)
                else:
                    self.edges[v1].append(v1)
                    self.weights[v1].append(weight)

    def _sum_neigh_w(self):
        self.neigh_w = []
        self.all_w = 0.0
        for v in self.weights:
            w_sum = sum(v)
            self.neigh_w.append(w_sum)
            self.all_w += w_sum
        self.all_w /= 2

    def _init_comm(self):
        self.comm = [i for i in range(self.vert_n)]
        self.comm_w = [w for w in self.neigh_w]
        self.comm_sz = [1] * len(self.comm)
        self.new_comm = [0] * len(self.comm)

    def _assign_new_comm(self, v_set):
        for v in v_set:
            self.comm_sz[self.new_comm[v]] += 1
            self.comm_sz[self.comm[v]] -= 1
            self.comm[v] = self.new_comm[v]

    def _calc_new_comm_w(self):
        self.comm_w = [0] * len(self.comm)
        for v, w in enumerate(self.neigh_w):
            self.comm_w[self.comm[v]] += w

    def modularity_optimization(self, threshold):
        buckets = [0, 4, 8, 16, 32, 84, 319, math.inf]

        self._sum_neigh_w()
        self._init_comm()
        act_modularity = self._compute_modularity()
        first_loop = True
        ret = True
        while True:
            for i in range(1, len(buckets)):
                v_set = [j for j, v in enumerate(self.edges)
                         if buckets[i - 1] < len(v) <= buckets[i]]
                if len(v_set) > 0:
                    self._compute_move(v_set)
                    self._assign_new_comm(v_set)
                    self._calc_new_comm_w()
            old_modularity = act_modularity
            act_modularity = self._compute_modularity()
            print(act_modularity, old_modularity)
            if first_loop and act_modularity - old_modularity < threshold:
                print(old_modularity)
                ret = False
            first_loop = False
            if act_modularity - old_modularity < threshold:
                return ret

    def _compute_vert_neigh_comm_sum(self):
        all_sum = 0
        for v, (neighs_w, neighs_v) in enumerate(zip(self.weights, self.edges)):
            for w, e in zip(neighs_w, neighs_v):
                if self.comm[v] == self.comm[e]:
                    all_sum += w

        return all_sum

    def _compute_modularity(self):
        comm_w_sum = sum(w ** 2 for w in self.comm_w)
        v_neigh_comm_s = self._compute_vert_neigh_comm_sum()
        return (v_neigh_comm_s / (2 * self.all_w)) - (comm_w_sum / (4 * self.all_w ** 2))

    def is_lower_idx(self, v, e):
        if self.comm_sz[self.comm[v]] == 1 and self.comm_sz[self.comm[e]] == 1:
            return self.comm[e] < self.comm[v]
        return True

    def _compute_move(self, v_set):
        for v in v_set:
            neigh_comm_w = collections.defaultdict(float)
            for w, e in zip(self.weights[v], self.edges[v]):
                neigh_comm_w[self.comm[e]] += w

            max_gain = float('-inf')
            max_e = -1
            max_comm = -1
            comm_e_w = 0.0
            for i, e in enumerate(self.edges[v]):
                if self.comm[e] == self.comm[v] and v != e:
                    comm_e_w += self.weights[v][i]

                if self.is_lower_idx(v, e):
                    mod_gain = (neigh_comm_w[self.comm[e]] / self.all_w) \
                               + (self.neigh_w[v]
                                  * ((self.comm_w[self.comm[v]] - self.neigh_w[v]) - self.comm_w[self.comm[e]])
                                  / (2 * self.all_w ** 2))
                    if debug:
                        print(f"VERT: {v}, EDGE: {e}, MOD_GAIN: {mod_gain}, "
                              f"e_i_C(j): {neigh_comm_w[self.comm[e]]}, k_i: {self.neigh_w[v]}, "
                              f"a_C(i)-i: {self.comm_w[self.comm[v]] - self.neigh_w[v]}, "
                              f"a_C(j): {self.comm_w[self.comm[e]]}, m: {self.all_w}")
                    if mod_gain > max_gain or (mod_gain == max_gain and self.comm[e] < max_comm):
                        max_gain = mod_gain
                        max_e = e
                        max_comm = self.comm[e]
                        if mod_gain > 0:
                            self.new_comm[v] = self.comm[e]
            if max_gain <=0:#- (comm_e_w / self.all_w) <= 0:
                self.new_comm[v] = self.comm[v]

    def print_graph(self):
        for i, (edges, weights) in enumerate(zip(self.edges, self.weights)):
            print(i, ':', list(zip(edges, weights)))

    def print_comm(self):
        for p in enumerate(self.comm):
            print(p, end=" ")
        print()


class CommAggregation(object):

    def __init__(self):
        self.comm_deg = []
        self.new_comm_id = []
        self.new_edges = []
        self.new_weights = []
        self.verts_by_comm = []
        self.new_vert_sz = 0
        self.new_comm_sz = 0

    def _sum_deg(self, graph):
        for i in range(graph.vert_n):
            self.comm_deg[graph.comm[i]] += len(graph.edges[i])

    def _set_new_id(self, graph):
        self.new_id = [0] * graph.vert_n
        c_id = 0
        for i, c_sz in enumerate(graph.comm_sz):
            if c_sz == 0:
                self.new_id[i] = 0
            else:
                self.new_id[i] = c_id
                c_id += 1
        return c_id

    def _set_verts_by_comm(self, graph):
        for v, c in enumerate(graph.comm):
            self.verts_by_comm[self.new_id[c]].append(v)

    def _merge_community(self, graph):
        for c in range(self.new_vert_sz):
            weights = collections.defaultdict(float)
            for v in self.verts_by_comm[c]:
                for i, e in enumerate(graph.edges[v]):
                    weights[graph.comm[e]] += graph.weights[v][i]

            for old_comm, comm_w in weights.items():
                self.new_edges[c].append(self.new_id[old_comm])
                self.new_weights[c].append(comm_w)

    def contract(self, graph):
        self.comm_deg = [0] * graph.vert_n
        self._sum_deg(graph)
        self.new_vert_sz = self._set_new_id(graph)
        self.verts_by_comm = [[] for _ in range(self.new_vert_sz)]
        self.new_edges = [[] for _ in range(self.new_vert_sz)]
        self.new_weights = [[] for _ in range(self.new_vert_sz)]
        self._set_verts_by_comm(graph)
        self._merge_community(graph)
        graph.edges = self.new_edges
        graph.weights = self.new_weights
        graph.vert_n = self.new_vert_sz


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", help="input file")
    parser.add_argument("-g", "--gain", type=float, help="gain")
    args = parser.parse_args()
    filename = args.file
    g = Graph()
    aggr = CommAggregation()
    g.parse_graph(filename)

    while True:
        if not g.modularity_optimization(args.gain):
            break
        aggr.contract(g)
        # g.print_graph()


if __name__ == '__main__':
    main()
