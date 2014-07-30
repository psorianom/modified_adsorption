'''
Created on 15/06/2013
@author: Pavel SORIANO-MORALES

Implementation of the random walk label-propagation Modified Adsorption algorithm by P. Talukdar.
A full explanation can be found in the reference New Regularized Algorithms for Transductive Learning.

It is coded with care and quite a lot of thought went to this. Nevertheless, it is not guaranteed
to be correctly implemented. The only small text that can asses the correctness is that the same results
from the junto (a Java package from P. Talukdar implementing his method) example are obtained.
Though it is a very (very) small test.

'''
from collections import defaultdict
import itertools

import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
from networkx import read_weighted_edgelist, to_scipy_sparse_matrix


class ModifiedAdsorption(object):
    '''
    classdocs
    '''


    def read_graph_seeds(self, graph_file, seed_file, nb_seeds=10, delimiter="\t"):
        """
        Quick and dirty way to get the adjacency matrix and seeds!
        """
        G = read_weighted_edgelist(graph_file, delimiter=delimiter)
        W = to_scipy_sparse_matrix(G)
        nodes_G = G.nodes()
        label_index = defaultdict(list)
        # Deal with seeds
        file_lines = open(seed_file, "r").readlines()
        seed_nodes, seed_labels, seed_values = [], [], []
        for idx, line in enumerate(file_lines):
            split_line = line.split(delimiter)
            node, label, value = split_line[0], split_line[1], float(split_line[2])
            seed_nodes.append(node)
            seed_labels.append(label)
            seed_values.append(value)
            label_index[label].append(idx)

        # Store golden_labels/seeds node name and their real value
        golden_labels = dict(zip(seed_nodes, seed_labels))

        unique_labels = sorted(set(seed_labels))
        labels_dict = {e: idx for idx, e in enumerate(unique_labels)}  # [L1, L2, L3,..., DUMMY]


        # Build the seeds matrix: number of nodes x number of labels + 1
        seeds_matrix = np.zeros(
            [W.shape[0], len(set(seed_labels)) + 1])  # We add 1 because of the "dummy" label used in the algorithm

        # Draw a n sample for each type of label
        label_samples = {}
        for k, v in label_index.items():
            temp_nb_seeds = nb_seeds if len(v) > nb_seeds else len(v)
            label_samples[k] = np.random.choice(v, temp_nb_seeds)

        # Once we have the sample for each class, we build the seeds matrix
        for label, seed_idx in label_samples.items():
            label_j = labels_dict.get(label)
            for i in seed_idx:
                seeds_matrix[i, label_j] = seed_values[i]

        return W, seeds_matrix, unique_labels, golden_labels, nodes_G

    def __init__(self, graph_file, seeds_file, nb_seeds=10, tol=1e-3, maxiter=5):

        '''
        Constructor
        '''
        self._W, self._Y, labels, self._golden_labels, self._nodes_list = self.read_graph_seeds(graph_file,
                                                                                                seeds_file,
                                                                                                nb_seeds=nb_seeds)
        self._labels = labels
        self._mu1 = 1.
        self._mu2 = 1e-2
        self._mu3 = 1e-2
        self._beta = 2.
        self._tol = tol
        self._get_initial_probs()
        self.max_iter = maxiter

    def _get_initial_probs(self):
        print "\t..Getting initial probabilities."
        nr_nodes = self._W.shape[0]

        # Calculate Pr transition probabilities matrix
        self._Pr = lil_matrix((nr_nodes, nr_nodes))
        W_coo = self._W.tocoo()
        col_sums = {k: v for k, v in enumerate(W_coo.sum(0).tolist()[0])}
        for i, j, v in itertools.izip(W_coo.row, W_coo.col, W_coo.data):
            # print "\t%d\t%d" % (i,j)
            self._Pr[i, j] = v / col_sums[j]

        # W_nnz = self._W.nonzero()
        # for v1, v2 in zip(W_nnz[0], W_nnz[1]):
        # # Get the sum fast
        # print "\t%d\t%d" % (v1,v2)
        # self._Pr[v1, v2] = self._W[v1, v2] / (self._W[:, v2].sum())

        print "\t\tPr matrix done."
        # Calculate H entropy vector for each node
        self._H = lil_matrix((nr_nodes, 1))
        self._Pr = self._Pr.tocoo()

        for i, _, v in itertools.izip(self._Pr.row, self._Pr.col, self._Pr.data):
            self._H[i, 0] += -(v * np.log(v))

        # Pr_nnz = self._Pr.nonzero()
        # for v, u in zip(Pr_nnz[0], Pr_nnz[1]):
        # self._H[v, 0] += -(self._Pr[v, u] * np.log(self._Pr[v, u]))
        print "\t\tH matrix done."

        # Calculate vector C (Cv)
        self._H = self._H.tocoo()
        # H_nnz = self._H.nonzero()
        self._C = lil_matrix((nr_nodes, 1))
        log_beta = np.log(self._beta)
        for i, _, v in itertools.izip(self._H.row, self._H.col, self._H.data):
            # print v
            self._C[i, 0] = (log_beta) / (np.log(self._beta + (1 / (np.exp(-v) + 0.00001))))

        # for i in H_nnz[0]:
        # self._C[i, 0] = (np.log(self._beta)) / (np.log(self._beta + np.exp(self._H[i, 0])))
        print "\t\tC matrix done."

        # Calculate vector D (dv)
        # Get nodes that are labeled
        Y_nnz = self._Y.nonzero()
        self._D = lil_matrix((nr_nodes, 1))
        self._H = self._H.tolil()

        for i in Y_nnz[0]:
            # Check if node v is labeled            
            self._D[i, 0] = (1. - self._C[i, 0]) * np.sqrt(self._H[i, 0])

        print "\t\tD matrix done."
        # Calculate Z vector
        self._Z = lil_matrix((nr_nodes, 1))
        c_v = self._C + self._D
        c_v_nnz = c_v.nonzero()
        for i in c_v_nnz[0]:
            self._Z[i, 0] = np.max([c_v[i, 0], 1.])
        print "\t\tZ matrix done."

        # Finally calculate p_cont, p_inj and p_abnd
        self._Pcont = lil_matrix((nr_nodes, 1))
        self._Pinj = lil_matrix((nr_nodes, 1))
        self._Pabnd = lil_matrix((nr_nodes, 1))
        C_nnz = self._C.nonzero()
        for i in C_nnz[0]:
            self._Pcont[i, 0] = self._C[i, 0] / self._Z[i, 0]
        for i in Y_nnz[0]:
            self._Pinj[i, 0] = self._D[i, 0] / self._Z[i, 0]

        self._Pabnd[:, :] = 1.
        pc_pa = self._Pcont + self._Pinj
        pc_pa_nnz = pc_pa.nonzero()
        for i in pc_pa_nnz[0]:
            self._Pabnd[i, 0] = 1. - pc_pa[i, 0]
        # for i in range(nr_nodes):
        # self._Pabnd[i, 0] = 1. - self._Pcont[i, 0] - self._Pinj[i, 0]

        self._Pabnd = csr_matrix(self._Pabnd)
        self._Pcont = csr_matrix(self._Pcont)
        self._Pinj = csr_matrix(self._Pinj)
        print "\n\nDone getting probabilities..."

    def results(self):
        """
        Return the class determined by the maximum in each row of the Yh matrix. Doesnt
        take into account the dummy label
        """
        result_complete = []
        self._mad_class_index = np.squeeze(np.asarray(self._Yh[:, :self._Yh.shape[1] - 1].todense().argmax(axis=1)))
        self._label_results = np.array([self._labels[r] for r in self._mad_class_index])
        print self._label_results
        for i in range(len(self._label_results)):
            result_complete.append((self._nodes_list[i], self._label_results[i],
                                    self._golden_labels.get(self._nodes_list[i], "NO GOLDEN LABEL")))
        print result_complete
        self._result_complete = result_complete
        return self._label_results, self._result_complete

    def calculate_mad(self):
        print "\n...Calculating modified adsorption."
        nr_nodes = self._W.shape[0]

        # 1. Initialize Yhat
        self._Yh = lil_matrix(self._Y.copy())

        # 2. Calculate Mvv
        self._M = lil_matrix((nr_nodes, nr_nodes))
        # self._M = lil_matrix(np.diag((self._mu1*self._Pinj).toarray().flatten()) + (np.eye(nr_nodes)*self._mu3))
        """
        TODO: This does not work cause flattening and to array-ing is not memory cool so it fails. Need to find a way to
            build this initial matrices with sparse matrices
        """
        for v in range(nr_nodes):
            first_part = self._mu1 * self._Pinj[v, 0]
            second_part = 0.

            for u in self._W[v, :].nonzero()[1]:
                if u != v:
                    second_part += (self._Pcont[v, 0] * self._W[v, u] + self._Pcont[u, 0] * self._W[u, v])
            self._M[v, v] = first_part + (self._mu2 * second_part) + self._mu3

        # W_coo = self._W.tocoo()
        #
        # for i, j, v in itertools.izip(W_coo.row, W_coo.col, W_coo.data):
        # second_part = self._mu2 * ((self._Pcont[i, 0] * v) + self._Pcont[j, 0] * v)
        # self._M[i, i] += second_part

        # self._M = csr_matrix(self._M)

        # 3. Begin main loop
        itero = 0
        r = lil_matrix((1, self._Y.shape[1]))
        r[-1, -1] = 1.
        Yh_old = lil_matrix((self._Y.shape[0], self._Y.shape[1]))

        # Main loop begins
        Pcont = self._Pcont.toarray()

        while not self._check_convergence(Yh_old, self._Yh, ) and self.max_iter > itero:
            itero += 1
            print ">>>>>Iteration:%d" % itero
            self._D = lil_matrix((nr_nodes, self._Y.shape[1]))
            # 4. Calculate Dv
            print "\t\tCalculating D..."
            time_d = time.time()
            W_coo = self._W.tocoo()
            for i, j, v in itertools.izip(W_coo.row, W_coo.col, W_coo.data):
                # self._D[i, :] += (Pcont[i][0] * v + Pcont[j][0] * v) * self._Yh[j, :]
                self._D[i, :] += (v * (Pcont[i][0] + Pcont[j][0])) * self._Yh[j, :]
                # print i

            print "\t\tTime it took to calculate D:", time.time() - time_d
            print
            # for v in range(nr_nodes):
            # for u in self._W[v, :].nonzero()[1]:
            # self._D[v, :] += (self._Pcont[v, 0] * self._W[v, u] + self._Pcont[u, 0] * self._W[u, v])\
            # * self._Yh[u, :]
            # print v,  self._D[v, :].todense()

            print "\t\tUpdating Y..."
            # 5. Update Yh
            time_y = time.time()
            Yh_old = self._Yh.copy()
            for v in range(nr_nodes):
                # 6.
                second_part = ((self._mu1 * self._Pinj[v, 0] * self._Y[v, :]) +
                               (self._mu2 * self._D[v, :]) +
                               (self._mu3 * self._Pabnd[v, 0] * r))
                self._Yh[v, :] = 1. / (self._M[v, v]) * second_part
                # print v
            # self._Yh = csr_matrix(self._Yh)
            print "\t\tTime it took to calculate Y:", time.time() - time_y
            print
            # repeat until convergence.

    def _check_convergence(self, A, B):
        if not type(A) is csr_matrix:
            A = csr_matrix(A)
        if not type(B) is csr_matrix:
            B = csr_matrix(B)

        norm_a = (A.data ** 2).sum()
        norm_b = (B.data ** 2).sum()
        diff = np.abs(norm_a - norm_b)
        if diff <= self._tol:
            return True
        else:
            print "\t\tNorm differences between Y_old and Y_hat: ", diff
            print
            return False


if __name__ == '__main__':
    import time

    graph_file = "input_graph"
    seed_file = "seeds"
    #
    # graph_file = "super_graph.edges"
    # seed_file = "gold_labels"

    # W2, Y2, labels, golden_labels, nodes_names = read_graph_seeds(graph_file, seed_file)
    timo = time.time()

    mad = ModifiedAdsorption(graph_file, seed_file)
    mad.calculate_mad()
    print "MAD took:", time.time() - timo
    y_pred = mad.results()

    # print y_pred
    pass
