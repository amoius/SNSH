import networkx as nx
from w2v import *
import tqdm
import random
import numpy as np
from torch import nn
import scipy.sparse as sp
import torch
import io
import process
import matplotlib.pyplot as plt
from shdgi import SHDGI


def generate_window(url):
    path_to_file = url
    with open(path_to_file) as f:
        lines = f.read().splitlines()
    window = {}
    for line in lines:
        li = [int(i) for i in line.split('\t')]
        window[li[0]] = li[1:]
    return window


class SignedGraphConvolutionalNetwork:
    def __init__(self, args):
        """
        SGCN Initialization.
        :param device: Device for calculations.
        :param args: Arguments object.
        :param X: Node features.
        """
        self.args = args
        self.G = nx.read_edgelist(args.edge_list_url,
                                  create_using=nx.Graph(), nodetype=int, data=(('relation', int),))
        self.vocab_size = len(self.G.nodes()) + 1
        self.num_ns = args.num_ns  # 负采样数量
        self.min_range = args.min_range  # 采样次数
        self.BATCH_SIZE = args.BATCH_SIZE
        self.SEED = args.seed
        self.AUTOTUNE = tf.data.AUTOTUNE
        self.bad_window = generate_window(args.bad_window_url)
        self.good_window = generate_window(args.good_window_url)

    def start(self):
        """
        SGCN training.
        """
        dataset_pos = self.generate_dataset(tuples=list(self.G.edges()), flag=1)
        dataset_neg = self.generate_dataset(tuples=list(self.G.edges()), flag=0)
        weights_pos = self.w2v_train(dataset_pos)[1:]
        weights_neg = self.w2v_train(dataset_neg)[1:]
        embeds = self.shdgi_train(weights_pos, weights_neg)
        self.make_graph(weights_pos)
        self.make_graph(embeds)

    def w2v_train(self, dataset):
        """
        Word2Vec training.
        :param dataset: 数据集
        :return:
        """
        # if len(self.G.edges()) > 10000:
        #     dataset = self.generate_dataset(tuples=list(self.G.edges())[0:10000], flag=flag)
        #     for i in range(int(len(self.G.edges()) / 10000)):
        #         dataset1 = self.generate_dataset(tuples=list(self.G.edges())[i * 10000 + 10000:i * 10000 + 20000],
        #                                          flag=flag)
        #         dataset = dataset.concatenate(dataset1)
        # else:
        #     dataset = self.generate_dataset(tuples=list(self.G.edges()), flag=flag)
        word2vec = Word2Vec(self.vocab_size, self.args.embedding_dim)
        word2vec.compile(optimizer='adam',
                         loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                         metrics=['accuracy'])
        word2vec.fit(dataset, epochs=self.args.epochs)
        weights = word2vec.get_layer("w2v_embedding").get_weights()[0]
        return weights

    def generate_dataset(self, tuples, flag):
        targets, contexts, labels = self.generate_training_data(
            tuples=tuples,
            flag=flag)

        targets = np.array(targets)
        contexts = np.array(contexts)[:, :, 0]
        labels = np.array(labels)
        BUFFER_SIZE = len(targets)
        dataset = tf.data.Dataset.from_tensor_slices(((targets, contexts), labels))
        dataset = dataset.shuffle(BUFFER_SIZE).batch(self.BATCH_SIZE, drop_remainder=True)
        dataset = dataset.cache().prefetch(buffer_size=self.AUTOTUNE)
        return dataset

    def random_sample(self, node_1, node_2, flag):
        # 返回target，context，和label
        # 采样方式，1、对所有节点，分别构造负采样，缺点会出现大量额外数据2、1个随机采样加所有节点的组合
        relation = self.G[node_1][node_2]['relation']
        if flag == 1:
            if relation == 1:
                window = self.bad_window[node_2] + self.bad_window[node_1]
                label = tf.constant([1] + [0] * self.num_ns, dtype="float32")
            else:
                window = self.good_window[node_2] + self.good_window[node_1]
                label = tf.constant([0] + [1 / self.num_ns] * self.num_ns, dtype="float32")
        else:
            if relation == -1:
                window = self.good_window[node_2] + self.good_window[node_1]
                label = tf.constant([1] + [0] * self.num_ns, dtype="float32")
            else:
                window = self.bad_window[node_2] + self.bad_window[node_1]
                label = tf.constant([0] + [1 / self.num_ns] * self.num_ns, dtype="float32")
        if len(window) == self.num_ns - 1:
            window += [0]
        elif len(window) < self.num_ns - 1:
            window += [0]
            window += random.sample(list(self.G.nodes()), self.num_ns - len(window))
        pairs = []
        for i in range(0, min(int((len(window) / self.num_ns)) + 1, self.min_range)):
            negative_sampling_candidates = random.sample(window, self.num_ns)
            negative_sampling_candidates = tf.expand_dims(
                negative_sampling_candidates, 1)
            context_class = tf.expand_dims(
                tf.constant([node_2], dtype="int32"), 1)
            context = tf.concat([context_class, negative_sampling_candidates], 0)
            pairs.append([node_1, context, label])
            negative_sampling_candidates = random.sample(window, self.num_ns)
            negative_sampling_candidates = tf.expand_dims(
                negative_sampling_candidates, 1)
            context_class = tf.expand_dims(
                tf.constant([node_1], dtype="int32"), 1)
            context = tf.concat([context_class, negative_sampling_candidates], 0)
            pairs.append([node_2, context, label])
        return pairs

    def generate_training_data(self, tuples, flag):
        targets, contexts, labels = [], [], []
        # 进行负采样，首先考虑节点i和节点j的所有负关系节点，若数量不够num_ns，则首先加入虚拟负节点0，随后在所有非邻居节点中随机选择
        # 节点对：正关系节点对和负关系节点对，对于正关系节点对，使用两个节点的负邻居作为负采样，节点对调换关系作为输入
        print('Generating training data...')
        for tri_tuple in tqdm.tqdm(tuples):
            target_word = tri_tuple[0]
            context_word = tri_tuple[1]
            window1 = self.random_sample(target_word, context_word, flag)

            # Append each element from the training example to global lists.
            for pair in window1:
                target = pair[0]
                context = pair[1]
                label = pair[2]
                targets.append(target)
                contexts.append(context)
                labels.append(label)

        return targets, contexts, labels

    def make_attributes(self):
        # 构造特征属性向量 f=(de, ne+, ne-, pr)
        F = []
        F_neg = []
        for node in self.G.nodes():
            neighbours = self.G.neighbors(node)
            l_ne = list(neighbours)
            de = len(l_ne)
            ne_p = 0
            ne_n = 0
            for n in l_ne:
                relation = self.G[node][n]['relation']
                if relation == 1:
                    ne_p += 1
                else:
                    ne_n += 1
            pr = (ne_p - ne_n) / (ne_p + ne_n)
            f = [de, ne_p, ne_n, pr]
            f_neg = [de, ne_n, ne_p, -pr]
            F.append(f)
            F_neg.append(f_neg)
        return F, F_neg

    def shdgi_train(self, weights_pos, weights_neg):
        batch_size = 1
        nb_epochs = 1000
        patience = 20
        nonlinearity = 'prelu'
        sparse = True
        lr = 0.01
        l2_coef = 0.0
        hid_units = self.args.embedding_dim  # 输出维度数
        n_f = 4
        w1 = 1  # 正关系权重
        w2 = 1  # 负关系权重
        features = torch.FloatTensor(weights_pos[np.newaxis])
        features_neg = torch.FloatTensor(weights_neg[np.newaxis])
        F, F_neg = self.make_attributes()
        F = np.array(F)
        F = torch.FloatTensor(F[np.newaxis])
        F_neg = np.array(F_neg)
        F_neg = torch.FloatTensor(F_neg[np.newaxis])
        edges = list(self.G.edges(data=True))
        P = []
        N = []
        for i in edges:
            if i[2]['relation'] == 1:
                P.append(i)
            else:
                N.append(i)

        G_P = nx.Graph()
        G_N = nx.Graph()
        G_P.add_nodes_from(self.G.nodes())
        G_N.add_nodes_from(self.G.nodes())
        G_P.add_edges_from(P)
        G_N.add_edges_from(N)
        nodes = [i for i in range(1, len(list(self.G.nodes())) + 1)]
        a_A = np.array(nx.adjacency_matrix(self.G, nodelist=self.G.nodes()).todense())
        a_P = np.array(nx.adjacency_matrix(G_P, nodelist=self.G.nodes()).todense())
        a_N = np.array(nx.adjacency_matrix(G_N, nodelist=self.G.nodes()).todense())
        adj_A = sp.csr_matrix(a_A)
        adj_P = sp.csr_matrix(a_P)
        adj_N = sp.csr_matrix(a_N)
        adj = adj_P.todense() * w1 - adj_N.todense() * w2
        adj = process.normalize_adj(adj + 10 * sp.eye(adj.shape[0]))  # 邻接矩阵
        sp_adj = process.sparse_mx_to_torch_sparse_tensor(adj)
        nb_nodes = features.shape[1]  # 节点数
        ft_size = features.shape[2]  # 特征数
        model = SHDGI(ft_size, hid_units, n_f, nonlinearity)
        optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_coef)
        b_xent = nn.BCEWithLogitsLoss()
        xent = nn.CrossEntropyLoss()
        best = 1e9
        best_t = 0
        cnt_wait = 0
        for epoch in range(nb_epochs):
            model.train()
            optimiser.zero_grad()

            idx = np.random.permutation(nb_nodes)
            shuf_fts = features_neg[:, idx, :]

            lbl_1 = torch.ones(batch_size, nb_nodes)
            lbl_2 = torch.zeros(batch_size, nb_nodes)
            lbl = torch.cat((lbl_1, lbl_2), 1)

            logits = model(features, shuf_fts, F, F_neg, sp_adj if sparse else adj, sparse, None, None, None)

            loss_E = b_xent(logits[0], lbl)
            loss_I = b_xent(logits[1], lbl)
            loss_J = b_xent(logits[2], lbl)
            loss = loss_E + loss_I + loss_J
            print('Loss:', loss)

            if loss < best:
                best = loss
                best_t = epoch
                cnt_wait = 0
                torch.save(model.state_dict(), 'best_dgi.pkl')
            else:
                cnt_wait += 1

            if cnt_wait == patience:
                print('Early stopping!')
                break

            loss.backward()
            optimiser.step()
        print('Loading {}th epoch'.format(best_t))
        model.load_state_dict(torch.load('best_dgi.pkl'))
        embeds, _ = model.embed(features, sp_adj if sparse else adj, sparse, None)
        embeds = embeds.numpy()[0]
        out_v = io.open(self.args.output_url + 'node_weights.tsv', 'w', encoding='utf-8')
        for node in self.G.nodes():
            out_v.write(
                '\t'.join([str(x) for x in embeds[node-1]]) + "\n")
        out_v.close()
        out_v = io.open(self.args.output_url + 'edge_weights.tsv', 'w', encoding='utf-8')
        rel_dict = self.read_rel_dict()
        for item in rel_dict:
            node_1, node_2 = item[0] - 1, item[1] - 1
            relation = (item[2] + 1) / 2
            out_v.write(
                '\t'.join([str(x) for x in embeds[node_1]]) + '\t' +
                '\t'.join([str(x) for x in embeds[node_2]]) + '\t' +
                str(relation) + "\n")
        out_v.close()
        return embeds

    def read_rel_dict(self):
        path_to_file = self.args.edge_list_url
        with open(path_to_file) as f:
            lines = f.read().splitlines()
        rel_dict = []
        for line in lines:
            li = [int(i) for i in line.split(' ')]
            rel_dict.append(li)
        return rel_dict

    def make_graph(self, embs):
        weights = embs
        pos = {}
        for i in self.G.nodes():
            pos[i] = weights[i - 1]
        color = []
        sty = []
        for i in self.G.edges():
            r = self.G[i[0]][i[1]]['relation']
            if r == 1:
                color.append('r')
                sty.append('solid')
            else:
                color.append('b')
                sty.append('dashed')
        nx.draw_networkx_edges(self.G, edge_color=color, width=0.7, pos=pos, alpha=0.6)
        nx.draw_networkx_nodes(self.G, node_size=150, pos=pos, node_color='mediumpurple', alpha=1)
        plt.show()
