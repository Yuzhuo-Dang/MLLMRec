import os
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F


from common.abstract_recommender import GeneralRecommender
from utils.utils import build_knn_neighbourhood, get_dense_laplacian

class MLLMREC(GeneralRecommender):
    def __init__(self, config, dataset):
        super(MLLMREC, self).__init__(config, dataset)
        self.n_nodes = self.n_users + self.n_items
        self.dim_feat = config['feat_embed_dim']
        self.n_layers = config['n_ii_layers']
        self.n_layers_ui = config['n_ui_layers']
        self.knn_k = config['knn_k']
        self.knn_jac = config['knn_jac']
        self.pure = config['pure']
        self.drop_g = config['drop_g'] if config['drop_g'] is not None else 0.1

        self.sem_weight = config['sem_weight'] if config['sem_weight'] is not None else 0.0001
        self.info_weight = config['info_weight'] if config['info_weight'] is not None else 0.01

        # load dataset info
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)

        dataset_path = os.path.abspath(config['data_path'] + config['dataset'])
        ii_mat_file = os.path.join(dataset_path + '/mat', 'ii_mat_{}_{}.pt'.format(self.knn_jac, self.pure))

        self.item_feat_embedding = nn.Embedding.from_pretrained(self.item_feat, freeze=True) #self.item_feat, self.t_feat, self.v_feat
        self.user_preference_feat_embedding = nn.Embedding.from_pretrained(self.user_preference_feat, freeze=True)
        if os.path.exists(ii_mat_file):
            self.ii_mat = torch.load(ii_mat_file)
        else:
            self.ii_mat = self.get_knn_adj_mat_adv(self.item_feat_embedding.weight.detach())
            torch.save(self.ii_mat, ii_mat_file)

        self.norm_adj = self.get_norm_adj_mat().to(self.device)

        self.masked_adj = self.norm_adj
        self.edge_indices, self.edge_values = self.get_edge_info()
        self.edge_indices = self.edge_indices.to(self.device)
        self.edge_values = self.edge_values.to(self.device)

        self.user_embedding = nn.Embedding(self.n_users, self.dim_feat)
        self.item_embedding = nn.Embedding(self.n_items, self.dim_feat)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)


        self.MLP = nn.Linear(self.item_feat.size(1), 4 * self.dim_feat)
        self.MLP_1 = nn.Linear(4 * self.dim_feat, self.dim_feat)

        self.MLP_u = nn.Linear(self.user_preference_feat.size(1), 4 * self.dim_feat)
        self.MLP_u_1 = nn.Linear(4 * self.dim_feat, self.dim_feat)

    def get_norm_adj_mat(self):
        A = sp.dok_matrix((self.n_users + self.n_items,
                           self.n_users + self.n_items), dtype=np.float32)
        inter_M = self.interaction_matrix
        inter_M_t = self.interaction_matrix.transpose()
        data_dict = dict(zip(zip(inter_M.row, inter_M.col + self.n_users),
                             [1] * inter_M.nnz))
        data_dict.update(dict(zip(zip(inter_M_t.row + self.n_users, inter_M_t.col),
                                  [1] * inter_M_t.nnz)))
        A._update(data_dict)
        # norm adj matrix
        sumArr = (A > 0).sum(axis=1)
        # add epsilon to avoid Devide by zero Warning
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D * A * D
        # covert norm_adj matrix to tensor
        L = sp.coo_matrix(L)
        row = L.row
        col = L.col
        i = torch.LongTensor(np.array([row, col]))
        data = torch.FloatTensor(L.data)

        return torch.sparse.FloatTensor(i, data, torch.Size((self.n_nodes, self.n_nodes)))

    def get_knn_adj_mat_adv(self, mm_embeddings):
        context_norm = mm_embeddings.div(torch.norm(mm_embeddings, p=2, dim=-1, keepdim=True))
        sim = torch.mm(context_norm, context_norm.transpose(1, 0))
        adj = build_knn_neighbourhood(sim, self.knn_k)
        # adj_size = sim.size()
        del sim
        adj[adj < self.pure] = 0
        adj[adj >= self.pure] = 1.

        dense_numpy = self.interaction_matrix.toarray()
        A = torch.tensor(dense_numpy).T
        intersection = torch.mm(A, A.T)
        sum_a = A.sum(dim=1, keepdim=True)
        sum_b = sum_a.T
        union = sum_a + sum_b - intersection
        jaccard = (intersection / (union + 1e-7)).to(self.device)
        jaccard[jaccard == 1.] = 0
        adj_jaccard = build_knn_neighbourhood(jaccard, self.knn_jac)

        mat = adj + adj_jaccard
        ii_mat = get_dense_laplacian(mat, normalization='sym')
        return ii_mat.to_sparse()

    def get_edge_info(self):
        rows = torch.from_numpy(self.interaction_matrix.row)
        cols = torch.from_numpy(self.interaction_matrix.col)
        edges = torch.stack([rows, cols]).type(torch.LongTensor)
        values = self.normalize_adj_g(edges, torch.Size((self.n_users, self.n_items)))
        return edges, values

    def normalize_adj_g(self, indices, adj_size):
        adj = torch.sparse.FloatTensor(indices, torch.ones_like(indices[0]), adj_size)
        row_sum = 1e-7 + torch.sparse.sum(adj, -1).to_dense()
        col_sum = 1e-7 + torch.sparse.sum(adj.t(), -1).to_dense()
        r_inv_sqrt = torch.pow(row_sum, -0.5)
        rows_inv_sqrt = r_inv_sqrt[indices[0]]
        c_inv_sqrt = torch.pow(col_sum, -0.5)
        cols_inv_sqrt = c_inv_sqrt[indices[1]]
        values = rows_inv_sqrt * cols_inv_sqrt
        return values

    def pre_epoch_processing(self):
        if self.drop_g <= .0:
            self.masked_adj = self.norm_adj
            return
        degree_len = int(self.edge_values.size(0) * (1. - self.drop_g))
        degree_idx = torch.multinomial(self.edge_values, degree_len)
        keep_indices = self.edge_indices[:, degree_idx]
        keep_values = self.normalize_adj_g(keep_indices, torch.Size((self.n_users, self.n_items)))
        all_values = torch.cat((keep_values, keep_values))
        keep_indices[1] += self.n_users
        all_indices = torch.cat((keep_indices, torch.flip(keep_indices, [0])), 1)
        self.masked_adj = torch.sparse.FloatTensor(all_indices, all_values, self.norm_adj.shape).to(self.device)

    def UI_GCN(self, mat, user, item):
        embeddings = torch.cat((user, item), dim=0)
        all_embeddings = [embeddings]
        for i in range(self.n_layers_ui):
            side_embeddings = torch.sparse.mm(mat, embeddings)
            embeddings = side_embeddings
            all_embeddings += [embeddings]
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = all_embeddings.sum(dim=1, keepdim=False)
        u_g_embeddings, i_g_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items], dim=0)
        return u_g_embeddings, i_g_embeddings

    def II_GCN(self, mat, embeddings):
        all_embeddings = [embeddings]
        for i in range(self.n_layers):
            side_embeddings = torch.sparse.mm(mat, embeddings)
            embeddings = side_embeddings
            all_embeddings += [embeddings]
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = all_embeddings.sum(dim=1, keepdim=False)
        return all_embeddings

    def forward(self, adj):
        e_u, e_i = self.UI_GCN(adj, self.user_embedding.weight, self.item_embedding.weight)
        e_i = e_i + self.II_GCN(self.ii_mat, e_i)

        h_u_sem = self.MLP_u_1(F.leaky_relu(self.MLP_u(self.user_preference_feat_embedding.weight)))
        e_i_sem = self.MLP_1(F.leaky_relu(self.MLP(self.item_feat_embedding.weight)))
        h_u_sem, e_i_sem = self.UI_GCN(adj, h_u_sem, e_i_sem)
        h_i_sem = e_i_sem + self.II_GCN(self.ii_mat, e_i_sem)

        h_u_sem = self.sem_weight * F.normalize(h_u_sem)
        h_i_sem = self.sem_weight * F.normalize(h_i_sem)

        return e_u, h_u_sem, e_i, h_i_sem

    def bpr_loss(self, users, pos_items, neg_items):
        pos_scores = torch.sum(torch.mul(users, pos_items), dim=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), dim=1)
        maxi = F.logsigmoid(pos_scores - neg_scores)
        mf_loss = -torch.mean(maxi)
        return mf_loss

    def InfoNCE(self, view1, view2, temperature):
        view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)
        pos_score = (view1 * view2).sum(dim=-1)
        pos_score = torch.exp(pos_score / temperature)
        ttl_score = torch.matmul(view1, view2.transpose(0, 1))
        ttl_score = torch.exp(ttl_score / temperature).sum(dim=1)
        cl_loss = -torch.log(pos_score / ttl_score)
        return torch.mean(cl_loss)

    def calculate_loss(self, interaction):
        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]

        u_collab, u_sem, i_collab, i_sem = self.forward(self.masked_adj)


        ua = torch.cat((u_collab, u_sem), dim=1)
        ia = torch.cat((i_collab, i_sem), dim=1)


        u_g = ua[users]
        pos_i_g = ia[pos_items]
        neg_i_g = ia[neg_items]

        batch_mf_loss = self.bpr_loss(u_g, pos_i_g, neg_i_g)


        info_loss = self.info_weight * (self.InfoNCE(u_collab[users], u_sem[users], 0.2)
                                        + self.InfoNCE(i_collab[pos_items], i_sem[pos_items], 0.2))
        return batch_mf_loss + info_loss


    def full_sort_predict(self, interaction):
        user = interaction[0]

        u_collab, u_sem, i_collab, i_sem = self.forward(self.norm_adj)

        restore_user_e = torch.cat((u_collab, u_sem), dim=1)
        restore_item_e = torch.cat((i_collab, i_sem), dim=1)

        u_embeddings = restore_user_e[user]

        # dot with all item embedding to accelerate
        scores = torch.matmul(u_embeddings, restore_item_e.transpose(0, 1))
        return scores

    # def save_embeddings(self):
    #     ua_feat, _ = self.forward(self.norm_adj)
    #     return ua_feat.cpu().detach().numpy()
