import os
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F


from common.abstract_recommender import GeneralRecommender
from utils.utils import build_knn_neighbourhood, get_dense_laplacian

class MLLMRec(GeneralRecommender):
    def __init__(self, config, dataset):
        super(MLLMRec, self).__init__(config, dataset)
        self.n_nodes = self.n_users + self.n_items
        self.dim_feat = config['feat_embed_dim']
        self.n_layers = config['n_ii_layers']
        self.n_layers_ui = config['n_ui_layers']
        self.knn_k = config['knn_k']
        self.knn_jac = config['knn_jac']
        self.pure = config['pure']

        # load dataset info
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)

        dataset_path = os.path.abspath(config['data_path'] + config['dataset'])
        ii_mat_file = os.path.join(dataset_path + '/', 'ii_mat_{}_{}.pt'.format(self.knn_jac, self.pure))

        self.item_feat_embedding = nn.Embedding.from_pretrained(self.item_feat)
        self.user_preference_feat_embedding = nn.Embedding.from_pretrained(self.user_preference_feat)
        if os.path.exists(ii_mat_file):
            self.ii_mat = torch.load(ii_mat_file)
        else:
            self.ii_mat = self.get_knn_adj_mat_adv(self.item_feat_embedding.weight.detach())
            torch.save(self.ii_mat, ii_mat_file)


        self.norm_adj = self.get_norm_adj_mat().to(self.device)

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

    def pre_epoch_processing(self):
        pass

    def UI_GCN(self, mat, user, item):
        embeddings = torch.cat((user, item), dim=0)
        embeddings = F.normalize(embeddings)
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
        h_u = self.MLP_u_1(F.leaky_relu(self.MLP_u(self.user_preference_feat_embedding.weight)))

        e_i_1 = self.II_GCN(self.ii_mat, self.item_feat_embedding.weight)
        h_i = self.MLP_1(F.leaky_relu(self.MLP(e_i_1)))

        ### w/o GCN_II
        # h_i = self.MLP_1(F.leaky_relu(self.MLP(self.item_feat_embedding.weight)))

        ### w/ GCN_UI
        # z_u, z_i = self.UI_GCN(adj, h_u, h_i)
        # h_u = h_u + z_u
        # h_i = h_i + z_i

        return h_u, h_i

    def bpr_loss(self, users, pos_items, neg_items):
        pos_scores = torch.sum(torch.mul(users, pos_items), dim=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), dim=1)
        maxi = F.logsigmoid(pos_scores - neg_scores)
        mf_loss = -torch.mean(maxi)
        return mf_loss

    def calculate_loss(self, interaction):
        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]

        ua_feat, ia_feat = self.forward(self.norm_adj)

        u_g_feat = ua_feat[users]
        pos_i_g_feat = ia_feat[pos_items]
        neg_i_g_feat = ia_feat[neg_items]

        batch_mf_loss = self.bpr_loss(u_g_feat, pos_i_g_feat, neg_i_g_feat)

        return batch_mf_loss


    def full_sort_predict(self, interaction):
        user = interaction[0]

        restore_user_e, restore_item_e = self.forward(self.norm_adj)
        u_embeddings = restore_user_e[user]

        # dot with all item embedding to accelerate
        scores = torch.matmul(u_embeddings, restore_item_e.transpose(0, 1))
        return scores

    # def save_embeddings(self):
    #     ua_feat, _ = self.forward(self.norm_adj)
    #     return ua_feat.cpu().detach().numpy()