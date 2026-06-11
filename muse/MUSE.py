# -*- coding: utf-8 -*-
# @Time    : 2020/9/18 11:33
# @Author  : Hui Wang
# @Email   : hui.wang@ruc.edu.cn

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.layers import TransformerEncoder
from recbole.model.loss import BPRLoss
from sklearn.cluster import MiniBatchKMeans
import copy

class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate):

        super(PointWiseFeedForward, self).__init__()

        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2) # as Conv1D requires (N, C, Length)
        outputs += inputs
        return outputs

class MUSE(SequentialRecommender):
    r"""
    SASRec is the first sequential recommender based on self-attentive mechanism.

    NOTE:
        In the author's implementation, the Point-Wise Feed-Forward Network (PFFN) is implemented
        by CNN with 1x1 kernel. In this implementation, we follows the original BERT implementation
        using Fully Connected Layer to implement the PFFN.
    """
    def __init__(self, config, dataset, co_data, colens):
        super(MUSE, self).__init__(config, dataset)
        
        self.device = config['device']
        self.co_seq = F.normalize(self.get_co(co_data), dim=1).to(self.device)
        self.co_seq[0, :] = 0
        self.co_seq[:, 0] = 0

        self.counts = torch.bincount(co_data.flatten(), minlength=co_data.max()+1)
        self.counts[0] = 0
    
        # load parameters info
        self.n_layers = config["n_layers"]
        self.n_heads = config["n_heads"]
        self.hidden_size = config["hidden_size"]  # same as embedding_size
        self.inner_size = config[
            "inner_size"
        ]  # the dimensionality in feed-forward layer
        self.hidden_dropout_prob = config["hidden_dropout_prob"]
        self.attn_dropout_prob = config["attn_dropout_prob"]
        self.hidden_act = config["hidden_act"]
        self.layer_norm_eps = config["layer_norm_eps"]
        self.emb_dropout = torch.nn.Dropout(p=0.5)

        self.initializer_range = config["initializer_range"]
        self.loss_type = config["loss_type"]

        self.fre = 2.5
        self.low_r = self.max_seq_length // self.fre
        self.high_r = self.max_seq_length // self.fre
        self.LPA = self.createLPAilter((self.max_seq_length, self.hidden_size), self.low_r)
        self.HPA = self.createHPAilter((self.max_seq_length, self.hidden_size), self.high_r)
        self.lmd = config['lmd']
        self.tau = config['tau']
        self.sim = config['sim']
        self.mask_default = self.mask_correlated_samples(batch_size=256)

        self.attention_layernorms = torch.nn.ModuleList() # to be Q for self-attention
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()

        self.attention_layernorms_l = torch.nn.ModuleList() # to be Q for self-attention
        self.attention_layers_l = torch.nn.ModuleList()
        self.forward_layernorms_l = torch.nn.ModuleList()
        self.forward_layers_l = torch.nn.ModuleList()

        self.attention_layernorms_h = torch.nn.ModuleList() # to be Q for self-attention
        self.attention_layers_h = torch.nn.ModuleList()
        self.forward_layernorms_h = torch.nn.ModuleList()
        self.forward_layers_h = torch.nn.ModuleList()

        self.last_layernorm = torch.nn.LayerNorm(self.hidden_size, eps=1e-8)

        for _ in range(2):
            new_attn_layernorm = torch.nn.LayerNorm(self.hidden_size, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer =  torch.nn.MultiheadAttention(self.hidden_size,
                                                            2,
                                                            self.hidden_dropout_prob)
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(self.hidden_size, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(self.hidden_size, self.hidden_dropout_prob)
            self.forward_layers.append(new_fwd_layer)

        for _ in range(2):
            new_attn_layernorm = torch.nn.LayerNorm(self.hidden_size, eps=1e-8)
            self.attention_layernorms_l.append(new_attn_layernorm)

            new_attn_layer =  torch.nn.MultiheadAttention(self.hidden_size,
                                                            2,
                                                            self.hidden_dropout_prob)
            self.attention_layers_l.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(self.hidden_size, eps=1e-8)
            self.forward_layernorms_l.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(self.hidden_size, self.hidden_dropout_prob)
            self.forward_layers_l.append(new_fwd_layer)

        for _ in range(2):
            new_attn_layernorm = torch.nn.LayerNorm(self.hidden_size, eps=1e-8)
            self.attention_layernorms_h.append(new_attn_layernorm)

            new_attn_layer =  torch.nn.MultiheadAttention(self.hidden_size,
                                                            2,
                                                            self.hidden_dropout_prob)
            self.attention_layers_h.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(self.hidden_size, eps=1e-8)
            self.forward_layernorms_h.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(self.hidden_size, self.hidden_dropout_prob)
            self.forward_layers_h.append(new_fwd_layer)

        # define layers and loss
        self.item_embedding = nn.Embedding(
            self.n_items, self.hidden_size, padding_idx=0
        )
        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)
        self.trm_encoder = TransformerEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps,
        )

        self.kk = 25
        self.kmeans = MiniBatchKMeans(n_clusters=self.kk, init_size=512, batch_size=512, random_state=2023)
        self.sense, self.sample = self.get_center(self.item_embedding.weight)
        self.vol_weight = 0.5
        self.sta_weight = 0.5
        

        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

        if self.loss_type == "BPR":
            self.loss_fct = BPRLoss()
        elif self.loss_type == "CE":
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")

        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
    
    def get_co(self, seqs):
        len = (seqs > 0).sum(dim=1)
        co_mat = torch.zeros(self.n_items,self.n_items)
        for i in range(seqs.shape[0]):
            for k in range(len[i]):
                for j in range(k+1,len[i]):
                    co_mat[seqs[i][k]][seqs[i][j]] +=1/((j-k))
                    co_mat[seqs[i][j]][seqs[i][k]] +=1/((j-k))
        return co_mat

    def get_center(self,embs):
        means = embs.detach().cpu().numpy()
        self.kmeans.fit(means)
        sample = torch.tensor(self.kmeans.labels_)
        # print(sample)
        o = torch.zeros(1, self.n_items).to(self.device)
        for i in range(max(sample)+1):
            op=copy.deepcopy(sample).unsqueeze(0).to(self.device)
            for j in range(self.n_items-1):
                if op[0,j]==i:
                    op[0,j]=1
                else:
                    op[0,j]=0
            o=torch.cat((o, op), 0)
        sense=o[1:]
        sense = sense/(1e-7+torch.sum(sense,dim=-1).unsqueeze(1))
        return sense,sample

    def get_more(self, result):
        most_frequent = torch.full((result.shape[0],), -1, dtype=result.dtype) 

        for i in range(result.shape[0]):
            row = result[i]
            positive_values = row[row > 0] 
            
            if positive_values.numel() > 0: 
                mode_value, _ = torch.mode(positive_values) 
                most_frequent[i] = mode_value
        return most_frequent

    def get_low_au(self, log_seqs, high_up, cate, item_seq_len):
        t = torch.zeros_like(log_seqs)
        k = (item_seq_len * 0.1).long()
        for i in range(t.shape[0]):
            if k[i] > 0:  
                topk_indices = torch.topk(high_up[i], k[i], largest=True).indices
                z_values = log_seqs[i, topk_indices]
                indexed_y1 = self.sample[z_values]
                mask_condition = (indexed_y1 == cate[i]).to(torch.bool)
                t[i, topk_indices[~mask_condition]] = -(log_seqs[i, topk_indices[~mask_condition]]+1)
        return t + log_seqs

    def forward(self, log_seqs1, item_seq_len): 
        log_seqs = log_seqs1.cpu()
        result = log_seqs.clone()
        result[log_seqs > 0] = self.sample[log_seqs[log_seqs > 0].long()].to(result.dtype)
        cate = self.get_more(result)
        ID = self.item_embedding.weight.cuda()

        high_up = torch.softmax(self.counts[log_seqs].float(), dim=1)
        low_ind = self.get_low_au(log_seqs, high_up, cate, item_seq_len)
        log_seqs = log_seqs.cpu().numpy()
        seqs = self.item_embedding(torch.LongTensor(log_seqs).to(self.device))
        seqs *= self.item_embedding.embedding_dim ** 0.5
        poss = np.tile(np.array(range(log_seqs.shape[1])), [log_seqs.shape[0], 1])

        seqs += self.position_embedding(torch.LongTensor(poss).to(self.device))
        timeline_mask = torch.BoolTensor(log_seqs == 0).to(self.device)

        seqs_low = seqs.clone()
        center_emb = self.sense.cuda() @ ID
        result = torch.bmm(seqs_low, center_emb.T.unsqueeze(0).repeat(seqs.shape[0], 1, 1))  
        result = torch.bmm(result, center_emb.unsqueeze(0).repeat(seqs.shape[0], 1, 1))  
        seqs_low = seqs_low + self.sta_weight * result 

        high_sen = self.co_seq[log_seqs1]
        high_sen = torch.bmm(high_sen, ID.unsqueeze(0).repeat(seqs.shape[0], 1, 1))
        seqs_high = seqs.clone()  
        seqs_high = seqs_high.cuda() * high_up.unsqueeze(-1).cuda() + self.vol_weight * high_sen

        seqs = self.emb_dropout(seqs)   
        seqs_low = self.emb_dropout(seqs_low) 
        seqs_high = self.emb_dropout(seqs_high)    

        tl = seqs.shape[1] # time dim len for enforce causality
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.device))

        seqs *= ~timeline_mask.unsqueeze(-1)  
        seqs_low *= ~timeline_mask.unsqueeze(-1) 
        seqs_high *= ~timeline_mask.unsqueeze(-1)  
        seqs_low[low_ind == -1] = 0
        seqs_l = seqs.clone() 
        seqs_h = seqs.clone() 
        for i in range(len(self.attention_layers)):
            seqs = torch.transpose(seqs, 0, 1)
            Q = self.attention_layernorms[i](seqs)
            mha_outputs, _ = self.attention_layers[i](Q, seqs, seqs, 
                                            attn_mask=attention_mask)                                           
            seqs = Q + mha_outputs
            seqs = torch.transpose(seqs, 0, 1)

            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)

            seqs_l = torch.transpose(seqs_l, 0, 1)
            seqs_low = torch.transpose(seqs_low, 0, 1)
            Q_low = self.attention_layernorms_l[i](seqs_low)           
            mha_outputs_l, _ = self.attention_layers_l[i](Q_low, seqs_low, seqs_low, 
                                            attn_mask=attention_mask)
            seqs_low = Q_low + mha_outputs_l
            seqs_low = torch.transpose(seqs_low, 0, 1)

            seqs_l = self.forward_layernorms_l[i](seqs_low)
            seqs_l = self.forward_layers_l[i](seqs_l)

            seqs_h = torch.transpose(seqs_h, 0, 1)
            seqs_high = torch.transpose(seqs_high, 0, 1)
            Q_high = self.attention_layernorms_h[i](seqs_high) 
            mha_outputs_h, _ = self.attention_layers_h[i](Q_high, seqs_high, seqs_high, 
                                            attn_mask=attention_mask)
            seqs_high = Q_high + mha_outputs_h
            seqs_high = torch.transpose(seqs_high, 0, 1)

            seqs_h = self.forward_layernorms_h[i](seqs_high)
            seqs_h = self.forward_layers_h[i](seqs_h)  

        output = self.last_layernorm(seqs) 
        output_aug_l = self.last_layernorm(seqs_l) 
        output_aug_h = self.last_layernorm(seqs_h) 

        output_a_l = self.fft_2(output_aug_l, self.LPA) 
        output_a_h = self.fft_2(output_aug_h, self.HPA)       

        high_weight = torch.norm(torch.abs(torch.fft.fft(output_aug_h, dim=1)), dim=(1, 2))
        low_weight = torch.norm(torch.abs(torch.fft.fft(output_aug_l, dim=1)), dim=(1, 2))

        softmax_result = torch.nn.functional.softmax(torch.stack(((low_weight - low_weight.mean()), high_weight - high_weight.mean()), dim=0), dim = 0)    
        
        output_i = self.gather_indexes(output, item_seq_len - 1)
        output_l = self.gather_indexes(output_aug_l, item_seq_len - 1)
        output_h = self.gather_indexes(output_aug_h, item_seq_len - 1)
        output_a_l = self.gather_indexes(output_a_l, item_seq_len - 1)
        output_a_h = self.gather_indexes(output_a_h, item_seq_len - 1)        

        return output_i, output_l, output_h, softmax_result[0].view(-1, 1), softmax_result[1].view(-1, 1), output_a_l, output_a_h, output # [B H]

    def calculate_loss(self, interaction):
        time_slow = torch.linspace(0, 6, 50)  
        time_slow = time_slow.unsqueeze(1).repeat(1, 128)  

        time_fast = torch.randn(50)  
        time_fast = time_fast.unsqueeze(1).repeat(1, 128)  
 
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output, output_l, output_h, low_weight, high_weight, output_a_l, output_a_h, output = self.forward(item_seq, item_seq_len)
        pos_items = interaction[self.POS_ITEM_ID]
        if self.loss_type == "BPR":
            neg_items = interaction[self.NEG_ITEM_ID]
            pos_items_emb = self.item_embedding(pos_items)
            neg_items_emb = self.item_embedding(neg_items)
            pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)  # [B]
            neg_score = torch.sum(seq_output * neg_items_emb, dim=-1)  # [B]
            loss = self.loss_fct(pos_score, neg_score)
            return loss
        else:  # self.loss_type = 'CE'
            test_item_emb = self.item_embedding.weight
            test_item_emb_l = self.zero_out_rows_by_frequency(test_item_emb, self.counts, 1//self.fre, 'low')
            test_item_emb_h = self.zero_out_rows_by_frequency(test_item_emb, self.counts, 1//self.fre, 'high')            
            # logits = torch.matmul(seq_output, (test_item_emb - test_item_emb_h - test_item_emb_l).transpose(0, 1))
            logits = torch.matmul(seq_output, (test_item_emb).transpose(0, 1))
            loss = self.loss_fct(logits, pos_items)
            logits = high_weight * torch.matmul(output_h, test_item_emb.transpose(0, 1))
            loss += self.loss_fct(logits, pos_items)
            logits = low_weight * torch.matmul(output_l, test_item_emb.transpose(0, 1))
            loss += self.loss_fct(logits, pos_items)
            nce_loss_l = self.ncelosss(self.tau, output_l.device, output_l, output_a_l)
            nce_loss_h = self.ncelosss(self.tau, output_l.device, output_h, output_a_h)            
            return loss + 0.1 * self.lmd * nce_loss_l + 0.1 * self.lmd * nce_loss_h
        
    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output, output_l, output_h, low_weight, high_weight, output_a_l, output_a_h, output = self.forward(item_seq, item_seq_len)
        test_items_emb = self.item_embedding.weight 
        test_items_emb_l = self.zero_out_rows_by_frequency(test_items_emb, self.counts, 1//self.fre, 'low')
        test_items_emb_h = self.zero_out_rows_by_frequency(test_items_emb, self.counts, 1//self.fre, 'high')
        scores = torch.matmul(seq_output, (test_items_emb).transpose(0, 1))  # [B n_items]
        # scores = torch.matmul(seq_output, (test_items_emb - test_items_emb_h - test_items_emb_l).transpose(0, 1))  # [B n_items]
        scores += high_weight * torch.matmul(output_h, test_items_emb.transpose(0, 1))  # [B n_items]
        scores += low_weight * torch.matmul(output_l, test_items_emb.transpose(0, 1))  # [B n_items]
        return scores

    def ncelosss(self, temperature, device, batch_sample_one, batch_sample_two):
        self.device = device
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.temperature = temperature
        b_size = batch_sample_one.shape[0]
        batch_sample_one = batch_sample_one.view(b_size, -1)
        batch_sample_two = batch_sample_two.view(b_size, -1)

        self.cossim = nn.CosineSimilarity(dim=-1).to(self.device)
        sim11 = torch.matmul(batch_sample_one, batch_sample_one.T) / self.temperature
        sim22 = torch.matmul(batch_sample_two, batch_sample_two.T) / self.temperature
        sim12 = torch.matmul(batch_sample_one, batch_sample_two.T) / self.temperature
        d = sim12.shape[-1]
        sim11[..., range(d), range(d)] = float('-inf')
        sim22[..., range(d), range(d)] = float('-inf')
        raw_scores1 = torch.cat([sim12, sim11], dim=-1)
        raw_scores2 = torch.cat([sim22, sim12.transpose(-1, -2)], dim=-1)
        logits = torch.cat([raw_scores1, raw_scores2], dim=-2)
        labels = torch.arange(2 * d, dtype=torch.long, device=logits.device)
        nce_loss = self.criterion(logits, labels)
        return nce_loss

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    def info_nce(self, z_i, z_j, temp, batch_size, sim='dot'):
        """
        We do not sample negative examples explicitly.
        Instead, given a positive pair, similar to (Chen et al., 2017), we treat the other 2(N − 1) augmented examples within a minibatch as negative examples.
        """
        N = 2 * batch_size
    
        z = torch.cat((z_i, z_j), dim=0)
    
        if sim == 'cos':
            sim = nn.functional.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2) / temp
        elif sim == 'dot':
            sim = torch.mm(z, z.T) / temp
    
        sim_i_j = torch.diag(sim, batch_size)
        sim_j_i = torch.diag(sim, -batch_size)
    
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        if batch_size != 256:
            mask = self.mask_correlated_samples(batch_size)
        else:
            mask = self.mask_default
        negative_samples = sim[mask].reshape(N, -1)
    
        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        return logits, labels

    def fft_2(self, x, filter):
        f = torch.fft.fft2(x)
        fshift = torch.fft.fftshift(f)
        return torch.abs(torch.fft.ifft2(torch.fft.ifftshift(fshift.cuda() * filter.cuda())))

    def zero_out_rows_by_frequency(self, c, b, ratio, fre):
        d = c.detach().clone()
        b_no_zero = b[1:]
        threshold = int(len(b_no_zero) * (1-ratio))
        if fre == 'high':
            _, indices = torch.topk(b_no_zero, threshold, largest=False)
        else:
            _, indices = torch.topk(b_no_zero, threshold, largest=True)
        indices = indices + 1
        d[indices, :] = 0
        return d
    
    def createLPAilter(self, shape, bandCenter):
        rows, cols = shape

        xx = torch.arange(0, cols, 1)
        yy = torch.arange(0, rows, 1)
        x = xx.repeat(rows, 1)
        y = yy.repeat(cols, 1).T

        x = x - cols // 2
        y = y - rows // 2

        d = (x.pow(2) + y.pow(2)).sqrt()

        lpFilter = torch.ones((rows, cols))
        lpFilter[d > bandCenter] = 0

        return lpFilter

    def createHPAilter(self, shape, bandCenter):
        rows, cols = shape

        xx = torch.arange(0, cols, 1)
        yy = torch.arange(0, rows, 1)
        x = xx.repeat(rows, 1)
        y = yy.repeat(cols, 1).T

        x = x - cols // 2
        y = y - rows // 2

        d = (x.pow(2) + y.pow(2)).sqrt()

        hpFilter = torch.ones((rows, cols))
        hpFilter[d < bandCenter] = 0

        return hpFilter
    
    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
        seq_output, output_l, output_h = self.forward(item_seq, item_seq_len)
        test_item_emb = self.item_embedding(test_item)
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)  # [B]
        return scores
 





