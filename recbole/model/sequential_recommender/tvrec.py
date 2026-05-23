# -*- coding: utf-8 -*-
"""
TVRec
################################################

A Graph Signal Processing-based sequential recommender model.

This model uses a graph signal processing approach with learnable basis functions
to capture sequential patterns in user interactions.

"""

import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as fn

from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.loss import BPRLoss


class TVRec(SequentialRecommender):
    """
    TVRec is a sequential recommender based on graph signal processing.

    It uses a spectral approach with learnable basis functions to model
    sequential user behavior patterns.
    """

    def __init__(self, config, dataset):
        super(TVRec, self).__init__(config, dataset)

        # load parameters info
        self.hidden_size = config["hidden_size"]
        self.inner_size = config["inner_size"]
        self.hidden_dropout_prob = config["hidden_dropout_prob"]
        self.hidden_act = config["hidden_act"]
        self.layer_norm_eps = config["layer_norm_eps"]
        self.num_hidden_layers = config["num_hidden_layers"]
        self.M = config["M"]
        self.reg_weight = config["reg_weight"]
        self.initializer_range = config["initializer_range"]
        self.loss_type = config["loss_type"]

        # define layers and loss
        self.item_embedding = nn.Embedding(
            self.n_items, self.hidden_size, padding_idx=0
        )
        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)

        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

        self.item_encoder = NVRecEncoder(
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps,
            num_hidden_layers=self.num_hidden_layers,
            max_seq_length=self.max_seq_length,
            M=self.M,
            device=self.device
        )

        if self.loss_type == "BPR":
            self.loss_fct = BPRLoss()
        elif self.loss_type == "CE":
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")

        # parameters initialization
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

    def get_sequence_emb(self, input_ids, pos_emb=False):
        """Get sequence embeddings with optional position embeddings"""
        item_emb = self.item_embedding(input_ids)
        if pos_emb:
            position_ids = torch.arange(
                input_ids.size(1), dtype=torch.long, device=input_ids.device
            )
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
            position_embedding = self.position_embedding(position_ids)
            item_emb = item_emb + position_embedding
        return self.dropout(self.LayerNorm(item_emb))

    def forward(self, item_seq, item_seq_len, all_sequence_output=False, inference=False):
        sequence_emb = self.get_sequence_emb(item_seq, pos_emb=False)
        pad_mask = (item_seq == 0)

        item_encoded_layers = self.item_encoder(
            [sequence_emb, pad_mask],
            output_all_encoded_layers=True,
            inference=inference
        )

        if all_sequence_output:
            sequence_output = item_encoded_layers
        else:
            sequence_output = item_encoded_layers[-1]

        output = self.gather_indexes(sequence_output, item_seq_len - 1)
        return output

    def orthogonal_regularization(self, B, weight=1.0):
        """Compute orthogonal regularization loss for basis matrix"""
        # B: [m, K+1, 2]
        B_real = B[:, :, 0]  # [m, K+1]
        B_imag = B[:, :, 1]  # [m, K+1]

        def ortho_loss(matrix):
            # Normalize rows
            matrix = nn.functional.normalize(matrix, p=2, dim=1)
            # Compute (B B^T - I)
            gram = matrix @ matrix.T
            I = torch.eye(matrix.size(0), device=matrix.device, dtype=matrix.dtype)
            return ((gram - I) ** 2).sum()

        loss_real = ortho_loss(B_real)
        loss_imag = ortho_loss(B_imag)

        return weight * (loss_real + loss_imag)

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        pos_items = interaction[self.POS_ITEM_ID]

        seq_output = self.forward(item_seq, item_seq_len)

        if self.loss_type == "BPR":
            neg_items = interaction[self.NEG_ITEM_ID]
            pos_items_emb = self.item_embedding(pos_items)
            neg_items_emb = self.item_embedding(neg_items)
            pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)
            neg_score = torch.sum(seq_output * neg_items_emb, dim=-1)
            loss = self.loss_fct(pos_score, neg_score)
        else:  # CE loss
            test_item_emb = self.item_embedding.weight
            logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
            loss = self.loss_fct(logits, pos_items)

        # Add orthogonal regularization
        ortho_reg = 0.0
        for block in self.item_encoder.blocks:
            if hasattr(block.layer, 'basis') and isinstance(block.layer.basis, nn.Parameter):
                basis = block.layer.basis  # [m, K+1, 2]
                ortho_reg += self.orthogonal_regularization(basis, weight=self.reg_weight)

        return loss + ortho_reg

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
        seq_output = self.forward(item_seq, item_seq_len, inference=True)
        test_item_emb = self.item_embedding(test_item)
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)
        return scores

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len, inference=True)
        test_items_emb = self.item_embedding.weight
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))
        return scores


class NVRecEncoder(nn.Module):
    """Encoder composed of multiple NVRecBlock layers"""

    def __init__(self, hidden_size, inner_size, hidden_dropout_prob, hidden_act,
                 layer_norm_eps, num_hidden_layers, max_seq_length, M, device):
        super(NVRecEncoder, self).__init__()

        block = NVRecBlock(
            hidden_size=hidden_size,
            inner_size=inner_size,
            hidden_dropout_prob=hidden_dropout_prob,
            hidden_act=hidden_act,
            layer_norm_eps=layer_norm_eps,
            max_seq_length=max_seq_length,
            M=M,
            device=device
        )
        self.blocks = nn.ModuleList([copy.deepcopy(block) for _ in range(num_hidden_layers)])

    def forward(self, hidden_states, output_all_encoded_layers=False, inference=False):
        hidden_states, pad_mask = hidden_states
        all_encoder_layers = [hidden_states]

        for layer_module in self.blocks:
            hidden_states[pad_mask] = 0.0
            hidden_states = layer_module(hidden_states, inference)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)

        return all_encoder_layers


class NVRecBlock(nn.Module):
    """A single block in the NVRec encoder"""

    def __init__(self, hidden_size, inner_size, hidden_dropout_prob, hidden_act,
                 layer_norm_eps, max_seq_length, M, device):
        super(NVRecBlock, self).__init__()
        self.layer = NVRecLayer(
            hidden_size=hidden_size,
            hidden_dropout_prob=hidden_dropout_prob,
            layer_norm_eps=layer_norm_eps,
            max_seq_length=max_seq_length,
            M=M,
            device=device
        )
        self.feed_forward = TVFeedForward(
            hidden_size=hidden_size,
            inner_size=inner_size,
            hidden_dropout_prob=hidden_dropout_prob,
            hidden_act=hidden_act,
            layer_norm_eps=layer_norm_eps
        )

    def forward(self, hidden_states, inference=False):
        layer_output = self.layer(hidden_states, inference)
        feedforward_output = self.feed_forward(layer_output)
        return feedforward_output


class NVRecLayer(nn.Module):
    """
    Graph signal processing layer using spectral transforms.

    This layer applies a learnable graph filter in the spectral domain
    using a basis of eigenvectors.
    """

    def __init__(self, hidden_size, hidden_dropout_prob, layer_norm_eps,
                 max_seq_length, M, device):
        super(NVRecLayer, self).__init__()
        self.hidden_size = hidden_size
        self.out_dropout = nn.Dropout(hidden_dropout_prob)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.seq_len = max_seq_length
        self.K = 50
        self.pad_seq_len = self.seq_len + self.K - 1
        self.device = device
        self.M = M

        if M > 0:
            self.basis = nn.Parameter(torch.randn(M, self.K + 1, 2, dtype=torch.float32) * 1e-3)
            self.c_t = nn.Parameter(torch.randn(self.pad_seq_len, M, dtype=torch.float32) * 1e-3)
        else:
            self.register_buffer('basis', torch.stack([
                torch.eye(self.seq_len + 1),
                torch.zeros(self.seq_len + 1, self.seq_len + 1)
            ], dim=2))
            self.c_t = nn.Parameter(torch.randn(self.pad_seq_len, self.seq_len + 1, dtype=torch.float32) * 1e-3)

        self.register_buffer('mask', torch.ones(self.pad_seq_len, self.seq_len, 2))
        self.mask[self.seq_len:] = 0

        self.func_H = None
        self.gft, self.igft, self.L = self._init_transform_matrix()

    def _adjacency_matrix(self):
        A = torch.zeros((self.pad_seq_len, self.pad_seq_len))
        A[0, -1] = 1
        A[1:, :-1].fill_diagonal_(1)
        return A.to(self.device)

    def _init_transform_matrix(self):
        A = self._adjacency_matrix()
        lmd, igft_matrix = torch.linalg.eig(A)
        L = torch.stack([lmd ** i for i in range(self.K + 1)], dim=1)
        return torch.linalg.inv(igft_matrix), igft_matrix, L

    def _pad_tensor(self, tensor):
        pad_item = torch.zeros_like(tensor[:, :self.K - 1, :])
        return torch.cat([tensor, pad_item], axis=1)

    def calculate_func_H(self):
        """Pre-compute the filter function for inference"""
        B = torch.view_as_complex(self.basis)
        H = self.c_t.to(torch.complex64) @ (B / torch.norm(B, p=2, dim=1, keepdim=True))
        L = self.L
        F = self.igft * (H @ L.T)
        func_H = F @ self.gft
        self.func_H = torch.real(func_H)

    def forward(self, input_tensor, inference=False):
        # [batch, seq_len, hidden]
        _, seq_len, _ = input_tensor.shape
        padded_tensor = self._pad_tensor(input_tensor)

        if inference:
            if self.func_H is None:
                self.calculate_func_H()
            x = self.func_H @ padded_tensor
        else:
            x_tilde = self.gft @ padded_tensor.to(torch.complex64)
            B = torch.view_as_complex(self.basis)
            H = self.c_t.to(torch.complex64) @ (B / torch.norm(B, p=2, dim=1, keepdim=True))
            L = self.L
            F = self.igft * (H @ L.T)
            x = F @ x_tilde
            x = torch.real(x)

        x = x[:, :seq_len, :]
        hidden_states = self.out_dropout(x)
        hidden_states = hidden_states + input_tensor

        hidden_states = self.LayerNorm(hidden_states)

        return hidden_states


class TVFeedForward(nn.Module):
    """
    Point-wise feed-forward layer with two dense layers.
    """

    def __init__(self, hidden_size, inner_size, hidden_dropout_prob, hidden_act, layer_norm_eps):
        super(TVFeedForward, self).__init__()
        self.dense_1 = nn.Linear(hidden_size, inner_size)
        self.intermediate_act_fn = self._get_hidden_act(hidden_act)
        self.dense_2 = nn.Linear(inner_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def _get_hidden_act(self, act):
        ACT2FN = {
            "gelu": self._gelu,
            "relu": fn.relu,
            "swish": self._swish,
            "tanh": torch.tanh,
            "sigmoid": torch.sigmoid,
        }
        return ACT2FN[act]

    def _gelu(self, x):
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

    def _swish(self, x):
        return x * torch.sigmoid(x)

    def forward(self, input_tensor):
        hidden_states = self.dense_1(input_tensor)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.dense_2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class TVLayerNorm(nn.Module):
    """Layer normalization module"""

    def __init__(self, hidden_size, eps=1e-12):
        super(TVLayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.eps = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight * x + self.bias
