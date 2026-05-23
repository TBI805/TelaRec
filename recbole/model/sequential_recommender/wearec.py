# -*- coding: utf-8 -*-

import torch
from torch import nn
import torch.nn.functional as F
import math
import copy

from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.layers import FeedForward
from recbole.model.loss import BPRLoss


class WEARecLayer(nn.Module):
    def __init__(self, config):
        super(WEARecLayer, self).__init__()

        self.hidden_size = config["hidden_size"]
        self.num_heads = config["n_heads"]

        if self.hidden_size % self.num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")

        self.head_dim = self.hidden_size // self.num_heads
        self.seq_len = config["MAX_ITEM_LIST_LENGTH"]

        # Hyperparameters retrieved directly from config (No .get method)
        self.adaptive = config["adaptive"]
        self.combine_mode = config["combine_mode"]
        self.alpha = config["alpha"]

        # Frequency bins for rFFT: (seq_len//2 + 1)
        self.freq_bins = self.seq_len // 2 + 1

        # Wavelet complex weight
        self.complex_weight = nn.Parameter(
            torch.randn(1, self.num_heads, self.seq_len // 2, self.head_dim, dtype=torch.float32) * 0.02
        )

        # Base multiplicative filter & additive bias for FFT
        self.base_filter = nn.Parameter(torch.ones(self.num_heads, self.freq_bins, 1))
        self.base_bias = nn.Parameter(torch.full((self.num_heads, self.freq_bins, 1), -0.1))

        if self.adaptive:
            # Adaptive MLP: produces 2 values per head & frequency bin
            self.adaptive_mlp = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.GELU(),
                nn.Linear(self.hidden_size, self.num_heads * self.freq_bins * 2)
            )

        if self.combine_mode == 'gate':
            # alpha is used directly for gating
            self.proj_concat = None
        elif self.combine_mode == 'concat':
            # A projection to bring concatenated (local + global) features back to embed_dim
            self.proj_concat = nn.Linear(2 * self.hidden_size, self.hidden_size)
        else:
            raise ValueError("combine_mode must be either 'gate' or 'concat'")

        self.out_dropout = nn.Dropout(config["hidden_dropout_prob"])
        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=config["layer_norm_eps"])

    def wavelet_transform(self, x_heads):
        B, H, N, D = x_heads.shape

        # For simplicity, if N is odd, truncate by one
        N_even = N if (N % 2) == 0 else (N - 1)
        x_heads_trunc = x_heads[:, :, :N_even, :]

        # Split even and odd positions along sequence dimension
        x_even = x_heads_trunc[:, :, 0::2, :]
        x_odd = x_heads_trunc[:, :, 1::2, :]

        # Haar wavelet decomposition
        approx = 0.5 * (x_even + x_odd)
        detail = 0.5 * (x_even - x_odd)

        detail = detail * self.complex_weight

        # Haar wavelet reconstruction
        x_even_recon = approx + detail
        x_odd_recon = approx - detail

        # Interleave even/odd back to original shape
        out = torch.zeros_like(x_heads_trunc)
        out[:, :, 0::2, :] = x_even_recon
        out[:, :, 1::2, :] = x_odd_recon

        # If we truncated one position, pad it back with zeros
        if N_even < N:
            pad = torch.zeros((B, H, 1, D), device=out.device, dtype=out.dtype)
            out = torch.cat([out, pad], dim=2)

        return out

    def forward(self, input_tensor):
        # [batch, seq_len, hidden]
        batch, seq_len, hidden = input_tensor.shape

        # Reshape to separate heads: (B, num_heads, seq_len, head_dim)
        x_heads = input_tensor.view(batch, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # ---- (1) FFT-based global features ----
        F_fft = torch.fft.rfft(x_heads, dim=2, norm='ortho')

        if self.adaptive:
            context = input_tensor.mean(dim=1)
            adapt_params = self.adaptive_mlp(context)
            adapt_params = adapt_params.view(batch, self.num_heads, self.freq_bins, 2)
            adaptive_scale = adapt_params[..., 0:1]
            adaptive_bias = adapt_params[..., 1:2]
        else:
            adaptive_scale = torch.zeros(batch, self.num_heads, self.freq_bins, 1, device=input_tensor.device)
            adaptive_bias = torch.zeros(batch, self.num_heads, self.freq_bins, 1, device=input_tensor.device)

        effective_filter = self.base_filter * (1 + adaptive_scale)
        effective_bias = self.base_bias + adaptive_bias

        F_fft_mod = F_fft * effective_filter + effective_bias

        x_fft = torch.fft.irfft(F_fft_mod, dim=2, n=self.seq_len, norm='ortho')

        # ---- (2) Wavelet-based local features ----
        x_wavelet = self.wavelet_transform(x_heads)

        # ---- (3) Combine local/global ----
        if self.combine_mode == 'gate':
            x_combined = (1.0 - self.alpha) * x_wavelet + self.alpha * x_fft
        else:  # concat
            x_wavelet_reshaped = x_wavelet.permute(0, 2, 1, 3).contiguous().reshape(batch, seq_len, hidden)
            x_fft_reshaped = x_fft.permute(0, 2, 1, 3).contiguous().reshape(batch, seq_len, hidden)

            x_cat = torch.cat([x_wavelet_reshaped, x_fft_reshaped], dim=-1)
            x_combined_proj = self.proj_concat(x_cat)

            x_combined = x_combined_proj.view(batch, seq_len, self.num_heads, self.head_dim)
            x_combined = x_combined.permute(0, 2, 1, 3)

        # Reshape: merge heads back into the embedding dimension.
        x_out = x_combined.permute(0, 2, 1, 3).contiguous().reshape(batch, seq_len, hidden)

        hidden_states = self.out_dropout(x_out)
        hidden_states = hidden_states + input_tensor
        hidden_states = self.LayerNorm(hidden_states)

        return hidden_states


class WEARecBlock(nn.Module):
    def __init__(self, config):
        super(WEARecBlock, self).__init__()
        self.layer = WEARecLayer(config)
        self.feed_forward = FeedForward(
            config["hidden_size"],
            config["inner_size"],
            config["hidden_dropout_prob"],
            config["hidden_act"],
            config["layer_norm_eps"]
        )

    def forward(self, hidden_states):
        layer_output = self.layer(hidden_states)
        feedforward_output = self.feed_forward(layer_output)
        return feedforward_output


class WEARecEncoder(nn.Module):
    def __init__(self, config):
        super(WEARecEncoder, self).__init__()
        block = WEARecBlock(config)
        self.blocks = nn.ModuleList([copy.deepcopy(block) for _ in range(config["n_layers"])])

    def forward(self, hidden_states, output_all_encoded_layers=False):
        all_encoder_layers = [hidden_states]

        for layer_module in self.blocks:
            hidden_states = layer_module(hidden_states)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)

        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)

        return all_encoder_layers


class WEARec(SequentialRecommender):
    def __init__(self, config, dataset):
        super(WEARec, self).__init__(config, dataset)

        # Config Parsing
        self.n_layers = config["n_layers"]
        self.n_heads = config["n_heads"]
        self.hidden_size = config["hidden_size"]
        self.inner_size = config["inner_size"]
        self.hidden_dropout_prob = config["hidden_dropout_prob"]
        self.layer_norm_eps = config["layer_norm_eps"]
        self.initializer_range = config["initializer_range"]
        self.loss_type = config["loss_type"]

        # Embeddings & Encoder
        self.item_embedding = nn.Embedding(self.n_items, self.hidden_size, padding_idx=0)
        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)
        self.item_encoder = WEARecEncoder(config)

        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

        # Loss Function
        if self.loss_type == "BPR":
            self.loss_fct = BPRLoss()
        elif self.loss_type == "CE":
            self.loss_fct = nn.CrossEntropyLoss(label_smoothing=0.1)
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, item_seq, item_seq_len):
        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)

        item_emb = self.item_embedding(item_seq)
        input_emb = item_emb + position_embedding

        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        item_encoded_layers = self.item_encoder(input_emb, output_all_encoded_layers=True)

        output = item_encoded_layers[-1]
        output = self.gather_indexes(output, item_seq_len - 1)
        return output

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        pos_items = interaction[self.POS_ITEM_ID]

        if self.loss_type == "BPR":
            neg_items = interaction[self.NEG_ITEM_ID]
            pos_items_emb = self.item_embedding(pos_items)
            neg_items_emb = self.item_embedding(neg_items)
            pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)
            neg_score = torch.sum(seq_output * neg_items_emb, dim=-1)
            loss = self.loss_fct(pos_score, neg_score)
            return loss
        else:  # self.loss_type == 'CE'
            test_item_emb = self.item_embedding.weight
            logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
            loss = self.loss_fct(logits, pos_items)
            return loss

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]

        seq_output = self.forward(item_seq, item_seq_len)
        test_item_emb = self.item_embedding(test_item)
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)
        return scores

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        test_items_emb = self.item_embedding.weight
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))
        return scores