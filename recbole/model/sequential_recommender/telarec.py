import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.loss import BPRLoss


class ThermoelasticLayer(nn.Module):
    def __init__(
        self,
        d_model,
        seq_len,
        dropout=0.0,
        c_init=1.0,
        alpha_init=0.1,
        kappa_init=0.5,
        spectral_dropout=0.1,
        lpf_ratio=0.1,
        tau_min=0.01,
    ):
        super().__init__()

        self.d_model = d_model
        self.seq_len = seq_len
        self.spectral_dropout = spectral_dropout
        self.lpf_ratio = lpf_ratio
        self.tau_min = tau_min

        self.linear = nn.Linear(d_model, 4 * d_model, bias=True)

        self.n_freq = seq_len // 2 + 1

        self.tau_base = nn.Parameter(torch.zeros(self.n_freq, 1))
        nn.init.trunc_normal_(self.tau_base, std=0.02)

        self.content_scale = nn.Parameter(torch.zeros(1))

        raw_c_init = math.log(math.expm1(max(c_init, 1e-6)))
        self.raw_c = nn.Parameter(torch.ones(self.n_freq, 1) * raw_c_init)

        self.alpha = nn.Parameter(torch.ones(1) * alpha_init)
        self.kappa = nn.Parameter(torch.ones(1) * kappa_init)
        self.beta = nn.Parameter(torch.ones(1) * 0.1)

        self.out_norm = nn.LayerNorm(d_model)
        self.out_linear = nn.Linear(d_model, d_model, bias=True)
        self.out_dropout = nn.Dropout(dropout)

        self.ffn = InertialMixer(d_model=d_model, hidden_rate=4, dropout=dropout)
        self._cached_heat_decay = None
        self._cached_L = None

    @staticmethod
    def fft1d(x):
        return torch.fft.rfft(x, dim=1, norm="ortho")

    @staticmethod
    def ifft1d(x, n):
        return torch.fft.irfft(x, n=n, dim=1, norm="ortho")

    @staticmethod
    def _compute_heat_decay(n_freq, device, dtype):
        weight_n = torch.linspace(0, math.pi, n_freq, device=device, dtype=dtype)
        return torch.exp(-torch.pow(weight_n, 2))

    def _get_heat_decay(self, n_freq, device, dtype):
        if (
            (self._cached_L != n_freq)
            or (self._cached_heat_decay is None)
            or (self._cached_heat_decay.device != device)
            or (self._cached_heat_decay.dtype != dtype)
        ):
            self._cached_heat_decay = self._compute_heat_decay(
                n_freq, device, dtype
            ).detach()
            self._cached_L = n_freq

        return self._cached_heat_decay

    def forward(self, input_tensor):
        B, L, D = input_tensor.shape

        xz = self.linear(input_tensor)
        x_u, x_v, x_theta, z = xz.chunk(4, dim=-1)

        Theta0 = self.fft1d(x_theta)
        U0 = self.fft1d(x_u)
        V0 = self.fft1d(x_v)

        n_freq = U0.shape[1]

        # Heat branch
        heat_decay = self._get_heat_decay(
            n_freq, input_tensor.device, input_tensor.dtype
        )

        heat_kernel = torch.pow(
            heat_decay.view(1, n_freq, 1),
            torch.abs(self.kappa),
        )
        Theta = heat_kernel * Theta0

        # UFC
        spec_context = torch.log1p(torch.abs(U0) + torch.abs(V0))
        spec_context = spec_context.mean(dim=-1, keepdim=True)
        tau_base = self.tau_base[:n_freq, :].unsqueeze(0)
        tau = self.tau_min + torch.abs(
            tau_base + self.content_scale * spec_context
        )

        c = F.softplus(self.raw_c[:n_freq, :]).unsqueeze(0) + 1e-6
        phase = c * tau
        cos_term = torch.cos(phase)
        sin_term = torch.sin(phase) / c

        # Wave branch
        U_wave = cos_term * U0 + sin_term * (
            V0 + (self.alpha / 2.0) * U0
        )

        u = U_wave - self.beta * Theta

        if self.spectral_dropout > 0.0:
            mask = torch.ones_like(u.real)
            mask = F.dropout(mask, p=self.spectral_dropout, training=self.training)
            u = u * mask

        if self.lpf_ratio > 0.0 and n_freq > 1:
            cutoff = int(n_freq * (1.0 - self.lpf_ratio))
            u[:, cutoff:, :] = 0.0

        x_out = self.ifft1d(u, n=L)

        x_out = self.out_norm(x_out)
        x_out = x_out * F.silu(z)
        x_out = self.out_linear(x_out)

        hidden_states = self.out_dropout(x_out)
        hidden_states = hidden_states + input_tensor

        return self.ffn(hidden_states)

class InertialMixer(nn.Module):
    def __init__(self, d_model, hidden_rate=4, dropout=0.2):
        super().__init__()
        self.d_model = d_model
        self.x_k = nn.Parameter(torch.zeros(self.d_model))

        hidden_sz = int(hidden_rate * self.d_model)
        self.key = nn.Linear(self.d_model, hidden_sz, bias=True)
        self.value = nn.Linear(hidden_sz, self.d_model, bias=True)

        self.dropout = nn.Dropout(dropout)
        self.LayerNorm = nn.LayerNorm(d_model, eps=1e-12)

    def sequence_shift(self, x):
        xx = F.pad(x, (0, 0, 1, 0), mode="constant", value=0)
        return xx[:, :-1, :]

    def forward(self, input_tensor):
        delta = self.sequence_shift(input_tensor)
        x_processed = input_tensor.addcmul(delta, self.x_k)

        k = self.key(x_processed)
        k_activated = torch.square(torch.relu(k))
        kv = self.value(k_activated)

        hidden = self.dropout(kv)
        hidden = self.LayerNorm(hidden + input_tensor)

        return hidden

class TelaRec(SequentialRecommender):
    def __init__(self, config, dataset):
        super(TelaRec, self).__init__(config, dataset)

        self.hidden_size = config["hidden_size"]
        self.loss_type = config["loss_type"]
        self.num_layers = config["num_layers"]
        self.dropout_prob = config["dropout_prob"]
        self.seq_len = config["MAX_ITEM_LIST_LENGTH"]

        self.c_init = config["c_init"]
        self.alpha_init = config["alpha_init"]
        self.kappa_init = config["kappa_init"]
        self.spectral_dropout_prob = config["spectral_dropout_prob"]
        self.lpf_ratio = config["lpf_ratio"]

        self.item_embedding = nn.Embedding(
            self.n_items, self.hidden_size, padding_idx=0
        )

        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(self.dropout_prob)

        self.telarec_layers = nn.ModuleList(
            [
                ThermoelasticLayer(
                    d_model=self.hidden_size,
                    seq_len=self.seq_len,
                    dropout=self.dropout_prob,
                    c_init=self.c_init,
                    alpha_init=self.alpha_init,
                    kappa_init=self.kappa_init,
                    spectral_dropout=self.spectral_dropout_prob,
                    lpf_ratio=self.lpf_ratio,
                )
                for _ in range(self.num_layers)
            ]
        )

        if self.loss_type == "BPR":
            self.loss_fct = BPRLoss()
        elif self.loss_type == "CE":
            self.loss_fct = nn.CrossEntropyLoss(label_smoothing=0.1)
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, item_seq, item_seq_len):
        item_emb = self.item_embedding(item_seq)
        item_emb = self.dropout(item_emb)
        item_emb = self.LayerNorm(item_emb)

        for layer in self.telarec_layers:
            item_emb = layer(item_emb)

        seq_output = self.gather_indexes(item_emb, item_seq_len - 1)
        return seq_output

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

        else:
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