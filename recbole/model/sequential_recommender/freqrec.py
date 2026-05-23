import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.loss import BPRLoss
from recbole.model.layers import TransformerEncoder, FeedForward


class FreqRec(SequentialRecommender):
    def __init__(self, config, dataset):
        super(FreqRec, self).__init__(config, dataset)

        # load parameters info
        self.hidden_size = config['hidden_size']
        self.num_hidden_layers = config['num_hidden_layers']
        self.num_attention_heads = config['num_attention_heads']
        self.inner_size = config['inner_size']
        self.hidden_dropout_prob = config['hidden_dropout_prob']
        self.attn_dropout_prob = config['attn_dropout_prob']
        self.hidden_act = config['hidden_act']
        self.layer_norm_eps = config['layer_norm_eps']
        self.initializer_range = config['initializer_range']

        self.alpha = config['alpha']
        self.gama = config['gama']
        self.chux = config['chux']
        self.fft_loss_type = config['fft_loss_type']
        self.fourier_loss_weight = config['alpha_loss']
        self.use_fourier_loss = config['fourier_loss']

        self.loss_type = config['loss_type']

        # define layers and loss
        self.item_embedding = nn.Embedding(self.n_items, self.hidden_size, padding_idx=0)
        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)

        self.item_encoder = FilterEncoder(config)

        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

        if self.loss_type == 'BPR':
            self.loss_fct = BPRLoss()
        elif self.loss_type == 'CE':
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")

        # parameters initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def get_attention_mask(self, item_seq):
        attention_mask = (item_seq > 0).long()
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        max_len = attention_mask.size(-1)
        attn_shape = (1, max_len, max_len)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1)
        subsequent_mask = (subsequent_mask == 0).unsqueeze(1)
        subsequent_mask = subsequent_mask.long().to(item_seq.device)
        extended_attention_mask = extended_attention_mask * subsequent_mask
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def forward(self, item_seq, item_seq_len):
        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)

        item_emb = self.item_embedding(item_seq)
        input_emb = item_emb + position_embedding

        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        extended_attention_mask = self.get_attention_mask(item_seq)

        item_encoded_layers = self.item_encoder(
            input_emb,
            extended_attention_mask,
            output_all_encoded_layers=True
        )

        sequence_output = item_encoded_layers[-1]

        return sequence_output, input_emb

    def get_loss_fn(self):
        if self.fft_loss_type == 'l1':
            return F.l1_loss
        elif self.fft_loss_type == 'l2':
            return F.mse_loss
        elif self.fft_loss_type == 'SmoothL1Loss':
            return F.smooth_l1_loss
        elif self.fft_loss_type == 'mix_loss':
            return self.mix_loss
        else:
            raise ValueError(f"invalid loss type {self.fft_loss_type}")

    def mix_loss(self, pred, true, reduction=None):
        l1 = F.l1_loss(pred, true)
        l2 = F.mse_loss(pred, true)
        return 0.5 * l1 + 0.5 * l2

    def fft_loss(self, model_out, target):
        fft1 = torch.fft.fft(model_out.transpose(1, 2), norm='forward')
        fft2 = torch.fft.fft(target.transpose(1, 2), norm='forward')
        fft1, fft2 = fft1.transpose(1, 2), fft2.transpose(1, 2)
        loss_fn = self.get_loss_fn()
        fourier_loss = (loss_fn(torch.real(fft1), torch.real(fft2))
                        + loss_fn(torch.imag(fft1), torch.imag(fft2)))
        return fourier_loss

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]

        seq_output, target_emb = self.forward(item_seq, item_seq_len)

        fourier_loss = self.fft_loss(seq_output, target_emb)
        fourier_loss = torch.mean(fourier_loss)

        seq_output = self.gather_indexes(seq_output, item_seq_len - 1)
        pos_items = interaction[self.POS_ITEM_ID]

        if self.loss_type == 'BPR':
            neg_items = interaction[self.NEG_ITEM_ID]
            pos_items_emb = self.item_embedding(pos_items)
            neg_items_emb = self.item_embedding(neg_items)
            pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)
            neg_score = torch.sum(seq_output * neg_items_emb, dim=-1)
            loss = self.loss_fct(pos_score, neg_score)
        else:  # CE
            test_item_emb = self.item_embedding.weight
            logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
            loss = self.loss_fct(logits, pos_items)

        if self.use_fourier_loss:
            loss_all = self.fourier_loss_weight * loss + (1 - self.fourier_loss_weight) * fourier_loss
        else:
            loss_all = loss

        return loss_all

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
        seq_output, _ = self.forward(item_seq, item_seq_len)
        seq_output = self.gather_indexes(seq_output, item_seq_len - 1)
        test_item_emb = self.item_embedding(test_item)
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)
        return scores

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output, _ = self.forward(item_seq, item_seq_len)
        seq_output = self.gather_indexes(seq_output, item_seq_len - 1)
        test_items_emb = self.item_embedding.weight
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))
        return scores


class FilterEncoder(nn.Module):
    def __init__(self, config):
        super(FilterEncoder, self).__init__()
        block = FilterBlock(config)
        self.blocks = nn.ModuleList([copy.deepcopy(block) for _ in range(config['num_hidden_layers'])])

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=False):
        all_encoder_layers = [hidden_states]
        for layer_module in self.blocks:
            hidden_states = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers


class FilterBlock(nn.Module):
    def __init__(self, config):
        super(FilterBlock, self).__init__()
        self.layer = FilterLayer(config)
        self.feed_forward = FeedForward(
            config['hidden_size'],
            config['inner_size'],
            config['hidden_dropout_prob'],
            config['hidden_act'],
            config['layer_norm_eps']
        )

    def forward(self, hidden_states, attention_mask):
        layer_output = self.layer(hidden_states, attention_mask)
        feedforward_output = self.feed_forward(layer_output)
        return feedforward_output


class FilterLayer(nn.Module):
    def __init__(self, config):
        super(FilterLayer, self).__init__()
        self.filter_layer = Filter_Model(config)

        from recbole.model.layers import MultiHeadAttention as RecBoleMHA
        self.attention_layer = RecBoleMHA(
            config['num_attention_heads'],
            config['hidden_size'],
            config['hidden_dropout_prob'],
            config['attn_dropout_prob'],
            config['layer_norm_eps']
        )
        self.alpha = config['alpha']

    def forward(self, input_tensor, attention_mask):
        filter_out = self.filter_layer(input_tensor)
        att_out = self.attention_layer(input_tensor, attention_mask)
        hidden_states = self.alpha * filter_out + (1 - self.alpha) * att_out
        return hidden_states


class Filter_Model(nn.Module):
    def __init__(self, config):
        super(Filter_Model, self).__init__()
        self.hidden_size = config['hidden_size']
        self.sparsity_threshold = 0.02
        self.scale = 0.02
        self.r1 = nn.Parameter(self.scale * torch.randn(self.hidden_size, self.hidden_size))
        self.i1 = nn.Parameter(self.scale * torch.randn(self.hidden_size, self.hidden_size))
        self.rb1 = nn.Parameter(self.scale * torch.randn(self.hidden_size))
        self.ib1 = nn.Parameter(self.scale * torch.randn(self.hidden_size))
        self.r2 = nn.Parameter(self.scale * torch.randn(self.hidden_size, self.hidden_size))
        self.i2 = nn.Parameter(self.scale * torch.randn(self.hidden_size, self.hidden_size))
        self.rb2 = nn.Parameter(self.scale * torch.randn(self.hidden_size))
        self.ib2 = nn.Parameter(self.scale * torch.randn(self.hidden_size))

        self.out_dropout = nn.Dropout(config['hidden_dropout_prob'])
        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=config['layer_norm_eps'])

        self.gama = config['gama']
        self.chux = config['chux']

    def FFN(self, x, y):
        x = self.out_dropout(x)
        x = self.LayerNorm(x + y)
        return x

    def MLP_temporal(self, x, B, S, H):
        x = torch.fft.rfft(x, dim=1, norm='ortho')
        y = self.FreMLP(B, S, H, x, self.r2, self.i2, self.rb2, self.ib2)
        x = torch.fft.irfft(y, n=S, dim=1, norm="ortho")
        return x

    def MLP_channel(self, x, B, S, H):
        x = x.permute(1, 0, 2)
        x = torch.fft.rfft(x, dim=1, norm='ortho')
        y = self.FreMLP(B, S, H, x, self.r1, self.i1, self.rb1, self.ib1)
        x = torch.fft.irfft(y, n=B, dim=1, norm="ortho")
        x = x.permute(1, 0, 2)
        return x

    def FreMLP(self, B, S, H, x, r, i, rb, ib):
        o1_real = torch.nn.functional.relu(
            torch.einsum('bid,dd->bid', x.real, r) - \
            torch.einsum('bid,dd->bid', x.imag, i) + \
            rb
        )
        o1_imag = torch.nn.functional.relu(
            torch.einsum('bid,dd->bid', x.imag, r) + \
            torch.einsum('bid,dd->bid', x.real, i) + \
            ib
        )
        y = torch.stack([o1_real, o1_imag], dim=-1)
        y = torch.nn.functional.softshrink(y, lambd=self.sparsity_threshold)
        y = torch.view_as_complex(y)
        return y

    def forward(self, x):
        B, S, H = x.shape
        bias = x
        x_use = self.MLP_channel(x, B, S, H)

        if self.chux == "p":
            x_squence = self.MLP_temporal(x, B, S, H)
            x = (1 - self.gama) * x_use + self.gama * x_squence
            x = self.FFN(x, bias)
            return x
        elif self.chux == "c":
            x = bias + x_use
            x = self.MLP_temporal(x, B, S, H)
            x = self.FFN(x, bias)
            return x
