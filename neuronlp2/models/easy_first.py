__author__ = 'max'

from overrides import overrides
import numpy as np
from enum import Enum
import torch
import torch.nn as nn
import torch.nn.functional as F
from neuronlp2.nn import TreeCRF, VarGRU, VarRNN, VarLSTM, VarFastLSTM
from neuronlp2.nn import BiAffine, BiAffine_v2, BiLinear, CharCNN
from neuronlp2.tasks import parser
from neuronlp2.nn.transformer import GraphAttentionConfig, GraphAttentionModel

class PriorOrder(Enum):
    DEPTH = 0
    INSIDE_OUT = 1
    LEFT2RIGTH = 2


class EasyFirst(nn.Module):
    def __init__(self, word_dim, num_words, char_dim, num_chars, pos_dim, num_pos, hidden_size, num_labels, arc_space, type_space,
                 num_attention_heads, intermediate_size, recomp_att_dim,
                 attention_probs_dropout_prob=0.1, graph_attention_probs_dropout_prob=0,
                 embedd_word=None, embedd_char=None, embedd_pos=None, p_in=0.33, p_out=0.33, pos=True, use_char=False, activation='elu'):
        super(EasyFirst, self).__init__()

        self.word_embed = nn.Embedding(num_words, word_dim, _weight=embedd_word, padding_idx=1)
        self.pos_embed = nn.Embedding(num_pos, pos_dim, _weight=embedd_pos, padding_idx=1) if pos else None
        self.char_embed = nn.Embedding(num_chars, char_dim, _weight=embedd_char, padding_idx=1) if use_char else None
        self.char_cnn = CharCNN(2, char_dim, char_dim, hidden_channels=char_dim * 4, activation=activation)

        self.dropout_in = nn.Dropout2d(p=p_in)
        self.dropout_out = nn.Dropout2d(p=p_out)
        self.num_labels = num_labels

        dim_enc = word_dim
        if use_char:
            dim_enc += char_dim
        if pos:
            dim_enc += pos_dim

        self.config = GraphAttentionConfig(vocab_size,
                                            input_size=dim_enc,
                                            hidden_size=hidden_size,
                                            arc_space=arc_space,
                                            num_attention_heads=num_attention_heads,
                                            intermediate_size=intermediate_size,
                                            hidden_act="gelu",
                                            hidden_dropout_prob=hidden_dropout_prob,
                                            attention_probs_dropout_prob=attention_probs_dropout_prob,
                                            graph_attention_probs_dropout_prob=graph_attention_probs_dropout_prob,
                                            max_position_embeddings=256,
                                            initializer_range=0.02)

        self.graph_attention = GraphAttentionModel(self.config)

        out_dim = hidden_size
        self.arc_h = nn.Linear(out_dim, arc_space)
        self.arc_c = nn.Linear(out_dim, arc_space)
        self.arc_attn = BiAffine_v2(arc_space, bias_x=True, bias_y=False)

        self.type_h = nn.Linear(out_dim, type_space)
        self.type_c = nn.Linear(out_dim, type_space)
        self.type_attn = BiAffine_v2(type_space, n_out=self.num_labels, bias_x=True, bias_y=True)

        self.arc_hidden_to_att = nn.Linear(2*arc_space, recomp_att_dim)
        self.encoder_to_att = nn.Linear(hidden_size, recomp_att_dim)
        self.recomp_vector = nn.Parameter(torch.Tensor(recomp_att_dim))
        self.tanh = nn.Tanh()
        self.recomp = nn.Linear(2*arc_space+hidden_size, 3)

        assert activation in ['elu', 'tanh']
        if activation == 'elu':
            self.activation = nn.ELU(inplace=True)
        else:
            self.activation = nn.Tanh()
        self.criterion = nn.CrossEntropyLoss(reduction='none')
        self.reset_parameters(embedd_word, embedd_char, embedd_pos)

    def reset_parameters(self, embedd_word, embedd_char, embedd_pos):
        if embedd_word is None:
            nn.init.uniform_(self.word_embed.weight, -0.1, 0.1)
        if embedd_char is None and self.char_embed is not None:
            nn.init.uniform_(self.char_embed.weight, -0.1, 0.1)
        if embedd_pos is None and self.pos_embed is not None:
            nn.init.uniform_(self.pos_embed.weight, -0.1, 0.1)

        with torch.no_grad():
            self.word_embed.weight[self.word_embed.padding_idx].fill_(0)
            if self.char_embed is not None:
                self.char_embed.weight[self.char_embed.padding_idx].fill_(0)
            if self.pos_embed is not None:
                self.pos_embed.weight[self.pos_embed.padding_idx].fill_(0)

        nn.init.xavier_uniform_(self.arc_h.weight)
        nn.init.constant_(self.arc_h.bias, 0.)
        nn.init.xavier_uniform_(self.arc_c.weight)
        nn.init.constant_(self.arc_c.bias, 0.)

        nn.init.xavier_uniform_(self.type_h.weight)
        nn.init.constant_(self.type_h.bias, 0.)
        nn.init.xavier_uniform_(self.type_c.weight)
        nn.init.constant_(self.type_c.bias, 0.)

        nn.init.xavier_uniform_(self.arc_hidden_to_att.weight)
        nn.init.constant_(self.arc_hidden_to_att.bias, 0.)
        nn.init.xavier_uniform_(self.encoder_to_att.weight)
        nn.init.constant_(self.encoder_to_att.bias, 0.)
        nn.init.xavier_uniform_(self.recomp_vector)
        nn.init.xavier_uniform_(self.recomp.weight)
        nn.init.constant_(self.recomp.bias, 0.)

    def _get_encoder_output(self, input_word, input_char, input_pos, graph_matrices, mask=None):
        # [batch, length, word_dim]
        word = self.word_embed(input_word)
        # apply dropout word on input
        word = self.dropout_in(word)
        enc = word

        if self.char_embed is not None:
            # [batch, length, char_length, char_dim]
            char = self.char_cnn(self.char_embed(input_char))
            char = self.dropout_in(char)
            # concatenate word and char [batch, length, word_dim+char_filter]
            enc = torch.cat([enc, char], dim=2)

        if self.pos_embed is not None:
            # [batch, length, pos_dim]
            pos = self.pos_embed(input_pos)
            # apply dropout on input
            pos = self.dropout_in(pos)
            enc = torch.cat([enc, pos], dim=2)

        all_encoder_layers = self.graph_attention(enc, graph_matrices, mask)
        # [batch, length, hidden_size]
        output = all_encoder_layers[-1]

        # apply dropout for output
        # [batch, length, hidden_size] --> [batch, hidden_size, length] --> [batch, length, hidden_size]
        output = self.dropout_out(output.transpose(1, 2)).transpose(1, 2)

        return output

    def _mlp(self, input_tensor):
        # output size [batch, length, arc_space]
        arc_h = self.activation(self.arc_h(input_tensor))
        arc_c = self.activation(self.arc_c(input_tensor))

        # output size [batch, length, type_space]
        type_h = self.activation(self.type_h(input_tensor))
        type_c = self.activation(self.type_c(input_tensor))

        # apply dropout on arc
        # [batch, length, dim] --> [batch, 2 * length, dim]
        arc = torch.cat([arc_h, arc_c], dim=1)
        type = torch.cat([type_h, type_c], dim=1)
        arc = self.dropout_out(arc.transpose(1, 2)).transpose(1, 2)
        arc_h, arc_c = arc.chunk(2, 1)

        # apply dropout on type
        # [batch, length, dim] --> [batch, 2 * length, dim]
        type = self.dropout_out(type.transpose(1, 2)).transpose(1, 2)
        type_h, type_c = type.chunk(2, 1)
        type_h = type_h.contiguous()
        type_c = type_c.contiguous()

        return (arc_h, arc_c), (type_h, type_c)


    def forward(self, input_word, input_char, input_pos, mask=None):
        raise RuntimeError('EasyFirst does not implement forward')

    def _get_top_arc_hidden_states(self, dep_hidden_states, head_hidden_states, arc_logp):
        """
        Gather the dep/head token hidden states of the top arc (based on predicted head logp)
        Input:
          dep_/head_hidden_sates: (batch, seq_len, hidden_size)
          arc_logp: (batch, seq_len, seq_len), the log probability of every arc
        """
        batch_size, seq_len, hidden_size = dep_hidden_states.size()
        # (batch, seq_len*seq_len)
        flatten_logp = arc_logp.view([batch_size, -1])
        # (batch)
        values, flatten_max_indices = torch.max(flatten_logp, -1)
        # (batch)
        max_indices_dep = flatten_max_indices // seq_len
        max_indices_head = flatten_max_indices % seq_len

        ids_dep = max_indices_dep.unsqueeze(1).unsqueeze(2).expand(batch_size,1,hidden_size)
        #print (id_dep)
        # (batch, hidden_size)
        selected_dep_hidden_states = dep_hidden_states.gather(dim=1, index=ids_dep).squeeze(1)
        #print (selected_dep_hidden_states)

        ids_head = max_indices_head.unsqueeze(1).unsqueeze(2).expand(batch_size,1,hidden_size)
        #print (id_head)
        # (batch, hidden_size)
        selected_head_hidden_states = head_hidden_states.gather(dim=1, index=ids_head).squeeze(1)
        #print (selected_head_hidden_states)
        # (batch, 2*hidden_size)
        top_arc_hidden_states = torch.cat([selected_dep_hidden_states, selected_head_hidden_states], -1)
        return top_arc_hidden_states

    def _get_recomp_prob(self, encoder_output, top_arc_hidden_states):
        """
        Use dep/head hidden states of top arc to attend encoder output
        Input:
            encoder_ouput: (batch, seq_len, hidden_size)
            top_arc_hidden_states: (batch, 2*arc_space)
        """
        # (batch, att_dim)
        arc_hidden_att = self.arc_hidden_to_att(top_arc_hidden_states)
        # (batch, seq_len, att_dim)
        enc_att = self.encoder_to_att(encoder_ouput)
        # (batch, seq_len)
        weight = torch.mm(self.tanh(enc_att + arc_hidden_att.unsqueeze(1)),self.recomp_vector.unsqueeze(-1))
        weight = F.softmax(weight, -1)
        # (batch, 1, seq_len) * (batch, seq_len, hidden_size)
        # -> (batch, 1, hidden_size) -> (batch, hidden_size)
        context_layer = torch.bmm(weight.unsqueeze(1), encoder_output).squeeze(1)
        # (batch, 2*arc_space+hidden_size) -> (batch, 3)
        recomp_logits = self.recomp(torch.cat([context_layer, top_arc_hidden_states], -1))
        # (batch, 3) logp of (no_recomp, do_recomp, eos)
        recomp_logp = F.log_softmax(recomp_logits, dim=-1)

        return recomp_logp

    def _get_arc_logp(self, arc_logits, arc_c, arc_h, encoder_ouput):
        """
        Input:
            arc_logits: (batch, seq_len, seq_len)
            arc_c/arc_h: (batch, seq_len, arc_space)
            encoder_output: (batch, seq_len, hidden_size)
        """
        # (batch, seq_len*seq_len)
        reshaped_logits = arc_logits.view([batch_size, -1])
        # (batch, seq_len, seq_len), log softmax over all possible arcs
        arc_logp = F.log_softmax(reshaped_logits, dim=-1).view(arc_logits.size())

        # (batch, 2*arc_space)
        top_arc_hidden_states = self._get_top_arc_hidden_states(arc_c, arc_h, arc_logp)
        # (batch, 3), recompute logp of (no_recomp, do_recomp, eos)
        recomp_logp = self._get_recomp_prob(encoder_output, top_arc_hidden_states)

        # (batch, seq_len, seq_len)
        arc_logp = arc_logp + no_recomp_logp.unsqueeze(1).unsqueeze(2).expand_as(arc_logp)

        return arc_logp, recomp_logp


    def loss(self, input_word, input_char, input_pos, heads, types, recomps, gen_heads, mask=None):
        """
        Input:
            input_word: (batch, seq_len)
            input_char: (batch, seq_len, char_len)
            input_pos: (batch, seq_len)
            heads: (batch, seq_len)
            types: (batch, seq_len)
            recomps: (batch)
            gen_heads: (n_layers, batch, seq_len), 0-1 mask
            mask: (batch, seq_len)
        """
        # preprocessing
        n_layers, batch_size, seq_len = gen_heads.size()
        # (batch, seq_len), the mask of generated heads
        generated_head_mask = gen_heads.sum(0)
        # (batch, seq_len), mask of heads to be generated
        ref_heads_mask = (1 - gen_heads) * mask
        # (batch, seq_len, seq_len)
        ref_heads_onehot = torch.zeros(batch_size, seq_len, seq_len, dtype=torch.int32)
        ref_heads_onehot.scatter_(-1, torch.unsqueeze(ref_heads_mask*heads, -1), 1)

        # (n_layers, batch, seq_len, seq_len)
        gen_heads_onehot = torch.zeros(n_layers, batch_size, seq_len, seq_len, dtype=torch.int32)
        gen_heads_onehot.scatter_(-1, torch.unsqueeze(gen_heads*heads, -1), 1)
        # (1, batch, seq_len, 1)
        expanded_mask = mask.unsqueeze(0).unsqueeze(-1)
        # (n_layers, batch, seq_len, seq_len)
        gen_heads_onehot = gen_heads_onehot * expanded_mask


        encoder_output = self._get_encoder_output(input_word, input_char, input_pos, gen_heads_onehot, mask=mask)

        arc, type = self._mlp(encoder_output)

        # compute arc loss
        arc_h, arc_c = arc
        # [batch, seq_len, seq_len]
        arc_logits = self.arc_attn(arc_c, arc_h)
        # mask invalid position to -inf for log_softmax
        if mask is not None:
            minus_mask = mask.eq(0).unsqueeze(2)
            arc_logits = arc_logits.masked_fill(minus_mask, float('-inf'))

        # (batch, seq_len, seq_len)
        arc_logp, recomp_logp = self._get_arc_logp(arc_logits, arc_c, arc_h, encoder_ouput)
        # (batch)
        no_recomp_logp = recomp_logp[:,0]
        do_recomp_logp = recomp_logp[:,1]
        eos_logp = recomp_logp[:,2]

        # (batch, seq_len, seq_len)
        neg_inf_like_logp = torch.Tensor(arc_logp.size()).fill_(-1e9)
        selected_gold_heads_logp = torch.where(torch.equal(ref_heads_onehot,1), arc_logp, neg_inf_like_logp)
        # (batch) number of ref heads in total
        n_heads = ref_heads_onehot.sum(dim=(1,2)) + 1e-5
        # (batch)
        logp_selected_gold_heads = torch.logsumexp(selected_gold_heads_logp, dim=(1, 2)) / n_heads

        # (batch), fill in no_recomp and do_recomp
        overall_logp = torch.where(recomps==0, logp_selected_gold_heads, do_recomp_logp)
        # add eos
        overall_logp = torch.where(recomps==2, eos_logp, overall_logp)

        loss_arc = -overall_logp.sum()

        # compute label loss
        # out_type shape [batch, length, type_space]
        type_h, type_c = type
        # [batch_size, seq_len, seq_len, n_rels]
        rel_logits = self.rel_attn(rel_c, rel_h).permute(0, 2, 3, 1)
        
        # (batch, seq_len, seq_len)
        types_3D = torch.zeros((batch_size, seq_len, seq_len), dtype=torch.int32)
        types_3D.scatter_(-1, torch.unsqueeze(heads, -1), torch.unsqueeze(types, -1))

        # (batch, seq_len, seq_len)
        heads_3D = torch.zeros((batch_size, seq_len, seq_len), dtype=torch.int32)
        heads_3D.scatter_(-1, torch.unsqueeze(heads, -1), 1)
        # (batch, seq_len, seq_len)
        loss_type = self.criterion(rel_logits, types_3D) * mask.unsqueeze(-1) * heads_3D
        loss_type = loss_type.sum()

        return loss_arc, loss_type

    def _decode_types(self, out_type, heads, leading_symbolic):
        # out_type shape [batch, length, type_space]
        type_h, type_c = out_type
        # get vector for heads [batch, length, type_space],
        type_h = type_h.gather(dim=1, index=heads.unsqueeze(2).expand(type_h.size()))
        # compute output for type [batch, length, num_labels]
        out_type = self.bilinear(type_h, type_c)
        # remove the first #leading_symbolic types.
        out_type = out_type[:, :, leading_symbolic:]
        # compute the prediction of types [batch, length]
        _, types = out_type.max(dim=2)
        return types + leading_symbolic

    def decode(self, input_word, input_char, input_pos, mask=None, max_layers=6, max_steps=100):
        """
        Input:
            input_word: (batch, seq_len)
            input_char: (batch, seq_len, char_len)
            input_pos: (batch, seq_len)
            mask: (batch, seq_len)
        """
        batch_size, seq_len = input_word.size()

        # (batch_size, seq_len)
        heads_pred = torch.ones((batch_size, seq_len), dtype=torch.int32) * -1
        types_pred = torch.zeros_like(heads_pred)

        for batch_id in range(batch_size):
            word = input_word[batch_id:batch_id+1, :]
            char = input_char[batch_id:batch_id+1, :, :]
            pos = input_pos[batch_id:batch_id+1, :]
            mask_ = mask[batch_id:batch_id+1, :]
            # (n_layers=1, batch=1, seq_len, seq_len)
            gen_heads_onehot = torch.zeros((1, 1, seq_len, seq_len), dtype=torch.int32)
            
            
            for n_step in range(max_steps):
                encoder_output = self._get_encoder_output(word, char, pos, gen_heads_onehot, mask=mask_)
                arc, type = self._mlp(encoder_output)

                # compute arc loss
                arc_h, arc_c = arc
                # [batch, seq_len, seq_len]
                arc_logits = self.arc_attn(arc_c, arc_h)

                # mask words that have heads, this only for tree parsing
                generated_mask = heads_pred[batch_id:batch_id+1,:].ne(-1)
                logit_mask = mask_.eq(0) + generated_mask
                # mask invalid position to -inf for log_softmax
                minus_mask = logit_mask.unsqueeze(2)
                arc_logits = arc_logits.masked_fill(minus_mask, float('-inf'))

                # (batch, seq_len, seq_len), (batch, 3)
                arc_logp, recomp_logp = self._get_arc_logp(arc_logits, arc_c, arc_h, encoder_ouput)
                # (seq_len*seq_len+2), the last two are do_recomp and eos
                overall_logp = torch.cat([arc_logp.view(-1),recomp_logp.view(-1)[1:]])
                eos_id = overall_logp.size(0) - 1
                do_recomp_id = eos_id - 1
                prediction = torch.argmax(overall_logp).cpu().numpy()

                if prediction == eos_id:
                    # predict label here
                    # out_type shape [batch, length, type_space]
                    type_h, type_c = type
                    # [batch_size=1, seq_len, seq_len, n_rels]
                    rel_logits = self.rel_attn(rel_c, rel_h).permute(0, 2, 3, 1)
                    # (batch_size=1, seq_len, seq_len)
                    rel_ids = rel_logits.argmax(rel_logits, dim=-1)
                    # (1, seq_len)
                    masked_heads_pred = heads_pred[batch_id:batch_id+1,:] * mask_
                    # (1, seq_len)
                    types_pred[batch_id:batch_id+1,:] = rel_ids.gather(dim=-1, index=masked_heads_pred.unsqueeze(-1))
                    break
                elif prediction == do_recomp_id:
                    # add a new layer to gen_heads
                    # (1, batch=1, seq_len, seq_len)
                    gen_heads_new_layer = torch.zeros((1, 1, seq_len, seq_len) dtype=torch.int32)
                    # (n_layers+1, batch=1, seq_len, seq_len)
                    gen_heads_onehot = torch.cat([gen_heads_onehot, gen_heads_new_layer], dim=0)
                else:
                    # calculate the predicted arc
                    dep = prediction // seq_len
                    head = prediction % seq_len
                    heads_pred[batch_id,dep] = head
                    # update it to gen_heads by layer for encoder input
                    gen_heads_onehot[-1,0,dep,head] = 1

        return heads_pred.cpu().numpy(), types_pred.cpu().numpy()


class EasyFirst(nn.Module):
    def __init__(self, word_dim, num_words, char_dim, num_chars, pos_dim, num_pos, rnn_mode, hidden_size,
                 encoder_layers, decoder_layers, num_labels, arc_space, type_space,
                 embedd_word=None, embedd_char=None, embedd_pos=None, p_in=0.33, p_out=0.33, p_rnn=(0.33, 0.33),
                 pos=True, prior_order='inside_out', grandPar=False, sibling=False, activation='elu'):

        super(StackPtrNet, self).__init__()
        self.word_embed = nn.Embedding(num_words, word_dim, _weight=embedd_word, padding_idx=1)
        self.pos_embed = nn.Embedding(num_pos, pos_dim, _weight=embedd_pos, padding_idx=1) if pos else None
        self.char_embed = nn.Embedding(num_chars, char_dim, _weight=embedd_char, padding_idx=1)
        self.char_cnn = CharCNN(2, char_dim, char_dim, hidden_channels=char_dim * 4, activation=activation)

        self.dropout_in = nn.Dropout2d(p=p_in)
        self.dropout_out = nn.Dropout2d(p=p_out)
        self.num_labels = num_labels

        if prior_order in ['deep_first', 'shallow_first']:
            self.prior_order = PriorOrder.DEPTH
        elif prior_order == 'inside_out':
            self.prior_order = PriorOrder.INSIDE_OUT
        elif prior_order == 'left2right':
            self.prior_order = PriorOrder.LEFT2RIGTH
        else:
            raise ValueError('Unknown prior order: %s' % prior_order)

        self.grandPar = grandPar
        self.sibling = sibling

        if rnn_mode == 'RNN':
            RNN_ENCODER = VarRNN
            RNN_DECODER = VarRNN
        elif rnn_mode == 'LSTM':
            RNN_ENCODER = VarLSTM
            RNN_DECODER = VarLSTM
        elif rnn_mode == 'FastLSTM':
            RNN_ENCODER = VarFastLSTM
            RNN_DECODER = VarFastLSTM
        elif rnn_mode == 'GRU':
            RNN_ENCODER = VarGRU
            RNN_DECODER = VarGRU
        else:
            raise ValueError('Unknown RNN mode: %s' % rnn_mode)

        dim_enc = word_dim + char_dim
        if pos:
            dim_enc += pos_dim

        self.encoder_layers = encoder_layers
        self.encoder = RNN_ENCODER(dim_enc, hidden_size, num_layers=encoder_layers, batch_first=True, bidirectional=True, dropout=p_rnn)

        dim_dec = hidden_size // 2
        self.src_dense = nn.Linear(2 * hidden_size, dim_dec)
        self.decoder_layers = decoder_layers
        self.decoder = RNN_DECODER(dim_dec, hidden_size, num_layers=decoder_layers, batch_first=True, bidirectional=False, dropout=p_rnn)

        self.hx_dense = nn.Linear(2 * hidden_size, hidden_size)

        self.arc_h = nn.Linear(hidden_size, arc_space) # arc dense for decoder
        self.arc_c = nn.Linear(hidden_size * 2, arc_space)  # arc dense for encoder
        self.biaffine = BiAffine(arc_space, arc_space)

        self.type_h = nn.Linear(hidden_size, type_space) # type dense for decoder
        self.type_c = nn.Linear(hidden_size * 2, type_space)  # type dense for encoder
        self.bilinear = BiLinear(type_space, type_space, self.num_labels)

        assert activation in ['elu', 'tanh']
        if activation == 'elu':
            self.activation = nn.ELU(inplace=True)
        else:
            self.activation = nn.Tanh()

        self.criterion = nn.CrossEntropyLoss(reduction='none')
        self.reset_parameters(embedd_word, embedd_char, embedd_pos)

    def reset_parameters(self, embedd_word, embedd_char, embedd_pos):
        if embedd_word is None:
            nn.init.uniform_(self.word_embed.weight, -0.1, 0.1)
        if embedd_char is None:
            nn.init.uniform_(self.char_embed.weight, -0.1, 0.1)
        if embedd_pos is None and self.pos_embed is not None:
            nn.init.uniform_(self.pos_embed.weight, -0.1, 0.1)

        with torch.no_grad():
            self.word_embed.weight[self.word_embed.padding_idx].fill_(0)
            self.char_embed.weight[self.char_embed.padding_idx].fill_(0)
            if self.pos_embed is not None:
                self.pos_embed.weight[self.pos_embed.padding_idx].fill_(0)

        nn.init.xavier_uniform_(self.arc_h.weight)
        nn.init.constant_(self.arc_h.bias, 0.)
        nn.init.xavier_uniform_(self.arc_c.weight)
        nn.init.constant_(self.arc_c.bias, 0.)

        nn.init.xavier_uniform_(self.type_h.weight)
        nn.init.constant_(self.type_h.bias, 0.)
        nn.init.xavier_uniform_(self.type_c.weight)
        nn.init.constant_(self.type_c.bias, 0.)

    def _get_encoder_output(self, input_word, input_char, input_pos, mask=None):
        # [batch, length, word_dim]
        word = self.word_embed(input_word)

        # [batch, length, char_length, char_dim]
        char = self.char_cnn(self.char_embed(input_char))

        # apply dropout word on input
        word = self.dropout_in(word)
        char = self.dropout_in(char)

        # concatenate word and char [batch, length, word_dim+char_filter]
        enc = torch.cat([word, char], dim=2)

        if self.pos_embed is not None:
            # [batch, length, pos_dim]
            pos = self.pos_embed(input_pos)
            # apply dropout on input
            pos = self.dropout_in(pos)
            enc = torch.cat([enc, pos], dim=2)

        # output from rnn [batch, length, hidden_size]
        output, hn = self.encoder(enc, mask)
        # apply dropout
        # [batch, length, hidden_size] --> [batch, hidden_size, length] --> [batch, length, hidden_size]
        output = self.dropout_out(output.transpose(1, 2)).transpose(1, 2)

        return output, hn

    def _get_decoder_output(self, output_enc, heads, heads_stack, siblings, hx, mask=None):
        # get vector for heads [batch, length_decoder, input_dim],
        enc_dim = output_enc.size(2)
        batch, length_dec = heads_stack.size()
        src_encoding = output_enc.gather(dim=1, index=heads_stack.unsqueeze(2).expand(batch, length_dec, enc_dim))

        if self.sibling:
            # [batch, length_decoder, hidden_size * 2]
            mask_sib = siblings.gt(0).float().unsqueeze(2)
            output_enc_sibling = output_enc.gather(dim=1, index=siblings.unsqueeze(2).expand(batch, length_dec, enc_dim)) * mask_sib
            src_encoding = src_encoding + output_enc_sibling

        if self.grandPar:
            # [batch, length_decoder, 1]
            gpars = heads.gather(dim=1, index=heads_stack).unsqueeze(2)
            # mask_gpar = gpars.ge(0).float()
            # [batch, length_decoder, hidden_size * 2]
            output_enc_gpar = output_enc.gather(dim=1, index=gpars.expand(batch, length_dec, enc_dim)) #* mask_gpar
            src_encoding = src_encoding + output_enc_gpar

        # transform to decoder input
        # [batch, length_decoder, dec_dim]
        src_encoding = self.activation(self.src_dense(src_encoding))
        # output from rnn [batch, length, hidden_size]
        output, hn = self.decoder(src_encoding, mask, hx=hx)
        # apply dropout
        # [batch, length, hidden_size] --> [batch, hidden_size, length] --> [batch, length, hidden_size]
        output = self.dropout_out(output.transpose(1, 2)).transpose(1, 2)

        return output, hn

    def forward(self, input_word, input_char, input_pos, mask=None, length=None, hx=None):
        raise RuntimeError('Stack Pointer Network does not implement forward')

    def _transform_decoder_init_state(self, hn):
        if isinstance(hn, tuple):
            hn, cn = hn
            _, batch, hidden_size = cn.size()
            # take the last layers
            # [batch, 2 * hidden_size] --> [1, batch, 2 * hidden_size]
            cn = torch.cat([cn[-2], cn[-1]], dim=1).unsqueeze(0)
            # take hx_dense to [1, batch, hidden_size]
            cn = self.hx_dense(cn)
            # [decoder_layers, batch, hidden_size]
            if self.decoder_layers > 1:
                cn = torch.cat([cn, cn.new_zeros(self.decoder_layers - 1, batch, hidden_size)], dim=0)
            # hn is tanh(cn)
            hn = torch.tanh(cn)
            hn = (hn, cn)
        else:
            # take the last layers
            # [2, batch, hidden_size]
            hn = hn[-2:]
            # hn [2, batch, hidden_size]
            _, batch, hidden_size = hn.size()
            # first convert hn t0 [batch, 2, hidden_size]
            hn = hn.transpose(0, 1).contiguous()
            # then view to [batch, 1, 2 * hidden_size] --> [1, batch, 2 * hidden_size]
            hn = hn.view(batch, 1, 2 * hidden_size).transpose(0, 1)
            # take hx_dense to [1, batch, hidden_size]
            hn = torch.tanh(self.hx_dense(hn))
            # [decoder_layers, batch, hidden_size]
            if self.decoder_layers > 1:
                hn = torch.cat([hn, hn.new_zeros(self.decoder_layers - 1, batch, hidden_size)], dim=0)
        return hn

    def loss(self, input_word, input_char, input_pos, heads, stacked_heads, children, siblings, stacked_types, mask_e=None, mask_d=None):
        # output from encoder [batch, length_encoder, hidden_size]
        output_enc, hn = self._get_encoder_output(input_word, input_char, input_pos, mask=mask_e)

        # output size [batch, length_encoder, arc_space]
        arc_c = self.activation(self.arc_c(output_enc))
        # output size [batch, length_encoder, type_space]
        type_c = self.activation(self.type_c(output_enc))

        # transform hn to [decoder_layers, batch, hidden_size]
        hn = self._transform_decoder_init_state(hn)

        # output from decoder [batch, length_decoder, tag_space]
        output_dec, _ = self._get_decoder_output(output_enc, heads, stacked_heads, siblings, hn, mask=mask_d)

        # output size [batch, length_decoder, arc_space]
        arc_h = self.activation(self.arc_h(output_dec))
        type_h = self.activation(self.type_h(output_dec))

        batch, max_len_d, type_space = type_h.size()
        # apply dropout
        # [batch, length_decoder, dim] + [batch, length_encoder, dim] --> [batch, length_decoder + length_encoder, dim]
        arc = self.dropout_out(torch.cat([arc_h, arc_c], dim=1).transpose(1, 2)).transpose(1, 2)
        arc_h = arc[:, :max_len_d]
        arc_c = arc[:, max_len_d:]

        type = self.dropout_out(torch.cat([type_h, type_c], dim=1).transpose(1, 2)).transpose(1, 2)
        type_h = type[:, :max_len_d].contiguous()
        type_c = type[:, max_len_d:]

        # [batch, length_decoder, length_encoder]
        out_arc = self.biaffine(arc_h, arc_c, mask_query=mask_d, mask_key=mask_e)

        # get vector for heads [batch, length_decoder, type_space],
        type_c = type_c.gather(dim=1, index=children.unsqueeze(2).expand(batch, max_len_d, type_space))
        # compute output for type [batch, length_decoder, num_labels]
        out_type = self.bilinear(type_h, type_c)

        # mask invalid position to -inf for log_softmax
        if mask_e is not None:
            minus_mask_e = mask_e.eq(0).unsqueeze(1)
            minus_mask_d = mask_d.eq(0).unsqueeze(2)
            out_arc = out_arc.masked_fill(minus_mask_d * minus_mask_e, float('-inf'))

        # losarc_logits shape [batch, length_decoder]
        losarc_logits = self.criterion(out_arc.transpose(1, 2), children)
        loss_type = self.criterion(out_type.transpose(1, 2), stacked_types)

        if mask_d is not None:
            losarc_logits = losarc_logits * mask_d
            loss_type = loss_type * mask_d

        return losarc_logits.sum(dim=1), loss_type.sum(dim=1)

    def decode(self, input_word, input_char, input_pos, mask=None, beam=1, leading_symbolic=0):
        # reset noise for decoder
        self.decoder.reset_noise(0)

        # output_enc [batch, length, model_dim]
        # arc_c [batch, length, arc_space]
        # type_c [batch, length, type_space]
        # hn [num_direction, batch, hidden_size]
        output_enc, hn = self._get_encoder_output(input_word, input_char, input_pos, mask=mask)
        enc_dim = output_enc.size(2)
        device = output_enc.device
        # output size [batch, length_encoder, arc_space]
        arc_c = self.activation(self.arc_c(output_enc))
        # output size [batch, length_encoder, type_space]
        type_c = self.activation(self.type_c(output_enc))
        type_space = type_c.size(2)
        # [decoder_layers, batch, hidden_size]
        hn = self._transform_decoder_init_state(hn)
        batch, max_len, _ = output_enc.size()

        heads = torch.zeros(batch, 1, max_len, device=device, dtype=torch.int64)
        types = torch.zeros(batch, 1, max_len, device=device, dtype=torch.int64)

        num_steps = 2 * max_len - 1
        stacked_heads = torch.zeros(batch, 1, num_steps + 1, device=device, dtype=torch.int64)
        siblings = torch.zeros(batch, 1, num_steps + 1, device=device, dtype=torch.int64) if self.sibling else None
        hypothesis_scores = output_enc.new_zeros((batch, 1))

        # [batch, beam, length]
        children = torch.arange(max_len, device=device, dtype=torch.int64).view(1, 1, max_len).expand(batch, beam, max_len)
        constraints = torch.zeros(batch, 1, max_len, device=device, dtype=torch.bool)
        constraints[:, :, 0] = True
        # [batch, 1]
        batch_index = torch.arange(batch, device=device, dtype=torch.int64).view(batch, 1)

        # compute lengths
        if mask is None:
            steps = torch.new_tensor([num_steps] * batch, dtype=torch.int64, device=device)
            mask_sent = torch.ones(batch, 1, max_len, dtype=torch.bool, device=device)
        else:
            steps = (mask.sum(dim=1) * 2 - 1).long()
            mask_sent = mask.unsqueeze(1).bool()

        num_hyp = 1
        mask_hyp = torch.ones(batch, 1, device=device)
        hx = hn
        for t in range(num_steps):
            # [batch, num_hyp]
            curr_heads = stacked_heads[:, :, t]
            curr_gpars = heads.gather(dim=2, index=curr_heads.unsqueeze(2)).squeeze(2)
            curr_sibs = siblings[:, :, t] if self.sibling else None
            # [batch, num_hyp, enc_dim]
            src_encoding = output_enc.gather(dim=1, index=curr_heads.unsqueeze(2).expand(batch, num_hyp, enc_dim))

            if self.sibling:
                mask_sib = curr_sibs.gt(0).float().unsqueeze(2)
                output_enc_sibling = output_enc.gather(dim=1, index=curr_sibs.unsqueeze(2).expand(batch, num_hyp, enc_dim)) * mask_sib
                src_encoding = src_encoding + output_enc_sibling

            if self.grandPar:
                output_enc_gpar = output_enc.gather(dim=1, index=curr_gpars.unsqueeze(2).expand(batch, num_hyp, enc_dim))
                src_encoding = src_encoding + output_enc_gpar

            # transform to decoder input
            # [batch, num_hyp, dec_dim]
            src_encoding = self.activation(self.src_dense(src_encoding))

            # output [batch * num_hyp, dec_dim]
            # hx [decoder_layer, batch * num_hyp, dec_dim]
            output_dec, hx = self.decoder.step(src_encoding.view(batch * num_hyp, -1), hx=hx)
            dec_dim = output_dec.size(1)
            # [batch, num_hyp, dec_dim]
            output_dec = output_dec.view(batch, num_hyp, dec_dim)

            # [batch, num_hyp, arc_space]
            arc_h = self.activation(self.arc_h(output_dec))
            # [batch, num_hyp, type_space]
            type_h = self.activation(self.type_h(output_dec))
            # [batch, num_hyp, length]
            out_arc = self.biaffine(arc_h, arc_c, mask_query=mask_hyp, mask_key=mask)
            # mask invalid position to -inf for log_softmax
            if mask is not None:
                minus_mask_enc = mask.eq(0).unsqueeze(1)
                out_arc.masked_fill_(minus_mask_enc, float('-inf'))

            # [batch]
            mask_last = steps.le(t + 1)
            mask_stop = steps.le(t)
            minus_mask_hyp = mask_hyp.eq(0).unsqueeze(2)
            # [batch, num_hyp, length]
            hyp_scores = F.log_softmax(out_arc, dim=2).masked_fill_(mask_stop.view(batch, 1, 1) + minus_mask_hyp, 0)
            # [batch, num_hyp, length]
            hypothesis_scores = hypothesis_scores.unsqueeze(2) + hyp_scores

            # [batch, num_hyp, length]
            mask_leaf = curr_heads.unsqueeze(2).eq(children[:, :num_hyp]) * mask_sent
            mask_non_leaf = (~mask_leaf) * mask_sent

            # apply constrains to select valid hyps
            # [batch, num_hyp, length]
            mask_leaf = mask_leaf * (mask_last.unsqueeze(1) + curr_heads.ne(0)).unsqueeze(2)
            mask_non_leaf = mask_non_leaf * (~constraints)

            hypothesis_scores.masked_fill_(~(mask_non_leaf + mask_leaf), float('-inf'))
            # [batch, num_hyp * length]
            hypothesis_scores, hyp_index = torch.sort(hypothesis_scores.view(batch, -1), dim=1, descending=True)

            # [batch]
            prev_num_hyp = num_hyp
            num_hyps = (mask_leaf + mask_non_leaf).long().view(batch, -1).sum(dim=1)
            num_hyp = num_hyps.max().clamp(max=beam).item()
            # [batch, hum_hyp]
            hyps = torch.arange(num_hyp, device=device, dtype=torch.int64).view(1, num_hyp)
            mask_hyp = hyps.lt(num_hyps.unsqueeze(1)).float()

            # [batch, num_hyp]
            hypothesis_scores = hypothesis_scores[:, :num_hyp]
            hyp_index = hyp_index[:, :num_hyp]
            base_index = hyp_index / max_len
            child_index = hyp_index % max_len

            # [batch, num_hyp]
            hyp_heads = curr_heads.gather(dim=1, index=base_index)
            hyp_gpars = curr_gpars.gather(dim=1, index=base_index)

            # [batch, num_hyp, length]
            base_index_expand = base_index.unsqueeze(2).expand(batch, num_hyp, max_len)
            constraints = constraints.gather(dim=1, index=base_index_expand)
            constraints.scatter_(2, child_index.unsqueeze(2), True)

            # [batch, num_hyp]
            mask_leaf = hyp_heads.eq(child_index)
            # [batch, num_hyp, length]
            heads = heads.gather(dim=1, index=base_index_expand)
            heads.scatter_(2, child_index.unsqueeze(2), torch.where(mask_leaf, hyp_gpars, hyp_heads).unsqueeze(2))
            types = types.gather(dim=1, index=base_index_expand)
            # [batch, num_hyp]
            org_types = types.gather(dim=2, index=child_index.unsqueeze(2)).squeeze(2)

            # [batch, num_hyp, num_steps]
            base_index_expand = base_index.unsqueeze(2).expand(batch, num_hyp, num_steps + 1)
            stacked_heads = stacked_heads.gather(dim=1, index=base_index_expand)
            stacked_heads[:, :, t + 1] = torch.where(mask_leaf, hyp_gpars, child_index)
            if self.sibling:
                siblings = siblings.gather(dim=1, index=base_index_expand)
                siblings[:, :, t + 1] = torch.where(mask_leaf, child_index, torch.zeros_like(child_index))

            # [batch, num_hyp, type_space]
            base_index_expand = base_index.unsqueeze(2).expand(batch, num_hyp, type_space)
            child_index_expand = child_index.unsqueeze(2).expand(batch, num_hyp, type_space)
            # [batch, num_hyp, num_labels]
            out_type = self.bilinear(type_h.gather(dim=1, index=base_index_expand), type_c.gather(dim=1, index=child_index_expand))
            hyp_type_scores = F.log_softmax(out_type, dim=2)
            # compute the prediction of types [batch, num_hyp]
            hyp_type_scores, hyp_types = hyp_type_scores.max(dim=2)
            hypothesis_scores = hypothesis_scores + hyp_type_scores.masked_fill_(mask_stop.view(batch, 1), 0)
            types.scatter_(2, child_index.unsqueeze(2), torch.where(mask_leaf, org_types, hyp_types).unsqueeze(2))

            # hx [decoder_layer, batch * num_hyp, dec_dim]
            # hack to handle LSTM
            hx_index = (base_index + batch_index * prev_num_hyp).view(batch * num_hyp)
            if isinstance(hx, tuple):
                hx, cx = hx
                hx = hx[:, hx_index]
                cx = cx[:, hx_index]
                hx = (hx, cx)
            else:
                hx = hx[:, hx_index]

        heads = heads[:, 0].cpu().numpy()
        types = types[:, 0].cpu().numpy()
        return heads, types




