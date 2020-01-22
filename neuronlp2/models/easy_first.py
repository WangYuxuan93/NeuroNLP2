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
from neuronlp2.nn.transformer import GraphAttentionConfig, GraphAttentionModel, GraphAttentionModelV2
from neuronlp2.nn.transformer import SelfAttentionConfig, SelfAttentionModel
from neuronlp2.models.parsing import PositionEmbeddingLayer


class EasyFirstV2(nn.Module):
    def __init__(self, word_dim, num_words, char_dim, num_chars, pos_dim, num_pos, hidden_size, num_labels, arc_space, type_space,
                 intermediate_size,
                 device=torch.device('cpu'),
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1, graph_attention_probs_dropout_prob=0,
                 embedd_word=None, embedd_char=None, embedd_pos=None, p_in=0.33, p_out=0.33, 
                 pos=True, use_char=False, activation='elu',
                 dep_prob_depend_on_head=False, use_top2_margin=False, target_recomp_prob=0.25,
                 extra_self_attention_layer=False, num_attention_heads=4,
                 input_encoder='Linear', num_layers=3, p_rnn=(0.33, 0.33),
                 input_self_attention_layer=False, num_input_attention_layers=3,
                 maximize_unencoded_arcs_for_norc=False,
                 encode_all_arc_for_rel=False):
        super(EasyFirstV2, self).__init__()
        self.device = device
        self.dep_prob_depend_on_head = dep_prob_depend_on_head
        self.use_top2_margin = use_top2_margin
        self.maximize_unencoded_arcs_for_norc = maximize_unencoded_arcs_for_norc
        self.encode_all_arc_for_rel = encode_all_arc_for_rel
        self.target_recomp_prob = target_recomp_prob

        self.word_embed = nn.Embedding(num_words, word_dim, _weight=embedd_word, padding_idx=1)
        self.pos_embed = nn.Embedding(num_pos, pos_dim, _weight=embedd_pos, padding_idx=1) if pos else None
        if use_char:
            self.char_embed = nn.Embedding(num_chars, char_dim, _weight=embedd_char, padding_idx=1)
            self.char_cnn = CharCNN(2, char_dim, char_dim, hidden_channels=char_dim * 4, activation=activation)
        else:
            self.char_embed = None
            self.char_cnn = None

        self.dropout_in = nn.Dropout2d(p=p_in)
        self.dropout_out = nn.Dropout2d(p=p_out)
        self.num_labels = num_labels

        dim_enc = word_dim
        if use_char:
            dim_enc += char_dim
        if pos:
            dim_enc += pos_dim

        self.input_encoder_type = input_encoder
        if input_encoder == 'Linear':
            self.input_encoder = nn.Linear(dim_enc, hidden_size)
            self.position_embedding_layer = PositionEmbeddingLayer(dim_enc, dropout_prob=0, 
                                                                max_position_embeddings=256)
            out_dim = hidden_size
        elif input_encoder == 'FastLSTM':
            self.input_encoder = VarFastLSTM(dim_enc, hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True, dropout=p_rnn)
            out_dim = hidden_size * 2
        elif input_encoder == 'Transformer':
            self.config = SelfAttentionConfig(input_size=dim_enc,
                                        hidden_size=hidden_size,
                                        num_hidden_layers=num_layers,
                                        num_attention_heads=num_attention_heads,
                                        intermediate_size=intermediate_size,
                                        hidden_act="gelu",
                                        hidden_dropout_prob=hidden_dropout_prob,
                                        attention_probs_dropout_prob=attention_probs_dropout_prob,
                                        max_position_embeddings=256,
                                        initializer_range=0.02)
            self.input_encoder = SelfAttentionModel(self.config)
            out_dim = hidden_size
        elif input_encoder == 'None':
            self.input_encoder = None
            out_dim = dim_enc
        else:
            self.input_encoder = None
            out_dim = dim_enc

        self.config = GraphAttentionConfig(input_size=out_dim,
                                            hidden_size=hidden_size,
                                            arc_space=arc_space,
                                            num_attention_heads=num_attention_heads,
                                            intermediate_size=intermediate_size,
                                            hidden_act="gelu",
                                            hidden_dropout_prob=hidden_dropout_prob,
                                            attention_probs_dropout_prob=attention_probs_dropout_prob,
                                            graph_attention_probs_dropout_prob=graph_attention_probs_dropout_prob,
                                            max_position_embeddings=256,
                                            initializer_range=0.02,
                                            extra_self_attention_layer=extra_self_attention_layer,
                                            input_self_attention_layer=input_self_attention_layer,
                                            num_input_attention_layers=num_input_attention_layers)

        self.graph_attention = GraphAttentionModelV2(self.config)

        out_dim = hidden_size
        self.arc_h = nn.Linear(out_dim, arc_space)
        self.arc_c = nn.Linear(out_dim, arc_space)
        self.arc_attn = BiAffine_v2(arc_space, bias_x=True, bias_y=False)

        self.rel_h = nn.Linear(out_dim, type_space)
        self.rel_c = nn.Linear(out_dim, type_space)
        self.rel_attn = BiAffine_v2(type_space, n_out=self.num_labels, bias_x=True, bias_y=True)

        self.dep_dense = nn.Linear(out_dim, 1)
        feature_dim = 3 + self.use_top2_margin
        self.recomp_dense = nn.Linear(feature_dim, 1)

        assert activation in ['elu', 'tanh']
        if activation == 'elu':
            self.activation = nn.ELU(inplace=True)
        else:
            self.activation = nn.Tanh()
        self.l2_loss = nn.MSELoss(reduction='mean')
        self.criterion = nn.CrossEntropyLoss(reduction='none')

        if torch.cuda.device_count() <= 1:
            if use_char:
                self.char_cnn.to(device)
            self.dropout_in.to(device)
            self.dropout_out.to(device)
            if self.input_encoder is not None:
                self.input_encoder.to(device)
            self.graph_attention.to(device)
            self.arc_h.to(device)
            self.arc_c.to(device)
            self.arc_attn.to(device)
            self.rel_h.to(device)
            self.rel_c.to(device)
            self.rel_attn.to(device)
            self.dep_dense.to(device)
            self.recomp_dense.to(device)
            self.activation.to(device)
            self.l2_loss.to(device)
            self.criterion.to(device)
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

        nn.init.xavier_uniform_(self.rel_h.weight)
        nn.init.constant_(self.rel_h.bias, 0.)
        nn.init.xavier_uniform_(self.rel_c.weight)
        nn.init.constant_(self.rel_c.bias, 0.)

        nn.init.xavier_uniform_(self.dep_dense.weight)
        nn.init.constant_(self.dep_dense.bias, 0.)
        nn.init.xavier_uniform_(self.recomp_dense.weight)
        nn.init.constant_(self.recomp_dense.bias, 0.)

        if self.input_encoder_type == 'Linear':
            nn.init.xavier_uniform_(self.input_encoder.weight)
            nn.init.constant_(self.input_encoder.bias, 0.)

    def _get_encoder_output(self, input_word, input_char, input_pos, graph_matrix, mask=None, device=torch.device('cpu')):
        
        #np.set_printoptions(threshold=np.inf)
        #print ("graph_matrix:\n", graph_matrix.cpu().numpy())
        
        # [batch, length, word_dim]
        word = self.word_embed(input_word)
        # apply dropout word on input
        word = self.dropout_in(word).to(device)
        enc = word
        
        if self.char_embed is not None:
            # [batch, length, char_length, char_dim]
            char = self.char_cnn(self.char_embed(input_char).to(device))
            char = self.dropout_in(char)
            # concatenate word and char [batch, length, word_dim+char_filter]
            enc = torch.cat([enc, char], dim=2)

        if self.pos_embed is not None:
            # [batch, length, pos_dim]
            pos = self.pos_embed(input_pos)
            # apply dropout on input
            pos = self.dropout_in(pos).to(device)
            enc = torch.cat([enc, pos], dim=2)

        # output from rnn [batch, length, hidden_size]
        if self.input_encoder is not None:
            if self.input_encoder_type == 'Linear':
                enc = self.position_embedding_layer(enc)
                input_encoder_output = self.input_encoder(enc)
            elif self.input_encoder_type == 'FastLSTM':
                input_encoder_output, _ = self.input_encoder(enc, mask)
            elif self.input_encoder_type == 'Transformer':
                all_encoder_layers = self.input_encoder(enc, mask)
                # [batch, length, hidden_size]
                input_encoder_output = all_encoder_layers[-1]
        else:
            input_encoder_output = enc
        all_encoder_layers = self.graph_attention(input_encoder_output, graph_matrix, mask)
        # [batch, length, hidden_size]
        output = all_encoder_layers[-1]

        # apply dropout for output
        # [batch, length, hidden_size] --> [batch, hidden_size, length] --> [batch, length, hidden_size]
        #output = self.dropout_out(output.transpose(1, 2)).transpose(1, 2)

        return input_encoder_output, output

    def _mlp(self, input_tensor):
        # output size [batch, length, arc_space]
        arc_h = self.activation(self.arc_h(input_tensor))
        arc_c = self.activation(self.arc_c(input_tensor))

        # output size [batch, length, type_space]
        rel_h = self.activation(self.rel_h(input_tensor))
        rel_c = self.activation(self.rel_c(input_tensor))

        # apply dropout on arc
        # [batch, length, dim] --> [batch, 2 * length, dim]
        arc = torch.cat([arc_h, arc_c], dim=1)
        rel = torch.cat([rel_h, rel_c], dim=1)
        arc = self.dropout_out(arc.transpose(1, 2)).transpose(1, 2)
        #arc = self.dropout_out(arc)
        arc_h, arc_c = arc.chunk(2, 1)

        # apply dropout on rel
        # [batch, length, dim] --> [batch, 2 * length, dim]
        rel = self.dropout_out(rel.transpose(1, 2)).transpose(1, 2)
        #rel = self.dropout_out(rel)
        rel_h, rel_c = rel.chunk(2, 1)
        rel_h = rel_h.contiguous()
        rel_c = rel_c.contiguous()

        return (arc_h, arc_c), (rel_h, rel_c)

    #def forward(self, input_word, input_char, input_pos, mask=None):
    #    raise RuntimeError('EasyFirst does not implement forward')

    def _get_recomp_logp(self, head_logp, rc_gen_mask, norc_gen_mask, mask, debug=False):
        """
        Input:
            head_logp: (batch, seq_len, seq_len)
            rc_gen_mask: (batch, seq_len)
            norc_gen_mask: (batch, seq_len)
            mask: (batch, seq_len)
        """
        batch_size, seq_len, _ = head_logp.size()
        features = []
        # (batch, seq_len*seq_len)
        flatten_logp = head_logp.view([batch_size, -1])
        # (batch, 2), get top 2 logp
        top2_values, _ = torch.topk(flatten_logp, k=2, dim=-1)
        # (batch)
        max_head_logp = top2_values[:,0]
        # (batch)
        second_head_logp = top2_values[:,1]
        # feature: max head logp
        features.append(max_head_logp.unsqueeze(-1))
        # feature: sentence length
        # (batch)
        sent_lens = mask.sum(-1)
        features.append(sent_lens.float().unsqueeze(-1))
        # feature: number of new arcs after last recompute
        # (batch)
        num_new_arcs = rc_gen_mask.sum(-1) - norc_gen_mask.sum(-1)
        features.append(num_new_arcs.float().unsqueeze(-1))

        if self.use_top2_margin:
            # feature: margin between top 2 head logp
            # (batch)
            top2_margin = max_head_logp - second_head_logp
            features.append(top2_margin.unsqueeze(-1))

        if debug:
            print ("flatten_logp:\n", flatten_logp)
            print ("max_head_logp:\n", max_head_logp)
            print ("sent_lens:\n", sent_lens)
            print ("rc_gen_mask:\n",rc_gen_mask)
            print ("norc_gen_mask:\n",norc_gen_mask)
            print ("num_new_arcs:\n", num_new_arcs)

        # (batch, n_feature)
        features = torch.cat(features, -1)
        # (batch)
        rc_probs = torch.sigmoid(self.recomp_dense(features)).squeeze(-1)
        # (batch)
        rc_logp = torch.log(rc_probs)
        norc_logp = torch.log(1.0 - rc_probs)
        return rc_probs, rc_logp, norc_logp

    def _get_head_logp(self, arc_c, arc_h, encoder_output, mask=None, debug=False):
        """
        Input:
            encoder_output: (batch, seq_len, hidden_size)
            mask: (batch, seq_len), seq mask, where at position 0 is 0
        """
        batch_size = encoder_output.size(0)

        # compute head logp given dep
        # (batch, seq_len, seq_len)
        head_logits = self.arc_attn(arc_c, arc_h)

        if debug:
            print ("arc_h:\n", arc_h)
            print ("arc_c:\n", arc_c)
            print ("head_logits:\n", head_logits)

        # mask invalid position to -inf for log_softmax
        #if mask is not None:
        #    minus_mask = mask.eq(0).unsqueeze(2)
        #    head_logits = head_logits.masked_fill(minus_mask, float('-inf'))
  
        # (batch, seq_len, seq_len), log softmax over all possible arcs
        head_logp_given_dep = F.log_softmax(head_logits, dim=-1)
        
        # compute dep logp
        if self.dep_prob_depend_on_head:
            # (batch, seq_len, seq_len) * (batch, seq_len, hidden_size) 
            # => (batch, seq_len, hidden_size)
            # stop grads, prevent it from messing up head probs
            context_layer = torch.matmul(head_logits.detach(), encoder_output.detach())
        else:
            # (batch, seq_len, hidden_size)
            context_layer = encoder_output.detach()
        # (batch, seq_len)
        dep_logp = F.log_softmax(self.dep_dense(context_layer).squeeze(-1), dim=-1)
        if debug:
            print ("head_logits:\n", head_logits)
            print ("head_logp_given_dep:\n", head_logp_given_dep)
            print ("dep_logits:\n",self.dep_dense(context_layer).squeeze(-1))
            print ("dep_logp:\n", dep_logp)
        # (batch, seq_len, seq_len)
        head_logp = head_logp_given_dep + dep_logp.unsqueeze(2)

        return head_logp


    def forward(self, input_word, input_char, input_pos, heads, rels, rc_gen_mask, 
             norc_gen_mask, mask=None, next_head_mask=None, device=torch.device('cpu'),
             debug=False):
        """
        Input:
            input_word: (batch, seq_len)
            input_char: (batch, seq_len, char_len)
            input_pos: (batch, seq_len)
            heads: (batch, seq_len)
            rels: (batch, seq_len)
            rc_gen_mask: (batch, seq_len), 0-1 mask
            norc_gen_mask: (batch, seq_len), 0-1 mask
            mask: (batch, seq_len)
            next_head_mask: (batch, seq_len), 0-1 mask of the next head prediction
        """
        # ----- preprocessing -----
        batch_size, seq_len = input_word.size()
        device = heads.device

        # (batch, seq_len), seq mask, where at position 0 is 0
        root_mask = torch.arange(seq_len, device=device).gt(0).float().unsqueeze(0) * mask
        # (batch, seq_len, seq_len)
        mask_3D = (root_mask.unsqueeze(-1) * mask.unsqueeze(1))

        # (batch, seq_len), the mask of generated heads
        generated_head_mask = rc_gen_mask
        if next_head_mask is None:
            # (batch, seq_len), mask of heads t.cuda()o be generated
            rc_ref_heads_mask = (1 - generated_head_mask) * root_mask
        else:
            rc_ref_heads_mask = next_head_mask
        # (batch, seq_len, seq_len)
        rc_ref_heads_onehot = torch.zeros(batch_size, seq_len, seq_len, dtype=torch.int32, device=device)
        rc_ref_heads_onehot.scatter_(-1, (rc_ref_heads_mask*heads).unsqueeze(-1).long(), 1)
        rc_ref_heads_onehot = rc_ref_heads_onehot * rc_ref_heads_mask.unsqueeze(2)
        if self.maximize_unencoded_arcs_for_norc:
            norc_ref_heads_mask = (1 - norc_gen_mask) * root_mask
            norc_ref_heads_onehot = torch.zeros(batch_size, seq_len, seq_len, dtype=torch.int32, device=device)
            norc_ref_heads_onehot.scatter_(-1, (norc_ref_heads_mask*heads).unsqueeze(-1).long(), 1)
            norc_ref_heads_onehot = norc_ref_heads_onehot * norc_ref_heads_mask.unsqueeze(2)
        else:
            norc_ref_heads_mask = None
            norc_ref_heads_onehot = rc_ref_heads_onehot
        
        # (batch, seq_len, seq_len)
        rels_3D = torch.zeros((batch_size, seq_len, seq_len), dtype=torch.long, device=device)
        rels_3D.scatter_(-1, heads.unsqueeze(-1), rels.unsqueeze(-1))

        # (batch, seq_len, seq_len)
        heads_3D = torch.zeros((batch_size, seq_len, seq_len), dtype=torch.int32, device=device)
        heads_3D.scatter_(-1, heads.unsqueeze(-1), 1)
        heads_3D = (heads_3D * mask_3D).int()

        #print ('mask:\n',mask)
        #print ("root_mask:\n",root_mask)
        if debug:
            print ("mask_3D:\n",mask_3D)
            print ('rc_ref_heads_mask:\n',rc_ref_heads_mask)
            if norc_ref_heads_mask is not None:
                print ('norc_ref_heads_mask:\n',norc_ref_heads_mask)
            print ('rc_ref_heads_onehot:\n',rc_ref_heads_onehot)
            print ('norc_ref_heads_onehot:\n',norc_ref_heads_onehot)
            print ('rc_gen_mask:\n', rc_gen_mask)
            print ('rc_gen_mask*heads:\n', rc_gen_mask*heads)
        
        # (batch, seq_len, seq_len)
        rc_gen_heads_onehot = torch.zeros(batch_size, seq_len, seq_len, dtype=torch.int32, device=device)
        rc_gen_heads_onehot.scatter_(-1, (rc_gen_mask*heads).unsqueeze(-1), 1)
        # (batch, seq_len, seq_len)
        rc_gen_heads_onehot = rc_gen_heads_onehot * rc_gen_mask.unsqueeze(-1)

        if debug:
            print ('rc_gen_heads_onehot:\n',rc_gen_heads_onehot.cpu().numpy())

        #"""
        # (batch, seq_len, seq_len)
        norc_gen_heads_onehot = torch.zeros(batch_size, seq_len, seq_len, dtype=torch.int32, device=device)
        norc_gen_heads_onehot.scatter_(-1, (norc_gen_mask*heads).unsqueeze(-1), 1)
        # (batch, seq_len, seq_len)
        norc_gen_heads_onehot = norc_gen_heads_onehot * norc_gen_mask.unsqueeze(-1)
        
        if debug:
            np.set_printoptions(threshold=np.inf)
            print ('rc_gen_heads_onehot:\n',rc_gen_heads_onehot.cpu().numpy())
            print ('norc_gen_mask:\n', norc_gen_mask)
            print ('norc_gen_mask*heads:\n', norc_gen_mask*heads)
            print ('norc_gen_heads_onehot:\n',norc_gen_heads_onehot.cpu().numpy())
        #"""

        # ----- encoding -----
        fast_mode = True
        if fast_mode:
            if self.encode_all_arc_for_rel:
                # (3*batch, seq_len)
                input_words = torch.cat([input_word,input_word,input_word], dim=0)
                input_chars = torch.cat([input_char,input_char,input_char], dim=0)
                input_poss = torch.cat([input_pos,input_pos,input_pos], dim=0)
                graph_matrices = torch.cat([rc_gen_heads_onehot,norc_gen_heads_onehot,heads_3D], dim=0)
                masks = torch.cat([root_mask,root_mask,root_mask], dim=0)
            else:
                # (2*batch, seq_len)
                input_words = torch.cat([input_word,input_word], dim=0)
                input_chars = torch.cat([input_char,input_char], dim=0)
                input_poss = torch.cat([input_pos,input_pos], dim=0)
                graph_matrices = torch.cat([rc_gen_heads_onehot, norc_gen_heads_onehot], dim=0)
                masks = torch.cat([root_mask,root_mask], dim=0)
            # (2*batch, seq_len, hidden_size) or (3*batch, seq_len, hidden_size)
            _, encoder_output = self._get_encoder_output(input_words, input_chars, input_poss, graph_matrices, mask=masks, device=device)
            if self.encode_all_arc_for_rel:
                # (batch, seq_len, hidden_size)
                rc_encoder_output, norc_encoder_output, _ = encoder_output.split(batch_size, dim=0)
                arc, rel = self._mlp(encoder_output)
                # (2*batch, seq_len, arc_space)
                arc_h, arc_c = arc
                rc_arc_h, norc_arc_h, _ = arc_h.split(batch_size, dim=0)
                rc_arc_c, norc_arc_c, _ = arc_c.split(batch_size, dim=0)
                # (2*batch, seq_len, rel_space)
                rel_h_, rel_c_ = rel
                rc_rel_h, norc_rel_h, rel_h = rel_h_.split(batch_size, dim=0)
                rc_rel_c, norc_rel_c, rel_c = rel_c_.split(batch_size, dim=0)
            else:
                # (batch, seq_len, hidden_size)
                rc_encoder_output, norc_encoder_output = encoder_output.split(batch_size, dim=0)
                arc, rel = self._mlp(encoder_output)
                # (2*batch, seq_len, arc_space)
                arc_h, arc_c = arc
                rc_arc_h, norc_arc_h = arc_h.split(batch_size, dim=0)
                rc_arc_c, norc_arc_c = arc_c.split(batch_size, dim=0)
                # (2*batch, seq_len, rel_space)
                rel_h, rel_c = rel
                rc_rel_h, norc_rel_h = rel_h.split(batch_size, dim=0)
                rc_rel_c, norc_rel_c = rel_c.split(batch_size, dim=0)
        else:
            # (batch, seq_len, hidden_size)
            input_encoder_output, rc_encoder_output = self._get_encoder_output(input_word, input_char, input_pos, rc_gen_heads_onehot, mask=root_mask, device=device)
            _, norc_encoder_output = self._get_encoder_output(input_word, input_char, input_pos, norc_gen_heads_onehot, mask=root_mask, device=device)
            #print ("rc_encoder_output:\n",rc_encoder_output)
            norc_arc, norc_rel = self._mlp(norc_encoder_output)
            norc_arc_h, norc_arc_c = norc_arc
            rc_arc, rc_rel = self._mlp(rc_encoder_output)
            rc_arc_h, rc_arc_c = rc_arc
            # (batch, length, type_space), out_type shape 
            rc_rel_h, rc_rel_c = rc_rel

        # ----- compute arc loss -----
        #"""
        # compute arc logp for no recompute generate mask
        # (batch, seq_len, seq_len)
        norc_head_logp_given_norc = self._get_head_logp(norc_arc_c, norc_arc_h, norc_encoder_output)
        
        # (batch, seq_len, seq_len)
        neg_inf_logp = torch.Tensor(norc_head_logp_given_norc.size()).fill_(-1e9).to(device)
        # (batch, seq_len, seq_len)
        logp_mask = (1-rc_gen_mask.unsqueeze(2)) * mask_3D
        # (batch, seq_len, seq_len), mask out generated heads
        masked_norc_head_logp = torch.where(logp_mask==1, norc_head_logp_given_norc.detach(), neg_inf_logp)
        # (batch)
        rc_probs, rc_logp, norc_logp = self._get_recomp_logp(masked_norc_head_logp, rc_gen_mask, norc_gen_mask, root_mask)
        # (batch, seq_len, seq_len)
        norc_head_logp = norc_logp.unsqueeze(1).unsqueeze(2) + norc_head_logp_given_norc
        #"""
        # compute arc logp for recompute generate mask
        # (batch, seq_len, seq_len)
        rc_head_logp_given_rc = self._get_head_logp(rc_arc_c, rc_arc_h, rc_encoder_output)
        # (batch, seq_len, seq_len)
        rc_head_logp = rc_logp.unsqueeze(1).unsqueeze(2) + rc_head_logp_given_rc

        # (batch), number of ref heads in total
        rc_num_heads = rc_ref_heads_onehot.sum() + 1e-5
        norc_num_heads = norc_ref_heads_onehot.sum() + 1e-5
        # (batch), reference loss for no recompute
        norc_ref_heads_logp = (norc_head_logp * norc_ref_heads_onehot).sum(dim=(1,2))
        # (batch), reference loss for recompute
        rc_ref_heads_logp = (rc_head_logp * rc_ref_heads_onehot).sum(dim=(1,2))
        
        loss_arc = -0.5*(norc_ref_heads_logp.sum()/norc_num_heads+rc_ref_heads_logp.sum()/rc_num_heads) 
        #loss_arc = - rc_ref_heads_logp.sum() / num_heads
        # regularizer of recompute prob, prevent always predicting recompute
        loss_recomp = self.l2_loss(rc_probs.mean(dim=-1,keepdim=True), torch.Tensor([self.target_recomp_prob]).to(device))
        #print ("rc_probs: ({})\n {}".format(self.target_recomp_prob, rc_probs))
        if debug:
            print ('rc_ref_heads_onehot:\n',rc_ref_heads_onehot)
            print ('rc_head_logp:\n', rc_head_logp)
            print ('rc_head_logp * rc_ref_heads_onehot:\n', rc_head_logp * rc_ref_heads_onehot)
        """
        if debug:
            print ('ref_heads_onehot:\n',ref_heads_onehot)
            print ('norc_head_logp:\n', norc_head_logp)
            print ('rc_head_logp:\n', rc_head_logp)
            print ('rc_head_logp * ref_heads_onehot:\n', rc_head_logp * ref_heads_onehot)
            print ('norc_head_logp * ref_heads_onehot:\n', norc_head_logp * ref_heads_onehot)
        """

        # ----- compute label loss -----
        num_total_heads = heads_3D.sum()
        # compute label loss for no recompute
        """
        # (batch, length, type_space), out_type shape 
        norc_rel_h, norc_rel_c = norc_type
        # (batch, n_rels, seq_len, seq_len)
        norc_rel_logits = self.rel_attn(norc_rel_c, norc_rel_h) #.permute(0, 2, 3, 1)
        # (batch, seq_len, seq_len)
        norc_rel_loss = self.criterion(norc_rel_logits, rels_3D) * (mask_3D * heads_3D)
        norc_rel_loss = norc_rel_loss.sum() / num_total_heads
        """

        # compute label loss for no recompute
        if debug:
            print ("rels:\n",rels)
            print ("rels_3D:\n",rels_3D)
            print ("heads_3D:\n", heads_3D)
            print ("num_total_heads:",num_total_heads)

        if not fast_mode and self.encode_all_arc_for_rel:
            print ("Encoding all arc for relation")
            # encode all arcs for relation prediction
            all_encoder_layers = self.graph_attention(input_encoder_output, heads_3D, root_mask)
            # (batch, length, hidden_size)
            graph_attention_output = all_encoder_layers[-1]
            # output size [batch, length, type_space]
            rel_h = self.activation(self.rel_h(graph_attention_output))
            rel_c = self.activation(self.rel_c(graph_attention_output))
            # apply dropout on arc
            # [batch, length, dim] --> [batch, 2 * length, dim]
            rel = torch.cat([rel_h, rel_c], dim=1)
            # apply dropout on rel
            # [batch, length, dim] --> [batch, 2 * length, dim]
            rel = self.dropout_out(rel.transpose(1, 2)).transpose(1, 2)
            #rel = self.dropout_out(rel)
            rel_h, rel_c = rel.chunk(2, 1)
            rel_h = rel_h.contiguous()
            rel_c = rel_c.contiguous()
        else:
            rel_h = rc_rel_h
            rel_c = rc_rel_c

        # (batch, n_rels, seq_len, seq_len)
        rel_logits = self.rel_attn(rel_c, rel_h) #.permute(0, 2, 3, 1)
        # (batch, seq_len, seq_len)
        loss_rel = self.criterion(rel_logits, rels_3D) * heads_3D

        #print ("rc_rel_loss:\n",rc_rel_loss)

        loss_rel = loss_rel.sum() / num_total_heads

        #loss_rel = 0.5*norc_rel_loss + 0.5*rc_rel_loss


        return loss_arc.unsqueeze(0), loss_rel.unsqueeze(0), loss_recomp.unsqueeze(0)


    def _decode_rels(self, out_type, heads, leading_symbolic):
        # out_type shape [batch, length, type_space]
        rel_h, rel_c = out_type
        # get vector for heads [batch, length, type_space],
        rel_h = rel_h.gather(dim=1, index=heads.unsqueeze(2).expand(rel_h.size()))
        # compute output for type [batch, length, num_labels]
        out_type = self.bilinear(rel_h, rel_c)
        # remove the first #leading_symbolic rels.
        out_type = out_type[:, :, leading_symbolic:]
        # compute the prediction of rels [batch, length]
        _, rels = out_type.max(dim=2)
        return rels + leading_symbolic


    def _decode_one_step(self, head_logp, heads_mask, mask, device=torch.device('cpu'), 
                            debug=False, get_order=False, random_recomp=False, recomp_prob=0.25):
        """
        Input:
            head_logp: (batch, seq_len, seq_len)
            heads_mask: (batch, seq_len)
            mask: (batch, seq_len)
        """
        batch_size, seq_len, _ = head_logp.size()
        order_mask = []
        # (batch)
        num_words = mask.sum(-1)
        # (batch, 1)
        sent_lens = mask.sum(-1).float().unsqueeze(-1)
        # (batch, seq_len, seq_len)
        masked_head_logp = head_logp.detach()
        # (batch, seq_len, seq_len)
        neg_inf_logp = torch.Tensor(head_logp.size()).fill_(-1e9).to(device)
        # (batch, seq_len, seq_len)
        new_heads_onehot = torch.zeros((batch_size, seq_len, seq_len), dtype=torch.int32, device=device)
        null_arc_adder = torch.zeros_like(new_heads_onehot)
        rc_probs_list = []
        if random_recomp:
            prev_rc_probs = torch.zeros(batch_size, dtype=torch.float, device=device)
        for i in range(seq_len):
            # (batch, seq_len*seq_len)
            flatten_logp = masked_head_logp.view([batch_size, -1])
            # (batch, 2), get top 2 logp
            top2_values, top2_indices = torch.topk(flatten_logp, k=2, dim=-1)
            
            if random_recomp:
                # ----- randomly generate recompute -----
                # (batch)
                rand_vec = torch.rand(batch_size).to(device)
                rc_probs = torch.where(rand_vec < recomp_prob, torch.ones_like(rand_vec), 
                                        torch.zeros_like(rand_vec))
                # once a sentence decide to recompute, 
                # make sure it will not generate anymore arc
                rc_probs = torch.where(prev_rc_probs.eq(1), prev_rc_probs, rc_probs)
                prev_rc_probs = rc_probs
            else:
                # ----- compute recompute prob -----
                features = []
                # (batch)
                max_head_logp = top2_values[:,0]
                # (batch)
                second_head_logp = top2_values[:,1]
                # feature: max head logp
                features.append(max_head_logp.unsqueeze(-1))
                # feature: sentence length
                # (batch)
                features.append(sent_lens)
                # feature: number of new arcs after last recompute
                num_new_arcs = new_heads_onehot.sum(dim=(1,2)).float()
                # (batch)
                features.append(num_new_arcs.unsqueeze(-1))

                if self.use_top2_margin:
                    # feature: margin between top 2 head logp
                    # (batch)
                    top2_margin = max_head_logp - second_head_logp
                    features.append(top2_margin.unsqueeze(-1))

                # (batch, n_feature)
                features = torch.cat(features, -1)
                # (batch)
                rc_probs = torch.sigmoid(self.recomp_dense(features)).squeeze(-1)
            rc_probs_list.append(rc_probs)

            # ----- get top arc & update state -----
            # (batch)
            max_dep_indices = top2_indices[:,0] // seq_len
            max_head_indices = top2_indices[:,0] % seq_len

            max_dep_mask = torch.zeros(batch_size, seq_len, seq_len, dtype=torch.int32, device=device)
            # (batch, seq_len, seq_len)
            max_dep_mask.scatter_(1, max_dep_indices.unsqueeze(1).unsqueeze(2).expand_as(max_dep_mask), 1)
            # (batch, seq_len, seq_len)
            max_head_onehot = torch.zeros(batch_size, seq_len, seq_len, dtype=torch.int32, device=device)
            max_head_onehot.scatter_(2, max_head_indices.unsqueeze(1).unsqueeze(2).expand_as(max_head_onehot), 1)
            # (batch, seq_len, seq_len)
            max_arc_onehot = max_dep_mask * max_head_onehot

            # (batch), number of generated heads of each sent
            num_heads = heads_mask.sum(-1)
            # (batch), = 1 if all heads found
            finish_mask = (num_words == num_heads).int()
            if i == 0:
                continue_mask = (1 - finish_mask)
                # if all heads generated
                if continue_mask.sum() == 0:
                    break
                # at least add one new arc
                #new_arc_onehot = max_arc_onehot
                #new_heads_onehot = new_heads_onehot + new_arc_onehot
                norc_mask = torch.ones_like(rc_probs)
                #keep_mask = (norc_mask).unsqueeze(1).expand_as(heads_mask)*(1-heads_mask)*mask
                #keep_mask_3D = keep_mask.unsqueeze(2).expand_as(new_heads_onehot)
            else:
                # (batch), True if do recompute, otherwise False
                norc_mask = torch.le(rc_probs, 0.5).int()
                continue_mask = norc_mask * (1 - finish_mask)
                # if all sent in the batch choose to recompute, return
                if continue_mask.sum() == 0:
                    break
            # (batch, seq_len), if 1, keep the arc
            keep_mask = norc_mask.unsqueeze(1).expand_as(heads_mask)*(1-heads_mask)*mask
            keep_mask_3D = keep_mask.unsqueeze(2).expand_as(new_heads_onehot)
            new_arc_onehot = torch.where(keep_mask_3D==0,
                                        null_arc_adder, max_arc_onehot)
            new_heads_onehot = new_heads_onehot + new_arc_onehot
            
            if debug:
                print ("norc_mask:\n",norc_mask)
                print ("continue_mask:\n",continue_mask)
                print ("keep_mask:\n",keep_mask)
                print ("max_dep_mask*keep_mask:\n",max_dep_mask*keep_mask_3D)
                print ("max_arc_onehot:\n",max_arc_onehot)
                print ("heads_mask:\n",heads_mask)
                print ("new_heads_onehot:\n",new_heads_onehot)
                print ("masked_head_logp:\n",masked_head_logp)

            if get_order:
                # n * (batch, seq_len), 1 for dep of new arc
                order_mask.append(new_arc_onehot.sum(-1))

            # update logp
            #norc_mask_3D = (1-rc_mask.int()).unsqueeze(1).unsqueeze(2).expand_as(max_dep_mask)
            # (batch, seq_len, seq_len), mask out the whole row of newly generated heads
            # ensure it will not be choosed in next step
            # only mask out rows that will be generated in this step
            masked_head_logp = torch.where(max_dep_mask*keep_mask_3D==1, neg_inf_logp, masked_head_logp)
            new_heads_mask = max_dep_mask[:,:,0]
            heads_mask = heads_mask + new_heads_mask*keep_mask
            
        return rc_probs_list, new_heads_onehot, order_mask


    def decode(self, input_word, input_char, input_pos, mask=None, debug=False, device=torch.device('cpu'),
                get_head_by_layer=False, random_recomp=False, recomp_prob=0.25):
        """
        Input:
            input_word: (batch, seq_len)
            input_char: (batch, seq_len, char_len)
            input_pos: (batch, seq_len)
            mask: (batch, seq_len)
        """
        batch_size, seq_len = input_word.size()
        # ----- preprocessing -----
        # (batch), the number of recompute of each sentence
        num_recomp = torch.zeros(batch_size, dtype=torch.int32, device=device)
        # (batch_size, seq_len)
        heads_pred = torch.zeros((batch_size, seq_len), dtype=torch.int64, device=device)
        heads_mask = torch.zeros_like(heads_pred, device=device)
        rels_pred = torch.zeros_like(heads_pred, device=device)
        # (batch, seq_len, seq_len)
        gen_heads_onehot = torch.zeros((batch_size, seq_len, seq_len), dtype=torch.int32, device=device)
        # (batch, seq_len), seq mask, where at position 0 is 0
        root_mask = torch.arange(seq_len, device=device).gt(0).float().unsqueeze(0) * mask
        # (batch, seq_len, seq_len)
        mask_3D = (root_mask.unsqueeze(-1) * mask.unsqueeze(1))
        heads_by_layer = []
        while True:
            # ----- encoding -----
            # (batch, seq_len, hidden_size)
            _, encoder_output = self._get_encoder_output(input_word, input_char, input_pos, gen_heads_onehot, mask=root_mask, device=device)
            # ----- compute arc probs -----
            # compute arc logp for no recompute generate mask
            arc, type = self._mlp(encoder_output)
            arc_h, arc_c = arc
            # (batch, seq_len, seq_len)
            head_logp = self._get_head_logp(arc_c, arc_h, encoder_output)
            
            # (batch, seq_len, seq_len)
            neg_inf_logp = torch.Tensor(head_logp.size()).fill_(-1e9).to(device)
            logp_mask = (1-heads_mask).unsqueeze(-1).expand_as(head_logp) * mask_3D
            # (batch, seq_len, seq_len), mask out generated heads
            masked_head_logp = torch.where(logp_mask==1, head_logp.detach(), neg_inf_logp)
            # rc_probs_list: k* (batch), the probability of recompute after each arc generated
            # new_heads_onehot: (batch, seq_len, seq_len), newly generated arcs
            rc_probs_list, new_heads_onehot, _ = self._decode_one_step(masked_head_logp, heads_mask, root_mask, 
                                                device=device, random_recomp=random_recomp, recomp_prob=recomp_prob)
            
            # prevent generating new arcs for rows that have heads
            # (batch, seq_len)
            allow_mask = (1 - heads_mask) * root_mask
            new_heads_onehot = (new_heads_onehot * allow_mask.unsqueeze(-1)).int()
            # update the generated head tensor
            gen_heads_onehot = gen_heads_onehot + new_heads_onehot
            # should convert this to 2D (heads_pred & heads_mask)
            # (batch, seq_len)
            new_heads_mask = new_heads_onehot.sum(-1)
            # (batch, seq_len)
            _, new_heads_pred = torch.max(new_heads_onehot, -1)
            new_heads_pred = new_heads_pred * new_heads_mask
            heads_mask = heads_mask + new_heads_mask
            heads_pred = heads_pred + new_heads_pred
            num_recomp = num_recomp + (new_heads_mask.sum(-1).ge(1)).int()
            if get_head_by_layer:
                heads_by_layer.append(new_heads_mask.unsqueeze(1))
            #print ("rc_probs_list:\n", rc_probs_list)
            if debug:
                print ("rc_probs_list:\n", rc_probs_list)
                print ("logp_mask:\n",logp_mask)
                print ("rc_probs_list:\n", rc_probs_list)
                print ("new_heads_mask:\n", new_heads_mask)
                print ("new_heads_pred:\n", new_heads_pred)
                print ("heads_pred:\n", heads_pred)
                print ("heads_mask:\n", heads_mask)
                print ("root_mask:\n", root_mask)
            # if every token has a head
            if torch.equal(heads_mask, root_mask.long()):
                break
        if debug:
            print ("heads_pred:\n", heads_pred)
            print ("heads_mask:\n", heads_mask)

        # ----- compute rel probs -----
        #encoder_output = self._get_encoder_output(input_word, input_char, input_pos, gen_heads_onehot, mask=root_mask)
        # compute arc logp for no recompute generate mask
        #arc, type = self._mlp(encoder_output)
        rel_h, rel_c = type
        # (batch_size, seq_len, seq_len, n_rels)
        rel_logits = self.rel_attn(rel_c, rel_h).permute(0, 2, 3, 1)
        # (batch_size, seq_len, seq_len)
        rel_ids = rel_logits.argmax(-1)
        # (batch_size, seq_len)
        masked_heads_pred = heads_pred * root_mask
        # (batch_size, seq_len)
        rels_pred = rel_ids.gather(dim=-1, index=masked_heads_pred.unsqueeze(-1).long()).squeeze(-1)
        
        if debug:
            print ("rel_ids:\n", rel_ids)
            print ("masked_heads_pred:\n", masked_heads_pred)
            print ("rels_pred:\n", rels_pred)

        # (batch)
        num_words = root_mask.sum(-1)
        freq_recomp = (num_recomp- 1) / num_words
        if debug:
            print ("num_recomp:\n", num_recomp)
            print ("num_words:\n", num_words)
            print ("freq_recomp:\n", freq_recomp)

        if get_head_by_layer:
            heads_by_layer_ = torch.cat(heads_by_layer, dim=1)
            heads_by_layer = heads_by_layer_.argmax(1).cpu().numpy()
            #print (heads_by_layer)
        else:
            heads_by_layer = None

        return heads_pred.cpu().numpy(), rels_pred.cpu().numpy(), freq_recomp.mean().cpu().numpy(), heads_by_layer


    def inference(self, input_word, input_char, input_pos, gold_heads, batch, mask=None, 
                  debug=False, device=torch.device('cpu')):
        """
        Input:
            input_word: (batch, seq_len)
            input_char: (batch, seq_len, char_len)
            input_pos: (batch, seq_len)
            gold_heads: (batch, seq_len), the gold heads
            mask: (batch, seq_len)
        """
        batch_size, seq_len = input_word.size()
        device = gold_heads.device

        # for neural network
        # (batch_size, seq_len)
        heads_pred = torch.zeros((batch_size, seq_len), dtype=torch.int64, device=device)
        heads_mask = torch.zeros_like(heads_pred, device=device)

        # collect inference results
        basic_keys = ['WORD', 'MASK', 'LENGTH', 'POS', 'CHAR', 'HEAD', 'TYPE']
        all_keys =  basic_keys + ['RECOMP_GEN_MASK', 'NO_RECOMP_GEN_MASK', 'NEXT_HEAD_MASK']
        sampled_batch = {key: [] for key in all_keys}

        next_list = []
        rc_gen_list = []
        norc_gen_list = []

        # (batch_size, seq_len)
        heads_pred = torch.zeros((batch_size, seq_len), dtype=torch.int64, device=device)
        heads_mask = torch.zeros_like(heads_pred, device=device)
        # (batch, seq_len, seq_len)
        gen_heads_onehot = torch.zeros((batch_size, seq_len, seq_len), dtype=torch.int32, device=device)
        # (batch, seq_len), seq mask, where at position 0 is 0
        root_mask = torch.arange(seq_len, device=device).gt(0).float().unsqueeze(0) * mask
        # (batch, seq_len, seq_len)
        mask_3D = (root_mask.unsqueeze(-1) * mask.unsqueeze(1))

        # (batch, seq_len, seq_len)
        gold_heads_3D = torch.zeros((batch_size, seq_len, seq_len), dtype=torch.int32, device=device)
        gold_heads_3D.scatter_(-1, gold_heads.unsqueeze(-1), 1)

        while True:
            # ----- encoding -----
            # (batch, seq_len, hidden_size)
            _, encoder_output = self._get_encoder_output(input_word, input_char, input_pos, gen_heads_onehot, mask=root_mask, device=device)
            # ----- compute arc probs -----
            # compute arc logp for no recompute generate mask
            arc, type = self._mlp(encoder_output)
            arc_h, arc_c = arc
            # (batch, seq_len, seq_len)
            head_logp = self._get_head_logp(arc_c, arc_h, encoder_output)
            
            # (batch, seq_len, seq_len)
            neg_inf_logp = torch.Tensor(head_logp.size()).fill_(-1e9).to(device)
            # only allow gold arcs
            logp_mask = gold_heads_3D * (1-heads_mask).unsqueeze(-1).expand_as(head_logp) * mask_3D
            # (batch, seq_len, seq_len), mask out generated heads
            masked_head_logp = torch.where(logp_mask==1, head_logp.detach(), neg_inf_logp)
            # rc_probs_list: k* (batch), the probability of recompute after each arc generated
            # new_heads_onehot: (batch, seq_len, seq_len), newly generated arcs
            rc_probs_list, new_heads_onehot, order_mask = self._decode_one_step(masked_head_logp, 
                                        heads_mask, root_mask, device=device, get_order=True)
            tmp_rc_mask = heads_mask
            heads_mask_ = heads_mask.cpu().numpy()
            # (batch, seq_len)
            for i, next_head_mask in enumerate(order_mask):
                # (batch)
                has_head = next_head_mask.sum(-1).cpu().numpy()
                next_head_mask_ = next_head_mask.cpu().numpy()
                tmp_rc_mask_ = tmp_rc_mask.cpu().numpy()
                for j in range(batch_size):
                    if has_head[j] == 1:
                        for key in basic_keys:
                            sampled_batch[key].append(batch[key][j])
                        sampled_batch['RECOMP_GEN_MASK'].append(tmp_rc_mask_[j])
                        sampled_batch['NO_RECOMP_GEN_MASK'].append(heads_mask_[j])
                        sampled_batch['NEXT_HEAD_MASK'].append(next_head_mask_[j])
                        #next_list.append(next_head_mask_[j])
                        #rc_gen_list.append(tmp_rc_mask_[j])
                        #norc_gen_list.append(heads_mask_[j])
                tmp_rc_mask = tmp_rc_mask + next_head_mask

            # prevent generating new arcs for rows that have heads
            # (batch, seq_len)
            #allow_mask = (1 - heads_mask) * root_mask
            #new_heads_onehot = (new_heads_onehot * allow_mask.unsqueeze(-1)).int()
            # update the generated head tensor
            gen_heads_onehot = gen_heads_onehot + new_heads_onehot
            # should convert this to 2D (heads_pred & heads_mask)
            # (batch, seq_len)
            new_heads_mask = new_heads_onehot.sum(-1)
            # (batch, seq_len)
            _, new_heads_pred = torch.max(new_heads_onehot, -1)
            new_heads_pred = new_heads_pred * new_heads_mask
            heads_mask = heads_mask + new_heads_mask
            heads_pred = heads_pred + new_heads_pred
            #print ("rc_probs_list:\n", rc_probs_list)
            if debug:
                print ("rc_probs_list:\n", rc_probs_list)
                print ("logp_mask:\n",logp_mask)
                print ("rc_probs_list:\n", rc_probs_list)
                print ("new_heads_mask:\n", new_heads_mask)
                print ("new_heads_pred:\n", new_heads_pred)
                print ("heads_pred:\n", heads_pred)
                print ("heads_mask:\n", heads_mask)
                print ("root_mask:\n", root_mask)
                print ("order_mask:\n", order_mask)
            # if every token has a head
            if torch.equal(heads_mask, root_mask.long()):
                break

        for key in sampled_batch.keys():
            sampled_batch[key] = np.stack(sampled_batch[key])

        if debug:
            for key in sampled_batch.keys():
                print ("%s\n"%key, sampled_batch[key])


        return sampled_batch


class EasyFirst(nn.Module):
    def __init__(self, word_dim, num_words, char_dim, num_chars, pos_dim, num_pos, hidden_size, num_labels, arc_space, type_space,
                 num_attention_heads, intermediate_size, recomp_att_dim,
                 device=torch.device('cpu'),
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1, graph_attention_probs_dropout_prob=0,
                 embedd_word=None, embedd_char=None, embedd_pos=None, p_in=0.33, p_out=0.33, pos=True, use_char=False, activation='elu'):
        super(EasyFirst, self).__init__()
        self.device = device

        self.word_embed = nn.Embedding(num_words, word_dim, _weight=embedd_word, padding_idx=1)
        self.pos_embed = nn.Embedding(num_pos, pos_dim, _weight=embedd_pos, padding_idx=1) if pos else None
        if use_char:
            self.char_embed = nn.Embedding(num_chars, char_dim, _weight=embedd_char, padding_idx=1)
            self.char_cnn = CharCNN(2, char_dim, char_dim, hidden_channels=char_dim * 4, activation=activation).to(device)
        else:
            self.char_embed = None
            self.char_cnn = None

        self.dropout_in = nn.Dropout2d(p=p_in).to(device)
        self.dropout_out = nn.Dropout2d(p=p_out).to(device)
        self.num_labels = num_labels

        dim_enc = word_dim
        if use_char:
            dim_enc += char_dim
        if pos:
            dim_enc += pos_dim

        self.config = GraphAttentionConfig(input_size=dim_enc,
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

        self.graph_attention = GraphAttentionModel(self.config).to(device)

        out_dim = hidden_size
        self.arc_h = nn.Linear(out_dim, arc_space).to(device)
        self.arc_c = nn.Linear(out_dim, arc_space).to(device)
        self.arc_attn = BiAffine_v2(arc_space, bias_x=True, bias_y=False).to(device)

        self.rel_h = nn.Linear(out_dim, type_space).to(device)
        self.rel_c = nn.Linear(out_dim, type_space).to(device)
        self.rel_attn = BiAffine_v2(type_space, n_out=self.num_labels, bias_x=True, bias_y=True).to(device)
        self.tanh = nn.Tanh().to(device)
        self.recomp = nn.Linear(2*arc_space+hidden_size, 3).to(device)

        assert activation in ['elu', 'tanh']
        if activation == 'elu':
            self.activation = nn.ELU(inplace=True).to(device)
        else:
            self.activation = nn.Tanh().to(device)
        self.criterion = nn.CrossEntropyLoss(reduction='none').to(device)
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

        nn.init.xavier_uniform_(self.rel_h.weight)
        nn.init.constant_(self.rel_h.bias, 0.)
        nn.init.xavier_uniform_(self.rel_c.weight)
        nn.init.constant_(self.rel_c.bias, 0.)

        nn.init.xavier_uniform_(self.arc_hidden_to_att.weight)
        nn.init.constant_(self.arc_hidden_to_att.bias, 0.)
        nn.init.xavier_uniform_(self.encoder_to_att.weight)
        nn.init.constant_(self.encoder_to_att.bias, 0.)
        nn.init.uniform_(self.recomp_att.weight)
        nn.init.xavier_uniform_(self.recomp.weight)
        nn.init.constant_(self.recomp.bias, 0.)

    def _get_encoder_output(self, input_word, input_char, input_pos, graph_matrices, mask=None):
        # [batch, length, word_dim]
        word = self.word_embed(input_word)
        # apply dropout word on input
        word = self.dropout_in(word).to(self.device)
        enc = word

        if self.char_embed is not None:
            # [batch, length, char_length, char_dim]
            char = self.char_cnn(self.char_embed(input_char).to(self.device))
            char = self.dropout_in(char)
            # concatenate word and char [batch, length, word_dim+char_filter]
            enc = torch.cat([enc, char], dim=2)

        if self.pos_embed is not None:
            # [batch, length, pos_dim]
            pos = self.pos_embed(input_pos)
            # apply dropout on input
            pos = self.dropout_in(pos).to(self.device)
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
        rel_h = self.activation(self.rel_h(input_tensor))
        rel_c = self.activation(self.rel_c(input_tensor))

        # apply dropout on arc
        # [batch, length, dim] --> [batch, 2 * length, dim]
        arc = torch.cat([arc_h, arc_c], dim=1)
        type = torch.cat([rel_h, rel_c], dim=1)
        arc = self.dropout_out(arc.transpose(1, 2)).transpose(1, 2)
        arc_h, arc_c = arc.chunk(2, 1)

        # apply dropout on type
        # [batch, length, dim] --> [batch, 2 * length, dim]
        type = self.dropout_out(type.transpose(1, 2)).transpose(1, 2)
        rel_h, rel_c = type.chunk(2, 1)
        rel_h = rel_h.contiguous()
        rel_c = rel_c.contiguous()

        return (arc_h, arc_c), (rel_h, rel_c)


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
        #print ("arc_logp:\n",arc_logp)
        #print ("dep:{}, head:{}".format(max_indices_dep, max_indices_head))
        #print ("dep_hidden_states:\n",dep_hidden_states[:,:,:3])

        ids_dep = max_indices_dep.unsqueeze(1).unsqueeze(2).expand(batch_size,1,hidden_size)
        #print (id_dep)
        # (batch, hidden_size)
        selected_dep_hidden_states = dep_hidden_states.gather(dim=1, index=ids_dep).squeeze(1)
        #print ("selected_dep_hidden_states:\n",selected_dep_hidden_states[:,:3])

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
        enc_att = self.encoder_to_att(encoder_output)
        # (batch, seq_len, att_dim) -> (batch, seq_len, 1) -> (batch, seq_len)
        weight = self.recomp_att(self.tanh(enc_att + arc_hidden_att.unsqueeze(1))).squeeze(-1)
        weight = F.softmax(weight, -1)
        # (batch, 1, seq_len) * (batch, seq_len, hidden_size)
        # -> (batch, 1, hidden_size) -> (batch, hidden_size)
        context_layer = torch.bmm(weight.unsqueeze(1), encoder_output).squeeze(1)
        # (batch, 2*arc_space+hidden_size) -> (batch, 3)
        recomp_logits = self.recomp(torch.cat([context_layer, top_arc_hidden_states], -1))
        # (batch, 3) logp of (no_recomp, do_recomp, eos)
        recomp_logp = F.log_softmax(recomp_logits, dim=-1)

        return recomp_logp

    def _get_arc_logp(self, arc_logits, arc_c, arc_h, encoder_output):
        """
        Input:
            arc_logits: (batch, seq_len, seq_len)
            arc_c/arc_h: (batch, seq_len, arc_space)
            encoder_output: (batch, seq_len, hidden_size)
        """
        batch_size = arc_logits.size(0)
        # (batch, seq_len*seq_len)
        reshaped_logits = arc_logits.view([batch_size, -1])
        # (batch, seq_len, seq_len), log softmax over all possible arcs
        arc_logp = F.log_softmax(reshaped_logits, dim=-1).view(arc_logits.size())

        # (batch, 2*arc_space)
        top_arc_hidden_states = self._get_top_arc_hidden_states(arc_c, arc_h, arc_logp)
        # (batch, 3), recompute logp of (no_recomp, do_recomp, eos)
        recomp_logp = self._get_recomp_prob(encoder_output, top_arc_hidden_states)
        # (batch)
        no_recomp_logp = recomp_logp[:,0]
        # (batch, seq_len, seq_len)
        arc_logp = arc_logp + no_recomp_logp.unsqueeze(1).unsqueeze(2).expand_as(arc_logp)

        return arc_logp, recomp_logp


    def forward(self, input_word, input_char, input_pos, heads, rels, recomps, gen_heads,  
             mask=None, next_head=None, device=torch.device('cpu')):
        """
        Input:
            input_word: (batch, seq_len)
            input_char: (batch, seq_len, char_len)
            input_pos: (batch, seq_len)
            heads: (batch, seq_len)
            rels: (batch, seq_len)
            recomps: (batch)
            gen_heads: (n_layers, batch, seq_len), 0-1 mask
            mask: (batch, seq_len)
            next_head: (batch, seq_len), 0-1 mask of the next head prediction
        """
        # preprocessing
        n_layers, batch_size, seq_len = gen_heads.size()

        # (batch, seq_len), at position 0 is 0
        root_mask = torch.arange(seq_len, device=device).gt(0).float().unsqueeze(0) * mask
        # (batch, seq_len, seq_len)
        mask_3D = (root_mask.unsqueeze(-1) * mask.unsqueeze(1))

        # (batch, seq_len), the mask of generated heads
        generated_head_mask = gen_heads.sum(0)
        if next_head is None:
            # (batch, seq_len), mask of heads to be generated
            ref_heads_mask = (1 - generated_head_mask) * root_mask
        else:
            ref_heads_mask = next_head
        # (batch, seq_len, seq_len)
        ref_heads_onehot = torch.zeros(batch_size, seq_len, seq_len, dtype=torch.int32, device=device)
        ref_heads_onehot.scatter_(-1, (ref_heads_mask*heads).unsqueeze(-1).long(), 1)
        ref_heads_onehot = ref_heads_onehot * ref_heads_mask.unsqueeze(2)
        
        #print ('mask:\n',mask)
        #print ("root_mask:\n",root_mask)
        """
        print ("mask_3D:\n",mask_3D)
        print ('ref_heads_mask:\n',ref_heads_mask)
        print ('ref_heads_onehot:\n',ref_heads_onehot)
        print ('gen_heads:\n', gen_heads)
        print ('gen_heads*heads:\n', gen_heads*heads)
        """
        
        # (n_layers, batch, seq_len, seq_len)
        gen_heads_onehot = torch.zeros(n_layers, batch_size, seq_len, seq_len, dtype=torch.int32, device=device)
        gen_heads_onehot.scatter_(-1, torch.unsqueeze(gen_heads*heads, -1), 1)
        # (1, batch, 1, seq_len)
        #expanded_mask = mask.unsqueeze(0).unsqueeze(2)
        # (n_layers, batch, seq_len, seq_len)
        gen_heads_onehot = gen_heads_onehot * gen_heads.unsqueeze(-1)
        
        #np.set_printoptions(threshold=np.inf)
        #print ('gen_heads_onehot:\n',gen_heads_onehot.cpu().numpy())

        encoder_output = self._get_encoder_output(input_word, input_char, input_pos, gen_heads_onehot, mask=mask)
        arc, type = self._mlp(encoder_output)

        # compute arc loss
        arc_h, arc_c = arc
        # [batch, seq_len, seq_len]
        arc_logits = self.arc_attn(arc_c, arc_h)
        # mask invalid position to -inf for log_softmax
        if mask is not None:
            minus_mask = root_mask.eq(0).unsqueeze(2)
            arc_logits = arc_logits.masked_fill(minus_mask, float('-inf'))

        # (batch, seq_len, seq_len)
        arc_logp, recomp_logp = self._get_arc_logp(arc_logits, arc_c, arc_h, encoder_output)
        # (batch)
        no_recomp_logp = recomp_logp[:,0]
        do_recomp_logp = recomp_logp[:,1]
        eos_logp = recomp_logp[:,2]

        # (batch, seq_len, seq_len)
        neg_inf_like_logp = torch.Tensor(arc_logp.size()).fill_(-1e9).to(device)
        selected_gold_heads_logp = torch.where(ref_heads_onehot==1, arc_logp, neg_inf_like_logp)
        # (batch) number of ref heads in total
        n_heads = ref_heads_onehot.sum(dim=(1,2)) + 1e-5
        # (batch)
        logp_selected_gold_heads = torch.logsumexp(selected_gold_heads_logp, dim=(1, 2)) #/ n_heads

        # (batch), fill in no_recomp and do_recomp
        overall_logp = torch.where(recomps==0, logp_selected_gold_heads, do_recomp_logp)
        # add eos
        overall_logp = torch.where(recomps==2, eos_logp, overall_logp)

        loss_arc = -overall_logp.sum()

        # compute label loss
        # out_type shape [batch, length, type_space]
        rel_h, rel_c = type
        # [batch, n_rels, seq_len, seq_len]
        rel_logits = self.rel_attn(rel_c, rel_h) #.permute(0, 2, 3, 1)
        
        # (batch, seq_len, seq_len)
        rels_3D = torch.zeros((batch_size, seq_len, seq_len), dtype=torch.long, device=device)
        rels_3D.scatter_(-1, heads.unsqueeze(-1), rels.unsqueeze(-1))

        # (batch, seq_len, seq_len)
        heads_3D = torch.zeros((batch_size, seq_len, seq_len), dtype=torch.int32, device=device)
        heads_3D.scatter_(-1, heads.unsqueeze(-1), 1)
        # (batch, seq_len, seq_len)
        loss_type = self.criterion(rel_logits, rels_3D) * (mask_3D * heads_3D)
        loss_type = loss_type.sum()

        return loss_arc, loss_type

    def _decode_rels(self, out_type, heads, leading_symbolic):
        # out_type shape [batch, length, type_space]
        rel_h, rel_c = out_type
        # get vector for heads [batch, length, type_space],
        rel_h = rel_h.gather(dim=1, index=heads.unsqueeze(2).expand(rel_h.size()))
        # compute output for type [batch, length, num_labels]
        out_type = self.bilinear(rel_h, rel_c)
        # remove the first #leading_symbolic rels.
        out_type = out_type[:, :, leading_symbolic:]
        # compute the prediction of rels [batch, length]
        _, rels = out_type.max(dim=2)
        return rels + leading_symbolic

    def decode(self, input_word, input_char, input_pos, mask=None, max_layers=6, max_steps=100,
                debug=False, device=torch.device('cpu')):
        """
        Input:
            input_word: (batch, seq_len)
            input_char: (batch, seq_len, char_len)
            input_pos: (batch, seq_len)
            mask: (batch, seq_len)
        """
        batch_size, seq_len = input_word.size()

        # (batch_size, seq_len)
        heads_pred = torch.zeros((batch_size, seq_len), dtype=torch.int64, device=device)
        heads_mask = torch.zeros_like(heads_pred, device=device)
        rels_pred = torch.zeros_like(heads_pred, device=device)

        for batch_id in range(batch_size):
            word = input_word[batch_id:batch_id+1, :]
            char = input_char[batch_id:batch_id+1, :, :]
            pos = input_pos[batch_id:batch_id+1, :]
            mask_ = mask[batch_id:batch_id+1, :]
            # (batch, seq_len), at position 0 is 0
            root_mask = torch.arange(seq_len, device=device).gt(0).float().unsqueeze(0) * mask_
            # (n_layers=1, batch=1, seq_len, seq_len)
            gen_heads_onehot = torch.zeros((1, 1, seq_len, seq_len), dtype=torch.int32, device=device)
            #recomp_minus_mask = torch.Tensor([0,0,0]).bool()
            recomp_minus_mask = torch.Tensor([0,0,1]).bool().to(device)

            for n_step in range(max_steps):
                if debug:
                    print ("step:{}\n".format(n_step))
                encoder_output = self._get_encoder_output(word, char, pos, gen_heads_onehot, mask=mask_)
                arc, type = self._mlp(encoder_output)

                # compute arc loss
                arc_h, arc_c = arc
                # [batch, seq_len, seq_len]
                arc_logits = self.arc_attn(arc_c, arc_h)

                # mask words that have heads, this only for tree parsing
                #generated_mask = heads_pred[batch_id:batch_id+1,:].ne(0)
                generated_mask = heads_mask[batch_id:batch_id+1,:]
                # (1, seq_len, 1)
                #logit_mask = (mask_.eq(0) + generated_mask).unsqueeze(2)
                logit_mask = (root_mask * (1-generated_mask)).unsqueeze(2)
                # (1, seq_len, seq_len)
                minus_mask = (logit_mask * mask_.unsqueeze(1)).eq(0)
                # mask invalid position to -inf for log_softmax
                #minus_mask = logit_mask.unsqueeze(2)
                arc_logits = arc_logits.masked_fill(minus_mask, float('-inf'))
                
                # (batch, seq_len, seq_len), (batch, 3)
                arc_logp, recomp_logp = self._get_arc_logp(arc_logits, arc_c, arc_h, encoder_output)
                
                recomp_logp = recomp_logp.masked_fill(recomp_minus_mask, float('-inf'))
                # (seq_len*seq_len+2), the last two are do_recomp and eos
                overall_logp = torch.cat([arc_logp.view(-1),recomp_logp.view(-1)[1:]])
                eos_id = overall_logp.size(0) - 1
                do_recomp_id = eos_id - 1

                if debug:
                    print ("heads_pred:\n", heads_pred)
                    print ("heads_mask:\n", heads_mask)
                    print ("root_mask:\n", root_mask)
                    print ("minus_mask:",minus_mask)
                    print ("arc_logp:\n", arc_logp)
                    print ("recomp_logp:\n", recomp_logp)

                prediction = torch.argmax(overall_logp)
                recomp_pred = prediction.cpu().numpy()
                if debug:
                    print ("prediction:",recomp_pred)
                    print ("recomp_mask:",recomp_minus_mask)
                if recomp_pred == eos_id:
                    # predict label here
                    # out_type shape [batch, length, type_space]
                    rel_h, rel_c = type
                    # [batch_size=1, seq_len, seq_len, n_rels]
                    rel_logits = self.rel_attn(rel_c, rel_h).permute(0, 2, 3, 1)
                    # (batch_size=1, seq_len, seq_len)
                    rel_ids = rel_logits.argmax(-1)
                    # (1, seq_len)
                    masked_heads_pred = heads_pred[batch_id:batch_id+1,:] * root_mask
                    # (1, seq_len)
                    gathered_rels_pred = rel_ids.gather(dim=-1, index=masked_heads_pred.unsqueeze(-1).long()).squeeze(-1)
                    # (1, seq_len)
                    rels_pred[batch_id:batch_id+1,:] = gathered_rels_pred
                    break
                elif recomp_pred == do_recomp_id:
                    # add a new layer to gen_heads
                    # (1, batch=1, seq_len, seq_len)
                    gen_heads_new_layer = torch.zeros((1, 1, seq_len, seq_len), dtype=torch.int32, device=device)
                    # (n_layers+1, batch=1, seq_len, seq_len)
                    gen_heads_onehot = torch.cat([gen_heads_onehot, gen_heads_new_layer], dim=0)
                    # update recomp_minus_mask, disable do_recomp action
                    n_layers = gen_heads_onehot.size(0)
                    if n_layers == max_layers:
                        #recomp_minus_mask = torch.Tensor([0,1,0]).bool()
                        recomp_minus_mask = torch.Tensor([0,1,1]).bool().to(device)
                else:
                    # calculate the predicted arc
                    dep = prediction // seq_len
                    head = prediction % seq_len
                    heads_pred[batch_id,dep] = head
                    heads_mask[batch_id,dep] = 1
                    # update it to gen_heads by layer for encoder input
                    gen_heads_onehot[-1,0,dep,head] = 1
                    #if torch.equal(heads_mask[batch_id], mask[batch_id].long()):
                    if torch.equal(heads_mask[batch_id], root_mask[0].long()):
                        # predict label here
                        # out_type shape [batch, length, type_space]
                        rel_h, rel_c = type
                        # [batch_size=1, seq_len, seq_len, n_rels]
                        rel_logits = self.rel_attn(rel_c, rel_h).permute(0, 2, 3, 1)
                        # (batch_size=1, seq_len, seq_len)
                        rel_ids = rel_logits.argmax(-1)
                        # (1, seq_len)
                        masked_heads_pred = heads_pred[batch_id:batch_id+1,:] * root_mask
                        # (1, seq_len)
                        gathered_rels_pred = rel_ids.gather(dim=-1, index=masked_heads_pred.unsqueeze(-1).long()).squeeze(-1)
                        # (1, seq_len)
                        rels_pred[batch_id:batch_id+1,:] = gathered_rels_pred
                        break

        return heads_pred.cpu().numpy(), rels_pred.cpu().numpy()


    def _get_best_gold_head(self, gold_heads, arc_logp, device=torch.device('cpu'), debug=False):

        batch_size, seq_len = gold_heads.size()
        # (batch, seq_len, seq_len)
        heads_3D = torch.zeros((batch_size, seq_len, seq_len), dtype=torch.int32, device=device)
        heads_3D.scatter_(-1, gold_heads.unsqueeze(-1), 1)
        minus_mask = heads_3D.eq(0)
        arc_logp = arc_logp.masked_fill(minus_mask, float('-inf'))
        best_head_index = torch.argmax(arc_logp)
        dep = best_head_index // seq_len
        head = best_head_index % seq_len

        if debug:
            print ("new arc_logp:",arc_logp)
            print ("dep:{}, head:{}".format(dep, head))
            print ("gold_heads:\n", gold_heads)

        return dep, head

    def inference(self, input_word, input_char, input_pos, gold_heads, batch, mask=None, 
                  max_layers=6, use_whole_seq=True, debug=False, device=torch.device('cpu')):
        """
        Input:
            input_word: (batch, seq_len)
            input_char: (batch, seq_len, char_len)
            input_pos: (batch, seq_len)
            gold_heads: (batch, seq_len), the gold heads
            mask: (batch, seq_len)
        """
        NO_RECOMP = 0
        DO_RECOMP = 1
        DO_EOS = 2

        batch_size, seq_len = input_word.size()

        # for neural network
        # (batch_size, seq_len)
        heads_pred = torch.zeros((batch_size, seq_len), dtype=torch.int64, device=device)
        heads_mask = torch.zeros_like(heads_pred, device=device)

        # collect inference results
        easyfirst_keys = ['WORD', 'MASK', 'LENGTH', 'POS', 'CHAR', 'HEAD', 'TYPE']
        all_keys =  easyfirst_keys + ['RECOMP', 'GEN_HEAD', 'NEXT_HEAD']
        batch_by_layer = {i: {key: [] for key in all_keys} for i in range(max_layers)}

        for batch_id in range(batch_size):
            generated_heads = np.zeros([1,seq_len], dtype=np.int32)
            zero_mask = np.zeros([seq_len], dtype=np.int32)
            # the input generated head list
            generated_heads_list = []
            # whether to recompute at this step
            recomp_list = []
            # the next head to be generated, in shape of 0-1 mask
            next_list = []

            word = input_word[batch_id:batch_id+1, :]
            char = input_char[batch_id:batch_id+1, :, :]
            pos = input_pos[batch_id:batch_id+1, :]
            mask_ = mask[batch_id:batch_id+1, :]
            # (batch, seq_len), at position 0 is 0
            root_mask = torch.arange(seq_len, device=device).gt(0).float().unsqueeze(0) * mask_
            # (n_layers=1, batch=1, seq_len, seq_len)
            gen_heads_onehot = torch.zeros((1, 1, seq_len, seq_len), dtype=torch.int32, device=device)
            #recomp_minus_mask = torch.Tensor([0,0,0]).bool()
            recomp_minus_mask = torch.Tensor([0,0,1]).bool().to(device)

            max_steps = batch['LENGTH'][batch_id] + max_layers

            for n_step in range(max_steps):
                if debug:
                    print ("step:{}\n".format(n_step))
                encoder_output = self._get_encoder_output(word, char, pos, gen_heads_onehot, mask=mask_)
                arc, type = self._mlp(encoder_output)

                # compute arc loss
                arc_h, arc_c = arc
                # [batch, seq_len, seq_len]
                arc_logits = self.arc_attn(arc_c, arc_h)

                # mask words that have heads, this only for tree parsing
                #generated_mask = heads_pred[batch_id:batch_id+1,:].ne(0)
                generated_mask = heads_mask[batch_id:batch_id+1,:]
                # (1, seq_len, 1)
                #logit_mask = (mask_.eq(0) + generated_mask).unsqueeze(2)
                logit_mask = (root_mask * (1-generated_mask)).unsqueeze(2)
                # (1, seq_len, seq_len)
                minus_mask = (logit_mask * mask_.unsqueeze(1)).eq(0)
                # mask invalid position to -inf for log_softmax
                #minus_mask = logit_mask.unsqueeze(2)
                arc_logits = arc_logits.masked_fill(minus_mask, float('-inf'))
                
                # (batch, seq_len, seq_len), (batch, 3)
                arc_logp, recomp_logp = self._get_arc_logp(arc_logits, arc_c, arc_h, encoder_output)
                
                recomp_logp = recomp_logp.masked_fill(recomp_minus_mask, float('-inf'))
                # (seq_len*seq_len+2), the last two are do_recomp and eos
                overall_logp = torch.cat([arc_logp.view(-1),recomp_logp.view(-1)[1:]])
                eos_id = overall_logp.size(0) - 1
                do_recomp_id = eos_id - 1

                if debug:
                    print ("heads_pred:\n", heads_pred)
                    print ("heads_mask:\n", heads_mask)
                    print ("root_mask:\n", root_mask)
                    print ("minus_mask:",minus_mask)
                    print ("arc_logp:\n", arc_logp)
                    print ("recomp_logp:\n", recomp_logp)

                prediction = torch.argmax(overall_logp)
                best_head_index = torch.argmax(arc_logp)
                recomp_pred = prediction.cpu().numpy()
                
                dep = best_head_index // seq_len
                head = best_head_index % seq_len
                #dep_id = dep.cpu().numpy()
                #head_id = head.cpu().numpy()
                if debug:
                    print ("best_head_index:",best_head_index)
                    print ("dep:{}, head:{}".format(dep, head))
                    #print ("recomp_mask:",recomp_minus_mask)
                    print ("gold_heads:\n", gold_heads)
                # if the predicted arc in gold, chose it as next prediction
                if gold_heads[batch_id][dep] == head and dep != 0:
                    next_list.append(np.copy(zero_mask))
                    next_list[-1][dep] = 1
                    generated_heads_list.append(np.copy(generated_heads))
                    recomp_list.append(NO_RECOMP)
                    # add one new head to the top layer of generated heads
                    generated_heads[-1,dep] = 1

                    # update states for input of next step
                    heads_pred[batch_id,dep] = head
                    heads_mask[batch_id,dep] = 1
                    # update it to gen_heads by layer for encoder input
                    gen_heads_onehot[-1,0,dep,head] = 1
                # if all arcs have been generated
                elif torch.equal(heads_mask[batch_id], root_mask[0].long()):
                    next_list.append(np.copy(zero_mask))
                    generated_heads_list.append(np.copy(generated_heads))
                    recomp_list.append(DO_EOS)
                    break
                # in this case, we predict DO_RECOMP
                else:
                    if len(generated_heads) == max_layers:
                        if use_whole_seq:
                            dep, head = self._get_best_gold_head(gold_heads[batch_id:batch_id+1, :], arc_logp, device=device, debug=debug)
                            next_list.append(np.copy(zero_mask))
                            next_list[-1][dep] = 1
                            generated_heads_list.append(np.copy(generated_heads))
                            recomp_list.append(NO_RECOMP)
                            # add one new head to the top layer of generated heads
                            generated_heads[-1,dep] = 1

                            # update states for input of next step
                            heads_pred[batch_id,dep] = head
                            heads_mask[batch_id,dep] = 1
                            # update it to gen_heads by layer for encoder input
                            gen_heads_onehot[-1,0,dep,head] = 1
                            continue
                        else:
                            break
                    next_list.append(np.copy(zero_mask))
                    generated_heads_list.append(np.copy(generated_heads))
                    recomp_list.append(DO_RECOMP)
                    prev_layers = generated_heads_list[-1]
                    # add a new layer
                    generated_heads = np.concatenate([prev_layers,np.zeros([1,seq_len], dtype=int)], axis=0)
                    # if the layer in next layer exceeds max_layers, break

                    # update states for input of next step
                    # add a new layer to gen_heads
                    # (1, batch=1, seq_len, seq_len)
                    gen_heads_new_layer = torch.zeros((1, 1, seq_len, seq_len), dtype=torch.int32, device=device)
                    # (n_layers+1, batch=1, seq_len, seq_len)
                    gen_heads_onehot = torch.cat([gen_heads_onehot, gen_heads_new_layer], dim=0)
                    # update recomp_minus_mask, disable do_recomp action
                    n_layers = gen_heads_onehot.size(0)
                    if n_layers == max_layers:
                        #recomp_minus_mask = torch.Tensor([0,1,0]).bool()
                        recomp_minus_mask = torch.Tensor([0,1,1]).bool().to(device)
                if debug:
                    print ("generated_heads_list:\n", generated_heads_list)
                    print ("next_list:\n",next_list)

            # category steps by number of layers
            for n_step in range(len(recomp_list)):
                n_layers = len(generated_heads_list[n_step]) - 1
                for key in easyfirst_keys:
                    batch_by_layer[n_layers][key].append(batch[key][batch_id])
                batch_by_layer[n_layers]['RECOMP'].append(recomp_list[n_step])
                batch_by_layer[n_layers]['GEN_HEAD'].append(generated_heads_list[n_step])
                batch_by_layer[n_layers]['NEXT_HEAD'].append(next_list[n_step])
        if debug:
            for i in batch_by_layer.keys():
                print('-' * 50)
                print ("layer-%d"%i)
                for key in batch_by_layer[i].keys():
                    print ("%s\n"%key, batch_by_layer[i][key])
        # convert batches into torch tensor
        for n_layers in batch_by_layer.keys():
            for key in batch_by_layer[n_layers].keys():
                if batch_by_layer[n_layers][key]:
                    batch_by_layer[n_layers][key] = torch.from_numpy(np.stack(batch_by_layer[n_layers][key]))
                else:
                    batch_by_layer[n_layers][key] = None
            if batch_by_layer[n_layers]['GEN_HEAD'] is not None:
                # (batch, n_layers, seq_len) -> (n_layers, batch, seq_len)
                batch_by_layer[n_layers]['GEN_HEAD'] = np.transpose(batch_by_layer[n_layers]['GEN_HEAD'], (1,0,2))

        return batch_by_layer
