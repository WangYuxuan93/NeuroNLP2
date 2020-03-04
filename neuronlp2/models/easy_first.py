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
from neuronlp2.nn.transformer import GraphAttentionV2Config, GraphAttentionModelV2
from neuronlp2.nn.transformer import SelfAttentionConfig, SelfAttentionModel
from neuronlp2.models.parsing import PositionEmbeddingLayer
from neuronlp2.nn.hard_concrete import HardConcreteDist


class EasyFirst(nn.Module):
    def __init__(self, word_dim, num_words, char_dim, num_chars, pos_dim, num_pos, hidden_size, num_labels, arc_space, type_space,
                 intermediate_size,
                 device=torch.device('cpu'),
                 hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1,
                 graph_attention_hidden_dropout_prob=0.1,
                 graph_attention_probs_dropout_prob=0,
                 embedd_word=None, embedd_char=None, embedd_pos=None, p_in=0.33, p_out=0.33, 
                 pos=True, use_char=False, activation='elu',
                 dep_prob_depend_on_head=False, use_top2_margin=False, target_recomp_prob=0.25,
                 extra_self_attention_layer=False, num_attention_heads=4,
                 input_encoder='Linear', num_layers=3, p_rnn=(0.33, 0.33),
                 input_self_attention_layer=False, num_input_attention_layers=3,
                 maximize_unencoded_arcs_for_norc=False,
                 encode_all_arc_for_rel=False, use_input_encode_for_rel=False,
                 always_recompute=False, use_hard_concrete_dist=True, 
                 hard_concrete_temp=0.1, hard_concrete_eps=0.1,
                 apply_recomp_prob_first=False, num_graph_attention_layers=1, share_params=False,
                 residual_from_input=False, transformer_drop_prob=0,
                 num_graph_attention_heads=1, only_value_weight=False,
                 encode_rel_type='gold', rel_dim=100):
        super(EasyFirst, self).__init__()
        self.device = device
        self.dep_prob_depend_on_head = dep_prob_depend_on_head
        self.use_top2_margin = use_top2_margin
        self.maximize_unencoded_arcs_for_norc = maximize_unencoded_arcs_for_norc
        self.encode_all_arc_for_rel = encode_all_arc_for_rel
        self.use_input_encode_for_rel = use_input_encode_for_rel
        self.always_recompute = always_recompute
        self.target_recomp_prob = target_recomp_prob
        self.use_hard_concrete_dist = use_hard_concrete_dist
        self.apply_recomp_prob_first = apply_recomp_prob_first
        self.residual_from_input = residual_from_input
        self.encode_rel_type = encode_rel_type
        if self.encode_rel_type == 'gold' or self.encode_rel_type == 'pred':
            self.do_encode_rel = True
        else:
            self.do_encode_rel = False

        self.word_embed = nn.Embedding(num_words, word_dim, _weight=embedd_word, padding_idx=1)
        self.pos_embed = nn.Embedding(num_pos, pos_dim, _weight=embedd_pos, padding_idx=1) if pos else None
        if use_char:
            self.char_embed = nn.Embedding(num_chars, char_dim, _weight=embedd_char, padding_idx=1)
            self.char_cnn = CharCNN(2, char_dim, char_dim, hidden_channels=char_dim * 4, activation=activation)
        else:
            self.char_embed = None
            self.char_cnn = None
        if self.do_encode_rel:
            self.rel_embed = nn.Embedding(num_labels, rel_dim, padding_idx=0)

        self.dropout_in = nn.Dropout2d(p=p_in)
        self.dropout_out = nn.Dropout2d(p=p_out)
        self.num_labels = num_labels
        if self.residual_from_input:
            self.transformer_dropout = nn.Dropout2d(p=transformer_drop_prob)

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
                                        embedding_dropout_prob=0.1,
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

        self.config = GraphAttentionV2Config(input_size=out_dim,
                                            hidden_size=hidden_size,
                                            arc_space=arc_space,
                                            num_attention_heads=num_graph_attention_heads,
                                            num_graph_attention_layers=num_graph_attention_layers,
                                            share_params=share_params,
                                            only_value_weight=only_value_weight,
                                            intermediate_size=intermediate_size,
                                            hidden_act="gelu",
                                            hidden_dropout_prob=graph_attention_hidden_dropout_prob,
                                            graph_attention_probs_dropout_prob=graph_attention_probs_dropout_prob,
                                            max_position_embeddings=256,
                                            initializer_range=0.02,
                                            extra_self_attention_layer=extra_self_attention_layer,
                                            input_self_attention_layer=input_self_attention_layer,
                                            num_input_attention_layers=num_input_attention_layers,
                                            rel_dim=rel_dim, do_encode_rel=self.do_encode_rel)

        self.graph_attention = GraphAttentionModelV2(self.config)

        graph_attention_dim = hidden_size
        self.arc_h = nn.Linear(graph_attention_dim, arc_space)
        self.arc_c = nn.Linear(graph_attention_dim, arc_space)
        self.arc_attn = BiAffine_v2(arc_space, bias_x=True, bias_y=False)

        if self.use_input_encode_for_rel:
            self.rel_h = nn.Linear(out_dim, type_space)
            self.rel_c = nn.Linear(out_dim, type_space)
        else:
            self.rel_h = nn.Linear(graph_attention_dim, type_space)
            self.rel_c = nn.Linear(graph_attention_dim, type_space)
        self.rel_attn = BiAffine_v2(type_space, n_out=self.num_labels, bias_x=True, bias_y=True)

        self.dep_dense = nn.Linear(graph_attention_dim, 1)
        if not self.always_recompute:
            if self.use_hard_concrete_dist:
                self.recomp_dist = HardConcreteDist(beta=hard_concrete_temp, eps=hard_concrete_eps)
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
            if self.residual_from_input:
                self.transformer_dropout.to(device)
            if self.input_encoder is not None:
                self.input_encoder.to(device)
            if input_encoder == 'Linear':
                self.position_embedding_layer.to(device)
            self.graph_attention.to(device)
            self.arc_h.to(device)
            self.arc_c.to(device)
            self.arc_attn.to(device)
            self.rel_h.to(device)
            self.rel_c.to(device)
            self.rel_attn.to(device)
            self.dep_dense.to(device)
            if not self.always_recompute:
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
        if self.do_encode_rel:
            nn.init.uniform_(self.rel_embed.weight, -0.1, 0.1)

        with torch.no_grad():
            self.word_embed.weight[self.word_embed.padding_idx].fill_(0)
            if self.char_embed is not None:
                self.char_embed.weight[self.char_embed.padding_idx].fill_(0)
            if self.pos_embed is not None:
                self.pos_embed.weight[self.pos_embed.padding_idx].fill_(0)
            if self.do_encode_rel:
                self.rel_embed.weight[self.rel_embed.padding_idx].fill_(0)

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
        if not self.always_recompute:
            nn.init.xavier_uniform_(self.recomp_dense.weight)
            nn.init.constant_(self.recomp_dense.bias, 0.)

        if self.input_encoder_type == 'Linear':
            nn.init.xavier_uniform_(self.input_encoder.weight)
            nn.init.constant_(self.input_encoder.bias, 0.)

    def _input_encoder(self, input_word, input_char, input_pos, mask=None, device=torch.device('cpu')):
        
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
                if self.residual_from_input:
                    positioned_input_embedding = self.input_encoder.get_embedding()
                    dropped_output = self.transformer_dropout(input_encoder_output)
                    input_encoder_output = dropped_output + positioned_input_embedding
        else:
            input_encoder_output = enc
        
        # apply dropout for output
        # [batch, length, hidden_size] --> [batch, hidden_size, length] --> [batch, length, hidden_size]
        #output = self.dropout_out(output.transpose(1, 2)).transpose(1, 2)

        return input_encoder_output


    def _arc_mlp(self, input_tensor):

        # output size [batch, length, arc_space]
        arc_h = self.activation(self.arc_h(input_tensor))
        arc_c = self.activation(self.arc_c(input_tensor))

        # apply dropout on arc
        # [batch, length, dim] --> [batch, 2 * length, dim]
        arc = torch.cat([arc_h, arc_c], dim=1)
        arc = self.dropout_out(arc.transpose(1, 2)).transpose(1, 2)
        arc_h, arc_c = arc.chunk(2, 1)

        return (arc_h, arc_c)


    def _rel_mlp(self, input_tensor):

        # output size [batch, length, rel_space]
        rel_h = self.activation(self.rel_h(input_tensor))
        rel_c = self.activation(self.rel_c(input_tensor))

        # apply dropout on arc
        # [batch, length, dim] --> [batch, 2 * length, dim]
        rel = torch.cat([rel_h, rel_c], dim=1)

        # apply dropout on rel
        # [batch, length, dim] --> [batch, 2 * length, dim]
        rel = self.dropout_out(rel.transpose(1, 2)).transpose(1, 2)
        rel_h, rel_c = rel.chunk(2, 1)
        rel_h = rel_h.contiguous()
        rel_c = rel_c.contiguous()

        return (rel_h, rel_c)


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
        if self.use_hard_concrete_dist:
            # (batch)
            rc_probs, l0_loss = self.recomp_dist._get_prob(self.recomp_dense(features).squeeze(-1))
            #rc_probs = rc_probs
            rc_logp = torch.log(rc_probs + 1e-9)
            norc_logp = torch.log(1.0 - rc_probs + 1e-9)
            loss_recomp = l0_loss.sum() / batch_size
        else:
            # (batch)
            rc_probs = torch.sigmoid(self.recomp_dense(features)).squeeze(-1)
            l0_loss = None
            rc_logp = torch.log(rc_probs)
            norc_logp = torch.log(1.0 - rc_probs)
            loss_recomp = self.l2_loss(rc_probs.mean(dim=-1,keepdim=True), torch.Tensor([self.target_recomp_prob]).to(head_logp.device))
        # (batch)
        
        #print ("rc_logp:\n",rc_logp)
        #print ("norc_logp:\n",norc_logp)
        return rc_probs, rc_logp, norc_logp, loss_recomp

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
             norc_gen_mask, ref_mask, mask=None, next_head_mask=None, device=torch.device('cpu'),
             debug=False, use_1d_mask=False):
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
        # (batch, seq_len, seq_len)
        rels_3D = torch.zeros((batch_size, seq_len, seq_len), dtype=torch.long, device=device)
        rels_3D.scatter_(-1, heads.unsqueeze(-1), rels.unsqueeze(-1))
        # (batch, seq_len, seq_len)
        heads_3D = torch.zeros((batch_size, seq_len, seq_len), dtype=torch.int32, device=device)
        heads_3D.scatter_(-1, heads.unsqueeze(-1), 1)
        heads_3D = (heads_3D * mask_3D).int()

        if use_1d_mask:
            # (batch, seq_len), the mask of generated heads
            #generated_head_mask = rc_gen_mask
            if next_head_mask is None:
                # (batch, seq_len), mask of heads t.cuda()o be generated
                #rc_ref_heads_mask = (1 - generated_head_mask) * root_mask
                rc_ref_heads_mask = ref_mask * root_mask
            else:
                rc_ref_heads_mask = next_head_mask
            # (batch, seq_len, seq_len)
            rc_ref_heads_onehot = torch.zeros(batch_size, seq_len, seq_len, dtype=torch.int32, device=device)
            rc_ref_heads_onehot.scatter_(-1, (rc_ref_heads_mask*heads).unsqueeze(-1).long(), 1)
            rc_ref_heads_onehot = rc_ref_heads_onehot * rc_ref_heads_mask.unsqueeze(2)
            """
            if self.maximize_unencoded_arcs_for_norc:
                norc_ref_heads_mask = (1 - norc_gen_mask) * root_mask
                norc_ref_heads_onehot = torch.zeros(batch_size, seq_len, seq_len, dtype=torch.int32, device=device)
                norc_ref_heads_onehot.scatter_(-1, (norc_ref_heads_mask*heads).unsqueeze(-1).long(), 1)
                norc_ref_heads_onehot = norc_ref_heads_onehot * norc_ref_heads_mask.unsqueeze(2)
            else:
                norc_ref_heads_mask = None
                norc_ref_heads_onehot = rc_ref_heads_onehot
            """
            norc_ref_heads_mask = None
            norc_ref_heads_onehot = rc_ref_heads_onehot

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
        else:
            rc_ref_heads_onehot = ref_mask
            norc_ref_heads_mask = None
            norc_ref_heads_onehot = rc_ref_heads_onehot
            rc_gen_heads_onehot = rc_gen_mask
            norc_gen_heads_onehot = norc_gen_mask

            if debug:
                np.set_printoptions(threshold=np.inf)
                print ("mask_3D:\n",mask_3D)
                print ('rc_ref_heads_onehot:\n',rc_ref_heads_onehot)
                print ('norc_ref_heads_onehot:\n',norc_ref_heads_onehot)
                print ('rc_gen_heads_onehot:\n',rc_gen_heads_onehot.cpu().numpy())
                print ('norc_gen_heads_onehot:\n',norc_gen_heads_onehot.cpu().numpy())

        #"""

        # ----- encoding -----
        # (batch, seq_len, hidden_size)
        input_encoder_output = self._input_encoder(input_word, input_char, input_pos, mask=root_mask, device=device)
        
        fast_mode = True
        if fast_mode:
            if self.encode_all_arc_for_rel:
                # (3*batch, seq_len)
                #input_words = torch.cat([input_word,input_word,input_word], dim=0)
                #input_chars = torch.cat([input_char,input_char,input_char], dim=0)
                #input_poss = torch.cat([input_pos,input_pos,input_pos], dim=0)
                input_encoder_outputs = torch.cat([input_encoder_output,input_encoder_output,input_encoder_output], dim=0)
                graph_matrices = torch.cat([rc_gen_heads_onehot,norc_gen_heads_onehot,heads_3D], dim=0)
                masks = torch.cat([root_mask,root_mask,root_mask], dim=0)
            else:
                # (2*batch, seq_len)
                #input_words = torch.cat([input_word,input_word], dim=0)
                #input_chars = torch.cat([input_char,input_char], dim=0)
                #input_poss = torch.cat([input_pos,input_pos], dim=0)
                input_encoder_outputs = torch.cat([input_encoder_output,input_encoder_output], dim=0)
                graph_matrices = torch.cat([rc_gen_heads_onehot, norc_gen_heads_onehot], dim=0)
                masks = torch.cat([root_mask,root_mask], dim=0)
            # (2*batch, seq_len, hidden_size) or (3*batch, seq_len, hidden_size)
            
            all_encoder_layers = self.graph_attention(input_encoder_outputs, graph_matrices, masks)
            encoder_output = all_encoder_layers[-1]

            if self.encode_all_arc_for_rel:
                # (batch, seq_len, hidden_size)
                rc_encoder_output, norc_encoder_output, _ = encoder_output.split(batch_size, dim=0)
                # (2*batch, seq_len, arc_space)
                arc_h, arc_c = self._arc_mlp(encoder_output)
                rc_arc_h, norc_arc_h, _ = arc_h.split(batch_size, dim=0)
                rc_arc_c, norc_arc_c, _ = arc_c.split(batch_size, dim=0)
                rel_h, rel_c = self._rel_mlp(encoder_output)
                _, _, rc_rel_h = rel_h.split(batch_size, dim=0)
                _, _, rc_rel_c = rel_c.split(batch_size, dim=0)
            else:
                # (batch, seq_len, hidden_size)
                rc_encoder_output, norc_encoder_output = encoder_output.split(batch_size, dim=0)
                # (2*batch, seq_len, arc_space)
                arc_h, arc_c = self._arc_mlp(encoder_output)
                rc_arc_h, norc_arc_h = arc_h.split(batch_size, dim=0)
                rc_arc_c, norc_arc_c = arc_c.split(batch_size, dim=0)
                if not self.use_input_encode_for_rel:
                    rel_h, rel_c = self._rel_mlp(encoder_output)
                    rc_rel_h, _ = rel_h.split(batch_size, dim=0)
                    rc_rel_c, _ = rel_c.split(batch_size, dim=0)
        else:
            # (batch, seq_len, hidden_size)
            #input_encoder_output = self._input_encoder(input_word, input_char, input_pos, mask=root_mask, device=device)
            rc_all_encoder_layers = self.graph_attention(input_encoder_output, rc_gen_heads_onehot, root_mask)
            rc_encoder_output = rc_all_encoder_layers[-1]
            norc_all_encoder_layers = self.graph_attention(input_encoder_output, norc_gen_heads_onehot, root_mask)
            norc_encoder_output = norc_all_encoder_layers[-1]
            #print ("rc_encoder_output:\n",rc_encoder_output)
            norc_arc_h, norc_arc_c = self._arc_mlp(norc_encoder_output)
            rc_arc_h, rc_arc_c = self._arc_mlp(rc_encoder_output)
            if not self.use_input_encode_for_rel:
                # (batch, length, type_space), out_type shape 
                rc_rel_h, rc_rel_c = self._rel_mlp(rc_encoder_output)

        # ----- compute arc loss -----
        if self.always_recompute:
            # (batch, seq_len, seq_len)
            rc_head_logp = self._get_head_logp(rc_arc_c, rc_arc_h, rc_encoder_output)
            # (batch), number of ref heads in total
            rc_num_heads = rc_ref_heads_onehot.sum() + 1e-5
            # (batch), reference loss for recompute
            rc_ref_heads_logp = (rc_head_logp * rc_ref_heads_onehot).sum(dim=(1,2))
            loss_arc = - rc_ref_heads_logp.sum()/rc_num_heads
            loss_recomp = torch.zeros_like(loss_arc)
        else:
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
            rc_probs, rc_logp, norc_logp, loss_recomp = self._get_recomp_logp(masked_norc_head_logp, rc_gen_mask, norc_gen_mask, root_mask)
            # compute arc logp for recompute generate mask
            # (batch, seq_len, seq_len)
            rc_head_logp_given_rc = self._get_head_logp(rc_arc_c, rc_arc_h, rc_encoder_output)
            if self.apply_recomp_prob_first:
                # (batch, seq_len, seq_len)
                head_probs = (1-rc_probs).unsqueeze(1).unsqueeze(2) * torch.exp(norc_head_logp_given_norc) + rc_probs.unsqueeze(1).unsqueeze(2) * torch.exp(rc_head_logp_given_rc)
                head_logp = torch.log(head_probs)
                # (batch), number of ref heads in total
                rc_num_heads = rc_ref_heads_onehot.sum() + 1e-5
                # (batch), reference loss for recompute
                ref_heads_logp = (head_logp * rc_ref_heads_onehot).sum(dim=(1,2))
                loss_arc = - ref_heads_logp.sum()/rc_num_heads
                if debug:
                    print ("rc_probs:\n", rc_probs)
                    print ("rc_head_logp_given_norc:\n", rc_head_logp_given_rc)
                    print ("norc_head_logp_given_norc:\n", norc_head_logp_given_norc)
                    print ("head_logp:\n", head_logp)
                if not torch.isfinite(loss_arc):
                    np.set_printoptions(threshold=np.inf)
                    print ("None Finite Loss Detected:", loss_arc)
                    print ("rc_probs:\n", rc_probs)
                    print ("rc_head_logp_given_rc:\n", rc_head_logp_given_rc)
                    print ("norc_head_logp_given_norc:\n", norc_head_logp_given_norc)
                    print ("torch.exp(rc_head_logp_given_rc):\n", torch.exp(rc_head_logp_given_rc))
                    print ("head_logp:\n", head_logp.detach().cpu().numpy())
                    print ("head_logp* rc_ref_heads_onehot:\n", (head_logp*rc_ref_heads_onehot).detach().cpu().numpy())
                    print ("ref_heads_logp:\n", ref_heads_logp)
            else:
                # (batch, seq_len, seq_len)
                norc_head_logp = norc_logp.unsqueeze(1).unsqueeze(2) + norc_head_logp_given_norc
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
            #if self.use_hard_concrete_dist:
            #    loss_recomp = l0_loss
            #else:
                # regularizer of recompute prob, prevent always predicting recompute
                
            
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
        norc_loss_rel = self.criterion(norc_rel_logits, rels_3D) * (mask_3D * heads_3D)
        norc_loss_rel = norc_loss_rel.sum() / num_total_heads
        """

        # compute label loss for no recompute
        if debug:
            print ("rels:\n",rels)
            print ("rels_3D:\n",rels_3D)
            print ("heads_3D:\n", heads_3D)
            print ("num_total_heads:",num_total_heads)

        if not fast_mode and self.encode_all_arc_for_rel:
            #print ("Encoding all arc for relation")
            # encode all arcs for relation prediction
            all_encoder_layers = self.graph_attention(input_encoder_output, heads_3D, root_mask)
            # (batch, length, hidden_size)
            graph_attention_output = all_encoder_layers[-1]
            # output size [batch, length, type_space]
            rel_h, rel_c = self._rel_mlp(graph_attention_output)
        elif self.use_input_encode_for_rel:
            rel_h, rel_c = self._rel_mlp(input_encoder_output)
        else:
            rel_h = rc_rel_h
            rel_c = rc_rel_c

        # (batch, n_rels, seq_len, seq_len)
        rel_logits = self.rel_attn(rel_c, rel_h) #.permute(0, 2, 3, 1)
        # (batch, seq_len, seq_len)
        loss_rel = self.criterion(rel_logits, rels_3D) * heads_3D

        #print ("rc_loss_rel:\n",rc_loss_rel)

        loss_rel = loss_rel.sum() / num_total_heads

        #loss_rel = 0.5*norc_loss_rel + 0.5*rc_loss_rel


        return loss_arc.unsqueeze(0), loss_rel.unsqueeze(0), loss_recomp.unsqueeze(0)


    def _decode_one_step(self, head_logp, heads_mask, mask, device=torch.device('cpu'), 
                            debug=False, get_order=False, random_recomp=False, recomp_prob=0.25,
                            use_1d_mask=False):
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
                if self.use_hard_concrete_dist:
                    # (batch)
                    rc_probs, _ = self.recomp_dist._get_prob(self.recomp_dense(features).squeeze(-1))
                else:
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
                if use_1d_mask:
                    # n * (batch, seq_len), 1 for dep of new arc
                    order_mask.append(new_arc_onehot.sum(-1))
                else:
                    # n * (batch, seq_len, seq_len)
                    order_mask.append(new_arc_onehot)

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
        if self.always_recompute:
            random_recomp = True
            recomp_prob = 1
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
        # (batch, seq_len, hidden_size)
        input_encoder_output = self._input_encoder(input_word, input_char, input_pos, mask=root_mask, device=device)
        
        # compute arc logp for no recompute generate mask
        if self.use_input_encode_for_rel:
            rel_h, rel_c = self._rel_mlp(input_encoder_output)
            # (batch_size, seq_len, seq_len, n_rels)
            rel_logits = self.rel_attn(rel_c, rel_h).permute(0, 2, 3, 1)
            # (batch_size, seq_len, seq_len)
            rel_ids = rel_logits.argmax(-1)

        rel_embeddings = None

        while True:
            # ----- encoding -----
            if self.do_encode_rel:
                # (batch, seq_len, seq_len)
                masked_rel_ids = rel_ids * gen_heads_onehot
                # (batch, seq_len, seq_len, rel_dim)
                rel_embeddings = self.rel_embed(masked_rel_ids)
                if debug:
                    np.set_printoptions(threshold=np.inf)
                    print ("rel_ids:\n", rel_ids)
                    print ("masked_rel_ids:\n", masked_rel_ids)
                    print ("rel_embeddings:\n", rel_embeddings.detach().numpy())
            # (batch, seq_len, hidden_size)
            #input_encoder_output = self._input_encoder(input_word, input_char, input_pos, mask=root_mask, device=device)
            all_encoder_layers = self.graph_attention(input_encoder_output, gen_heads_onehot, root_mask,
                                                        rel_embeddings=rel_embeddings)
            encoder_output = all_encoder_layers[-1]
            # ----- compute arc probs -----
            # compute arc logp for no recompute generate mask
            arc_h, arc_c = self._arc_mlp(encoder_output)
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
        if self.encode_all_arc_for_rel:
            #input_encoder_output = self._input_encoder(input_word, input_char, input_pos, mask=root_mask)
            all_encoder_layers = self.graph_attention(input_encoder_output, gen_heads_onehot, root_mask)
            encoder_output = all_encoder_layers[-1]
            # compute arc logp for no recompute generate mask
            rel_h, rel_c = self._rel_mlp(encoder_output)
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
        num_words = root_mask.sum(-1) - 1 + 1e-9
        freq_recomp = (num_recomp-1) / num_words
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
                  explore=False, debug=False, device=torch.device('cpu'), use_1d_mask=False):
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

        if self.always_recompute:
            random_recomp = True
            recomp_prob = 1
        else:
            random_recomp = False
            recomp_prob = 0

        # for neural network
        # (batch_size, seq_len)
        heads_pred = torch.zeros((batch_size, seq_len), dtype=torch.int64, device=device)
        heads_mask = torch.zeros_like(heads_pred, device=device)

        # collect inference results
        basic_keys = ['WORD', 'MASK', 'LENGTH', 'POS', 'CHAR', 'HEAD', 'TYPE']
        all_keys =  basic_keys + ['RECOMP_GEN_MASK', 'NO_RECOMP_GEN_MASK', 'REF_MASK', 'NEXT_HEAD_MASK']
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
        gold_heads_3D = gold_heads_3D * mask_3D

        while True:
            # ----- encoding -----
            # (batch, seq_len, hidden_size)
            input_encoder_output = self._input_encoder(input_word, input_char, input_pos, mask=root_mask, device=device)
            all_encoder_layers = self.graph_attention(input_encoder_output, gen_heads_onehot, root_mask)
            encoder_output = all_encoder_layers[-1]
            # ----- compute arc probs -----
            # compute arc logp for no recompute generate mask
            arc_h, arc_c = self._arc_mlp(encoder_output)
            # (batch, seq_len, seq_len)
            head_logp = self._get_head_logp(arc_c, arc_h, encoder_output)
            
            # (batch, seq_len, seq_len)
            neg_inf_logp = torch.Tensor(head_logp.size()).fill_(-1e9).to(device)
            if explore:
                logp_mask = (1-heads_mask).unsqueeze(-1).expand_as(head_logp) * mask_3D
            else:
                # only allow gold arcs
                logp_mask = gold_heads_3D * (1-heads_mask).unsqueeze(-1).expand_as(head_logp) * mask_3D
            # (batch, seq_len, seq_len), mask out generated heads
            masked_head_logp = torch.where(logp_mask==1, head_logp.detach(), neg_inf_logp)
            # rc_probs_list: k* (batch), the probability of recompute after each arc generated
            # new_heads_onehot: (batch, seq_len, seq_len), newly generated arcs
            rc_probs_list, new_heads_onehot, order_mask = self._decode_one_step(masked_head_logp, 
                                        heads_mask, root_mask, device=device, get_order=True, 
                                        random_recomp=random_recomp, recomp_prob=recomp_prob,
                                        use_1d_mask=use_1d_mask)
            
            if use_1d_mask:
                tmp_rc_mask = heads_mask
                heads_mask_ = heads_mask.cpu().numpy()
                # (batch, seq_len)
                for i, next_head_mask in enumerate(order_mask):
                    # (batch)
                    has_head = next_head_mask.sum(-1).cpu().numpy()
                    next_head_mask_ = next_head_mask.cpu().numpy()
                    tmp_rc_mask_ = tmp_rc_mask.cpu().numpy()
                    ref_mask_ = (root_mask.int() - tmp_rc_mask).cpu().numpy()
                    for j in range(batch_size):
                        if has_head[j] == 1:
                            for key in basic_keys:
                                sampled_batch[key].append(batch[key][j])
                            sampled_batch['RECOMP_GEN_MASK'].append(tmp_rc_mask_[j])
                            sampled_batch['NO_RECOMP_GEN_MASK'].append(heads_mask_[j])
                            sampled_batch['NEXT_HEAD_MASK'].append(next_head_mask_[j])
                            sampled_batch['REF_MASK'].append(ref_mask_[j])
                            #next_list.append(next_head_mask_[j])
                            #rc_gen_list.append(tmp_rc_mask_[j])
                            #norc_gen_list.append(heads_mask_[j])
                    tmp_rc_mask = tmp_rc_mask + next_head_mask
            else:
                tmp_rc_mask = gen_heads_onehot
                heads_mask_ = gen_heads_onehot.cpu().numpy()
                # (batch, seq_len, seq_len)
                for i, next_head_mask in enumerate(order_mask):
                    # (batch)
                    has_head = next_head_mask.sum(dim=(-1,-2)).cpu().numpy()
                    next_head_mask_ = (gold_heads_3D * next_head_mask.sum(-1).unsqueeze(-1)).cpu().numpy()
                    tmp_rc_mask_ = tmp_rc_mask.cpu().numpy()
                    # (batch, seq_len)
                    tmp_rc_mask_1d = 1 - tmp_rc_mask.sum(-1)
                    ref_mask_ = (gold_heads_3D * tmp_rc_mask_1d.unsqueeze(-1)).cpu().numpy()
                    for j in range(batch_size):
                        if has_head[j] == 1:
                            for key in basic_keys:
                                sampled_batch[key].append(batch[key][j])
                            sampled_batch['RECOMP_GEN_MASK'].append(tmp_rc_mask_[j])
                            sampled_batch['NO_RECOMP_GEN_MASK'].append(heads_mask_[j])
                            sampled_batch['NEXT_HEAD_MASK'].append(next_head_mask_[j])
                            sampled_batch['REF_MASK'].append(ref_mask_[j])
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


class EasyFirstV2(EasyFirst):
    def __init__(self, *args, **kwargs):
        super(EasyFirstV2, self).__init__(*args, **kwargs)

    def _max_3D(self, tensor):
        """
        Input:
            tensor: (batch, seq_len, seq_len)
        Return:
            max_val: (batch), the max value
            max_tensor_3D: (batch, seq_len, seq_len)
            max_heads_2D: (batch, seq_len), each entry is the index of head
        """
        batch_size, seq_len, _ = tensor.size()
        # (batch, seq_len*seq_len)
        flatten_tensor = tensor.view([batch_size, -1])
        # (batch)
        max_val, max_indices = flatten_tensor.max(dim=1)
        max_dep_indices = max_indices // seq_len
        max_head_indices = max_indices % seq_len

        dep_mask = torch.zeros(batch_size, seq_len, dtype=torch.int32, device=tensor.device)
        dep_mask.scatter_(1, max_dep_indices.unsqueeze(1), 1)

        max_heads_2D = torch.zeros(batch_size, seq_len, dtype=torch.int32, device=tensor.device)
        max_heads_2D.scatter_(1, max_dep_indices.unsqueeze(1), max_head_indices.unsqueeze(1).int())

        max_tensor_3D = torch.zeros((batch_size, seq_len, seq_len), dtype=torch.int32, device=tensor.device)
        max_tensor_3D.scatter_(-1, max_heads_2D.unsqueeze(-1).long(), 1)
        max_tensor_3D = max_tensor_3D * dep_mask.unsqueeze(-1)

        return max_val, max_tensor_3D, max_heads_2D


    def _get_loss_arc_one_step(self, head_logp, gen_arcs_3D, gold_arcs_3D, order_mask, 
                                mask_3D=None, margin=1, debug=False):

        # (batch, seq_len, seq_len)
        neg_inf_tensor = torch.Tensor(head_logp.size()).fill_(-1e9).to(head_logp.device)
        # (batch, seq_len)
        unfinished_token_mask_2D = (1 - gen_arcs_3D.sum(-1))
        # (batch, seq_len, seq_len), the gold arcs not generated
        ref_heads_mask = unfinished_token_mask_2D.unsqueeze(-1) * gold_arcs_3D
        # (batch, seq_len, seq_len), mask out all rows whose heads are found
        valid_mask = unfinished_token_mask_2D.unsqueeze(-1) * mask_3D
        # (batch, seq_len, seq_len), mask out invalid positions
        head_logp = torch.where(valid_mask==1, head_logp, neg_inf_tensor)

        # (batch), number of ref heads in total
        num_heads = ref_heads_mask.sum() + 1e-5
        # (batch), sum of reference losses
        ref_heads_logp = (head_logp * ref_heads_mask).sum(dim=(1,2))
        loss_arc = - ref_heads_logp.sum()/num_heads
        loss_recomp = torch.zeros_like(loss_arc)

        # Select arcs with max scores
        order_masked_head_logp = torch.where(order_mask.unsqueeze(-1).expand_as(head_logp)==1, head_logp, neg_inf_tensor)
        # (batch, seq_len, seq_len), only remained gold arcs are left
        gold_head_logp = torch.where(ref_heads_mask==1, order_masked_head_logp, neg_inf_tensor)
        # (batch, seq_len, seq_len), false arcs are left
        false_head_logp = torch.where(gold_arcs_3D==0, order_masked_head_logp, neg_inf_tensor)

        gold_max_val, gold_max_tensor, _ = self._max_3D(gold_head_logp)
        false_max_val, false_max_tensor, _ = self._max_3D(false_head_logp)
        general_max_val, general_max_tensor, _ = self._max_3D(order_masked_head_logp)

        if debug:
            print ("gold_arcs_3D:\n", gold_arcs_3D)
            print ("head_logp:\n", head_logp)
            print ("order_mask:\n", order_mask)
            print ("order_masked_head_logp:\n", order_masked_head_logp)
            print ("gold_max_tensor:\n", gold_max_tensor)
            print ("general_max_tensor:\n", general_max_tensor)
            print ("false_max_tensor:\n", false_max_tensor)

        return loss_arc, loss_recomp, general_max_tensor, gold_max_tensor, false_max_tensor

    def _get_input_encoder_output(self, input_word, input_char, input_pos, mask):
        batch_size, seq_len = input_word.size()
        # (batch, seq_len), seq mask, where at position 0 is 0
        root_mask = torch.arange(seq_len, device=mask.device).gt(0).float().unsqueeze(0) * mask
        # (batch, seq_len, hidden_size)
        input_encoder_output = self._input_encoder(input_word, input_char, input_pos, mask=root_mask, device=mask.device)
        return input_encoder_output

    def forward(self, input_word, input_char, input_pos, gen_arcs_3D, heads, rels, order_mask, 
                mask=None, explore=True, debug=False):
        if explore and self.encode_rel_type == 'gold':
            print ("### Error: Training with explore but encoding gold relation! ###")
            exit()
        input_encoder_output = self._get_input_encoder_output(input_word, input_char, input_pos, mask)
        # Pre-processing
        batch_size, seq_len = heads.size()
        # (batch, seq_len), seq mask, where at position 0 is 0
        root_mask = torch.arange(seq_len, device=heads.device).gt(0).float().unsqueeze(0) * mask
        # (batch, seq_len, seq_len)
        mask_3D = (root_mask.unsqueeze(-1) * mask.unsqueeze(1))
        # (batch, seq_len, seq_len)
        gold_arcs_3D = torch.zeros((batch_size, seq_len, seq_len), dtype=torch.int32, device=heads.device)
        gold_arcs_3D.scatter_(-1, heads.unsqueeze(-1), 1)
        gold_arcs_3D = gold_arcs_3D * mask_3D
        # (batch, seq_len, seq_len)
        rels_3D = torch.zeros((batch_size, seq_len, seq_len), dtype=torch.long, device=heads.device)
        #rels[:,0] = 0
        rels_3D.scatter_(-1, heads.unsqueeze(-1), rels.unsqueeze(-1))
        
        # Compute relation loss
        if self.use_input_encode_for_rel:
            # get vector for heads [batch, length, rel_space]     
            rel_h, rel_c = self._rel_mlp(input_encoder_output)
        else:
            print ("Not Implemented!")
        # (batch, n_rels, seq_len, seq_len)
        rel_logits = self.rel_attn(rel_c, rel_h)
        loss_rel = (self.criterion(rel_logits, rels_3D) * gold_arcs_3D).sum(-1)
        if mask is not None:
            loss_rel = loss_rel * mask
        loss_rel = loss_rel[:, 1:].sum() / gold_arcs_3D.sum()


        if self.encode_rel_type == "gold":
            rel_ids = rels_3D
        elif self.encode_rel_type == "pred":
            # (batch, n_rels, seq_len, seq_len) => (batch, seq_len, seq_len, n_rels)
            reformed_rel_logits = rel_logits.permute(0, 2, 3, 1)
            # (batch_size, seq_len, seq_len)
            rel_ids = reformed_rel_logits.argmax(-1)
        else:
            rel_ids = None
        rel_embeddings = None
        if self.always_recompute:
            if self.do_encode_rel:
                # (batch, seq_len, seq_len)
                masked_rel_ids = rel_ids * gen_arcs_3D
                # (batch, seq_len, seq_len, rel_dim)
                rel_embeddings = self.rel_embed(masked_rel_ids)
            if debug:
                np.set_printoptions(threshold=np.inf)
                print ("rel_ids:\n", rel_ids)
                print ("masked_rel_ids:\n", masked_rel_ids)
                print ("rel_embeddings:\n", rel_embeddings.detach().numpy())

            all_encoder_layers = self.graph_attention(input_encoder_output, gen_arcs_3D, root_mask, 
                                                        rel_embeddings=rel_embeddings)
            # [batch, length, hidden_size]
            encoder_output = all_encoder_layers[-1]
            arc_h, arc_c = self._arc_mlp(encoder_output)
            #arc_logits = self.arc_attn(arc_c, arc_h)
            # (batch, seq_len, seq_len)
            head_logp = self._get_head_logp(arc_c, arc_h, encoder_output)
            # loss_arc: (batch)
            loss_arc, loss_recomp, general_max_tensor, gold_max_tensor, false_max_tensor = self._get_loss_arc_one_step(head_logp, gen_arcs_3D, gold_arcs_3D, order_mask, mask_3D)
            # (batch), count the number of error predictions
            if explore:
                # * order_mask to remove out of length arcs
                # update with max arcs among all
                gen_arcs_3D = gen_arcs_3D + (general_max_tensor * order_mask.unsqueeze(-1)).int()
            else:
                # update with max gold arcs
                gen_arcs_3D = gen_arcs_3D + (gold_max_tensor * order_mask.unsqueeze(-1)).int()
            #print ("### gen_arcs_3D:\n",gen_arcs_3D)
        else:
            print ("Not Implemented!")
        return loss_arc.unsqueeze(0), loss_rel.unsqueeze(0), loss_recomp.unsqueeze(0), gen_arcs_3D.detach()


    def inference(self, input_word, input_char, input_pos, heads, mask=None, explore=True):
        # Pre-processing
        batch_size, seq_len = input_word.size()
        # (batch, seq_len), seq mask, where at position 0 is 0
        root_mask = torch.arange(seq_len, device=heads.device).gt(0).float().unsqueeze(0) * mask
        # (batch, seq_len, seq_len)
        mask_3D = (root_mask.unsqueeze(-1) * mask.unsqueeze(1))
        # (batch, seq_len, seq_len)
        gold_arcs_3D = torch.zeros((batch_size, seq_len, seq_len), dtype=torch.int32, device=heads.device)
        gold_arcs_3D.scatter_(-1, heads.unsqueeze(-1), 1)
        gold_arcs_3D = gold_arcs_3D * mask_3D

        # (batch, seq_len, seq_len)
        gen_arcs_3D = torch.zeros((batch_size, seq_len, seq_len), dtype=torch.int32, device=heads.device)
        # arc_logits shape [batch, seq_len, hidden_size]
        input_encoder_output = self._input_encoder(input_word, input_char, input_pos, mask=root_mask, device=heads.device)
        
        # (batch, seq_len), 1 represent the token whose head is to be generated at this step
        order_masks = []
        # (batch, seq_len)
        ones = torch.ones_like(heads)
        for i in range(seq_len-1):
            if self.always_recompute:
                all_encoder_layers = self.graph_attention(input_encoder_output, gen_arcs_3D, root_mask)
                # [batch, length, hidden_size]
                encoder_output = all_encoder_layers[-1]
                arc_h, arc_c = self._arc_mlp(encoder_output)
                #arc_logits = self.arc_attn(arc_c, arc_h)
                # (batch, seq_len, seq_len)
                head_logp = self._get_head_logp(arc_c, arc_h, encoder_output)
                # loss_arc: (batch)
                _, _, general_max_tensor, gold_max_tensor, false_max_tensor = self._get_loss_arc_one_step(head_logp, gen_arcs_3D, gold_arcs_3D, ones, mask_3D)
                # (batch, seq_len)
                order_mask = gold_max_tensor.sum(-1)
                order_masks.append(order_mask.detach())
                # update with max gold arcs
                gen_arcs_3D = gen_arcs_3D + gold_max_tensor.detach()
            else:
                print ("Not Implemented!")
        # (seq_len-1, batch, seq_len)
        order_masks = torch.stack(order_masks)
        trans_mask = root_mask.permute(1,0)[1:,:]
        order_masks = order_masks * trans_mask.int().unsqueeze(-1)

        return order_masks.detach()

    """
    def forward(self, input_word, input_char, input_pos, heads, rels, order_masks,
                mask=None, explore=True):
        # Pre-processing
        batch_size, seq_len = input_word.size()
        # (batch, seq_len), seq mask, where at position 0 is 0
        root_mask = torch.arange(seq_len, device=heads.device).gt(0).float().unsqueeze(0) * mask
        # (batch, seq_len, seq_len)
        mask_3D = (root_mask.unsqueeze(-1) * mask.unsqueeze(1))
        # (batch, seq_len, seq_len)
        gold_arcs_3D = torch.zeros((batch_size, seq_len, seq_len), dtype=torch.int32, device=heads.device)
        gold_arcs_3D.scatter_(-1, heads.unsqueeze(-1), 1)
        gold_arcs_3D = gold_arcs_3D * mask_3D
        # (batch, seq_len, seq_len)
        rels_3D = torch.zeros((batch_size, seq_len, seq_len), dtype=torch.long, device=heads.device)
        rels_3D.scatter_(-1, heads.unsqueeze(-1), rels.unsqueeze(-1))

        # (batch, seq_len, seq_len)
        gen_arcs_3D = torch.zeros((batch_size, seq_len, seq_len), dtype=torch.int32, device=heads.device)
        # arc_logits shape [batch, seq_len, hidden_size]
        input_encoder_output = self._input_encoder(input_word, input_char, input_pos, mask=root_mask, device=heads.device)
        
        # Compute relation loss
        if self.use_input_encode_for_rel:
            # get vector for heads [batch, length, rel_space]     
            rel_h, rel_c = self._rel_mlp(input_encoder_output)
        else:
            print ("Not Implemented!")
        # (batch, n_rels, seq_len, seq_len)
        rel_logits = self.rel_attn(rel_c, rel_h)
        loss_rel = (self.criterion(rel_logits, rels_3D) * gold_arcs_3D).sum(-1)
        if mask is not None:
            loss_rel = loss_rel * mask
        loss_rel = loss_rel[:, 1:].sum() / gold_arcs_3D.sum()

        losses_arc = []
        losses_recomp = []
        # (batch, seq_len, seq_len) => (seq_len, batch, seq_len)
        order_masks = order_masks.permute(1,0,2)
        for i in range(seq_len-1):
            # (batch, seq_len), 1 represent the token whose head is to be generated at this step
            order_mask = order_masks[i]
            if self.always_recompute:
                all_encoder_layers = self.graph_attention(input_encoder_output, gen_arcs_3D, root_mask)
                # [batch, length, hidden_size]
                encoder_output = all_encoder_layers[-1]
                arc_h, arc_c = self._arc_mlp(encoder_output)
                #arc_logits = self.arc_attn(arc_c, arc_h)
                # (batch, seq_len, seq_len)
                head_logp = self._get_head_logp(arc_c, arc_h, encoder_output)
                # loss_arc: (batch)
                loss_arc, loss_recomp, general_max_tensor, gold_max_tensor, false_max_tensor = self._get_loss_arc_one_step(head_logp, gen_arcs_3D, gold_arcs_3D, order_mask, mask_3D)
                # (batch), count the number of error predictions
                losses_arc.append(loss_arc.unsqueeze(0))
                losses_recomp.append(loss_recomp.unsqueeze(0))
                if explore:
                    # update with max arcs among all
                    gen_arcs_3D = gen_arcs_3D + general_max_tensor.detach()
                else:
                    # update with max gold arcs
                    gen_arcs_3D = gen_arcs_3D + gold_max_tensor.detach()
                yield loss_arc.unsqueeze(0), loss_rel.unsqueeze(0), loss_recomp.unsqueeze(0)
            else:
                print ("Not Implemented!")
        # (batch)
        loss_arc = torch.cat(losses_arc).mean()
        loss_recomp = torch.cat(losses_recomp).mean()

        # [batch, length - 1] -> [batch] remove the symbolic root.
        #return loss_arc.unsqueeze(0), loss_rel.unsqueeze(0), loss_recomp.unsqueeze(0)
    """
