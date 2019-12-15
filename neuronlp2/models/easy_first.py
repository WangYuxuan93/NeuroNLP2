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
                 hidden_dropout_prob=0.1,
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

        self.graph_attention = GraphAttentionModel(self.config)

        out_dim = hidden_size
        self.arc_h = nn.Linear(out_dim, arc_space)
        self.arc_c = nn.Linear(out_dim, arc_space)
        self.arc_attn = BiAffine_v2(arc_space, bias_x=True, bias_y=False)

        self.rel_h = nn.Linear(out_dim, type_space)
        self.rel_c = nn.Linear(out_dim, type_space)
        self.rel_attn = BiAffine_v2(type_space, n_out=self.num_labels, bias_x=True, bias_y=True)

        self.arc_hidden_to_att = nn.Linear(2*arc_space, recomp_att_dim)
        self.encoder_to_att = nn.Linear(hidden_size, recomp_att_dim)
        self.recomp_att = nn.Linear(recomp_att_dim, 1, bias=False)
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


    def loss(self, input_word, input_char, input_pos, heads, rels, recomps, gen_heads,  
             mask=None, next_head=None):
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
        root_mask = torch.arange(seq_len).gt(0).float().unsqueeze(0) * mask
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
        ref_heads_onehot = torch.zeros(batch_size, seq_len, seq_len, dtype=torch.int32)
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
        gen_heads_onehot = torch.zeros(n_layers, batch_size, seq_len, seq_len, dtype=torch.int32)
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
        neg_inf_like_logp = torch.Tensor(arc_logp.size()).fill_(-1e9)
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
        rels_3D = torch.zeros((batch_size, seq_len, seq_len), dtype=torch.long)
        rels_3D.scatter_(-1, heads.unsqueeze(-1), rels.unsqueeze(-1))

        # (batch, seq_len, seq_len)
        heads_3D = torch.zeros((batch_size, seq_len, seq_len), dtype=torch.int32)
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
                debug=False):
        """
        Input:
            input_word: (batch, seq_len)
            input_char: (batch, seq_len, char_len)
            input_pos: (batch, seq_len)
            mask: (batch, seq_len)
        """
        batch_size, seq_len = input_word.size()

        # (batch_size, seq_len)
        heads_pred = torch.zeros((batch_size, seq_len), dtype=torch.int64)
        heads_mask = torch.zeros_like(heads_pred)
        rels_pred = torch.zeros_like(heads_pred)

        for batch_id in range(batch_size):
            word = input_word[batch_id:batch_id+1, :]
            char = input_char[batch_id:batch_id+1, :, :]
            pos = input_pos[batch_id:batch_id+1, :]
            mask_ = mask[batch_id:batch_id+1, :]
            # (batch, seq_len), at position 0 is 0
            root_mask = torch.arange(seq_len).gt(0).float().unsqueeze(0) * mask_
            # (n_layers=1, batch=1, seq_len, seq_len)
            gen_heads_onehot = torch.zeros((1, 1, seq_len, seq_len), dtype=torch.int32)
            #recomp_minus_mask = torch.Tensor([0,0,0]).bool()
            recomp_minus_mask = torch.Tensor([0,0,1]).bool()

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
                    gen_heads_new_layer = torch.zeros((1, 1, seq_len, seq_len), dtype=torch.int32)
                    # (n_layers+1, batch=1, seq_len, seq_len)
                    gen_heads_onehot = torch.cat([gen_heads_onehot, gen_heads_new_layer], dim=0)
                    # update recomp_minus_mask, disable do_recomp action
                    n_layers = gen_heads_onehot.size(0)
                    if n_layers == max_layers:
                        #recomp_minus_mask = torch.Tensor([0,1,0]).bool()
                        recomp_minus_mask = torch.Tensor([0,1,1]).bool()
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