from overrides import overrides
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from neuronlp2.nn import TreeCRF, VarGRU, VarRNN, VarLSTM, VarFastLSTM
from neuronlp2.nn import BiAffine, BiLinear, CharCNN, BiAffine_v2
from neuronlp2.tasks import parser
from neuronlp2.nn.transformer import GraphAttentionConfig, GraphAttentionModelV2
from neuronlp2.nn.transformer import SelfAttentionConfig, SelfAttentionModel
from neuronlp2.models.parsing import PositionEmbeddingLayer

class TransEasyFirst(nn.Module):
    def __init__(self, word_dim, num_words, char_dim, num_chars, pos_dim, num_pos, input_encoder, 
                 hidden_size, num_layers, num_labels, arc_space, rel_space,
                 embedd_word=None, embedd_char=None, embedd_pos=None, p_in=0.33, p_out=0.33, 
                 p_rnn=(0.33, 0.33), pos=True, use_char=False, activation='elu',
                 num_attention_heads=8, intermediate_size=1024, minimize_logp=False,
                 num_graph_attention_layers=1, share_params=False):
        super(TransEasyFirst, self).__init__()

        self.minimize_logp = minimize_logp
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
        if input_encoder == 'RNN':
            RNN = VarRNN
        elif input_encoder == 'LSTM':
            RNN = VarLSTM
        elif input_encoder == 'FastLSTM':
            RNN = VarFastLSTM
        elif input_encoder == 'GRU':
            RNN = VarGRU
        elif input_encoder == 'Linear':
            self.position_embedding_layer = PositionEmbeddingLayer(dim_enc, dropout_prob=0, 
                                                                max_position_embeddings=256)
            print ("Using Linear Encoder!")
            #self.linear_encoder = True
        elif input_encoder == 'Transformer':
            print ("Using Transformer Encoder!")
        else:
            raise ValueError('Unknown RNN mode: %s' % input_encoder)
        print ("Use POS tag: %s" % pos)
        print ("Use Char: %s" % use_char)
        print ("Input Encoder Type: %s" % self.input_encoder_type)

        if self.input_encoder_type == 'Linear':
            self.input_encoder = nn.Linear(dim_enc, hidden_size)
            out_dim = hidden_size
        elif self.input_encoder_type == 'Transformer':
            self.config = SelfAttentionConfig(input_size=dim_enc,
                                        hidden_size=hidden_size,
                                        num_hidden_layers=num_layers,
                                        num_attention_heads=num_attention_heads,
                                        intermediate_size=intermediate_size,
                                        hidden_act="gelu",
                                        hidden_dropout_prob=0.1,
                                        attention_probs_dropout_prob=0.1,
                                        max_position_embeddings=256,
                                        initializer_range=0.02)
            self.input_encoder = SelfAttentionModel(self.config)
            out_dim = hidden_size
        else:
            self.input_encoder = RNN(dim_enc, hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True, dropout=p_rnn) 
            out_dim = hidden_size * 2

        self.config = GraphAttentionConfig(input_size=out_dim,
                                            hidden_size=hidden_size,
                                            arc_space=arc_space,
                                            num_attention_heads=num_attention_heads,
                                            num_graph_attention_layers=num_graph_attention_layers,
                                            share_params=share_params,
                                            intermediate_size=intermediate_size,
                                            hidden_act="gelu",
                                            hidden_dropout_prob=0.1,
                                            attention_probs_dropout_prob=0.1,
                                            graph_attention_probs_dropout_prob=0,
                                            max_position_embeddings=256,
                                            initializer_range=0.02,
                                            extra_self_attention_layer=False,
                                            input_self_attention_layer=False,
                                            num_input_attention_layers=0)

        self.graph_attention = GraphAttentionModelV2(self.config)
        encode_dim = hidden_size

        self.arc_h = nn.Linear(encode_dim, arc_space)
        self.arc_c = nn.Linear(encode_dim, arc_space)
        #self.biaffine = BiAffine(arc_space, arc_space)
        self.biaffine = BiAffine_v2(arc_space, bias_x=True, bias_y=False)

        self.rel_h = nn.Linear(out_dim, rel_space)
        self.rel_c = nn.Linear(out_dim, rel_space)
        #self.bilinear = BiLinear(rel_space, rel_space, self.num_labels)
        self.bilinear = BiAffine_v2(rel_space, n_out=self.num_labels, bias_x=True, bias_y=True)

        self.dep_dense = nn.Linear(out_dim, 1)

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

        nn.init.xavier_uniform_(self.dep_dense.weight)
        nn.init.constant_(self.dep_dense.bias, 0.)

        if self.input_encoder_type == 'Linear':
            nn.init.xavier_uniform_(self.input_encoder.weight)
            nn.init.constant_(self.input_encoder.bias, 0.)

    def _input_encoder(self, input_word, input_char, input_pos, mask=None):
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

        # output from rnn [batch, length, hidden_size]
        if self.input_encoder_type == 'Linear':
            enc = self.position_embedding_layer(enc)
            output = self.input_encoder(enc)
        elif self.input_encoder_type == 'Transformer':
            all_encoder_layers = self.input_encoder(enc, mask)
            # [batch, length, hidden_size]
            output = all_encoder_layers[-1]
        else:
            output, _ = self.input_encoder(enc, mask)

        # apply dropout for output
        # [batch, length, hidden_size] --> [batch, hidden_size, length] --> [batch, length, hidden_size]
        output = self.dropout_out(output.transpose(1, 2)).transpose(1, 2)
        self.encoder_output = output

        return output

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

    def _get_arc_loss(self, logits, gold_arcs_3D, debug=False):
        # in the original code, the i,j = 1 means i is head of j
        # but in gold_arcs_3D, it means j is head of i
        logits = logits.permute(0,2,1)
        # (batch, seq_len, seq_len), log softmax over all possible arcs
        head_logp_given_dep = F.log_softmax(logits, dim=-1)
        
        # compute dep logp
        #if self.dep_prob_depend_on_head:
            # (batch, seq_len, seq_len) * (batch, seq_len, hidden_size) 
            # => (batch, seq_len, hidden_size)
            # stop grads, prevent it from messing up head probs
        #    context_layer = torch.matmul(logits.detach(), encoder_output.detach())
        #else:
        # (batch, seq_len, hidden_size)
        context_layer = self.encoder_output.detach()
        # (batch, seq_len)
        dep_logp = F.log_softmax(self.dep_dense(context_layer).squeeze(-1), dim=-1)
        # (batch, seq_len, seq_len)
        head_logp = head_logp_given_dep + dep_logp.unsqueeze(2)
        # (batch, seq_len)
        ref_heads_logp = (head_logp * gold_arcs_3D).sum(dim=-1)
        # (batch, seq_len)
        loss_arc = - ref_heads_logp

        if debug:
            print ("logits:\n",logits)
            print ("softmax:\n", torch.softmax(logits, dim=-1))
            print ("gold_arcs_3D:\n", gold_arcs_3D)
            print ("head_logp_given_dep:\n", head_logp_given_dep)
            #print ("dep_logits:\n",self.dep_dense(context_layer).squeeze(-1))
            print ("dep_logp:\n", dep_logp)
            print ("head_logp:\n", head_logp)
            print ("heads_logp*gold_arcs_3D:\n", head_logp * gold_arcs_3D)
            print ("loss_arc:\n", loss_arc)

        return loss_arc

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


    def _get_arc_loss_one_step(self, arc_logits, gen_arcs_3D, gold_arcs_3D, mask_3D=None, margin=1):

        # (batch, seq_len, seq_len)
        neg_inf_tensor = torch.Tensor(arc_logits.size()).fill_(-1e9).to(arc_logits.device)
        # (batch, seq_len)
        finished_token_mask = gen_arcs_3D.sum(-1)
        # (batch, seq_len, seq_len), mask out all rows whose heads are found
        valid_mask = (1-finished_token_mask).unsqueeze(-1) * mask_3D
        # (batch, seq_len, seq_len), mask out invalid positions
        arc_logits = torch.where(valid_mask==1, arc_logits, neg_inf_tensor)
        
        # (batch, seq_len, seq_len), the gold arcs not generated
        remained_gold_mask = gold_arcs_3D - gen_arcs_3D
        # (batch, seq_len, seq_len), only remained gold arcs are left
        gold_arc_logits = torch.where(remained_gold_mask==1, arc_logits, neg_inf_tensor)
        # (batch, seq_len, seq_len), false arcs are left
        false_arc_logits = torch.where(gold_arcs_3D==0, arc_logits, neg_inf_tensor)

        gold_max_val, gold_max_tensor, _ = self._max_3D(gold_arc_logits)
        false_max_val, false_max_tensor, _ = self._max_3D(false_arc_logits)
        general_max_val, general_max_tensor, _ = self._max_3D(arc_logits)
        # (batch)
        margin_loss = torch.max(margin - gold_max_val + false_max_val, torch.zeros_like(gold_max_val))

        return margin_loss, general_max_tensor, gold_max_tensor, false_max_tensor


    def forward(self, input_word, input_char, input_pos, heads, rels, mask=None):

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
        err_count = torch.zeros((batch_size), dtype=torch.int32, device=heads.device)
        # arc_logits shape [batch, seq_len, hidden_size]
        encoder_output = self._input_encoder(input_word, input_char, input_pos, mask=mask)
        arc_losses = []
        for i in range(seq_len):
            all_encoder_layers = self.graph_attention(encoder_output, gen_arcs_3D, root_mask)
            # [batch, length, hidden_size]
            graph_attention_output = all_encoder_layers[-1]
            arc_h, arc_c = self._arc_mlp(graph_attention_output)
            arc_logits = self.biaffine(arc_c, arc_h)
            # arc_loss: (batch)
            arc_loss, general_max_tensor, gold_max_tensor, false_max_tensor = self._get_arc_loss_one_step(arc_logits, gen_arcs_3D, gold_arcs_3D, mask_3D)
            # (batch), count the number of error predictions
            err_mask = 1 - (general_max_tensor * gold_max_tensor).sum(dim=(-1,-2))
            err_count = err_count + err_mask
            arc_losses.append(arc_loss.unsqueeze(-1))
            # update with max arcs among all
            gen_arcs_3D = gen_arcs_3D + general_max_tensor
        # (batch, seq_len), remove the invalid loss
        arc_loss = torch.cat(arc_losses, dim=-1) * root_mask
        arc_loss = arc_loss[:,1:].sum(dim=1)

        # get vector for heads [batch, length, rel_space]     
        rel_h, rel_c = self._rel_mlp(encoder_output)
        # (batch, n_rels, seq_len, seq_len)
        rel_logits = self.bilinear(rel_c, rel_h)
        #loss_rel = self.criterion(rel_logits.transpose(1, 2), rels)
        rel_loss = (self.criterion(rel_logits, rels_3D) * gold_arcs_3D).sum(-1)
        if mask is not None:
            rel_loss = rel_loss * mask
        rel_loss = rel_loss[:, 1:].sum(dim=1)

        # [batch, length - 1] -> [batch] remove the symbolic root.
        return arc_loss, rel_loss, err_count


    def _decode_one_step(self, arc_logits, gen_arcs_3D, mask_3D=None):

        # (batch, seq_len, seq_len)
        neg_inf_tensor = torch.Tensor(arc_logits.size()).fill_(-1e9).to(arc_logits.device)
        # (batch, seq_len)
        finished_token_mask = gen_arcs_3D.sum(-1)
        # (batch, seq_len, seq_len), mask out all rows whose heads are found
        valid_mask = (1-finished_token_mask).unsqueeze(-1) * mask_3D
        # (batch, seq_len, seq_len), mask out invalid positions
        arc_logits = torch.where(valid_mask==1, arc_logits, neg_inf_tensor)
        
        general_max_val, general_max_tensor, _ = self._max_3D(arc_logits)

        #print ("arc_logits:\n", arc_logits)

        return general_max_tensor


    def decode(self, input_word, input_char, input_pos, mask=None, leading_symbolic=0, debug=False):
        """
        Args:
            input_word: Tensor
                the word input tensor with shape = [batch, length]
            input_char: Tensor
                the character input tensor with shape = [batch, length, char_length]
            input_pos: Tensor
                the pos input tensor with shape = [batch, length]
            mask: Tensor or None
                the mask tensor with shape = [batch, length]
            length: Tensor or None
                the length tensor with shape = [batch]
            hx: Tensor or None
                the initial states of RNN
            leading_symbolic: int
                number of symbolic labels leading in type alphabets (set it to 0 if you are not sure)

        Returns: (Tensor, Tensor)
                predicted heads and types.

        """
        # Pre-processing
        batch_size, seq_len = input_word.size()
        # (batch, seq_len), seq mask, where at position 0 is 0
        root_mask = torch.arange(seq_len, device=input_word.device).gt(0).float().unsqueeze(0) * mask
        # (batch, seq_len, seq_len)
        mask_3D = (root_mask.unsqueeze(-1) * mask.unsqueeze(1))

        encoder_output = self._input_encoder(input_word, input_char, input_pos, mask=mask)
        # (batch, seq_len, seq_len)
        gen_arcs_3D = torch.zeros((batch_size, seq_len, seq_len), dtype=torch.int32, device=input_word.device)
        for i in range(seq_len):
            if debug:
                print ("gen_arcs_3D:\n", gen_arcs_3D)
            all_encoder_layers = self.graph_attention(encoder_output, gen_arcs_3D, root_mask)
            # [batch, length, hidden_size]
            graph_attention_output = all_encoder_layers[-1]
            # to add loop
            arc_h, arc_c = self._arc_mlp(graph_attention_output)
            arc_logits = self.biaffine(arc_c, arc_h)
            # (batch, seq_len, seq_len)
            general_max_tensor = self._decode_one_step(arc_logits, gen_arcs_3D, mask_3D)
            gen_arcs_3D = gen_arcs_3D + general_max_tensor

        # (batch, seq_len)
        _, heads_pred = torch.max(gen_arcs_3D, dim=-1)


        # get vector for heads [batch, length, rel_space]     
        rel_h, rel_c = self._rel_mlp(encoder_output)
        # (batch, n_rels, seq_len, seq_len)
        # => (batch, seq_len, seq_len, n_rels)
        rel_logits = self.bilinear(rel_c, rel_h).permute(0,2,3,1)
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

        # compute lengths
        length = mask.sum(dim=1).long().cpu().numpy()
        return heads_pred.cpu().numpy(), rels_pred.cpu().numpy()
