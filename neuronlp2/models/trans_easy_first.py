from overrides import overrides
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from neuronlp2.nn import TreeCRF, VarGRU, VarRNN, VarLSTM, VarFastLSTM
from neuronlp2.nn import BiAffine, BiLinear, CharCNN, BiAffine_v2
from neuronlp2.tasks import parser
from neuronlp2.nn.transformer import SelfAttentionConfig, SelfAttentionModel
from neuronlp2.models.parsing import PositionEmbeddingLayer

class TransEasyFirst(nn.Module):
    def __init__(self, word_dim, num_words, char_dim, num_chars, pos_dim, num_pos, rnn_mode, 
                 hidden_size, num_layers, num_labels, arc_space, rel_space,
                 embedd_word=None, embedd_char=None, embedd_pos=None, p_in=0.33, p_out=0.33, 
                 p_rnn=(0.33, 0.33), pos=True, use_char=False, activation='elu',
                 num_attention_heads=8, intermediate_size=1024, minimize_logp=False):
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

        self.input_encoder_type = rnn_mode
        if rnn_mode == 'RNN':
            RNN = VarRNN
        elif rnn_mode == 'LSTM':
            RNN = VarLSTM
        elif rnn_mode == 'FastLSTM':
            RNN = VarFastLSTM
        elif rnn_mode == 'GRU':
            RNN = VarGRU
        elif rnn_mode == 'Linear':
            self.position_embedding_layer = PositionEmbeddingLayer(dim_enc, dropout_prob=0, 
                                                                max_position_embeddings=256)
            print ("Using Linear Encoder!")
            #self.linear_encoder = True
        elif rnn_mode == 'Transformer':
            print ("Using Transformer Encoder!")
        else:
            raise ValueError('Unknown RNN mode: %s' % rnn_mode)
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

        self.arc_h = nn.Linear(out_dim, arc_space)
        self.arc_c = nn.Linear(out_dim, arc_space)
        #self.biaffine = BiAffine(arc_space, arc_space)
        self.biaffine = BiAffine_v2(arc_space, bias_x=True, bias_y=False)

        self.type_h = nn.Linear(out_dim, rel_space)
        self.type_c = nn.Linear(out_dim, rel_space)
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

        nn.init.xavier_uniform_(self.type_h.weight)
        nn.init.constant_(self.type_h.bias, 0.)
        nn.init.xavier_uniform_(self.type_c.weight)
        nn.init.constant_(self.type_c.bias, 0.)

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

    def _mlp(self, input_tensor):

        # output size [batch, length, arc_space]
        arc_h = self.activation(self.arc_h(input_tensor))
        arc_c = self.activation(self.arc_c(input_tensor))

        # output size [batch, length, rel_space]
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

    def _get_score(self, input_word, input_char, input_pos, mask=None):
        # output from rnn [batch, length, dim]
        arc, type = self._input_encoder(input_word, input_char, input_pos, mask=mask)
        # [batch, length_head, length_child]
        #arc_logits = self.biaffine(arc[0], arc[1], mask_query=mask, mask_key=mask)
        arc_logits = self.biaffine(arc[1], arc[0])
        return arc_logits, type

    def _get_arc_loss(self, logits, heads_3D, debug=False):
        # in the original code, the i,j = 1 means i is head of j
        # but in heads_3D, it means j is head of i
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
        ref_heads_logp = (head_logp * heads_3D).sum(dim=-1)
        # (batch, seq_len)
        loss_arc = - ref_heads_logp

        if debug:
            print ("logits:\n",logits)
            print ("softmax:\n", torch.softmax(logits, dim=-1))
            print ("heads_3D:\n", heads_3D)
            print ("head_logp_given_dep:\n", head_logp_given_dep)
            #print ("dep_logits:\n",self.dep_dense(context_layer).squeeze(-1))
            print ("dep_logp:\n", dep_logp)
            print ("head_logp:\n", head_logp)
            print ("heads_logp*heads_3D:\n", head_logp * heads_3D)
            print ("loss_arc:\n", loss_arc)

        return loss_arc

   # def _train_one_step(self, ):


    def forward(self, input_word, input_char, input_pos, heads, rels, mask=None):

        # Pre-processing
        batch_size, seq_len = input_word.size()
        # (batch, seq_len), seq mask, where at position 0 is 0
        root_mask = torch.arange(seq_len, device=heads.device).gt(0).float().unsqueeze(0) * mask
        # (batch, seq_len, seq_len)
        mask_3D = (root_mask.unsqueeze(-1) * mask.unsqueeze(1))
        # (batch, seq_len, seq_len)
        heads_3D = torch.zeros((batch_size, seq_len, seq_len), dtype=torch.int32, device=heads.device)
        heads_3D.scatter_(-1, heads.unsqueeze(-1), 1)
        heads_3D = heads_3D * mask_3D
        # (batch, seq_len, seq_len)
        rels_3D = torch.zeros((batch_size, seq_len, seq_len), dtype=torch.long, device=heads.device)
        rels_3D.scatter_(-1, heads.unsqueeze(-1), rels.unsqueeze(-1))

        # arc_logits shape [batch, seq_len, hidden_size]
        encoder_output = self._input_encoder(input_word, input_char, input_pos, mask=mask)
        
        arc, rel = self._mlp(encoder_output)
        
        arc_logits = self.biaffine(arc[1], arc[0])
        
        # get vector for heads [batch, length, rel_space]
        rel_h, rel_c = rel

        # (batch, n_rels, seq_len, seq_len)
        rel_logits = self.bilinear(rel_c, rel_h)

        if self.minimize_logp:
            # (batch, seq_len)
            loss_arc = self._get_arc_loss(arc_logits, heads_3D)
        else:
            # mask invalid position to -inf for log_softmax
            if mask is not None:
                minus_mask = mask.eq(0).unsqueeze(2)
                arc_logits = arc_logits.masked_fill(minus_mask, float('-inf'))
            # loss_arc shape [batch, length_c]
            loss_arc = self.criterion(arc_logits, heads)
        #loss_rel = self.criterion(rel_logits.transpose(1, 2), rels)
        loss_rel = (self.criterion(rel_logits, rels_3D) * heads_3D).sum(-1)

        # mask invalid position to 0 for sum loss
        if mask is not None:
            loss_arc = loss_arc * mask
            loss_rel = loss_rel * mask

        # [batch, length - 1] -> [batch] remove the symbolic root.
        return loss_arc[:, 1:].sum(dim=1), loss_rel[:, 1:].sum(dim=1)


    def decode(self, input_word, input_char, input_pos, mask=None, leading_symbolic=0):
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

        encoder_output = self._input_encoder(input_word, input_char, input_pos, mask=mask)
        
        arc, rel = self._mlp(encoder_output)
        arc_logits = self.biaffine(arc[1], arc[0])

        # get vector for heads [batch, length, rel_space]
        rel_h, rel_c = rel
        # (batch, n_rels, seq_len, seq_len)
        rel_logits = self.bilinear(rel_c, rel_h).permute(0,3,2,1)

        batch, max_len, rel_space = rel_h.size()

        # => (batch, length_h, length_c, num_labels)
        #out_type = self.bilinear(type_c, type_h).permute(0,3,2,1)

        if mask is not None:
            minus_mask = mask.eq(0).unsqueeze(2)
            arc_logits.masked_fill_(minus_mask, float('-inf'))
        # loss_arc shape [batch, length_h, length_c]
        loss_arc = F.log_softmax(arc_logits, dim=1)
        # loss_rel shape [batch, length_h, length_c, num_labels]
        loss_rel = F.log_softmax(rel_logits, dim=3).permute(0, 3, 1, 2)
        # [batch, num_labels, length_h, length_c]
        energy = loss_arc.unsqueeze(1) + loss_rel

        # compute lengths
        length = mask.sum(dim=1).long().cpu().numpy()
        return parser.decode_MST(energy.cpu().numpy(), length, leading_symbolic=leading_symbolic, labeled=True)
