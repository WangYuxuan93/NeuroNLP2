__author__ = 'max'

from overrides import overrides
import numpy as np
from enum import Enum
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from neuronlp2.io import get_logger
from neuronlp2.nn import TreeCRF, VarGRU, VarRNN, VarLSTM, VarFastLSTM
from neuronlp2.nn import CharCNN #, BiAffine_v2
from neuronlp2.tasks import parser
from neuronlp2.nn.transformer import SelfAttentionConfig, SelfAttentionModel
from neuronlp2.nn.self_attention import AttentionEncoderConfig, AttentionEncoder
from torch.autograd import Variable

class PriorOrder(Enum):
    DEPTH = 0
    INSIDE_OUT = 1
    LEFT2RIGTH = 2


class PositionEmbeddingLayer(nn.Module):
    def __init__(self, embedding_size, dropout_prob=0, max_position_embeddings=256):
        super(PositionEmbeddingLayer, self).__init__()
        """Adding position embeddings to input layer
        """
        self.position_embeddings = nn.Embedding(max_position_embeddings, embedding_size)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, input_tensor, debug=False):
        """
        input_tensor: (batch, seq_len, input_size)
        """
        seq_length = input_tensor.size(1)
        batch_size = input_tensor.size(0)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_tensor.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size,-1)
        position_embeddings = self.position_embeddings(position_ids)
        embeddings = input_tensor + position_embeddings
        embeddings = self.dropout(embeddings)
        if debug:
            print ("input_tensor:",input_tensor)
            print ("position_embeddings:",position_embeddings)
            print ("embeddings:",embeddings)
        return embeddings

def drop_input_independent(word_embeddings, tag_embeddings, dropout_emb):
    batch_size, seq_length, _ = word_embeddings.size()
    word_masks = word_embeddings.data.new(batch_size, seq_length).fill_(1 - dropout_emb)#6*98 ;0.67
    #print(word_masks)
    word_masks = Variable(torch.bernoulli(word_masks), requires_grad=False)#6*78 ;0,1
    #print(word_masks)
    tag_masks = tag_embeddings.data.new(batch_size, seq_length).fill_(1 - dropout_emb)
    tag_masks = Variable(torch.bernoulli(tag_masks), requires_grad=False)
    scale = 3.0 / (2.0 * word_masks + tag_masks + 1e-12)#batch_size*seq_length 1,1.5,3
    #print(scale)
    word_masks *= scale#batch_size*seq_length 0,1,1.5
    tag_masks *= scale
    #print(word_masks)
    word_masks = word_masks.unsqueeze(dim=2)#6*78*1
    #print(word_masks)
    tag_masks = tag_masks.unsqueeze(dim=2)
    word_embeddings = word_embeddings * word_masks
    tag_embeddings = tag_embeddings * tag_masks

    return word_embeddings, tag_embeddings

def drop_sequence_sharedmask(inputs, dropout, batch_first=True):
    if batch_first:
        inputs = inputs.transpose(0, 1)
    seq_length, batch_size, hidden_size = inputs.size()
    drop_masks = inputs.data.new(batch_size, hidden_size).fill_(1 - dropout)
    drop_masks = Variable(torch.bernoulli(drop_masks), requires_grad=False)
    drop_masks = drop_masks / (1 - dropout)
    drop_masks = torch.unsqueeze(drop_masks, dim=2).expand(-1, -1, seq_length).permute(2, 0, 1)
    inputs = inputs * drop_masks

    return inputs.transpose(1, 0)

def orthonormal_initializer(output_size, input_size):
    """
    adopted from Timothy Dozat https://github.com/tdozat/Parser/blob/master/lib/linalg.py
    """
    print(output_size, input_size)
    I = np.eye(output_size)
    lr = .1
    eps = .05 / (output_size + input_size)
    success = False
    tries = 0
    while not success and tries < 10:
        Q = np.random.randn(input_size, output_size) / np.sqrt(output_size)
        for i in range(100):
            QTQmI = Q.T.dot(Q) - I
            loss = np.sum(QTQmI ** 2 / 2)
            Q2 = Q ** 2
            Q -= lr * Q.dot(QTQmI) / (
                    np.abs(Q2 + Q2.sum(axis=0, keepdims=True) + Q2.sum(axis=1, keepdims=True) - 1) + eps)
            if np.max(Q) > 1e6 or loss > 1e6 or not np.isfinite(loss):
                tries += 1
                lr /= 2
                break
        success = True
    if success:
        print('Orthogonal pretrainer loss: %.2e' % loss)
    else:
        print('Orthogonal pretrainer failed, using non-orthogonal random matrix')
        Q = np.random.randn(input_size, output_size) / np.sqrt(output_size)
    return np.transpose(Q.astype(np.float32))

def reset_bias_with_orthogonal(bias):
    bias_temp = torch.nn.Parameter(torch.FloatTensor(bias.size()[0], 1))
    nn.init.orthogonal_(bias_temp)
    bias_temp = bias_temp.view(-1)
    bias.data = bias_temp.data

class NonLinear(nn.Module):
    def __init__(self, input_size, hidden_size, activation=None, initializer='orthogonal'):
        super(NonLinear, self).__init__()
        self.initializer = initializer
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.linear = nn.Linear(in_features=input_size, out_features=hidden_size)
        if activation is None:
            self._activate = lambda x: x
        else:
            if not callable(activation):
                raise ValueError("activation must be callable: rel={}".format(rel(activation)))
            self._activate = activation

        self.reset_parameters()

    def forward(self, x):
        y = self.linear(x)
        return self._activate(y)

    def reset_parameters(self):
        if self.initializer == 'orthogonal':
            nn.init.orthogonal_(self.linear.weight)
            reset_bias_with_orthogonal(self.linear.bias)
        elif self.initializer == 'orthonormal':
            W = orthonormal_initializer(self.hidden_size, self.input_size)
            self.linear.weight.data.copy_(torch.from_numpy(W))

            b = np.zeros(self.hidden_size, dtype=np.float32)
            self.linear.bias.data.copy_(torch.from_numpy(b))
        

class Biaffine(nn.Module):
    def __init__(self, in1_features, in2_features, out_features,
                 bias=(True, True)):
        super(Biaffine, self).__init__()
        self.in1_features = in1_features
        self.in2_features = in2_features
        self.out_features = out_features
        self.bias = bias
        self.linear_input_size = in1_features + int(bias[0])
        self.linear_output_size = out_features * (in2_features + int(bias[1]))
        self.linear = nn.Linear(in_features=self.linear_input_size,
                                out_features=self.linear_output_size,
                                bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        W = np.zeros((self.linear_output_size, self.linear_input_size), dtype=np.float32)
        self.linear.weight.data.copy_(torch.from_numpy(W))
       #nn.init.orthogonal(self.linear.weight)

    def forward(self, input1, input2):
        batch_size, len1, dim1 = input1.size()
        batch_size, len2, dim2 = input2.size()
        if self.bias[0]:
            ones = input1.data.new(batch_size, len1, 1).zero_().fill_(1)
            #print(ones)#6*73*1 value=1
            #print(input1) #6*73*500
            input1 = torch.cat((input1, Variable(ones)), dim=2)
            #print(input1) #6*73*501
            dim1 += 1
        if self.bias[1]:
            ones = input2.data.new(batch_size, len2, 1).zero_().fill_(1)
            input2 = torch.cat((input2, Variable(ones)), dim=2)
            dim2 += 1

        affine = self.linear(input1)

        #print("affine before view")
        #print(affine) #compute arc: affine before view is the same as affine after view 6*73*500
        affine = affine.view(batch_size, len1*self.out_features, dim2)
        #print("affine after view")
        #print(affine)
        input2 = torch.transpose(input2, 1, 2)

        biaffine = torch.transpose(torch.bmm(affine, input2), 1, 2)
        
        #print(biaffine)#compute arc:biaffine before view is 6*73*73, after view is 6*73*73*1
        biaffine = biaffine.contiguous().view(batch_size, len2, len1, self.out_features)
        #print(biaffine)

        return biaffine


class DeepBiAffineV2(nn.Module):
    def __init__(self, num_pretrained, word_dim, num_words, char_dim, num_chars, pos_dim, num_pos, rnn_mode, 
                 hidden_size, num_layers, num_labels, arc_space, rel_space,
                 basic_word_embedding=True,
                 embedd_word=None, embedd_char=None, embedd_pos=None, p_in=0.33, p_out=0.33, 
                 p_rnn=(0.33, 0.33), pos=True, use_char=False, activation='elu',
                 num_attention_heads=8, intermediate_size=1024, minimize_logp=False,
                 use_input_layer=True, use_sin_position_embedding=False, 
                 freeze_position_embedding=True, hidden_act="gelu", dropout_type="seq",
                 initializer="default", embedding_dropout_prob=0.33, hidden_dropout_prob=0.2,
                 inter_dropout_prob=0.1, attention_probs_dropout_prob=0.1,
                 mlp_initializer="orthogonal", emb_initializer="default", ff_first=True):
        super(DeepBiAffineV2, self).__init__()

        self.basic_word_embedding = basic_word_embedding
        self.minimize_logp = minimize_logp
        self.act_func = activation
        self.initializer = initializer
        self.emb_initializer = emb_initializer
        self.p_out = p_out
        self.p_in = p_in
        self.arc_space = arc_space
        self.rel_space = rel_space
        if self.basic_word_embedding:
            self.basic_word_embed = nn.Embedding(num_words, word_dim, padding_idx=1)
        else:
            self.basic_word_embed = None

        self.word_embed = nn.Embedding(num_pretrained, word_dim, _weight=embedd_word, padding_idx=1)
        self.word_embed.weight.requires_grad=False
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

        self.input_encoder_rel = rnn_mode
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
        print ("Input Encoder rel: %s" % self.input_encoder_rel)

        if self.input_encoder_rel == 'Linear':
            self.input_encoder = nn.Linear(dim_enc, hidden_size)
            out_dim = hidden_size
        elif self.input_encoder_rel == 'Transformer':
            #self.config = SelfAttentionConfig(input_size=dim_enc,
            self.config = AttentionEncoderConfig(input_size=dim_enc,
                                        hidden_size=hidden_size,
                                        num_hidden_layers=num_layers,
                                        num_attention_heads=num_attention_heads,
                                        intermediate_size=intermediate_size,
                                        hidden_act=hidden_act,
                                        dropout_type=dropout_type,
                                        embedding_dropout_prob=embedding_dropout_prob,
                                        hidden_dropout_prob=hidden_dropout_prob,
                                        inter_dropout_prob=inter_dropout_prob,
                                        attention_probs_dropout_prob=attention_probs_dropout_prob,
                                        use_input_layer=use_input_layer,
                                        use_sin_position_embedding=use_sin_position_embedding,
                                        freeze_position_embedding=freeze_position_embedding,
                                        max_position_embeddings=256,
                                        initializer=initializer,
                                        initializer_range=0.02,
                                        ff_first=ff_first)
            self.input_encoder = AttentionEncoder(self.config)
            #self.input_encoder = SelfAttentionModel(self.config)
            out_dim = hidden_size
        else:
            self.input_encoder = RNN(dim_enc, hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True, dropout=p_rnn) 
            out_dim = hidden_size * 2

        assert activation in ['elu', 'leaky_relu', 'tanh']
        if activation == 'elu':
            self.activation = nn.ELU(inplace=True)
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.1)
        else:
            self.activation = nn.Tanh()

        self.mlp_arc_dep = NonLinear(
            input_size = out_dim,
            hidden_size = arc_space+rel_space,
            activation = self.activation,
            initializer=mlp_initializer)
        self.mlp_arc_head = NonLinear(
            input_size = out_dim,
            hidden_size = arc_space+rel_space,
            activation = self.activation,
            initializer=mlp_initializer)

        self.arc_biaffine = Biaffine(arc_space, arc_space, \
                                     1, bias=(True, False))
        self.rel_biaffine = Biaffine(rel_space, rel_space, \
                                     self.num_labels, bias=(True, True))
        """ #legacy
        self.arc_h = nn.Linear(out_dim, arc_space)
        self.arc_c = nn.Linear(out_dim, arc_space)
        #self.biaffine = BiAffine(arc_space, arc_space)
        self.biaffine = BiAffine_v2(arc_space, bias_x=True, bias_y=False)

        self.rel_h = nn.Linear(out_dim, rel_space)
        self.rel_c = nn.Linear(out_dim, rel_space)
        #self.bilinear = BiLinear(rel_space, rel_space, self.num_labels)
        self.bilinear = BiAffine_v2(rel_space, n_out=self.num_labels, bias_x=True, bias_y=True)
        """

        if self.minimize_logp:
            self.dep_dense = nn.Linear(out_dim, 1)

        self.criterion = nn.CrossEntropyLoss(reduction='none')
        self.reset_parameters(embedd_word, embedd_char, embedd_pos)

    def reset_parameters(self, embedd_word, embedd_char, embedd_pos):
        if embedd_word is None:
            nn.init.uniform_(self.word_embed.weight, -0.1, 0.1)
        if embedd_char is None and self.char_embed is not None:
            nn.init.uniform_(self.char_embed.weight, -0.1, 0.1)
        if embedd_pos is None and self.pos_embed is not None:
            if self.emb_initializer == 'uniform':
                nn.init.uniform_(self.pos_embed.weight, -0.1, 0.1)
            else:
                pos_num, pos_dim = list(self.pos_embed.weight.size())
                tag_init = np.random.randn(pos_num, pos_dim).astype(np.float32)
                self.pos_embed.weight.data.copy_(torch.from_numpy(tag_init))
        if self.basic_word_embed is not None:
            if self.emb_initializer == 'uniform':
                nn.init.uniform_(self.basic_word_embed.weight, -0.1, 0.1)
            else:
                nn.init.normal_(self.basic_word_embed.weight, 0.0, 1.0 / (200 ** 0.5))

        with torch.no_grad():
            self.word_embed.weight[self.word_embed.padding_idx].fill_(0)
            if self.basic_word_embed is not None:
                self.basic_word_embed.weight[self.basic_word_embed.padding_idx].fill_(0)
            if self.char_embed is not None:
                self.char_embed.weight[self.char_embed.padding_idx].fill_(0)
            if self.pos_embed is not None:
                self.pos_embed.weight[self.pos_embed.padding_idx].fill_(0)
        """
        if self.initializer == 'orthogonal':
            nn.init.orthogonal_(self.arc_h.weight)
            nn.init.orthogonal_(self.arc_c.weight)
            nn.init.orthogonal_(self.rel_h.weight)
            nn.init.orthogonal_(self.rel_c.weight)
        elif self.initializer == 'default':
            nn.init.kaiming_uniform_(self.arc_h.weight, a=0.1, nonlinearity='leaky_relu')
            nn.init.kaiming_uniform_(self.arc_c.weight, a=0.1, nonlinearity='leaky_relu')
            nn.init.kaiming_uniform_(self.rel_h.weight, a=0.1, nonlinearity='leaky_relu')
            nn.init.kaiming_uniform_(self.rel_c.weight, a=0.1, nonlinearity='leaky_relu')
        elif self.initializer == 'xavier_uniform':
            nn.init.xavier_uniform_(self.arc_h.weight)
            nn.init.xavier_uniform_(self.arc_c.weight)
            nn.init.xavier_uniform_(self.rel_h.weight)
            nn.init.xavier_uniform_(self.rel_c.weight)

        nn.init.constant_(self.arc_h.bias, 0.)
        nn.init.constant_(self.arc_c.bias, 0.)
        nn.init.constant_(self.rel_h.bias, 0.)
        nn.init.constant_(self.rel_c.bias, 0.)
        """

        if self.minimize_logp:
            nn.init.xavier_uniform_(self.dep_dense.weight)
            nn.init.constant_(self.dep_dense.bias, 0.)

        if self.input_encoder_rel == 'Linear':
            nn.init.xavier_uniform_(self.input_encoder.weight)
            nn.init.constant_(self.input_encoder.bias, 0.)

    def _get_logits(self, input_word, input_pretrained, input_char, input_pos, mask=None):
        #print ("word:\n", input_word)
        #print ("pretrained:\n",input_pretrained)
        # [batch, length, word_dim]
        #pre_word = self.word_embed(input_word)
        pre_word = self.word_embed(input_pretrained)
        # apply dropout word on input
        #word = self.dropout_in(word)
        enc_word = pre_word
        if self.basic_word_embedding:
            basic_word = self.basic_word_embed(input_word)
            #print ("pre_emb:\n", pre_word)
            #print ("basic_emb:\n", basic_word)
            #basic_word = self.dropout_in(basic_word)
            enc_word = enc_word + basic_word

        if self.char_embed is not None:
            # [batch, length, char_length, char_dim]
            char = self.char_cnn(self.char_embed(input_char))
            #char = self.dropout_in(char)
            # concatenate word and char [batch, length, word_dim+char_filter]
            enc_word = torch.cat([enc_word, char], dim=2)

        if self.pos_embed is not None:
            # [batch, length, pos_dim]
            enc_pos = self.pos_embed(input_pos)
            # apply dropout on input
            #pos = self.dropout_in(pos)
            if self.training:
                #print ("enc_word:\n", enc_word)
                # mask by token dropout
                enc_word, enc_pos = drop_input_independent(enc_word, enc_pos, self.p_in)
                #print ("enc_word (a):\n", enc_word)
            enc = torch.cat([enc_word, enc_pos], dim=2)
        # output from rnn [batch, length, hidden_size]
        if self.input_encoder_rel == 'Linear':
            # sequence shared mask dropout
            enc = self.dropout_in(enc.transpose(1, 2)).transpose(1, 2)
            enc = self.position_embedding_layer(enc)
            output = self.input_encoder(enc)
        elif self.input_encoder_rel == 'Transformer':
            # sequence shared mask dropout
            # apply this dropout in transformer after added position embedding
            #enc = self.dropout_in(enc.transpose(1, 2)).transpose(1, 2)
            all_encoder_layers = self.input_encoder(enc, mask)
            # [batch, length, hidden_size]
            output = all_encoder_layers[-1]
        else:
            # sequence shared mask dropout
            enc = self.dropout_in(enc.transpose(1, 2)).transpose(1, 2)
            output, _ = self.input_encoder(enc, mask)

        # apply dropout for output
        # [batch, length, hidden_size] --> [batch, hidden_size, length] --> [batch, length, hidden_size]
        #output = self.dropout_out(output.transpose(1, 2)).transpose(1, 2)
        if self.training:
            output = drop_sequence_sharedmask(output, self.p_out)
        self.encoder_output = output

        # (batch, seq_len, arc_space + rel_space)
        x_all_dep = self.mlp_arc_dep(output)
        x_all_head = self.mlp_arc_head(output)

        if self.training:
            x_all_dep = drop_sequence_sharedmask(x_all_dep, self.p_out)
            x_all_head = drop_sequence_sharedmask(x_all_head, self.p_out)
        #print(x_all_dep)#6*73*600
        
        # (batch, seq_len, arc_space), (batch, seq_len, rel_space)
        x_arc_dep, x_rel_dep = torch.split(x_all_dep, [self.arc_space, self.rel_space], dim=2)
        # (batch, seq_len, arc_space), (batch, seq_len, rel_space)
        x_arc_head, x_rel_head = torch.split(x_all_head, [self.arc_space, self.rel_space], dim=2)

        arc_logit = self.arc_biaffine(x_arc_dep, x_arc_head)#6*73*73*1
        # (batch, seq_len, seq_len)
        arc_logit = torch.squeeze(arc_logit, dim=3)#6*73*73
        #print(arc_logit)
        # (batch, seq_len, seq_len, rel_size)
        rel_logit = self.rel_biaffine(x_rel_dep, x_rel_head)#6*73*73*43
        #print ("x_arc_dep, x_rel_dep, x_arc_head, x_rel_head:",x_arc_dep.size(), x_rel_dep.size()
        #                                                        , x_arc_head.size(), x_rel_head.size())
        #print ("arc_logit, rel_logit:",arc_logit.size(), rel_logit.size())

        """
        # legacy
        # output size [batch, length, arc_space]
        arc_h = self.activation(self.arc_h(output))
        arc_c = self.activation(self.arc_c(output))

        # output size [batch, length, rel_space]
        rel_h = self.activation(self.rel_h(output))
        rel_c = self.activation(self.rel_c(output))

        # apply dropout on arc
        # [batch, length, dim] --> [batch, 2 * length, dim]
        arc = torch.cat([arc_h, arc_c], dim=1)
        rel = torch.cat([rel_h, rel_c], dim=1)
        arc = self.dropout_out(arc.transpose(1, 2)).transpose(1, 2)
        arc_h, arc_c = arc.chunk(2, 1)

        # apply dropout on rel
        # [batch, length, dim] --> [batch, 2 * length, dim]
        rel = self.dropout_out(rel.transpose(1, 2)).transpose(1, 2)
        rel_h, rel_c = rel.chunk(2, 1)
        rel_h = rel_h.contiguous()
        rel_c = rel_c.contiguous()
        """

        return arc_logit, rel_logit

    def forward(self, input_word, input_char, input_pos, mask=None):
        # output from rnn [batch, length, dim]
        arc, rel = self._get_rnn_output(input_word, input_char, input_pos, mask=mask)
        # [batch, length_head, length_child]
        #out_arc = self.biaffine(arc[0], arc[1], mask_query=mask, mask_key=mask)
        out_arc = self.biaffine(arc[1], arc[0])
        return out_arc, rel

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

    def accuracy(self, arc_logits, rel_logits, heads, rels, mask, debug=False):
        """
        arc_logits: (batch, seq_len, seq_len)
        rel_logits: (batch, n_rels, seq_len, seq_len)
        heads: (batch, seq_len)
        rels: (batch, seq_len)
        mask: (batch, seq_len)
        """
        total_arcs = mask.sum()
        # (batch, seq_len)
        arc_preds = arc_logits.argmax(-2)
        # (batch_size, seq_len, seq_len, n_rels)
        transposed_rel_logits = rel_logits.permute(0, 2, 3, 1)
        # (batch_size, seq_len, seq_len)
        rel_ids = transposed_rel_logits.argmax(-1)
        # (batch, seq_len)
        rel_preds = rel_ids.gather(-1, heads.unsqueeze(-1)).squeeze()

        ones = torch.ones_like(heads)
        zeros = torch.zeros_like(heads)
        arc_correct = (torch.where(arc_preds==heads, ones, zeros) * mask).sum()
        rel_correct = (torch.where(rel_preds==rels, ones, zeros) * mask).sum()

        if debug:
            print ("arc_logits:\n", arc_logits)
            print ("arc_preds:\n", arc_preds)
            print ("heads:\n", heads)
            print ("rel_ids:\n", rel_ids)
            print ("rel_preds:\n", rel_preds)
            print ("rels:\n", rels)
            print ("mask:\n", mask)
            print ("total_arcs:\n", total_arcs)
            print ("arc_correct:\n", arc_correct)
            print ("rel_correct:\n", rel_correct)

        return arc_correct.cpu().numpy(), rel_correct.cpu().numpy(), total_arcs.cpu().numpy()

    def loss(self, input_word, input_pretrained, input_char, input_pos, heads, rels, mask=None):
        # out_arc shape [batch, length_head, length_child]
        #out_arc, out_rel  = self(input_word, input_char, input_pos, mask=mask)
        # out_rel shape [batch, length, rel_space]
        #rel_h, rel_c = out_rel

        # get vector for heads [batch, length, rel_space],
        #rel_h = rel_h.gather(dim=1, index=heads.unsqueeze(2).expand(rel_h.size()))
        # compute output for rel [batch, length, num_labels]
        #out_rel = self.bilinear(rel_h, rel_c)
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
        # (batch, n_rels, seq_len, seq_len)
        #out_rel = self.bilinear(rel_c, rel_h)

        arc_logit, rel_logit = self._get_logits(input_word, input_pretrained, input_char, input_pos, root_mask)
        # (batch, seq_len, seq_len, n_rels) => (batch, n_rels, seq_len, seq_len)
        rel_logit = rel_logit.permute(0,3,1,2)
        if self.minimize_logp:
            # (batch, seq_len)
            loss_arc = self._get_arc_loss(arc_logit, heads_3D)
        else:
            # mask invalid position to -inf for log_softmax
            if mask is not None:
                minus_mask = mask.eq(0).unsqueeze(2)
                arc_logit = arc_logit.masked_fill(minus_mask, float('-inf'))
            # loss_arc shape [batch, length_c]
            loss_arc = self.criterion(arc_logit, heads)
        #loss_rel = self.criterion(out_rel.transpose(1, 2), rels)
        loss_rel = (self.criterion(rel_logit, rels_3D) * heads_3D).sum(-1)

        arc_correct, rel_correct, total_arcs = self.accuracy(arc_logit, rel_logit, heads, rels, root_mask)

        # mask invalid position to 0 for sum loss
        if mask is not None:
            loss_arc = loss_arc * mask
            loss_rel = loss_rel * mask

        # [batch, length - 1] -> [batch] remove the symbolic root.
        return loss_arc[:, 1:].sum(dim=1), loss_rel[:, 1:].sum(dim=1), arc_correct, rel_correct, total_arcs 

    def _decode_rels(self, out_rel, heads, leading_symbolic):
        # out_rel shape [batch, length, rel_space]
        rel_h, rel_c = out_rel
        # get vector for heads [batch, length, rel_space],
        rel_h = rel_h.gather(dim=1, index=heads.unsqueeze(2).expand(rel_h.size()))
        # compute output for rel [batch, length, num_labels]
        out_rel = self.bilinear(rel_h, rel_c)
        # remove the first #leading_symbolic rels.
        out_rel = out_rel[:, :, leading_symbolic:]
        # compute the prediction of rels [batch, length]
        _, rels = out_rel.max(dim=2)
        return rels + leading_symbolic

    def decode_local(self, input_word, input_char, input_pos, mask=None, leading_symbolic=0):
        # out_arc shape [batch, length_h, length_c]
        out_arc, out_rel = self(input_word, input_char, input_pos, mask=mask)
        batch, max_len, _ = out_arc.size()
        # set diagonal elements to -inf
        diag_mask = torch.eye(max_len, device=out_arc.device, dtype=torch.uint8).unsqueeze(0)
        out_arc.masked_fill_(diag_mask, float('-inf'))
        # set invalid positions to -inf
        if mask is not None:
            minus_mask = mask.eq(0).unsqueeze(2)
            out_arc.masked_fill_(minus_mask, float('-inf'))

        # compute naive predictions.
        # predition shape = [batch, length_c]
        _, heads = out_arc.max(dim=1)

        rels = self._decode_rels(out_rel, heads, leading_symbolic)

        return heads.cpu().numpy(), rels.cpu().numpy()

    def decode(self, input_word, input_pretrained, input_char, input_pos, mask=None, leading_symbolic=0):
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
                number of symbolic labels leading in rel alphabets (set it to 0 if you are not sure)

        Returns: (Tensor, Tensor)
                predicted heads and rels.

        """
        # out_arc shape [batch, length_h, length_c]
        #out_arc, out_rel = self(input_word, input_char, input_pos, mask=mask)

        # out_rel shape [batch, length, rel_space]
        #rel_h, rel_c = out_rel
        #batch, max_len, rel_space = rel_h.size()

        #rel_h = rel_h.unsqueeze(2).expand(batch, max_len, max_len, rel_space).contiguous()
        #rel_c = rel_c.unsqueeze(1).expand(batch, max_len, max_len, rel_space).contiguous()
        # compute output for rel [batch, length_h, length_c, num_labels]
        #out_rel = self.bilinear(rel_h, rel_c)
        # (batch, n_rels, seq_len_c, seq_len_h)
        # => (batch, length_h, length_c, num_labels)
        #out_rel = self.bilinear(rel_c, rel_h).permute(0,3,2,1)
        # (batch, seq_len), seq mask, where at position 0 is 0
        seq_len = input_word.size(1)
        root_mask = torch.arange(seq_len, device=input_word.device).gt(0).float().unsqueeze(0) * mask
        arc_logit, rel_logit = self._get_logits(input_word, input_pretrained, input_char, input_pos, root_mask)
        #rel_logit = rel_logit.permute(0,3,2,1)

        if mask is not None:
            minus_mask = mask.eq(0).unsqueeze(2)
            arc_logit.masked_fill_(minus_mask, float('-inf'))
        # loss_arc shape [batch, length_h, length_c]
        loss_arc = F.log_softmax(arc_logit, dim=1)
        # loss_rel shape [batch, length_h, length_c, num_labels]
        loss_rel = F.log_softmax(rel_logit, dim=3).permute(0, 3, 2, 1)
        # [batch, num_labels, length_h, length_c]
        energy = loss_arc.unsqueeze(1) + loss_rel

        # compute lengths
        length = mask.sum(dim=1).long().cpu().numpy()
        return parser.decode_MST(energy.cpu().numpy(), length, leading_symbolic=leading_symbolic, labeled=True)