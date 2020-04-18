__author__ = 'max'

from overrides import overrides
import numpy as np
from enum import Enum
import torch
import torch.nn as nn
import torch.nn.functional as F
from neuronlp2.io import get_logger
from neuronlp2.nn import TreeCRF, VarGRU, VarRNN, VarLSTM, VarFastLSTM
from neuronlp2.nn import BiAffine, BiLinear, CharCNN, BiAffine_v2
from neuronlp2.tasks import parser
from neuronlp2.nn.self_attention import AttentionEncoderConfig, AttentionEncoder
from neuronlp2.nn.graph_attention_network import GraphAttentionNetworkConfig, GraphAttentionNetwork
from neuronlp2.nn.dropout import drop_input_independent
from torch.autograd import Variable


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


class RefinementParser(nn.Module):
    def __init__(self, hyps, num_pretrained, num_words, num_chars, num_pos, num_labels,
                 device=torch.device('cpu'), basic_word_embedding=True, 
                 embedd_word=None, embedd_char=None, embedd_pos=None):
        super(RefinementParser, self).__init__()

        self.hyps = hyps
        self.device = device
        # for input embeddings
        use_pos = hyps['input']['use_pos']
        use_char = hyps['input']['use_char']
        word_dim = hyps['input']['word_dim']
        pos_dim = hyps['input']['pos_dim']
        char_dim = hyps['input']['char_dim']
        self.basic_word_embedding = basic_word_embedding
        # for biaffine layer
        arc_mlp_dim = hyps['biaffine']['arc_mlp_dim']
        rel_mlp_dim = hyps['biaffine']['rel_mlp_dim']
        p_in = hyps['biaffine']['p_in']
        self.p_in = p_in
        p_out = hyps['biaffine']['p_out']
        activation = hyps['biaffine']['activation']
        self.act_func = activation
        self.minimize_logp = hyps['biaffine']['minimize_logp']
        # for input encoder
        input_encoder_name = hyps['input_encoder']['name']
        hidden_size = hyps['input_encoder']['hidden_size']
        num_layers = hyps['input_encoder']['num_layers']
        p_rnn = hyps['input_encoder']['p_rnn']
        # for graph encoder
        graph_encoder_name = hyps['graph_encoder']['name']
        self.encode_arc_type = hyps['graph_encoder']['encode_arc_type']
        if self.encode_arc_type.startswith('hard-'):
            self.encode_arc_topk = int(self.encode_arc_type.split('-')[1])
        self.encode_rel_type = hyps['graph_encoder']['encode_rel_type']
        if self.encode_rel_type == 'gold' or self.encode_rel_type == 'pred':
            self.do_encode_rel = True
        else:
            self.do_encode_rel = False
        self.use_null_att_pos = hyps['graph_encoder']['use_null_att_pos']
        #self.end_word_id = end_word_id

        logger = get_logger("Network")
        model = "{}-{}-{}".format(hyps['model'], input_encoder_name, graph_encoder_name)
        logger.info("Network: %s, hidden=%d, act=%s" % (model, hidden_size, activation))
        logger.info("##### Embeddings (POS tag: %s, Char: %s) #####" % (use_pos, use_char))
        logger.info("dropout(in, out): (%.2f, %.2f)" % (p_in, p_out))
        logger.info("Use Randomly Init Word Emb: %s" % (basic_word_embedding))
        logger.info("##### Input Encoder (Type: %s, Layer: %d, Hidden: %d) #####" % (input_encoder_name, num_layers, hidden_size))

        # Initialization
        if self.basic_word_embedding:
            self.basic_word_embed = nn.Embedding(num_words, word_dim, padding_idx=1)
            self.word_embed = nn.Embedding(num_pretrained, word_dim, _weight=embedd_word, padding_idx=1)
        else:
            self.basic_word_embed = None
            self.word_embed = nn.Embedding(num_words, word_dim, _weight=embedd_word, padding_idx=1)

        #self.word_embed.weight.requires_grad=False
        self.pos_embed = nn.Embedding(num_pos, pos_dim, _weight=embedd_pos, padding_idx=1) if use_pos else None
        if use_char:
            self.char_embed = nn.Embedding(num_chars, char_dim, _weight=embedd_char, padding_idx=1)
            self.char_cnn = CharCNN(2, char_dim, char_dim, hidden_channels=char_dim * 4, activation=activation)
        else:
            self.char_embed = None
            self.char_cnn = None

        self.dropout_in = nn.Dropout2d(p=p_in)
        self.dropout_out = nn.Dropout2d(p=p_out)
        self.num_labels = num_labels

        enc_dim = word_dim
        if use_char:
            enc_dim += char_dim
        if use_pos:
            enc_dim += pos_dim

        self.input_encoder_name = input_encoder_name
        if input_encoder_name == 'Linear':
            self.input_encoder = nn.Linear(enc_dim, hidden_size)
            self.position_embedding_layer = PositionEmbeddingLayer(enc_dim, dropout_prob=0, 
                                                                max_position_embeddings=256)
            out_dim = hidden_size
        elif input_encoder_name == 'FastLSTM':
            self.input_encoder = VarFastLSTM(enc_dim, hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True, dropout=p_rnn)
            out_dim = hidden_size * 2
            logger.info("dropout(p_rnn): (%.2f, %.2f)" % (p_rnn[0], p_rnn[1]))
        elif input_encoder_name == 'Transformer':
            num_attention_heads = hyps['input_encoder']['num_attention_heads']
            intermediate_size = hyps['input_encoder']['intermediate_size']
            hidden_act = hyps['input_encoder']['hidden_act']
            dropout_type = hyps['input_encoder']['dropout_type']
            embedding_dropout_prob = hyps['input_encoder']['embedding_dropout_prob']
            hidden_dropout_prob = hyps['input_encoder']['hidden_dropout_prob']
            inter_dropout_prob = hyps['input_encoder']['inter_dropout_prob']
            attention_probs_dropout_prob = hyps['input_encoder']['attention_probs_dropout_prob']
            use_input_layer = hyps['input_encoder']['use_input_layer']
            use_sin_position_embedding = hyps['input_encoder']['use_sin_position_embedding']
            freeze_position_embedding = hyps['input_encoder']['freeze_position_embedding']
            initializer = hyps['input_encoder']['initializer']
            if not use_input_layer and not enc_dim == hidden_size:
                print ("enc_dim ({}) does not match hidden_size ({}) with no input layer!".format(enc_dim, hidden_size))
                exit()

            self.attention_config = AttentionEncoderConfig(input_size=enc_dim,
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
                                                    initializer_range=0.02)
            self.input_encoder = AttentionEncoder(self.attention_config)
            out_dim = hidden_size
            logger.info("dropout(emb, hidden, inter, att): (%.2f, %.2f, %.2f, %.2f)" % (embedding_dropout_prob, 
                                hidden_dropout_prob, inter_dropout_prob, attention_probs_dropout_prob))
            logger.info("Use Sin Position Embedding: %s (Freeze it: %s)" % (use_sin_position_embedding, freeze_position_embedding))
            logger.info("Use Input Layer: %s" % use_input_layer)
        elif input_encoder_name == 'None':
            self.input_encoder = None
            out_dim = enc_dim
        else:
            self.input_encoder = None
            out_dim = enc_dim
        
        # for graph encoder
        logger.info("##### Graph Encoder (Type: %s, Layers: %d, Hidden: %d, Share:%s) #####"% (graph_encoder_name,
         hyps['graph_encoder']['num_layers'], hyps['graph_encoder']['hidden_size'], hyps['graph_encoder']['share_params']))
        if graph_encoder_name == 'GAT':
            self.gat_config = GraphAttentionNetworkConfig(input_size=out_dim,
                                                hidden_size=hyps['graph_encoder']['hidden_size'],
                                                num_layers=hyps['graph_encoder']['num_layers'],
                                                num_attention_heads=hyps['graph_encoder']['num_attention_heads'],
                                                share_params=hyps['graph_encoder']['share_params'],
                                                only_value_weight=hyps['graph_encoder']['only_value_weight'],
                                                intermediate_size=hyps['graph_encoder']['intermediate_size'],
                                                hidden_act=hyps['graph_encoder']['hidden_act'],
                                                dropout_type=hyps['graph_encoder']['dropout_type'],
                                                embedding_dropout_prob=hyps['graph_encoder']['embedding_dropout_prob'],
                                                hidden_dropout_prob=hyps['graph_encoder']['hidden_dropout_prob'],
                                                inter_dropout_prob=hyps['graph_encoder']['inter_dropout_prob'],
                                                attention_probs_dropout_prob=hyps['graph_encoder']['attention_probs_dropout_prob'],
                                                use_input_layer=hyps['graph_encoder']['use_input_layer'],
                                                use_sin_position_embedding=hyps['graph_encoder']['use_sin_position_embedding'],
                                                freeze_position_embedding=hyps['graph_encoder']['freeze_position_embedding'],
                                                max_position_embeddings=256, initializer_range=0.02,
                                                initializer=hyps['graph_encoder']['initializer'],
                                                rel_dim=hyps['graph_encoder']['rel_dim'], do_encode_rel=self.do_encode_rel,
                                                use_null_att_pos=hyps['graph_encoder']['use_null_att_pos'])

            self.graph_attention = GraphAttentionNetwork(self.gat_config)
        else:
            logger.info("Unrecognized graph encoder: %s", graph_encoder_name)
        
        logger.info("dropout(emb, hidden, inter, att): (%.2f, %.2f, %.2f, %.2f)" % (hyps['graph_encoder']['embedding_dropout_prob'],
                hyps['graph_encoder']['hidden_dropout_prob'],hyps['graph_encoder']['inter_dropout_prob'], hyps['graph_encoder']['attention_probs_dropout_prob']))
        logger.info("Use Input Layer: %s" % hyps['graph_encoder']['use_input_layer'])
        logger.info("Only Use Value Weight: %s" % hyps['graph_encoder']['only_value_weight'])
        logger.info("Attend to END if no head: %s" % hyps['graph_encoder']['use_null_att_pos'])
        logger.info("Encode Arc Type: %s" % (self.encode_arc_type))
        logger.info("Encode Relation Type: %s (rel embed dim: %d)" % (self.encode_rel_type, hyps['graph_encoder']['rel_dim']))

        # for biaffine scorer
        self.arc_h = nn.Linear(out_dim, arc_mlp_dim)
        self.arc_c = nn.Linear(out_dim, arc_mlp_dim)
        #self.arc_attention = BiAffine(arc_mlp_dim, arc_mlp_dim)
        self.arc_attention = BiAffine_v2(arc_mlp_dim, bias_x=True, bias_y=False)

        self.rel_h = nn.Linear(out_dim, rel_mlp_dim)
        self.rel_c = nn.Linear(out_dim, rel_mlp_dim)
        #self.rel_attention = BiLinear(rel_mlp_dim, rel_mlp_dim, self.num_labels)
        self.rel_attention = BiAffine_v2(rel_mlp_dim, n_out=self.num_labels, bias_x=True, bias_y=True)

        if self.minimize_logp:
            self.dep_dense = nn.Linear(out_dim, 1)

        assert activation in ['elu', 'leaky_relu', 'tanh']
        if activation == 'elu':
            self.activation = nn.ELU(inplace=True)
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.1)
        else:
            self.activation = nn.Tanh()
        self.criterion = nn.CrossEntropyLoss(reduction='none')
        self.reset_parameters(embedd_word, embedd_char, embedd_pos)
        logger.info('# of Parameters: %d' % (sum([param.numel() for param in self.parameters()])))

    def reset_parameters(self, embedd_word, embedd_char, embedd_pos):
        if embedd_word is None:
            nn.init.uniform_(self.word_embed.weight, -0.1, 0.1)
        if embedd_char is None and self.char_embed is not None:
            nn.init.uniform_(self.char_embed.weight, -0.1, 0.1)
        if embedd_pos is None and self.pos_embed is not None:
            nn.init.uniform_(self.pos_embed.weight, -0.1, 0.1)
        if self.basic_word_embed is not None:
            nn.init.uniform_(self.basic_word_embed.weight, -0.1, 0.1)

        with torch.no_grad():
            self.word_embed.weight[self.word_embed.padding_idx].fill_(0)
            if self.basic_word_embed is not None:
                self.basic_word_embed.weight[self.basic_word_embed.padding_idx].fill_(0)
            if self.char_embed is not None:
                self.char_embed.weight[self.char_embed.padding_idx].fill_(0)
            if self.pos_embed is not None:
                self.pos_embed.weight[self.pos_embed.padding_idx].fill_(0)

        if self.act_func == 'leaky_relu':
            nn.init.kaiming_uniform_(self.arc_h.weight, a=0.1, nonlinearity='leaky_relu')
            nn.init.kaiming_uniform_(self.arc_c.weight, a=0.1, nonlinearity='leaky_relu')
            nn.init.kaiming_uniform_(self.rel_h.weight, a=0.1, nonlinearity='leaky_relu')
            nn.init.kaiming_uniform_(self.rel_c.weight, a=0.1, nonlinearity='leaky_relu')
        else:
            nn.init.xavier_uniform_(self.arc_h.weight)
            nn.init.xavier_uniform_(self.arc_c.weight)
            nn.init.xavier_uniform_(self.rel_h.weight)
            nn.init.xavier_uniform_(self.rel_c.weight)

        nn.init.constant_(self.arc_h.bias, 0.)
        nn.init.constant_(self.arc_c.bias, 0.)
        nn.init.constant_(self.rel_h.bias, 0.)
        nn.init.constant_(self.rel_c.bias, 0.)

        if self.minimize_logp:
            nn.init.xavier_uniform_(self.dep_dense.weight)
            nn.init.constant_(self.dep_dense.bias, 0.)

        if self.input_encoder_name == 'Linear':
            nn.init.xavier_uniform_(self.input_encoder.weight)
            nn.init.constant_(self.input_encoder.bias, 0.)

    def _get_rnn_output(self, input_word, input_pretrained, input_char, input_pos, mask=None):
        
        #print ("input_word:\n", input_word)
        #print ("input_pretrained:\n", input_pretrained)
        
        # apply dropout word on input
        #word = self.dropout_in(word)
        
        if self.basic_word_embedding:
            # [batch, length, word_dim]
            pre_word = self.word_embed(input_pretrained)
            enc_word = pre_word
            basic_word = self.basic_word_embed(input_word)
            #print ("pre_word:\n", pre_word)
            #print ("basic_word:\n", basic_word)
            #basic_word = self.dropout_in(basic_word)
            enc_word = enc_word + basic_word
        else:
            # if not basic word emb, still use input_word as index
            pre_word = self.word_embed(input_word)
            enc_word = pre_word

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
        if self.input_encoder_name == 'Linear':
            # sequence shared mask dropout
            enc = self.dropout_in(enc.transpose(1, 2)).transpose(1, 2)
            enc = self.position_embedding_layer(enc)
            output = self.input_encoder(enc)
        elif self.input_encoder_name == 'Transformer':
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
        output = self.dropout_out(output.transpose(1, 2)).transpose(1, 2)
        self.encoder_output = output

        # output size [batch, length, arc_mlp_dim]
        arc_h = self.activation(self.arc_h(output))
        arc_c = self.activation(self.arc_c(output))

        # output size [batch, length, rel_mlp_dim]
        rel_h = self.activation(self.rel_h(output))
        rel_c = self.activation(self.rel_c(output))

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

        return arc_correct.unsqueeze(0), rel_correct.unsqueeze(0), total_arcs.unsqueeze(0)


    def forward(self, input_word, input_pretrained, input_char, input_pos, heads, rels, mask=None):
        # output from rnn [batch, length, dim]
        arc_hidden, rel_hidden = self._get_rnn_output(input_word, input_pretrained, input_char, input_pos, mask=mask)
        # [batch, length_head, length_child]
        arc_logits = self.arc_attention(arc_hidden[1], arc_hidden[0])

        # [batch, length, rel_mlp_dim]
        rel_h, rel_c = rel_hidden

        # get vector for heads [batch, length, rel_mlp_dim],
        #rel_h = rel_h.gather(dim=1, index=heads.unsqueeze(2).expand(rel_h.size()))
        # compute output for type [batch, length, num_labels]
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
        rel_logits = self.rel_attention(rel_c, rel_h)

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
        #loss_rel = self.criterion(out_type.transpose(1, 2), rels)
        loss_rel = (self.criterion(rel_logits, rels_3D) * heads_3D).sum(-1)

        statistics = self.accuracy(arc_logits, rel_logits, heads, rels, root_mask)

        # mask invalid position to 0 for sum loss
        if mask is not None:
            loss_arc = loss_arc * mask
            loss_rel = loss_rel * mask

        # [batch, length - 1] -> [batch] remove the symbolic root.
        return loss_arc[:, 1:].sum(dim=1), loss_rel[:, 1:].sum(dim=1), statistics


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
                number of symbolic labels leading in type alphabets (set it to 0 if you are not sure)

        Returns: (Tensor, Tensor)
                predicted heads and rels.

        """
        # output from rnn [batch, length, dim]
        arc_hidden, rel_hidden = self._get_rnn_output(input_word, input_pretrained, input_char, input_pos, mask=mask)
        # [batch, length_head, length_child]
        arc_logits = self.arc_attention(arc_hidden[1], arc_hidden[0])

        # out_type shape [batch, length, rel_mlp_dim]
        rel_h, rel_c = rel_hidden
        batch, max_len, rel_mlp_dim = rel_h.size()

        #rel_h = rel_h.unsqueeze(2).expand(batch, max_len, max_len, rel_mlp_dim).contiguous()
        #rel_c = rel_c.unsqueeze(1).expand(batch, max_len, max_len, rel_mlp_dim).contiguous()
        # compute output for type [batch, length_h, length_c, num_labels]
        #out_type = self.rel_attention(rel_h, rel_c)
        # (batch, n_rels, seq_len_c, seq_len_h)
        # => (batch, length_h, length_c, num_labels)
        rel_logits = self.rel_attention(rel_c, rel_h).permute(0,3,2,1)

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


