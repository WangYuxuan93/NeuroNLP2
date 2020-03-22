import torch.nn.functional as F
from neuronlp2.tasks.MST import arc_argmax, rel_argmax, softmax2d
import torch.optim.lr_scheduler
#from driver.Layer import *
import numpy as np


def pad_sequence(xs, length=None, padding=-1, dtype=np.float64):
    lengths = [len(x) for x in xs]
    if length is None:
        length = max(lengths)
    y = np.array([np.pad(x.astype(dtype), (0, length - l),
                         mode="constant", constant_values=padding)
                  for x, l in zip(xs, lengths)])
    return torch.from_numpy(y)

def _model_var(model, x):
    p = next(filter(lambda p: p.requires_grad, model.parameters()))
    if p.is_cuda:
        x = x.cuda(p.get_device())
    return torch.autograd.Variable(x)


class BiaffineParser(object):
    def __init__(self, model, root_id):
        self.model = model
        self.root = root_id
        p = next(filter(lambda p: p.requires_grad, model.parameters()))
        self.use_cuda = p.is_cuda
        self.device = p.get_device() if self.use_cuda else None

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def parameters(self):
        return self.model.parameters()

    def state_dict(self):
        return self.model.state_dict()

    def forward(self, words, extwords, tags, masks, positions):
        if self.use_cuda:
            words, extwords = words.cuda(self.device), extwords.cuda(self.device),
            tags = tags.cuda(self.device)
            masks = masks.cuda(self.device)
            positions=positions.cuda(self.device)

        arc_logits, rel_logits = self.model.forward(words, extwords, tags, masks, positions)
        # cache
        self.arc_logits = arc_logits
        self.rel_logits = rel_logits


    def compute_loss(self, true_arcs, true_rels, mask):
        b, l1, l2 = self.arc_logits.size()
        # pad = 0
        index_true_arcs = true_arcs
        neg_ones = torch.ones_like(true_arcs) * -1
        # pad = -1
        true_arcs = torch.where(mask.eq(1), true_arcs, neg_ones)
        #print ("true_arcs:\n", true_arcs)
        #print ("index_true_arcs:\n", index_true_arcs)
        """
        index_true_arcs = _model_var(
            self.model,
            pad_sequence(true_arcs, length=l1, padding=0, dtype=np.int64))
        true_arcs = _model_var(
            self.model,
            pad_sequence(true_arcs, length=l1, padding=-1, dtype=np.int64))
        #print(index_true_arcs)#6*73
        #print(true_arcs)#6*73
        masks = []
        for length in lengths:
            mask = torch.FloatTensor([0] * length + [-10000] * (l2 - length))
            #print("mask before _model_var")
            #print(mask)#size:l2 73;value:0 and -1000
            mask = _model_var(self.model, mask)
            #print("mask  after _model_var")
            #print(mask)#size and valus are the same as mask before _model_var
            mask = torch.unsqueeze(mask, dim=1).expand(-1, l1)
            #print("mask before append")
            #print(mask)#size:73(l2)*73(l1)
            masks.append(mask.transpose(0, 1))
            #print("masks after append")
            #print(masks)
        #print("final length_mask")
        length_mask = torch.stack(masks, 0)
        """
        minus_mask = mask.eq(0).unsqueeze(2)
        arc_logits = self.arc_logits.masked_fill(minus_mask, float('-inf'))
        #print(length_mask)#size:6*73(l2)*73(l1)
        #arc_logits = self.arc_logits + length_mask
        #print(arc_logits)#size:6*73*73
        #print(arc_logits.view(b*l1,l2))#size:438(6*73)*73
        #print(true_arcs.view(b*l1))#size:438

        arc_loss = F.cross_entropy(
            arc_logits.view(b * l1, l2), true_arcs.view(b * l1),
            ignore_index=-1)
        #print(arc_loss)#one number of cross_entropy ;when arc_loss is smaller,the result is more correct

        size = self.rel_logits.size()
        output_logits = _model_var(self.model, torch.zeros(size[0], size[1], size[3]))

        for batch_index, (logits, arcs) in enumerate(zip(self.rel_logits, index_true_arcs)):
            #print(batch_index)#one number form 0 to 6
            #print(logits)#73*73*43
            #print(arcs)#tensor size of 73
            rel_probs = []
            for i in range(l1):
                rel_probs.append(logits[i][int(arcs[i])])
            rel_probs = torch.stack(rel_probs, dim=0)
            output_logits[batch_index] = torch.squeeze(rel_probs, dim=1)

        #print(rel_probs)#tensor size of 73*43
        #print(output_logits)#tensor size of 6*73*43
        b, l1, d = output_logits.size()
        neg_ones = torch.ones_like(true_rels) * -1
        # pad = -1
        true_rels = torch.where(mask.eq(1), true_rels, neg_ones)
        #true_rels = _model_var(self.model, pad_sequence(true_rels, padding=-1, dtype=np.int64))

        rel_loss = F.cross_entropy(
            output_logits.view(b * l1, d), true_rels.view(b * l1), ignore_index=-1)

        #loss = arc_loss + rel_loss

        return arc_loss, rel_loss

    def loss(self, input_word, input_pre, input_char, input_pos, heads, types, mask=None):
        batch_size, seq_len  = input_word.size()
        position_ids = torch.arange(seq_len, dtype=torch.long, device=input_word.device)
        # (batch, seq_len)
        position_ids = position_ids.unsqueeze(0).expand(batch_size,-1) * mask.long()
        self.forward(input_word, input_pre, input_pos, mask, position_ids)
        arc_loss, rel_loss = self.compute_loss(heads, types, mask)
        arc_correct, label_correct, total_arcs = self.compute_accuracy(heads, types, mask)
        return arc_loss, rel_loss, arc_correct, label_correct, total_arcs

    def compute_accuracy(self, true_arcs, true_rels, mask):
        b, l1, l2 = self.arc_logits.size()
        pred_arcs = self.arc_logits.data.max(2)[1]#.cpu()

        neg_ones = torch.ones_like(true_arcs) * -1
        # pad = -1
        true_arcs = torch.where(mask.eq(1), true_arcs, neg_ones)
        index_true_arcs = true_arcs
        #index_true_arcs = pad_sequence(true_arcs, padding=-1, dtype=np.int64)
        #true_arcs = pad_sequence(true_arcs, padding=-1, dtype=np.int64)
        arc_correct = pred_arcs.eq(true_arcs).cpu().sum()


        size = self.rel_logits.size()
        output_logits = _model_var(self.model, torch.zeros(size[0], size[1], size[3]))

        for batch_index, (logits, arcs) in enumerate(zip(self.rel_logits, index_true_arcs)):
            rel_probs = []
            for i in range(l1):
                rel_probs.append(logits[i][arcs[i]])
            rel_probs = torch.stack(rel_probs, dim=0)
            output_logits[batch_index] = torch.squeeze(rel_probs, dim=1)

        pred_rels = output_logits.data.max(2)[1]#.cpu()
        #print ("pred_rels:\n", pred_rels)
        #print ("true_rels:\n", true_rels)
        neg_ones = torch.ones_like(true_rels) * -1
        # pad = -1
        true_rels = torch.where(mask.eq(1), true_rels, neg_ones)
        #true_rels = pad_sequence(true_rels, padding=-1, dtype=np.int64)
        label_correct = pred_rels.eq(true_rels).cpu().sum()

        total_arcs = b * l1 - np.sum(true_arcs.cpu().numpy() == -1)

        return arc_correct, label_correct, total_arcs

    def decode(self, input_word, input_pre, input_char, input_pos, mask=None, leading_symbolic=None):
        batch_size, seq_len  = input_word.size()
        position_ids = torch.arange(seq_len, dtype=torch.long, device=input_word.device)
        # (batch, seq_len)
        position_ids = position_ids.unsqueeze(0).expand(batch_size,-1) * mask.long()
        # (batch)
        lengths = mask.sum(-1).int().cpu().numpy()
        heads_pred, types_pred = self.parse(input_word, input_pre, input_pos, lengths, mask, position_ids)
        #print ("heads_pred:\n", heads_pred)
        #print ("types_pred:\n", types_pred)
        heads_pred_ = np.zeros((batch_size, seq_len), dtype=np.int32)
        types_pred_ = np.zeros((batch_size, seq_len), dtype=np.int32)
        for b, (head, type) in enumerate(zip(heads_pred, types_pred)):
            heads_pred_[b][:len(head)] = head
            types_pred_[b][:len(type)] = type
        return heads_pred_, types_pred_

    def parse(self, words, extwords, tags, lengths, masks, positions):
        if words is not None:
            self.forward(words, extwords, tags, masks, positions)
        ROOT = self.root
        arcs_batch, rels_batch = [], []
        arc_logits = self.arc_logits.data.cpu().numpy()
        rel_logits = self.rel_logits.data.cpu().numpy()

        for arc_logit, rel_logit, length in zip(arc_logits, rel_logits, lengths):
            arc_probs = softmax2d(arc_logit, length, length)
            arc_pred = arc_argmax(arc_probs, length)
            
            rel_probs = rel_logit[np.arange(len(arc_pred)), arc_pred]
            rel_pred = rel_argmax(rel_probs, length, ROOT)

            arcs_batch.append(arc_pred)
            rels_batch.append(rel_pred)

        return arcs_batch, rels_batch
