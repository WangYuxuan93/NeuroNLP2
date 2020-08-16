import os
import json
from overrides import overrides
import numpy as np
from enum import Enum
import torch
import torch.nn as nn
import torch.nn.functional as F
from neuronlp2.tasks import parser
from neuronlp2.io import get_logger
from neuronlp2.models.robust_parsing import RobustParser

class EnsembleParser(nn.Module):
    def __init__(self, hyps, num_pretrained, num_words, num_chars, num_pos, num_labels,
                 device=torch.device('cpu'), basic_word_embedding=True, 
                 embedd_word=None, embedd_char=None, embedd_pos=None,
                 pretrained_lm='none', lm_path=None, num_lans=1, model_paths=None,
                 merge_by='logits'):
        super(EnsembleParser, self).__init__()

        self.pretrained_lm = pretrained_lm
        self.merge_by = merge_by
        self.networks = []
        assert merge_by in ['logits', 'probs']
        logger = get_logger("Ensemble")
        logger.info("Number of models: %d (merge by: %s)" % (len(model_paths), merge_by))
        for path in model_paths:
            model_name = os.path.join(path, 'model.pt')
            logger.info("Loading sub-model from: %s" % model_name)
            network = RobustParser(hyps, num_pretrained, num_words, num_chars, num_pos,
                               num_labels, device=device, basic_word_embedding=basic_word_embedding, 
                               pretrained_lm=pretrained_lm, lm_path=lm_path,
                               num_lans=num_lans, log_name='Network-'+str(len(self.networks)))
            network = network.to(device)
            network.load_state_dict(torch.load(model_name, map_location=device))
            self.networks.append(network)

    def eval(self):
        for i in range(len(self.networks)):
            self.networks[i].eval()

    def decode(self, input_word, input_pretrained, input_char, input_pos, mask=None, 
                bpes=None, first_idx=None, lan_id=None, leading_symbolic=0):
        if self.merge_by == 'logits':
            arc_logits_list, rel_logits_list = [], []
            for network in self.networks:
                arc_logits, rel_logits = network.get_logits(input_word, input_pretrained, input_char, 
                    input_pos, mask=mask, bpes=bpes, first_idx=first_idx, lan_id=lan_id, leading_symbolic=leading_symbolic)
                arc_logits_list.append(arc_logits)
                rel_logits_list.append(rel_logits)
            arc_logits = sum(arc_logits_list)
            rel_logits = sum(rel_logits_list)
        elif self.merge_by == 'probs':
            arc_logits_list, rel_logits_list = [], []
            for network in self.networks:
                arc_logits, rel_logits = network.get_probs(input_word, input_pretrained, input_char, 
                    input_pos, mask=mask, bpes=bpes, first_idx=first_idx, lan_id=lan_id, leading_symbolic=leading_symbolic)
                arc_logits_list.append(arc_logits)
                rel_logits_list.append(rel_logits)
            arc_logits = sum(arc_logits_list)
            rel_logits = sum(rel_logits_list)

        # arc_loss shape [batch, length_h, length_c]
        arc_loss = F.log_softmax(arc_logits, dim=1)
        # rel_loss shape [batch, length_h, length_c, num_labels]
        rel_loss = F.log_softmax(rel_logits, dim=3).permute(0, 3, 1, 2)
        # [batch, num_labels, length_h, length_c]
        energy = arc_loss.unsqueeze(1) + rel_loss

        # compute lengths
        length = mask.sum(dim=1).long().cpu().numpy()
        return parser.decode_MST(energy.cpu().numpy(), length, leading_symbolic=leading_symbolic, labeled=True)