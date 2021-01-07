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
from neuronlp2.models.biaffine_parser import BiaffineParser
from neuronlp2.models.stack_pointer_parser import StackPointerParser

class EnsembleParser(nn.Module):
    def __init__(self, hyps, num_pretrained, num_words, num_chars, num_pos, num_labels,
                 device=torch.device('cpu'), model_type="Biaffine",
                 embedd_word=None, embedd_char=None, embedd_pos=None,
                 use_pretrained_static=True, use_random_static=False,
                 use_elmo=False, elmo_path=None, 
                 pretrained_lm='none', lm_path=None, num_lans=1, model_paths=None,
                 merge_by='logits',beam=5):
        super(EnsembleParser, self).__init__()

        self.pretrained_lm = pretrained_lm
        self.merge_by = merge_by
        self.networks = []
        self.use_pretrained_static = use_pretrained_static
        self.use_random_static = use_random_static
        self.model_type = model_type
        self.beam=beam
        assert merge_by in ['logits', 'probs']
        logger = get_logger("Ensemble")
        logger.info("Number of models: %d (merge by: %s)" % (len(model_paths), merge_by))
        if model_type == "Biaffine":
            for i, path in enumerate(model_paths):
                model_name = os.path.join(path, 'model.pt')
                logger.info("Loading sub-model from: %s" % model_name) 
                hyp = hyps[i]
                network = BiaffineParser(hyp, num_pretrained[i], num_words[i], num_chars[i], num_pos[i],
                                   num_labels[i], device=device,
                                   pretrained_lm=pretrained_lm, lm_path=lm_path,
                                   use_pretrained_static=use_pretrained_static, 
                                   use_random_static=use_random_static,
                                   use_elmo=use_elmo, elmo_path=elmo_path,
                                   num_lans=num_lans, log_name='Network-'+str(len(self.networks)))
                network = network.to(device)
                network.load_state_dict(torch.load(model_name, map_location=device))
                self.networks.append(network)
        elif model_type == "StackPointer":
            for i, path in enumerate(model_paths):
                model_name = os.path.join(path, 'model.pt')
                logger.info("Loading sub-model[stack-ptr] from: %s" % model_name)
                network = StackPointerParser(hyps, num_pretrained[i], num_words[i], num_chars[i], num_pos[i],
                                   num_labels[i], device=device,
                                   pretrained_lm=pretrained_lm, lm_path=lm_path,
                                   use_pretrained_static=use_pretrained_static,
                                   use_random_static=use_random_static,
                                   use_elmo=use_elmo, elmo_path=elmo_path,
                                   num_lans=num_lans)
                network = network.to(device)
                network.load_state_dict(torch.load(model_name, map_location=device))
                self.networks.append(network)
        else:
            print ("Ensembling %s not supported." % model_type)
            exit()
        self.hyps = self.networks[0].hyps
        self.use_elmo = any([network.use_elmo for network in self.networks])
        #has_roberta = any([network.pretrained_lm == "roberta" for network in self.networks])
        #if has_roberta:
        #    self.pretrained_lm = "roberta"
        #else:
        #    self.pretrained_lm = pretrained_lm
        self.pretrained_lms = [network.pretrained_lm for network in self.networks]
        self.lm_paths = [network.lm_path for network in self.networks]
        self.lan_emb_as_input = False

    def eval(self):
        for i in range(len(self.networks)):
            self.networks[i].eval()

    def decode(self, input_words, input_pretrained, input_chars, input_poss, mask=None, 
                bpes=None, first_idx=None, input_elmo=None, lan_id=None, leading_symbolic=0):
        if self.merge_by == 'logits':
            arc_logits_list, rel_logits_list = [], []
            for i, network in enumerate(self.networks):
                input_word, input_char, input_pos = input_words[i], input_chars[i], input_poss[i]
                sub_bpes, sub_first_idx = bpes[i], first_idx[i]
                arc_logits, rel_logits = network.get_logits(input_word, input_pretrained, input_char, 
                    input_pos, mask=mask, bpes=sub_bpes, first_idx=sub_first_idx, input_elmo=input_elmo, 
                    lan_id=lan_id, leading_symbolic=leading_symbolic)
                arc_logits_list.append(arc_logits)
                rel_logits_list.append(rel_logits)
            arc_logits = sum(arc_logits_list)
            rel_logits = sum(rel_logits_list)
        elif self.merge_by == 'probs':
            arc_logits_list, rel_logits_list = [], []
            for i, network in enumerate(self.networks):
                input_word, input_char, input_pos = input_words[i], input_chars[i], input_poss[i]
                sub_bpes, sub_first_idx = bpes[i], first_idx[i]
                arc_logits, rel_logits = network.get_probs(input_word, input_pretrained, input_char, 
                    input_pos, mask=mask, bpes=sub_bpes, first_idx=sub_first_idx, input_elmo=input_elmo, 
                    lan_id=lan_id, leading_symbolic=leading_symbolic)
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

    """
    def decode(self, input_words, input_pretrained, input_chars, input_poss, mask=None,
            bpes=None, first_idx=None, input_elmo=None, lan_id=None, leading_symbolic=0,beam=5):

        if self.model_type == "Biaffine":
            if self.merge_by == 'logits':
                arc_logits_list, rel_logits_list = [], []
                for i, network in enumerate(self.networks):
                    input_word, input_char, input_pos = input_words[i], input_chars[i], input_poss[i]
                    arc_logits, rel_logits = network.get_logits(input_word, input_pretrained, input_char,
                        input_pos, mask=mask, bpes=bpes, first_idx=first_idx, input_elmo=input_elmo,
                        lan_id=lan_id, leading_symbolic=leading_symbolic)
                    arc_logits_list.append(arc_logits)
                    rel_logits_list.append(rel_logits)
                arc_logits = sum(arc_logits_list)
                rel_logits = sum(rel_logits_list)
            elif self.merge_by == 'probs':
                arc_logits_list, rel_logits_list = [], []
                for i, network in enumerate(self.networks):
                    input_word, input_char, input_pos = input_words[i], input_chars[i], input_poss[i]
                    arc_logits, rel_logits = network.get_probs(input_word, input_pretrained, input_char,
                        input_pos, mask=mask, bpes=bpes, first_idx=first_idx, input_elmo=input_elmo,
                        lan_id=lan_id, leading_symbolic=leading_symbolic)
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
        elif self.model_type == "StackPointer":
            beam=beam
            network1 = self.networks[0]
            network2 = self.networks[1]
            input_word1, input_char1, input_pos1 = input_words[0], input_chars[0], input_poss[0]
            input_word2, input_char2, input_pos2 = input_words[1], input_chars[1], input_poss[1]
            # 初始化1 ********************
            network1.decoder.reset_noise(0)
            embeddings1 = network1._embed(input_word1, input_pretrained, input_char1, input_pos1, bpes=bpes, first_idx=first_idx, input_elmo=input_elmo, lan_id=lan_id)
            # (batch, seq_len, hidden_size)
            output_enc1, hn1 = network1._input_encoder(embeddings1, mask=mask, lan_id=lan_id)

            # output size [batch, length_encoder, arc_space]
            arc_c1 = network1.activation(network1.arc_c(output_enc1))
            # output size [batch, length_encoder, rel_space]
            rel_c1 = network1.activation(network1.rel_c(output_enc1))

            device = output_enc1.device
            rel_space = rel_c1.size(2)
            # [decoder_layers, batch, hidden_size]
            hn1 = network1._transform_decoder_init_state(hn1)
            batch, max_len, enc_dim = output_enc1.size()

            heads = torch.zeros(batch, 1, max_len, device=device, dtype=torch.int64)
            rels = torch.zeros(batch, 1, max_len, device=device, dtype=torch.int64)

            num_steps = 2 * max_len - 1
            stacked_heads = torch.zeros(batch, 1, num_steps + 1, device=device, dtype=torch.int64)
            siblings = torch.zeros(batch, 1, num_steps + 1, device=device, dtype=torch.int64) if network1.sibling else None
            hypothesis_scores = output_enc1.new_zeros((batch, 1))

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
            hx1 = hn1
            # 初始化2 ********************
            network2.decoder.reset_noise(0)
            embeddings2 = network2._embed(input_word2, input_pretrained, input_char2, input_pos2, bpes=bpes, first_idx=first_idx, input_elmo=input_elmo, lan_id=lan_id)
            # (batch, seq_len, hidden_size)
            output_enc2, hn2 = network2._input_encoder(embeddings2, mask=mask, lan_id=lan_id)

            #device = output_enc1.device
            # output size [batch, length_encoder, arc_space]
            arc_c2 = network2.activation(network2.arc_c(output_enc2))
            # output size [batch, length_encoder, rel_space]
            rel_c2 = network2.activation(network2.rel_c(output_enc2))

            # [decoder_layers, batch, hidden_size]
            hn2 = network2._transform_decoder_init_state(hn2)
            #batch2, max_len, _ = output_enc2.size()

            hx2 = hn2
            for t in range(num_steps):
                # [batch, num_hyp]
                curr_heads = stacked_heads[:, :, t]
                curr_gpars = heads.gather(dim=2, index=curr_heads.unsqueeze(2)).squeeze(2)
                curr_sibs = siblings[:, :, t] if network1.sibling else None
                # [batch, num_hyp, enc_dim]
                src_encoding1 = output_enc1.gather(dim=1, index=curr_heads.unsqueeze(2).expand(batch, num_hyp, enc_dim))
                src_encoding2 = output_enc2.gather(dim=1, index=curr_heads.unsqueeze(2).expand(batch, num_hyp, enc_dim))
                if network1.sibling:
                    mask_sib = curr_sibs.gt(0).float().unsqueeze(2)
                    output_enc_sibling1 = output_enc1.gather(dim=1, index=curr_sibs.unsqueeze(2).expand(batch, num_hyp, enc_dim)) * mask_sib
                    output_enc_sibling2 = output_enc2.gather(dim=1, index=curr_sibs.unsqueeze(2).expand(batch, num_hyp, enc_dim)) * mask_sib
                    src_encoding1 = src_encoding1 + output_enc_sibling1
                    src_encoding2 = src_encoding2 + output_enc_sibling1

                if network1.grandPar:
                    output_enc_gpar1 = output_enc1.gather(dim=1, index=curr_gpars.unsqueeze(2).expand(batch, num_hyp, enc_dim))
                    output_enc_gpar2 = output_enc2.gather(dim=1, index=curr_gpars.unsqueeze(2).expand(batch, num_hyp, enc_dim))
                    src_encoding1 = src_encoding1 + output_enc_gpar1
                    src_encoding2 = src_encoding2 + output_enc_gpar2

                # transform to decoder input
                # [batch, num_hyp, dec_dim]
                src_encoding1 = network1.activation(network1.src_dense(src_encoding1))
                src_encoding2 = network2.activation(network2.src_dense(src_encoding2))
                # output [batch * num_hyp, dec_dim]
                # hx [decoder_layer, batch * num_hyp, dec_dim]
                output_dec1, hx1 = network1.decoder.step(src_encoding1.view(batch * num_hyp, -1), hx=hx1)
                output_dec2, hx2 = network2.decoder.step(src_encoding2.view(batch * num_hyp, -1), hx=hx2)
                dec_dim = output_dec1.size(1)
                # [batch, num_hyp, dec_dim]
                output_dec1 = output_dec1.view(batch, num_hyp, dec_dim)
                output_dec2 = output_dec2.view(batch, num_hyp, dec_dim)

                # [batch, num_hyp, arc_space]
                arc_h1 = network1.activation(network1.arc_h(output_dec1))
                arc_h2 = network2.activation(network2.arc_h(output_dec2))
                # [batch, num_hyp, rel_space]
                rel_h1 = network1.activation(network1.rel_h(output_dec1))
                rel_h2 = network2.activation(network2.rel_h(output_dec2))
                # [batch, num_hyp, length]
                arc_logits1 = network1.arc_attention(arc_h1, arc_c1, mask_query=mask_hyp, mask_key=mask)
                arc_logits2 = network2.arc_attention(arc_h2, arc_c2, mask_query=mask_hyp, mask_key=mask)
                # mask invalid position to -inf for log_softmax
                if mask is not None:
                    minus_mask_enc = mask.eq(0).unsqueeze(1)
                    arc_logits1.masked_fill_(minus_mask_enc, float('-inf'))
                    arc_logits2.masked_fill_(minus_mask_enc, float('-inf'))

                arc_logits = arc_logits1+arc_logits2  # Jeffrey: add
                # [batch]
                mask_last = steps.le(t + 1)
                mask_stop = steps.le(t)
                minus_mask_hyp = mask_hyp.eq(0).unsqueeze(2)
                # [batch, num_hyp, length]
                hyp_scores = F.log_softmax(arc_logits, dim=2).masked_fill_(mask_stop.view(batch, 1, 1) + minus_mask_hyp, 0)
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
                base_index = hyp_index // max_len
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
                # *** Jeffrey: 完成head的求解******************

                rels = rels.gather(dim=1, index=base_index_expand)

                # [batch, num_hyp]
                org_rels = rels.gather(dim=2, index=child_index.unsqueeze(2)).squeeze(2)

                # [batch, num_hyp, num_steps]
                base_index_expand = base_index.unsqueeze(2).expand(batch, num_hyp, num_steps + 1)
                stacked_heads = stacked_heads.gather(dim=1, index=base_index_expand)
                stacked_heads[:, :, t + 1] = torch.where(mask_leaf, hyp_gpars, child_index)
                if network1.sibling:
                    siblings = siblings.gather(dim=1, index=base_index_expand)
                    siblings[:, :, t + 1] = torch.where(mask_leaf, child_index, torch.zeros_like(child_index))

                # [batch, num_hyp, rel_space]
                base_index_expand = base_index.unsqueeze(2).expand(batch, num_hyp, rel_space)
                child_index_expand = child_index.unsqueeze(2).expand(batch, num_hyp, rel_space)
                # [batch, num_hyp, num_labels]
                rel_logits1 = network1.rel_attention(rel_h1.gather(dim=1, index=base_index_expand), rel_c1.gather(dim=1, index=child_index_expand))
                rel_logits2 = network2.rel_attention(rel_h2.gather(dim=1, index=base_index_expand), rel_c2.gather(dim=1, index=child_index_expand))
                rel_logits = rel_logits1+rel_logits2
                hyp_type_scores = F.log_softmax(rel_logits, dim=2)
                # compute the prediction of rels [batch, num_hyp]
                hyp_type_scores, hyp_rels = hyp_type_scores.max(dim=2)
                hypothesis_scores = hypothesis_scores + hyp_type_scores.masked_fill_(mask_stop.view(batch, 1), 0)
                rels.scatter_(2, child_index.unsqueeze(2), torch.where(mask_leaf, org_rels, hyp_rels).unsqueeze(2))

                # hx [decoder_layer, batch * num_hyp, dec_dim]
                # hack to handle LSTM
                hx_index = (base_index + batch_index * prev_num_hyp).view(batch * num_hyp)
                if isinstance(hx1, tuple):
                    hx1, cx1 = hx1
                    hx1 = hx1[:, hx_index]
                    cx1 = cx1[:, hx_index]
                    hx1 = (hx1, cx1)
                    # *************
                    hx2, cx2 = hx2
                    hx2 = hx2[:, hx_index]
                    cx2 = cx2[:, hx_index]
                    hx2 = (hx2, cx2)
                else:
                    hx1 = hx1[:, hx_index]
                    # *******************
                    hx2 = hx2[:, hx_index]

            # 综合

            heads = heads[:, 0].cpu().numpy()
            rels = rels[:, 0].cpu().numpy()
            return heads, rels
        else:
            print("Ensembling %s not supported." % self.model_type)
            exit()
    """
