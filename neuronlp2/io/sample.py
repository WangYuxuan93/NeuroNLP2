__author__ = 'max'

import numpy as np
import torch
"""
def random_sample_(data, batch_size, step_batch_size=None, unk_replace=0., shuffle=False, 
                    target_recomp_prob=0.25):
    data_tensor, bucket_sizes = data

    bucket_indices = np.arange(len(bucket_sizes))
    if shuffle:
        np.random.shuffle((bucket_indices))

    easyfirst_keys = ['WORD', 'MASK', 'POS', 'CHAR', 'HEAD', 'TYPE']
    for bucket_id in bucket_indices:
        data = data_tensor[bucket_id]
        bucket_size = bucket_sizes[bucket_id]
        if bucket_size == 0:
            continue

        words = data['WORD']
        single = data['SINGLE']
        bucket_length = words.size(1)
        if unk_replace:
            ones = single.new_ones(bucket_size, bucket_length)
            noise = single.new_empty(bucket_size, bucket_length).bernoulli_(unk_replace).long()
            words = words * (ones - single * noise)

        indices = None
        if shuffle:
            indices = torch.randperm(bucket_size).long()
            indices = indices.to(words.device)
        for start_idx in range(0, bucket_size, batch_size):
            if shuffle:
                excerpt = indices[start_idx:start_idx + batch_size]
            else:
                excerpt = slice(start_idx, start_idx + batch_size)

            lengths = data['LENGTH'][excerpt]
            batch_length = lengths.max().item()
            # [batch_size, batch_len]
            heads = data['HEAD'][excerpt, :batch_length]
            types = data['TYPE'][excerpt, :batch_length]
            batch = {'WORD': words[excerpt, :batch_length], 'LENGTH': lengths}
            batch.update({key: field[excerpt, :batch_length] for key, field in data.items() if key in easyfirst_keys})
            
            sampled_batch = sample_generate_order(batch, lengths, target_recomp_prob=target_recomp_prob)
            yield sampled_batch
"""

def random_sample(data, batch_size, step_batch_size=None, unk_replace=0., shuffle=False, 
                    target_recomp_prob=0.25, debug=False):
    data_tensor, bucket_sizes = data

    bucket_indices = np.arange(len(bucket_sizes))
    if shuffle:
        np.random.shuffle((bucket_indices))

    easyfirst_keys = ['MASK', 'POS', 'CHAR', 'HEAD', 'TYPE','RECOMP_GEN_MASK', 'NO_RECOMP_GEN_MASK', 'NEXT_HEAD_MASK']
    for bucket_id in bucket_indices:
        data = data_tensor[bucket_id]
        bucket_size = bucket_sizes[bucket_id]
        if bucket_size == 0:
            continue

        sampled_data = sample_generate_order(data, data['LENGTH'], target_recomp_prob=target_recomp_prob)
        sample_size = sampled_data['WORD'].size(0)
        if sample_size == 0:
            continue

        indices = None
        if shuffle:
            indices = torch.randperm(sample_size).long()
            indices = indices.to(data['WORD'].device)

        for start_idx in range(0, sample_size, batch_size):
            if shuffle:
                excerpt = indices[start_idx:start_idx + batch_size]
            else:
                excerpt = slice(start_idx, start_idx + batch_size)

            words = sampled_data['WORD']
            single = sampled_data['SINGLE']
            bucket_length = words.size(1)

            if unk_replace:
                ones = single.new_ones(sample_size, bucket_length)
                noise = single.new_empty(sample_size, bucket_length).bernoulli_(unk_replace).long()
                words = words * (ones - single * noise)

            lengths = sampled_data['LENGTH'][excerpt]
            batch_length = lengths.max().item()
            # [batch_size, batch_len]
            batch = {'WORD': words[excerpt, :batch_length], 'LENGTH': lengths}
            batch.update({key: field[excerpt, :batch_length] for key, field in sampled_data.items() if key in easyfirst_keys})
            
            if debug:
                for key in batch.keys():
                    print ("%s\n"%key, batch[key])

            yield batch

def sample_generate_order(batch, lengths, target_recomp_prob=0.25, recomp_in_prev=False, debug=False):

    RECOMP = -1
    #EOS = -2
    NO_RECOMP = 0
    DO_RECOMP = 1
    #DO_EOS = 2
    batch_length = lengths.max().item()

    basic_keys = ['WORD', 'MASK', 'LENGTH', 'POS', 'CHAR', 'HEAD', 'TYPE', 'SINGLE']
    all_keys = basic_keys + ['RECOMP_GEN_MASK', 'NO_RECOMP_GEN_MASK', 'NEXT_HEAD_MASK']
    sampled_batch = {key: [] for key in all_keys}

    # for every sentence
    for i in range(len(lengths)):
        seq_len = lengths[i]
        n_recomp = int(seq_len * target_recomp_prob)
        arc_order = np.arange(1,seq_len)
        np.random.shuffle(arc_order)
        recomp_pos = np.arange(1,seq_len-1)
        np.random.shuffle(recomp_pos)
        assert len(recomp_pos) >= n_recomp
        sample_order = []
        for j, dep_id in enumerate(arc_order):
            if j in recomp_pos[:n_recomp]:
                sample_order.append(RECOMP)
                sample_order.append(dep_id)
            else:
                sample_order.append(dep_id)
        if debug:
            print ("new_order:", arc_order)
            print ("recomp_pos:",recomp_pos)
            print ("sample_order:",sample_order)
        n_step = 0
        zero_mask = np.zeros([batch_length], dtype=np.int32)
        recomp_gen_heads = np.zeros([batch_length], dtype=np.int32)
        no_recomp_gen_heads = np.zeros([batch_length], dtype=np.int32)
        # the input generated head list if do recompute before predicting
        recomp_gen_list = []
        # the input generated head list if not do recompute before predicting
        no_recomp_gen_list = []
        # whether to recompute at this step
        recomp_list = []
        # the next head to be generated, in shape of 0-1 mask
        next_list = []
        while n_step < len(sample_order):
            next_step = sample_order[n_step]
            if next_step != RECOMP:
                next_list.append(np.copy(zero_mask))
                next_list[-1][next_step] = 1
                recomp_gen_list.append(np.copy(recomp_gen_heads))
                no_recomp_gen_list.append(np.copy(no_recomp_gen_heads))
                #recomp_list.append(NO_RECOMP)
                # add one new head to the generated heads with recomp
                recomp_gen_heads[next_step] = 1
            else:
                #next_list.append(np.copy(zero_mask))
                #recomp_list.append(DO_RECOMP)
                no_recomp_gen_heads = np.copy(recomp_gen_heads)
            n_step += 1

        if debug:
            print ("recomp_gen_list:\n", recomp_gen_list)
            print ("no_recomp_gen_list:\n", no_recomp_gen_list)
            print ("next_list:\n", next_list)
        
        for n_step in range(len(next_list)):
            for key in basic_keys:
                sampled_batch[key].append(batch[key][i])
            sampled_batch['RECOMP_GEN_MASK'].append(recomp_gen_list[n_step])
            sampled_batch['NO_RECOMP_GEN_MASK'].append(no_recomp_gen_list[n_step])
            sampled_batch['NEXT_HEAD_MASK'].append(next_list[n_step])

    for key in sampled_batch.keys():
        sampled_batch[key] = torch.from_numpy(np.stack(sampled_batch[key]))

    if debug:
        for key in sampled_batch.keys():
            print ("%s\n"%key, sampled_batch[key])
    return sampled_batch


def from_model_sample(network, data, batch_size, unk_replace=0., shuffle=False, 
                      device=torch.device('cpu'), debug=False):
    data_tensor, bucket_sizes = data

    bucket_indices = np.arange(len(bucket_sizes))
    if shuffle:
        np.random.shuffle((bucket_indices))

    basic_keys = ['MASK', 'POS', 'CHAR', 'HEAD', 'TYPE']
    all_keys =  basic_keys + ['WORD', 'LENGTH', 'RECOMP_GEN_MASK', 'NO_RECOMP_GEN_MASK', 'NEXT_HEAD_MASK']
    easyfirst_keys = ['MASK', 'POS', 'CHAR', 'HEAD', 'TYPE','RECOMP_GEN_MASK', 'NO_RECOMP_GEN_MASK', 'NEXT_HEAD_MASK']

    for bucket_id in bucket_indices:
        data = data_tensor[bucket_id]
        bucket_size = bucket_sizes[bucket_id]
        if bucket_size == 0:
            continue

        words = data['WORD']
        single = data['SINGLE']
        bucket_length = words.size(1)
        if unk_replace:
            ones = single.new_ones(bucket_size, bucket_length)
            noise = single.new_empty(bucket_size, bucket_length).bernoulli_(unk_replace).long()
            words = words * (ones - single * noise)

        sampled_batches = {key: [] for key in all_keys}
        #indices = None
        #if shuffle:
        #    indices = torch.randperm(bucket_size).long()
        #    indices = indices.to(words.device)
        batch_length = data['LENGTH'].max().item()
        # sample data from model
        for start_idx in range(0, bucket_size, batch_size):
            #if shuffle:
            #    excerpt = indices[start_idx:start_idx + batch_size]
            #else:
            excerpt = slice(start_idx, start_idx + batch_size)

            lengths = data['LENGTH'][excerpt]
            # [batch_size, batch_len]
            batch = {'WORD': words[excerpt, :batch_length], 'LENGTH': lengths}
            batch.update({key: field[excerpt, :batch_length] for key, field in data.items() if key in basic_keys})
            # pre-process the input
            if torch.cuda.device_count() > 1:
                input_word = batch['WORD'].to(device)
                input_char = batch['CHAR'].to(device)
                input_pos = batch['POS'].to(device)
            else:
                input_word = batch['WORD']#.to(device)
                input_char = batch['CHAR']#.to(device)
                input_pos = batch['POS']#.to(device)
            gold_heads = batch['HEAD'].to(device)
            mask = batch['MASK'].to(device)
            sampled_batch = network.inference(input_word, input_char, input_pos, gold_heads, 
                                batch, mask=mask, device=device)
            for key in all_keys:
                sampled_batches[key].append(sampled_batch[key])

            if debug:
                for key in sampled_batch.keys():
                    print ("%s\n"%key, sampled_batch[key])
            #yield sampled_batch

        # Merging
        #if torch.cuda.is_available():
        #    torch.cuda.empty_cache()
        sampled_data = {}
        for key in all_keys:
            sampled_data[key] = torch.from_numpy(np.concatenate(sampled_batches[key], axis=0))

        if debug:
            print ("Merged Batch")
            for key in sampled_data.keys():
                print ("%s\n"%key, sampled_data[key])

        # batching
        sample_size = sampled_data['WORD'].size(0)
        indices = None
        if shuffle:
            indices = torch.randperm(sample_size).long()
            indices = indices.to(words.device)
        # sample data from model
        for start_idx in range(0, sample_size, batch_size):
            if shuffle:
                excerpt = indices[start_idx:start_idx + batch_size]
            else:
                excerpt = slice(start_idx, start_idx + batch_size)

            lengths = sampled_data['LENGTH'][excerpt]
            batch_length = lengths.max().item()
            batch = {'WORD': sampled_data['WORD'][excerpt, :batch_length], 'LENGTH': lengths}
            batch.update({key: field[excerpt, :batch_length] for key, field in sampled_data.items() if key in easyfirst_keys})

            if debug:
                for key in batch.keys():
                    print ("%s\n"%key, batch[key])
            yield batch

def split_batch_by_layer(batch_by_layer, step_batch_size, shuffle=False, debug=False):

    batches = []
    keys_ = batch_by_layer[0].keys() - ['LENGTH', 'RECOMP', 'GEN_HEAD']
    for n_layers in batch_by_layer.keys():
        if batch_by_layer[n_layers]['LENGTH'] is None: continue
        bucket_size = len(batch_by_layer[n_layers]['LENGTH'])
        for start_idx in range(0, bucket_size, step_batch_size):
            if shuffle:
                excerpt = indices[start_idx:start_idx + step_batch_size]
            else:
                excerpt = slice(start_idx, start_idx + step_batch_size)
            # [step_batch_size, batch_len]
            batch = {'LENGTH': batch_by_layer[n_layers]['LENGTH'][excerpt],
                     'RECOMP': batch_by_layer[n_layers]['RECOMP'][excerpt],
                     'GEN_HEAD': batch_by_layer[n_layers]['GEN_HEAD'][:,excerpt,:]}
            batch.update({key: field[excerpt, :] for key, field in batch_by_layer[n_layers].items() if key in keys_})
            batches.append(batch)

    if debug:
        print ("Split batches:")
        for batch in batches:
            print('-' * 50)
            for key in batch.keys():
                print ("%s\n"%key, batch[key])

    return batches


if __name__ == '__main__':
    easyfirst_keys = ['WORD', 'MASK', 'LENGTH', 'POS', 'CHAR', 'HEAD', 'TYPE']
    batch = {'WORD':[[0,1,2,3,4,5],[0,6,7,8,0,0]],'MASK':[[1,1,1,1,1,0],[1,1,1,1,1,1]],
             'POS':[[0,1,2,3,4,5],[0,6,7,8,0,0]], 'LENGTH':np.array([6,5]),
             'CHAR':[[0,1,2,3,4,5],[0,6,7,8,0,0]],'HEAD':[[0,3,1,0,3,5],[0,6,7,8,0,0]],'TYPE':[[0,1,2,3,4,5],[0,6,7,8,0,0]]}
    lengths = batch['LENGTH']
    sample_generate_order(batch, lengths, target_recomp_prob=0.25)
