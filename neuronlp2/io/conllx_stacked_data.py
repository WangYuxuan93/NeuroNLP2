__author__ = 'max'

import numpy as np
import torch
from neuronlp2.io.conllx_data import _buckets, PAD_ID_WORD, PAD_ID_CHAR, PAD_ID_TAG, UNK_ID
from neuronlp2.io.conllx_data import NUM_SYMBOLIC_TAGS
from neuronlp2.io.conllx_data import create_alphabets
from neuronlp2.io import utils
from neuronlp2.io.reader import CoNLLXReader


def _obtain_child_index_for_left2right(heads):
    child_ids = [[] for _ in range(len(heads))]
    # skip the symbolic root.
    for child in range(1, len(heads)):
        head = heads[child]
        child_ids[head].append(child)
    return child_ids


def _obtain_child_index_for_inside_out(heads):
    child_ids = [[] for _ in range(len(heads))]
    for head in range(len(heads)):
        # first find left children inside-out
        for child in reversed(range(1, head)):
            if heads[child] == head:
                child_ids[head].append(child)
        # second find right children inside-out
        for child in range(head + 1, len(heads)):
            if heads[child] == head:
                child_ids[head].append(child)
    return child_ids


def _obtain_child_index_for_depth(heads, reverse):
    def calc_depth(head):
        children = child_ids[head]
        max_depth = 0
        for child in children:
            depth = calc_depth(child)
            child_with_depth[head].append((child, depth))
            max_depth = max(max_depth, depth + 1)
        child_with_depth[head] = sorted(child_with_depth[head], key=lambda x: x[1], reverse=reverse)
        return max_depth

    child_ids = _obtain_child_index_for_left2right(heads)
    child_with_depth = [[] for _ in range(len(heads))]
    calc_depth(0)
    return [[child for child, depth in child_with_depth[head]] for head in range(len(heads))]


def _generate_stack_inputs(heads, types, prior_order):
    if prior_order == 'deep_first':
        child_ids = _obtain_child_index_for_depth(heads, True)
    elif prior_order == 'shallow_first':
        child_ids = _obtain_child_index_for_depth(heads, False)
    elif prior_order == 'left2right':
        child_ids = _obtain_child_index_for_left2right(heads)
    elif prior_order == 'inside_out':
        child_ids = _obtain_child_index_for_inside_out(heads)
    else:
        raise ValueError('Unknown prior order: %s' % prior_order)

    stacked_heads = []
    children = []
    siblings = []
    stacked_types = []
    skip_connect = []
    prev = [0 for _ in range(len(heads))]
    sibs = [0 for _ in range(len(heads))]
    stack = [0]
    position = 1
    while len(stack) > 0:
        head = stack[-1]
        stacked_heads.append(head)
        siblings.append(sibs[head])
        child_id = child_ids[head]
        skip_connect.append(prev[head])
        prev[head] = position
        if len(child_id) == 0:
            children.append(head)
            sibs[head] = 0
            stacked_types.append(PAD_ID_TAG)
            stack.pop()
        else:
            child = child_id.pop(0)
            children.append(child)
            sibs[head] = child
            stack.append(child)
            stacked_types.append(types[child])
        position += 1

    return stacked_heads, children, siblings, stacked_types, skip_connect


def read_data(source_path, word_alphabet, char_alphabet, pos_alphabet, type_alphabet,
              max_size=None, normalize_digits=True, prior_order='inside_out', device=torch.device('cpu')):
    data = []
    max_length = 0
    max_char_length = 0
    print('Reading data from %s' % source_path)
    counter = 0
    reader = CoNLLXReader(source_path, word_alphabet, char_alphabet, pos_alphabet, type_alphabet)
    inst = reader.getNext(normalize_digits=normalize_digits, symbolic_root=True, symbolic_end=False)
    while inst is not None and (not max_size or counter < max_size):
        counter += 1
        if counter % 10000 == 0:
            print("reading data: %d" % counter)

        sent = inst.sentence
        stacked_heads, children, siblings, stacked_types, skip_connect = _generate_stack_inputs(inst.heads, inst.type_ids, prior_order)
        data.append([sent.word_ids, sent.char_id_seqs, inst.pos_ids, inst.heads, inst.type_ids, stacked_heads, children, siblings, stacked_types, skip_connect])
        max_len = max([len(char_seq) for char_seq in sent.char_seqs])
        if max_char_length < max_len:
            max_char_length = max_len
        if max_length < inst.length():
            max_length = inst.length()
        inst = reader.getNext(normalize_digits=normalize_digits, symbolic_root=True, symbolic_end=False)
    reader.close()
    print("Total number of data: %d" % counter)

    data_size = len(data)
    char_length = min(utils.MAX_CHAR_LENGTH, max_char_length)
    wid_inputs = np.empty([data_size, max_length], dtype=np.int64)
    cid_inputs = np.empty([data_size, max_length, char_length], dtype=np.int64)
    pid_inputs = np.empty([data_size, max_length], dtype=np.int64)
    hid_inputs = np.empty([data_size, max_length], dtype=np.int64)
    tid_inputs = np.empty([data_size, max_length], dtype=np.int64)

    masks_e = np.zeros([data_size, max_length], dtype=np.float32)
    single = np.zeros([data_size, max_length], dtype=np.int64)
    lengths_e = np.empty(data_size, dtype=np.int64)

    stack_hid_inputs = np.empty([data_size, 2 * max_length - 1], dtype=np.int64)
    chid_inputs = np.empty([data_size, 2 * max_length - 1], dtype=np.int64)
    ssid_inputs = np.empty([data_size, 2 * max_length - 1], dtype=np.int64)
    stack_tid_inputs = np.empty([data_size, 2 * max_length - 1], dtype=np.int64)
    skip_connect_inputs = np.empty([data_size, 2 * max_length - 1], dtype=np.int64)

    masks_d = np.zeros([data_size, 2 * max_length - 1], dtype=np.float32)
    lengths_d = np.empty(data_size, dtype=np.int64)

    for i, inst in enumerate(data):
        wids, cid_seqs, pids, hids, tids, stack_hids, chids, ssids, stack_tids, skip_ids = inst
        inst_size = len(wids)
        lengths_e[i] = inst_size
        # word ids
        wid_inputs[i, :inst_size] = wids
        wid_inputs[i, inst_size:] = PAD_ID_WORD
        for c, cids in enumerate(cid_seqs):
            cid_inputs[i, c, :len(cids)] = cids
            cid_inputs[i, c, len(cids):] = PAD_ID_CHAR
        cid_inputs[i, inst_size:, :] = PAD_ID_CHAR
        # pos ids
        pid_inputs[i, :inst_size] = pids
        pid_inputs[i, inst_size:] = PAD_ID_TAG
        # type ids
        tid_inputs[i, :inst_size] = tids
        tid_inputs[i, inst_size:] = PAD_ID_TAG
        # heads
        hid_inputs[i, :inst_size] = hids
        hid_inputs[i, inst_size:] = PAD_ID_TAG
        # masks_e
        masks_e[i, :inst_size] = 1.0
        for j, wid in enumerate(wids):
            if word_alphabet.is_singleton(wid):
                single[i, j] = 1

        inst_size_decoder = 2 * inst_size - 1
        lengths_d[i] = inst_size_decoder
        # stacked heads
        stack_hid_inputs[i, :inst_size_decoder] = stack_hids
        stack_hid_inputs[i, inst_size_decoder:] = PAD_ID_TAG
        # children
        chid_inputs[i, :inst_size_decoder] = chids
        chid_inputs[i, inst_size_decoder:] = PAD_ID_TAG
        # siblings
        ssid_inputs[i, :inst_size_decoder] = ssids
        ssid_inputs[i, inst_size_decoder:] = PAD_ID_TAG
        # stacked types
        stack_tid_inputs[i, :inst_size_decoder] = stack_tids
        stack_tid_inputs[i, inst_size_decoder:] = PAD_ID_TAG
        # skip connects
        skip_connect_inputs[i, :inst_size_decoder] = skip_ids
        skip_connect_inputs[i, inst_size_decoder:] = PAD_ID_TAG
        # masks_d
        masks_d[i, :inst_size_decoder] = 1.0

    words = torch.from_numpy(wid_inputs).to(device)
    chars = torch.from_numpy(cid_inputs).to(device)
    pos = torch.from_numpy(pid_inputs).to(device)
    heads = torch.from_numpy(hid_inputs).to(device)
    types = torch.from_numpy(tid_inputs).to(device)
    masks_e = torch.from_numpy(masks_e).to(device)
    single = torch.from_numpy(single).to(device)
    lengths_e = torch.from_numpy(lengths_e).to(device)

    stacked_heads = torch.from_numpy(stack_hid_inputs).to(device)
    children = torch.from_numpy(chid_inputs).to(device)
    siblings = torch.from_numpy(ssid_inputs).to(device)
    stacked_types = torch.from_numpy(stack_tid_inputs).to(device)
    skip_connect = torch.from_numpy(skip_connect_inputs).to(device)
    masks_d = torch.from_numpy(masks_d).to(device)
    lengths_d = torch.from_numpy(lengths_d).to(device)

    data_tensor = {'WORD': words, 'CHAR': chars, 'POS': pos, 'HEAD': heads, 'TYPE': types, 'MASK_ENC': masks_e,
                   'SINGLE': single, 'LENGTH_ENC': lengths_e, 'STACK_HEAD': stacked_heads, 'CHILD': children,
                   'SIBLING': siblings, 'STACK_TYPE': stacked_types, 'SKIP_CONNECT': skip_connect, 'MASK_DEC': masks_d,
                   'LENGTH_DEC': lengths_d}
    return data_tensor, data_size


def read_bucketed_data(source_path, word_alphabet, char_alphabet, pos_alphabet, type_alphabet,
                       max_size=None, normalize_digits=True, prior_order='inside_out', device=torch.device('cpu')):
    data = [[] for _ in _buckets]
    max_char_length = [0 for _ in _buckets]
    print('Reading data from %s' % source_path)
    counter = 0
    reader = CoNLLXReader(source_path, word_alphabet, char_alphabet, pos_alphabet, type_alphabet)
    inst = reader.getNext(normalize_digits=normalize_digits, symbolic_root=True, symbolic_end=False)
    while inst is not None and (not max_size or counter < max_size):
        counter += 1
        if counter % 10000 == 0:
            print("reading data: %d" % counter)

        inst_size = inst.length()
        sent = inst.sentence
        for bucket_id, bucket_size in enumerate(_buckets):
            if inst_size < bucket_size:
                stacked_heads, children, siblings, stacked_types, skip_connect = _generate_stack_inputs(inst.heads, inst.type_ids, prior_order)
                data[bucket_id].append([sent.word_ids, sent.char_id_seqs, inst.pos_ids, inst.heads, inst.type_ids, stacked_heads, children, siblings, stacked_types, skip_connect])
                max_len = max([len(char_seq) for char_seq in sent.char_seqs])
                if max_char_length[bucket_id] < max_len:
                    max_char_length[bucket_id] = max_len
                break

        inst = reader.getNext(normalize_digits=normalize_digits, symbolic_root=True, symbolic_end=False)
    reader.close()
    print("Total number of data: %d" % counter)

    bucket_sizes = [len(data[b]) for b in range(len(_buckets))]
    data_tensors = []
    for bucket_id in range(len(_buckets)):
        bucket_size = bucket_sizes[bucket_id]
        if bucket_size == 0:
            data_tensors.append((1, 1))
            continue

        bucket_length = _buckets[bucket_id]
        char_length = min(utils.MAX_CHAR_LENGTH, max_char_length[bucket_id])
        wid_inputs = np.empty([bucket_size, bucket_length], dtype=np.int64)
        cid_inputs = np.empty([bucket_size, bucket_length, char_length], dtype=np.int64)
        pid_inputs = np.empty([bucket_size, bucket_length], dtype=np.int64)
        hid_inputs = np.empty([bucket_size, bucket_length], dtype=np.int64)
        tid_inputs = np.empty([bucket_size, bucket_length], dtype=np.int64)

        masks_e = np.zeros([bucket_size, bucket_length], dtype=np.float32)
        single = np.zeros([bucket_size, bucket_length], dtype=np.int64)
        lengths_e = np.empty(bucket_size, dtype=np.int64)

        stack_hid_inputs = np.empty([bucket_size, 2 * bucket_length - 1], dtype=np.int64)
        chid_inputs = np.empty([bucket_size, 2 * bucket_length - 1], dtype=np.int64)
        ssid_inputs = np.empty([bucket_size, 2 * bucket_length - 1], dtype=np.int64)
        stack_tid_inputs = np.empty([bucket_size, 2 * bucket_length - 1], dtype=np.int64)
        skip_connect_inputs = np.empty([bucket_size, 2 * bucket_length - 1], dtype=np.int64)

        masks_d = np.zeros([bucket_size, 2 * bucket_length - 1], dtype=np.float32)
        lengths_d = np.empty(bucket_size, dtype=np.int64)

        for i, inst in enumerate(data[bucket_id]):
            wids, cid_seqs, pids, hids, tids, stack_hids, chids, ssids, stack_tids, skip_ids = inst
            inst_size = len(wids)
            lengths_e[i] = inst_size
            # word ids
            wid_inputs[i, :inst_size] = wids
            wid_inputs[i, inst_size:] = PAD_ID_WORD
            for c, cids in enumerate(cid_seqs):
                cid_inputs[i, c, :len(cids)] = cids
                cid_inputs[i, c, len(cids):] = PAD_ID_CHAR
            cid_inputs[i, inst_size:, :] = PAD_ID_CHAR
            # pos ids
            pid_inputs[i, :inst_size] = pids
            pid_inputs[i, inst_size:] = PAD_ID_TAG
            # type ids
            tid_inputs[i, :inst_size] = tids
            tid_inputs[i, inst_size:] = PAD_ID_TAG
            # heads
            hid_inputs[i, :inst_size] = hids
            hid_inputs[i, inst_size:] = PAD_ID_TAG
            # masks_e
            masks_e[i, :inst_size] = 1.0
            for j, wid in enumerate(wids):
                if word_alphabet.is_singleton(wid):
                    single[i, j] = 1

            inst_size_decoder = 2 * inst_size - 1
            lengths_d[i] = inst_size_decoder
            # stacked heads
            stack_hid_inputs[i, :inst_size_decoder] = stack_hids
            stack_hid_inputs[i, inst_size_decoder:] = PAD_ID_TAG
            # children
            chid_inputs[i, :inst_size_decoder] = chids
            chid_inputs[i, inst_size_decoder:] = PAD_ID_TAG
            # siblings
            ssid_inputs[i, :inst_size_decoder] = ssids
            ssid_inputs[i, inst_size_decoder:] = PAD_ID_TAG
            # stacked types
            stack_tid_inputs[i, :inst_size_decoder] = stack_tids
            stack_tid_inputs[i, inst_size_decoder:] = PAD_ID_TAG
            # skip connects
            skip_connect_inputs[i, :inst_size_decoder] = skip_ids
            skip_connect_inputs[i, inst_size_decoder:] = PAD_ID_TAG
            # masks_d
            masks_d[i, :inst_size_decoder] = 1.0

        words = torch.from_numpy(wid_inputs).to(device)
        chars = torch.from_numpy(cid_inputs).to(device)
        pos = torch.from_numpy(pid_inputs).to(device)
        heads = torch.from_numpy(hid_inputs).to(device)
        types = torch.from_numpy(tid_inputs).to(device)
        masks_e = torch.from_numpy(masks_e).to(device)
        single = torch.from_numpy(single).to(device)
        lengths_e = torch.from_numpy(lengths_e).to(device)

        stacked_heads = torch.from_numpy(stack_hid_inputs).to(device)
        children = torch.from_numpy(chid_inputs).to(device)
        siblings = torch.from_numpy(ssid_inputs).to(device)
        stacked_types = torch.from_numpy(stack_tid_inputs).to(device)
        skip_connect = torch.from_numpy(skip_connect_inputs).to(device)
        masks_d = torch.from_numpy(masks_d).to(device)
        lengths_d = torch.from_numpy(lengths_d).to(device)

        data_tensor = {'WORD': words, 'CHAR': chars, 'POS': pos, 'HEAD': heads, 'TYPE': types, 'MASK_ENC': masks_e,
                       'SINGLE': single, 'LENGTH_ENC': lengths_e, 'STACK_HEAD': stacked_heads, 'CHILD': children,
                       'SIBLING': siblings, 'STACK_TYPE': stacked_types, 'SKIP_CONNECT': skip_connect, 'MASK_DEC': masks_d,
                       'LENGTH_DEC': lengths_d}
        data_tensors.append(data_tensor)

    return data_tensors, bucket_sizes


def iterate_batch(data, batch_size, bucketed=False, unk_replace=0., shuffle=False):
    if bucketed:
        return utils.iterate_bucketed_batch(data, batch_size, unk_replace==unk_replace, shuffle=shuffle)
    else:
        return utils.iterate_batch(data, batch_size, unk_replace==unk_replace, shuffle=shuffle)
