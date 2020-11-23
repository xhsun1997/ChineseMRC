import os
import collections
import re
import torch
from glob import glob


def torch_show_all_params(model, rank=0):
    params = list(model.parameters())
    k = 0
    for i in params:
        l = 1
        for j in i.size():
            l *= j
        k = k + l
    if rank == 0:
        print("Total param num：" + str(k))


def torch_init_model(model, init_checkpoint):
    state_dict = torch.load(init_checkpoint, map_location='cpu')
    missing_keys = []
    unexpected_keys = []
    error_msgs = []
    # copy state_dict so _load_from_state_dict can modify it
    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=''):
        local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})

        module._load_from_state_dict(
            state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    load(model, prefix='' if hasattr(model, 'bert') else 'bert.')

    print("missing keys:{}".format(missing_keys))
    print('unexpected keys:{}'.format(unexpected_keys))
    print('error msgs:{}'.format(error_msgs))


def torch_save_model(model, output_dir, scores, max_save_num=1):
    # Save model checkpoint
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
    saved_pths = glob(os.path.join(output_dir, '*.pth'))
    saved_pths.sort()
    while len(saved_pths) >= max_save_num:
        if os.path.exists(saved_pths[0].replace('//', '/')):
            os.remove(saved_pths[0].replace('//', '/'))
            del saved_pths[0]

    save_prex = "checkpoint_score"
    for k in scores:
        save_prex += ('_' + k + '-' + str(scores[k])[:6])
    save_prex += '.pth'

    torch.save(model_to_save.state_dict(),
               os.path.join(output_dir, save_prex))
    print("Saving model checkpoint to %s", output_dir)
