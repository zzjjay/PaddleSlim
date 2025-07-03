# 

#OMP_NUM_THREADS=16 TRITON_PRINT_AUTOTUNING=1 CUDA_VISIBLE_DEVICES=0 ipython3 benchmark_triton.py #select the right number of threads based on your machine
#################################################################################################################################
import torch
import numpy as np


device = 'gpu:2'

in_features, out_features = 4096, 4096
#in_features, out_features = 4096*2, 4096*2
#in_features, out_features = 4096*4, 4096*4 
#in_features, out_features = 4096*8, 4096*8 

#W_nbits, group_size = 8, in_features 
W_nbits, group_size = 4, 128 
#W_nbits, group_size = 2, 128
import paddle


import random
import numpy as np
def eval_time(fct, params, rep=1000, return_mode='min'):
    # Follow https://github.com/mobiusml/gemlite/blob/master/examples/benchmark_triton.py
    cache = torch.empty(int(256 * 1024 * 1024 // 4), dtype=torch.int, device='cuda')

    t = []
    for _ in range(rep):
        start_event = paddle.device.cuda.Event(enable_timing=True)
        end_event = paddle.device.cuda.Event(enable_timing=True)
        # start_event = torch.cuda.Event(enable_timing=True)
        # end_event = torch.cuda.Event(enable_timing=True)

        cache.zero_() #fast_flush
        start_event.record()
        fct(**params)
        end_event.record()
        paddle.device.cuda.synchronize()
        t.append(start_event.elapsed_time(end_event))
        cache += int(random.random()*1000)  #change cache

    return np.min(t) if return_mode=='min' else np.mean(t[rep//2:])