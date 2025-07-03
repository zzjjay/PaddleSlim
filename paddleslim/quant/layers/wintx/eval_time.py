import random
import numpy as np
import paddle

def eval_time_paddle(fct, rep=50, return_mode='mean'):
    # Follow https://github.com/mobiusml/gemlite/blob/master/examples/benchmark_triton.py
    cache = paddle.empty((int(256e6 // 4),), dtype='int32')

    with paddle.no_grad():
        t = []
        for _ in range(rep):
            start_event = paddle.device.Event(enable_timing=True)
            end_event = paddle.device.Event(enable_timing=True)
            # start_event = torch.cuda.Event(enable_timing=True)
            # end_event = torch.cuda.Event(enable_timing=True)

            cache.zero_() #fast_flush
            start_event.record()
            fct()
            end_event.record()
            paddle.device.synchronize()
            t.append(start_event.elapsed_time(end_event))
            cache += int(random.random()*1000)  #change cache
    del cache

    return np.quantile(t, 0.5), np.quantile(t, 0.2), np.quantile(t, 0.8)
    # return np.min(t), np.mean(t), np.max(t)