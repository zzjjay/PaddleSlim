import paddle
from wint2.wint2_gemm import *
from wint2.quant import *
from wint2.pack import *
import triton

paddle.set_device('gpu:3')


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['M'],  # argument names to use as an x-axis for the plot
        x_vals=[1024*i for i in range(1,16,2)],  # different possible values for `x_name`
        line_arg='provider',  # argument name whose value corresponds to a different line in the plot
        line_vals=['paddle_triton', 
                #    'torch_triton',
                #    'gemlite', 
                   'fp_paddle',
                   'fp_unpack_paddle'
                   ],  # possible values for `line_arg``
        line_names=[
                    'paddle_triton', 
                    # 'torch_triton',
                #    'gemlite', 
                   'fp_paddle',
                   'fp_unpack_paddle'
        ],  # label name for the lines
        # styles=[('blue', '-'), ('green', '-')],  # line styles
        ylabel="ms",  # label name for the y-axis
        plot_name="wint2 gemm",  # name for the plot. Used also as a file name for saving the plot.
        args={'N':256, 'K':7168, 'dtype': 'float16', 'asymm': True},  # values for function arguments not in `x_names` and `y_name`
    ))
def benchmark(provider, M, N, K, dtype,asymm, warmup=16):

    inp_paddle = paddle.randn((M,K)).astype(dtype).contiguous()
    w_paddle = paddle.randn((K,N)).astype(dtype).contiguous()

        
    if provider in ['paddle_triton','fp_unpack_paddle']:

        with paddle.no_grad():
            
            W_q, scales, zeros = quantize(w_paddle,asymm=True)
            pack_w = pack_col(W_q.T).T.contiguous()
            pack_zp = pack_col(zeros)
            
        

    
    ############## calculate ###############
    if provider == 'paddle_triton':
        
        for _ in range(warmup):
            gemm_int2_paddle(inp_paddle, pack_w, scales, pack_zp, group_size=32)
        # ms = triton.testing.do_bench(lambda: matmul_dequantize_int2_v2(inp, pack_w, scale, pack_zp))
        ms = triton.testing.do_bench(lambda: gemm_int2_paddle(inp_paddle, pack_w, scales, pack_zp, group_size=128),warmup=1000, rep=1000,)
        del inp_paddle, w_paddle, W_q, scales, zeros

        
    if provider == 'fp_paddle':
        
        for _ in range(warmup):
            inp_paddle @ dequantize(*quantize(w_paddle,asymm=True))
        ms = triton.testing.do_bench(lambda: inp_paddle @ dequantize(*quantize(w_paddle,asymm=True)),warmup=1000, rep=1000,)
    
    if provider == 'fp_unpack_paddle':
        
        for _ in range(warmup):
            inp_paddle @ dequantize(unpack_col(pack_w.T).T.cast(w_paddle.dtype), scales, unpack_col(pack_zp).cast(w_paddle.dtype))
        ms = triton.testing.do_bench(lambda: inp_paddle @ dequantize(unpack_col(pack_w.T).T.cast(w_paddle.dtype), scales, unpack_col(pack_zp).cast(w_paddle.dtype)),warmup=100, rep=1000,)

    return ms
benchmark.run(show_plots=False, print_data=True)