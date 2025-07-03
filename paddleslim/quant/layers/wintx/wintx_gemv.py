import paddle
import paddle.nn.functional as F

import triton.language as tl
import triton

import use_triton_in_paddle; use_triton_in_paddle.make_triton_compatible_with_paddle()

import custom_autotune

from .utils import *

__all__ = [
    'gemv_int2_paddle',
    'gemv_int4_paddle',
    'gemv_kernel'
]


def get_default_config():
    config = triton.Config({'BLOCK_SIZE_M':1, 'BLOCK_SIZE_N':128, 'BLOCK_SIZE_K':32},
                            num_warps=4, num_stages=1)
    return [config]

def get_wint2_kernel_config():
    configs = []
    for num_stages in [1,2]:
        for block_m in [1]:
            for block_n in [64,128,256,512]:#, 256, 512]:
                for block_k in [16,32,64,128,256]:#, 256, 512]: # block must larger than 16
                    for split_k in [1]:
                        for warps in [2,4]:
                            configs.append(
                                triton.Config(
                                {
                                    "BLOCK_SIZE_M": block_m,
                                    "BLOCK_SIZE_N": block_n,
                                    "BLOCK_SIZE_K": block_k,
                                    "GROUP_SIZE_M": 1,
                                    "num_stages": num_stages,
                                    "num_warps": warps,
                                    # "pre_hook": init_to_zero("c_ptr")
                                },
                                )
                            )
    return configs


@custom_autotune.autotune(
        configs=get_default_config(), #get_wint2_kernel_config(),
        key=["M", "N", "K"],
        nearest_power_of_two=True,
        prune_configs_by={
            "early_config_prune": custom_autotune.kernel_config_pruner,
            "perf_model": None,
            "top_k": None,
        },
        reset_to_zero=["c_ptr"]
    )
# @triton.autotune(
#     configs=get_default_config(),
#     key=["M", "N", "K"],
# )
@triton.jit
def gemv_kernel(
    a_ptr, b_ptr, c_ptr,
    bs_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    stride_bsk, stride_bsn,
    group_size: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr=1,
    n_bit:tl.constexpr=2, w_mask:tl.constexpr=0x3,
    # dot_prod_mode: tl.constexpr=1
):
    """
    """
    pid = tl.program_id(axis=0)

    repeat_chunks = (group_size-1) // BLOCK_SIZE_K + 1
    # tl.static_print(BLOCK_SIZE_K)
    pid_k = tl.program_id(axis=1)

    pack_num: tl.constexpr = 32 // n_bit
    # swizzle_tile, maybe work...
    pid_m, pid_n = linear_tile(pid, M, N, BLOCK_SIZE_M, BLOCK_SIZE_N, None)

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M

    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N

    offs_k = (pid_k * BLOCK_SIZE_K * repeat_chunks + tl.arange(0, BLOCK_SIZE_K)) % K

    bk_nums: tl.constexpr = int((BLOCK_SIZE_K-1) // pack_num + 1)
    offs_bk = (repeat_chunks * pid_k * bk_nums + tl.arange(0, bk_nums)) % K

    bzn_nums: tl.constexpr = int(BLOCK_SIZE_N-1) // pack_num + 1
    offs_bzn = (pid_n * bzn_nums + tl.arange(0,bzn_nums)) % (N // pack_num)

    group_nums: tl.constexpr = (BLOCK_SIZE_K-1) // group_size + 1
    offs_bzk = (pid_k * group_nums  + tl.arange(0, group_nums)) % (K // group_size)

    a_ptrs = a_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_bk[:, None] * stride_bk + offs_bn[None, :] * stride_bn

    # a_mask = offs_am[:,None] < M

    # if dot_prod_mode == 1:
    #     accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    # else:
    #     accumulator = tl.zeros((BLOCK_SIZE_K, BLOCK_SIZE_N), dtype=tl.float32)

    # get bit shifts
    b_shift_bits = (tl.arange(0, BLOCK_SIZE_K) % pack_num) * n_bit
    b_shift_bits = b_shift_bits[:,None]
    # bzp_shift_bits = (offs_bn[None, :] % pack_num) * w_mask

    #### load bzp,bs
    # zero-points ptr
    # bzp_ptrs = bzp_ptr + offs_bzk[:, None] * stride_bzpk \
    #     + offs_bzn[None, :] * stride_bzpn
    # bzp_ptrs = bzp_ptr + offs_bzk[:, None] * stride_bzpk \
    #     + offs_bzn[None, :] * stride_bzpn

    # bzp = tl.load(bzp_ptrs)
    bzp = 1 << (n_bit- 1)
    # scales ptr
    bs_ptrs = bs_ptr + offs_bzk[:, None] * stride_bsk \
                + offs_bn[None, :] * stride_bsn
    bs = tl.load(bs_ptrs)

    # unpack zero-points and scales
    # bzp,bs = unpack_zp_bs(bzp,bs,
    #         BLOCK_SIZE_N,BLOCK_SIZE_K,
    #         pack_num, group_nums)

    bs = unpack_bs(bs,
                    BLOCK_SIZE_N,BLOCK_SIZE_K,
                    group_nums)
    # bzp = (bzp >> bzp_shift_bits) & 0x3



    # Load B
    b = tl.load(b_ptrs)

    # Load A
    a = tl.load(a_ptrs)

    # dequant
    b = dequant(b, bs, bzp, b_shift_bits,
        BLOCK_SIZE_N,BLOCK_SIZE_K,
        pack_num,w_mask)

    a = a.reshape(BLOCK_SIZE_K,1)

    # if dot_prod_mode == 1:
    accumulator = tl.sum(a * b.to(a.dtype), axis=0, keep_dims=True)
    # elif dot_prod_mode == 0:
    #     accumulator = a * b.to(a.dtype)

    # tl.static_print(accumulator.shape)

    # next chunk
    for chunks_idx in tl.range(repeat_chunks-1):
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk // pack_num


        # Load B
        b = tl.load(b_ptrs)
        # Load A
        a = tl.load(a_ptrs)
        # dequant
        b = dequant(b, bs, bzp, b_shift_bits,
            BLOCK_SIZE_N,BLOCK_SIZE_K,
            pack_num,w_mask)


        # tl.static_print(b.shape)
        a = a.reshape(BLOCK_SIZE_K,1)
        # accumulator += tl.sum(a * b.to(a.dtype), axis=0, keep_dims=True)
        # if dot_prod_mode == 1:
        accumulator += tl.sum(a * b.to(a.dtype), axis=0, keep_dims=True)
        # elif dot_prod_mode == 0:
        #     accumulator += a * b.to(a.dtype)



    # if dot_prod_mode == 0:
    #     accumulator = tl.sum(accumulator, axis=0, keep_dims=True)
    # You can fuse arbitrary activation functions here
    # while the accumulator is still in FP32!
    c = accumulator.to(c_ptr.dtype.element_ty)


    # -----------------------------------------------------------
    # Write back the block of the output matrix C with masks.
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    # offs_cn = vectorize_load(offs_cn, BLOCK_SIZE_N)
    # offs_cm = vectorize_load(offs_cm, BLOCK_SIZE_M)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)

    tl.atomic_add(c_ptrs, c, mask=c_mask)


def gemv_int2_paddle(x, qw, scales, group_size=None,output=None):
    assert x.is_contiguous(), "A must be contiguous"
    assert qw.is_contiguous(), "B must be contiguous"

    M,K = x.shape
    N = qw.shape[1]

    dtype = x.dtype

    if group_size is None:
        group_size = K // scales.shape[0]

    if output is None:
        output = paddle.zeros([M,N], dtype='float32')
        # output = paddle.empty([M,N], dtype=x.dtype)

    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_SIZE_M']) * triton.cdiv(N, meta['BLOCK_SIZE_N']),
                         triton.cdiv(K,  group_size))

    # import pdb;pdb.set_trace()
    x_stride0,x_stride1 = x.shape[1], 1
    qw_stride0,qw_stride1 = qw.shape[1], 1
    scales_stride0,scales_stride1 = scales.shape[1], 1
    output_stride0,output_stride1 = output.shape[1], 1

    # print(output.dtype)
    gemv_kernel[grid](
        x, qw, output,
        scales,
        M, N, K,
        x_stride0, x_stride1,
        qw_stride0, qw_stride1,
        output_stride0, output_stride1,
        scales_stride0, scales_stride1,
        group_size=group_size,
        n_bit=2,w_mask=0x3,
        # BLOCK_SIZE_M=128, BLOCK_SIZE_N=16, BLOCK_SIZE_K=32,
        # GROUP_SIZE_M=8, SPLIT_K=1
    )

    # if dtype is paddle.bfloat16:
    output = output.cast(dtype)
    return output


def gemv_int4_paddle(x, qw, scales, group_size=None,output=None):
    assert x.is_contiguous(), "A must be contiguous"
    assert qw.is_contiguous(), "B must be contiguous"

    M,K = x.shape
    N = qw.shape[1]

    dtype = x.dtype

    if group_size is None:
        group_size = K // scales.shape[0]

    if output is None:
        output = paddle.zeros([M,N], dtype='float32')
        # output = paddle.empty([M,N], dtype=x.dtype)

    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_SIZE_M']) * triton.cdiv(N, meta['BLOCK_SIZE_N']),
                         triton.cdiv(K,  group_size))

    # import pdb;pdb.set_trace()
    x_stride0,x_stride1 = x.shape[1], 1
    qw_stride0,qw_stride1 = qw.shape[1], 1
    scales_stride0,scales_stride1 = scales.shape[1], 1
    output_stride0,output_stride1 = output.shape[1], 1

    # print(output.dtype)
    gemv_kernel[grid](
        x, qw, output,
        scales,
        M, N, K,
        x_stride0, x_stride1,
        qw_stride0, qw_stride1,
        output_stride0, output_stride1,
        scales_stride0, scales_stride1,
        group_size=group_size,
        n_bit=4,w_mask=0xF,
        # BLOCK_SIZE_M=128, BLOCK_SIZE_N=16, BLOCK_SIZE_K=32,
        # GROUP_SIZE_M=8, SPLIT_K=1
    )

    # if dtype is paddle.bfloat16:
    output = output.cast(dtype)
    return output



if __name__ == '__main__':
    import os
    os.environ['TRITON_PRINT_AUTOTUNING'] = '1'
    from quant import *
    from pack import *
    import timeit

    from eval_time import eval_time_paddle
    paddle.set_device('gpu:2')
    # inp,w = paddle.randn([4096,4096]).cast('float16'),paddle.randn([4096,1024]).cast('float16')
    inp,w = paddle.randn([1,7168]).cast('float16'),paddle.randn([7168,4096]).cast('float16')
    print(f'weight dtype:{w.dtype}')
    Qw,scale,_ = quantize2(w, asymm=False,group_size=32)
    # zp = zp.cast('uint8')

    pack_w = pack_col(Qw.T)
    pack_w = pack_w.T
    # pack_zp = pack_col(zp)

    pack_w = pack_w.contiguous()


    latency,_,_ = eval_time_paddle(lambda:gemv_int2_paddle(inp, pack_w, scale))
    pp_latency,_,_ = eval_time_paddle(lambda:inp @ dequantize2(unpack_col(pack_w.T).T.cast(w.dtype), scale))
    pp_fp_latency,_,_ = eval_time_paddle(lambda: inp @ w)
    print(f'triton int2 gemm:\t{latency}ms\npaddle int2 gemm:\t{pp_latency}ms\nfp gemm:\t{pp_fp_latency}ms')


    Qw,scale,_ = quantize2(w, asymm=False,group_size=32, bits=4)
    # zp = zp.cast('uint8')
    pack_w = pack_col(Qw.T,tgt_bit=4)
    pack_w = pack_w.T
    pack_w = pack_w.contiguous()
    latency,_,_ = eval_time_paddle(lambda:gemv_int4_paddle(inp, pack_w, scale))
    pp_latency,_,_ = eval_time_paddle(lambda:inp @ dequantize2(unpack_col(pack_w.T,tgt_bit=4).T.cast(w.dtype), scale))
    pp_fp_latency,_,_ = eval_time_paddle(lambda: inp @ w)
    print(f'triton int2 gemm:\t{latency}ms\npaddle int2 gemm:\t{pp_latency}ms\nfp gemm:\t{pp_fp_latency}ms')