import paddle
import paddle.nn.functional as F

import triton.language as tl
import triton



@triton.jit
def unpack_zp_bs(bzp,bs,
            BLOCK_SIZE_N,BLOCK_SIZE_K,
            pack_num, group_nums):
    
    bzp = bzp.permute(1,0)
    bzp = bzp.expand_dims(axis=-1).expand_dims(axis=0)
    bzp = bzp.broadcast_to(pack_num, BLOCK_SIZE_N//pack_num, group_nums, BLOCK_SIZE_K // group_nums)
    bzp = bzp.permute(1,0,2,3).reshape(BLOCK_SIZE_N,BLOCK_SIZE_K).permute(1,0)

    
    
    bs = bs.permute(1,0)
    bs = bs.expand_dims(axis=-1)
    bs = bs.broadcast_to(BLOCK_SIZE_N, group_nums, BLOCK_SIZE_K // group_nums).reshape(BLOCK_SIZE_N, BLOCK_SIZE_K)
    bs = bs.permute(1,0)
    
    return bzp, bs

@triton.jit
def dequant(b, bs, bzp, b_shift_bits,
            BLOCK_SIZE_N,BLOCK_SIZE_K,
            pack_num,w_mask):
    
    b = b.permute(1,0)
    b = b.expand_dims(axis=-1)
    b = b.broadcast_to(BLOCK_SIZE_N,BLOCK_SIZE_K//pack_num, pack_num).reshape(BLOCK_SIZE_N,BLOCK_SIZE_K)
    b = b.permute(1,0)
    # b = b.permute(1,0)
    # b = tl.interleave(b, b)
    # b = tl.interleave(b, b)
    # b = tl.interleave(b, b)
    # b = tl.interleave(b, b)
    # b = b.permute(1,0)
    
    
    int_b = (b >> b_shift_bits) & w_mask
    # int_bzp = (bzp >> bzp_shift_bits) & 0x3
    b = (int_b - bzp) * bs #.to(a.dtype)
    return b

@triton.jit
def unpack_bs(bs,
            BLOCK_SIZE_N,BLOCK_SIZE_K,
            group_nums):
    bs = bs.permute(1,0)
    bs = bs.expand_dims(axis=-1)
    bs = bs.broadcast_to(BLOCK_SIZE_N, group_nums, BLOCK_SIZE_K // group_nums).reshape(BLOCK_SIZE_N, BLOCK_SIZE_K)
    bs = bs.permute(1,0)
    
    return bs

@triton.jit
def swizzle_tile(pid, M, N, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, GROUP_SIZE_M: tl.constexpr):
    grid_m     = tl.cdiv(M, BLOCK_SIZE_M)
    grid_n     = tl.cdiv(N, BLOCK_SIZE_N)
    width      = GROUP_SIZE_M * grid_n
    group_id   = pid // width
    group_size = tl.minimum(grid_m - group_id * GROUP_SIZE_M, GROUP_SIZE_M)
    pid_m      = group_id * GROUP_SIZE_M + (pid % group_size)
    pid_n      = (pid % width) // group_size
    return pid_m, pid_n

def init_to_zero(name):
    return lambda nargs: nargs[name].zero_()


def init_to_zero(name):
    return lambda nargs: nargs[name].zero_()

@triton.jit
def linear_tile(pid, M, N, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, GROUP_SIZE_M: tl.constexpr):
    pid_m = pid // tl.cdiv(N, BLOCK_SIZE_N)
    pid_n = pid % tl.cdiv(N, BLOCK_SIZE_N)
    return pid_m, pid_n

@triton.jit
def vectorize_load(offset, block_size):
    return tl.max_contiguous(tl.multiple_of(offset, block_size), block_size) 