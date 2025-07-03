import use_triton_in_paddle; use_triton_in_paddle.make_triton_compatible_with_paddle()

from .wintx_gemm import *
from .wintx_gemv import *
from .wintx_gemm_splitk import *

from .pack import *


MAX_TRITON_M = 4096

WINT2_GEMM_MAP = {
    'gemm': gemm_int2_paddle,
    'gemv': gemv_int2_paddle,
    'gemm_splitk': gemm_int2_paddle_splitk,
    'unpack': gemm_unpack_int2_paddle
}

WINT4_GEMM_MAP = {
    'gemm': gemm_int4_paddle,
    'gemv': gemv_int4_paddle,
    'gemm_splitk': gemm_int4_paddle_splitk,
    'unpack': gemm_unpack_int4_paddle
}
def weight_only_linear_int2_symm(x, pack_w, bias, scale,matmul_type='auto'):
    H = x.shape[-1]
    other_shape = x.shape[:-1]
    x = x.reshape((-1,H))#.contiguous()

    S = x.shape[0]
    # print(f"S={S}, H={H}, matmul_type={matmul_type}")
    if matmul_type == 'auto':
        if S > MAX_TRITON_M:
            matmul_type = 'unpack'
        elif S > 64:
            matmul_type = 'gemm'
        elif S > 2:
            matmul_type = 'gemm_splitk'
        else:
            matmul_type = 'gemv'

    gemm = WINT2_GEMM_MAP.get(matmul_type, None)
    assert gemm is not None, f"{matmul_type} is not supported"

    if bias is None:
        out = gemm(x, pack_w, scale)
    else:
        out = gemm(x, pack_w, scale) + bias
    return out.reshape(other_shape+out.shape[-1:])

def weight_only_linear_int4_symm(x, pack_w, bias, scale,matmul_type='auto'):
    H = x.shape[-1]
    other_shape = x.shape[:-1]
    x = x.reshape((-1,H))#.contiguous()

    S = x.shape[0]
    # print(f"S={S}, H={H}, matmul_type={matmul_type}")
    if matmul_type == 'auto':
        if S > MAX_TRITON_M:
            matmul_type = 'unpack'
        elif S > 64:
            matmul_type = 'gemm'
        elif S > 2:
            matmul_type = 'gemm_splitk'
        else:
            matmul_type = 'gemv'

    gemm = WINT4_GEMM_MAP.get(matmul_type, None)
    assert gemm is not None, f"{matmul_type} is not supported"

    if bias is None:
        out = gemm(x, pack_w, scale)
    else:
        out = gemm(x, pack_w, scale) + bias
    return out.reshape(other_shape+out.shape[-1:])



