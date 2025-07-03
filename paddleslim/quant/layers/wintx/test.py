from quant import *
from pack import *
from wintx.wintx_gemm import *
from wintx.wintx_gemv import *

from wintx import *

def test_wint2(inp,w, dtype,asymm,group_size,matmul='gemm'):

    inp = inp.astype(dtype)
    w = w.astype(dtype)

    Qw,scale,_ = quantize2(w, asymm=False,group_size=group_size)


    pack_w = pack_col(Qw.T)
    pack_w = pack_w.T


    pack_w = pack_w.contiguous()

    # if matmul == 'gemm':
    #     out_triton = gemm_int2_paddle(inp, pack_w, scale)
    # elif matmul == 'gemv':
    #     # import pdb;pdb.set_trace()
    #     out_triton = gemv_int2_paddle(inp, pack_w, scale)

    out_triton = weight_only_linear_int2_symm(inp, pack_w, None, scale, matmul)

    QdQW1 = dequantize2(Qw,scale,None)
    out_paddle = inp @ QdQW1

    loss = (out_triton-out_paddle).abs().max()
    out_fp = inp @ w
    loss_fp = (out_fp-out_paddle).abs().max()
    return loss.cast('float32').tolist(),loss_fp.cast('float32').tolist(),out_triton.max().cast('float32').tolist()


def test_wint4(inp,w, dtype,asymm,group_size,matmul='gemm'):

    inp = inp.astype(dtype)
    w = w.astype(dtype)

    Qw,scale,_ = quantize2(w, asymm=asymm,group_size=group_size,bits=4)

    # cast to uint8

    # Qw = Qw.cast('uint8')
    # zp = zp.cast('uint8')

    pack_w = pack_col(Qw.T, tgt_bit=4)
    pack_w = pack_w.T

    pack_w = pack_w.contiguous()

    # if matmul == 'gemm':
    #     out_triton = gemm_int4_paddle(inp, pack_w, scale)
    # elif matmul == 'gemv':
    #     out_triton = gemv_int4_paddle(inp, pack_w, scale)

    out_triton = weight_only_linear_int4_symm(inp, pack_w, None, scale, matmul)

    QdQW1 = dequantize2(Qw,scale,None,bits=4)
    out_paddle = inp @ QdQW1

    loss = (out_triton-out_paddle).abs().max()
    out_fp = inp @ w
    loss_fp = (out_fp-out_paddle).abs().max()
    return loss.cast('float32').tolist(),loss_fp.cast('float32').tolist(),out_triton.max().cast('float32').tolist()


if __name__ == "__main__":
    import paddle
    paddle.set_device('gpu:1')
    test_config = {
        'dtype': ['float16','bfloat16'],
        'asymm': [False],
        'group_size': [32,64,128],
        'matmul': ['gemv','gemm','gemm_splitk']
    }

    inp,w = paddle.randn([64,7168]),paddle.randn([7168,4096])
    # inp,w = paddle.randn([2,128*1024]),paddle.randn([128*1024,4096])

    print('***INFO***')
    print(f'input shape:{inp.shape}, weight shape:{w.shape}')
    print("***TEST***")
    for group_size in test_config['group_size']:
        print(f'---group_size:{group_size}---')
        for matmul in test_config['matmul']:
            print(f'---matmul:{matmul}---')
            for dtype in test_config['dtype']:
                for asymm in test_config['asymm']:
                        print("***Testing WINT2****")
                        loss,loss_fp, max_v = test_wint2(inp,w,dtype,asymm,group_size,matmul=matmul)
                        print(f'dtype:{dtype}\tasymm:{asymm}\t\tdiff:{loss:.2f}\tloss_fp:{loss_fp:.2f}\tmax_v:{max_v:.2f}')

                        print("***Testing WINT4****")
                        loss,loss_fp, max_v = test_wint4(inp,w,dtype,asymm,group_size,matmul=matmul)
                        print(f'dtype:{dtype}\tasymm:{asymm}\t\tdiff:{loss:.2f}\tloss_fp:{loss_fp:.2f}\tmax_v:{max_v:.2f}')