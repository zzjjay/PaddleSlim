from quant import *
from pack import *
from wintx.wintx_gemm import *
from wintx.wintx_gemv import *

def test_wint2(inp,w, dtype,asymm,group_size,matmul='gemm'):
    
    inp = inp.astype(dtype)
    w = w.astype(dtype)
    
    Qw,scale,_ = quantize2(w, asymm=False,group_size=group_size)
    
    # cast to uint8
    
    # Qw = Qw.cast('uint8')
    # zp = zp.cast('uint8')
    
    pack_w = pack_col(Qw.T)
    pack_w = pack_w.T
    
    pack_w = pack_w.contiguous()
    
    if matmul == 'gemm':
        out_triton = gemm_int2_paddle(inp, pack_w, scale)
    elif matmul == 'gemv':
        # import pdb;pdb.set_trace()
        out_triton = gemv_int2_paddle(inp, pack_w, scale)
    
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
    
    if matmul == 'gemm':
        out_triton = gemm_int4_paddle(inp, pack_w, scale)
    elif matmul == 'gemv':
        out_triton = gemv_int4_paddle(inp, pack_w, scale)
    
    QdQW1 = dequantize2(Qw,scale,None,bits=4)
    out_paddle = inp @ QdQW1
    
    loss = (out_triton-out_paddle).abs().max()
    out_fp = inp @ w    
    loss_fp = (out_fp-out_paddle).abs().max()
    return loss.cast('float32').tolist(),loss_fp.cast('float32').tolist(),out_triton.max().cast('float32').tolist()
    
    
if __name__ == "__main__":
    import paddle
    from wintx import weight_only_linear_int4_symm
    paddle.set_device('gpu:1')
    # test_config = {
    #     'dtype': ['float16','bfloat16'],
    #     'asymm': [False],
    #     'group_size': [32,64,128],
    #     'matmul': ['gemm']
    # }

    # inp,w = paddle.randn([64,7168])*0.125,paddle.randn([7168,4096])

    import pickle
    import os
    test_pth = '/root/paddlejob/workspace/env_run/output/lixunchao/'


    # import pdb;pdb.set_trace()
    for i in range(8):

        inp_pth = os.path.join(test_pth,f'test_input_{i:02d}.pkl')
        quant_weight_pth = os.path.join(test_pth,f'test_quant_weight_{i:02d}.pkl')
        quant_scale_pth = os.path.join(test_pth,f'test_quant_scale_{i:02d}.pkl')

        with open(inp_pth,'rb') as f:
            inp = pickle.load(f)
        with open(quant_weight_pth,'rb') as f:
            w = pickle.load(f)
        with open(quant_scale_pth,'rb') as f:
            scale = pickle.load(f)
        # inp,w = paddle.randn([2,128*1024]),paddle.randn([128*1024,4096])
        


        inp = paddle.to_tensor(inp).cast('bfloat16')
        w = paddle.to_tensor(w).cast('int32')
        scale = paddle.to_tensor(scale).cast('bfloat16')
        # print("input")
        # print(inp)
        # print("weight")
        # print(w)
        # print("scale")
        # print(scale)
        print(f'test tp{i:02d}')
        inp2 = inp[:,:,i*2048:(i+1)*2048].contiguous()
        # out = gemm_int4_paddle(inp,w,scale)
        out = weight_only_linear_int4_symm(inp2, w, None, scale)
        print("Is nan?")
        print(paddle.any(out.isnan()).tolist())

        if paddle.any(out.isnan()).tolist():

            unpack_w = unpack_col(w.T, tgt_bit=4).T.cast('float32')
            import pdb;pdb.set_trace()
            dw = dequantize(unpack_w, scale, bits=4)
            out2 = inp2 @ dw

            print("out1")
            print(out)
            print('out2')
            print(out2)



    