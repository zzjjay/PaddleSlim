import paddle
import paddle.nn.functional as F
from pack import pack_col

def quantize2(W, asymm=False, group_size=128,bits = 2):
    
    W = W.T
    row,col = W.shape # [Cout, Cin]
    W = W.reshape((-1,group_size))
    dtype = W.dtype
    W = W.cast("float32")
    with paddle.no_grad():
        if asymm:
            bnt = 2**bits - 1
            
            w_max =  F.relu(W).max(-1,keepdim=True)
            w_min = -F.relu(-W).max(-1,keepdim=True)
            w_range = w_max - w_min
            scale = w_range / bnt
            zp = (-w_min / scale).round().clip(0, bnt)
            qw = (W / scale + zp).round().clip(0, bnt)
            del w_range, w_max, w_min
            zp = zp.reshape((row,-1)).cast(dtype).T.contiguous()
            
        else:
            bnt = 2**(bits-1) - 1
            quant_scale = paddle.mean(paddle.abs(W.cast("float32")), axis=-1,keepdim=True)
            quant_scale = paddle.where(quant_scale == paddle.to_tensor( 0, dtype=W.dtype),paddle.to_tensor(1e-8, dtype=W.dtype),quant_scale)
            scale = quant_scale/bnt
            qw = paddle.clip(
                            paddle.round(W.cast("float32") / scale),
                            -bnt-1, bnt)
            qw = qw + bnt + 1
            zp = None
            del quant_scale
            
        qw = qw.reshape((row, col)).cast(dtype).T.contiguous()
        scale = scale.reshape((row,-1)).cast(dtype).T.contiguous()
        
    return qw, scale, zp 

def dequantize2(qx, scale, zp=None,bits=2):
    qx = qx.T
    row,col = qx.shape[0], qx.shape[1]
    group_size = col//scale.shape[0]
    with paddle.no_grad():
        qx = qx.reshape((-1,group_size))
        scale = scale.T.reshape((-1,1))
        
        if zp is None:
            zp = 2**(bits-1)
        else:
            zp = zp.T.reshape((-1,1))
        out=(qx - zp) * scale
        
        out = out.reshape((row,col)).T
    return out
    
def quantize3(W, asymm=False, group_size=128):
    bits = 2
    
    W = W.T
    row,col = W.shape # [Cout, Cin]
    W = W.reshape((-1,group_size))
    dtype = W.dtype
    W = W.cast("float32")
    if asymm:
        bnt = 2**bits - 1
        w_max =  F.relu(W).max(-1,keepdim=True)
        w_min = -F.relu(-W).max(-1,keepdim=True)
        w_range = w_max - w_min
        scale = w_range / bnt
        zp = (-w_min / scale).round().clip(0, bnt)
        qw0 = W / scale
        qw0 = qw0.abs().pow(0.5)*paddle.sign(qw0)
        qw = (qw0.clip(-1,1) + zp).round().clip(0, bnt)
        new_scale = (W).abs().sum(-1,keepdim=True) / (qw-zp).abs().sum(-1,keepdim=True)
        scale = new_scale
        
    else:
        bnt = 2**(bits-1) - 1
        quant_scale = paddle.max(paddle.abs(W), axis=-1,keepdim=True)
        # w_max =  F.relu(W).max(-1,keepdim=True)
        # w_min = -F.relu(-W).max(-1,keepdim=True)
        quant_scale = paddle.where(quant_scale == paddle.to_tensor( 0, dtype=W.dtype),paddle.to_tensor(1e-8, dtype=W.dtype),quant_scale)
        scale = quant_scale/bnt
        qw0 = W.cast("float32") / scale
        # qw0 = qw0.abs().pow(0.98) * paddle.sign(qw0)
        qw0 = qw0.abs().pow(0.5)*paddle.sign(qw0)
        qw = paddle.clip(
                        paddle.round(qw0),
                        -bnt, bnt)
        qw = qw + bnt
        zp = paddle.ones_like(quant_scale) * bnt
        
        new_scale = (W).abs().sum(-1,keepdim=True) / (qw-bnt).abs().sum(-1,keepdim=True)
        scale = new_scale
        
    qw = qw.reshape((row, col)).cast(dtype).T.contiguous()
    scale = scale.reshape((row,-1)).cast(dtype).T.contiguous()
    zp = zp.reshape((row,-1)).cast(dtype).T.contiguous()
    return qw, scale, zp 

def dequantize3(qx, scale, zp):
    qx = qx.T
    row,col = qx.shape[0], qx.shape[1]
    group_size = col//scale.shape[0]
    
    # qx = qx.reshape((-1,group_size,col))
    qx = qx.reshape((-1,group_size))
    scale = scale.T.reshape((-1,1))
    zp = zp.T.reshape((-1,1))
    
    if zp is None:
        out= qx*scale
    else:
        out=(qx - zp) * scale
        
    out = out.reshape((row,col)).T
    return out


quantize = quantize2
dequantize = dequantize2
def wint2_quant_pack(W, asymm=True, group_size=64):

    Qw,scale,zp = quantize(W, asymm=asymm,group_size=group_size)
    # cast to uint8
    Qw = Qw.cast('uint8')
    zp = zp.cast('uint8')
    pack_w = pack_col(Qw.T)
    pack_w = pack_w.T
    pack_zp = pack_col(zp)
    pack_w = pack_w.contiguous()
    return pack_w, scale, pack_zp
