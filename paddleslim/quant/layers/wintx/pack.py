import paddle


def pack_col(src, tgt_bit=2):
    src_bitwidth=32
    bnt = 2**tgt_bit - 1
    if src.dtype != paddle.int32:
        src = src.cast('int32')

    row,col = src.shape
    pack_num = src_bitwidth // tgt_bit # there will be pack_num UINT2 value in one UINT32 value
    
    int_tgt = paddle.zeros((row, col// pack_num)).astype('int32')
    shift_bits = (paddle.arange(0,pack_num)*tgt_bit).cast('int32')
    
    src = src.reshape((row, col//pack_num, pack_num))
    src = src << shift_bits
    int_tgt = src.sum(axis=-1).cast('int32')
    return int_tgt


def unpack_col(src, tgt_bit=2):
    pack_num = 32 // tgt_bit # 4 UINT2 value in one UINT32 value
    bnt = 2**tgt_bit - 1
    row,col = src.shape
    # int_tgt = paddle.zeros((row,col* pack_num)).astype('uint32')
    int_tgt = src.reshape((row,col,1)).expand((row,col,pack_num))
    shift_bits = (paddle.arange(0,pack_num)*tgt_bit).cast('int32')
    
    int_tgt = int_tgt >> shift_bits & paddle.to_tensor(bnt).cast('int32')
    int_tgt = int_tgt.reshape((row,col*pack_num))
    # for pack in range(0, col):
    #     for i in range(pack_num):
    #         int_tgt[:,pack * pack_num + i] = src[:,pack] >> (i*2) & paddle.to_tensor(0x3).cast('uint32')
    return int_tgt.cast('float16')


def unpack_int4_super_scale(scale, super_scale):
    
    scale = unpack_col(scale.T, tgt_bit=4).T
    row,col = scale.shape
    with paddle.no_grad():

        group_nums, _ = scale.shape
        super_group_nums, _ = super_scale.shape
        scale = scale.T.reshape((-1, group_nums // super_group_nums)) * super_scale.T.reshape((-1,1))
        scale = scale.reshape((col,row)).T
    return scale