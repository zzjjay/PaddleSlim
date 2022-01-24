import os
import sys
import numpy as np
import paddle
import paddle.fluid as fluid
from paddleslim.core import GraphWrapper
sys.path.append(os.path.join(os.path.dirname("__file__"), os.path.pardir))
import models
import copy
paddle.enable_static()
# 

import paddle
import paddle.static as static
import numpy as np

paddle.enable_static()


image = paddle.static.data(
        name='image', shape=[None] + [3,224,224], dtype='float32')
label = paddle.static.data(name='label', shape=[None, 1], dtype='int64')
z= paddle.static.data(name='z', shape=[None] + [3,224,224], dtype='float32')
model = models.__dict__['MobileNet']()
out = model.net(input=image, class_dim=1000)

places = paddle.static.cpu_places()
place = places[0]
exe = paddle.static.Executor(place)
train_program = paddle.static.default_main_program()

exe.run(paddle.static.default_startup_program())
for param in train_program.list_vars():
    if param.name.split('_')[-1] == 'weights':
        # z = param.detach().clone()
        if len(param.shape) == 4 and param.shape[2] == 1 and param.shape[3] == 1:
        # new_variable = cur_block.create_var(name="X",
        #                             shape=[32, 3, 3, 3],
        #                             dtype='float32')
            # z = exe.run(program=train_program, feed={"image": data}, fetch_list=[param.name], return_numpy=False)[0]
            # Z += (z,)
            # print(z)
            mask = train_program.global_block().create_var(
                name=param.name + "_mask",
                shape=param.shape,
                dtype=param.dtype,
                type=param.type,
                persistable=param.persistable,
                stop_gradient=True)
            paddle.static.global_scope().var(param.name + "_mask").get_tensor().set(
                np.ones(param.shape).astype("float32"), place)

data = np.random.random(size=(1,3,224,224)).astype('float32')
data2 = np.random.random(size=(1,1)).astype('float32')
exe.run(train_program, feed={'image':data}, fetch_list=[out, ])

Z = ()
U = ()
scope = paddle.static.global_scope()
for param in train_program.list_vars():
    if param.name.split('_')[-1] == 'weights':
        # z = param.detach().clone()
        if len(param.shape) == 4 and param.shape[2] == 1 and param.shape[3] == 1:
            print(param)
        # new_variable = cur_block.create_var(name="X",
        #                             shape=[32, 3, 3, 3],
        #                             dtype='float32')
            # z = exe.run(program=train_program, feed={"image": data}, fetch_list=[param.name], return_numpy=False)[0]
            # Z += (z,)
            # print(z)
            # mask = train_program.global_block().create_var(
            #     name=param.name + "_mask",
            #     shape=param.shape,
            #     dtype=param.dtype,
            #     type=param.type,
            #     persistable=param.persistable,
            #     stop_gradient=True)
            mask = scope.find_var(param.name+'_mask').get_tensor()
            print(mask)
            # paddle.static.global_scope().var(param.name+'_mask').get_tensor().set(w.astype("float32"), place)
            
            print(type(w))
            stop
            threshold = np.percentile(abs(z), 50)
            under_thres = abs(z) < threshold
            z[under_thres] = 0
            print(z)
            stop

    # if param.name.split('_')[-1] == 'weights':
        
        # # print(param)
        # print(idx)
        # w = paddle.static.global_scope().find_var(param.name)
        # print(type(w))
        # # z = fluid.create_lod_tensor(np.array(w), w.lod(), place=place)
        # z = w + w
        # print(id(w), id(z))

        # Z += (z,)
        # u = fluid.create_lod_tensor(np.zeros(w.shape()), w.lod(), place=place)
        # U += (u,)
        # # res = fluid.layers.sum([z, u])
        # # print(type(res))
        # x = fluid.layers.data(name='temp', shape=w.shape(), dtype='float32')
        # print(type(x))
        # idx+=1

        