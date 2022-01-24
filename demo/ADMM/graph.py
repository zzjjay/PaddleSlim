import paddle
import pickle
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
import sys
sys.path.append(os.path.join(os.path.dirname("__file__"), os.path.pardir)) 
import models

def get_args():
    """Get arguments.
        Returns:
        Namespace, arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--layer', type=int, default=1, required=True, help='layer nums')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    paddle.enable_static()
    # pretrained_model = 'save_models' # pretrained
    # pretrained_model = 'admm_models'  # admm models
    # pretrained_model = 'retrain_models' # retrain
    layer = args.layer
    weight = []

    places = paddle.static.cpu_places()
    place = places[0]
    exe = paddle.static.Executor(place)

    # from lenet5 import LeNet5
    # model = LeNet5()
    model = models.__dict__['MobileNet']() 
    
    image = paddle.static.data(
        name='image', shape=[None] + [3, 224, 224], dtype='float32')
    label = paddle.static.data(name='label', shape=[None, 1], dtype='int64')
    out = model.net(image)
    train_program = paddle.static.default_main_program()

    exe.run(paddle.static.default_startup_program())

    for model_path in ['MobileNetV1_pretrained', 'admm_models_mbv1', 'MobileNetV1_pretrained']: 
        assert os.path
        def if_exist(var):
            return os.path.exists(os.path.join(model_path, var.name))
        paddle.fluid.io.load_vars(exe, model_path, predicate=if_exist)

        # data = [[], [], [], []]
        idx = 0
        for param in train_program.list_vars():
            if param.name.split('_')[-1] != 'weights':
                continue
            if not (len(param.shape) == 4 and param.shape[2] == 1 
                    and param.shape[3] == 1):
                continue
            if idx == layer:
                weight.append(np.array(param.get_value()))
                break
            idx += 1   
        # print(idx)

    plt.figure(num=0, figsize=[15,5])
    for i in range(3):
        plt.subplot(1, 3, i+1)

        target = np.array(weight[i]).flatten()
        print(min(target))
        print(max(target))
        x = np.array(target)
        width = 0.01
        begin = min(target)-0.01
        end = max(target)+0.01
        bins=np.arange(begin, end, width)#设置连续的边界值，即直方图的分布区间[0,10],[10,20]...
        #直方图会进行统计各个区间的数值
        frequency, _, _ =plt.hist(x,bins,color='blue',alpha=0.8, edgecolor='white')#alpha设置透明度，0为完全透明

        plt.xlabel('value')
        plt.ylabel('nums')
        plt.xlim(begin, end)#设置x轴分布范围
        
        plt.plot(bins[1:]-(width/2),frequency,color='red')
    # plt.show()
    name = f'layer_{layer}.jpg'
    plt.savefig(name)
    print("successfully save!")
