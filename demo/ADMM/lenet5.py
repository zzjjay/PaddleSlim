import paddle.fluid as fluid
from paddle.fluid.initializer import MSRA
from paddle.fluid.param_attr import ParamAttr

class LeNet5():
    def __init__(self):
        self.params = None

    def net(self, input, class_dim=10):
        input = self.conv(input, num_filters=20, kernel=5, stride=1, name='conv1')
        input = fluid.layers.pool2d(
            input=input,
            pool_size=2,
            pool_stride=2,
            pool_type='max')
        input = self.conv(input, 50, 5, 1, name='conv2')
        input = fluid.layers.pool2d(
            input=input,
            pool_size=2,
            pool_stride=2,
            pool_type='max')
        with fluid.name_scope('last_fc1'):
            input = fluid.layers.fc(input=input,
                                     size=500,
                                     act='relu',
                                     param_attr=ParamAttr(
                                         initializer=MSRA(),
                                         name="fc1_weights"),
                                     bias_attr=ParamAttr(name="fc1_offset"))
        with fluid.name_scope('last_fc2'):
            output = fluid.layers.fc(input=input,
                                     size=class_dim,
                                     act='relu',
                                     param_attr=ParamAttr(
                                         initializer=MSRA(),
                                         name="fc2_weights"),
                                     bias_attr=ParamAttr(name="fc2_offset"))

        return output
        

    
    def conv(self, input, num_filters, kernel, stride, name):
        output = fluid.layers.conv2d(
            input=input,
            num_filters=num_filters,
            filter_size=kernel,
            stride=stride,
            padding=0,
            act='relu',
            use_cudnn=True,
            param_attr=ParamAttr(
                initializer=MSRA(), name=name + "_weights"),
            bias_attr=False)
        return output