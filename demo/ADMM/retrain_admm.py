import os
import sys
import logging
import paddle
import argparse
import functools
import time
import numpy as np
import paddle.fluid as fluid
from paddleslim.prune.unstructured_pruner import UnstructuredPruner, GMPUnstructuredPruner
from paddleslim.common import get_logger
sys.path.append(os.path.join(os.path.dirname("__file__"), os.path.pardir))
import models
from utility import add_arguments, print_arguments
import paddle.vision.transforms as T
from paddle.fluid.incubate.fleet.collective import fleet, DistributedStrategy
from paddle.fluid.incubate.fleet.base import role_maker
import pickle
from ADMM_pruner import AdmmPruner
_logger = get_logger(__name__, level=logging.INFO)

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
# yapf: disable
add_arg('use_gpu',          bool, True,               "Whether to use gpu for traning or not. Defauly: True")
add_arg('batch_size',       int,  64,                 "Minibatch size. Default: 64")
add_arg('batch_size_for_validation',       int,  64,                 "Minibatch size for validation. Default: 64")
add_arg('model',            str,  "MobileNet",                "The target model.")
add_arg('pretrained_model', str,  None,                "Whether to use pretrained model. Default: None")
add_arg('checkpoint',       str, None, "The model to load for resuming training. Default: None")
add_arg('lr',               float,  0.1,               "The learning rate used to fine-tune pruned model. Default: 0.1")
add_arg('lr_strategy',      str,  "piecewise_decay",   "The learning rate decay strategy. Default: piecewise_decay")
add_arg('l2_decay',         float,  5e-5,               "The l2_decay parameter. Default: 3e-5")
add_arg('momentum_rate',    float,  0.9,               "The value of momentum_rate. Default: 0.9")
add_arg('pruning_strategy', str,    'base',            "The pruning strategy, currently we support base and gmp. Default: base")
add_arg('threshold',        float,  0.01,               "The threshold to set zeros, the abs(weights) lower than which will be zeros. Default: 0.01")
add_arg('pruning_mode',            str,  'ratio',               "the pruning mode: whether by ratio or by threshold. Default: ratio")
add_arg('ratio',            float,  0.55,               "The ratio to set zeros, the smaller portion will be zeros. Default: 0.55")
add_arg('num_epochs',       int,  120,               "The number of total epochs. Default: 120")
parser.add_argument('--step_epochs', nargs='+', type=int, default=[30, 60, 90], help="piecewise decay step")
add_arg('data',             str, "imagenet",                 "Which data to use. 'mnist' or 'imagenet'. Default: imagenet")
add_arg('log_period',       int, 100,                 "Log period in batches. Default: 100")
add_arg('test_period',      int, 5,                 "Test period in epoches. Default: 5")
add_arg('model_path',       str, "./retrain_models",         "The path to save model. Default: ./models")
add_arg('model_period',     int, 10,             "The period to save model in epochs. Default: 10")
add_arg('last_epoch',     int, -1,             "The last epoch we could train from. Default: -1")
add_arg('stable_epochs',    int, 0,              "The epoch numbers used to stablize the model before pruning. Default: 0")
add_arg('pruning_epochs',   int, 60,             "The epoch numbers used to prune the model by a ratio step. Default: 60")
add_arg('tunning_epochs',   int, 60,             "The epoch numbers used to tune the after-pruned models. Default: 60")
add_arg('pruning_steps',    int, 120,        "How many times you want to increase your ratio during training. Default: 120")
add_arg('initial_ratio',    float, 0.15,         "The initial pruning ratio used at the start of pruning stage. Default: 0.15")
add_arg('prune_params_type', str, None,           "Which kind of params should be pruned, we only support None (all but norms) and conv1x1_only for now. Default: None")
add_arg('local_sparsity', bool, False,            "Whether to prune all the parameter matrix at the same ratio or not. Default: False")
add_arg('rho', type=float, default=1e-3,   help='cardinality weight (default: 1e-3)')
# yapf: enable

model_list = models.__all__


def piecewise_decay(args, step_per_epoch):
    bd = [step_per_epoch * e for e in args.step_epochs]
    lr = [args.lr * (0.1**i) for i in range(len(bd) + 1)]
    last_iter = (1 + args.last_epoch) * step_per_epoch
    learning_rate = paddle.optimizer.lr.PiecewiseDecay(
        boundaries=bd, values=lr, last_epoch=last_iter)

    optimizer = paddle.optimizer.Momentum(
        learning_rate=learning_rate,
        momentum=args.momentum_rate,
        weight_decay=paddle.regularizer.L2Decay(args.l2_decay))
    return optimizer, learning_rate


def cosine_decay(args, step_per_epoch):
    last_iter = (1 + args.last_epoch) * step_per_epoch
    learning_rate = paddle.optimizer.lr.CosineAnnealingDecay(
        learning_rate=args.lr,
        T_max=args.num_epochs * step_per_epoch,
        last_epoch=last_iter)
    optimizer = paddle.optimizer.Momentum(
        learning_rate=learning_rate,
        momentum=args.momentum_rate,
        weight_decay=paddle.regularizer.L2Decay(args.l2_decay))
    return optimizer, learning_rate


def create_optimizer(args, step_per_epoch):
    if args.lr_strategy == "piecewise_decay":
        return piecewise_decay(args, step_per_epoch)
    elif args.lr_strategy == "cosine_decay":
        return cosine_decay(args, step_per_epoch)


def create_unstructured_pruner(train_program, args, place, configs, ratio):
    if configs is None:
        return UnstructuredPruner(
            train_program,
            mode=args.pruning_mode,
            ratio=args.ratio,
            threshold=args.threshold,
            prune_params_type=args.prune_params_type,
            place=place,
            local_sparsity=args.local_sparsity)
    else:
        print("local sparsity:", args.local_sparsity)
        return GMPUnstructuredPruner(
            train_program,
            ratio=ratio,
            prune_params_type=args.prune_params_type,
            place=place,
            local_sparsity=args.local_sparsity,
            configs=configs)


def create_Z_U(program):
    Z = []
    U = []

    for param in program.all_parameters():
        if param.name.split('_')[-1] != 'weights':
            continue
        if not (len(param.shape) == 4 and param.shape[2] == 1 
                    and param.shape[3] == 1):
                continue
        name_z = f'z_{len(Z)}'
        z = paddle.static.data(name=name_z, shape=param.shape, dtype='float32')
        Z.append(z)
        name_u = f'u_{len(U)}'
        u = paddle.static.data(name=name_u, shape=param.shape, dtype='float32')
        U.append(u)
    return Z, U

def initialize_Z_U(program):
    Z = []
    U = []
    for param in program.list_vars():
        if param.name.split('_')[-1] != 'weights':
            continue
        if not (len(param.shape) == 4 and param.shape[2] == 1 
                    and param.shape[3] == 1):
                continue
        z = np.array(param.get_value())
        Z.append(z)
        u = np.zeros_like(z)
        U.append(u)
    return Z, U

def ADMM_loss(args, program, Z, U, output, label):
    cost = paddle.nn.functional.loss.cross_entropy(input=output, label=label)
    loss = paddle.mean(x=cost)
    idx = 0
    for param in program.list_vars():
        if param.name.split('_')[-1] != 'weights':
            continue
        if not (len(param.shape) == 4 and param.shape[2] == 1 
                    and param.shape[3] == 1):
                continue
        loss += args.rho / 2 * (param - Z[idx] + U[idx]).norm()
        idx += 1
    return loss
    
def update_Z_U(program, Z, U, args):
    idx = 0
    percent = int(100*args.ratio)
    for param in program.list_vars():
        if param.name.split('_')[-1] != 'weights':
            continue
        if not (len(param.shape) == 4 and param.shape[2] == 1 
                    and param.shape[3] == 1):
                continue
        w = np.array(param.get_value())
        z = w + U[idx]
        threshold = np.percentile(abs(z), percent)
        z[abs(z) < threshold] = 0
        
        Z[idx] = z
        U[idx] = U[idx] + w - z
        idx += 1
    
    return Z, U

def print_convergence(program, Z):
    from numpy.linalg import norm
    idx = 0 
    for param in program.list_vars():
        if param.name.split('_')[-1] != 'weights':
            continue
        if not (len(param.shape) == 4 and param.shape[2] == 1 
                    and param.shape[3] == 1):
                continue
        w = np.array(param.get_value())
        norm_value = norm(w - Z[idx]) #/ norm(w)
        print("{}, {}: {:.4f}".format(idx, param.name, norm_value))
        idx += 1
   
def cal_params(program):
    total_params = 0
    for param in program.all_parameters():
        total_params += np.product(param.shape)
    return total_params

def compress(args):
    env = os.environ
    num_trainers = int(env.get('PADDLE_TRAINERS_NUM', 1))
    use_data_parallel = num_trainers > 1

    if use_data_parallel:
        # Fleet step 1: initialize the distributed environment
        role = role_maker.PaddleCloudRoleMaker(is_collective=True)
        fleet.init(role)

    train_reader = None
    test_reader = None
    if args.data == "mnist":
        transform = T.Compose([T.Transpose(), T.Normalize([127.5], [127.5])])
        train_dataset = paddle.vision.datasets.MNIST(
            mode='train', backend="cv2", transform=transform)
        val_dataset = paddle.vision.datasets.MNIST(
            mode='test', backend="cv2", transform=transform)
        class_dim = 10
        image_shape = "1,28,28"
        # args.pretrained_model = False
    elif args.data == "imagenet":
        import imagenet_reader as reader
        train_dataset = reader.ImageNetDataset(mode='train')
        val_dataset = reader.ImageNetDataset(mode='val')
        class_dim = 1000
        image_shape = "3,224,224"
    else:
        raise ValueError("{} is not supported.".format(args.data))
    image_shape = [int(m) for m in image_shape.split(",")]
    assert args.model in model_list, "{} is not in lists: {}".format(args.model,
                                                                     model_list)
    if args.use_gpu:
        places = paddle.static.cuda_places()
    else:
        places = paddle.static.cpu_places()
    place = places[0]
    exe = paddle.static.Executor(place)

    image = paddle.static.data(
        name='image', shape=[None] + image_shape, dtype='float32')
    label = paddle.static.data(name='label', shape=[None, 1], dtype='int64')
    
    batch_size_per_card = args.batch_size
    batch_sampler = paddle.io.DistributedBatchSampler(
        train_dataset,
        batch_size=batch_size_per_card,
        shuffle=True,
        drop_last=True)

    train_loader = paddle.io.DataLoader(
        train_dataset,
        places=place,
        batch_sampler=batch_sampler,
        feed_list=[image, label],
        return_list=False,
        use_shared_memory=True,
        num_workers=32)

    valid_loader = paddle.io.DataLoader(
        val_dataset,
        places=place,
        feed_list=[image, label],
        drop_last=False,
        return_list=False,
        use_shared_memory=True,
        batch_size=args.batch_size_for_validation,
        shuffle=False)

    step_per_epoch = int(
        np.ceil(len(train_dataset) * 1. / args.batch_size / num_trainers))

    # model definition
    model = models.__dict__[args.model]()
    # from lenet5 import LeNet5
    # model = LeNet5()
    out = model.net(input=image, class_dim=class_dim)
    cost = paddle.nn.functional.loss.cross_entropy(input=out, label=label)
    avg_cost = paddle.mean(x=cost)
    acc_top1 = paddle.metric.accuracy(input=out, label=label, k=1)
    acc_top5 = paddle.metric.accuracy(input=out, label=label, k=5)

    val_program = paddle.static.default_main_program().clone(for_test=True)

    opt, learning_rate = create_optimizer(args, step_per_epoch)
    # optimizer: admm loss
    # admm_opt, admm_lr =  create_optimizer(args, step_per_epoch)

    # Fleet step 2: distributed strategy
    if use_data_parallel:
        dist_strategy = DistributedStrategy()
        dist_strategy.sync_batch_norm = False
        dist_strategy.exec_strategy = paddle.static.ExecutionStrategy()
        dist_strategy.fuse_all_reduce_ops = False

    train_program = paddle.static.default_main_program()

    # GMP pruner step 1: initialize a pruner object by calling entry function.
    # ratio = {}
    # path = '/home/PaddleSlim/demo/unstructured_prune/sparse.pkl'
    # with open(path, 'rb') as f:
    #     ratio = pickle.load(f)
    #     print("ratio:", ratio)
    #     print("successfully!")

    # pruner = create_unstructured_pruner(
    #     train_program, args, place, configs, ratio)
    
    
    if use_data_parallel:
        # Fleet step 3: decorate the origial optimizer and minimize it
        opt = fleet.distributed_optimizer(opt, strategy=dist_strategy)

    admm_pruner = AdmmPruner(train_program, args.ratio)
    opt.minimize(avg_cost, no_grad_set=admm_pruner.no_grad_set)
    # opt.minimize(avg_cost)

    # Z, U = create_Z_U(train_program)

    # admm_loss = ADMM_loss(args, train_program, Z, U, out, label)
    # opt.minimize(admm_loss)

    exe.run(paddle.static.default_startup_program())

    
    if args.last_epoch > -1:
        assert args.checkpoint is not None and os.path.exists(
            args.checkpoint), "Please specify a valid checkpoint path."
        paddle.fluid.io.load_persistables(
            executor=exe, dirname=args.checkpoint, main_program=train_program)

    elif args.pretrained_model:
        assert os.path.exists(
            args.
            pretrained_model), "Pretrained model path {} doesn't exist".format(
                args.pretrained_model)

        def if_exist(var):
            return os.path.exists(os.path.join(args.pretrained_model, var.name))

        _logger.info("Load pretrained model from {}".format(
            args.pretrained_model))
        # NOTE: We are using fluid.io.load_vars() because the pretrained model is from an older version which requires this API. 
        # Please consider using paddle.static.load(program, model_path) when possible
        paddle.fluid.io.load_vars(
            exe, args.pretrained_model, predicate=if_exist)

    admm_pruner.initial_masks(train_program)
    admm_pruner.update_params()
    def test(epoch, program):
        admm_pruner.update_params()
        acc_top1_ns = []
        acc_top5_ns = []

        _logger.info(
            "The current sparsity of the inference model is {}%".format(
                round(100 * AdmmPruner.total_sparse(
                    paddle.static.default_main_program()), 2)))
        
        for batch_id, data in enumerate(valid_loader):
            start_time = time.time()
            acc_top1_n, acc_top5_n = exe.run(
                program, feed=data, fetch_list=[acc_top1.name, acc_top5.name])
            end_time = time.time()
            if batch_id % args.log_period == 0:
                _logger.info(
                    "Eval epoch[{}] batch[{}] - acc_top1: {}; acc_top5: {}; time: {}".
                    format(epoch, batch_id,
                           np.mean(acc_top1_n),
                           np.mean(acc_top5_n), end_time - start_time))
            acc_top1_ns.append(np.mean(acc_top1_n))
            acc_top5_ns.append(np.mean(acc_top5_n))

        _logger.info("Final eval epoch[{}] - acc_top1: {}; acc_top5: {}".format(
            epoch,
            np.mean(np.array(acc_top1_ns)), np.mean(np.array(acc_top5_ns))))

    def ADMM_train(args, program):
        train_reader_cost = 0.0
        train_run_cost = 0.0
        total_samples = 0
        reader_start = time.time()
        
        # Z_value, U_value = initialize_Z_U(train_program)
        # dict_ZU = {}
        # for idx in range(len(Z_value)):
        #     name_z = f"z_{idx}"
        #     dict_ZU[name_z] = Z_value[idx]
        #     name_u = f"u_{idx}"
        #     dict_ZU[name_u] = U_value[idx]

        for i in range(args.last_epoch + 1, args.num_epochs):
            for batch_id, data in enumerate(train_loader):
                train_reader_cost += time.time() - reader_start
                train_start = time.time()
                # print("data size:", len(data), type(data))
                # print("every data:", len(data[0]), type(data[0]))
                
                # add Z_value, U_value to data
                # data[0].update(dict_ZU)
                
                # sys.exit()
                loss_n, acc_top1_n, acc_top5_n = exe.run(
                    program,
                    feed=data,
                    fetch_list=[avg_cost.name, acc_top1.name, acc_top5.name])
                # admm_pruner.update_params()
                
                train_run_cost += time.time() - train_start
                total_samples += args.batch_size
                loss_n = np.mean(loss_n)
                acc_top1_n = np.mean(acc_top1_n)
                acc_top5_n = np.mean(acc_top5_n)
                if batch_id % args.log_period == 0:
                    _logger.info(
                        "epoch[{}]-batch[{}] lr: {:.6f} - loss: {}; acc_top1: {}; acc_top5: {}; avg_reader_cost: {:.5f} sec, avg_batch_cost: {:.5f} sec, avg_samples: {:.5f}, ips: {:.5f} images/sec".
                        format(i, batch_id,
                            learning_rate.get_lr(), loss_n, acc_top1_n,
                            acc_top5_n, train_reader_cost / args.log_period, (
                                train_reader_cost + train_run_cost
                            ) / args.log_period, total_samples / args.log_period,
                            total_samples / (train_reader_cost + train_run_cost
                                                )))
                    # scope = paddle.static.global_scope()
                    # for name in admm_pruner.mask_value:
                    #     real_mask_value = np.array(scope.find_var(name).get_tensor())
                    #     _logger.info("{}:{}".format(name, 
                    #     np.count_nonzero(real_mask_value)/np.product(real_mask_value.shape)))
                    
                    train_reader_cost = 0.0
                    train_run_cost = 0.0
                    total_samples = 0
                learning_rate.step()
                reader_start = time.time()

        # update Z, U
            # Z_value, U_value = update_Z_U(train_program, Z_value, U_value, args)
            # _logger.info("normalized norm of (weight - projection)")
            # print_convergence(train_program, Z_value)
            
            _logger.info("The current sparsity of the pruned model is: {}%".format(
            round(100 * AdmmPruner.total_sparse(
                paddle.static.default_main_program()), 2)))


            if (i + 1) % args.test_period == 0:
                test(i, val_program)
            if (i + 1) % args.model_period == 0:
                if use_data_parallel:
                    fleet.save_persistables(executor=exe, dirname=args.model_path)
                else:
                    paddle.fluid.io.save_persistables(
                        executor=exe, dirname=args.model_path)
    
    if use_data_parallel:
        # Fleet step 4: get the compiled program from fleet
        compiled_train_program = fleet.main_program
    else:
        compiled_train_program = paddle.static.CompiledProgram(
            paddle.static.default_main_program())

    test(-1, val_program)
    before = cal_params(val_program)
    print("befor:", before)
    ADMM_train(args, compiled_train_program)
    total_params = 0
    for param in val_program.all_parameters():
        total_params += np.count_nonzero(
                np.array(paddle.static.global_scope().find_var(param.name)
                         .get_tensor()))

    test(args.num_epochs-1, val_program)
    _logger.info("params before:{} after:{}".format(before, total_params))
    if use_data_parallel:
        fleet.save_persistables(executor=exe, dirname=args.model_path)
    else:
        paddle.fluid.io.save_persistables(
            executor=exe, dirname=args.model_path)
    # GMP pruner step 3: update params before summrizing sparsity, saving model or evaluation. 
    # pruner.update_params()


def main():
    paddle.enable_static()
    args = parser.parse_args()
    print_arguments(args)
    compress(args)


if __name__ == '__main__':
    main()
