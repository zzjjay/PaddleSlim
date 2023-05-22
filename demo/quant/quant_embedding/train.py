from __future__ import print_function
import argparse
import logging
import os
import time
import random
import numpy as np
import paddle
import reader
from net import skip_gram_word2vec
import paddle

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("paddle")
logger.setLevel(logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser(
        description="PaddlePaddle Word2vec example")
    parser.add_argument(
        '--train_data_dir',
        type=str,
        default='./data/text',
        help="The path of taining dataset")
    parser.add_argument(
        '--base_lr',
        type=float,
        default=0.01,
        help="The number of learing rate (default: 0.01)")
    parser.add_argument(
        '--save_step',
        type=int,
        default=500000,
        help="The number of step to save (default: 500000)")
    parser.add_argument(
        '--print_batch',
        type=int,
        default=10,
        help="The number of print_batch (default: 10)")
    parser.add_argument(
        '--dict_path',
        type=str,
        default='./data/1-billion_dict',
        help="The path of data dict")
    parser.add_argument(
        '--batch_size',
        type=int,
        default=500,
        help="The size of mini-batch (default:500)")
    parser.add_argument(
        '--num_passes',
        type=int,
        default=10,
        help="The number of passes to train (default: 10)")
    parser.add_argument(
        '--model_output_dir',
        type=str,
        default='models',
        help='The path for model to store (default: models)')
    parser.add_argument('--nce_num', type=int, default=5, help='nce_num')
    parser.add_argument(
        '--embedding_size',
        type=int,
        default=64,
        help='sparse feature hashing space for index processing')
    parser.add_argument(
        '--is_sparse',
        action='store_true',
        required=False,
        default=False,
        help='embedding and nce will use sparse or not, (default: False)')
    parser.add_argument(
        '--with_speed',
        action='store_true',
        required=False,
        default=False,
        help='print speed or not , (default: False)')
    parser.add_argument(
        '--ce_test',
        required=False,
        default=False,
        help='Whether to CE test, (default: False)')

    return parser.parse_args()


def train_loop(args, train_program, dataset, inputs, loss, trainer_id, weight,
               lr):

    place = paddle.CPUPlace()
    exe = paddle.static.Executor(place)
    batch_sampler = reader.MySampler(dataset, args.batch_size, args.nce_num,
                                     weight)
    train_loader = paddle.io.DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        places=place,
        feed_list=inputs,
        return_list=False,
        use_shared_memory=True, )

    exe.run(paddle.static.default_startup_program())

    exec_strategy = paddle.static.ExecutionStrategy()
    exec_strategy.use_experimental_executor = True

    print("CPU_NUM:" + str(os.getenv("CPU_NUM")))
    exec_strategy.num_threads = int(os.getenv("CPU_NUM"))

    build_strategy = paddle.static.BuildStrategy()
    if int(os.getenv("CPU_NUM")) > 1:
        build_strategy.reduce_strategy = paddle.static.BuildStrategy.ReduceStrategy.Reduce

    program = paddle.static.CompiledProgram(
        train_program, build_strategy=build_strategy)

    for pass_id in range(args.num_passes):
        time.sleep(10)
        epoch_start = time.time()
        for data in train_loader():
            batch_id = 0
            start = time.time()

            loss_val = exe.run(program, fetch_list=[loss.name], feed=data)
            loss_val = np.mean(loss_val)

            if batch_id % args.print_batch == 0:
                logger.info("TRAIN --> pass: {} batch: {} loss: {}".format(
                    pass_id, batch_id, loss_val.mean()))
            if args.with_speed:
                if batch_id % 500 == 0 and batch_id != 0:
                    elapsed = (time.time() - start)
                    start = time.time()
                    samples = 1001 * args.batch_size * int(os.getenv("CPU_NUM"))
                    logger.info("Time used: {}, Samples/Sec: {}".format(
                        elapsed, samples / elapsed))
            lr.step()

            if batch_id % args.save_step == 0 and batch_id != 0:
                model_dir = args.model_output_dir + '/pass-' + str(pass_id) + (
                    '/batch-' + str(batch_id))
                if trainer_id == 0:
                    paddle.static.save(train_program, model_dir)
                    print("model saved in %s" % model_dir)
            batch_id += 1

        epoch_end = time.time()
        logger.info("Epoch: {0}, Train total expend: {1} ".format(
            pass_id, epoch_end - epoch_start))
        model_dir = args.model_output_dir + '/pass-' + str(pass_id)
        if trainer_id == 0:
            paddle.static.save(train_program, model_dir)
            print("model saved in %s" % model_dir)


def GetFileList(data_path):
    return os.listdir(data_path)


def train(args):
    if args.ce_test:
        # set seed
        seed = 111
        paddle.seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    if not os.path.isdir(args.model_output_dir):
        os.mkdir(args.model_output_dir)

    filelist = GetFileList(args.train_data_dir)
    word2vec_dataset = reader.Word2VecDataset(
        args.dict_path, args.train_data_dir, filelist, 0, 1)

    logger.info("dict_size: {}".format(word2vec_dataset.dict_size))
    np_power = np.power(np.array(word2vec_dataset.id_frequencys), 0.75)
    id_frequencys_pow = np_power / np_power.sum()

    loss, words = skip_gram_word2vec(
        word2vec_dataset.dict_size,
        args.embedding_size,
        args.batch_size,
        is_sparse=args.is_sparse,
        neg_num=args.nce_num)

    learning_rate = paddle.optimizer.lr.ExponentialDecay(
        args.base_lr, gamma=0.999)

    optimizer = paddle.optimizer.SGD(learning_rate=learning_rate)

    optimizer.minimize(loss)

    # do local training
    logger.info("run local training")
    main_program = paddle.static.default_main_program()
    train_loop(args, main_program, word2vec_dataset, words, loss, 0,
               id_frequencys_pow, learning_rate)


if __name__ == '__main__':
    paddle.enable_static()
    args = parse_args()
    train(args)
