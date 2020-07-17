import argparse
import math
import h5py
import numpy as np
import tensorflow as tf
import socket
import importlib
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))
import provider
from utils import tf_util

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='dgcnn', help='Model name: dgcnn')
parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
parser.add_argument('--num_point', type=int, default=1024,
                    help='How many points will be taken from each cloud during training and testing [256/512/1024/2048] [default: 1024]')
parser.add_argument('--max_epoch', type=int, default=250, help='Epoch to run [default: 250]')
parser.add_argument('--batch_size', type=int, default=24, help='Batch Size during training [default: 32]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=200000, help='Decay step for lr decay [default: 200000]')
parser.add_argument('--decay_rate', type=float, default=0.7, help='Decay rate for lr decay [default: 0.8]')
FLAGS = parser.parse_args()

BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
GPU_INDEX = FLAGS.gpu
MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate

MODEL = importlib.import_module(FLAGS.model)  # import network module
MODEL_FILE = os.path.join(BASE_DIR, 'models', FLAGS.model + '.py')
LOG_DIR = FLAGS.log_dir
if not os.path.exists(LOG_DIR):
    os.mkdir(LOG_DIR)
# os.system('cp %s %s' % (MODEL_FILE, LOG_DIR)) # bkp of model def
# os.system('cp train.py %s' % (LOG_DIR)) # bkp of train procedure
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS) + '\n')

MAX_NUM_POINT = 4096 #2048
NUM_CLASSES = 40

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

HOSTNAME = socket.gethostname()

# ModelNet40 official train/test split
TRAIN_FILES = provider.get_data_files( \
    os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/train_files.txt'))
TEST_FILES = provider.get_data_files( \
    os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/test_files.txt'))


# def log_string2(out_str):
#     LOG_FOUT.write(out_str + '\n')
#     LOG_FOUT.flush()
#     print(out_str)


# def log_string(log_file, out_str):
#     log_file.write(out_str + '\n')
#     log_file.flush()
#     print(out_str)


def get_learning_rate(batch):
    learning_rate = tf.train.exponential_decay(
        BASE_LEARNING_RATE,  # Base learning rate.
        batch * BATCH_SIZE,  # Current index into the dataset.
        DECAY_STEP,  # Decay step.
        DECAY_RATE,  # Decay rate.
        staircase=True)
    learning_rate = tf.maximum(learning_rate, 0.00001)  # don't allow learning rate go beyond 0.00001
    return learning_rate


def get_bn_decay(batch):
    bn_momentum = tf.train.exponential_decay(
        BN_INIT_DECAY,
        batch * BATCH_SIZE,
        BN_DECAY_DECAY_STEP,
        BN_DECAY_DECAY_RATE,
        staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay


def train():
    with tf.Graph().as_default():
        with tf.device('/gpu:' + str(GPU_INDEX)):
            pointclouds_pl, labels_pl = MODEL.placeholder_inputs(BATCH_SIZE,
                                                                 NUM_POINT)  # create placeholders using method
            is_training_pl = tf.placeholder(tf.bool, shape=())
            print(is_training_pl)

            # Note the global_step=batch parameter to minimize. 
            # That tells the optimizer to helpfully increment the 'batch' parameter for you every time it trains.
            batch = tf.Variable(0)
            bn_decay = get_bn_decay(batch)
            tf.summary.scalar('bn_decay', bn_decay)

            # Get prediction and loss based on placeholders
            pred, end_points = MODEL.get_model(pointclouds_pl, is_training_pl,
                                               bn_decay=bn_decay)  # predictions on probabilities of BATCH_SIZE clouds belonging to 40 classes
            loss = MODEL.get_loss(pred, labels_pl,
                                  end_points)  # value of loss for batch, based on difference of predictions and true values of classes for BATCH_SIZE clouds
            tf.summary.scalar('loss', loss)

            correct = tf.equal(tf.argmax(pred, 1), tf.to_int64(
                labels_pl))  # vector of booleans of correct guesses, based on predictions and and true values of classes for BATCH_SIZE clouds
            accuracy = tf.reduce_sum(tf.cast(correct, tf.float32)) / float(
                BATCH_SIZE)  # cast vector of corrects to float, define number of corrects and divide by total number
            tf.summary.scalar('accuracy', accuracy)

            # Get training operator
            learning_rate = get_learning_rate(batch)  # calculate current value of learning rate
            tf.summary.scalar('learning_rate', learning_rate)
            if OPTIMIZER == 'momentum':
                optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
            elif OPTIMIZER == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate)
            train_op = optimizer.minimize(loss, global_step=batch)  # define what does optimizer minimize

            # Add ops to save and restore all the variables.
            saver = tf.train.Saver()

        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)

        # Add summary writers
        # merged = tf.merge_all_summaries()
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'),
                                             sess.graph)
        test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'test'))

        # Init variables
        init = tf.global_variables_initializer()
        # To fix the bug introduced in TF 0.12.1 as in
        # http://stackoverflow.com/questions/41543774/invalidargumenterror-for-tensor-bool-tensorflow-0-12-1
        # sess.run(init)
        sess.run(init, {is_training_pl: True})

        ops = {'pointclouds_pl': pointclouds_pl,
               'labels_pl': labels_pl,
               'is_training_pl': is_training_pl,
               'pred': pred,
               'loss': loss,
               'train_op': train_op,
               'merged': merged,
               'step': batch}

        # for every epoch in range 0 to MAX_EPOCH - 1 do train, evaluation and saving
        for epoch in range(MAX_EPOCH):
            log_string(LOG_FOUT, '**** EPOCH %03d ****' % (epoch))
            sys.stdout.flush()

            # training within every epoch will be done using 5 train-files,
            # dividing each into pieces of size BATCH_SIZE
            # BATCH_SIZE of clouds with different labels
            train_one_epoch(sess, ops, train_writer)

            # evaluation within every epoch
            eval_one_epoch(sess, ops, test_writer)

            # Save the variables to disk.
            if epoch % 10 == 0:
                save_path = saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"))
                log_string(LOG_FOUT, "Model saved in file: %s" % save_path)


# define function which creates data batches and feeds them to a built graph for training
def train_one_epoch(sess, ops, train_writer):
    """ ops: dict mapping from string to tf ops """
    is_training = True

    # Shuffle train files
    train_file_idxs = np.arange(0, len(TRAIN_FILES))
    np.random.shuffle(train_file_idxs)  # train_file_idxs will be some random sequence of indexes of train files
    print("train_file_idxs is", train_file_idxs)

    # training within every epoch will be done using 5 train-files
    for fn in range(len(TRAIN_FILES)):
        log_string(LOG_FOUT, 'train file ----' + str(fn) + '-----')
        # train_file_idxs[fn] is some random index of train file.
        # current_data will be 2048 clouds with 2048 (MAX_NUM_POINT) points in each, 3 coors per point
        # current_label will be 2048 clouds with 1 label each
        # current_data, current_label = provider.loadDataFile(TRAIN_FILES[train_file_idxs[fn]])
        current_data, current_label = provider.load_h5(TRAIN_FILES[train_file_idxs[fn]])
        current_data = current_data[:, 0:NUM_POINT, :]  # cut only specified number of points from each cloud
        current_data, current_label, _ = provider.shuffle_data(current_data, np.squeeze(current_label))
        # current_label = np.squeeze(current_label)

        file_size = current_data.shape[0]
        num_batches = file_size // BATCH_SIZE

        total_correct = 0
        total_seen = 0
        loss_sum = 0

        # clouds within every file will be divided into groups of size BATCH_SIZE
        for batch_idx in range(num_batches):
            start_idx = batch_idx * BATCH_SIZE
            end_idx = (batch_idx + 1) * BATCH_SIZE

            # Cut a sequence of BATCH_SIZE clouds from current_data
            # And augment batched point clouds by rotation and jittering
            rotated_data = provider.rotate_point_cloud(current_data[start_idx:end_idx, :, :])
            jittered_data = provider.jitter_point_cloud(rotated_data)
            jittered_data = provider.random_scale_point_cloud(jittered_data)
            jittered_data = provider.rotate_perturbation_point_cloud(jittered_data)
            jittered_data = provider.shift_point_cloud(jittered_data)

            # define dict and perform feeding of this batch
            feed_dict = {ops['pointclouds_pl']: jittered_data,
                         ops['labels_pl']: current_label[start_idx:end_idx],
                         ops['is_training_pl']: is_training, }
            summary, step, _, loss_value, pred_value = sess.run([ops['merged'], ops['step'],
                                                                 ops['train_op'], ops['loss'], ops['pred']],
                                                                feed_dict=feed_dict)
            train_writer.add_summary(summary, step)
            pred_value = np.argmax(pred_value, 1)  # pred_value is argmax of pred_value probabilities on axis 1
            correct = np.sum(pred_value == current_label[start_idx:end_idx])  # number of correct predictions
            total_correct += correct
            total_seen += BATCH_SIZE
            loss_sum += loss_value

        log_string(LOG_FOUT, 'mean loss: %f' % (loss_sum / float(num_batches)))  # mean loss of current epoch
        log_string(LOG_FOUT, 'accuracy: %f' % (
                    total_correct / float(total_seen)))  # accuracy of current epoch. Both numbers count clouds


# define function which creates data batches and feeds them to a built graph for testing
def eval_one_epoch(sess, ops, test_writer):
    """ ops: dict mapping from string to tf ops """
    is_training = False
    total_correct = 0
    total_seen = 0
    loss_sum = 0
    total_seen_class = [0 for _ in range(NUM_CLASSES)]
    total_correct_class = [0 for _ in range(NUM_CLASSES)]

    # evaluation within every epoch will be done using 5 train-files
    for fn in range(len(TEST_FILES)):
        log_string(LOG_FOUT, 'eval file ----' + str(fn) + '-----')
        #current_data, current_label = provider.loadDataFile(TEST_FILES[fn])  # load data and labels from file
        current_data, current_label = provider.load_h5(TEST_FILES[fn])  # load data and labels from file
        current_data = current_data[:, 0:NUM_POINT, :]  # crop only specific number of points from current_data
        current_label = np.squeeze(current_label)  # no need to shuffle, so that's it

        file_size = current_data.shape[0]
        num_batches = file_size // BATCH_SIZE

        # clouds within every file will be divided into groups of size BATCH_SIZE
        for batch_idx in range(num_batches):
            start_idx = batch_idx * BATCH_SIZE
            end_idx = (batch_idx + 1) * BATCH_SIZE

            # define dict and perform feeding of this batch
            feed_dict = {ops['pointclouds_pl']: current_data[start_idx:end_idx, :, :],
                         ops['labels_pl']: current_label[start_idx:end_idx],
                         ops['is_training_pl']: is_training}
            summary, step, loss_value, pred_value = sess.run([ops['merged'], ops['step'],
                                                              ops['loss'], ops['pred']], feed_dict=feed_dict)
            pred_value = np.argmax(pred_value, 1)  # pred_value is argmax of pred_value probabilities on axis 1
            correct = np.sum(pred_value == current_label[start_idx:end_idx])  # number of correct predictions
            total_correct += correct
            total_seen += BATCH_SIZE
            loss_sum += (loss_value * BATCH_SIZE)
            #loss_sum += loss_value
            for i in range(start_idx, end_idx):
                l = current_label[i]
                total_seen_class[l] += 1
                total_correct_class[l] += (pred_value[i - start_idx] == l)

    log_string(LOG_FOUT, 'eval mean loss: %f' % (loss_sum / float(total_seen)))
    #log_string(LOG_FOUT, 'eval mean loss: %f' % (loss_sum / float(num_batches)))
    log_string(LOG_FOUT, 'eval accuracy: %f' % (total_correct / float(total_seen)))
    log_string(LOG_FOUT, 'eval avg class acc: %f' % (
        np.mean(np.array(total_correct_class) / np.array(total_seen_class, dtype=np.float))))


if __name__ == "__main__":
    train()
    LOG_FOUT.close()
