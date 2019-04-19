import tensorflow as tf
import numpy as np
import parallax_debugger as md

from tensorflow.keras.datasets.mnist import load_data
import os, sys, subprocess
import argparse
from time import sleep
import time

"""
ProfileSessionContext example with LeNet.
This script should create '{curr_dir}/tmp' which contains
'run_meta_{step}' and 'worker-{id}_metagraph'.
"""

def next_batch(num, data, labels):
    idx = np.arange(0, len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[i] for i in idx]
    labels_shuffle = [labels[i] for i in idx]

    return [np.asarray(data_shuffle), np.asarray(labels_shuffle)]

def next_batch_for_replicas(num_replica, num_data, data, labels):
    data_label_pair_list = [next_batch(num_data, data, labels) for _ in range(num_replica)]
    return [list(l) for l in zip(*data_label_pair_list)]

def assertShape(tensor, shape):
    msg = "shape of {} is {}, expected to be {}".format(tensor, tensor.shape, shape)

    if len(tensor.shape) != len(shape):
        raise AssertionError(msg)

    for i, dim in enumerate(shape):
        if dim is None or dim == tensor.shape[i]:
            pass
        else:
            raise AssertionError(msg)

def pad2by2(tensor):
    return np.pad(tensor, ((0,0), (2, 2), (2, 2)), 'constant', constant_values=((0, 0), (0, 0), (0, 0)))


def one_hot(label, length):
    one_hot_vector = [0 for _ in range(length)]
    one_hot_vector[label] = 1
    return one_hot_vector

def remote_exec(bash_script,
        remote_machine,
        stdout=None,
        stderr=None,
        env={},
        python_venv=None,
        port=22):
    full_cmd = ' '.join(
            map(lambda (k, v): 'export %s=%s;' % (k, v), env.iteritems()))
    if python_venv is not None:
        full_cmd += ' source %s/bin/activate; ' % python_venv
    full_cmd += bash_script

    remote_cmd = 'ssh -tt -p %d %s \'bash -c "%s"\' </dev/null' % (
            port, remote_machine, full_cmd)

    proc = subprocess.Popen(args=remote_cmd, shell=True, stdout=stdout,
            stderr=stderr, preexec_fn=os.setsid)
    return proc

def _get_empty_port(hostname, num_ports):
    try:
        python_venv = os.environ['VIRTUAL_ENV']
    except:
        python_venv = None

    ports = []
    for i in range(num_ports):
        proc = remote_exec('python -m ephemeral_port_reserve', hostname, stdout=subprocess.PIPE, python_venv=python_venv)
        port = int(proc.stdout.readline())
        proc.wait()
        ports.append(port)
    return ports

def _dump_profile(metadata, task_index):
    profile_dir = os.path.join('profile',
                               'elsa-10',
                               'worekr:{}'.format(task_index),
                               'run_meta')
    print profile_dir
    if not tf.gfile.Exists(profile_dir):
        tf.gfile.MakeDirs(profile_dir)
    with tf.gfile.Open(os.path.join(profile_dir, 'run_meta_1'), 'wb') as f:
        f.write(metadata.SerializeToString())

def main(_):
    num_epoch = 20

    job_name = args.job_name
    task_index = args.task_index
    tf_cluster_dict = {'ps': ['elsa-10:80000'], 'worker': ['elsa-10:80001', 'elsa-10:80002']} 
    cluster_spec = tf.train.ClusterSpec(tf_cluster_dict)
    
    if job_name == 'server' or job_name == 's':
        server = tf.train.Server(cluster_spec, job_name='ps',
                                 task_index=task_index,
                                 protocol='grpc')
        server.join()
    else:
        server = tf.train.Server(cluster_spec, job_name='worker',
                                 task_index=task_index,
                                 protocol='grpc')

        is_chief = (task_index == 0)

        with tf.device(tf.train.replica_device_setter(
              "/job:worker/task:%d" % task_index,
              cluster=cluster_spec)):
            print "/job:worker/task:%d" % task_index
            x = tf.placeholder(tf.float32, shape=[None, 32, 32, 1])
            y = tf.placeholder(tf.float32, shape=[None, 10])
            keep_prob = tf.placeholder(tf.float32)

            W_conv1 = tf.Variable(tf.truncated_normal(shape=[5, 5, 1, 6], stddev=5e-2))  # what initializer?
            b_conv1 = tf.Variable(tf.constant(0.1, shape=[6]))
            h_conv1 = tf.nn.relu(tf.nn.conv2d(x, W_conv1, strides=[1, 1, 1, 1], padding='VALID') + b_conv1)
            assertShape(h_conv1, (None, 28, 28, 6))

            h_pool1 = tf.nn.avg_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
            assertShape(h_pool1, (None, 14, 14, 6))

            W_conv2 = tf.Variable(tf.truncated_normal(shape=[5, 5, 6, 16], stddev=5e-2))
            b_conv2 = tf.Variable(tf.constant(0.1, shape=[16]))
            h_conv2 = tf.nn.relu(tf.nn.conv2d(h_pool1, W_conv2, strides=[1, 1, 1, 1], padding='VALID') + b_conv2)
            assertShape(h_conv2, (None, 10, 10, 16))

            h_pool2 = tf.nn.avg_pool(h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
            assertShape(h_pool2, (None, 5, 5, 16))

            W_conv3 = tf.Variable(tf.truncated_normal(shape=[5, 5, 16, 16], stddev=5e-2))
            b_conv3 = tf.Variable(tf.constant(0.1, shape=[16]))
            h_conv3 = tf.nn.relu(tf.nn.conv2d(h_pool2, W_conv3, strides=[1, 1, 1, 1], padding='VALID') + b_conv3)
            h_conv3 = tf.squeeze(h_conv3, axis=[1, 2])
            assertShape(h_conv3, (None, 16))

            W_fc0 = tf.Variable(tf.truncated_normal(shape=[16, 120], stddev=5e-2))
            b_fc0 = tf.Variable(tf.constant(0.1, shape=[120]))

            h_fc0 = tf.nn.relu(tf.matmul(h_conv3, W_fc0) + b_fc0)

            W_fc1 = tf.Variable(tf.truncated_normal(shape=[120, 84], stddev=5e-2))
            b_fc1 = tf.Variable(tf.constant(0.1, shape=[84]))

            h_fc1 = tf.nn.relu(tf.matmul(h_fc0, W_fc1) + b_fc1)

            W_fc2 = tf.Variable(tf.truncated_normal(shape=[84, 10], stddev=5e-2))
            b_fc2 = tf.Variable(tf.constant(0.1, shape=[10]))
            logits = tf.matmul(h_fc1, W_fc2) + b_fc2
            y_pred = tf.nn.softmax(logits)

            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits))
            global_step = tf.train.get_or_create_global_step()
            optimizer = tf.train.GradientDescentOptimizer(0.01)
            optimizer = tf.train.SyncReplicasOptimizer(optimizer,
                                                       replicas_to_aggregate=len(tf_cluster_dict['worker']),
                                                       total_num_replicas=len(tf_cluster_dict['worker']))
            train_step = optimizer.minimize(loss, global_step=global_step)

            correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

            (x_train, y_train), (x_test, y_test) = load_data()
            x_train = pad2by2(x_train)
            x_train = np.expand_dims(x_train, axis=3)
            y_train_one_hot = np.array([one_hot(y_train[i], 10) for i in range(len(y_train))])
            
            init = tf.global_variables_initializer()

        curr_dir = os.path.dirname(os.path.abspath(__file__))
        profile_dir = os.path.join(curr_dir, 'profile_dir')
        profile_steps = range(10, 20)

        sync_replicas_hook = optimizer.make_session_run_hook(is_chief)
        stop_hook = tf.train.StopAtStepHook(last_step=num_epoch)
        hooks = [sync_replicas_hook,stop_hook]

        ps = md.ProfileSessionContext(profile_dir,
                                          task_index,
                                          profile_steps=profile_steps)
        with tf.train.MonitoredTrainingSession(
                master=server.target,
                is_chief=is_chief,
                hooks=hooks) as sess:
            sess.run(init)
            """
            with md.ProfileSessionContext(profile_dir,
                                          task_index,
                                          profile_steps=profile_steps):"""
            while not sess.should_stop():
                start = time.time()
                run_metadata = tf.RunMetadata()
                options = tf.RunOptions(
                    trace_level=tf.RunOptions.FULL_TRACE)
                feed_dict = dict()
                feed_dict[x], feed_dict[y] = next_batch_for_replicas(1, 128, x_train, y_train_one_hot)
                feed_dict[x] = feed_dict[x][0]
                feed_dict[y] = feed_dict[y][0]
                fetches = {
                    'global_step': global_step,
                    'loss': loss,
                    'train_op': train_step,
                    'accuracy': accuracy
                }
                results = sess.run(fetches, feed_dict=feed_dict, 
                                   options=options, 
                                   run_metadata=run_metadata)
                _dump_profile(run_metadata, task_index)
                finish = time.time()
                if i % 1 == 0:
                    print("global step: %d, loss: %f, acc: %f, latency: %f secs" % (results['global_step'], results['loss'], results['accuracy'], (finish-start)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("job_name", choices=['server', 's', 'worker', 'w'])
    parser.add_argument('task_index', type=int)
    parser.add_argument("-p", "--ports", nargs='+')
    args = parser.parse_args()
    tf.app.run()
