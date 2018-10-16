# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

import argparse
import os
import tensorflow as tf
import tensorflow.contrib.tensorrt as trt
import time
import numpy as np
import sys
from classification import build_classification_graph, get_preprocess_fn, get_netdef

class LoggerHook(tf.train.SessionRunHook):
    """Logs runtime of each iteration"""
    def __init__(self, batch_size, num_records, display_every):
        self.iter_times = []
        self.display_every = display_every
        self.num_steps = (num_records + batch_size - 1) / batch_size
        self.batch_size = batch_size

    def begin(self):
        self.start_time = time.time()

    def after_run(self, run_context, run_values):
        current_time = time.time()
        duration = current_time - self.start_time
        self.start_time = current_time
        self.iter_times.append(duration)
        current_step = len(self.iter_times)
        if current_step % self.display_every == 0:
            print("    step %d/%d, iter_time(ms)=%.4f, images/sec=%d" % (
                current_step, self.num_steps, duration * 1000,
                self.batch_size / self.iter_times[-1]))

def run(frozen_graph, model, data_dir, batch_size,
    num_iterations, num_warmup_iterations, use_synthetic, display_every=100):
    """Evaluates a frozen graph
    
    This function evaluates a graph on the ImageNet validation set.
    tf.estimator.Estimator is used to evaluate the accuracy of the model
    and a few other metrics. The results are returned as a dict.

    frozen_graph: GraphDef, a graph containing input node 'input' and outputs 'logits' and 'classes'
    model: string, the model name (see NETS table in graph.py)
    data_dir: str, directory containing ImageNet validation TFRecord files
    batch_size: int, batch size for TensorRT optimizations
    num_iterations: int, number of iterations(batches) to run for
    """
    # Define model function for tf.estimator.Estimator
    def model_fn(features, labels, mode):
        logits_out, classes_out = tf.import_graph_def(frozen_graph,
            input_map={'input': features},
            return_elements=['logits:0', 'classes:0'],
            name='')
        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits_out)
        accuracy = tf.metrics.accuracy(labels=labels, predictions=classes_out, name='acc_op')
        if mode == tf.estimator.ModeKeys.EVAL:
            return tf.estimator.EstimatorSpec(
                mode,
                loss=loss,
                eval_metric_ops={'accuracy': accuracy})

    # Create the dataset
    preprocess_fn = get_preprocess_fn(model)
    validation_files = tf.gfile.Glob(os.path.join(data_dir, 'validation*'))

    def get_tfrecords_count(files):
        num_records = 0
        for fn in files:
            for record in tf.python_io.tf_record_iterator(fn):
                num_records += 1
        return num_records

    # Define the dataset input function for tf.estimator.Estimator
    def eval_input_fn():
        if use_synthetic:
            input_width, input_height = get_netdef(model).get_input_dims()
            features = np.random.normal(
                loc=112, scale=70,
                size=(batch_size, input_height, input_width, 3)).astype(np.float32)
            features = np.clip(features, 0.0, 255.0)
            features = tf.identity(tf.constant(features))
            labels = np.random.randint(
                low=0,
                high=get_netdef(model).get_num_classes(),
                size=(batch_size),
                dtype=np.int32)
            labels = tf.identity(tf.constant(labels))
        else:
            dataset = tf.data.TFRecordDataset(validation_files)
            dataset = dataset.apply(tf.contrib.data.map_and_batch(map_func=preprocess_fn, batch_size=batch_size, num_parallel_calls=8))
            dataset = dataset.prefetch(buffer_size=tf.contrib.data.AUTOTUNE)
            dataset = dataset.repeat(count=1)
            iterator = dataset.make_one_shot_iterator()
            features, labels = iterator.get_next()
        return features, labels

    # Evaluate model
    logger = LoggerHook(
        display_every=display_every,
        batch_size=batch_size,
        num_records=get_tfrecords_count(validation_files))
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        config=tf.estimator.RunConfig(session_config=tf_config))
    results = estimator.evaluate(eval_input_fn, steps=num_iterations, hooks=[logger])
    
    # Gather additional results
    iter_times = np.array(logger.iter_times[num_warmup_iterations:])
    results['total_time'] = np.sum(iter_times)
    results['images_per_sec'] = np.mean(batch_size / iter_times)
    results['99th_percentile'] = np.percentile(iter_times, q=99, interpolation='lower') * 1000
    results['latency_mean'] = np.mean(iter_times) * 1000
    return results

def get_frozen_graph(
    model,
    use_trt=False,
    use_dynamic_op=False,
    precision='fp32',
    batch_size=8,
    minimum_segment_size=2,
    calib_data_dir=None,
    num_calib_inputs=None,
    use_synthetic=False,
    cache=False,
    download_dir='./data'):
    """Retreives a frozen GraphDef from model definitions in classification.py and applies TF-TRT

    model: str, the model name (see NETS table in classification.py)
    use_trt: bool, if true, use TensorRT
    precision: str, floating point precision (fp32, fp16, or int8)
    batch_size: int, batch size for TensorRT optimizations
    returns: tensorflow.GraphDef, the TensorRT compatible frozen graph
    """
    num_nodes = {}
    times = {}

    # Load from pb file if frozen graph was already created and cached
    if cache:
        # Graph must match the model, TRT mode, precision, and batch size
        prebuilt_graph_path = "graphs/frozen_graph_%s_%d_%s_%d.pb" % (model, int(use_trt), precision, batch_size)
        if os.path.isfile(prebuilt_graph_path):
            print('Loading cached frozen graph from \'%s\'' % prebuilt_graph_path)
            start_time = time.time()
            with tf.gfile.GFile(prebuilt_graph_path, "rb") as f:
                frozen_graph = tf.GraphDef()
                frozen_graph.ParseFromString(f.read())
            times['loading_frozen_graph'] = time.time() - start_time
            num_nodes['loaded_frozen_graph'] = len(frozen_graph.node)
            num_nodes['trt_only'] = len([1 for n in frozen_graph.node if str(n.op)=='TRTEngineOp'])
            return frozen_graph, num_nodes, times

    # Build graph and load weights
    frozen_graph = build_classification_graph(model, download_dir)
    num_nodes['native_tf'] = len(frozen_graph.node)

    # Convert to TensorRT graph
    if use_trt:
        start_time = time.time()
        frozen_graph = trt.create_inference_graph(
            input_graph_def=frozen_graph,
            outputs=['logits', 'classes'],
            max_batch_size=batch_size,
            max_workspace_size_bytes=4096 << 20,
            precision_mode=precision,
            minimum_segment_size=minimum_segment_size,
            is_dynamic_op=use_dynamic_op
        )
        times['trt_conversion'] = time.time() - start_time
        num_nodes['tftrt_total'] = len(frozen_graph.node)
        num_nodes['trt_only'] = len([1 for n in frozen_graph.node if str(n.op)=='TRTEngineOp'])

        if precision == 'int8':
            calib_graph = frozen_graph
            # INT8 calibration step
            print('Calibrating INT8...')
            start_time = time.time()
            run(calib_graph, model, calib_data_dir, batch_size,
                num_calib_inputs // batch_size, 0, False)
            times['trt_calibration'] = time.time() - start_time

            start_time = time.time()
            frozen_graph = trt.calib_graph_to_infer_graph(calib_graph)
            times['trt_int8_conversion'] = time.time() - start_time

            del calib_graph
            print('INT8 graph created.')

    # Cache graph to avoid long conversions each time
    if cache:
        if not os.path.exists(os.path.dirname(prebuilt_graph_path)):
            try:
                os.makedirs(os.path.dirname(prebuilt_graph_path))
            except Exception as e:
                raise e
        start_time = time.time()
        with tf.gfile.GFile(prebuilt_graph_path, "wb") as f:
            f.write(frozen_graph.SerializeToString())
        times['saving_frozen_graph'] = time.time() - start_time

    return frozen_graph, num_nodes, times

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate model')
    parser.add_argument('--model', type=str, default='inception_v4',
        choices=['mobilenet_v1', 'mobilenet_v2', 'nasnet_mobile', 'nasnet_large',
                 'resnet_v1_50', 'resnet_v2_50', 'vgg_16', 'vgg_19', 'inception_v3', 'inception_v4'],
        help='Which model to use.')
    parser.add_argument('--data_dir', type=str, required=True,
        help='Directory containing validation set TFRecord files.')
    parser.add_argument('--calib_data_dir', type=str,
        help='Directory containing TFRecord files for calibrating int8.')
    parser.add_argument('--download_dir', type=str, default='./data',
        help='Directory where downloaded model checkpoints will be stored.')
    parser.add_argument('--use_trt', action='store_true',
        help='If set, the graph will be converted to a TensorRT graph.')
    parser.add_argument('--use_trt_dynamic_op', action='store_true',
        help='If set, TRT conversion will be done using dynamic op instead of statically.')
    parser.add_argument('--precision', type=str, choices=['fp32', 'fp16', 'int8'], default='fp32',
        help='Precision mode to use. FP16 and INT8 only work in conjunction with --use_trt')
    parser.add_argument('--batch_size', type=int, default=8,
        help='Number of images per batch.')
    parser.add_argument('--minimum_segment_size', type=int, default=2,
        help='Minimum number of TF ops in a TRT engine.')
    parser.add_argument('--num_iterations', type=int, default=None,
        help='How many iterations(batches) to evaluate. If not supplied, the whole set will be evaluated.')
    parser.add_argument('--display_every', type=int, default=100,
        help='Number of iterations executed between two consecutive display of metrics')
    parser.add_argument('--use_synthetic', action='store_true',
        help='If set, one batch of random data is generated and used at every iteration.')
    parser.add_argument('--num_warmup_iterations', type=int, default=50,
        help='Number of initial iterations skipped from timing')
    parser.add_argument('--num_calib_inputs', type=int, default=500,
        help='Number of inputs (e.g. images) used for calibration '
        '(last batch is skipped in case it is not full)')
    parser.add_argument('--cache', action='store_true',
        help='If set, graphs will be saved to disk after conversion. If a converted graph is present on disk, it will be loaded instead of building the graph again.')
    args = parser.parse_args()

    if args.precision != 'fp32' and not args.use_trt:
        raise ValueError('TensorRT must be enabled for fp16 or int8 modes (--use_trt).')
    if args.precision == 'int8' and not args.calib_data_dir:
        raise ValueError('--calib_data_dir is required for int8 mode')
    if args.num_iterations is not None and args.num_iterations <= args.num_warmup_iterations:
        raise ValueError('--num_iterations must be larger than --num_warmup_iterations '
            '({} <= {})'.format(args.num_iterations, args.num_warmup_iterations))
    if args.num_calib_inputs < args.batch_size:
        raise ValueError('--num_calib_inputs must not be smaller than --batch_size'
            '({} <= {})'.format(args.num_calib_inputs, args.batch_size))

    # Retreive graph using NETS table in graph.py
    frozen_graph, num_nodes, times = get_frozen_graph(
        model=args.model,
        use_trt=args.use_trt,
        use_dynamic_op=args.use_trt_dynamic_op,
        precision=args.precision,
        batch_size=args.batch_size,
        minimum_segment_size=args.minimum_segment_size,
        calib_data_dir=args.calib_data_dir,
        num_calib_inputs=args.num_calib_inputs,
        use_synthetic=args.use_synthetic,
        cache=args.cache,
        download_dir=args.download_dir)

    def print_dict(input_dict, str=''):
        for k, v in sorted(input_dict.items()):
            headline = '{}({}): '.format(str, k) if str else '{}: '.format(k)
            print('{}{}'.format(headline, '%.1f'%v if type(v)==float else v))
    print_dict(vars(args))
    print_dict(num_nodes, str='num_nodes')
    print_dict(times, str='time(s)')

    # Evaluate model
    print('running inference...')
    results = run(
        frozen_graph,
        model=args.model,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_iterations=args.num_iterations,
        num_warmup_iterations=args.num_warmup_iterations,
        use_synthetic=args.use_synthetic,
        display_every=args.display_every)

    # Display results
    print('results of {}:'.format(args.model))
    print('    accuracy: %.2f' % (results['accuracy'] * 100))
    print('    images/sec: %d' % results['images_per_sec'])
    print('    99th_percentile(ms): %.1f' % results['99th_percentile'])
    print('    total_time(s): %.1f' % results['total_time'])
    print('    latency_mean(ms): %.1f' % results['latency_mean'])
