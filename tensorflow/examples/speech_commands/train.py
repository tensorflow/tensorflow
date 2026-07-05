# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
r"""Simple speech recognition to spot a limited number of keywords.

This is a self-contained example script that will train a very basic audio
recognition model in TensorFlow. It downloads the necessary training data and
runs with reasonable defaults to train within a few hours even only using a CPU.
For more information, please see
https://www.tensorflow.org/tutorials/audio/simple_audio.

It is intended as an introduction to using neural networks for audio
recognition, and is not a full speech recognition system. For more advanced
speech systems, I recommend looking into Kaldi. This network uses a keyword
detection style to spot discrete words from a small vocabulary, consisting of
"yes", "no", "up", "down", "left", "right", "on", "off", "stop", and "go".

To run the training process, use:

bazel run tensorflow/examples/speech_commands:train

This will write out checkpoints to /tmp/speech_commands_train/, and will
download over 1GB of open source training data, so you'll need enough free space
and a good internet connection. The default data is a collection of thousands of
one-second .wav files, each containing one spoken word. Learn more at
https://blog.research.google/2017/08/launching-speech-commands-dataset.html.

As training progresses, it will print out its accuracy metrics, which should
rise above 90% by the end. Once it's complete, you can run the freeze script to
get a binary GraphDef that you can easily deploy on mobile applications.

If you want to train on your own data, you'll need to create .wavs with your
recordings, all at a consistent length, and then arrange them into subfolders
organized by label. For example, here's a possible file structure:

my_wavs >
  up >
    audio_0.wav
    audio_1.wav
  down >
    audio_2.wav
    audio_3.wav
  other>
    audio_4.wav
    audio_5.wav

You'll also need to tell the script what labels to look for, using the
`--wanted_words` argument. In this case, 'up,down' might be what you want, and
the audio in the 'other' folder would be used to train an 'unknown' category.

To pull this all together, you'd run:

bazel run tensorflow/examples/speech_commands:train -- \
--data_dir=my_wavs --wanted_words=up,down

"""

import argparse
import logging
import os.path

import numpy as np
import tensorflow as tf

import input_data
import models
from tensorflow.python.platform import gfile

FLAGS = None


def main(_):
  # Set the verbosity based on flags (default is INFO, so we see all messages).
  tf.get_logger().setLevel(FLAGS.verbosity)

  if FLAGS.check_nans:
    tf.debugging.enable_check_numerics()

  if FLAGS.quantize:
    # TODO: Quantization-aware training doesn't have a TF2 story yet in this
    # example. Since deployment targets TFLite, the intended follow-up is to
    # train a float model as usual and then apply
    # tf.lite.TFLiteConverter post-training quantization (or, if that isn't
    # accurate enough, quantization-aware training via
    # tensorflow_model_optimization) at export time -- see the matching TODO
    # in freeze.py for where that conversion would happen.
    raise Exception(
        '--quantize is not yet supported for TF2 training. See the TODO '
        'comment above main() in train.py for the intended follow-up '
        '(TFLite post-training quantization).')

  # Begin by making sure we have the training data we need. If you already have
  # training data of your own, use `--data_url= ` on the command line to avoid
  # downloading.
  model_settings = models.prepare_model_settings(
      len(input_data.prepare_words_list(FLAGS.wanted_words.split(','))),
      FLAGS.sample_rate, FLAGS.clip_duration_ms, FLAGS.window_size_ms,
      FLAGS.window_stride_ms, FLAGS.feature_bin_count, FLAGS.preprocess)
  audio_processor = input_data.AudioProcessor(
      FLAGS.data_url, FLAGS.data_dir, FLAGS.silence_percentage,
      FLAGS.unknown_percentage, FLAGS.wanted_words.split(','),
      FLAGS.validation_percentage, FLAGS.testing_percentage, model_settings,
      FLAGS.summaries_dir)
  label_count = model_settings['label_count']
  time_shift_samples = int((FLAGS.time_shift_ms * FLAGS.sample_rate) / 1000)
  # Figure out the learning rates for each training phase. Since it's often
  # effective to have high learning rates at the start of training, followed by
  # lower levels towards the end, the number of steps and learning rates can be
  # specified as comma-separated lists to define the rate at each stage. For
  # example --how_many_training_steps=10000,3000 --learning_rate=0.001,0.0001
  # will run 13,000 training loops in total, with a rate of 0.001 for the first
  # 10,000, and 0.0001 for the final 3,000.
  training_steps_list = list(map(int, FLAGS.how_many_training_steps.split(',')))
  learning_rates_list = list(map(float, FLAGS.learning_rate.split(',')))
  if len(training_steps_list) != len(learning_rates_list):
    raise Exception(
        '--how_many_training_steps and --learning_rate must be equal length '
        'lists, but are %d and %d long instead' %
        (len(training_steps_list), len(learning_rates_list)))

  # A PiecewiseConstantDecay schedule reproduces the staged learning rates
  # directly in the optimizer.
  step_boundaries = np.cumsum(training_steps_list)[:-1].tolist()
  lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
      boundaries=step_boundaries, values=learning_rates_list)

  model = models.create_model(model_settings, FLAGS.model_architecture)

  if FLAGS.optimizer == 'gradient_descent':
    optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule)
  elif FLAGS.optimizer == 'momentum':
    optimizer = tf.keras.optimizers.SGD(
        learning_rate=lr_schedule, momentum=0.9, nesterov=True)
  else:
    raise Exception('Invalid Optimizer')

  # The models output raw logits, so from_logits=True.
  loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

  train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
      name='train_accuracy')
  val_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
      name='val_accuracy')
  test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
      name='test_accuracy')

  train_writer = tf.summary.create_file_writer(
      os.path.join(FLAGS.summaries_dir, 'train'))
  val_writer = tf.summary.create_file_writer(
      os.path.join(FLAGS.summaries_dir, 'validation'))

  checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
  checkpoint_manager = tf.train.CheckpointManager(
      checkpoint, directory=FLAGS.train_dir, max_to_keep=5)

  start_step = 1
  if FLAGS.start_checkpoint:
    checkpoint.restore(FLAGS.start_checkpoint)
    # The optimizer's iteration count tells us how many steps were already
    # trained.
    start_step = optimizer.iterations.numpy() + 1
    tf.get_logger().info(
        'Restored from checkpoint. Training from step: %d' % start_step)

  # Save list of words.
  with gfile.GFile(
      os.path.join(FLAGS.train_dir, FLAGS.model_architecture + '_labels.txt'),
      'w') as f:
    f.write('\n'.join(audio_processor.words_list))

  @tf.function
  def train_step(fingerprints, ground_truth):
    with tf.GradientTape() as tape:
      # Pass training=True so layers like Dropout behave correctly.
      logits = model(fingerprints, training=True)
      loss = loss_fn(ground_truth, logits)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_accuracy.update_state(ground_truth, logits)
    return loss

  @tf.function
  def eval_step(fingerprints, ground_truth, accuracy_metric):
    logits = model(fingerprints, training=False)
    accuracy_metric.update_state(ground_truth, logits)
    return tf.argmax(logits, axis=1)

  # Training loop.
  training_steps_max = np.sum(training_steps_list)
  for training_step in range(start_step, training_steps_max + 1):
    # Figure out what the current learning rate is, for logging purposes.
    learning_rate_value = lr_schedule(optimizer.iterations)

    # Pull the audio samples we'll use for training.
    train_fingerprints, train_ground_truth = audio_processor.get_data(
        FLAGS.batch_size, 0, model_settings, FLAGS.background_frequency,
        FLAGS.background_volume, time_shift_samples, 'training',
        step=training_step)

    loss_value = train_step(train_fingerprints, train_ground_truth)

    with train_writer.as_default():
      tf.summary.scalar('cross_entropy', loss_value, step=training_step)
      tf.summary.scalar(
          'accuracy', train_accuracy.result(), step=training_step)
      tf.summary.scalar(
          'learning_rate', learning_rate_value, step=training_step)

    tf.get_logger().debug(
        'Step #%d: rate %f, accuracy %.1f%%, cross entropy %f' %
        (training_step, learning_rate_value, train_accuracy.result() * 100,
         loss_value))

    is_last_step = (training_step == training_steps_max)
    if (training_step % FLAGS.eval_step_interval) == 0 or is_last_step:
      tf.get_logger().info(
          'Step #%d: rate %f, accuracy %.1f%%, cross entropy %f' %
          (training_step, learning_rate_value, train_accuracy.result() * 100,
           loss_value))

      val_accuracy.reset_states()
      set_size = audio_processor.set_size('validation')
      total_conf_matrix = np.zeros((label_count, label_count), dtype=np.int32)
      for i in range(0, set_size, FLAGS.batch_size):
        val_fingerprints, val_ground_truth = audio_processor.get_data(
            FLAGS.batch_size, i, model_settings, 0.0, 0.0, 0, 'validation')
        predictions = eval_step(val_fingerprints, val_ground_truth,
                                val_accuracy)
        batch_conf_matrix = tf.math.confusion_matrix(
            labels=val_ground_truth, predictions=predictions,
            num_classes=label_count).numpy()
        total_conf_matrix += batch_conf_matrix

      with val_writer.as_default():
        tf.summary.scalar(
            'accuracy', val_accuracy.result(), step=training_step)

      tf.get_logger().info('Confusion Matrix:\n %s' % (total_conf_matrix,))
      tf.get_logger().info(
          'Step %d: Validation accuracy = %.1f%% (N=%d)' %
          (training_step, val_accuracy.result() * 100, set_size))

    # Save the model checkpoint periodically.
    if (training_step % FLAGS.save_step_interval == 0 or
        training_step == training_steps_max):
      save_path = checkpoint_manager.save(checkpoint_number=training_step)
      tf.get_logger().info('Saved checkpoint to "%s"' % save_path)

    train_accuracy.reset_states()

  set_size = audio_processor.set_size('testing')
  tf.get_logger().info('set_size=%d' % set_size)
  test_accuracy.reset_states()
  total_conf_matrix = np.zeros((label_count, label_count), dtype=np.int32)
  for i in range(0, set_size, FLAGS.batch_size):
    test_fingerprints, test_ground_truth = audio_processor.get_data(
        FLAGS.batch_size, i, model_settings, 0.0, 0.0, 0, 'testing')
    predictions = eval_step(test_fingerprints, test_ground_truth,
                            test_accuracy)
    batch_conf_matrix = tf.math.confusion_matrix(
        labels=test_ground_truth, predictions=predictions,
        num_classes=label_count).numpy()
    total_conf_matrix += batch_conf_matrix
  tf.get_logger().warning('Confusion Matrix:\n %s' % (total_conf_matrix,))
  tf.get_logger().warning(
      'Final test accuracy = %.1f%% (N=%d)' %
      (test_accuracy.result() * 100, set_size))


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--data_url',
      type=str,
      # pylint: disable=line-too-long
      default='https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz',
      # pylint: enable=line-too-long
      help='Location of speech training data archive on the web.')
  parser.add_argument(
      '--data_dir',
      type=str,
      default='/tmp/speech_dataset/',
      help="""\
      Where to download the speech training data to.
      """)
  parser.add_argument(
      '--background_volume',
      type=float,
      default=0.1,
      help="""\
      How loud the background noise should be, between 0 and 1.
      """)
  parser.add_argument(
      '--background_frequency',
      type=float,
      default=0.8,
      help="""\
      How many of the training samples have background noise mixed in.
      """)
  parser.add_argument(
      '--silence_percentage',
      type=float,
      default=10.0,
      help="""\
      How much of the training data should be silence.
      """)
  parser.add_argument(
      '--unknown_percentage',
      type=float,
      default=10.0,
      help="""\
      How much of the training data should be unknown words.
      """)
  parser.add_argument(
      '--time_shift_ms',
      type=float,
      default=100.0,
      help="""\
      Range to randomly shift the training audio by in time.
      """)
  parser.add_argument(
      '--testing_percentage',
      type=int,
      default=10,
      help='What percentage of wavs to use as a test set.')
  parser.add_argument(
      '--validation_percentage',
      type=int,
      default=10,
      help='What percentage of wavs to use as a validation set.')
  parser.add_argument(
      '--sample_rate',
      type=int,
      default=16000,
      help='Expected sample rate of the wavs',)
  parser.add_argument(
      '--clip_duration_ms',
      type=int,
      default=1000,
      help='Expected duration in milliseconds of the wavs',)
  parser.add_argument(
      '--window_size_ms',
      type=float,
      default=30.0,
      help='How long each spectrogram timeslice is.',)
  parser.add_argument(
      '--window_stride_ms',
      type=float,
      default=10.0,
      help='How far to move in time between spectrogram timeslices.',
  )
  parser.add_argument(
      '--feature_bin_count',
      type=int,
      default=40,
      help='How many bins to use for the MFCC fingerprint',
  )
  parser.add_argument(
      '--how_many_training_steps',
      type=str,
      default='15000,3000',
      help='How many training loops to run',)
  parser.add_argument(
      '--eval_step_interval',
      type=int,
      default=400,
      help='How often to evaluate the training results.')
  parser.add_argument(
      '--learning_rate',
      type=str,
      default='0.001,0.0001',
      help='How large a learning rate to use when training.')
  parser.add_argument(
      '--batch_size',
      type=int,
      default=100,
      help='How many items to train with at once',)
  parser.add_argument(
      '--summaries_dir',
      type=str,
      default='/tmp/retrain_logs',
      help='Where to save summary logs for TensorBoard.')
  parser.add_argument(
      '--wanted_words',
      type=str,
      default='yes,no,up,down,left,right,on,off,stop,go',
      help='Words to use (others will be added to an unknown label)',)
  parser.add_argument(
      '--train_dir',
      type=str,
      default='/tmp/speech_commands_train',
      help='Directory to write event logs and checkpoint.')
  parser.add_argument(
      '--save_step_interval',
      type=int,
      default=100,
      help='Save model checkpoint every save_steps.')
  parser.add_argument(
      '--start_checkpoint',
      type=str,
      default='',
      help='If specified, restore this pretrained model before any training.')
  parser.add_argument(
      '--model_architecture',
      type=str,
      default='conv',
      help='What model architecture to use')
  parser.add_argument(
      '--check_nans',
      type=bool,
      default=False,
      help='Whether to check for invalid numbers during processing')
  parser.add_argument(
      '--quantize',
      type=bool,
      default=False,
      help='Whether to train the model for eight-bit deployment')
  parser.add_argument(
      '--preprocess',
      type=str,
      default='mfcc',
      help='Spectrogram processing mode. Can be "mfcc", "average", or "micro"')

  # Function used to parse --verbosity argument.
  def verbosity_arg(value):
    """Parses verbosity argument.

    Args:
      value: A member of logging.
    Raises:
      ArgumentTypeError: Not an expected value.
    """
    value = value.upper()
    if value == 'DEBUG':
      return logging.DEBUG
    elif value == 'INFO':
      return logging.INFO
    elif value == 'WARN':
      return logging.WARN
    elif value == 'ERROR':
      return logging.ERROR
    elif value == 'FATAL':
      return logging.CRITICAL
    else:
      raise argparse.ArgumentTypeError('Not an expected value')

  parser.add_argument(
      '--verbosity',
      type=verbosity_arg,
      default=logging.INFO,
      help='Log verbosity. Can be "DEBUG", "INFO", "WARN", "ERROR", or "FATAL"')
  parser.add_argument(
      '--optimizer',
      type=str,
      default='gradient_descent',
      help='Optimizer (gradient_descent or momentum)')

  # parse_known_args (rather than parse_args) tolerates extra flags that test
  # runners like bazel sometimes inject; main() doesn't look at argv, so any
  # leftovers are simply unused, matching the previous tf.compat.v1.app.run
  # behavior.
  FLAGS, _ = parser.parse_known_args()
  main(None)
