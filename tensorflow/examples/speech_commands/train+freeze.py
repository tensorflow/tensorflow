'''
Code to train and create frozen graph for your speech model together
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os.path
import sys

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

import input_data
import models
from tensorflow.python.platform import gfile
from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio
from tensorflow.python.framework import graph_util

FLAGS = None

def create_inference_graph(wanted_words, sample_rate, clip_duration_ms,
                           clip_stride_ms, window_size_ms, window_stride_ms,
                           feature_bin_count, model_architecture, preprocess):
  """Creates an audio model with the nodes needed for inference.

  Uses the supplied arguments to create a model, and inserts the input and
  output nodes that are needed to use the graph for inference.

  Args:
    wanted_words: Comma-separated list of the words we're trying to recognize.
    sample_rate: How many samples per second are in the input audio files.
    clip_duration_ms: How many samples to analyze for the audio pattern.
    clip_stride_ms: How often to run recognition. Useful for models with cache.
    window_size_ms: Time slice duration to estimate frequencies from.
    window_stride_ms: How far apart time slices should be.
    feature_bin_count: Number of frequency bands to analyze.
    model_architecture: Name of the kind of model to generate.
    preprocess: How the spectrogram is processed to produce features, for
      example 'mfcc' or 'average'.

  Raises:
    Exception: If the preprocessing mode isn't recognized.
  """

  words_list = input_data.prepare_words_list(wanted_words.split(','))
  model_settings = models.prepare_model_settings(
      len(words_list), sample_rate, clip_duration_ms, window_size_ms,
      window_stride_ms, feature_bin_count, preprocess)
  runtime_settings = {'clip_stride_ms': clip_stride_ms}

  wav_data_placeholder = tf.placeholder(tf.string, [], name='wav_data')
  decoded_sample_data = contrib_audio.decode_wav(
      wav_data_placeholder,
      desired_channels=1,
      desired_samples=model_settings['desired_samples'],
      name='decoded_sample_data')
  spectrogram = contrib_audio.audio_spectrogram(
      decoded_sample_data.audio,
      window_size=model_settings['window_size_samples'],
      stride=model_settings['window_stride_samples'],
      magnitude_squared=True)

  if preprocess == 'average':
    fingerprint_input = tf.nn.pool(
        tf.expand_dims(spectrogram, -1),
        window_shape=[1, model_settings['average_window_width']],
        strides=[1, model_settings['average_window_width']],
        pooling_type='AVG',
        padding='SAME')
  elif preprocess == 'mfcc':
    fingerprint_input = contrib_audio.mfcc(
        spectrogram,
        sample_rate,
        dct_coefficient_count=model_settings['fingerprint_width'])
  else:
    raise Exception('Unknown preprocess mode "%s" (should be "mfcc" or'
                    ' "average")' % (preprocess))

  fingerprint_size = model_settings['fingerprint_size']
  reshaped_input = tf.reshape(fingerprint_input, [-1, fingerprint_size])

  logits = models.create_model(
      reshaped_input, model_settings, model_architecture, is_training=False,
      runtime_settings=runtime_settings)

  # Create an output to use for inference.
  tf.nn.softmax(logits, name='labels_softmax')



def main(_):
  # We want to see all the logging messages for this tutorial.
  tf.logging.set_verbosity(tf.logging.INFO)

  # Start a new TensorFlow session.
  sess = tf.InteractiveSession()

  # Begin by making sure we have the training data we need. If you already have
  # training data of your own, use `--data_url= ` on the command line to avoid
  # downloading.
  model_settings = models.prepare_model_settings(
      len(input_data.prepare_words_list(FLAGS.wanted_words.split(','))),
      FLAGS.sample_rate, FLAGS.clip_duration_ms, FLAGS.window_size_ms,
      FLAGS.window_stride_ms, FLAGS.feature_bin_count, FLAGS.preprocess)
  audio_processor = input_data.AudioProcessor(
      FLAGS.data_url, FLAGS.data_dir,
      FLAGS.silence_percentage, FLAGS.unknown_percentage,
      FLAGS.wanted_words.split(','), FLAGS.validation_percentage,
      FLAGS.testing_percentage, model_settings, FLAGS.summaries_dir)
  fingerprint_size = model_settings['fingerprint_size']
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
        'lists, but are %d and %d long instead' % (len(training_steps_list),
                                                   len(learning_rates_list)))

  input_placeholder = tf.placeholder(
      tf.float32, [None, fingerprint_size], name='fingerprint_input')
  if FLAGS.quantize:
    # TODO(petewarden): These values have been derived from the observed ranges
    # of spectrogram and MFCC inputs. If the preprocessing pipeline changes,
    # they may need to be updated.
    if FLAGS.preprocess == 'average':
      fingerprint_min = 0.0
      fingerprint_max = 2048.0
    elif FLAGS.preprocess == 'mfcc':
      fingerprint_min = -247.0
      fingerprint_max = 30.0
    else:
      raise Exception('Unknown preprocess mode "%s" (should be "mfcc" or'
                      ' "average")' % (FLAGS.preprocess))
    fingerprint_input = tf.fake_quant_with_min_max_args(
        input_placeholder, fingerprint_min, fingerprint_max)
  else:
    fingerprint_input = input_placeholder

  logits, dropout_prob = models.create_model(
      fingerprint_input,
      model_settings,
      FLAGS.model_architecture,
      is_training=True)

  # Define loss and optimizer
  ground_truth_input = tf.placeholder(
      tf.int64, [None], name='groundtruth_input')

  # Optionally we can add runtime checks to spot when NaNs or other symptoms of
  # numerical errors start occurring during training.
  control_dependencies = []
  if FLAGS.check_nans:
    checks = tf.add_check_numerics_ops()
    control_dependencies = [checks]

  # Create the back propagation and training evaluation machinery in the graph.
  with tf.name_scope('cross_entropy'):
    cross_entropy_mean = tf.losses.sparse_softmax_cross_entropy(
        labels=ground_truth_input, logits=logits)
  if FLAGS.quantize:
    tf.contrib.quantize.create_training_graph(quant_delay=0)
  with tf.name_scope('train'), tf.control_dependencies(control_dependencies):
    learning_rate_input = tf.placeholder(
        tf.float32, [], name='learning_rate_input')
    train_step = tf.train.GradientDescentOptimizer(
        learning_rate_input).minimize(cross_entropy_mean)
  predicted_indices = tf.argmax(logits, 1)
  correct_prediction = tf.equal(predicted_indices, ground_truth_input)
  confusion_matrix = tf.confusion_matrix(
      ground_truth_input, predicted_indices, num_classes=label_count)
  evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  with tf.get_default_graph().name_scope('eval'):
    tf.summary.scalar('cross_entropy', cross_entropy_mean)
    tf.summary.scalar('accuracy', evaluation_step)

  global_step = tf.train.get_or_create_global_step()
  increment_global_step = tf.assign(global_step, global_step + 1)

  saver = tf.train.Saver(tf.global_variables())

  # Merge all the summaries and write them out to /tmp/retrain_logs (by default)
  merged_summaries = tf.summary.merge_all(scope='eval')
  train_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/train',
                                       sess.graph)
  validation_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/validation')

  tf.global_variables_initializer().run()

  start_step = 1

  if FLAGS.start_checkpoint:
    models.load_variables_from_checkpoint(sess, FLAGS.start_checkpoint)
    start_step = global_step.eval(session=sess)

  tf.logging.info('Training from step: %d ', start_step)

  # Save graph.pbtxt.
  tf.train.write_graph(sess.graph_def, FLAGS.train_dir,
                       FLAGS.model_architecture + '.pbtxt')

  # Save list of words.
  with gfile.GFile(
      os.path.join(FLAGS.train_dir, FLAGS.model_architecture + '_labels.txt'),
      'w') as f:
    f.write('\n'.join(audio_processor.words_list))

  # Training loop.
  training_steps_max = np.sum(training_steps_list)
  for training_step in xrange(start_step, training_steps_max + 1):
    # Figure out what the current learning rate is.
    training_steps_sum = 0
    for i in range(len(training_steps_list)):
      training_steps_sum += training_steps_list[i]
      if training_step <= training_steps_sum:
        learning_rate_value = learning_rates_list[i]
        break
    # Pull the audio samples we'll use for training.
    train_fingerprints, train_ground_truth = audio_processor.get_data(
        FLAGS.batch_size, 0, model_settings, FLAGS.background_frequency,
        FLAGS.background_volume, time_shift_samples, 'training', sess)
    # Run the graph with this batch of training data.
    train_summary, train_accuracy, cross_entropy_value, _, _ = sess.run(
        [
            merged_summaries,
            evaluation_step,
            cross_entropy_mean,
            train_step,
            increment_global_step,
        ],
        feed_dict={
            fingerprint_input: train_fingerprints,
            ground_truth_input: train_ground_truth,
            learning_rate_input: learning_rate_value,
            dropout_prob: 0.5
        })
    train_writer.add_summary(train_summary, training_step)
    tf.logging.info('Step #%d: rate %f, accuracy %.1f%%, cross entropy %f' %
                    (training_step, learning_rate_value, train_accuracy * 100,
                     cross_entropy_value))
    is_last_step = (training_step == training_steps_max)
    if (training_step % FLAGS.eval_step_interval) == 0 or is_last_step:
      set_size = audio_processor.set_size('validation')
      total_accuracy = 0
      total_conf_matrix = None
      for i in xrange(0, set_size, FLAGS.batch_size):
        validation_fingerprints, validation_ground_truth = (
            audio_processor.get_data(FLAGS.batch_size, i, model_settings, 0.0,
                                     0.0, 0, 'validation', sess))
        # Run a validation step and capture training summaries for TensorBoard
        # with the `merged` op.
        validation_summary, validation_accuracy, conf_matrix = sess.run(
            [merged_summaries, evaluation_step, confusion_matrix],
            feed_dict={
                fingerprint_input: validation_fingerprints,
                ground_truth_input: validation_ground_truth,
                dropout_prob: 1.0
            })
        validation_writer.add_summary(validation_summary, training_step)
        batch_size = min(FLAGS.batch_size, set_size - i)
        total_accuracy += (validation_accuracy * batch_size) / set_size
        if total_conf_matrix is None:
          total_conf_matrix = conf_matrix
        else:
          total_conf_matrix += conf_matrix
      tf.logging.info('Confusion Matrix:\n %s' % (total_conf_matrix))
      tf.logging.info('Step %d: Validation accuracy = %.1f%% (N=%d)' %
                      (training_step, total_accuracy * 100, set_size))

    # Save the model checkpoint periodically.
    if (training_step % FLAGS.save_step_interval == 0 or
        training_step == training_steps_max):
      checkpoint_path = os.path.join(FLAGS.train_dir,
                                     FLAGS.model_architecture + '.ckpt')
      tf.logging.info('Saving to "%s-%d"', checkpoint_path, training_step)
      saver.save(sess, checkpoint_path, global_step=training_step)

  set_size = audio_processor.set_size('testing')
  tf.logging.info('set_size=%d', set_size)
  total_accuracy = 0
  total_conf_matrix = None
  for i in xrange(0, set_size, FLAGS.batch_size):
    test_fingerprints, test_ground_truth = audio_processor.get_data(
        FLAGS.batch_size, i, model_settings, 0.0, 0.0, 0, 'testing', sess)
    test_accuracy, conf_matrix = sess.run(
        [evaluation_step, confusion_matrix],
        feed_dict={
            fingerprint_input: test_fingerprints,
            ground_truth_input: test_ground_truth,
            dropout_prob: 1.0
        })
    batch_size = min(FLAGS.batch_size, set_size - i)
    total_accuracy += (test_accuracy * batch_size) / set_size
    if total_conf_matrix is None:
      total_conf_matrix = conf_matrix
    else:
      total_conf_matrix += conf_matrix
  tf.logging.info('Confusion Matrix:\n %s' % (total_conf_matrix))
  tf.logging.info('Final test accuracy = %.1f%% (N=%d)' % (total_accuracy * 100,
                                                           set_size))

  # Create the model and load its weights.
  sess = tf.InteractiveSession()
  create_inference_graph(
      FLAGS.wanted_words, FLAGS.sample_rate, FLAGS.clip_duration_ms,
      FLAGS.clip_stride_ms, FLAGS.window_size_ms, FLAGS.window_stride_ms,
      FLAGS.feature_bin_count, FLAGS.model_architecture, FLAGS.preprocess)
  if FLAGS.quantize:
    tf.contrib.quantize.create_eval_graph()
  models.load_variables_from_checkpoint(sess, FLAGS.start_checkpoint)

  # Turn all the variables into inline constants inside the graph and save it.
  frozen_graph_def = graph_util.convert_variables_to_constants(
      sess, sess.graph_def, ['labels_softmax'])
  tf.train.write_graph(
      frozen_graph_def,
      os.path.dirname(FLAGS.output_file),
      os.path.basename(FLAGS.output_file),
      as_text=False)
  tf.logging.info('Saved frozen graph to %s', FLAGS.output_file)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()

  #train.py arguments
  parser.add_argument(
      '--data_url',
      type=str,
      # pylint: disable=line-too-long
      default='',
      # pylint: enable=line-too-long
      help='Location of speech training data archive on the web.')
  parser.add_argument(
      '--data_dir',
      type=str,
      default='/home/darpan/Documents/tensorflow-master/tensorflow/examples/speech_commands/dataset/',
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
  # parser.add_argument(
  #     '--sample_rate',
  #     type=int,
  #     default=16000,
  #     help='Expected sample rate of the wavs',)
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
      help='How far to move in time between spectogram timeslices.',)
  parser.add_argument(
      '--feature_bin_count',
      type=int,
      default=40,
      help='How many bins to use for the MFCC fingerprint',
  )
  parser.add_argument(
      '--how_many_training_steps',
      type=str,
      default='1500,500',
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
      default='/home/darpan/Documents/tensorflow-master/tensorflow/examples/speech_commands/retrain_logs/',
      help='Where to save summary logs for TensorBoard.')
  parser.add_argument(
      '--wanted_words',
      type=str,
      default='laddu,modak,pedha',
      help='Words to use (others will be added to an unknown label)',)
  parser.add_argument(
      '--train_dir',
      type=str,
      default='/home/darpan/Documents/tensorflow-master/tensorflow/examples/speech_commands/train_dir/',
      help='Directory to write event logs and checkpoint.')
  parser.add_argument(
      '--save_step_interval',
      type=int,
      default=200,
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
      help='Spectrogram processing mode. Can be "mfcc" or "average"')
  parser.add_argument(
        '--sample_rate',
        type=int,
        default=44100,
        help='Expected sample rate of the wavs',)
  # parser.add_argument(
  #       '--clip_duration_ms',
  #       type=int,
  #       default=1000,
  #       help='Expected duration in milliseconds of the wavs',)
  parser.add_argument(
        '--clip_stride_ms',
        type=int,
        default=30,
        help='How often to run recognition. Useful for models with cache.',)
  # parser.add_argument(
  #       '--window_size_ms',
  #       type=float,
  #       default=30.0,
  #       help='How long each spectrogram timeslice is',)
  # parser.add_argument(
  #       '--window_stride_ms',
  #       type=float,
  #       default=10.0,
  #       help='How long the stride is between spectrogram timeslices',)
  # parser.add_argument(
  #       '--feature_bin_count',
  #       type=int,
  #       default=40,
  #       help='How many bins to use for the MFCC fingerprint',)
  # parser.add_argument(
  #       '--start_checkpoint',
  #       type=str,
  #       default='',
  #       help='If specified, restore this pretrained model before any training.')
  # parser.add_argument(
  #       '--model_architecture',
  #       type=str,
  #       default='conv',
  #       help='What model architecture to use')

  # parser.add_argument(
  #       '--wanted_words',
  #       type=str,
  #       default='modak, laddu, pedha',
  #       help='Words to use (others will be added to an unknown label)',)

  parser.add_argument(
        '--output_file', default='/home/darpan/Documents/tensorflow-master/tensorflow/examples/speech_commands/my_frozen_graph.pb',
        type=str, help='Where to save the frozen graph.')

  # parser.add_argument(
  #       '--quantize',
  #       type=bool,
  #       default=False,
  #       help='Whether to train the model for eight-bit deployment')
  # parser.add_argument(
  #       '--preprocess',
  #       type=str,
  #       default='mfcc',
  #       help='Spectrogram processing mode. Can be "mfcc" or "average"')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
