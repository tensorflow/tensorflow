# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
r"""Tool to create accuracy statistics on a continuous stream of samples.

This is designed to be an environment for running experiments on new models and
settings to understand the effects they will have in a real application. You
need to supply it with a long audio file containing sounds you want to recognize
and a text file listing the labels of each sound along with the time they occur.
With this information, and a frozen model, the tool will process the audio
stream, apply the model, and keep track of how many mistakes and successes the
model achieved.

The matched percentage is the number of sounds that were correctly classified,
as a percentage of the total number of sounds listed in the ground truth file.
A correct classification is when the right label is chosen within a short time
of the expected ground truth, where the time tolerance is controlled by the
'time_tolerance_ms' command line flag.

The wrong percentage is how many sounds triggered a detection (the classifier
figured out it wasn't silence or background noise), but the detected class was
wrong. This is also a percentage of the total number of ground truth sounds.

The false positive percentage is how many sounds were detected when there was
only silence or background noise. This is also expressed as a percentage of the
total number of ground truth sounds, though since it can be large it may go
above 100%.

The easiest way to get an audio file and labels to test with is by using the
'generate_streaming_test_wav' script. This will synthesize a test file with
randomly placed sounds and background noise, and output a text file with the
ground truth.

If you want to test natural data, you need to use a .wav with the same sample
rate as your model (often 16,000 samples per second), and note down where the
sounds occur in time. Save this information out as a comma-separated text file,
where the first column is the label and the second is the time in seconds from
the start of the file that it occurs.

Here's an example of how to run the tool:

bazel run tensorflow/examples/speech_commands:test_streaming_accuracy_py -- \
--wav=/tmp/streaming_test_bg.wav \
--ground-truth=/tmp/streaming_test_labels.txt --verbose \
--model=/tmp/conv_frozen.pb \
--labels=/tmp/speech_commands_train/conv_labels.txt \
--clip_duration_ms=1000 --detection_threshold=0.70 --average_window_ms=500 \
--suppression_ms=500 --time_tolerance_ms=1500
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

import numpy
import tensorflow as tf

from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio
from tensorflow.examples.speech_commands.accuracy_utils import StreamingAccuracyStats
from tensorflow.examples.speech_commands.recognize_commands import RecognizeCommands
from tensorflow.examples.speech_commands.recognize_commands import RecognizeResult
from tensorflow.python.ops import io_ops

FLAGS = None


def load_graph(mode_file):
  """Read a tensorflow model, and creates a default graph object."""
  graph = tf.Graph()
  with graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(mode_file, 'rb') as fid:
      serialized_graph = fid.read()
      od_graph_def.ParseFromString(serialized_graph)
      tf.import_graph_def(od_graph_def, name='')
  return graph


def read_label_file(file_name):
  """Load a list of label."""
  label_list = []
  with open(file_name, 'r') as f:
    for line in f:
      label_list.append(line.strip())
  return label_list


def read_wav_file(filename):
  """Load a wav file and return sample_rate and numpy data of float64 type."""
  with tf.Session(graph=tf.Graph()) as sess:
    wav_filename_placeholder = tf.placeholder(tf.string, [])
    wav_loader = io_ops.read_file(wav_filename_placeholder)
    wav_decoder = contrib_audio.decode_wav(wav_loader, desired_channels=1)
    res = sess.run(wav_decoder, feed_dict={wav_filename_placeholder: filename})
  return res.sample_rate, res.audio.flatten()


def main(_):
  label_list = read_label_file(FLAGS.labels)
  sample_rate, data = read_wav_file(FLAGS.wav)
  # Init instance of RecognizeCommands with given parameters.
  recognize_commands = RecognizeCommands(
      labels=label_list,
      average_window_duration_ms=FLAGS.average_window_duration_ms,
      detection_threshold=FLAGS.detection_threshold,
      suppression_ms=FLAGS.suppression_ms,
      minimum_count=4)

  # Init instance of StreamingAccuracyStats and load ground truth.
  stats = StreamingAccuracyStats()
  stats.read_ground_truth_file(FLAGS.ground_truth)
  recognize_element = RecognizeResult()
  all_found_words = []
  data_samples = data.shape[0]
  clip_duration_samples = int(FLAGS.clip_duration_ms * sample_rate / 1000)
  clip_stride_samples = int(FLAGS.clip_stride_ms * sample_rate / 1000)
  audio_data_end = data_samples - clip_duration_samples

  # Load model and create a tf session to process audio pieces
  recognize_graph = load_graph(FLAGS.model)
  with recognize_graph.as_default():
    with tf.Session() as sess:

      # Get input and output tensor
      data_tensor = tf.get_default_graph().get_tensor_by_name(
          FLAGS.input_names[0])
      sample_rate_tensor = tf.get_default_graph().get_tensor_by_name(
          FLAGS.input_names[1])
      output_softmax_tensor = tf.get_default_graph().get_tensor_by_name(
          FLAGS.output_name)

      # Inference along audio stream.
      for audio_data_offset in range(0, audio_data_end, clip_stride_samples):
        input_start = audio_data_offset
        input_end = audio_data_offset + clip_duration_samples
        outputs = sess.run(
            output_softmax_tensor,
            feed_dict={
                data_tensor:
                    numpy.expand_dims(data[input_start:input_end], axis=-1),
                sample_rate_tensor:
                    sample_rate
            })
        outputs = numpy.squeeze(outputs)
        current_time_ms = int(audio_data_offset * 1000 / sample_rate)
        try:
          recognize_commands.process_latest_result(outputs, current_time_ms,
                                                   recognize_element)
        except ValueError as e:
          tf.logging.error('Recognition processing failed: {}' % e)
          return
        if (recognize_element.is_new_command and
            recognize_element.founded_command != '_silence_'):
          all_found_words.append(
              [recognize_element.founded_command, current_time_ms])
          if FLAGS.verbose:
            stats.calculate_accuracy_stats(all_found_words, current_time_ms,
                                           FLAGS.time_tolerance_ms)
            try:
              recognition_state = stats.delta()
            except ValueError as e:
              tf.logging.error(
                  'Statistics delta computing failed: {}'.format(e))
            else:
              tf.logging.info('{}ms {}:{}{}'.format(
                  current_time_ms, recognize_element.founded_command,
                  recognize_element.score, recognition_state))
              stats.print_accuracy_stats()
  stats.calculate_accuracy_stats(all_found_words, -1, FLAGS.time_tolerance_ms)
  stats.print_accuracy_stats()


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='test_streaming_accuracy')
  parser.add_argument(
      '--wav', type=str, default='', help='The wave file path to evaluate.')
  parser.add_argument(
      '--ground-truth',
      type=str,
      default='',
      help='The ground truth file path corresponding to wav file.')
  parser.add_argument(
      '--labels',
      type=str,
      default='',
      help='The label file path containing all possible classes.')
  parser.add_argument(
      '--model', type=str, default='', help='The model used for inference')
  parser.add_argument(
      '--input-names',
      type=str,
      nargs='+',
      default=['decoded_sample_data:0', 'decoded_sample_data:1'],
      help='Input name list involved in model graph.')
  parser.add_argument(
      '--output-name',
      type=str,
      default='labels_softmax:0',
      help='Output name involved in model graph.')
  parser.add_argument(
      '--clip-duration-ms',
      type=int,
      default=1000,
      help='Length of each audio clip fed into model.')
  parser.add_argument(
      '--clip-stride-ms',
      type=int,
      default=30,
      help='Length of audio clip stride over main trap.')
  parser.add_argument(
      '--average_window_duration_ms',
      type=int,
      default=500,
      help='Length of average window used for smoothing results.')
  parser.add_argument(
      '--detection-threshold',
      type=float,
      default=0.7,
      help='The confidence for filtering unreliable commands')
  parser.add_argument(
      '--suppression_ms',
      type=int,
      default=500,
      help='The time interval between every two adjacent commands')
  parser.add_argument(
      '--time-tolerance-ms',
      type=int,
      default=1500,
      help='Time tolerance before and after the timestamp of this audio clip '
      'to match ground truth')
  parser.add_argument(
      '--verbose',
      action='store_true',
      default=False,
      help='Whether to print streaming accuracy on stdout.')

  FLAGS, unparsed = parser.parse_known_args()
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
