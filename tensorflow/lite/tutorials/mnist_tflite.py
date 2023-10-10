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
# ==============================================================================
"""Script to evaluate accuracy of TFLite flatbuffer model on mnist dataset."""

import numpy as np
import tensorflow as tf  # pylint: disable=g-bad-import-order
from tensorflow.lite.tutorials import dataset
flags = tf.app.flags

flags.DEFINE_string('data_dir', '/tmp/data_dir',
                    'Directory where data is stored.')
flags.DEFINE_string('model_file', '',
                    'The path to the TFLite flatbuffer model file.')


flags = flags.FLAGS


def test_image_generator():
  # Generates an iterator over images
  with tf.compat.v1.Session() as sess:
    input_data = tf.compat.v1.data.make_one_shot_iterator(dataset.test(
        flags.data_dir)).get_next()
    try:
      while True:
        yield sess.run(input_data)
    except tf.errors.OutOfRangeError:
      pass


def run_eval(interpreter, input_image):
  """Performs evaluation for input image over specified model.

  Args:
      interpreter: TFLite interpreter initialized with model to execute.
      input_image: Image input to the model.

  Returns:
      output: output tensor of model being executed.
  """

  # Get input and output tensors.
  input_details = interpreter.get_input_details()
  output_details = interpreter.get_output_details()

  # Test model on the input images.
  input_image = np.reshape(input_image, input_details[0]['shape'])
  interpreter.set_tensor(input_details[0]['index'], input_image)

  interpreter.invoke()
  output_data = interpreter.get_tensor(output_details[0]['index'])
  output = np.squeeze(output_data)
  return output


def main(_):
  interpreter = tf.lite.Interpreter(model_path=flags.model_file)
  interpreter.allocate_tensors()
  num_correct, total = 0, 0
  for input_data in test_image_generator():
    output = run_eval(interpreter, input_data[0])
    total += 1
    if output == input_data[1]:
      num_correct += 1
    if total % 500 == 0:
      print('Accuracy after %i images: %f' %
            (total, float(num_correct) / float(total)))


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.compat.v1.app.run(main)
