# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
r"""Modify a quantized model's interface from float to integer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags

from tensorflow.lite.tools.optimize.python import modify_model_interface_constants as mmi_constants
from tensorflow.lite.tools.optimize.python import modify_model_interface_lib as mmi_lib

FLAGS = flags.FLAGS

flags.DEFINE_string('input_tflite_file', None,
                    'Full path name to the input TFLite file.')
flags.DEFINE_string('output_tflite_file', None,
                    'Full path name to the output TFLite file.')
flags.DEFINE_enum('input_type', mmi_constants.DEFAULT_STR_TYPE,
                  mmi_constants.STR_TYPES,
                  'Modified input integer interface type.')
flags.DEFINE_enum('output_type', mmi_constants.DEFAULT_STR_TYPE,
                  mmi_constants.STR_TYPES,
                  'Modified output integer interface type.')

flags.mark_flag_as_required('input_tflite_file')
flags.mark_flag_as_required('output_tflite_file')


def main(_):
  input_type = mmi_constants.STR_TO_TFLITE_TYPES[FLAGS.input_type]
  output_type = mmi_constants.STR_TO_TFLITE_TYPES[FLAGS.output_type]

  mmi_lib.modify_model_interface(FLAGS.input_file, FLAGS.output_file,
                                 input_type, output_type)

  print('Successfully modified the model input type from FLOAT to '
        '{input_type} and output type from FLOAT to {output_type}.'.format(
            input_type=FLAGS.input_type, output_type=FLAGS.output_type))


if __name__ == '__main__':
  app.run(main)
