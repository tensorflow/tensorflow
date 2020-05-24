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
"""Generates Android Java sources from a TFLite model with metadata."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil
from absl import app
from absl import flags
from absl import logging

from tensorflow.lite.experimental.support.codegen.python import _pywrap_codegen

FLAGS = flags.FLAGS

flags.DEFINE_string('model', None, 'Path to model (.tflite) flatbuffer file.')
flags.DEFINE_string('destination', None, 'Path of destination of generation.')
flags.DEFINE_string('package_name', 'org.tensorflow.lite.support',
                    'Name of generated java package to put the wrapper class.')
flags.DEFINE_string(
    'model_class_name', 'MyModel',
    'Name of generated wrapper class (should not contain package name).')
flags.DEFINE_string(
    'model_asset_path', '',
    '(Optional) Path to the model in generated assets/ dir. If not set, '
    'generator will use base name of input model.'
)


def get_model_buffer(path):
  if not os.path.isfile(path):
    logging.error('Cannot find model at path %s.', path)
  with open(path, 'rb') as f:
    buf = f.read()
    return buf


def prepare_directory_for_file(file_path):
  target_dir = os.path.dirname(file_path)
  if not os.path.exists(target_dir):
    os.makedirs(target_dir)
    return
  if not os.path.isdir(target_dir):
    logging.error('Cannot write to %s', target_dir)


def main(argv):
  if len(argv) > 1:
    logging.error('None flag arguments found: [%s]', ', '.join(argv[1:]))

  codegen = _pywrap_codegen.AndroidJavaGenerator(FLAGS.destination)
  model_buffer = get_model_buffer(FLAGS.model)
  model_asset_path = FLAGS.model_asset_path
  if not model_asset_path:
    model_asset_path = os.path.basename(FLAGS.model)
  result = codegen.generate(model_buffer, FLAGS.package_name,
                            FLAGS.model_class_name, model_asset_path)
  error_message = codegen.get_error_message().strip()
  if error_message:
    logging.error(error_message)
  if not result.files:
    logging.error('Generation failed!')
    return

  for each in result.files:
    prepare_directory_for_file(each.path)
    with open(each.path, 'w') as f:
      f.write(each.content)

  logging.info('Generation succeeded!')
  model_asset_path = os.path.join(FLAGS.destination, 'src/main/assets',
                                  model_asset_path)
  prepare_directory_for_file(model_asset_path)
  shutil.copy(FLAGS.model, model_asset_path)
  logging.info('Model copied into assets!')


if __name__ == '__main__':
  flags.mark_flag_as_required('model')
  flags.mark_flag_as_required('destination')
  app.run(main)
