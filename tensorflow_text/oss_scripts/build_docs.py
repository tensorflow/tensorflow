# coding=utf-8
# Copyright 2025 TF.Text Authors.
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

# Copyright 2024 TF.Text Authors.
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

r"""Tool to generate external api_docs.

python build_docs.py --output_dir=/tmp/text_api
"""
import os

from absl import app
from absl import flags

import tensorflow as tf

from tensorflow_docs.api_generator import doc_controls
from tensorflow_docs.api_generator import generate_lib
from tensorflow_docs.api_generator import public_api

import tensorflow_text as text

PROJECT_SHORT_NAME = "text"
PROJECT_FULL_NAME = "TensorFlow Text"

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "output_dir",
    default="/tmp/text_api",
    help="Where to write the resulting docs to.")
flags.DEFINE_string(
    "code_url_prefix",
    "http://github.com/tensorflow/text/blob/master/tensorflow_text",
    "The url prefix for links to code.")

flags.DEFINE_bool("search_hints", True,
                  "Include metadata search hints in the generated files")

flags.DEFINE_string("site_path", "/text/api_docs/python",
                    "Path prefix in the _toc.yaml")


def _hide_layer_and_module_methods():
  """Hide methods and properties defined in the base classes of keras layers."""
  # __dict__ only sees attributes defined in *this* class, not on parent classes
  # Needed to ignore redudant subclass documentation
  module_contents = list(tf.Module.__dict__.items())
  layer_contents = list(tf.keras.layers.Layer.__dict__.items())

  for name, obj in module_contents + layer_contents:
    if name == "__init__":
      continue

    if isinstance(obj, property):
      obj = obj.fget

    if isinstance(obj, (staticmethod, classmethod)):
      obj = obj.__func__

    try:
      doc_controls.do_not_doc_in_subclasses(obj)
    except AttributeError:
      pass


def build_docs():
  """Build api docs for tensorflow_text."""
  _hide_layer_and_module_methods()
  del text.keras  # keras is empty.

  doc_generator = generate_lib.DocGenerator(
      root_title="TensorFlow Text",
      py_modules=[("text", text)],
      base_dir=os.path.dirname(text.__file__),
      search_hints=True,
      code_url_prefix=FLAGS.code_url_prefix,
      site_path="text/api_docs/python",
      callbacks=[public_api.explicit_package_contents_filter])
  doc_generator.build(FLAGS.output_dir)


def main(_):
  # Build API docs
  build_docs()


if __name__ == "__main__":
  app.run(main)
