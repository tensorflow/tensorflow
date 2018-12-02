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
r"""A tool to generate api_docs for TensorFlow2.

```
python generate2.py --output_dir=/tmp/out
```

Requires a local installation of:
  https://github.com/tensorflow/docs/tree/master/tools
  tf-nightly-2.0-preview
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os import path

from absl import app
from absl import flags

import tensorflow as tf

from tensorflow_docs.api_generator import generate_lib

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "code_url_prefix",
    "/code/stable/tensorflow/",
    "A url to prepend to code paths when creating links to defining code")

flags.DEFINE_string(
    "output_dir", "/tmp/out",
    "A directory, where the docs will be output to.")


def build_docs(output_dir, code_url_prefix):
  """Build api docs for tensorflow v2.

  Args:
    output_dir: A string path, where to put the files.
    code_url_prefix: prefix for "Defined in" links.
  """
  base_dir = path.dirname(tf.__file__)
  doc_generator = generate_lib.DocGenerator(
      root_title="TensorFlow 2.0 Preview",
      py_modules=[("tf", tf)],
      base_dir=base_dir,
      search_hints=True,
      code_url_prefix=code_url_prefix,
      site_path="api_docs/")

  doc_generator.build(output_dir)


def main(argv):
  del argv
  build_docs(output_dir=FLAGS.output_dir,
             code_url_prefix=FLAGS.code_url_prefix)


if __name__ == "__main__":
  app.run(main)
