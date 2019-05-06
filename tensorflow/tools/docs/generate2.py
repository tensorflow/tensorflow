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
"""A tool to generate api_docs for TensorFlow2.

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

from tensorflow_docs.api_generator import doc_controls
from tensorflow_docs.api_generator import doc_generator_visitor
from tensorflow_docs.api_generator import generate_lib
from tensorflow_docs.api_generator import parser

import tensorboard
import tensorflow_estimator
from tensorflow.python.util import tf_export
from tensorflow.python.util import tf_inspect

# Use tensorflow's `tf_inspect`, which is aware of `tf_decorator`.
parser.tf_inspect = tf_inspect

# `tf` has an `__all__` that doesn't list important things like `keras`.
# The doc generator recognizes `__all__` as the list of public symbols.
# So patch `tf.__all__` to list everything.
tf.__all__ = [item_name for item_name, value in tf_inspect.getmembers(tf)]


FLAGS = flags.FLAGS

flags.DEFINE_string(
    "code_url_prefix",
    "/code/stable/tensorflow",
    "A url to prepend to code paths when creating links to defining code")

flags.DEFINE_string(
    "output_dir", "/tmp/out",
    "A directory, where the docs will be output to.")

flags.DEFINE_bool("search_hints", True,
                  "Include meta-data search hints at the top of each file.")

flags.DEFINE_string("site_path", "",
                    "The prefix ({site-path}/api_docs/python/...) used in the "
                    "`_toc.yaml` and `_redirects.yaml` files")


# The doc generator isn't aware of tf_export.
# So prefix the score tuples with -1 when this is the canonical name, +1
# otherwise. The generator chooses the name with the lowest score.
class TfExportAwareDocGeneratorVisitor(
    doc_generator_visitor.DocGeneratorVisitor):
  """A `tf_export` aware doc_visitor."""

  def _score_name(self, name):
    canonical = tf_export.get_canonical_name_for_symbol(self._index[name])

    canonical_score = 1
    if canonical is not None and name == "tf." + canonical:
      canonical_score = -1

    scores = super(TfExportAwareDocGeneratorVisitor, self)._score_name(name)
    return (canonical_score,) + scores


def build_docs(output_dir, code_url_prefix, search_hints=True):
  """Build api docs for tensorflow v2.

  Args:
    output_dir: A string path, where to put the files.
    code_url_prefix: prefix for "Defined in" links.
    search_hints: Bool. Include meta-data search hints at the top of each file.
  """
  try:
    doc_controls.do_not_generate_docs(tf.tools)
  except AttributeError:
    pass

  base_dir = path.dirname(tf.__file__)
  base_dirs = (
      base_dir,
      path.normpath(path.join(base_dir, "../../tensorflow")),
      path.dirname(tensorboard.__file__),
      path.dirname(tensorflow_estimator.__file__),
  )

  code_url_prefixes = (
      code_url_prefix,
      # External packages source repositories
      "https://github.com/tensorflow/tensorboard/tree/master/tensorboard"
      "https://github.com/tensorflow/estimator/tree/master/tensorflow_estimator"
  )

  doc_generator = generate_lib.DocGenerator(
      root_title="TensorFlow 2.0 Preview",
      py_modules=[("tf", tf)],
      base_dir=base_dirs,
      search_hints=search_hints,
      code_url_prefix=code_url_prefixes,
      site_path=FLAGS.site_path,
      visitor_cls=TfExportAwareDocGeneratorVisitor)

  doc_generator.build(output_dir)


def main(argv):
  del argv
  build_docs(output_dir=FLAGS.output_dir,
             code_url_prefix=FLAGS.code_url_prefix,
             search_hints=FLAGS.search_hints)


if __name__ == "__main__":
  app.run(main)
