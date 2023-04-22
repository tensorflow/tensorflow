# lint as: python3
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

Requires a local installation of `tensorflow_docs`:

```
pip install git+https://github.com/tensorflow/docs
```
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pathlib
import textwrap

from absl import app
from absl import flags

import tensorflow as tf

from tensorflow_docs.api_generator import doc_controls
from tensorflow_docs.api_generator import doc_generator_visitor
from tensorflow_docs.api_generator import generate_lib

from tensorflow.python.framework import ops
from tensorflow.python.util import tf_export
from tensorflow.python.util import tf_inspect

# Caution: the google and oss versions of this import are different.
import base_dir

# pylint: disable=g-import-not-at-top
try:
  from tensorflow.python.types import doc_typealias
  _EXTRA_DOCS = getattr(doc_typealias, "_EXTRA_DOCS", {})
  del doc_typealias
except ImportError:
  _EXTRA_DOCS = {}
# pylint: enable=g-import-not-at-top

# `tf` has an `__all__` that doesn't list important things like `keras`.
# The doc generator recognizes `__all__` as the list of public symbols.
# So patch `tf.__all__` to list everything.
tf.__all__ = [item_name for item_name, value in tf_inspect.getmembers(tf)]

# tf_export generated two copies of the module objects.
# This will just list compat.v2 as an alias for tf. Close enough, let's not
# duplicate all the module skeleton files.
tf.compat.v2 = tf

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "code_url_prefix",
    "/code/stable/tensorflow",
    "A url to prepend to code paths when creating links to defining code")

flags.DEFINE_string("output_dir", "/tmp/out",
                    "A directory, where the docs will be output to.")

flags.DEFINE_bool("search_hints", True,
                  "Include meta-data search hints at the top of each file.")

flags.DEFINE_string(
    "site_path", "",
    "The path prefix (up to `.../api_docs/python`) used in the "
    "`_toc.yaml` and `_redirects.yaml` files")

flags.DEFINE_bool("gen_report", False,
                  ("Generate an API report containing the health of the"
                   "docstrings of the public API."))

_PRIVATE_MAP = {
    "tf": ["python", "core", "compiler", "examples", "tools", "contrib"],
    # There's some aliasing between the compats and v1/2s, so it's easier to
    # block by name and location than by deleting, or hiding objects.
    "tf.compat.v1.compat": ["v1", "v2"],
    "tf.compat.v2.compat": ["v1", "v2"]
}

tf.__doc__ = """
  ## TensorFlow

  ```
  pip install tensorflow
  ```
  """


def generate_raw_ops_doc():
  """Generates docs for `tf.raw_ops`."""

  warning = textwrap.dedent("""\n
    Note: `tf.raw_ops` provides direct/low level access to all TensorFlow ops.
    See [the RFC](https://github.com/tensorflow/community/blob/master/rfcs/20181225-tf-raw-ops.md)
    for details. Unless you are library writer, you likely do not need to use
    these ops directly.""")

  table_header = textwrap.dedent("""

      | Op Name | Has Gradient |
      |---------|:------------:|""")

  parts = [warning, table_header]

  for op_name in sorted(dir(tf.raw_ops)):
    try:
      ops._gradient_registry.lookup(op_name)  # pylint: disable=protected-access
      has_gradient = "\N{HEAVY CHECK MARK}\N{VARIATION SELECTOR-16}"
    except LookupError:
      has_gradient = "\N{CROSS MARK}"

    if not op_name.startswith("_"):
      path = pathlib.Path("/") / FLAGS.site_path / "tf/raw_ops" / op_name
      path = path.with_suffix(".md")
      link = ('<a id={op_name} href="{path}">{op_name}</a>').format(
          op_name=op_name, path=str(path))
      parts.append("| {link} | {has_gradient} |".format(
          link=link, has_gradient=has_gradient))

  return "\n".join(parts)


# The doc generator isn't aware of tf_export.
# So prefix the score tuples with -1 when this is the canonical name, +1
# otherwise. The generator chooses the name with the lowest score.
class TfExportAwareVisitor(doc_generator_visitor.DocGeneratorVisitor):
  """A `tf_export`, `keras_export` and `estimator_export` aware doc_visitor."""

  def _score_name(self, name):
    all_exports = [tf_export.TENSORFLOW_API_NAME,
                   tf_export.KERAS_API_NAME,
                   tf_export.ESTIMATOR_API_NAME]

    for api_name in all_exports:
      canonical = tf_export.get_canonical_name_for_symbol(
          self._index[name], api_name=api_name)
      if canonical is not None:
        break

    canonical_score = 1
    if canonical is not None and name == "tf." + canonical:
      canonical_score = -1

    scores = super()._score_name(name)
    return (canonical_score,) + scores


def build_docs(output_dir, code_url_prefix, search_hints, gen_report):
  """Build api docs for tensorflow v2.

  Args:
    output_dir: A string path, where to put the files.
    code_url_prefix: prefix for "Defined in" links.
    search_hints: Bool. Include meta-data search hints at the top of each file.
    gen_report: Bool. Generates an API report containing the health of the
      docstrings of the public API.
  """
  # The custom page will be used for raw_ops.md not the one generated above.
  doc_controls.set_custom_page_content(tf.raw_ops, generate_raw_ops_doc())

  # Hide raw_ops from search.
  for name, obj in tf_inspect.getmembers(tf.raw_ops):
    if not name.startswith("_"):
      doc_controls.hide_from_search(obj)

  for cls in [tf.Module, tf.keras.layers.Layer, tf.keras.optimizers.Optimizer]:
    doc_controls.decorate_all_class_attributes(
        decorator=doc_controls.do_not_doc_in_subclasses,
        cls=cls,
        skip=["__init__"])

  try:
    doc_controls.do_not_generate_docs(tf.__internal__)
  except AttributeError:
    pass

  try:
    doc_controls.do_not_generate_docs(tf.keras.__internal__)
  except AttributeError:
    pass

  try:
    doc_controls.do_not_generate_docs(tf.__operators__)
  except AttributeError:
    pass

  try:
    doc_controls.do_not_generate_docs(tf.tools)
  except AttributeError:
    pass

  try:
    doc_controls.do_not_generate_docs(tf.compat.v1.pywrap_tensorflow)
  except AttributeError:
    pass

  try:
    doc_controls.do_not_generate_docs(tf.pywrap_tensorflow)
  except AttributeError:
    pass

  try:
    doc_controls.do_not_generate_docs(tf.flags)
  except AttributeError:
    pass

  base_dirs, code_url_prefixes = base_dir.get_base_dirs_and_prefixes(
      code_url_prefix)
  doc_generator = generate_lib.DocGenerator(
      root_title="TensorFlow 2",
      py_modules=[("tf", tf)],
      base_dir=base_dirs,
      search_hints=search_hints,
      code_url_prefix=code_url_prefixes,
      site_path=FLAGS.site_path,
      visitor_cls=TfExportAwareVisitor,
      private_map=_PRIVATE_MAP,
      gen_report=gen_report,
      extra_docs=_EXTRA_DOCS
  )

  doc_generator.build(output_dir)

  if gen_report:
    return

  out_path = pathlib.Path(output_dir)

  expected_path_contents = {
      "tf/summary/audio.md":
          "tensorboard/plugins/audio/summary_v2.py",
      "tf/estimator/DNNClassifier.md":
          "tensorflow_estimator/python/estimator/canned/dnn.py",
      "tf/nn/sigmoid_cross_entropy_with_logits.md":
          "python/ops/nn_impl.py",
      "tf/keras/Model.md":
          "keras/engine/training.py",
      "tf/keras/preprocessing/image/random_brightness.md":
          "keras_preprocessing/image/affine_transformations.py"
  }

  all_passed = True
  error_msg_parts = [
      'Some "view source" links seem to be broken, please check:'
  ]

  for (rel_path, contents) in expected_path_contents.items():
    path = out_path / rel_path
    if contents not in path.read_text():
      all_passed = False
      error_msg_parts.append("  " + str(path))

  if not all_passed:
    raise ValueError("\n".join(error_msg_parts))

  rejected_path_contents = {
      "tf/keras/optimizers.md": "keras/optimizers/__init__.py",
  }

  all_passed = True
  error_msg_parts = [
      'Bad "view source" links in generated files, please check:'
  ]
  for rel_path, content in rejected_path_contents.items():
    path = out_path / rel_path
    if content in path.read_text():
      all_passed = False
      error_msg_parts.append("  " + str(path))

  if not all_passed:
    raise ValueError("\n".join(error_msg_parts))

  num_files = len(list(out_path.rglob("*")))
  if num_files < 2000:
    raise ValueError("The TensorFlow api should be more than 2000 files"
                     "(found {}).".format(num_files))


def main(argv):
  del argv
  build_docs(
      output_dir=FLAGS.output_dir,
      code_url_prefix=FLAGS.code_url_prefix,
      search_hints=FLAGS.search_hints,
      gen_report=FLAGS.gen_report,)


if __name__ == "__main__":
  app.run(main)
