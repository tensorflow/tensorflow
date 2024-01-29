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
"""Opensource base_dir configuration for tensorflow doc-generator."""
import pathlib

import keras
from packaging import version
import tensorboard
import tensorflow as tf
from tensorflow_docs.api_generator import public_api

try:
  import tensorflow_estimator  # pylint: disable=[g-import-not-at-top, g-deprecated-tf-checker]
except ImportError:
  tensorflow_estimator = None


def get_base_dirs_and_prefixes(code_url_prefix):
  """Returns the base_dirs and code_prefixes for OSS TensorFlow api gen."""
  base_dir = pathlib.Path(tf.__file__).parent

  if "dev" in tf.__version__:
    keras_url_prefix = "https://github.com/keras-team/keras/tree/master/keras"
  else:
    keras_url_prefix = (
        f"https://github.com/keras-team/keras/tree/v{keras.__version__}/keras"
    )

  if version.parse(tf.__version__) >= version.parse("2.16"):
    # First match takes precedence.
    # Objects are dropped if they have no match.
    base_dirs = [
        # The real keras source files are now in `site-packages/keras/src/...`
        pathlib.Path(keras.__file__).parent / "src",
        # The generated module files in tensorflow are in keras
        # under `site-packages/keras/api/_v2/keras/...`.
        pathlib.Path(tf.keras.__file__).parent,
        # The generated api-module files are now in `site-packages/keras/...`
        pathlib.Path(keras.__file__).parent,
        pathlib.Path(tensorboard.__file__).parent,
        # The tensorflow base dir goes last because `tf.keras``
        base_dir,
    ]

    code_url_prefixes = (
        keras_url_prefix,
        # None -> don't link to the generated keras api-module files.
        None,
        None,
        f"https://github.com/tensorflow/tensorboard/tree/{tensorboard.__version__}/tensorboard",
        code_url_prefix,
    )

  elif version.parse(tf.__version__) >= version.parse("2.13"):
    # First match takes precedence.
    # Objects are dropped if they have no match.
    base_dirs = [
        # The real keras source files are now in `site-packages/keras/src/...`
        pathlib.Path(keras.__file__).parent / "src",
        # The generated module files in tensorflow are in keras
        # under `site-packages/keras/api/_v2/keras/...`.
        pathlib.Path(tf.keras.__file__).parent,
        # The generated api-module files are now in `site-packages/keras/...`
        pathlib.Path(keras.__file__).parent,
        pathlib.Path(tensorboard.__file__).parent,
        pathlib.Path(tensorflow_estimator.__file__).parent,
        # The tensorflow base dir goes last because `tf.keras``
        base_dir,
    ]

    code_url_prefixes = (
        keras_url_prefix,
        # None -> don't link to the generated keras api-module files.
        None,
        None,
        f"https://github.com/tensorflow/tensorboard/tree/{tensorboard.__version__}/tensorboard",
        "https://github.com/tensorflow/estimator/tree/master/tensorflow_estimator",
        code_url_prefix,
    )
  elif version.parse(tf.__version__) >= version.parse("2.9"):
    base_dirs = [
        base_dir,
        pathlib.Path(keras.__file__).parent,
        pathlib.Path(tensorboard.__file__).parent,
        pathlib.Path(tensorflow_estimator.__file__).parent,
    ]
    code_url_prefixes = (
        code_url_prefix,
        keras_url_prefix,
        f"https://github.com/tensorflow/tensorboard/tree/{tensorboard.__version__}/tensorboard",
        "https://github.com/tensorflow/estimator/tree/master/tensorflow_estimator",
    )
  else:
    raise ValueError("Unsupported: version < 2.9")

  return base_dirs, code_url_prefixes


def explicit_filter_keep_keras(parent_path, parent, children):
  """Like explicit_package_contents_filter, but keeps keras."""
  new_children = public_api.explicit_package_contents_filter(
      parent_path, parent, children)

  if parent_path[-1] not in ["tf", "v1", "v2"]:
    return new_children

  had_keras = any(name == "keras" for name, child in children)
  has_keras = any(name == "keras" for name, child in new_children)

  if had_keras and not has_keras:
    new_children.append(("keras", parent.keras))

  return sorted(new_children, key=lambda x: x[0])


def get_callbacks():
  return [explicit_filter_keep_keras]
