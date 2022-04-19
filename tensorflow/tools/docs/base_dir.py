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
import distutils
from os import path

import keras_preprocessing
import tensorboard
import tensorflow as tf
from tensorflow_docs.api_generator import public_api
import tensorflow_estimator


try:
  import keras  # pylint: disable=g-import-not-at-top
except ImportError:
  pass


def get_base_dirs_and_prefixes(code_url_prefix):
  """Returns the base_dirs and code_prefixes for OSS TensorFlow api gen."""
  base_dir = path.dirname(tf.__file__)

  if distutils.version.LooseVersion(tf.__version__) >= "2.9":
    base_dirs = [
        base_dir,
        path.dirname(keras.__file__),
        path.dirname(tensorboard.__file__),
        path.dirname(tensorflow_estimator.__file__),
    ]

  elif distutils.version.LooseVersion(tf.__version__) >= "2.6":
    base_dirs = [
        base_dir,
        path.dirname(keras.__file__),
        path.dirname(keras_preprocessing.__file__),
        path.dirname(tensorboard.__file__),
        path.dirname(tensorflow_estimator.__file__),
    ]
  elif distutils.version.LooseVersion(tf.__version__) >= "2.2":
    base_dirs = [
        base_dir,
        path.dirname(keras_preprocessing.__file__),
        path.dirname(tensorboard.__file__),
        path.dirname(tensorflow_estimator.__file__),
    ]
  else:
    base_dirs = [
        path.normpath(path.join(base_dir, "../tensorflow_core")),
        path.dirname(keras_preprocessing.__file__),
        path.dirname(tensorboard.__file__),
        path.dirname(tensorflow_estimator.__file__),
    ]

  if "dev" in tf.__version__:
    keras_url_prefix = "https://github.com/keras-team/keras/tree/master/keras"
  else:
    keras_url_prefix = f"https://github.com/keras-team/keras/tree/v{keras.__version__}/keras"

  if distutils.version.LooseVersion(tf.__version__) >= "2.9":
    code_url_prefixes = (
        code_url_prefix,
        keras_url_prefix,
        f"https://github.com/tensorflow/tensorboard/tree/{tensorboard.__version__}/tensorboard",
        "https://github.com/tensorflow/estimator/tree/master/tensorflow_estimator",
    )
  elif distutils.version.LooseVersion(tf.__version__) >= "2.6":
    code_url_prefixes = (
        code_url_prefix,
        keras_url_prefix,
        f"https://github.com/keras-team/keras-preprocessing/tree/{keras_preprocessing.__version__}/keras_preprocessing",
        f"https://github.com/tensorflow/tensorboard/tree/{tensorboard.__version__}/tensorboard",
        "https://github.com/tensorflow/estimator/tree/master/tensorflow_estimator",
    )
  else:
    code_url_prefixes = (
        code_url_prefix,
        f"https://github.com/keras-team/keras-preprocessing/tree/{keras_preprocessing.__version__}/keras_preprocessing",
        f"https://github.com/tensorflow/tensorboard/tree/{tensorboard.__version__}/tensorboard",
        "https://github.com/tensorflow/estimator/tree/master/tensorflow_estimator",
    )

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
  if distutils.version.LooseVersion(tf.__version__) >= "2.9":
    return [explicit_filter_keep_keras]
  else:
    return []
