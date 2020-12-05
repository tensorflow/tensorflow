# coding=utf-8
# Copyright 2020 The TensorFlow Datasets Authors.
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

# Copied over from https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/core/tf_compat.py - all code belongs to original authors


"""TensorFlow compatibility utilities."""

# pylint: disable=g-import-not-at-top,g-direct-tensorflow-import

import contextlib
import distutils.version
import functools
import os

MIN_TF_VERSION = "2.1.0"

_ensure_tf_install_called = False


# Ensure TensorFlow is importable and its version is sufficiently recent. This
# needs to happen before anything else, since the imports below will try to
# import tensorflow, too.
def ensure_tf_install():  # pylint: disable=g-statement-before-imports
  """Attempt to import tensorflow, and ensure its version is sufficient.
  Raises:
    ImportError: if either tensorflow is not importable or its version is
    inadequate.
  """
  # Only check the first time.
  global _ensure_tf_install_called
  if _ensure_tf_install_called:
    return
  _ensure_tf_install_called = True

  try:
    import tensorflow.compat.v2 as tf  # pylint: disable=import-outside-toplevel
  except ImportError:
    # Print more informative error message, then reraise.
    print("\n\nFailed to import TensorFlow. Please note that TensorFlow is not "
          "installed by default when you install TensorFlow Datasets. This is "
          "so that users can decide whether to install the GPU-enabled "
          "TensorFlow package. To use TensorFlow Datasets, please install the "
          "most recent version of TensorFlow, by following instructions at "
          "https://tensorflow.org/install.\n\n")
    raise

  tf_version = distutils.version.LooseVersion(tf.__version__)
  min_tf_version = distutils.version.LooseVersion(MIN_TF_VERSION)
  if tf_version < min_tf_version:
    raise ImportError(
        "This version of TensorFlow Datasets requires TensorFlow "
        f"version >= {MIN_TF_VERSION}; Detected an installation of version "
        f"{tf.__version__}. Please upgrade TensorFlow to proceed."
    )


def is_dataset(ds):
  """Whether ds is a Dataset. Compatible across TF versions."""
  import tensorflow.compat.v2 as tf  # pylint: disable=import-outside-toplevel
  return isinstance(ds, (tf.data.Dataset, tf.compat.v1.data.Dataset))


def _make_pathlike_fn(fn, nb_path_arg=1):
  """Wrap the function in a PathLike-compatible function."""

  @functools.wraps(fn)
  def new_fn(*args, **kwargs):
    # Normalize PathLike objects
    args = tuple(os.fspath(arg) for arg in args[:nb_path_arg])
    return fn(*args, **kwargs)

  return new_fn


@contextlib.contextmanager
def mock_gfile_pathlike():
  """Contextmanager which patch the `tf.io.gfile` API to be PathLike compatible.
  Before TF 2.4, GFile API is not PathLike compatible.
  After TF2.4, this function is a no-op.
  Yields:
    None
  """
  import tensorflow.compat.v2 as tf  # pylint: disable=import-outside-toplevel
  import tensorflow_datasets.testing as tfds_test  # pytype: disable=import-error

  class GFile(tf.io.gfile.GFile):

    def __init__(self, fpath, *args, **kwargs):
      super().__init__(os.fspath(fpath), *args, **kwargs)

  tf_version = distutils.version.LooseVersion(tf.__version__)
  min_tf_version = distutils.version.LooseVersion("2.4.0")
  if tf_version >= min_tf_version:
    yield  # No-op for recent TF versions
  else:  # Legacy TF, patch TF to restore PathLike compatibility
    with contextlib.ExitStack() as stack:
      for fn_name, nb_path_arg in [
          ("copy", 2),  # Use str, as tf.io.gfile.copy.__name__ == 'copy_v2'
          ("exists", 1),
          ("glob", 1),
          ("isdir", 1),
          ("listdir", 1),
          ("makedirs", 1),
          ("mkdir", 1),
          ("remove", 1),
          ("rename", 2),
          ("rmtree", 1),
          ("stat", 1),
          ("walk", 1),
      ]:
        fn = getattr(tf.io.gfile, fn_name)
        new_fn = _make_pathlike_fn(fn, nb_path_arg)
        stack.enter_context(tfds_test.mock_tf(f"tf.io.gfile.{fn_name}", new_fn))
      stack.enter_context(tfds_test.mock_tf("tf.io.gfile.GFile", GFile))
      yield
