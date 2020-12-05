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

# Copied over from https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/core/as_dataframe.py - all code belongs to original authors


"""Lazy imports for heavy dependencies."""

import importlib

from tensorflow.data.experimental.core.utils import py_utils as utils


def _try_import(module_name):
  """Try importing a module, with an informative error message on failure."""
  try:
    mod = importlib.import_module(module_name)
    return mod
  except ImportError as e:
    err_msg = ("Failed importing {name}. This likely means that the dataset "
               "requires additional dependencies that have to be "
               "manually installed (usually with `pip install {name}`). See "
               "setup.py extras_require.").format(name=module_name)
    utils.reraise(e, suffix=err_msg)


class LazyImporter(object):
  """Lazy importer for heavy dependencies.
  Some datasets require heavy dependencies for data generation. To allow for
  the default installation to remain lean, those heavy dependencies are
  lazily imported here.
  """

  @utils.classproperty
  @classmethod
  def apache_beam(cls):
    return _try_import("apache_beam")

  @utils.classproperty
  @classmethod
  def crepe(cls):
    return _try_import("crepe")

  @utils.classproperty
  @classmethod
  def cv2(cls):
    return _try_import("cv2")  # pylint: disable=unreachable

  @utils.classproperty
  @classmethod
  def gcld3(cls):
    return _try_import("gcld3")  # pylint: disable=unreachable

  @utils.classproperty
  @classmethod
  def h5py(cls):
    return _try_import("h5py")

  @utils.classproperty
  @classmethod
  def langdetect(cls):
    return _try_import("langdetect")

  @utils.classproperty
  @classmethod
  def librosa(cls):
    return _try_import("librosa")

  @utils.classproperty
  @classmethod
  def matplotlib(cls):
    _try_import("matplotlib.pyplot")
    return _try_import("matplotlib")

  @utils.classproperty
  @classmethod
  def mwparserfromhell(cls):
    return _try_import("mwparserfromhell")

  @utils.classproperty
  @classmethod
  def nltk(cls):
    return _try_import("nltk")

  @utils.classproperty
  @classmethod
  def pandas(cls):
    return _try_import("pandas")

  @utils.classproperty
  @classmethod
  def PIL_Image(cls):  # pylint: disable=invalid-name
    # TiffImagePlugin need to be activated explicitly on some systems
    # https://github.com/python-pillow/Pillow/blob/5.4.x/src/PIL/Image.py#L407
    _try_import("PIL.TiffImagePlugin")
    return _try_import("PIL.Image")

  @utils.classproperty
  @classmethod
  def pretty_midi(cls):
    return _try_import("pretty_midi")

  @utils.classproperty
  @classmethod
  def pydub(cls):
    return _try_import("pydub")

  @utils.classproperty
  @classmethod
  def scipy(cls):
    _try_import("scipy.io")
    _try_import("scipy.io.wavfile")
    _try_import("scipy.ndimage")
    return _try_import("scipy")

  @utils.classproperty
  @classmethod
  def skimage(cls):
    _try_import("skimage.color")
    _try_import("skimage.filters")
    try:
      _try_import("skimage.external.tifffile")
    except ImportError:
      pass
    return _try_import("skimage")

  @utils.classproperty
  @classmethod
  def tifffile(cls):
    return _try_import("tifffile")

  @utils.classproperty
  @classmethod
  def tensorflow_data_validation(cls):
    return _try_import("tensorflow_data_validation")

  @utils.classproperty
  @classmethod
  def tensorflow_io(cls):
    return _try_import("tensorflow_io")

  @utils.classproperty
  @classmethod
  def tldextract(cls):
    return _try_import("tldextract")

  @utils.classproperty
  @classmethod
  def os(cls):
    """For testing purposes only."""
    return _try_import("os")

  @utils.classproperty
  @classmethod
  def test_foo(cls):
    """For testing purposes only."""
    return _try_import("test_foo")


lazy_imports = LazyImporter  # pylint: disable=invalid-name
