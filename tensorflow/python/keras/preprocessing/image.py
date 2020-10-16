# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
# pylint: disable=invalid-name
# pylint: disable=g-import-not-at-top
# pylint: disable=g-classes-have-attributes
"""Set of tools for real-time data augmentation on image data.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras_preprocessing import image
import numpy as np
try:
  from scipy import linalg  # pylint: disable=unused-import
  from scipy import ndimage  # pylint: disable=unused-import
except ImportError:
  pass

from tensorflow.python.framework import ops
from tensorflow.python.keras import backend
from tensorflow.python.keras.preprocessing.image_dataset import image_dataset_from_directory  # pylint: disable=unused-import
from tensorflow.python.keras.utils import data_utils
from tensorflow.python.keras.utils import tf_inspect
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import image_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import tf_logging
from tensorflow.python.util.tf_export import keras_export

random_rotation = image.random_rotation
random_shift = image.random_shift
random_shear = image.random_shear
random_zoom = image.random_zoom
apply_channel_shift = image.apply_channel_shift
random_channel_shift = image.random_channel_shift
apply_brightness_shift = image.apply_brightness_shift
random_brightness = image.random_brightness
apply_affine_transform = image.apply_affine_transform


@keras_export('keras.preprocessing.image.smart_resize', v1=[])
def smart_resize(x, size, interpolation='bilinear'):
  """Resize images to a target size without aspect ratio distortion.

  TensorFlow image datasets typically yield images that have each a different
  size. However, these images need to be batched before they can be
  processed by Keras layers. To be batched, images need to share the same height
  and width.

  You could simply do:

  ````python
  size = (200, 200)
  ds = ds.map(lambda img: tf.image.resize(img, size))
  ```

  However, if you do this, you distort the aspect ratio of your images, since
  in general they do not all have the same aspect ratio as `size`. This is
  fine in many cases, but not always (e.g. for GANs this can be a problem).

  Note that passing the argument `preserve_aspect_ratio=True` to `resize`
  will preserve the aspect ratio, but at the cost of no longer respecting the
  provided target size. Because `tf.image.resize` doesn't crop images,
  your output images will still have different sizes.

  This calls for:

  ```python
  size = (200, 200)
  ds = ds.map(lambda img: smart_resize(img, size))
  ```

  Your output images will actually be `(200, 200)`, and will not be distorted.
  Instead, the parts of the image that do not fit within the target size
  get cropped out.

  The resizing process is:

  1. Take the largest centered crop of the image that has the same aspect ratio
  as the target size. For instance, if `size=(200, 200)` and the input image has
  size `(340, 500)`, we take a crop of `(340, 340)` centered along the width.
  2. Resize the cropped image to the target size. In the example above,
  we resize the `(340, 340)` crop to `(200, 200)`.

  Arguments:
    x: Input image (as a tensor or NumPy array). Must be in format
      `(height, width, channels)`.
    size: Tuple of `(height, width)` integer. Target size.
    interpolation: String, interpolation to use for resizing.
      Defaults to `'bilinear'`. Supports `bilinear`, `nearest`, `bicubic`,
      `area`, `lanczos3`, `lanczos5`, `gaussian`, `mitchellcubic`.

  Returns:
    Array with shape `(size[0], size[1], channels)`. If the input image was a
    NumPy array, the output is a NumPy array, and if it was a TF tensor,
    the output is a TF tensor.
  """
  if len(size) != 2:
    raise ValueError('Expected `size` to be a tuple of 2 integers, '
                     'but got: %s' % (size,))
  img = ops.convert_to_tensor_v2_with_dispatch(x)
  if img.shape.rank is not None:
    if img.shape.rank != 3:
      raise ValueError(
          'Expected an image array with shape `(height, width, channels)`, but '
          'got input with incorrect rank, of shape %s' % (img.shape,))
  shape = array_ops.shape(img)
  height, width = shape[0], shape[1]
  target_height, target_width = size

  crop_height = math_ops.cast(
      math_ops.cast(width * target_height, 'float32') / target_width, 'int32')
  crop_width = math_ops.cast(
      math_ops.cast(height * target_width, 'float32') / target_height, 'int32')

  # Set back to input height / width if crop_height / crop_width is not smaller.
  crop_height = math_ops.minimum(height, crop_height)
  crop_width = math_ops.minimum(width, crop_width)

  crop_box_hstart = math_ops.cast(
      math_ops.cast(height - crop_height, 'float32') / 2, 'int32')
  crop_box_wstart = math_ops.cast(
      math_ops.cast(width - crop_width, 'float32') / 2, 'int32')

  crop_box_start = array_ops.stack([crop_box_hstart, crop_box_wstart, 0])
  crop_box_size = array_ops.stack([crop_height, crop_width, -1])

  img = array_ops.slice(img, crop_box_start, crop_box_size)
  img = image_ops.resize_images_v2(
      images=img,
      size=size,
      method=interpolation)
  if isinstance(x, np.ndarray):
    return img.numpy()
  return img


@keras_export('keras.preprocessing.image.array_to_img')
def array_to_img(x, data_format=None, scale=True, dtype=None):
  """Converts a 3D Numpy array to a PIL Image instance.

  Usage:

  ```python
  from PIL import Image
  img = np.random.random(size=(100, 100, 3))
  pil_img = tf.keras.preprocessing.image.array_to_img(img)
  ```


  Arguments:
      x: Input Numpy array.
      data_format: Image data format, can be either "channels_first" or
        "channels_last". Defaults to `None`, in which case the global setting
        `tf.keras.backend.image_data_format()` is used (unless you changed it,
        it defaults to "channels_last").
      scale: Whether to rescale image values to be within `[0, 255]`. Defaults
        to `True`.
      dtype: Dtype to use. Default to `None`, in which case the global setting
      `tf.keras.backend.floatx()` is used (unless you changed it, it defaults
      to "float32")

  Returns:
      A PIL Image instance.

  Raises:
      ImportError: if PIL is not available.
      ValueError: if invalid `x` or `data_format` is passed.
  """

  if data_format is None:
    data_format = backend.image_data_format()
  kwargs = {}
  if 'dtype' in tf_inspect.getfullargspec(image.array_to_img)[0]:
    if dtype is None:
      dtype = backend.floatx()
    kwargs['dtype'] = dtype
  return image.array_to_img(x, data_format=data_format, scale=scale, **kwargs)


@keras_export('keras.preprocessing.image.img_to_array')
def img_to_array(img, data_format=None, dtype=None):
  """Converts a PIL Image instance to a Numpy array.

  Usage:

  ```python
  from PIL import Image
  img_data = np.random.random(size=(100, 100, 3))
  img = tf.keras.preprocessing.image.array_to_img(img_data)
  array = tf.keras.preprocessing.image.img_to_array(img)
  ```


  Arguments:
      img: Input PIL Image instance.
      data_format: Image data format, can be either "channels_first" or
        "channels_last". Defaults to `None`, in which case the global setting
        `tf.keras.backend.image_data_format()` is used (unless you changed it,
        it defaults to "channels_last").
      dtype: Dtype to use. Default to `None`, in which case the global setting
      `tf.keras.backend.floatx()` is used (unless you changed it, it defaults
      to "float32")

  Returns:
      A 3D Numpy array.

  Raises:
      ValueError: if invalid `img` or `data_format` is passed.
  """

  if data_format is None:
    data_format = backend.image_data_format()
  kwargs = {}
  if 'dtype' in tf_inspect.getfullargspec(image.img_to_array)[0]:
    if dtype is None:
      dtype = backend.floatx()
    kwargs['dtype'] = dtype
  return image.img_to_array(img, data_format=data_format, **kwargs)


@keras_export('keras.preprocessing.image.save_img')
def save_img(path,
             x,
             data_format=None,
             file_format=None,
             scale=True,
             **kwargs):
  """Saves an image stored as a Numpy array to a path or file object.

  Arguments:
      path: Path or file object.
      x: Numpy array.
      data_format: Image data format,
          either "channels_first" or "channels_last".
      file_format: Optional file format override. If omitted, the
          format to use is determined from the filename extension.
          If a file object was used instead of a filename, this
          parameter should always be used.
      scale: Whether to rescale image values to be within `[0, 255]`.
      **kwargs: Additional keyword arguments passed to `PIL.Image.save()`.
  """
  if data_format is None:
    data_format = backend.image_data_format()
  image.save_img(path,
                 x,
                 data_format=data_format,
                 file_format=file_format,
                 scale=scale, **kwargs)


def load_img(path, grayscale=False, color_mode='rgb', target_size=None,
             interpolation='nearest'):
  """Loads an image into PIL format.

  Usage:

  ```
  image = tf.keras.preprocessing.image.load_img(image_path)
  input_arr = keras.preprocessing.image.img_to_array(image)
  input_arr = np.array([input_arr])  # Convert single image to a batch.
  predictions = model.predict(input_arr)
  ```

  Arguments:
      path: Path to image file.
      grayscale: DEPRECATED use `color_mode="grayscale"`.
      color_mode: One of "grayscale", "rgb", "rgba". Default: "rgb".
          The desired image format.
      target_size: Either `None` (default to original size)
          or tuple of ints `(img_height, img_width)`.
      interpolation: Interpolation method used to resample the image if the
          target size is different from that of the loaded image.
          Supported methods are "nearest", "bilinear", and "bicubic".
          If PIL version 1.1.3 or newer is installed, "lanczos" is also
          supported. If PIL version 3.4.0 or newer is installed, "box" and
          "hamming" are also supported. By default, "nearest" is used.

  Returns:
      A PIL Image instance.

  Raises:
      ImportError: if PIL is not available.
      ValueError: if interpolation method is not supported.
  """
  return image.load_img(path, grayscale=grayscale, color_mode=color_mode,
                        target_size=target_size, interpolation=interpolation)


@keras_export('keras.preprocessing.image.Iterator')
class Iterator(image.Iterator, data_utils.Sequence):
  pass


@keras_export('keras.preprocessing.image.DirectoryIterator')
class DirectoryIterator(image.DirectoryIterator, Iterator):
  """Iterator capable of reading images from a directory on disk.

  Arguments:
      directory: Path to the directory to read images from.
          Each subdirectory in this directory will be
          considered to contain images from one class,
          or alternatively you could specify class subdirectories
          via the `classes` argument.
      image_data_generator: Instance of `ImageDataGenerator`
          to use for random transformations and normalization.
      target_size: tuple of integers, dimensions to resize input images to.
      color_mode: One of `"rgb"`, `"rgba"`, `"grayscale"`.
          Color mode to read images.
      classes: Optional list of strings, names of subdirectories
          containing images from each class (e.g. `["dogs", "cats"]`).
          It will be computed automatically if not set.
      class_mode: Mode for yielding the targets:
          `"binary"`: binary targets (if there are only two classes),
          `"categorical"`: categorical targets,
          `"sparse"`: integer targets,
          `"input"`: targets are images identical to input images (mainly
              used to work with autoencoders),
          `None`: no targets get yielded (only input images are yielded).
      batch_size: Integer, size of a batch.
      shuffle: Boolean, whether to shuffle the data between epochs.
      seed: Random seed for data shuffling.
      data_format: String, one of `channels_first`, `channels_last`.
      save_to_dir: Optional directory where to save the pictures
          being yielded, in a viewable format. This is useful
          for visualizing the random transformations being
          applied, for debugging purposes.
      save_prefix: String prefix to use for saving sample
          images (if `save_to_dir` is set).
      save_format: Format to use for saving sample images
          (if `save_to_dir` is set).
      subset: Subset of data (`"training"` or `"validation"`) if
          validation_split is set in ImageDataGenerator.
      interpolation: Interpolation method used to resample the image if the
          target size is different from that of the loaded image.
          Supported methods are "nearest", "bilinear", and "bicubic".
          If PIL version 1.1.3 or newer is installed, "lanczos" is also
          supported. If PIL version 3.4.0 or newer is installed, "box" and
          "hamming" are also supported. By default, "nearest" is used.
      dtype: Dtype to use for generated arrays.
  """

  def __init__(self, directory, image_data_generator,
               target_size=(256, 256),
               color_mode='rgb',
               classes=None,
               class_mode='categorical',
               batch_size=32,
               shuffle=True,
               seed=None,
               data_format=None,
               save_to_dir=None,
               save_prefix='',
               save_format='png',
               follow_links=False,
               subset=None,
               interpolation='nearest',
               dtype=None):
    if data_format is None:
      data_format = backend.image_data_format()
    kwargs = {}
    if 'dtype' in tf_inspect.getfullargspec(
        image.ImageDataGenerator.__init__)[0]:
      if dtype is None:
        dtype = backend.floatx()
      kwargs['dtype'] = dtype
    super(DirectoryIterator, self).__init__(
        directory, image_data_generator,
        target_size=target_size,
        color_mode=color_mode,
        classes=classes,
        class_mode=class_mode,
        batch_size=batch_size,
        shuffle=shuffle,
        seed=seed,
        data_format=data_format,
        save_to_dir=save_to_dir,
        save_prefix=save_prefix,
        save_format=save_format,
        follow_links=follow_links,
        subset=subset,
        interpolation=interpolation,
        **kwargs)


@keras_export('keras.preprocessing.image.NumpyArrayIterator')
class NumpyArrayIterator(image.NumpyArrayIterator, Iterator):
  """Iterator yielding data from a Numpy array.

  Arguments:
      x: Numpy array of input data or tuple.
          If tuple, the second elements is either
          another numpy array or a list of numpy arrays,
          each of which gets passed
          through as an output without any modifications.
      y: Numpy array of targets data.
      image_data_generator: Instance of `ImageDataGenerator`
          to use for random transformations and normalization.
      batch_size: Integer, size of a batch.
      shuffle: Boolean, whether to shuffle the data between epochs.
      sample_weight: Numpy array of sample weights.
      seed: Random seed for data shuffling.
      data_format: String, one of `channels_first`, `channels_last`.
      save_to_dir: Optional directory where to save the pictures
          being yielded, in a viewable format. This is useful
          for visualizing the random transformations being
          applied, for debugging purposes.
      save_prefix: String prefix to use for saving sample
          images (if `save_to_dir` is set).
      save_format: Format to use for saving sample images
          (if `save_to_dir` is set).
      subset: Subset of data (`"training"` or `"validation"`) if
          validation_split is set in ImageDataGenerator.
      dtype: Dtype to use for the generated arrays.
  """

  def __init__(self, x, y, image_data_generator,
               batch_size=32,
               shuffle=False,
               sample_weight=None,
               seed=None,
               data_format=None,
               save_to_dir=None,
               save_prefix='',
               save_format='png',
               subset=None,
               dtype=None):
    if data_format is None:
      data_format = backend.image_data_format()
    kwargs = {}
    if 'dtype' in tf_inspect.getfullargspec(
        image.NumpyArrayIterator.__init__)[0]:
      if dtype is None:
        dtype = backend.floatx()
      kwargs['dtype'] = dtype
    super(NumpyArrayIterator, self).__init__(
        x, y, image_data_generator,
        batch_size=batch_size,
        shuffle=shuffle,
        sample_weight=sample_weight,
        seed=seed,
        data_format=data_format,
        save_to_dir=save_to_dir,
        save_prefix=save_prefix,
        save_format=save_format,
        subset=subset,
        **kwargs)


class DataFrameIterator(image.DataFrameIterator, Iterator):
  """Iterator capable of reading images from a directory on disk as a dataframe.

  Arguments:
      dataframe: Pandas dataframe containing the filepaths relative to
        `directory` (or absolute paths if `directory` is None) of the images in
        a string column. It should include other column/s
          depending on the `class_mode`: - if `class_mode` is `"categorical"`
            (default value) it must include the `y_col` column with the class/es
            of each image. Values in column can be string/list/tuple if a single
            class or list/tuple if multiple classes. - if `class_mode` is
            `"binary"` or `"sparse"` it must include the given `y_col` column
            with class values as strings. - if `class_mode` is `"raw"` or
            `"multi_output"` it should contain the columns specified in `y_col`.
            - if `class_mode` is `"input"` or `None` no extra column is needed.
      directory: string, path to the directory to read images from. If `None`,
        data in `x_col` column should be absolute paths.
      image_data_generator: Instance of `ImageDataGenerator` to use for random
        transformations and normalization. If None, no transformations and
        normalizations are made.
      x_col: string, column in `dataframe` that contains the filenames (or
        absolute paths if `directory` is `None`).
      y_col: string or list, column/s in `dataframe` that has the target data.
      weight_col: string, column in `dataframe` that contains the sample
          weights. Default: `None`.
      target_size: tuple of integers, dimensions to resize input images to.
      color_mode: One of `"rgb"`, `"rgba"`, `"grayscale"`. Color mode to read
        images.
      classes: Optional list of strings, classes to use (e.g. `["dogs",
        "cats"]`). If None, all classes in `y_col` will be used.
      class_mode: one of "binary", "categorical", "input", "multi_output",
          "raw", "sparse" or None. Default: "categorical".
          Mode for yielding the targets:
          - `"binary"`: 1D numpy array of binary labels,
          - `"categorical"`: 2D numpy array of one-hot encoded labels. Supports
            multi-label output.
          - `"input"`: images identical to input images (mainly used to work
            with autoencoders),
          - `"multi_output"`: list with the values of the different columns,
          - `"raw"`: numpy array of values in `y_col` column(s),
          - `"sparse"`: 1D numpy array of integer labels, - `None`, no targets
            are returned (the generator will only yield batches of image data,
            which is useful to use in `model.predict()`).
      batch_size: Integer, size of a batch.
      shuffle: Boolean, whether to shuffle the data between epochs.
      seed: Random seed for data shuffling.
      data_format: String, one of `channels_first`, `channels_last`.
      save_to_dir: Optional directory where to save the pictures being yielded,
        in a viewable format. This is useful for visualizing the random
        transformations being applied, for debugging purposes.
      save_prefix: String prefix to use for saving sample images (if
        `save_to_dir` is set).
      save_format: Format to use for saving sample images (if `save_to_dir` is
        set).
      subset: Subset of data (`"training"` or `"validation"`) if
        validation_split is set in ImageDataGenerator.
      interpolation: Interpolation method used to resample the image if the
        target size is different from that of the loaded image. Supported
        methods are "nearest", "bilinear", and "bicubic". If PIL version 1.1.3
        or newer is installed, "lanczos" is also supported. If PIL version 3.4.0
        or newer is installed, "box" and "hamming" are also supported. By
        default, "nearest" is used.
      dtype: Dtype to use for the generated arrays.
      validate_filenames: Boolean, whether to validate image filenames in
        `x_col`. If `True`, invalid images will be ignored. Disabling this
        option
      can lead to speed-up in the instantiation of this class. Default: `True`.
  """

  def __init__(
      self,
      dataframe,
      directory=None,
      image_data_generator=None,
      x_col='filename',
      y_col='class',
      weight_col=None,
      target_size=(256, 256),
      color_mode='rgb',
      classes=None,
      class_mode='categorical',
      batch_size=32,
      shuffle=True,
      seed=None,
      data_format='channels_last',
      save_to_dir=None,
      save_prefix='',
      save_format='png',
      subset=None,
      interpolation='nearest',
      dtype='float32',
      validate_filenames=True):
    super(DataFrameIterator, self).__init__(
        dataframe=dataframe,
        directory=directory,
        image_data_generator=image_data_generator,
        x_col=x_col,
        y_col=y_col,
        weight_col=weight_col,
        target_size=target_size,
        color_mode=color_mode,
        classes=classes,
        class_mode=class_mode,
        batch_size=batch_size,
        shuffle=shuffle,
        seed=seed,
        data_format=data_format,
        save_to_dir=save_to_dir,
        save_prefix=save_prefix,
        save_format=save_format,
        subset=subset,
        interpolation=interpolation,
        dtype=dtype,
        validate_filenames=validate_filenames
    )


@keras_export('keras.preprocessing.image.ImageDataGenerator')
class ImageDataGenerator(image.ImageDataGenerator):
  """Generate batches of tensor image data with real-time data augmentation.

   The data will be looped over (in batches).

  Arguments:
      featurewise_center: Boolean.
          Set input mean to 0 over the dataset, feature-wise.
      samplewise_center: Boolean. Set each sample mean to 0.
      featurewise_std_normalization: Boolean.
          Divide inputs by std of the dataset, feature-wise.
      samplewise_std_normalization: Boolean. Divide each input by its std.
      zca_epsilon: epsilon for ZCA whitening. Default is 1e-6.
      zca_whitening: Boolean. Apply ZCA whitening.
      rotation_range: Int. Degree range for random rotations.
      width_shift_range: Float, 1-D array-like or int
          - float: fraction of total width, if < 1, or pixels if >= 1.
          - 1-D array-like: random elements from the array.
          - int: integer number of pixels from interval
              `(-width_shift_range, +width_shift_range)`
          - With `width_shift_range=2` possible values
              are integers `[-1, 0, +1]`,
              same as with `width_shift_range=[-1, 0, +1]`,
              while with `width_shift_range=1.0` possible values are floats
              in the interval [-1.0, +1.0).
      height_shift_range: Float, 1-D array-like or int
          - float: fraction of total height, if < 1, or pixels if >= 1.
          - 1-D array-like: random elements from the array.
          - int: integer number of pixels from interval
              `(-height_shift_range, +height_shift_range)`
          - With `height_shift_range=2` possible values
              are integers `[-1, 0, +1]`,
              same as with `height_shift_range=[-1, 0, +1]`,
              while with `height_shift_range=1.0` possible values are floats
              in the interval [-1.0, +1.0).
      brightness_range: Tuple or list of two floats. Range for picking
          a brightness shift value from.
      shear_range: Float. Shear Intensity
          (Shear angle in counter-clockwise direction in degrees)
      zoom_range: Float or [lower, upper]. Range for random zoom.
          If a float, `[lower, upper] = [1-zoom_range, 1+zoom_range]`.
      channel_shift_range: Float. Range for random channel shifts.
      fill_mode: One of {"constant", "nearest", "reflect" or "wrap"}.
          Default is 'nearest'.
          Points outside the boundaries of the input are filled
          according to the given mode:
          - 'constant': kkkkkkkk|abcd|kkkkkkkk (cval=k)
          - 'nearest':  aaaaaaaa|abcd|dddddddd
          - 'reflect':  abcddcba|abcd|dcbaabcd
          - 'wrap':  abcdabcd|abcd|abcdabcd
      cval: Float or Int.
          Value used for points outside the boundaries
          when `fill_mode = "constant"`.
      horizontal_flip: Boolean. Randomly flip inputs horizontally.
      vertical_flip: Boolean. Randomly flip inputs vertically.
      rescale: rescaling factor. Defaults to None.
          If None or 0, no rescaling is applied,
          otherwise we multiply the data by the value provided
          (after applying all other transformations).
      preprocessing_function: function that will be applied on each input.
          The function will run after the image is resized and augmented.
          The function should take one argument:
          one image (Numpy tensor with rank 3),
          and should output a Numpy tensor with the same shape.
      data_format: Image data format,
          either "channels_first" or "channels_last".
          "channels_last" mode means that the images should have shape
          `(samples, height, width, channels)`,
          "channels_first" mode means that the images should have shape
          `(samples, channels, height, width)`.
          It defaults to the `image_data_format` value found in your
          Keras config file at `~/.keras/keras.json`.
          If you never set it, then it will be "channels_last".
      validation_split: Float. Fraction of images reserved for validation
          (strictly between 0 and 1).
      dtype: Dtype to use for the generated arrays.

  Examples:

  Example of using `.flow(x, y)`:

  ```python
  (x_train, y_train), (x_test, y_test) = cifar10.load_data()
  y_train = np_utils.to_categorical(y_train, num_classes)
  y_test = np_utils.to_categorical(y_test, num_classes)
  datagen = ImageDataGenerator(
      featurewise_center=True,
      featurewise_std_normalization=True,
      rotation_range=20,
      width_shift_range=0.2,
      height_shift_range=0.2,
      horizontal_flip=True)
  # compute quantities required for featurewise normalization
  # (std, mean, and principal components if ZCA whitening is applied)
  datagen.fit(x_train)
  # fits the model on batches with real-time data augmentation:
  model.fit(datagen.flow(x_train, y_train, batch_size=32),
            steps_per_epoch=len(x_train) / 32, epochs=epochs)
  # here's a more "manual" example
  for e in range(epochs):
      print('Epoch', e)
      batches = 0
      for x_batch, y_batch in datagen.flow(x_train, y_train, batch_size=32):
          model.fit(x_batch, y_batch)
          batches += 1
          if batches >= len(x_train) / 32:
              # we need to break the loop by hand because
              # the generator loops indefinitely
              break
  ```

  Example of using `.flow_from_directory(directory)`:

  ```python
  train_datagen = ImageDataGenerator(
          rescale=1./255,
          shear_range=0.2,
          zoom_range=0.2,
          horizontal_flip=True)
  test_datagen = ImageDataGenerator(rescale=1./255)
  train_generator = train_datagen.flow_from_directory(
          'data/train',
          target_size=(150, 150),
          batch_size=32,
          class_mode='binary')
  validation_generator = test_datagen.flow_from_directory(
          'data/validation',
          target_size=(150, 150),
          batch_size=32,
          class_mode='binary')
  model.fit(
          train_generator,
          steps_per_epoch=2000,
          epochs=50,
          validation_data=validation_generator,
          validation_steps=800)
  ```

  Example of transforming images and masks together.

  ```python
  # we create two instances with the same arguments
  data_gen_args = dict(featurewise_center=True,
                       featurewise_std_normalization=True,
                       rotation_range=90,
                       width_shift_range=0.1,
                       height_shift_range=0.1,
                       zoom_range=0.2)
  image_datagen = ImageDataGenerator(**data_gen_args)
  mask_datagen = ImageDataGenerator(**data_gen_args)
  # Provide the same seed and keyword arguments to the fit and flow methods
  seed = 1
  image_datagen.fit(images, augment=True, seed=seed)
  mask_datagen.fit(masks, augment=True, seed=seed)
  image_generator = image_datagen.flow_from_directory(
      'data/images',
      class_mode=None,
      seed=seed)
  mask_generator = mask_datagen.flow_from_directory(
      'data/masks',
      class_mode=None,
      seed=seed)
  # combine generators into one which yields image and masks
  train_generator = zip(image_generator, mask_generator)
  model.fit(
      train_generator,
      steps_per_epoch=2000,
      epochs=50)
  ```
  """

  def __init__(self,
               featurewise_center=False,
               samplewise_center=False,
               featurewise_std_normalization=False,
               samplewise_std_normalization=False,
               zca_whitening=False,
               zca_epsilon=1e-6,
               rotation_range=0,
               width_shift_range=0.,
               height_shift_range=0.,
               brightness_range=None,
               shear_range=0.,
               zoom_range=0.,
               channel_shift_range=0.,
               fill_mode='nearest',
               cval=0.,
               horizontal_flip=False,
               vertical_flip=False,
               rescale=None,
               preprocessing_function=None,
               data_format=None,
               validation_split=0.0,
               dtype=None):
    if data_format is None:
      data_format = backend.image_data_format()
    kwargs = {}
    if 'dtype' in tf_inspect.getfullargspec(
        image.ImageDataGenerator.__init__)[0]:
      if dtype is None:
        dtype = backend.floatx()
      kwargs['dtype'] = dtype
    super(ImageDataGenerator, self).__init__(
        featurewise_center=featurewise_center,
        samplewise_center=samplewise_center,
        featurewise_std_normalization=featurewise_std_normalization,
        samplewise_std_normalization=samplewise_std_normalization,
        zca_whitening=zca_whitening,
        zca_epsilon=zca_epsilon,
        rotation_range=rotation_range,
        width_shift_range=width_shift_range,
        height_shift_range=height_shift_range,
        brightness_range=brightness_range,
        shear_range=shear_range,
        zoom_range=zoom_range,
        channel_shift_range=channel_shift_range,
        fill_mode=fill_mode,
        cval=cval,
        horizontal_flip=horizontal_flip,
        vertical_flip=vertical_flip,
        rescale=rescale,
        preprocessing_function=preprocessing_function,
        data_format=data_format,
        validation_split=validation_split,
        **kwargs)

  def flow(self,
           x,
           y=None,
           batch_size=32,
           shuffle=True,
           sample_weight=None,
           seed=None,
           save_to_dir=None,
           save_prefix='',
           save_format='png',
           subset=None):
    """Takes data & label arrays, generates batches of augmented data.

    Arguments:
        x: Input data. Numpy array of rank 4 or a tuple. If tuple, the first
          element should contain the images and the second element another numpy
          array or a list of numpy arrays that gets passed to the output without
          any modifications. Can be used to feed the model miscellaneous data
          along with the images. In case of grayscale data, the channels axis of
          the image array should have value 1, in case of RGB data, it should
          have value 3, and in case of RGBA data, it should have value 4.
        y: Labels.
        batch_size: Int (default: 32).
        shuffle: Boolean (default: True).
        sample_weight: Sample weights.
        seed: Int (default: None).
        save_to_dir: None or str (default: None). This allows you to optionally
          specify a directory to which to save the augmented pictures being
          generated (useful for visualizing what you are doing).
        save_prefix: Str (default: `''`). Prefix to use for filenames of saved
          pictures (only relevant if `save_to_dir` is set).
        save_format: one of "png", "jpeg"
            (only relevant if `save_to_dir` is set). Default: "png".
        subset: Subset of data (`"training"` or `"validation"`) if
          `validation_split` is set in `ImageDataGenerator`.

    Returns:
        An `Iterator` yielding tuples of `(x, y)`
            where `x` is a numpy array of image data
            (in the case of a single image input) or a list
            of numpy arrays (in the case with
            additional inputs) and `y` is a numpy array
            of corresponding labels. If 'sample_weight' is not None,
            the yielded tuples are of the form `(x, y, sample_weight)`.
            If `y` is None, only the numpy array `x` is returned.
    """
    return NumpyArrayIterator(
        x,
        y,
        self,
        batch_size=batch_size,
        shuffle=shuffle,
        sample_weight=sample_weight,
        seed=seed,
        data_format=self.data_format,
        save_to_dir=save_to_dir,
        save_prefix=save_prefix,
        save_format=save_format,
        subset=subset)

  def flow_from_directory(self,
                          directory,
                          target_size=(256, 256),
                          color_mode='rgb',
                          classes=None,
                          class_mode='categorical',
                          batch_size=32,
                          shuffle=True,
                          seed=None,
                          save_to_dir=None,
                          save_prefix='',
                          save_format='png',
                          follow_links=False,
                          subset=None,
                          interpolation='nearest'):
    """Takes the path to a directory & generates batches of augmented data.

    Arguments:
        directory: string, path to the target directory. It should contain one
          subdirectory per class. Any PNG, JPG, BMP, PPM or TIF images inside
          each of the subdirectories directory tree will be included in the
          generator. See [this script](
            https://gist.github.com/fchollet/0830affa1f7f19fd47b06d4cf89ed44d)
              for more details.
        target_size: Tuple of integers `(height, width)`, defaults to `(256,
          256)`. The dimensions to which all images found will be resized.
        color_mode: One of "grayscale", "rgb", "rgba". Default: "rgb". Whether
          the images will be converted to have 1, 3, or 4 channels.
        classes: Optional list of class subdirectories
            (e.g. `['dogs', 'cats']`). Default: None. If not provided, the list
              of classes will be automatically inferred from the subdirectory
              names/structure under `directory`, where each subdirectory will be
              treated as a different class (and the order of the classes, which
              will map to the label indices, will be alphanumeric). The
              dictionary containing the mapping from class names to class
              indices can be obtained via the attribute `class_indices`.
        class_mode: One of "categorical", "binary", "sparse",
            "input", or None. Default: "categorical".
            Determines the type of label arrays that are returned: -
              "categorical" will be 2D one-hot encoded labels, - "binary" will
              be 1D binary labels, "sparse" will be 1D integer labels, - "input"
              will be images identical to input images (mainly used to work with
              autoencoders). - If None, no labels are returned (the generator
              will only yield batches of image data, which is useful to use with
              `model.predict()`). Please note that in case of
              class_mode None, the data still needs to reside in a subdirectory
              of `directory` for it to work correctly.
        batch_size: Size of the batches of data (default: 32).
        shuffle: Whether to shuffle the data (default: True) If set to False,
          sorts the data in alphanumeric order.
        seed: Optional random seed for shuffling and transformations.
        save_to_dir: None or str (default: None). This allows you to optionally
          specify a directory to which to save the augmented pictures being
          generated (useful for visualizing what you are doing).
        save_prefix: Str. Prefix to use for filenames of saved pictures (only
          relevant if `save_to_dir` is set).
        save_format: One of "png", "jpeg"
            (only relevant if `save_to_dir` is set). Default: "png".
        follow_links: Whether to follow symlinks inside
            class subdirectories (default: False).
        subset: Subset of data (`"training"` or `"validation"`) if
          `validation_split` is set in `ImageDataGenerator`.
        interpolation: Interpolation method used to resample the image if the
          target size is different from that of the loaded image. Supported
          methods are `"nearest"`, `"bilinear"`, and `"bicubic"`. If PIL version
          1.1.3 or newer is installed, `"lanczos"` is also supported. If PIL
          version 3.4.0 or newer is installed, `"box"` and `"hamming"` are also
          supported. By default, `"nearest"` is used.

    Returns:
        A `DirectoryIterator` yielding tuples of `(x, y)`
            where `x` is a numpy array containing a batch
            of images with shape `(batch_size, *target_size, channels)`
            and `y` is a numpy array of corresponding labels.
    """
    return DirectoryIterator(
        directory,
        self,
        target_size=target_size,
        color_mode=color_mode,
        classes=classes,
        class_mode=class_mode,
        data_format=self.data_format,
        batch_size=batch_size,
        shuffle=shuffle,
        seed=seed,
        save_to_dir=save_to_dir,
        save_prefix=save_prefix,
        save_format=save_format,
        follow_links=follow_links,
        subset=subset,
        interpolation=interpolation)

  def flow_from_dataframe(self,
                          dataframe,
                          directory=None,
                          x_col='filename',
                          y_col='class',
                          weight_col=None,
                          target_size=(256, 256),
                          color_mode='rgb',
                          classes=None,
                          class_mode='categorical',
                          batch_size=32,
                          shuffle=True,
                          seed=None,
                          save_to_dir=None,
                          save_prefix='',
                          save_format='png',
                          subset=None,
                          interpolation='nearest',
                          validate_filenames=True,
                          **kwargs):
    """Takes the dataframe and the path to a directory + generates batches.

     The generated batches contain augmented/normalized data.

    **A simple tutorial can be found **[here](
                                http://bit.ly/keras_flow_from_dataframe).

    Arguments:
        dataframe: Pandas dataframe containing the filepaths relative to
          `directory` (or absolute paths if `directory` is None) of the images
          in a string column. It should include other column/s
            depending on the `class_mode`: - if `class_mode` is `"categorical"`
              (default value) it must include the `y_col` column with the
              class/es of each image. Values in column can be string/list/tuple
              if a single class or list/tuple if multiple classes. - if
              `class_mode` is `"binary"` or `"sparse"` it must include the given
              `y_col` column with class values as strings. - if `class_mode` is
              `"raw"` or `"multi_output"` it should contain the columns
              specified in `y_col`. - if `class_mode` is `"input"` or `None` no
              extra column is needed.
        directory: string, path to the directory to read images from. If `None`,
          data in `x_col` column should be absolute paths.
        x_col: string, column in `dataframe` that contains the filenames (or
          absolute paths if `directory` is `None`).
        y_col: string or list, column/s in `dataframe` that has the target data.
        weight_col: string, column in `dataframe` that contains the sample
            weights. Default: `None`.
        target_size: tuple of integers `(height, width)`, default: `(256, 256)`.
          The dimensions to which all images found will be resized.
        color_mode: one of "grayscale", "rgb", "rgba". Default: "rgb". Whether
          the images will be converted to have 1 or 3 color channels.
        classes: optional list of classes (e.g. `['dogs', 'cats']`). Default is
          None. If not provided, the list of classes will be automatically
          inferred from the `y_col`, which will map to the label indices, will
          be alphanumeric). The dictionary containing the mapping from class
          names to class indices can be obtained via the attribute
          `class_indices`.
        class_mode: one of "binary", "categorical", "input", "multi_output",
            "raw", sparse" or None. Default: "categorical".
            Mode for yielding the targets:
            - `"binary"`: 1D numpy array of binary labels,
            - `"categorical"`: 2D numpy array of one-hot encoded labels.
              Supports multi-label output.
            - `"input"`: images identical to input images (mainly used to work
              with autoencoders),
            - `"multi_output"`: list with the values of the different columns,
            - `"raw"`: numpy array of values in `y_col` column(s),
            - `"sparse"`: 1D numpy array of integer labels, - `None`, no targets
              are returned (the generator will only yield batches of image data,
              which is useful to use in `model.predict()`).
        batch_size: size of the batches of data (default: 32).
        shuffle: whether to shuffle the data (default: True)
        seed: optional random seed for shuffling and transformations.
        save_to_dir: None or str (default: None). This allows you to optionally
          specify a directory to which to save the augmented pictures being
          generated (useful for visualizing what you are doing).
        save_prefix: str. Prefix to use for filenames of saved pictures (only
          relevant if `save_to_dir` is set).
        save_format: one of "png", "jpeg"
            (only relevant if `save_to_dir` is set). Default: "png".
        subset: Subset of data (`"training"` or `"validation"`) if
          `validation_split` is set in `ImageDataGenerator`.
        interpolation: Interpolation method used to resample the image if the
          target size is different from that of the loaded image. Supported
          methods are `"nearest"`, `"bilinear"`, and `"bicubic"`. If PIL version
          1.1.3 or newer is installed, `"lanczos"` is also supported. If PIL
          version 3.4.0 or newer is installed, `"box"` and `"hamming"` are also
          supported. By default, `"nearest"` is used.
        validate_filenames: Boolean, whether to validate image filenames in
          `x_col`. If `True`, invalid images will be ignored. Disabling this
          option can lead to speed-up in the execution of this function.
          Defaults to `True`.
        **kwargs: legacy arguments for raising deprecation warnings.

    Returns:
        A `DataFrameIterator` yielding tuples of `(x, y)`
        where `x` is a numpy array containing a batch
        of images with shape `(batch_size, *target_size, channels)`
        and `y` is a numpy array of corresponding labels.
    """
    if 'has_ext' in kwargs:
      tf_logging.warn(
          'has_ext is deprecated, filenames in the dataframe have '
          'to match the exact filenames in disk.', DeprecationWarning)
    if 'sort' in kwargs:
      tf_logging.warn(
          'sort is deprecated, batches will be created in the'
          'same order than the filenames provided if shuffle'
          'is set to False.', DeprecationWarning)
    if class_mode == 'other':
      tf_logging.warn(
          '`class_mode` "other" is deprecated, please use '
          '`class_mode` "raw".', DeprecationWarning)
      class_mode = 'raw'
    if 'drop_duplicates' in kwargs:
      tf_logging.warn(
          'drop_duplicates is deprecated, you can drop duplicates '
          'by using the pandas.DataFrame.drop_duplicates method.',
          DeprecationWarning)

    return DataFrameIterator(
        dataframe,
        directory,
        self,
        x_col=x_col,
        y_col=y_col,
        weight_col=weight_col,
        target_size=target_size,
        color_mode=color_mode,
        classes=classes,
        class_mode=class_mode,
        data_format=self.data_format,
        batch_size=batch_size,
        shuffle=shuffle,
        seed=seed,
        save_to_dir=save_to_dir,
        save_prefix=save_prefix,
        save_format=save_format,
        subset=subset,
        interpolation=interpolation,
        validate_filenames=validate_filenames)


keras_export('keras.preprocessing.image.random_rotation')(random_rotation)
keras_export('keras.preprocessing.image.random_shift')(random_shift)
keras_export('keras.preprocessing.image.random_shear')(random_shear)
keras_export('keras.preprocessing.image.random_zoom')(random_zoom)
keras_export(
    'keras.preprocessing.image.apply_channel_shift')(apply_channel_shift)
keras_export(
    'keras.preprocessing.image.random_channel_shift')(random_channel_shift)
keras_export(
    'keras.preprocessing.image.apply_brightness_shift')(apply_brightness_shift)
keras_export('keras.preprocessing.image.random_brightness')(random_brightness)
keras_export(
    'keras.preprocessing.image.apply_affine_transform')(apply_affine_transform)
keras_export('keras.preprocessing.image.load_img')(load_img)
