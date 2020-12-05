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

"""Bounding boxes feature."""

import collections
from typing import Any

import numpy as np
import tensorflow.compat.v2 as tf

from tensorflow.data.experimental.core import lazy_imports_lib
from tensorflow.data.experimental.core import utils
from tensorflow.data.experimental.core.features import feature
from tensorflow.data.experimental.core.features import image_feature
from tensorflow.data.experimental.core.utils import type_utils

Json = type_utils.Json

BBox = collections.namedtuple('BBox', 'ymin, xmin, ymax, xmax')

PilImage = Any


class BBoxFeature(feature.Tensor):
  """`FeatureConnector` for a normalized bounding box.

  Note: If you have multiple bounding boxes, you may want to wrap the feature
  inside a `tfds.feature.Sequence`.

  Input:
    * `tfds.features.BBox` tuple.

  Output:
    bbox: tf.Tensor of type `tf.float32` and shape `[4,]` which contains the
      normalized coordinates of the bounding box `[ymin, xmin, ymax, xmax]`

  Example:
    * In the DatasetInfo object:

    ```
    features=features.FeatureDict({
        'bbox': features.BBox(shape=(None, 64, 64, 3)),
    })
    ```

    * During generation:

    ```
    yield {
        'input': tfds.feature.BBox(ymin=0.3, xmin=0.8, ymax=0.5, xmax=1.0),
    }
    ```
  """

  def __init__(self):
    super(BBoxFeature, self).__init__(shape=(4,), dtype=tf.float32)

  def encode_example(self, bbox):
    """See base class for details."""
    # Validate the coordinates
    for coordinate in bbox:
      if not isinstance(coordinate, float):
        raise ValueError(
            'BBox coordinates should be float. Got {}.'.format(bbox))
      if not 0.0 <= coordinate <= 1.0:
        raise ValueError(
            'BBox coordinates should be between 0 and 1. Got {}.'.format(bbox))
      if bbox.xmax < bbox.xmin or bbox.ymax < bbox.ymin:
        raise ValueError(
            'BBox coordinates should have min <= max. Got {}.'.format(bbox))

    return super(BBoxFeature, self).encode_example(
        [bbox.ymin, bbox.xmin, bbox.ymax, bbox.xmax]
    )

  def repr_html(self, ex: np.ndarray) -> str:
    """Returns the HTML str representation of an Image with BBoxes."""
    ex = np.expand_dims(ex, axis=0)  # Expand single bounding box to batch.
    return _repr_html(ex)

  def repr_html_batch(self, ex: np.ndarray) -> str:
    """Returns the HTML str representation of an Image with BBoxes (Sequence)."""
    return _repr_html(ex)

  @classmethod
  def from_json_content(cls, value: Json) -> 'BBoxFeature':
    del value  # Unused
    return cls()

  def to_json_content(self) -> Json:
    return dict()


def _repr_html(ex: np.ndarray) -> str:
  """Returns the HTML str representation of an Image with BBoxes."""
  img = _build_thumbnail_with_bbox(ex)
  img_str = utils.get_base64(lambda buff: img.save(buff, format='PNG'))
  return f'<img src="data:image/png;base64,{img_str}" alt="Img" />'


def _build_thumbnail_with_bbox(ex: np.ndarray) -> PilImage:
  """Returns blank image with Bboxes drawn on it."""
  PIL_Image = lazy_imports_lib.lazy_imports.PIL_Image  # pylint: disable=invalid-name
  PIL_ImageDraw = lazy_imports_lib.lazy_imports.PIL_ImageDraw  # pylint: disable=invalid-name

  shape = (image_feature.THUMBNAIL_SIZE, image_feature.THUMBNAIL_SIZE)
  blank_img = PIL_Image.new('RGB', shape, (255, 255, 255))
  draw = PIL_ImageDraw.Draw(blank_img)
  rs = np.random.RandomState(97531)  # freeze random state

  for i in range(ex.shape[0]):
    # Rescale coordinates to match size of blank_image
    ymin, xmin, ymax, xmax = ex[i, :] * image_feature.THUMBNAIL_SIZE
    # Generate random rgb values for Bbox ouline
    r, g, b = list(rs.randint(0, 256, size=3))
    draw.rectangle(((xmin, ymin), (xmax, ymax)), outline=(r, g, b))
  return blank_img
