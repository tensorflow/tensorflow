# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests for slim.data.tfexample_decoder."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np
import tensorflow as tf

import tensorflow.contrib.slim as slim


class TFExampleDecoderTest(tf.test.TestCase):

  def _EncodedFloatFeature(self, ndarray):
    return tf.train.Feature(float_list=tf.train.FloatList(
        value=ndarray.flatten().tolist()))

  def _EncodedInt64Feature(self, ndarray):
    return tf.train.Feature(int64_list=tf.train.Int64List(
        value=ndarray.flatten().tolist()))

  def _EncodedBytesFeature(self, tf_encoded):
    with self.test_session():
      encoded = tf_encoded.eval()

    def BytesList(value):
      return tf.train.BytesList(value=[value])

    return tf.train.Feature(bytes_list=BytesList(encoded))

  def _BytesFeature(self, ndarray):
    values = ndarray.flatten().tolist()
    for i in range(len(values)):
      values[i] = values[i].encode('utf-8')
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=values))

  def _StringFeature(self, value):
    value = value.encode('utf-8')
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

  def _Encoder(self, image, image_format):
    assert image_format  in ['jpeg', 'JPEG', 'png', 'PNG', 'raw', 'RAW']
    if image_format in ['jpeg', 'JPEG']:
      tf_image = tf.constant(image, dtype=tf.uint8)
      return tf.image.encode_jpeg(tf_image)
    if image_format in ['png', 'PNG']:
      tf_image = tf.constant(image, dtype=tf.uint8)
      return tf.image.encode_png(tf_image)
    if image_format in ['raw', 'RAW']:
      return tf.constant(image.tostring(), dtype=tf.string)

  def GenerateImage(self, image_format, image_shape):
    """Generates an image and an example containing the encoded image.

    Args:
      image_format: the encoding format of the image.
      image_shape: the shape of the image to generate.

    Returns:
      image: the generated image.
      example: a TF-example with a feature key 'image/encoded' set to the
        serialized image and a feature key 'image/format' set to the image
        encoding format ['jpeg', 'JPEG', 'png', 'PNG', 'raw'].
    """
    num_pixels = image_shape[0] * image_shape[1] * image_shape[2]
    image = np.linspace(0, num_pixels-1, num=num_pixels).reshape(
        image_shape).astype(np.uint8)
    tf_encoded = self._Encoder(image, image_format)
    example = tf.train.Example(features=tf.train.Features(feature={
        'image/encoded': self._EncodedBytesFeature(tf_encoded),
        'image/format': self._StringFeature(image_format)
    }))

    return image, example.SerializeToString()

  def DecodeExample(self, serialized_example, item_handler, image_format):
    """Decodes the given serialized example with the specified item handler.

    Args:
      serialized_example: a serialized TF example string.
      item_handler: the item handler used to decode the image.
      image_format: the image format being decoded.

    Returns:
      the decoded image found in the serialized Example.
    """
    serialized_example = tf.reshape(serialized_example, shape=[])
    decoder = slim.tfexample_decoder.TFExampleDecoder(
        keys_to_features={
            'image/encoded': tf.FixedLenFeature(
                (), tf.string, default_value=''),
            'image/format': tf.FixedLenFeature(
                (), tf.string, default_value=image_format),
        },
        items_to_handlers={'image': item_handler}
    )
    [tf_image] = decoder.decode(serialized_example, ['image'])
    return tf_image

  def RunDecodeExample(self, serialized_example, item_handler, image_format):
    tf_image = self.DecodeExample(serialized_example, item_handler,
                                  image_format)

    with self.test_session():
      decoded_image = tf_image.eval()

      # We need to recast them here to avoid some issues with uint8.
      return decoded_image.astype(np.float32)

  def testDecodeExampleWithJpegEncoding(self):
    image_shape = (2, 3, 3)
    image, serialized_example = self.GenerateImage(
        image_format='jpeg',
        image_shape=image_shape)

    decoded_image = self.RunDecodeExample(
        serialized_example,
        slim.tfexample_decoder.Image(),
        image_format='jpeg')

    # Need to use a tolerance of 1 because of noise in the jpeg encode/decode
    self.assertAllClose(image, decoded_image, atol=1.001)

  def testDecodeExampleWithJPEGEncoding(self):
    test_image_channels = [1, 3]
    for channels in test_image_channels:
      image_shape = (2, 3, channels)
      image, serialized_example = self.GenerateImage(
          image_format='JPEG',
          image_shape=image_shape)

      decoded_image = self.RunDecodeExample(
          serialized_example,
          slim.tfexample_decoder.Image(channels=channels),
          image_format='JPEG')

      # Need to use a tolerance of 1 because of noise in the jpeg encode/decode
      self.assertAllClose(image, decoded_image, atol=1.001)

  def testDecodeExampleWithNoShapeInfo(self):
    test_image_channels = [1, 3]
    for channels in test_image_channels:
      image_shape = (2, 3, channels)
      _, serialized_example = self.GenerateImage(
          image_format='jpeg',
          image_shape=image_shape)

      tf_decoded_image = self.DecodeExample(
          serialized_example,
          slim.tfexample_decoder.Image(shape=None, channels=channels),
          image_format='jpeg')
      self.assertEqual(tf_decoded_image.get_shape().ndims, 3)

  def testDecodeExampleWithPngEncoding(self):
    test_image_channels = [1, 3]
    for channels in test_image_channels:
      image_shape = (2, 3, channels)
      image, serialized_example = self.GenerateImage(
          image_format='png',
          image_shape=image_shape)

      decoded_image = self.RunDecodeExample(
          serialized_example,
          slim.tfexample_decoder.Image(channels=channels),
          image_format='png')

      self.assertAllClose(image, decoded_image, atol=0)

  def testDecodeExampleWithPNGEncoding(self):
    test_image_channels = [1, 3]
    for channels in test_image_channels:
      image_shape = (2, 3, channels)
      image, serialized_example = self.GenerateImage(
          image_format='PNG',
          image_shape=image_shape)

      decoded_image = self.RunDecodeExample(
          serialized_example,
          slim.tfexample_decoder.Image(channels=channels),
          image_format='PNG')

      self.assertAllClose(image, decoded_image, atol=0)

  def testDecodeExampleWithRawEncoding(self):
    image_shape = (2, 3, 3)
    image, serialized_example = self.GenerateImage(
        image_format='raw',
        image_shape=image_shape)

    decoded_image = self.RunDecodeExample(
        serialized_example,
        slim.tfexample_decoder.Image(shape=image_shape),
        image_format='raw')

    self.assertAllClose(image, decoded_image, atol=0)

  def testDecodeExampleWithRAWEncoding(self):
    image_shape = (2, 3, 3)
    image, serialized_example = self.GenerateImage(
        image_format='RAW',
        image_shape=image_shape)

    decoded_image = self.RunDecodeExample(
        serialized_example,
        slim.tfexample_decoder.Image(shape=image_shape),
        image_format='RAW')

    self.assertAllClose(image, decoded_image, atol=0)

  def testDecodeExampleWithStringTensor(self):
    tensor_shape = (2, 3, 1)
    np_array = np.array([[['ab'], ['cd'], ['ef']],
                         [['ghi'], ['jkl'], ['mnop']]])

    example = tf.train.Example(features=tf.train.Features(feature={
        'labels': self._BytesFeature(np_array),
    }))

    serialized_example = example.SerializeToString()

    with self.test_session():
      serialized_example = tf.reshape(serialized_example, shape=[])
      keys_to_features = {
          'labels': tf.FixedLenFeature(
              tensor_shape, tf.string, default_value=tf.constant(
                  '', shape=tensor_shape, dtype=tf.string))
      }
      items_to_handlers = {
          'labels': slim.tfexample_decoder.Tensor('labels'),
      }
      decoder = slim.tfexample_decoder.TFExampleDecoder(
          keys_to_features, items_to_handlers)
      [tf_labels] = decoder.decode(serialized_example, ['labels'])
      labels = tf_labels.eval()

      labels = labels.astype(np_array.dtype)
      self.assertTrue(np.array_equal(np_array, labels))

  def testDecodeExampleWithFloatTensor(self):
    np_array = np.random.rand(2, 3, 1).astype('f')

    example = tf.train.Example(features=tf.train.Features(feature={
        'array': self._EncodedFloatFeature(np_array),
    }))

    serialized_example = example.SerializeToString()

    with self.test_session():
      serialized_example = tf.reshape(serialized_example, shape=[])
      keys_to_features = {
          'array': tf.FixedLenFeature(np_array.shape, tf.float32)
      }
      items_to_handlers = {
          'array': slim.tfexample_decoder.Tensor('array'),
      }
      decoder = slim.tfexample_decoder.TFExampleDecoder(
          keys_to_features, items_to_handlers)
      [tf_array] = decoder.decode(serialized_example, ['array'])
      self.assertAllEqual(tf_array.eval(), np_array)

  def testDecodeExampleWithInt64Tensor(self):
    np_array = np.random.randint(1, 10, size=(2, 3, 1))

    example = tf.train.Example(features=tf.train.Features(feature={
        'array': self._EncodedInt64Feature(np_array),
    }))

    serialized_example = example.SerializeToString()

    with self.test_session():
      serialized_example = tf.reshape(serialized_example, shape=[])
      keys_to_features = {
          'array': tf.FixedLenFeature(np_array.shape, tf.int64)
      }
      items_to_handlers = {
          'array': slim.tfexample_decoder.Tensor('array'),
      }
      decoder = slim.tfexample_decoder.TFExampleDecoder(
          keys_to_features, items_to_handlers)
      [tf_array] = decoder.decode(serialized_example, ['array'])
      self.assertAllEqual(tf_array.eval(), np_array)

  def testDecodeExampleWithVarLenTensor(self):
    np_array = np.array([[[1], [2], [3]],
                         [[4], [5], [6]]])

    example = tf.train.Example(features=tf.train.Features(feature={
        'labels': self._EncodedInt64Feature(np_array),
    }))

    serialized_example = example.SerializeToString()

    with self.test_session():
      serialized_example = tf.reshape(serialized_example, shape=[])
      keys_to_features = {
          'labels': tf.VarLenFeature(dtype=tf.int64),
      }
      items_to_handlers = {
          'labels': slim.tfexample_decoder.Tensor('labels'),
      }
      decoder = slim.tfexample_decoder.TFExampleDecoder(
          keys_to_features, items_to_handlers)
      [tf_labels] = decoder.decode(serialized_example, ['labels'])
      labels = tf_labels.eval()
      self.assertAllEqual(labels, np_array.flatten())

  def testDecodeExampleWithFixLenTensorWithShape(self):
    np_array = np.array([[1, 2, 3],
                         [4, 5, 6]])

    example = tf.train.Example(features=tf.train.Features(feature={
        'labels': self._EncodedInt64Feature(np_array),
    }))

    serialized_example = example.SerializeToString()

    with self.test_session():
      serialized_example = tf.reshape(serialized_example, shape=[])
      keys_to_features = {
          'labels': tf.FixedLenFeature(np_array.shape, dtype=tf.int64),
      }
      items_to_handlers = {
          'labels': slim.tfexample_decoder.Tensor('labels',
                                                  shape=np_array.shape),
      }
      decoder = slim.tfexample_decoder.TFExampleDecoder(
          keys_to_features, items_to_handlers)
      [tf_labels] = decoder.decode(serialized_example, ['labels'])
      labels = tf_labels.eval()
      self.assertAllEqual(labels, np_array)

  def testDecodeExampleWithVarLenTensorToDense(self):
    np_array = np.array([[1, 2, 3],
                         [4, 5, 6]])
    example = tf.train.Example(features=tf.train.Features(feature={
        'labels': self._EncodedInt64Feature(np_array),
    }))

    serialized_example = example.SerializeToString()

    with self.test_session():
      serialized_example = tf.reshape(serialized_example, shape=[])
      keys_to_features = {
          'labels': tf.VarLenFeature(dtype=tf.int64),
      }
      items_to_handlers = {
          'labels': slim.tfexample_decoder.Tensor('labels',
                                                  shape=np_array.shape),
      }
      decoder = slim.tfexample_decoder.TFExampleDecoder(
          keys_to_features, items_to_handlers)
      [tf_labels] = decoder.decode(serialized_example, ['labels'])
      labels = tf_labels.eval()
      self.assertAllEqual(labels, np_array)

  def testDecodeExampleShapeKeyTensor(self):
    np_image = np.random.rand(2, 3, 1).astype('f')
    np_labels = np.array([[[1], [2], [3]],
                          [[4], [5], [6]]])

    example = tf.train.Example(features=tf.train.Features(feature={
        'image': self._EncodedFloatFeature(np_image),
        'image/shape': self._EncodedInt64Feature(np.array(np_image.shape)),
        'labels': self._EncodedInt64Feature(np_labels),
        'labels/shape': self._EncodedInt64Feature(np.array(np_labels.shape)),

    }))

    serialized_example = example.SerializeToString()

    with self.test_session():
      serialized_example = tf.reshape(serialized_example, shape=[])
      keys_to_features = {
          'image': tf.VarLenFeature(dtype=tf.float32),
          'image/shape': tf.VarLenFeature(dtype=tf.int64),
          'labels': tf.VarLenFeature(dtype=tf.int64),
          'labels/shape': tf.VarLenFeature(dtype=tf.int64),
      }
      items_to_handlers = {
          'image': slim.tfexample_decoder.Tensor('image',
                                                 shape_keys='image/shape'),
          'labels': slim.tfexample_decoder.Tensor('labels',
                                                  shape_keys='labels/shape'),
      }
      decoder = slim.tfexample_decoder.TFExampleDecoder(
          keys_to_features, items_to_handlers)
      [tf_image, tf_labels] = decoder.decode(serialized_example,
                                             ['image', 'labels'])
      self.assertAllEqual(tf_image.eval(), np_image)
      self.assertAllEqual(tf_labels.eval(), np_labels)

  def testDecodeExampleMultiShapeKeyTensor(self):
    np_image = np.random.rand(2, 3, 1).astype('f')
    np_labels = np.array([[[1], [2], [3]],
                          [[4], [5], [6]]])
    height, width, depth = np_labels.shape

    example = tf.train.Example(features=tf.train.Features(feature={
        'image': self._EncodedFloatFeature(np_image),
        'image/shape': self._EncodedInt64Feature(np.array(np_image.shape)),
        'labels': self._EncodedInt64Feature(np_labels),
        'labels/height': self._EncodedInt64Feature(np.array([height])),
        'labels/width': self._EncodedInt64Feature(np.array([width])),
        'labels/depth': self._EncodedInt64Feature(np.array([depth])),
    }))

    serialized_example = example.SerializeToString()

    with self.test_session():
      serialized_example = tf.reshape(serialized_example, shape=[])
      keys_to_features = {
          'image': tf.VarLenFeature(dtype=tf.float32),
          'image/shape': tf.VarLenFeature(dtype=tf.int64),
          'labels': tf.VarLenFeature(dtype=tf.int64),
          'labels/height': tf.VarLenFeature(dtype=tf.int64),
          'labels/width': tf.VarLenFeature(dtype=tf.int64),
          'labels/depth': tf.VarLenFeature(dtype=tf.int64),
      }
      items_to_handlers = {
          'image': slim.tfexample_decoder.Tensor(
              'image', shape_keys='image/shape'),
          'labels': slim.tfexample_decoder.Tensor(
              'labels',
              shape_keys=['labels/height', 'labels/width', 'labels/depth']),
      }
      decoder = slim.tfexample_decoder.TFExampleDecoder(
          keys_to_features, items_to_handlers)
      [tf_image, tf_labels] = decoder.decode(serialized_example,
                                             ['image', 'labels'])
      self.assertAllEqual(tf_image.eval(), np_image)
      self.assertAllEqual(tf_labels.eval(), np_labels)

  def testDecodeExampleWithSparseTensor(self):
    np_indices = np.array([[1], [2], [5]])
    np_values = np.array([0.1, 0.2, 0.6]).astype('f')
    example = tf.train.Example(features=tf.train.Features(feature={
        'indices': self._EncodedInt64Feature(np_indices),
        'values': self._EncodedFloatFeature(np_values),
    }))

    serialized_example = example.SerializeToString()

    with self.test_session():
      serialized_example = tf.reshape(serialized_example, shape=[])
      keys_to_features = {
          'indices': tf.VarLenFeature(dtype=tf.int64),
          'values': tf.VarLenFeature(dtype=tf.float32),
      }
      items_to_handlers = {
          'labels': slim.tfexample_decoder.SparseTensor(),
      }
      decoder = slim.tfexample_decoder.TFExampleDecoder(
          keys_to_features, items_to_handlers)
      [tf_labels] = decoder.decode(serialized_example, ['labels'])
      labels = tf_labels.eval()
      self.assertAllEqual(labels.indices, np_indices)
      self.assertAllEqual(labels.values, np_values)
      self.assertAllEqual(labels.shape, np_values.shape)

  def testDecodeExampleWithSparseTensorWithKeyShape(self):
    np_indices = np.array([[1], [2], [5]])
    np_values = np.array([0.1, 0.2, 0.6]).astype('f')
    np_shape = np.array([6])
    example = tf.train.Example(features=tf.train.Features(feature={
        'indices': self._EncodedInt64Feature(np_indices),
        'values': self._EncodedFloatFeature(np_values),
        'shape': self._EncodedInt64Feature(np_shape),
    }))

    serialized_example = example.SerializeToString()

    with self.test_session():
      serialized_example = tf.reshape(serialized_example, shape=[])
      keys_to_features = {
          'indices': tf.VarLenFeature(dtype=tf.int64),
          'values': tf.VarLenFeature(dtype=tf.float32),
          'shape': tf.VarLenFeature(dtype=tf.int64),
      }
      items_to_handlers = {
          'labels': slim.tfexample_decoder.SparseTensor(shape_key='shape'),
      }
      decoder = slim.tfexample_decoder.TFExampleDecoder(
          keys_to_features, items_to_handlers)
      [tf_labels] = decoder.decode(serialized_example, ['labels'])
      labels = tf_labels.eval()
      self.assertAllEqual(labels.indices, np_indices)
      self.assertAllEqual(labels.values, np_values)
      self.assertAllEqual(labels.shape, np_shape)

  def testDecodeExampleWithSparseTensorWithGivenShape(self):
    np_indices = np.array([[1], [2], [5]])
    np_values = np.array([0.1, 0.2, 0.6]).astype('f')
    np_shape = np.array([6])
    example = tf.train.Example(features=tf.train.Features(feature={
        'indices': self._EncodedInt64Feature(np_indices),
        'values': self._EncodedFloatFeature(np_values),
    }))

    serialized_example = example.SerializeToString()

    with self.test_session():
      serialized_example = tf.reshape(serialized_example, shape=[])
      keys_to_features = {
          'indices': tf.VarLenFeature(dtype=tf.int64),
          'values': tf.VarLenFeature(dtype=tf.float32),
      }
      items_to_handlers = {
          'labels': slim.tfexample_decoder.SparseTensor(shape=np_shape),
      }
      decoder = slim.tfexample_decoder.TFExampleDecoder(
          keys_to_features, items_to_handlers)
      [tf_labels] = decoder.decode(serialized_example, ['labels'])
      labels = tf_labels.eval()
      self.assertAllEqual(labels.indices, np_indices)
      self.assertAllEqual(labels.values, np_values)
      self.assertAllEqual(labels.shape, np_shape)

  def testDecodeExampleWithSparseTensorToDense(self):
    np_indices = np.array([1, 2, 5])
    np_values = np.array([0.1, 0.2, 0.6]).astype('f')
    np_shape = np.array([6])
    np_dense = np.array([0.0, 0.1, 0.2, 0.0, 0.0, 0.6]).astype('f')
    example = tf.train.Example(features=tf.train.Features(feature={
        'indices': self._EncodedInt64Feature(np_indices),
        'values': self._EncodedFloatFeature(np_values),
    }))

    serialized_example = example.SerializeToString()

    with self.test_session():
      serialized_example = tf.reshape(serialized_example, shape=[])
      keys_to_features = {
          'indices': tf.VarLenFeature(dtype=tf.int64),
          'values': tf.VarLenFeature(dtype=tf.float32),
      }
      items_to_handlers = {
          'labels': slim.tfexample_decoder.SparseTensor(shape=np_shape,
                                                        densify=True),
      }
      decoder = slim.tfexample_decoder.TFExampleDecoder(
          keys_to_features, items_to_handlers)
      [tf_labels] = decoder.decode(serialized_example, ['labels'])
      labels = tf_labels.eval()
      self.assertAllClose(labels, np_dense)

  def testDecodeExampleWithTensor(self):
    tensor_shape = (2, 3, 1)
    np_array = np.random.rand(2, 3, 1)

    example = tf.train.Example(features=tf.train.Features(feature={
        'image/depth_map': self._EncodedFloatFeature(np_array),
    }))

    serialized_example = example.SerializeToString()

    with self.test_session():
      serialized_example = tf.reshape(serialized_example, shape=[])

      keys_to_features = {
          'image/depth_map': tf.FixedLenFeature(
              tensor_shape, tf.float32, default_value=tf.zeros(tensor_shape))
      }

      items_to_handlers = {
          'depth': slim.tfexample_decoder.Tensor('image/depth_map')
      }

      decoder = slim.tfexample_decoder.TFExampleDecoder(
          keys_to_features, items_to_handlers)
      [tf_depth] = decoder.decode(serialized_example, ['depth'])
      depth = tf_depth.eval()

    self.assertAllClose(np_array, depth)

  def testDecodeExampleWithItemHandlerCallback(self):
    np.random.seed(0)
    tensor_shape = (2, 3, 1)
    np_array = np.random.rand(2, 3, 1)

    example = tf.train.Example(features=tf.train.Features(feature={
        'image/depth_map': self._EncodedFloatFeature(np_array),
    }))

    serialized_example = example.SerializeToString()

    with self.test_session():
      serialized_example = tf.reshape(serialized_example, shape=[])

      keys_to_features = {
          'image/depth_map': tf.FixedLenFeature(
              tensor_shape, tf.float32, default_value=tf.zeros(tensor_shape))
      }

      def HandleDepth(keys_to_tensors):
        depth = list(keys_to_tensors.values())[0]
        depth += 1
        return depth

      items_to_handlers = {
          'depth': slim.tfexample_decoder.ItemHandlerCallback(
              'image/depth_map', HandleDepth)
      }

      decoder = slim.tfexample_decoder.TFExampleDecoder(
          keys_to_features, items_to_handlers)
      [tf_depth] = decoder.decode(serialized_example, ['depth'])
      depth = tf_depth.eval()

    self.assertAllClose(np_array, depth-1)

  def testDecodeImageWithItemHandlerCallback(self):
    image_shape = (2, 3, 3)
    for image_encoding in ['jpeg', 'png']:
      image, serialized_example = self.GenerateImage(
          image_format=image_encoding,
          image_shape=image_shape)

      with self.test_session():

        def ConditionalDecoding(keys_to_tensors):
          """See base class."""
          image_buffer = keys_to_tensors['image/encoded']
          image_format = keys_to_tensors['image/format']

          def DecodePng():
            return tf.image.decode_png(image_buffer, 3)
          def DecodeJpg():
            return tf.image.decode_jpeg(image_buffer, 3)

          image = tf.case({
              tf.equal(image_format, 'png'): DecodePng,
          }, default=DecodeJpg, exclusive=True)
          image = tf.reshape(image, image_shape)
          return image

        keys_to_features = {
            'image/encoded': tf.FixedLenFeature(
                (), tf.string, default_value=''),
            'image/format': tf.FixedLenFeature(
                (), tf.string, default_value='jpeg')
        }

        items_to_handlers = {
            'image': slim.tfexample_decoder.ItemHandlerCallback(
                ['image/encoded', 'image/format'], ConditionalDecoding)
        }

        decoder = slim.tfexample_decoder.TFExampleDecoder(
            keys_to_features, items_to_handlers)
        [tf_image] = decoder.decode(serialized_example, ['image'])
        decoded_image = tf_image.eval()
        if image_encoding == 'jpeg':
          # For jenkins:
          image = image.astype(np.float32)
          decoded_image = decoded_image.astype(np.float32)
          self.assertAllClose(image, decoded_image, rtol=.5, atol=1.001)
        else:
          self.assertAllClose(image, decoded_image, atol=0)

  def testDecodeExampleWithBoundingBox(self):
    num_bboxes = 10
    np_ymin = np.random.rand(num_bboxes, 1)
    np_xmin = np.random.rand(num_bboxes, 1)
    np_ymax = np.random.rand(num_bboxes, 1)
    np_xmax = np.random.rand(num_bboxes, 1)
    np_bboxes = np.hstack([np_ymin, np_xmin, np_ymax, np_xmax])

    example = tf.train.Example(features=tf.train.Features(feature={
        'image/object/bbox/ymin': self._EncodedFloatFeature(np_ymin),
        'image/object/bbox/xmin': self._EncodedFloatFeature(np_xmin),
        'image/object/bbox/ymax': self._EncodedFloatFeature(np_ymax),
        'image/object/bbox/xmax': self._EncodedFloatFeature(np_xmax),
    }))
    serialized_example = example.SerializeToString()

    with self.test_session():
      serialized_example = tf.reshape(serialized_example, shape=[])

      keys_to_features = {
          'image/object/bbox/ymin': tf.VarLenFeature(tf.float32),
          'image/object/bbox/xmin': tf.VarLenFeature(tf.float32),
          'image/object/bbox/ymax': tf.VarLenFeature(tf.float32),
          'image/object/bbox/xmax': tf.VarLenFeature(tf.float32),
      }

      items_to_handlers = {
          'object/bbox': slim.tfexample_decoder.BoundingBox(
              ['ymin', 'xmin', 'ymax', 'xmax'], 'image/object/bbox/'),
      }

      decoder = slim.tfexample_decoder.TFExampleDecoder(
          keys_to_features,
          items_to_handlers)
      [tf_bboxes] = decoder.decode(serialized_example, ['object/bbox'])
      bboxes = tf_bboxes.eval()

    self.assertAllClose(np_bboxes, bboxes)

if __name__ == '__main__':
  tf.test.main()
