# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for slim.data.tfexample_decoder."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.contrib.slim.python.slim.data import tfexample_decoder
from tensorflow.core.example import example_pb2
from tensorflow.core.example import feature_pb2
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import image_ops
from tensorflow.python.ops import lookup_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import parsing_ops
from tensorflow.python.platform import test


class TFExampleDecoderTest(test.TestCase):

  def _EncodedFloatFeature(self, ndarray):
    return feature_pb2.Feature(float_list=feature_pb2.FloatList(
        value=ndarray.flatten().tolist()))

  def _EncodedInt64Feature(self, ndarray):
    return feature_pb2.Feature(int64_list=feature_pb2.Int64List(
        value=ndarray.flatten().tolist()))

  def _EncodedBytesFeature(self, tf_encoded):
    with self.test_session():
      encoded = tf_encoded.eval()

    def BytesList(value):
      return feature_pb2.BytesList(value=[value])

    return feature_pb2.Feature(bytes_list=BytesList(encoded))

  def _BytesFeature(self, ndarray):
    values = ndarray.flatten().tolist()
    for i in range(len(values)):
      values[i] = values[i].encode('utf-8')
    return feature_pb2.Feature(bytes_list=feature_pb2.BytesList(value=values))

  def _StringFeature(self, value):
    value = value.encode('utf-8')
    return feature_pb2.Feature(bytes_list=feature_pb2.BytesList(value=[value]))

  def _Encoder(self, image, image_format):
    assert image_format in ['jpeg', 'JPEG', 'png', 'PNG', 'raw', 'RAW']
    if image_format in ['jpeg', 'JPEG']:
      tf_image = constant_op.constant(image, dtype=dtypes.uint8)
      return image_ops.encode_jpeg(tf_image)
    if image_format in ['png', 'PNG']:
      tf_image = constant_op.constant(image, dtype=dtypes.uint8)
      return image_ops.encode_png(tf_image)
    if image_format in ['raw', 'RAW']:
      return constant_op.constant(image.tostring(), dtype=dtypes.string)

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
    image = np.linspace(
        0, num_pixels - 1, num=num_pixels).reshape(image_shape).astype(np.uint8)
    tf_encoded = self._Encoder(image, image_format)
    example = example_pb2.Example(features=feature_pb2.Features(feature={
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
    serialized_example = array_ops.reshape(serialized_example, shape=[])
    decoder = tfexample_decoder.TFExampleDecoder(
        keys_to_features={
            'image/encoded':
                parsing_ops.FixedLenFeature(
                    (), dtypes.string, default_value=''),
            'image/format':
                parsing_ops.FixedLenFeature(
                    (), dtypes.string, default_value=image_format),
        },
        items_to_handlers={'image': item_handler})
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
        image_format='jpeg', image_shape=image_shape)

    decoded_image = self.RunDecodeExample(
        serialized_example, tfexample_decoder.Image(), image_format='jpeg')

    # Need to use a tolerance of 1 because of noise in the jpeg encode/decode
    self.assertAllClose(image, decoded_image, atol=1.001)

  def testDecodeExampleWithJPEGEncoding(self):
    test_image_channels = [1, 3]
    for channels in test_image_channels:
      image_shape = (2, 3, channels)
      image, serialized_example = self.GenerateImage(
          image_format='JPEG', image_shape=image_shape)

      decoded_image = self.RunDecodeExample(
          serialized_example,
          tfexample_decoder.Image(channels=channels),
          image_format='JPEG')

      # Need to use a tolerance of 1 because of noise in the jpeg encode/decode
      self.assertAllClose(image, decoded_image, atol=1.001)

  def testDecodeExampleWithNoShapeInfo(self):
    test_image_channels = [1, 3]
    for channels in test_image_channels:
      image_shape = (2, 3, channels)
      _, serialized_example = self.GenerateImage(
          image_format='jpeg', image_shape=image_shape)

      tf_decoded_image = self.DecodeExample(
          serialized_example,
          tfexample_decoder.Image(
              shape=None, channels=channels),
          image_format='jpeg')
      self.assertEqual(tf_decoded_image.get_shape().ndims, 3)

  def testDecodeExampleWithPngEncoding(self):
    test_image_channels = [1, 3, 4]
    for channels in test_image_channels:
      image_shape = (2, 3, channels)
      image, serialized_example = self.GenerateImage(
          image_format='png', image_shape=image_shape)

      decoded_image = self.RunDecodeExample(
          serialized_example,
          tfexample_decoder.Image(channels=channels),
          image_format='png')

      self.assertAllClose(image, decoded_image, atol=0)

  def testDecodeExampleWithPNGEncoding(self):
    test_image_channels = [1, 3, 4]
    for channels in test_image_channels:
      image_shape = (2, 3, channels)
      image, serialized_example = self.GenerateImage(
          image_format='PNG', image_shape=image_shape)

      decoded_image = self.RunDecodeExample(
          serialized_example,
          tfexample_decoder.Image(channels=channels),
          image_format='PNG')

      self.assertAllClose(image, decoded_image, atol=0)

  def testDecodeExampleWithRawEncoding(self):
    image_shape = (2, 3, 3)
    image, serialized_example = self.GenerateImage(
        image_format='raw', image_shape=image_shape)

    decoded_image = self.RunDecodeExample(
        serialized_example,
        tfexample_decoder.Image(shape=image_shape),
        image_format='raw')

    self.assertAllClose(image, decoded_image, atol=0)

  def testDecodeExampleWithRAWEncoding(self):
    image_shape = (2, 3, 3)
    image, serialized_example = self.GenerateImage(
        image_format='RAW', image_shape=image_shape)

    decoded_image = self.RunDecodeExample(
        serialized_example,
        tfexample_decoder.Image(shape=image_shape),
        image_format='RAW')

    self.assertAllClose(image, decoded_image, atol=0)

  def testDecodeExampleWithJpegEncodingAt16BitCausesError(self):
    image_shape = (2, 3, 3)
    unused_image, serialized_example = self.GenerateImage(
        image_format='jpeg', image_shape=image_shape)
    # decode_raw support uint16 now so ValueError will be thrown instead.
    with self.assertRaisesRegexp(
        ValueError,
        'true_fn and false_fn must have the same type: uint16, uint8'):
      unused_decoded_image = self.RunDecodeExample(
          serialized_example,
          tfexample_decoder.Image(dtype=dtypes.uint16),
          image_format='jpeg')

  def testDecodeExampleWithStringTensor(self):
    tensor_shape = (2, 3, 1)
    np_array = np.array([[['ab'], ['cd'], ['ef']],
                         [['ghi'], ['jkl'], ['mnop']]])

    example = example_pb2.Example(features=feature_pb2.Features(feature={
        'labels': self._BytesFeature(np_array),
    }))

    serialized_example = example.SerializeToString()

    with self.test_session():
      serialized_example = array_ops.reshape(serialized_example, shape=[])
      keys_to_features = {
          'labels':
              parsing_ops.FixedLenFeature(
                  tensor_shape,
                  dtypes.string,
                  default_value=constant_op.constant(
                      '', shape=tensor_shape, dtype=dtypes.string))
      }
      items_to_handlers = {'labels': tfexample_decoder.Tensor('labels'),}
      decoder = tfexample_decoder.TFExampleDecoder(keys_to_features,
                                                   items_to_handlers)
      [tf_labels] = decoder.decode(serialized_example, ['labels'])
      labels = tf_labels.eval()

      labels = labels.astype(np_array.dtype)
      self.assertTrue(np.array_equal(np_array, labels))

  def testDecodeExampleWithFloatTensor(self):
    np_array = np.random.rand(2, 3, 1).astype('f')

    example = example_pb2.Example(features=feature_pb2.Features(feature={
        'array': self._EncodedFloatFeature(np_array),
    }))

    serialized_example = example.SerializeToString()

    with self.test_session():
      serialized_example = array_ops.reshape(serialized_example, shape=[])
      keys_to_features = {
          'array': parsing_ops.FixedLenFeature(np_array.shape, dtypes.float32)
      }
      items_to_handlers = {'array': tfexample_decoder.Tensor('array'),}
      decoder = tfexample_decoder.TFExampleDecoder(keys_to_features,
                                                   items_to_handlers)
      [tf_array] = decoder.decode(serialized_example, ['array'])
      self.assertAllEqual(tf_array.eval(), np_array)

  def testDecodeExampleWithInt64Tensor(self):
    np_array = np.random.randint(1, 10, size=(2, 3, 1))

    example = example_pb2.Example(features=feature_pb2.Features(feature={
        'array': self._EncodedInt64Feature(np_array),
    }))

    serialized_example = example.SerializeToString()

    with self.test_session():
      serialized_example = array_ops.reshape(serialized_example, shape=[])
      keys_to_features = {
          'array': parsing_ops.FixedLenFeature(np_array.shape, dtypes.int64)
      }
      items_to_handlers = {'array': tfexample_decoder.Tensor('array'),}
      decoder = tfexample_decoder.TFExampleDecoder(keys_to_features,
                                                   items_to_handlers)
      [tf_array] = decoder.decode(serialized_example, ['array'])
      self.assertAllEqual(tf_array.eval(), np_array)

  def testDecodeExampleWithVarLenTensor(self):
    np_array = np.array([[[1], [2], [3]], [[4], [5], [6]]])

    example = example_pb2.Example(features=feature_pb2.Features(feature={
        'labels': self._EncodedInt64Feature(np_array),
    }))

    serialized_example = example.SerializeToString()

    with self.test_session():
      serialized_example = array_ops.reshape(serialized_example, shape=[])
      keys_to_features = {
          'labels': parsing_ops.VarLenFeature(dtype=dtypes.int64),
      }
      items_to_handlers = {'labels': tfexample_decoder.Tensor('labels'),}
      decoder = tfexample_decoder.TFExampleDecoder(keys_to_features,
                                                   items_to_handlers)
      [tf_labels] = decoder.decode(serialized_example, ['labels'])
      labels = tf_labels.eval()
      self.assertAllEqual(labels, np_array.flatten())

  def testDecodeExampleWithFixLenTensorWithShape(self):
    np_array = np.array([[1, 2, 3], [4, 5, 6]])

    example = example_pb2.Example(features=feature_pb2.Features(feature={
        'labels': self._EncodedInt64Feature(np_array),
    }))

    serialized_example = example.SerializeToString()

    with self.test_session():
      serialized_example = array_ops.reshape(serialized_example, shape=[])
      keys_to_features = {
          'labels':
              parsing_ops.FixedLenFeature(
                  np_array.shape, dtype=dtypes.int64),
      }
      items_to_handlers = {
          'labels': tfexample_decoder.Tensor(
              'labels', shape=np_array.shape),
      }
      decoder = tfexample_decoder.TFExampleDecoder(keys_to_features,
                                                   items_to_handlers)
      [tf_labels] = decoder.decode(serialized_example, ['labels'])
      labels = tf_labels.eval()
      self.assertAllEqual(labels, np_array)

  def testDecodeExampleWithVarLenTensorToDense(self):
    np_array = np.array([[1, 2, 3], [4, 5, 6]])
    example = example_pb2.Example(features=feature_pb2.Features(feature={
        'labels': self._EncodedInt64Feature(np_array),
    }))

    serialized_example = example.SerializeToString()

    with self.test_session():
      serialized_example = array_ops.reshape(serialized_example, shape=[])
      keys_to_features = {
          'labels': parsing_ops.VarLenFeature(dtype=dtypes.int64),
      }
      items_to_handlers = {
          'labels': tfexample_decoder.Tensor(
              'labels', shape=np_array.shape),
      }
      decoder = tfexample_decoder.TFExampleDecoder(keys_to_features,
                                                   items_to_handlers)
      [tf_labels] = decoder.decode(serialized_example, ['labels'])
      labels = tf_labels.eval()
      self.assertAllEqual(labels, np_array)

  def testDecodeExampleShapeKeyTensor(self):
    np_image = np.random.rand(2, 3, 1).astype('f')
    np_labels = np.array([[[1], [2], [3]], [[4], [5], [6]]])

    example = example_pb2.Example(features=feature_pb2.Features(feature={
        'image': self._EncodedFloatFeature(np_image),
        'image/shape': self._EncodedInt64Feature(np.array(np_image.shape)),
        'labels': self._EncodedInt64Feature(np_labels),
        'labels/shape': self._EncodedInt64Feature(np.array(np_labels.shape)),
    }))

    serialized_example = example.SerializeToString()

    with self.test_session():
      serialized_example = array_ops.reshape(serialized_example, shape=[])
      keys_to_features = {
          'image': parsing_ops.VarLenFeature(dtype=dtypes.float32),
          'image/shape': parsing_ops.VarLenFeature(dtype=dtypes.int64),
          'labels': parsing_ops.VarLenFeature(dtype=dtypes.int64),
          'labels/shape': parsing_ops.VarLenFeature(dtype=dtypes.int64),
      }
      items_to_handlers = {
          'image':
              tfexample_decoder.Tensor(
                  'image', shape_keys='image/shape'),
          'labels':
              tfexample_decoder.Tensor(
                  'labels', shape_keys='labels/shape'),
      }
      decoder = tfexample_decoder.TFExampleDecoder(keys_to_features,
                                                   items_to_handlers)
      [tf_image, tf_labels] = decoder.decode(serialized_example,
                                             ['image', 'labels'])
      self.assertAllEqual(tf_image.eval(), np_image)
      self.assertAllEqual(tf_labels.eval(), np_labels)

  def testDecodeExampleMultiShapeKeyTensor(self):
    np_image = np.random.rand(2, 3, 1).astype('f')
    np_labels = np.array([[[1], [2], [3]], [[4], [5], [6]]])
    height, width, depth = np_labels.shape

    example = example_pb2.Example(features=feature_pb2.Features(feature={
        'image': self._EncodedFloatFeature(np_image),
        'image/shape': self._EncodedInt64Feature(np.array(np_image.shape)),
        'labels': self._EncodedInt64Feature(np_labels),
        'labels/height': self._EncodedInt64Feature(np.array([height])),
        'labels/width': self._EncodedInt64Feature(np.array([width])),
        'labels/depth': self._EncodedInt64Feature(np.array([depth])),
    }))

    serialized_example = example.SerializeToString()

    with self.test_session():
      serialized_example = array_ops.reshape(serialized_example, shape=[])
      keys_to_features = {
          'image': parsing_ops.VarLenFeature(dtype=dtypes.float32),
          'image/shape': parsing_ops.VarLenFeature(dtype=dtypes.int64),
          'labels': parsing_ops.VarLenFeature(dtype=dtypes.int64),
          'labels/height': parsing_ops.VarLenFeature(dtype=dtypes.int64),
          'labels/width': parsing_ops.VarLenFeature(dtype=dtypes.int64),
          'labels/depth': parsing_ops.VarLenFeature(dtype=dtypes.int64),
      }
      items_to_handlers = {
          'image':
              tfexample_decoder.Tensor(
                  'image', shape_keys='image/shape'),
          'labels':
              tfexample_decoder.Tensor(
                  'labels',
                  shape_keys=['labels/height', 'labels/width', 'labels/depth']),
      }
      decoder = tfexample_decoder.TFExampleDecoder(keys_to_features,
                                                   items_to_handlers)
      [tf_image, tf_labels] = decoder.decode(serialized_example,
                                             ['image', 'labels'])
      self.assertAllEqual(tf_image.eval(), np_image)
      self.assertAllEqual(tf_labels.eval(), np_labels)

  def testDecodeExampleWithSparseTensor(self):
    np_indices = np.array([[1], [2], [5]])
    np_values = np.array([0.1, 0.2, 0.6]).astype('f')
    example = example_pb2.Example(features=feature_pb2.Features(feature={
        'indices': self._EncodedInt64Feature(np_indices),
        'values': self._EncodedFloatFeature(np_values),
    }))

    serialized_example = example.SerializeToString()

    with self.test_session():
      serialized_example = array_ops.reshape(serialized_example, shape=[])
      keys_to_features = {
          'indices': parsing_ops.VarLenFeature(dtype=dtypes.int64),
          'values': parsing_ops.VarLenFeature(dtype=dtypes.float32),
      }
      items_to_handlers = {'labels': tfexample_decoder.SparseTensor(),}
      decoder = tfexample_decoder.TFExampleDecoder(keys_to_features,
                                                   items_to_handlers)
      [tf_labels] = decoder.decode(serialized_example, ['labels'])
      labels = tf_labels.eval()
      self.assertAllEqual(labels.indices, np_indices)
      self.assertAllEqual(labels.values, np_values)
      self.assertAllEqual(labels.dense_shape, np_values.shape)

  def testDecodeExampleWithSparseTensorWithKeyShape(self):
    np_indices = np.array([[1], [2], [5]])
    np_values = np.array([0.1, 0.2, 0.6]).astype('f')
    np_shape = np.array([6])
    example = example_pb2.Example(features=feature_pb2.Features(feature={
        'indices': self._EncodedInt64Feature(np_indices),
        'values': self._EncodedFloatFeature(np_values),
        'shape': self._EncodedInt64Feature(np_shape),
    }))

    serialized_example = example.SerializeToString()

    with self.test_session():
      serialized_example = array_ops.reshape(serialized_example, shape=[])
      keys_to_features = {
          'indices': parsing_ops.VarLenFeature(dtype=dtypes.int64),
          'values': parsing_ops.VarLenFeature(dtype=dtypes.float32),
          'shape': parsing_ops.VarLenFeature(dtype=dtypes.int64),
      }
      items_to_handlers = {
          'labels': tfexample_decoder.SparseTensor(shape_key='shape'),
      }
      decoder = tfexample_decoder.TFExampleDecoder(keys_to_features,
                                                   items_to_handlers)
      [tf_labels] = decoder.decode(serialized_example, ['labels'])
      labels = tf_labels.eval()
      self.assertAllEqual(labels.indices, np_indices)
      self.assertAllEqual(labels.values, np_values)
      self.assertAllEqual(labels.dense_shape, np_shape)

  def testDecodeExampleWithSparseTensorWithGivenShape(self):
    np_indices = np.array([[1], [2], [5]])
    np_values = np.array([0.1, 0.2, 0.6]).astype('f')
    np_shape = np.array([6])
    example = example_pb2.Example(features=feature_pb2.Features(feature={
        'indices': self._EncodedInt64Feature(np_indices),
        'values': self._EncodedFloatFeature(np_values),
    }))

    serialized_example = example.SerializeToString()

    with self.test_session():
      serialized_example = array_ops.reshape(serialized_example, shape=[])
      keys_to_features = {
          'indices': parsing_ops.VarLenFeature(dtype=dtypes.int64),
          'values': parsing_ops.VarLenFeature(dtype=dtypes.float32),
      }
      items_to_handlers = {
          'labels': tfexample_decoder.SparseTensor(shape=np_shape),
      }
      decoder = tfexample_decoder.TFExampleDecoder(keys_to_features,
                                                   items_to_handlers)
      [tf_labels] = decoder.decode(serialized_example, ['labels'])
      labels = tf_labels.eval()
      self.assertAllEqual(labels.indices, np_indices)
      self.assertAllEqual(labels.values, np_values)
      self.assertAllEqual(labels.dense_shape, np_shape)

  def testDecodeExampleWithSparseTensorToDense(self):
    np_indices = np.array([1, 2, 5])
    np_values = np.array([0.1, 0.2, 0.6]).astype('f')
    np_shape = np.array([6])
    np_dense = np.array([0.0, 0.1, 0.2, 0.0, 0.0, 0.6]).astype('f')
    example = example_pb2.Example(features=feature_pb2.Features(feature={
        'indices': self._EncodedInt64Feature(np_indices),
        'values': self._EncodedFloatFeature(np_values),
    }))

    serialized_example = example.SerializeToString()

    with self.test_session():
      serialized_example = array_ops.reshape(serialized_example, shape=[])
      keys_to_features = {
          'indices': parsing_ops.VarLenFeature(dtype=dtypes.int64),
          'values': parsing_ops.VarLenFeature(dtype=dtypes.float32),
      }
      items_to_handlers = {
          'labels':
              tfexample_decoder.SparseTensor(
                  shape=np_shape, densify=True),
      }
      decoder = tfexample_decoder.TFExampleDecoder(keys_to_features,
                                                   items_to_handlers)
      [tf_labels] = decoder.decode(serialized_example, ['labels'])
      labels = tf_labels.eval()
      self.assertAllClose(labels, np_dense)

  def testDecodeExampleWithTensor(self):
    tensor_shape = (2, 3, 1)
    np_array = np.random.rand(2, 3, 1)

    example = example_pb2.Example(features=feature_pb2.Features(feature={
        'image/depth_map': self._EncodedFloatFeature(np_array),
    }))

    serialized_example = example.SerializeToString()

    with self.test_session():
      serialized_example = array_ops.reshape(serialized_example, shape=[])

      keys_to_features = {
          'image/depth_map':
              parsing_ops.FixedLenFeature(
                  tensor_shape,
                  dtypes.float32,
                  default_value=array_ops.zeros(tensor_shape))
      }

      items_to_handlers = {'depth': tfexample_decoder.Tensor('image/depth_map')}

      decoder = tfexample_decoder.TFExampleDecoder(keys_to_features,
                                                   items_to_handlers)
      [tf_depth] = decoder.decode(serialized_example, ['depth'])
      depth = tf_depth.eval()

    self.assertAllClose(np_array, depth)

  def testDecodeExampleWithItemHandlerCallback(self):
    np.random.seed(0)
    tensor_shape = (2, 3, 1)
    np_array = np.random.rand(2, 3, 1)

    example = example_pb2.Example(features=feature_pb2.Features(feature={
        'image/depth_map': self._EncodedFloatFeature(np_array),
    }))

    serialized_example = example.SerializeToString()

    with self.test_session():
      serialized_example = array_ops.reshape(serialized_example, shape=[])

      keys_to_features = {
          'image/depth_map':
              parsing_ops.FixedLenFeature(
                  tensor_shape,
                  dtypes.float32,
                  default_value=array_ops.zeros(tensor_shape))
      }

      def HandleDepth(keys_to_tensors):
        depth = list(keys_to_tensors.values())[0]
        depth += 1
        return depth

      items_to_handlers = {
          'depth':
              tfexample_decoder.ItemHandlerCallback('image/depth_map',
                                                    HandleDepth)
      }

      decoder = tfexample_decoder.TFExampleDecoder(keys_to_features,
                                                   items_to_handlers)
      [tf_depth] = decoder.decode(serialized_example, ['depth'])
      depth = tf_depth.eval()

    self.assertAllClose(np_array, depth - 1)

  def testDecodeImageWithItemHandlerCallback(self):
    image_shape = (2, 3, 3)
    for image_encoding in ['jpeg', 'png']:
      image, serialized_example = self.GenerateImage(
          image_format=image_encoding, image_shape=image_shape)

      with self.test_session():

        def ConditionalDecoding(keys_to_tensors):
          """See base class."""
          image_buffer = keys_to_tensors['image/encoded']
          image_format = keys_to_tensors['image/format']

          def DecodePng():
            return image_ops.decode_png(image_buffer, 3)

          def DecodeJpg():
            return image_ops.decode_jpeg(image_buffer, 3)

          image = control_flow_ops.case(
              {
                  math_ops.equal(image_format, 'png'): DecodePng,
              },
              default=DecodeJpg,
              exclusive=True)
          image = array_ops.reshape(image, image_shape)
          return image

        keys_to_features = {
            'image/encoded':
                parsing_ops.FixedLenFeature(
                    (), dtypes.string, default_value=''),
            'image/format':
                parsing_ops.FixedLenFeature(
                    (), dtypes.string, default_value='jpeg')
        }

        items_to_handlers = {
            'image':
                tfexample_decoder.ItemHandlerCallback(
                    ['image/encoded', 'image/format'], ConditionalDecoding)
        }

        decoder = tfexample_decoder.TFExampleDecoder(keys_to_features,
                                                     items_to_handlers)
        [tf_image] = decoder.decode(serialized_example, ['image'])
        decoded_image = tf_image.eval()
        if image_encoding == 'jpeg':
          # For jenkins:
          image = image.astype(np.float32)
          decoded_image = decoded_image.astype(np.float32)
          self.assertAllClose(image, decoded_image, rtol=.5, atol=1.001)
        else:
          self.assertAllClose(image, decoded_image, atol=0)

  def testDecodeExampleWithBoundingBoxSparse(self):
    num_bboxes = 10
    np_ymin = np.random.rand(num_bboxes, 1)
    np_xmin = np.random.rand(num_bboxes, 1)
    np_ymax = np.random.rand(num_bboxes, 1)
    np_xmax = np.random.rand(num_bboxes, 1)
    np_bboxes = np.hstack([np_ymin, np_xmin, np_ymax, np_xmax])

    example = example_pb2.Example(features=feature_pb2.Features(feature={
        'image/object/bbox/ymin': self._EncodedFloatFeature(np_ymin),
        'image/object/bbox/xmin': self._EncodedFloatFeature(np_xmin),
        'image/object/bbox/ymax': self._EncodedFloatFeature(np_ymax),
        'image/object/bbox/xmax': self._EncodedFloatFeature(np_xmax),
    }))
    serialized_example = example.SerializeToString()

    with self.test_session():
      serialized_example = array_ops.reshape(serialized_example, shape=[])

      keys_to_features = {
          'image/object/bbox/ymin': parsing_ops.VarLenFeature(dtypes.float32),
          'image/object/bbox/xmin': parsing_ops.VarLenFeature(dtypes.float32),
          'image/object/bbox/ymax': parsing_ops.VarLenFeature(dtypes.float32),
          'image/object/bbox/xmax': parsing_ops.VarLenFeature(dtypes.float32),
      }

      items_to_handlers = {
          'object/bbox':
              tfexample_decoder.BoundingBox(['ymin', 'xmin', 'ymax', 'xmax'],
                                            'image/object/bbox/'),
      }

      decoder = tfexample_decoder.TFExampleDecoder(keys_to_features,
                                                   items_to_handlers)
      [tf_bboxes] = decoder.decode(serialized_example, ['object/bbox'])
      bboxes = tf_bboxes.eval()

    self.assertAllClose(np_bboxes, bboxes)

  def testDecodeExampleWithBoundingBoxDense(self):
    num_bboxes = 10
    np_ymin = np.random.rand(num_bboxes, 1)
    np_xmin = np.random.rand(num_bboxes, 1)
    np_ymax = np.random.rand(num_bboxes, 1)
    np_xmax = np.random.rand(num_bboxes, 1)
    np_bboxes = np.hstack([np_ymin, np_xmin, np_ymax, np_xmax])

    example = example_pb2.Example(features=feature_pb2.Features(feature={
        'image/object/bbox/ymin': self._EncodedFloatFeature(np_ymin),
        'image/object/bbox/xmin': self._EncodedFloatFeature(np_xmin),
        'image/object/bbox/ymax': self._EncodedFloatFeature(np_ymax),
        'image/object/bbox/xmax': self._EncodedFloatFeature(np_xmax),
    }))
    serialized_example = example.SerializeToString()

    with self.test_session():
      serialized_example = array_ops.reshape(serialized_example, shape=[])

      keys_to_features = {
          'image/object/bbox/ymin': parsing_ops.FixedLenSequenceFeature(
              [], dtypes.float32, allow_missing=True),
          'image/object/bbox/xmin': parsing_ops.FixedLenSequenceFeature(
              [], dtypes.float32, allow_missing=True),
          'image/object/bbox/ymax': parsing_ops.FixedLenSequenceFeature(
              [], dtypes.float32, allow_missing=True),
          'image/object/bbox/xmax': parsing_ops.FixedLenSequenceFeature(
              [], dtypes.float32, allow_missing=True),
      }

      items_to_handlers = {
          'object/bbox':
              tfexample_decoder.BoundingBox(['ymin', 'xmin', 'ymax', 'xmax'],
                                            'image/object/bbox/'),
      }

      decoder = tfexample_decoder.TFExampleDecoder(keys_to_features,
                                                   items_to_handlers)
      [tf_bboxes] = decoder.decode(serialized_example, ['object/bbox'])
      bboxes = tf_bboxes.eval()

    self.assertAllClose(np_bboxes, bboxes)

  def testDecodeExampleWithRepeatedImages(self):
    image_shape = (2, 3, 3)
    image_format = 'png'
    image, _ = self.GenerateImage(
        image_format=image_format, image_shape=image_shape)
    tf_encoded = self._Encoder(image, image_format)
    with self.test_session():
      tf_string = tf_encoded.eval()

    example = example_pb2.Example(features=feature_pb2.Features(feature={
        'image/encoded': feature_pb2.Feature(bytes_list=feature_pb2.BytesList(
            value=[tf_string, tf_string])),
        'image/format': self._StringFeature(image_format),
    }))
    serialized_example = example.SerializeToString()

    with self.test_session():
      serialized_example = array_ops.reshape(serialized_example, shape=[])

      decoder = tfexample_decoder.TFExampleDecoder(
          keys_to_features={
              'image/encoded':
                  parsing_ops.FixedLenFeature(
                      (2,), dtypes.string),
              'image/format':
                  parsing_ops.FixedLenFeature(
                      (), dtypes.string, default_value=image_format),
          },
          items_to_handlers={'image': tfexample_decoder.Image(repeated=True)})
      [tf_image] = decoder.decode(serialized_example, ['image'])

      output_image = tf_image.eval()

      self.assertEqual(output_image.shape, (2, 2, 3, 3))
      self.assertAllEqual(np.squeeze(output_image[0, :, :, :]), image)
      self.assertAllEqual(np.squeeze(output_image[1, :, :, :]), image)

  def testDecodeExampleWithLookup(self):

    example = example_pb2.Example(features=feature_pb2.Features(feature={
        'image/object/class/text': self._BytesFeature(
            np.array(['cat', 'dog', 'guinea pig'])),
    }))
    serialized_example = example.SerializeToString()
    # 'dog' -> 0, 'guinea pig' -> 1, 'cat' -> 2
    table = lookup_ops.index_table_from_tensor(
        constant_op.constant(['dog', 'guinea pig', 'cat']))

    with self.test_session() as sess:
      sess.run(lookup_ops.tables_initializer())

      serialized_example = array_ops.reshape(serialized_example, shape=[])

      keys_to_features = {
          'image/object/class/text': parsing_ops.VarLenFeature(dtypes.string),
      }

      items_to_handlers = {
          'labels':
              tfexample_decoder.LookupTensor('image/object/class/text', table),
      }

      decoder = tfexample_decoder.TFExampleDecoder(keys_to_features,
                                                   items_to_handlers)
      obtained_class_ids = decoder.decode(serialized_example)[0].eval()

    self.assertAllClose([2, 0, 1], obtained_class_ids)

  def testDecodeExampleWithBackupHandlerLookup(self):

    example1 = example_pb2.Example(
        features=feature_pb2.Features(
            feature={
                'image/object/class/text':
                    self._BytesFeature(np.array(['cat', 'dog', 'guinea pig'])),
                'image/object/class/label':
                    self._EncodedInt64Feature(np.array([42, 10, 900]))
            }))
    example2 = example_pb2.Example(
        features=feature_pb2.Features(
            feature={
                'image/object/class/text':
                    self._BytesFeature(np.array(['cat', 'dog', 'guinea pig'])),
            }))
    example3 = example_pb2.Example(
        features=feature_pb2.Features(
            feature={
                'image/object/class/label':
                    self._EncodedInt64Feature(np.array([42, 10, 901]))
            }))
    # 'dog' -> 0, 'guinea pig' -> 1, 'cat' -> 2
    table = lookup_ops.index_table_from_tensor(
        constant_op.constant(['dog', 'guinea pig', 'cat']))
    keys_to_features = {
        'image/object/class/text': parsing_ops.VarLenFeature(dtypes.string),
        'image/object/class/label': parsing_ops.VarLenFeature(dtypes.int64),
    }
    backup_handler = tfexample_decoder.BackupHandler(
        handler=tfexample_decoder.Tensor('image/object/class/label'),
        backup=tfexample_decoder.LookupTensor('image/object/class/text', table))
    items_to_handlers = {
        'labels': backup_handler,
    }
    decoder = tfexample_decoder.TFExampleDecoder(keys_to_features,
                                                 items_to_handlers)
    obtained_class_ids_each_example = []
    with self.test_session() as sess:
      sess.run(lookup_ops.tables_initializer())
      for example in [example1, example2, example3]:
        serialized_example = array_ops.reshape(
            example.SerializeToString(), shape=[])
        obtained_class_ids_each_example.append(
            decoder.decode(serialized_example)[0].eval())

    self.assertAllClose([42, 10, 900], obtained_class_ids_each_example[0])
    self.assertAllClose([2, 0, 1], obtained_class_ids_each_example[1])
    self.assertAllClose([42, 10, 901], obtained_class_ids_each_example[2])


if __name__ == '__main__':
  test.main()
