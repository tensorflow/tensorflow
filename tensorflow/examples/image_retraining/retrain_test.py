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
# pylint: disable=g-bad-import-order,unused-import
"""Tests the graph freezing tool."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.examples.image_retraining import retrain
from tensorflow.python.framework import test_util


class ImageRetrainingTest(test_util.TensorFlowTestCase):

  def dummyImageLists(self):
    return {'label_one': {'dir': 'somedir', 'training': ['image_one.jpg',
                                                         'image_two.jpg'],
                          'testing': ['image_three.jpg', 'image_four.jpg'],
                          'validation': ['image_five.jpg', 'image_six.jpg']},
            'label_two': {'dir': 'otherdir', 'training': ['image_one.jpg',
                                                          'image_two.jpg'],
                          'testing': ['image_three.jpg', 'image_four.jpg'],
                          'validation': ['image_five.jpg', 'image_six.jpg']}}

  def testGetImagePath(self):
    image_lists = self.dummyImageLists()
    self.assertEqual('image_dir/somedir/image_one.jpg', retrain.get_image_path(
        image_lists, 'label_one', 0, 'image_dir', 'training'))
    self.assertEqual('image_dir/otherdir/image_four.jpg',
                     retrain.get_image_path(image_lists, 'label_two', 1,
                                            'image_dir', 'testing'))

  def testGetBottleneckPath(self):
    image_lists = self.dummyImageLists()
    self.assertEqual('bottleneck_dir/somedir/image_five.jpg.txt',
                     retrain.get_bottleneck_path(
                         image_lists, 'label_one', 0, 'bottleneck_dir',
                         'validation'))

  def testShouldDistortImage(self):
    self.assertEqual(False, retrain.should_distort_images(False, 0, 0, 0))
    self.assertEqual(True, retrain.should_distort_images(True, 0, 0, 0))
    self.assertEqual(True, retrain.should_distort_images(False, 10, 0, 0))
    self.assertEqual(True, retrain.should_distort_images(False, 0, 1, 0))
    self.assertEqual(True, retrain.should_distort_images(False, 0, 0, 50))

  def testAddInputDistortions(self):
    with tf.Graph().as_default():
      with tf.Session() as sess:
        retrain.add_input_distortions(True, 10, 10, 10)
        self.assertIsNotNone(sess.graph.get_tensor_by_name('DistortJPGInput:0'))
        self.assertIsNotNone(sess.graph.get_tensor_by_name('DistortResult:0'))

  def testAddFinalTrainingOps(self):
    with tf.Graph().as_default():
      with tf.Session() as sess:
        bottleneck = tf.placeholder(
            tf.float32, [1, retrain.BOTTLENECK_TENSOR_SIZE],
            name=retrain.BOTTLENECK_TENSOR_NAME.split(':')[0])
        retrain.add_final_training_ops(5, 'final', bottleneck)
        self.assertIsNotNone(sess.graph.get_tensor_by_name('final:0'))

  def testAddEvaluationStep(self):
    with tf.Graph().as_default():
      final = tf.placeholder(tf.float32, [1], name='final')
      gt = tf.placeholder(tf.float32, [1], name='gt')
      self.assertIsNotNone(retrain.add_evaluation_step(final, gt))

if __name__ == '__main__':
  tf.test.main()
