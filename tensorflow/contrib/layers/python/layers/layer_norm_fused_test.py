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
"""Tests for tf.contrib.layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


class LayerNormFusedTest(tf.test.TestCase):

    def testUnknownShape(self):
        with tf.Graph().as_default() as g, self.test_session(g):
            inputs = tf.placeholder(dtype=tf.float32)
            with self.assertRaisesRegexp(ValueError, 'undefined rank'):
                tf.contrib.layers.layer_norm_fused(inputs)

    def testUnknownLastDim(self):
        with tf.Graph().as_default() as g, self.test_session(g):
            inputs = tf.placeholder(dtype=tf.float32)
            inputs.set_shape(tf.TensorShape((5, 3, 3, None)))
            with self.assertRaisesRegexp(ValueError, 'undefined last dimension'):
                tf.contrib.layers.layer_norm_fused(inputs)

    def testCreateOp(self):
        height, width = 3, 3
        with self.test_session():
            images = np.random.uniform(size=(5, height, width, 3))
            output = tf.contrib.layers.layer_norm_fused(images)
            self.assertTrue(output.op.name.startswith(
                'LayerNorm/LayerNormFusedCustom'))
            self.assertListEqual(output.get_shape().as_list(), [
                                 5, height, width, 3])

    def testCreateVariables(self):
        height, width = 3, 3
        with self.test_session():
            images = tf.random_uniform((5, height, width, 3), seed=1)
            tf.contrib.layers.layer_norm_fused(images)
            beta = tf.contrib.framework.get_variables_by_name('beta')[0]
            gamma = tf.contrib.framework.get_variables_by_name('gamma')[0]
            self.assertEqual(beta.op.name, 'LayerNorm/beta')
            self.assertEqual(gamma.op.name, 'LayerNorm/gamma')

    def testReuseVariables(self):
        height, width = 3, 3
        with self.test_session():
            images = tf.random_uniform((5, height, width, 3), seed=1)
            tf.contrib.layers.layer_norm_fused(images, scope='ln')
            tf.contrib.layers.layer_norm_fused(images, scope='ln', reuse=True)
            beta = tf.contrib.framework.get_variables_by_name('beta')
            gamma = tf.contrib.framework.get_variables_by_name('gamma')
            self.assertEqual(len(beta), 1)
            self.assertEqual(len(gamma), 1)

    def testReuseVars(self):
        height, width = 3, 3
        with self.test_session() as sess:
            image_shape = (10, height, width, 3)
            image_values = np.random.rand(*image_shape)
            images = tf.constant(
                image_values, shape=image_shape, dtype=tf.float32)
            output_train = tf.contrib.layers.layer_norm_fused(
                images, scope='LN')
            output_eval = tf.contrib.layers.layer_norm_fused(images,
                                                             scope='LN',
                                                             reuse=True)
            # Initialize all variables
            sess.run(tf.global_variables_initializer())
            # output_train and output_eval should be the same.
            self.assertAllClose(
                sess.run([output_train]), sess.run([output_eval]))

    def doOutputTest(self, input_shape, center=False, scale=False):
        with self.test_session() as sess:
            input_values = np.random.rand(*input_shape)
            inputs = tf.constant(
                input_values, shape=input_shape, dtype=tf.float32)
            output_op = tf.contrib.layers.layer_norm_fused(
                inputs, center=center, scale=scale,
                scope='LN')
            # reshape input to 2D so the built-in layer_norm layer
            # will generate equivalent results
            g_inputs = tf.reshape(inputs, [-1, input_shape[-1]])
            gold_op = tf.contrib.layers.layer_norm(
                g_inputs, center=center, scale=scale,
                scope="gold_LN")
            # Initialize all variables
            sess.run(tf.global_variables_initializer())
            # sess.run(tf.global_variables_initializer())
            outputs = sess.run(output_op)
            golds = sess.run(gold_op)

            tol = 1e-4
            self.assertAllClose(
                outputs.ravel(), golds.ravel(), rtol=tol, atol=tol)

    def testVanillaOutput2DInput(self):
        self.doOutputTest((10, 300))

    def testVanillaOutput4DInput(self):
        self.doOutputTest((100, 10, 10, 4))

    def testScaleOutput2DInput(self):
        self.doOutputTest((10, 300), center=False, scale=True)

    def testScaleOutput4DInput(self):
        self.doOutputTest((100, 10, 10, 4), center=False, scale=True)

    def testCenterOutput2DInput(self):
        self.doOutputTest((10, 300), center=True, scale=False)

    def testCenterOutput4DInput(self):
        self.doOutputTest((100, 10, 10, 4), center=True, scale=False)

    def testFusedOutput2DInput(self):
        self.doOutputTest((10, 300), center=True, scale=True)

    def testFusedOutput4DInput(self):
        self.doOutputTest((100, 10, 10, 4), center=True, scale=True)

    def doGradientTest(self, input_shape, center=False, scale=False):
        with self.test_session() as sess:
            # np.random.seed(5)
            input_values = np.random.rand(*input_shape)
            inputs = tf.constant(
                input_values, shape=input_shape, dtype=tf.float32)
            output_op = tf.contrib.layers.layer_norm_fused(
                inputs, center=center, scale=scale,
                variables_collections=["_var_cust"],
                scope='LN')
            # reshape input to 2D so the built-in layer_norm layer
            # will generate equivalent results
            g_inputs = tf.reshape(inputs, [-1, input_shape[-1]])
            gold_op = tf.contrib.layers.layer_norm(
                g_inputs, center=center, scale=scale,
                variables_collections=["_var_gold"],
                scope="gold_LN")
            cust_vars = tf.get_collection("_var_cust")
            gold_vars = tf.get_collection("_var_gold")
            cust_grads = tf.gradients(tf.nn.relu(
                output_op), [inputs] + cust_vars)
            gold_grads = tf.gradients(
                tf.nn.relu(gold_op), [inputs] + gold_vars)
            # Initialize all variables
            sess.run(tf.global_variables_initializer())
            # sess.run(tf.global_variables_initializer())
            _c_grads = sess.run(cust_grads)
            _g_grads = sess.run(gold_grads)
            for _c, _g in zip(_c_grads, _g_grads):
                tol = 1e-4 * _g.max()
                self.assertAllClose(
                    _c.ravel(), _g.ravel(), rtol=tol, atol=tol)

    def testVanillaGradient2DInput(self):
        self.doGradientTest((10, 300))

    def testVanillaGradient4DInput(self):
        self.doGradientTest((100, 10, 10, 64))

    def testVanillaGradientSmallDepth(self):
        self.doGradientTest((100, 10, 10, 16))

    def testScaleGradient2DInput(self):
        self.doGradientTest((2, 300), center=False, scale=True)

    def testScaleGradient4DInput(self):
        self.doGradientTest((100, 10, 10, 64), center=False, scale=True)

    def testScaleGradientSmallDepth(self):
        self.doGradientTest((100, 10, 10, 16), center=False, scale=True)

    def testCenterGradient2DInput(self):
        self.doGradientTest((10, 300), center=True, scale=False)

    def testCenterGradient4DInput(self):
        self.doGradientTest((100, 10, 10, 64), center=True, scale=False)

    def testCenterGradient4DSmallDepth(self):
        self.doGradientTest((100, 10, 10, 16), center=True, scale=False)

    def testFusedGradient2DInput(self):
        self.doGradientTest((10, 300), center=True, scale=True)

    def testFusedGradient4DInput(self):
        self.doGradientTest((100, 10, 10, 64), center=True, scale=True)

    def testFusedGradientSmallDepth(self):
        self.doGradientTest((100, 10, 10, 16), center=True, scale=True)


if __name__ == '__main__':
    tf.test.main()
