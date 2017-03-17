# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for contrib.seq2seq.python.ops.dynamic_attention_wrapper."""
# pylint: disable=unused-import,g-bad-import-order
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# pylint: enable=unused-import

import functools

import numpy as np

from tensorflow.contrib.rnn import core_rnn_cell
from tensorflow.contrib.seq2seq.python.ops import decoder
from tensorflow.contrib.seq2seq.python.ops import dynamic_attention_wrapper as wrapper
from tensorflow.contrib.seq2seq.python.ops import helper as helper_py
from tensorflow.contrib.seq2seq.python.ops import basic_decoder
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.platform import test
from tensorflow.python.util import nest

# pylint: enable=g-import-not-at-top


class DynamicAttentionWrapperTest(test.TestCase):

  def assertAllClose(self, *args, **kwargs):
    kwargs["atol"] = 1e-4  # For GPU tests
    kwargs["rtol"] = 1e-4  # For GPU tests
    return super(DynamicAttentionWrapperTest, self).assertAllClose(
        *args, **kwargs)

  def _testWithAttention(self,
                         create_attention_mechanism,
                         expected_final_outputs,
                         expected_final_state,
                         attention_mechanism_depth=3):
    encoder_sequence_length = [3, 2, 3, 1, 0]
    decoder_sequence_length = [2, 0, 1, 2, 3]
    batch_size = 5
    encoder_max_time = 8
    decoder_max_time = 4
    input_depth = 7
    encoder_output_depth = 10
    cell_depth = 9
    attention_depth = 6

    decoder_inputs = np.random.randn(batch_size, decoder_max_time,
                                     input_depth).astype(np.float32)
    encoder_outputs = np.random.randn(batch_size, encoder_max_time,
                                      encoder_output_depth).astype(np.float32)

    attention_mechanism = create_attention_mechanism(
        num_units=attention_mechanism_depth,
        memory=encoder_outputs,
        memory_sequence_length=encoder_sequence_length)

    with self.test_session() as sess:
      with vs.variable_scope(
          "root",
          initializer=init_ops.random_normal_initializer(stddev=0.01, seed=3)):
        cell = core_rnn_cell.LSTMCell(cell_depth)
        cell = wrapper.DynamicAttentionWrapper(
            cell, attention_mechanism, attention_size=attention_depth)
        helper = helper_py.TrainingHelper(decoder_inputs,
                                          decoder_sequence_length)
        my_decoder = basic_decoder.BasicDecoder(
            cell=cell,
            helper=helper,
            initial_state=cell.zero_state(
                dtype=dtypes.float32, batch_size=batch_size))

        final_outputs, final_state = decoder.dynamic_decode(my_decoder)

      self.assertTrue(
          isinstance(final_outputs, basic_decoder.BasicDecoderOutput))
      self.assertTrue(
          isinstance(final_state, wrapper.DynamicAttentionWrapperState))
      self.assertTrue(
          isinstance(final_state.cell_state, core_rnn_cell.LSTMStateTuple))

      self.assertEqual((batch_size, None, attention_depth),
                       tuple(final_outputs.rnn_output.get_shape().as_list()))
      self.assertEqual((batch_size, None),
                       tuple(final_outputs.sample_id.get_shape().as_list()))

      self.assertEqual((batch_size, attention_depth),
                       tuple(final_state.attention.get_shape().as_list()))
      self.assertEqual((batch_size, cell_depth),
                       tuple(final_state.cell_state.c.get_shape().as_list()))
      self.assertEqual((batch_size, cell_depth),
                       tuple(final_state.cell_state.h.get_shape().as_list()))

      sess.run(variables.global_variables_initializer())
      sess_results = sess.run({
          "final_outputs": final_outputs,
          "final_state": final_state
      })

      nest.map_structure(self.assertAllClose, expected_final_outputs,
                         sess_results["final_outputs"])
      nest.map_structure(self.assertAllClose, expected_final_state,
                         sess_results["final_state"])

  def testBahndahauNotNormalized(self):
    create_attention_mechanism = wrapper.BahdanauAttention

    array = np.array
    float32 = np.float32
    int32 = np.int32

    expected_final_outputs = basic_decoder.BasicDecoderOutput(
        rnn_output=array(
            [[[
                1.25166783e-02, -6.88887993e-03, 3.17239435e-03,
                -1.98234897e-03, 4.77387803e-03, -1.38330357e-02
            ], [
                1.28883058e-02, -6.76271692e-03, 3.13419267e-03,
                -2.02183682e-03, 5.62057737e-03, -1.35373026e-02
            ], [
                1.24917831e-02, -6.71574520e-03, 3.42238229e-03,
                -1.79501204e-03, 5.33161033e-03, -1.36620644e-02
            ]], [[
                1.55150667e-02, -1.07274549e-02, 4.44198400e-03,
                -9.73310322e-04, 1.27242506e-02, -1.21861566e-02
            ], [
                1.57585666e-02, -1.07965544e-02, 4.61554807e-03,
                -1.01510016e-03, 1.22341057e-02, -1.27029382e-02
            ], [
                1.58304181e-02, -1.09712025e-02, 4.67861444e-03,
                -1.03920139e-03, 1.23004699e-02, -1.25949886e-02
            ]], [[
                9.26700700e-03, -9.75431874e-03, -9.95740294e-04,
                -1.27463136e-06, 3.81659716e-03, -1.64887272e-02
            ], [
                9.25191958e-03, -9.80092678e-03, -8.48566880e-04,
                5.02091134e-05, 3.46567202e-03, -1.67435352e-02
            ], [
                9.48173273e-03, -9.52653307e-03, -8.79382715e-04,
                -3.07094306e-05, 4.05955408e-03, -1.67226996e-02
            ]], [[
                1.21462569e-02, -1.27578378e-02, 1.54045003e-04, 2.70257704e-03,
                7.79421115e-03, -8.14041123e-04
            ], [
                1.18412934e-02, -1.33513296e-02, 3.54760559e-05, 2.67801876e-03,
                6.99122995e-03, -9.46014654e-04
            ], [
                1.16087487e-02, -1.31632648e-02, -2.98853614e-04,
                2.49515846e-03, 6.92677684e-03, -6.92734495e-04
            ]], [[
                1.02377674e-02, -8.72955937e-03, 1.22555892e-03, 2.03830865e-03,
                8.93574394e-03, -7.28237582e-03
            ], [
                1.05115287e-02, -8.92531779e-03, 1.14568521e-03, 1.91635895e-03,
                8.94328393e-03, -7.39541650e-03
            ], [
                1.07398070e-02, -8.56867433e-03, 1.52354129e-03, 2.06834078e-03,
                9.36511997e-03, -7.64556089e-03
            ]]],
            dtype=float32),
        sample_id=array(
            [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
            dtype=int32))

    expected_final_state = wrapper.DynamicAttentionWrapperState(
        cell_state=core_rnn_cell.LSTMStateTuple(
            c=array(
                [[
                    -0.0220502, -0.008058, -0.00160266, 0.01609341, -0.01380513,
                    -0.00749483, -0.00816989, -0.01210028, 0.01795324
                ], [
                    0.01727026, -0.0142065, -0.00399991, 0.03195379,
                    -0.03547479, -0.02138772, -0.00610318, -0.00191625,
                    -0.01937846
                ], [
                    -0.0116077, 0.00876439, -0.01641787, -0.01400803,
                    0.01347527, -0.01036386, 0.00627491, -0.0096361, -0.00650565
                ], [
                    -0.04763387, -0.01192631, -0.00019412, 0.04103886,
                    -0.00137999, 0.02126684, -0.02793711, -0.05467696,
                    -0.02912051
                ], [
                    0.02241185, -0.00141741, 0.01911988, 0.00547728,
                    -0.01280068, -0.00307024, -0.00494239, 0.02169247,
                    0.01631995
                ]],
                dtype=float32),
            h=array(
                [[
                    -1.10613741e-02, -3.98175791e-03, -8.15514475e-04,
                    7.90482666e-03, -7.02390168e-03, -3.76394135e-03,
                    -4.16183751e-03, -6.17114361e-03, 8.95532221e-03
                ], [
                    8.60657450e-03, -7.17655150e-03, -1.94156705e-03,
                    1.62583217e-02, -1.76821016e-02, -1.06200138e-02,
                    -3.01904045e-03, -9.57608980e-04, -9.95732192e-03
                ], [
                    -5.78935863e-03, 4.49362956e-03, -8.13615043e-03,
                    -6.95384294e-03, 6.75151078e-03, -5.07845683e-03,
                    3.11869266e-03, -4.72904649e-03, -3.20469099e-03
                ], [
                    -2.38025561e-02, -5.89242764e-03, -9.76260417e-05,
                    2.01697368e-02, -6.82076614e-04, 1.07111251e-02,
                    -1.42077375e-02, -2.70790439e-02, -1.44685479e-02
                ], [
                    1.11825848e-02, -6.99267141e-04, 9.82748345e-03,
                    2.74566701e-03, -6.56377291e-03, -1.53681310e-03,
                    -2.48806458e-03, 1.10462429e-02, 7.97568541e-03
                ]],
                dtype=float32)),
        attention=array(
            [[
                1.24917831e-02, -6.71574520e-03, 3.42238229e-03,
                -1.79501204e-03, 5.33161033e-03, -1.36620644e-02
            ], [
                1.58304181e-02, -1.09712025e-02, 4.67861444e-03,
                -1.03920139e-03, 1.23004699e-02, -1.25949886e-02
            ], [
                9.48173273e-03, -9.52653307e-03, -8.79382715e-04,
                -3.07094306e-05, 4.05955408e-03, -1.67226996e-02
            ], [
                1.16087487e-02, -1.31632648e-02, -2.98853614e-04,
                2.49515846e-03, 6.92677684e-03, -6.92734495e-04
            ], [
                1.07398070e-02, -8.56867433e-03, 1.52354129e-03, 2.06834078e-03,
                9.36511997e-03, -7.64556089e-03
            ]],
            dtype=float32))
    self._testWithAttention(create_attention_mechanism, expected_final_outputs,
                            expected_final_state)

  def testBahndahauNormalized(self):
    create_attention_mechanism = functools.partial(
        wrapper.BahdanauAttention, normalize=True, attention_r_initializer=2.0)

    array = np.array
    float32 = np.float32
    int32 = np.int32

    expected_final_output = basic_decoder.BasicDecoderOutput(
        rnn_output=array(
            [[[
                1.72670335e-02, -5.83671592e-03, 6.38638902e-03,
                -8.11776379e-04, 1.12681929e-03, -1.24236047e-02
            ], [
                1.75918192e-02, -5.73426578e-03, 6.29768707e-03,
                -8.63141613e-04, 2.03352375e-03, -1.21420780e-02
            ], [
                1.72424167e-02, -5.66471322e-03, 6.63427915e-03,
                -6.23903936e-04, 1.68706616e-03, -1.22524602e-02
            ]], [[
                1.79958157e-02, -9.80986748e-03, 4.73218597e-03,
                -3.89962713e-03, 1.41502675e-02, -1.48344040e-02
            ], [
                1.82184577e-02, -9.88379307e-03, 4.90130857e-03,
                -3.91892251e-03, 1.36479288e-02, -1.53291579e-02
            ], [
                1.83001235e-02, -1.00617753e-02, 4.97077405e-03,
                -3.94908339e-03, 1.37211196e-02, -1.52311027e-02
            ]], [[
                7.93476030e-03, -8.46967567e-03, -7.16930721e-04,
                4.37953044e-04, 1.04503892e-03, -1.82424393e-02
            ], [
                7.90629163e-03, -8.48819874e-03, -5.57833235e-04,
                5.02390554e-04, 6.79406337e-04, -1.84837580e-02
            ], [
                8.14734399e-03, -8.23053624e-03, -5.92814526e-04,
                4.16347990e-04, 1.29250437e-03, -1.84548404e-02
            ]], [[
                1.21026095e-02, -1.26739489e-02, 1.78718648e-04, 2.68748170e-03,
                7.80996867e-03, -9.69076063e-04
            ], [
                1.17978491e-02, -1.32678337e-02, 6.00410858e-05, 2.66301399e-03,
                7.00691342e-03, -1.10030361e-03
            ], [
                1.15651665e-02, -1.30795036e-02, -2.74205930e-04,
                2.48012133e-03, 6.94250735e-03, -8.47495161e-04
            ]], [[
                1.02377674e-02, -8.72955937e-03, 1.22555892e-03, 2.03830865e-03,
                8.93574394e-03, -7.28237582e-03
            ], [
                1.05115287e-02, -8.92531779e-03, 1.14568521e-03, 1.91635895e-03,
                8.94328393e-03, -7.39541650e-03
            ], [
                1.07398070e-02, -8.56867433e-03, 1.52354129e-03, 2.06834078e-03,
                9.36511997e-03, -7.64556089e-03
            ]]],
            dtype=float32),
        sample_id=array(
            [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
            dtype=int32))

    expected_final_state = wrapper.DynamicAttentionWrapperState(
        cell_state=core_rnn_cell.LSTMStateTuple(
            c=array(
                [[
                    -0.02209264, -0.00794879, -0.00157153, 0.01614309,
                    -0.01383773, -0.00750943, -0.00824213, -0.01210296,
                    0.01794949
                ], [
                    0.01726926, -0.01418139, -0.0040099, 0.0319339, -0.03545783,
                    -0.02142831, -0.00609501, -0.00195033, -0.01938949
                ], [
                    -0.01159083, 0.0087524, -0.01639001, -0.01400012,
                    0.01342422, -0.01041037, 0.00620991, -0.00960796,
                    -0.00650131
                ], [
                    -0.04763237, -0.01192762, -0.00019377, 0.04103839,
                    -0.00138058, 0.02126443, -0.02793917, -0.05467755,
                    -0.02912025
                ], [
                    0.02241185, -0.00141741, 0.01911988, 0.00547728,
                    -0.01280068, -0.00307024, -0.00494239, 0.02169247,
                    0.01631995
                ]],
                dtype=float32),
            h=array(
                [[
                    -1.10821165e-02, -3.92766716e-03, -7.99638336e-04,
                    7.92923011e-03, -7.04019284e-03, -3.77124036e-03,
                    -4.19876305e-03, -6.17261464e-03, 8.95325281e-03
                ], [
                    8.60597286e-03, -7.16368994e-03, -1.94644753e-03,
                    1.62479617e-02, -1.76739115e-02, -1.06403306e-02,
                    -3.01484042e-03, -9.74688213e-04, -9.96260438e-03
                ], [
                    -5.78098884e-03, 4.48751403e-03, -8.12216662e-03,
                    -6.94991415e-03, 6.72604749e-03, -5.10144979e-03,
                    3.08637507e-03, -4.71517537e-03, -3.20256175e-03
                ], [
                    -2.38018110e-02, -5.89307398e-03, -9.74484938e-05,
                    2.01694984e-02, -6.82370039e-04, 1.07099237e-02,
                    -1.42087601e-02, -2.70793457e-02, -1.44684138e-02
                ], [
                    1.11825848e-02, -6.99267141e-04, 9.82748345e-03,
                    2.74566701e-03, -6.56377291e-03, -1.53681310e-03,
                    -2.48806458e-03, 1.10462429e-02, 7.97568541e-03
                ]],
                dtype=float32)),
        attention=array(
            [[
                0.01724242, -0.00566471, 0.00663428, -0.0006239, 0.00168707,
                -0.01225246
            ], [
                0.01830012, -0.01006178, 0.00497077, -0.00394908, 0.01372112,
                -0.0152311
            ], [
                0.00814734, -0.00823054, -0.00059281, 0.00041635, 0.0012925,
                -0.01845484
            ], [
                0.01156517, -0.0130795, -0.00027421, 0.00248012, 0.00694251,
                -0.0008475
            ], [
                0.01073981, -0.00856867, 0.00152354, 0.00206834, 0.00936512,
                -0.00764556
            ]],
            dtype=float32))

    self._testWithAttention(create_attention_mechanism, expected_final_output,
                            expected_final_state)

  def testLuongNotNormalized(self):
    create_attention_mechanism = wrapper.LuongAttention

    array = np.array
    float32 = np.float32
    int32 = np.int32

    expected_final_output = basic_decoder.BasicDecoderOutput(
        rnn_output=array(
            [[[
                1.23641128e-02, -6.82715839e-03, 3.24165262e-03,
                -1.90772023e-03, 4.69654519e-03, -1.37025211e-02
            ], [
                1.29463980e-02, -6.79699238e-03, 3.10124992e-03,
                -2.02869414e-03, 5.66399656e-03, -1.35517996e-02
            ], [
                1.22659411e-02, -6.81970268e-03, 3.15135531e-03,
                -1.96937821e-03, 5.62768336e-03, -1.39173865e-02
            ]], [[
                1.53944232e-02, -1.07725551e-02, 4.42822604e-03,
                -8.30623554e-04, 1.26549732e-02, -1.20573286e-02
            ], [
                1.57453734e-02, -1.08157266e-02, 4.62466478e-03,
                -9.88351414e-04, 1.22286947e-02, -1.26876952e-02
            ], [
                1.57857724e-02, -1.09536834e-02, 4.64798324e-03,
                -1.01319887e-03, 1.22695938e-02, -1.25500849e-02
            ]], [[
                9.23123397e-03, -9.42669343e-03, -9.09919385e-04,
                6.09827694e-05, 3.90436035e-03, -1.63374804e-02
            ], [
                9.22935922e-03, -9.57853813e-03, -7.92966573e-04,
                8.89014918e-05, 3.52671882e-03, -1.66499857e-02
            ], [
                9.49526206e-03, -9.39475093e-03, -8.49372707e-04,
                -1.72815053e-05, 4.16132808e-03, -1.66336838e-02
            ]], [[
                1.21248290e-02, -1.27166547e-02, 1.66158192e-04, 2.69516627e-03,
                7.80194718e-03, -8.90152063e-04
            ], [
                1.17861275e-02, -1.32453050e-02, 6.66640699e-05, 2.65894993e-03,
                7.01114535e-03, -1.14195189e-03
            ], [
                1.15833860e-02, -1.31145213e-02, -2.84505659e-04,
                2.48642010e-03, 6.93593081e-03, -7.82784075e-04
            ]], [[
                1.02377674e-02, -8.72955937e-03, 1.22555892e-03, 2.03830865e-03,
                8.93574394e-03, -7.28237582e-03
            ], [
                1.05115287e-02, -8.92531779e-03, 1.14568521e-03, 1.91635895e-03,
                8.94328393e-03, -7.39541650e-03
            ], [
                1.07398070e-02, -8.56867433e-03, 1.52354129e-03, 2.06834078e-03,
                9.36511997e-03, -7.64556089e-03
            ]]],
            dtype=float32),
        sample_id=array(
            [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
            dtype=int32))
    expected_final_state = wrapper.DynamicAttentionWrapperState(
        cell_state=core_rnn_cell.LSTMStateTuple(
            c=array(
                [[
                    -0.02204997, -0.00805805, -0.00160245, 0.01609369,
                    -0.01380494, -0.00749439, -0.00817, -0.01209992, 0.01795316
                ], [
                    0.01727016, -0.01420713, -0.00399972, 0.03195436,
                    -0.03547532, -0.02138666, -0.00610335, -0.00191557,
                    -0.01937821
                ], [
                    -0.01160429, 0.00876595, -0.01641685, -0.01400784,
                    0.01348004, -0.01036458, 0.00627241, -0.00963544,
                    -0.00650568
                ], [
                    -0.04763246, -0.01192755, -0.00019379, 0.04103841,
                    -0.00138055, 0.02126456, -0.02793905, -0.0546775,
                    -0.02912027
                ], [
                    0.02241185, -0.00141741, 0.01911988, 0.00547728,
                    -0.01280068, -0.00307024, -0.00494239, 0.02169247,
                    0.01631995
                ]],
                dtype=float32),
            h=array(
                [[
                    -1.10612623e-02, -3.98178305e-03, -8.15406092e-04,
                    7.90496264e-03, -7.02379830e-03, -3.76371504e-03,
                    -4.16189339e-03, -6.17096573e-03, 8.95528216e-03
                ], [
                    8.60652886e-03, -7.17687514e-03, -1.94147555e-03,
                    1.62586085e-02, -1.76823605e-02, -1.06194830e-02,
                    -3.01912241e-03, -9.57269047e-04, -9.95719433e-03
                ], [
                    -5.78764686e-03, 4.49441886e-03, -8.13564472e-03,
                    -6.95375400e-03, 6.75391173e-03, -5.07880514e-03,
                    3.11744539e-03, -4.72871540e-03, -3.20470310e-03
                ], [
                    -2.38018595e-02, -5.89303859e-03, -9.74571449e-05,
                    2.01695058e-02, -6.82353624e-04, 1.07099945e-02,
                    -1.42086931e-02, -2.70793252e-02, -1.44684194e-02
                ], [
                    1.11825848e-02, -6.99267141e-04, 9.82748345e-03,
                    2.74566701e-03, -6.56377291e-03, -1.53681310e-03,
                    -2.48806458e-03, 1.10462429e-02, 7.97568541e-03
                ]],
                dtype=float32)),
        attention=array(
            [[
                1.22659411e-02, -6.81970268e-03, 3.15135531e-03,
                -1.96937821e-03, 5.62768336e-03, -1.39173865e-02
            ], [
                1.57857724e-02, -1.09536834e-02, 4.64798324e-03,
                -1.01319887e-03, 1.22695938e-02, -1.25500849e-02
            ], [
                9.49526206e-03, -9.39475093e-03, -8.49372707e-04,
                -1.72815053e-05, 4.16132808e-03, -1.66336838e-02
            ], [
                1.15833860e-02, -1.31145213e-02, -2.84505659e-04,
                2.48642010e-03, 6.93593081e-03, -7.82784075e-04
            ], [
                1.07398070e-02, -8.56867433e-03, 1.52354129e-03, 2.06834078e-03,
                9.36511997e-03, -7.64556089e-03
            ]],
            dtype=float32))

    self._testWithAttention(
        create_attention_mechanism,
        expected_final_output,
        expected_final_state,
        attention_mechanism_depth=9)

  def testLuongNormalized(self):
    create_attention_mechanism = functools.partial(
        wrapper.LuongAttention, normalize=True, attention_r_initializer=2.0)

    array = np.array
    float32 = np.float32
    int32 = np.int32

    expected_final_output = basic_decoder.BasicDecoderOutput(
        rnn_output=array(
            [[[
                1.23956744e-02, -6.88115368e-03, 3.15234554e-03,
                -1.97300944e-03, 4.79680905e-03, -1.38076628e-02
            ], [
                1.28376717e-02, -6.78718928e-03, 3.07988771e-03,
                -2.03956687e-03, 5.68403490e-03, -1.35601182e-02
            ], [
                1.23463338e-02, -6.76322030e-03, 3.28891934e-03,
                -1.86874042e-03, 5.47897862e-03, -1.37654068e-02
            ]], [[
                1.54412268e-02, -1.07613346e-02, 4.43824846e-03,
                -8.81063985e-04, 1.26828086e-02, -1.21067995e-02
            ], [
                1.57206059e-02, -1.08218864e-02, 4.61952807e-03,
                -9.61483689e-04, 1.22140013e-02, -1.26614980e-02
            ], [
                1.57821011e-02, -1.09842420e-02, 4.66934917e-03,
                -9.85997496e-04, 1.22719472e-02, -1.25438003e-02
            ]], [[
                9.27361846e-03, -9.66077764e-03, -9.69522633e-04,
                1.48308463e-05, 3.88664147e-03, -1.64083000e-02
            ], [
                9.26287938e-03, -9.74234194e-03, -8.32488062e-04,
                5.83778601e-05, 3.52663640e-03, -1.66827720e-02
            ], [
                9.50474478e-03, -9.49789397e-03, -8.71829456e-04,
                -3.09986062e-05, 4.13423358e-03, -1.66635048e-02
            ]], [[
                1.21398102e-02, -1.27454493e-02, 1.57688977e-04, 2.70034792e-03,
                7.79653806e-03, -8.36936757e-04
            ], [
                1.18234595e-02, -1.33170560e-02, 4.55579720e-05, 2.67185434e-03,
                6.99766818e-03, -1.00935437e-03
            ], [
                1.16009805e-02, -1.31483339e-02, -2.94458936e-04,
                2.49248254e-03, 6.92958105e-03, -7.20315147e-04
            ]], [[
                1.02377674e-02, -8.72955937e-03, 1.22555892e-03, 2.03830865e-03,
                8.93574394e-03, -7.28237582e-03
            ], [
                1.05115287e-02, -8.92531779e-03, 1.14568521e-03, 1.91635895e-03,
                8.94328393e-03, -7.39541650e-03
            ], [
                1.07398070e-02, -8.56867433e-03, 1.52354129e-03, 2.06834078e-03,
                9.36511997e-03, -7.64556089e-03
            ]]],
            dtype=float32),
        sample_id=array(
            [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
            dtype=int32))
    expected_final_state = wrapper.DynamicAttentionWrapperState(
        cell_state=core_rnn_cell.LSTMStateTuple(
            c=array(
                [[
                    -0.02204949, -0.00805957, -0.001603, 0.01609283,
                    -0.01380462, -0.0074945, -0.00816895, -0.01210009,
                    0.01795324
                ], [
                    0.01727016, -0.01420708, -0.00399973, 0.03195432,
                    -0.03547529, -0.02138673, -0.00610332, -0.00191565,
                    -0.01937822
                ], [
                    -0.01160676, 0.00876512, -0.01641791, -0.01400807,
                    0.01347767, -0.01036341, 0.00627499, -0.00963627,
                    -0.00650573
                ], [
                    -0.04763342, -0.01192671, -0.00019402, 0.04103871,
                    -0.00138017, 0.02126611, -0.02793773, -0.05467714,
                    -0.02912043
                ], [
                    0.02241185, -0.00141741, 0.01911988, 0.00547728,
                    -0.01280068, -0.00307024, -0.00494239, 0.02169247,
                    0.01631995
                ]],
                dtype=float32),
            h=array(
                [[
                    -1.10610286e-02, -3.98253463e-03, -8.15684092e-04,
                    7.90454168e-03, -7.02364743e-03, -3.76377185e-03,
                    -4.16135695e-03, -6.17104582e-03, 8.95532966e-03
                ], [
                    8.60653073e-03, -7.17685232e-03, -1.94147974e-03,
                    1.62585936e-02, -1.76823437e-02, -1.06195193e-02,
                    -3.01911240e-03, -9.57308919e-04, -9.95720550e-03
                ], [
                    -5.78888878e-03, 4.49400023e-03, -8.13617278e-03,
                    -6.95386063e-03, 6.75271638e-03, -5.07823005e-03,
                    3.11873178e-03, -4.72912844e-03, -3.20472987e-03
                ], [
                    -2.38023344e-02, -5.89262368e-03, -9.75721487e-05,
                    2.01696623e-02, -6.82163402e-04, 1.07107637e-02,
                    -1.42080421e-02, -2.70791352e-02, -1.44685050e-02
                ], [
                    1.11825848e-02, -6.99267141e-04, 9.82748345e-03,
                    2.74566701e-03, -6.56377291e-03, -1.53681310e-03,
                    -2.48806458e-03, 1.10462429e-02, 7.97568541e-03
                ]],
                dtype=float32)),
        attention=array(
            [[
                1.23463338e-02, -6.76322030e-03, 3.28891934e-03,
                -1.86874042e-03, 5.47897862e-03, -1.37654068e-02
            ], [
                1.57821011e-02, -1.09842420e-02, 4.66934917e-03,
                -9.85997496e-04, 1.22719472e-02, -1.25438003e-02
            ], [
                9.50474478e-03, -9.49789397e-03, -8.71829456e-04,
                -3.09986062e-05, 4.13423358e-03, -1.66635048e-02
            ], [
                1.16009805e-02, -1.31483339e-02, -2.94458936e-04,
                2.49248254e-03, 6.92958105e-03, -7.20315147e-04
            ], [
                1.07398070e-02, -8.56867433e-03, 1.52354129e-03, 2.06834078e-03,
                9.36511997e-03, -7.64556089e-03
            ]],
            dtype=float32))
    self._testWithAttention(
        create_attention_mechanism,
        expected_final_output,
        expected_final_state,
        attention_mechanism_depth=9)


if __name__ == "__main__":
  test.main()
