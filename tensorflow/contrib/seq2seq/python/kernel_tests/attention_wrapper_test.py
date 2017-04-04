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
"""Tests for contrib.seq2seq.python.ops.attention_wrapper."""
# pylint: disable=unused-import,g-bad-import-order
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# pylint: enable=unused-import

import sys
import functools

import numpy as np

from tensorflow.contrib.rnn import core_rnn_cell
from tensorflow.contrib.seq2seq.python.ops import decoder
from tensorflow.contrib.seq2seq.python.ops import attention_wrapper as wrapper
from tensorflow.contrib.seq2seq.python.ops import helper as helper_py
from tensorflow.contrib.seq2seq.python.ops import basic_decoder
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.platform import test
from tensorflow.python.util import nest

# pylint: enable=g-import-not-at-top


# for testing
AttentionWrapperState = wrapper.AttentionWrapperState  # pylint: disable=invalid-name
LSTMStateTuple = core_rnn_cell.LSTMStateTuple  # pylint: disable=invalid-name
BasicDecoderOutput = basic_decoder.BasicDecoderOutput  # pylint: disable=invalid-name
float32 = np.float32
int32 = np.int32
array = np.array


class AttentionWrapperTest(test.TestCase):

  def assertAllClose(self, *args, **kwargs):
    kwargs["atol"] = 1e-4  # For GPU tests
    kwargs["rtol"] = 1e-4  # For GPU tests
    return super(AttentionWrapperTest, self).assertAllClose(
        *args, **kwargs)

  def testAttentionWrapperState(self):
    num_fields = len(wrapper.AttentionWrapperState._fields)  # pylint: disable=protected-access
    state = wrapper.AttentionWrapperState(*([None] * num_fields))
    new_state = state.clone(time=1)
    self.assertEqual(state.time, None)
    self.assertEqual(new_state.time, 1)

  def _testWithAttention(self,
                         create_attention_mechanism,
                         expected_final_output,
                         expected_final_state,
                         attention_mechanism_depth=3,
                         attention_history=False,
                         name=""):
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
        cell = wrapper.AttentionWrapper(
            cell, attention_mechanism, attention_size=attention_depth,
            attention_history=attention_history)
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
          isinstance(final_state, wrapper.AttentionWrapperState))
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

      if attention_history:
        state_attention_history = final_state.attention_history.stack()
        # Remove the history from final_state for purposes of the
        # remainder of the tests.
        final_state = final_state._replace(attention_history=())  # pylint: disable=protected-access
        self.assertEqual((None, batch_size, attention_depth),
                         tuple(state_attention_history.get_shape().as_list()))
      else:
        state_attention_history = ()

      sess.run(variables.global_variables_initializer())
      sess_results = sess.run({
          "final_outputs": final_outputs,
          "final_state": final_state,
          "state_attention_history": state_attention_history,
      })

      print("Copy/paste (%s)\nexpected_final_output = " % name,
            sess_results["final_outputs"])
      sys.stdout.flush()
      print("Copy/paste (%s)\nexpected_final_state = " % name,
            sess_results["final_state"])
      sys.stdout.flush()
      nest.map_structure(self.assertAllClose, expected_final_output,
                         sess_results["final_outputs"])
      nest.map_structure(self.assertAllClose, expected_final_state,
                         sess_results["final_state"])
      if attention_history:  # by default, the wrapper emits attention as output
        self.assertAllClose(
            # outputs are batch major but the stacked TensorArray is time major
            sess_results["state_attention_history"],
            np.transpose(sess_results["final_outputs"].rnn_output,
                         (1, 0, 2)))

  def testBahdanauNotNormalized(self):
    create_attention_mechanism = wrapper.BahdanauAttention

    expected_final_output = BasicDecoderOutput(
        rnn_output=array(
            [[[
                1.89980457e-03, 1.89681584e-03, 2.05339328e-03, -3.83376027e-03,
                -4.31808922e-03, -6.45466987e-03
            ], [
                2.27232254e-03, 2.02509761e-03, 2.01666891e-03, -3.87230632e-03,
                -3.47119337e-03, -6.15991233e-03
            ], [
                1.87640532e-03, 2.07374478e-03, 2.30582547e-03, -3.64564802e-03,
                -3.75995948e-03, -6.28685066e-03
            ]], [[
                4.89835022e-03, -1.94158917e-03, 3.32316267e-03,
                -2.82446202e-03, 3.63192149e-03, -4.80734091e-03
            ], [
                5.14256489e-03, -2.00877781e-03, 3.49807227e-03,
                -2.86567654e-03, 3.14202951e-03, -5.32575324e-03
            ], [
                5.21511910e-03, -2.18198029e-03, 3.56219849e-03,
                -2.88951304e-03, 3.20866983e-03, -5.21918852e-03
            ]], [[
                -1.34951377e-03, -9.68646549e-04, -2.11444520e-03,
                -1.85243192e-03, -5.27541339e-03, -9.10969637e-03
            ], [
                -1.36390887e-03, -1.01293903e-03, -1.96592091e-03,
                -1.80044665e-03, -5.62618347e-03, -9.36636236e-03
            ], [
                -1.13357347e-03, -7.37126335e-04, -1.99582824e-03,
                -1.88097963e-03, -5.03196474e-03, -9.34652984e-03
            ]], [[
                1.52963377e-03, -3.97205260e-03, -9.64675564e-04,
                8.51404853e-04, -1.29804458e-03, 6.56467676e-03
            ], [
                1.22557906e-03, -4.56343032e-03, -1.08188344e-03,
                8.27252632e-04, -2.10058759e-03, 6.43082103e-03
            ], [
                9.93478228e-04, -4.37378604e-03, -1.41531695e-03,
                6.44775166e-04, -2.16480484e-03, 6.68286439e-03
            ]], [[
                -3.78854020e-04, 5.62231544e-05, 1.06837302e-04, 1.87137164e-04,
                -1.56512906e-04, 9.63474595e-05
            ], [
                -1.04306288e-04, -1.37411975e-04, 2.82689070e-05,
                6.56487318e-05, -1.48634164e-04, -1.84347919e-05
            ], [
                1.24452345e-04, 2.20821079e-04, 4.07114130e-04, 2.18028668e-04,
                2.73401442e-04, -2.69805576e-04
            ]]],
            dtype=float32),
        sample_id=array(
            [[2, 0, 2], [0, 0, 0], [1, 1, 1], [5, 5, 5], [3, 3, 2]],
            dtype=int32))

    expected_final_state = AttentionWrapperState(
        cell_state=LSTMStateTuple(
            c=array(
                [[
                    -2.18963176e-02, -8.04424379e-03, -1.48289464e-03,
                    1.61068402e-02, -1.37983467e-02, -7.57976994e-03,
                    -8.28560349e-03, -1.18737305e-02, 1.78835373e-02
                ], [
                    1.74205080e-02, -1.41929444e-02, -3.88092734e-03,
                    3.19708064e-02, -3.54689620e-02, -2.14698724e-02,
                    -6.21716119e-03, -1.69295724e-03, -1.94495302e-02
                ], [
                    -1.14528481e-02, 8.77819210e-03, -1.62970200e-02,
                    -1.39963552e-02, 1.34831406e-02, -1.04494914e-02,
                    6.16127765e-03, -9.41022579e-03, -6.57590060e-03
                ], [
                    -4.74753827e-02, -1.19123599e-02, -7.40140676e-05,
                    4.10552323e-02, -1.36711076e-03, 2.11795457e-02,
                    -2.80460119e-02, -5.44509329e-02, -2.91906092e-02
                ], [
                    2.25644894e-02, -1.40382675e-03, 1.92396250e-02,
                    5.49034867e-03, -1.27930511e-02, -3.15603940e-03,
                    -5.05525898e-03, 2.19191350e-02, 1.62497871e-02
                ]],
                dtype=float32),
            h=array(
                [[
                    -1.09840557e-02, -3.97477299e-03, -7.54582870e-04,
                    7.91188516e-03, -7.02184858e-03, -3.80711886e-03,
                    -4.22059745e-03, -6.05464494e-03, 8.92061181e-03
                ], [
                    8.68131686e-03, -7.16938032e-03, -1.88384682e-03,
                    1.62678920e-02, -1.76827926e-02, -1.06622791e-02,
                    -3.07528162e-03, -8.45885137e-04, -9.99388192e-03
                ], [
                    -5.71205560e-03, 4.50050412e-03, -8.07640795e-03,
                    -6.94844872e-03, 6.75682165e-03, -5.12113515e-03,
                    3.06208082e-03, -4.61743120e-03, -3.23931244e-03
                ], [
                    -2.37231534e-02, -5.88526297e-03, -3.72226204e-05,
                    2.01789513e-02, -6.75848918e-04, 1.06686354e-02,
                    -1.42624676e-02, -2.69628745e-02, -1.45034352e-02
                ], [
                    1.12585640e-02, -6.92534202e-04, 9.88917705e-03,
                    2.75237625e-03, -6.56115822e-03, -1.57997780e-03,
                    -2.54477374e-03, 1.11598391e-02, 7.94144534e-03
                ]],
                dtype=float32)),
        attention=array(
            [[
                0.00187641, 0.00207374, 0.00230583, -0.00364565, -0.00375996,
                -0.00628685
            ], [
                0.00521512, -0.00218198, 0.0035622, -0.00288951, 0.00320867,
                -0.00521919
            ], [
                -0.00113357, -0.00073713, -0.00199583, -0.00188098, -0.00503196,
                -0.00934653
            ], [
                0.00099348, -0.00437379, -0.00141532, 0.00064478, -0.0021648,
                0.00668286
            ], [
                0.00012445, 0.00022082, 0.00040711, 0.00021803, 0.0002734,
                -0.00026981
            ]],
            dtype=float32),
        time=3,
        attention_history=())

    self._testWithAttention(
        create_attention_mechanism,
        expected_final_output,
        expected_final_state,
        attention_history=True,
        name="testBahdanauNotNormalized")

  def testBahdanauNormalized(self):
    create_attention_mechanism = functools.partial(
        wrapper.BahdanauAttention, normalize=True)

    expected_final_output = BasicDecoderOutput(
        rnn_output=array(
            [[[
                6.64783875e-03, 2.94425711e-03, 5.26542449e-03, -2.64955591e-03,
                -7.95925129e-03, -5.02286293e-03
            ], [
                7.01954123e-03, 3.07301106e-03, 5.22849336e-03, -2.68844375e-03,
                -7.11239874e-03, -4.72904276e-03
            ], [
                6.62360899e-03, 3.12234787e-03, 5.51807694e-03, -2.46222341e-03,
                -7.40198931e-03, -4.85701021e-03
            ]], [[
                7.37589924e-03, -1.02620223e-03, 3.61374952e-03,
                -5.74620720e-03, 5.05625410e-03, -7.45209027e-03
            ], [
                7.61946291e-03, -1.09287468e-03, 3.78817180e-03,
                -5.78709645e-03, 4.56611114e-03, -7.96987582e-03
            ], [
                7.69207766e-03, -1.26582675e-03, 3.85218812e-03,
                -5.81111759e-03, 4.63287206e-03, -7.86337163e-03
            ]], [[
                -2.69413739e-03, 3.47183552e-04, -1.82145904e-03,
                -1.39805069e-03, -8.05486552e-03, -1.08372131e-02
            ], [
                -2.70848931e-03, 3.03293345e-04, -1.67230750e-03,
                -1.34555507e-03, -8.40565283e-03, -1.10935047e-02
            ], [
                -2.47822329e-03, 5.79408603e-04, -1.70188327e-03,
                -1.42583530e-03, -7.81180616e-03, -1.10740755e-02
            ]], [[
                1.48582947e-03, -3.88786104e-03, -9.39912978e-04,
                8.36255029e-04, -1.28223014e-03, 6.40908210e-03
            ], [
                1.18177081e-03, -4.47923271e-03, -1.05711201e-03,
                8.12121783e-04, -2.08477327e-03, 6.27523474e-03
            ], [
                9.49664740e-04, -4.28957958e-03, -1.39053771e-03,
                6.29657647e-04, -2.14899099e-03, 6.52727811e-03
            ]], [[
                -3.78854020e-04, 5.62231544e-05, 1.06837302e-04, 1.87137164e-04,
                -1.56512906e-04, 9.63474595e-05
            ], [
                -1.04306288e-04, -1.37411975e-04, 2.82689070e-05,
                6.56487318e-05, -1.48634164e-04, -1.84347919e-05
            ], [
                1.24452345e-04, 2.20821079e-04, 4.07114130e-04, 2.18028668e-04,
                2.73401442e-04, -2.69805576e-04
            ]]],
            dtype=float32),
        sample_id=array(
            [[0, 0, 0], [0, 0, 0], [1, 1, 1], [5, 5, 5], [3, 3, 2]],
            dtype=int32))

    expected_final_state = AttentionWrapperState(
        cell_state=LSTMStateTuple(
            c=array(
                [[
                    -2.19389871e-02, -7.93421268e-03, -1.45148858e-03,
                    1.61569901e-02, -1.38310911e-02, -7.59426132e-03,
                    -8.35836027e-03, -1.18763093e-02, 1.78797375e-02
                ], [
                    1.74194798e-02, -1.41677596e-02, -3.89095861e-03,
                    3.19508761e-02, -3.54519747e-02, -2.15105712e-02,
                    -6.20894879e-03, -1.72719418e-03, -1.94605980e-02
                ], [
                    -1.14357909e-02, 8.76635592e-03, -1.62690803e-02,
                    -1.39883338e-02, 1.34323873e-02, -1.04959216e-02,
                    6.09614328e-03, -9.38197412e-03, -6.57159975e-03
                ], [
                    -4.74738739e-02, -1.19136795e-02, -7.36564398e-05,
                    4.10547666e-02, -1.36771239e-03, 2.11771261e-02,
                    -2.80481018e-02, -5.44515178e-02, -2.91903559e-02
                ], [
                    2.25644894e-02, -1.40382675e-03, 1.92396250e-02,
                    5.49034867e-03, -1.27930511e-02, -3.15603940e-03,
                    -5.05525898e-03, 2.19191350e-02, 1.62497871e-02
                ]],
                dtype=float32),
            h=array(
                [[
                    -1.10049099e-02, -3.92028037e-03, -7.38571223e-04,
                    7.93652050e-03, -7.03821564e-03, -3.81436548e-03,
                    -4.25778655e-03, -6.05606195e-03, 8.91851448e-03
                ], [
                    8.68070032e-03, -7.15647917e-03, -1.88874488e-03,
                    1.62575077e-02, -1.76745858e-02, -1.06826536e-02,
                    -3.07105901e-03, -8.63034453e-04, -9.99918394e-03
                ], [
                    -5.70359221e-03, 4.49446775e-03, -8.06238409e-03,
                    -6.94446685e-03, 6.73149945e-03, -5.14409645e-03,
                    3.02969781e-03, -4.60351165e-03, -3.23720207e-03
                ], [
                    -2.37224046e-02, -5.88591257e-03, -3.70427515e-05,
                    2.01787166e-02, -6.76146999e-04, 1.06674293e-02,
                    -1.42635051e-02, -2.69631781e-02, -1.45033030e-02
                ], [
                    1.12585640e-02, -6.92534202e-04, 9.88917705e-03,
                    2.75237625e-03, -6.56115822e-03, -1.57997780e-03,
                    -2.54477374e-03, 1.11598391e-02, 7.94144534e-03
                ]],
                dtype=float32)),
        attention=array(
            [[
                0.00662361, 0.00312235, 0.00551808, -0.00246222, -0.00740199,
                -0.00485701
            ], [
                0.00769208, -0.00126583, 0.00385219, -0.00581112, 0.00463287,
                -0.00786337
            ], [
                -0.00247822, 0.00057941, -0.00170188, -0.00142584, -0.00781181,
                -0.01107408
            ], [
                0.00094966, -0.00428958, -0.00139054, 0.00062966, -0.00214899,
                0.00652728
            ], [
                0.00012445, 0.00022082, 0.00040711, 0.00021803, 0.0002734,
                -0.00026981
            ]],
            dtype=float32),
        time=3,
        attention_history=())

    self._testWithAttention(
        create_attention_mechanism,
        expected_final_output,
        expected_final_state,
        name="testBahdanauNormalized")

  def testLuongNotNormalized(self):
    create_attention_mechanism = wrapper.LuongAttention

    expected_final_output = BasicDecoderOutput(
        rnn_output=array(
            [[[
                1.74749165e-03, 1.95862399e-03, 2.12293095e-03, -3.75889172e-03,
                -4.39571124e-03, -6.32379763e-03
            ], [
                2.33045570e-03, 1.99094601e-03, 1.98377599e-03, -3.87950847e-03,
                -3.42792575e-03, -6.17497414e-03
            ], [
                1.65032526e-03, 1.96972815e-03, 2.03462853e-03, -3.82007333e-03,
                -3.46369296e-03, -6.54224353e-03
            ]], [[
                4.77780215e-03, -1.98677275e-03, 3.30950436e-03,
                -2.68179504e-03, 3.56271653e-03, -4.67860466e-03
            ], [
                5.13039157e-03, -2.02797214e-03, 3.50760575e-03,
                -2.83981953e-03, 3.13726603e-03, -5.31156827e-03
            ], [
                5.17205056e-03, -2.16446724e-03, 3.53219034e-03,
                -2.86490913e-03, 3.17879021e-03, -5.17592067e-03
            ]], [[
                -1.38538703e-03, -6.40910701e-04, -2.02864106e-03,
                -1.79018872e-03, -5.18789608e-03, -8.95875692e-03
            ], [
                -1.38620089e-03, -7.92010222e-04, -1.91070826e-03,
                -1.76206254e-03, -5.56525169e-03, -9.27332044e-03
            ], [
                -1.11966045e-03, -6.07630936e-04, -1.96643686e-03,
                -1.86803937e-03, -4.93048411e-03, -9.25842486e-03
            ]], [[
                1.50820788e-03, -3.93087184e-03, -9.52563598e-04,
                8.43994785e-04, -1.29030924e-03, 6.48857141e-03
            ], [
                1.17029145e-03, -4.45716921e-03, -1.05062663e-03,
                8.08141369e-04, -2.08062865e-03, 6.23444980e-03
            ], [
                9.67921398e-04, -4.32466762e-03, -1.40085898e-03,
                6.35969569e-04, -2.15558149e-03, 6.59212377e-03
            ]], [[
                -3.78854020e-04, 5.62231544e-05, 1.06837302e-04, 1.87137164e-04,
                -1.56512906e-04, 9.63474595e-05
            ], [
                -1.04306288e-04, -1.37411975e-04, 2.82689070e-05,
                6.56487318e-05, -1.48634164e-04, -1.84347919e-05
            ], [
                1.24452345e-04, 2.20821079e-04, 4.07114130e-04, 2.18028668e-04,
                2.73401442e-04, -2.69805576e-04
            ]]],
            dtype=float32),
        sample_id=array(
            [[2, 0, 2], [0, 0, 0], [1, 1, 1], [5, 5, 5], [3, 3, 2]],
            dtype=int32))

    expected_final_state = AttentionWrapperState(
        cell_state=LSTMStateTuple(
            c=array(
                [[
                    -2.18960866e-02, -8.04429129e-03, -1.48267671e-03,
                    1.61071159e-02, -1.37981661e-02, -7.57933082e-03,
                    -8.28570686e-03, -1.18733812e-02, 1.78834442e-02
                ], [
                    1.74204130e-02, -1.41935758e-02, -3.88074201e-03,
                    3.19713727e-02, -3.54694910e-02, -2.14688145e-02,
                    -6.21731905e-03, -1.69229065e-03, -1.94492843e-02
                ], [
                    -1.14494488e-02, 8.77974741e-03, -1.62960067e-02,
                    -1.39961652e-02, 1.34879015e-02, -1.04502086e-02,
                    6.15879148e-03, -9.40956455e-03, -6.57592434e-03
                ], [
                    -4.74739634e-02, -1.19136050e-02, -7.36759976e-05,
                    4.10547927e-02, -1.36767328e-03, 2.11772677e-02,
                    -2.80479677e-02, -5.44514805e-02, -2.91903690e-02
                ], [
                    2.25644894e-02, -1.40382675e-03, 1.92396250e-02,
                    5.49034867e-03, -1.27930511e-02, -3.15603940e-03,
                    -5.05525898e-03, 2.19191350e-02, 1.62497871e-02
                ]],
                dtype=float32),
            h=array(
                [[
                    -1.09839402e-02, -3.97479767e-03, -7.54472159e-04,
                    7.91201927e-03, -7.02175125e-03, -3.80689627e-03,
                    -4.22065007e-03, -6.05447078e-03, 8.92056432e-03
                ], [
                    8.68127123e-03, -7.16970162e-03, -1.88375649e-03,
                    1.62681788e-02, -1.76830534e-02, -1.06617520e-02,
                    -3.07536125e-03, -8.45551898e-04, -9.99375992e-03
                ], [
                    -5.71034756e-03, 4.50129062e-03, -8.07590690e-03,
                    -6.94835978e-03, 6.75921654e-03, -5.12148207e-03,
                    3.06083867e-03, -4.61710012e-03, -3.23932176e-03
                ], [
                    -2.37224493e-02, -5.88587578e-03, -3.70525813e-05,
                    2.01787278e-02, -6.76127791e-04, 1.06675029e-02,
                    -1.42634306e-02, -2.69631632e-02, -1.45033058e-02
                ], [
                    1.12585640e-02, -6.92534202e-04, 9.88917705e-03,
                    2.75237625e-03, -6.56115822e-03, -1.57997780e-03,
                    -2.54477374e-03, 1.11598391e-02, 7.94144534e-03
                ]],
                dtype=float32)),
        attention=array(
            [[
                0.00165033, 0.00196973, 0.00203463, -0.00382007, -0.00346369,
                -0.00654224
            ], [
                0.00517205, -0.00216447, 0.00353219, -0.00286491, 0.00317879,
                -0.00517592
            ], [
                -0.00111966, -0.00060763, -0.00196644, -0.00186804, -0.00493048,
                -0.00925842
            ], [
                0.00096792, -0.00432467, -0.00140086, 0.00063597, -0.00215558,
                0.00659212
            ], [
                0.00012445, 0.00022082, 0.00040711, 0.00021803, 0.0002734,
                -0.00026981
            ]],
            dtype=float32),
        time=3,
        attention_history=())

    self._testWithAttention(
        create_attention_mechanism,
        expected_final_output,
        expected_final_state,
        attention_mechanism_depth=9,
        name="testLuongNotNormalized")

  def testLuongScaled(self):
    create_attention_mechanism = functools.partial(
        wrapper.LuongAttention, scale=True)

    expected_final_output = BasicDecoderOutput(
        rnn_output=array(
            [[[
                1.74749165e-03, 1.95862399e-03, 2.12293095e-03, -3.75889172e-03,
                -4.39571124e-03, -6.32379763e-03
            ], [
                2.33045570e-03, 1.99094601e-03, 1.98377599e-03, -3.87950847e-03,
                -3.42792575e-03, -6.17497414e-03
            ], [
                1.65032526e-03, 1.96972815e-03, 2.03462853e-03, -3.82007333e-03,
                -3.46369296e-03, -6.54224353e-03
            ]], [[
                4.77780215e-03, -1.98677275e-03, 3.30950436e-03,
                -2.68179504e-03, 3.56271653e-03, -4.67860466e-03
            ], [
                5.13039157e-03, -2.02797214e-03, 3.50760575e-03,
                -2.83981953e-03, 3.13726603e-03, -5.31156827e-03
            ], [
                5.17205056e-03, -2.16446724e-03, 3.53219034e-03,
                -2.86490913e-03, 3.17879021e-03, -5.17592067e-03
            ]], [[
                -1.38538703e-03, -6.40910701e-04, -2.02864106e-03,
                -1.79018872e-03, -5.18789608e-03, -8.95875692e-03
            ], [
                -1.38620089e-03, -7.92010222e-04, -1.91070826e-03,
                -1.76206254e-03, -5.56525169e-03, -9.27332044e-03
            ], [
                -1.11966045e-03, -6.07630936e-04, -1.96643686e-03,
                -1.86803937e-03, -4.93048411e-03, -9.25842486e-03
            ]], [[
                1.50820788e-03, -3.93087184e-03, -9.52563598e-04,
                8.43994785e-04, -1.29030924e-03, 6.48857141e-03
            ], [
                1.17029145e-03, -4.45716921e-03, -1.05062663e-03,
                8.08141369e-04, -2.08062865e-03, 6.23444980e-03
            ], [
                9.67921398e-04, -4.32466762e-03, -1.40085898e-03,
                6.35969569e-04, -2.15558149e-03, 6.59212377e-03
            ]], [[
                -3.78854020e-04, 5.62231544e-05, 1.06837302e-04, 1.87137164e-04,
                -1.56512906e-04, 9.63474595e-05
            ], [
                -1.04306288e-04, -1.37411975e-04, 2.82689070e-05,
                6.56487318e-05, -1.48634164e-04, -1.84347919e-05
            ], [
                1.24452345e-04, 2.20821079e-04, 4.07114130e-04, 2.18028668e-04,
                2.73401442e-04, -2.69805576e-04
            ]]],
            dtype=float32),
        sample_id=array(
            [[2, 0, 2], [0, 0, 0], [1, 1, 1], [5, 5, 5], [3, 3, 2]],
            dtype=int32))

    expected_final_state = AttentionWrapperState(
        cell_state=LSTMStateTuple(
            c=array(
                [[
                    -2.18960866e-02, -8.04429129e-03, -1.48267671e-03,
                    1.61071159e-02, -1.37981661e-02, -7.57933082e-03,
                    -8.28570686e-03, -1.18733812e-02, 1.78834442e-02
                ], [
                    1.74204130e-02, -1.41935758e-02, -3.88074201e-03,
                    3.19713727e-02, -3.54694910e-02, -2.14688145e-02,
                    -6.21731905e-03, -1.69229065e-03, -1.94492843e-02
                ], [
                    -1.14494488e-02, 8.77974741e-03, -1.62960067e-02,
                    -1.39961652e-02, 1.34879015e-02, -1.04502086e-02,
                    6.15879148e-03, -9.40956455e-03, -6.57592434e-03
                ], [
                    -4.74739634e-02, -1.19136050e-02, -7.36759976e-05,
                    4.10547927e-02, -1.36767328e-03, 2.11772677e-02,
                    -2.80479677e-02, -5.44514805e-02, -2.91903690e-02
                ], [
                    2.25644894e-02, -1.40382675e-03, 1.92396250e-02,
                    5.49034867e-03, -1.27930511e-02, -3.15603940e-03,
                    -5.05525898e-03, 2.19191350e-02, 1.62497871e-02
                ]],
                dtype=float32),
            h=array(
                [[
                    -1.09839402e-02, -3.97479767e-03, -7.54472159e-04,
                    7.91201927e-03, -7.02175125e-03, -3.80689627e-03,
                    -4.22065007e-03, -6.05447078e-03, 8.92056432e-03
                ], [
                    8.68127123e-03, -7.16970162e-03, -1.88375649e-03,
                    1.62681788e-02, -1.76830534e-02, -1.06617520e-02,
                    -3.07536125e-03, -8.45551898e-04, -9.99375992e-03
                ], [
                    -5.71034756e-03, 4.50129062e-03, -8.07590690e-03,
                    -6.94835978e-03, 6.75921654e-03, -5.12148207e-03,
                    3.06083867e-03, -4.61710012e-03, -3.23932176e-03
                ], [
                    -2.37224493e-02, -5.88587578e-03, -3.70525813e-05,
                    2.01787278e-02, -6.76127791e-04, 1.06675029e-02,
                    -1.42634306e-02, -2.69631632e-02, -1.45033058e-02
                ], [
                    1.12585640e-02, -6.92534202e-04, 9.88917705e-03,
                    2.75237625e-03, -6.56115822e-03, -1.57997780e-03,
                    -2.54477374e-03, 1.11598391e-02, 7.94144534e-03
                ]],
                dtype=float32)),
        attention=array(
            [[
                0.00165033, 0.00196973, 0.00203463, -0.00382007, -0.00346369,
                -0.00654224
            ], [
                0.00517205, -0.00216447, 0.00353219, -0.00286491, 0.00317879,
                -0.00517592
            ], [
                -0.00111966, -0.00060763, -0.00196644, -0.00186804, -0.00493048,
                -0.00925842
            ], [
                0.00096792, -0.00432467, -0.00140086, 0.00063597, -0.00215558,
                0.00659212
            ], [
                0.00012445, 0.00022082, 0.00040711, 0.00021803, 0.0002734,
                -0.00026981
            ]],
            dtype=float32),
        time=3,
        attention_history=())

    self._testWithAttention(
        create_attention_mechanism,
        expected_final_output,
        expected_final_state,
        attention_mechanism_depth=9,
        name="testLuongScaled")


if __name__ == "__main__":
  test.main()
