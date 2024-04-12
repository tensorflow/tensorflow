# Copyright 2024 The TensorFlow Authors. All Rights Reserved.
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
"""Test Output for Multi-signature Multi-block Transformer Models.

Autoregressive decode is done in TF Lite graphs via multi-signature support.
(One signature for prefill, another for decdoe). A prefill of N tokens
followed by a decode on 1 token should be equivalent to a prefill of N+1
tokens.
"""


import numpy as np

from tensorflow.lite.python import interpreter
from tensorflow.python.framework import test_util
from tensorflow.python.platform import resource_loader
from tensorflow.python.platform import test


def run_prefill_alone(model_path, seq_len):
  interp = interpreter.InterpreterWithCustomOps(
      model_path=model_path,
      custom_op_registerers=['GenAIOpsRegisterer'],
  )

  # Allocate tensors.
  interp.allocate_tensors()
  prefill_runner = interp.get_signature_runner('prefill')

  # prefill
  token_data = np.zeros(
      prefill_runner.get_input_details()['args_0']['shape'],
      dtype=prefill_runner.get_input_details()['args_0']['dtype'],
  )
  custom_inputs = [1, 2, 3, 4, 5, 6, 1]
  token_data[0, : len(custom_inputs)] = custom_inputs
  pos_data = np.arange(
      0,
      seq_len,
      dtype=prefill_runner.get_input_details()['args_1']['dtype'],
  )
  return prefill_runner(args_0=token_data, args_1=pos_data)['output_0']


def run_prefill_then_decode(model_path, seq_len):
  interp = interpreter.InterpreterWithCustomOps(
      model_path=model_path,
      custom_op_registerers=['GenAIOpsRegisterer'],
  )

  # Allocate tensors.
  interp.allocate_tensors()
  prefill_runner = interp.get_signature_runner('prefill')
  decode_runner = interp.get_signature_runner('decode')

  # prefill
  token_data = np.zeros(
      prefill_runner.get_input_details()['args_0']['shape'],
      dtype=prefill_runner.get_input_details()['args_0']['dtype'],
  )
  custom_inputs = [1, 2, 3, 4, 5, 6]
  token_data[0, : len(custom_inputs)] = custom_inputs
  pos_data = np.arange(
      0,
      seq_len,
      dtype=prefill_runner.get_input_details()['args_1']['dtype'],
  )
  _ = prefill_runner(args_0=token_data, args_1=pos_data)

  token_data = np.array(
      [[1]], dtype=decode_runner.get_input_details()['args_0']['dtype']
  )
  pos_data = np.array(
      [6], dtype=decode_runner.get_input_details()['args_1']['dtype']
  )
  return decode_runner(args_0=token_data, args_1=pos_data)['output_0']


class TransformerTest(test_util.TensorFlowTestCase):

  def test_prefill_decode(self):
    seq_len = 100
    model_path = resource_loader.get_path_to_datafile(
        'testdata/toy_model_prefill_decode.tflite'
    )
    prefill_logits = run_prefill_alone(model_path, seq_len)
    decode_logits = run_prefill_then_decode(model_path, seq_len)
    # The output at sequence index = 6 (7th token output) must match
    # the decode logits.
    self.assertTrue(np.allclose(prefill_logits[:, 6, :], decode_logits))


if __name__ == '__main__':
  test.main()
