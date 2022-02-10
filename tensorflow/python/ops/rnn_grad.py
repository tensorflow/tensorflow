# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Gradients for (block) GRU/LSTM operators."""
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_rnn_ops


def _block_lstm_grad(op, *grads):
  """Gradient for the BlockLSTM op."""
  seq_len_max, x, cs_prev, h_prev, w, wci, wcf, wco, b = op.inputs
  i, cs, f, o, ci, co, h = op.outputs
  _, cs_grad, _, _, _, _, h_grad = grads
  (x_grad, cs_prev_grad, h_prev_grad, w_grad, wci_grad, wcf_grad, wco_grad,
   b_grad) = gen_rnn_ops.block_lstm_grad(
       seq_len_max=seq_len_max,
       x=x,
       cs_prev=cs_prev,
       h_prev=h_prev,
       w=w,
       wci=wci,
       wcf=wcf,
       wco=wco,
       b=b,
       i=i,
       cs=cs,
       f=f,
       o=o,
       ci=ci,
       co=co,
       h=h,
       cs_grad=cs_grad,
       h_grad=h_grad,
       use_peephole=op.get_attr("use_peephole"))
  return (None, x_grad, cs_prev_grad, h_prev_grad, w_grad, wci_grad, wcf_grad,
          wco_grad, b_grad)


ops.RegisterGradient("BlockLSTM")(_block_lstm_grad)
ops.RegisterGradient("BlockLSTMV2")(_block_lstm_grad)
