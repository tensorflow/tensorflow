// Copyright 2026 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ==============================================================================

func.func @init(%in_ch: !tfrt.chain) -> !tfrt.chain {
  %ch0 = "tfrt_fallback_async.createop"(%in_ch)
    {
      device = "cpu",
      num_args = 2 : i64,
      op_attrs = [["T", i32]],
      op_func_attrs = [],
      op_key = 0 : i64,
      op_name = "tf.AddV2"
    } : (!tfrt.chain) -> (!tfrt.chain)
  tfrt.return %ch0 : !tfrt.chain
}

func.func @run(%in_ch: !tfrt.chain) -> (!tfrt.chain, !tfrt_fallback.tf_tensor){
  %x_th= corert.const_dense_tensor dense<1> : tensor<i32>
  %x = tfrt_fallback_async.corert_tensorhandle_to_fallback_tensor %x_th {_tfrt_cost = 1 : i64, device = "cpu"} : (!corert.tensorhandle) -> (!tfrt_fallback.tf_tensor)
  %res = tfrt_fallback_async.executeop key(0) cost(100) device("cpu") "tf.AddV2"(%x, %x) { T = i32 } : 1
  tfrt.return %in_ch, %res : !tfrt.chain, !tfrt_fallback.tf_tensor
}
