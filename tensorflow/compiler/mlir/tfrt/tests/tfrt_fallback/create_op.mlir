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
  %ch1 = "tfrt_fallback_async.createop"(%ch0)
   {
      device = "cpu",
      num_args = 1 : i64,
      op_attrs = [["Targuments", []],
                  ["output_shapes", [#corert.shape<>]],
                  ["output_types", [i64]]],
      op_func_attrs = [["f", "dummy_fn"]],
      op_key = 1 : i64,
      op_name = "tf.FlatMapDataset"
   } : (!tfrt.chain) -> (!tfrt.chain)
  tfrt.return %ch1 : !tfrt.chain
}
