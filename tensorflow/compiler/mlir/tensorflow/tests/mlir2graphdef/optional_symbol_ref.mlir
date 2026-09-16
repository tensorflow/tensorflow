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
// RUN: tf-mlir-translate -mlir-to-graphdef %s | tf-mlir-translate -graphdef-to-mlir | tf-mlir-translate -mlir-to-graphdef | FileCheck %s

// Verifies that optional symbol ref attributes that aren't optional in TensorFlow are handled by setting the value to an empty string.
module attributes {tf.versions = {bad_consumers = [], min_consumer = 12 : i32, producer = 458 : i32}} {
  func.func @main() {
    tf_executor.graph {
      %control = tf_executor.island wraps "tf.XlaHostCompute"() {_xla_original_oc_node_name = "Add", _xla_token_input_nodes = ["_xla_token_arg_node"], ancestors = [], cost_estimate_ns = 1024 : i64, key = "host_compute_channel_1_retvals", send_key = "", recv_key = "", shapes = [], tpu_core = 0 : i64} : () -> ()
      tf_executor.fetch
    }
    func.return
  }
}

// CHECK: op: "XlaHostCompute"

// CHECK:       attr {
// CHECK:         key: "shape_inference_graph"
// CHECK:         value {
// CHECK:           func {
// CHECK-NEXT       }
