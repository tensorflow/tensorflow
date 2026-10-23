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

// Same TPU cluster as ifrt_cluster_variable_arg.mlir, but its argument is a
// regular input, so the IfrtCall gets variable_arg_indices = [].
module attributes {tf.versions = {producer = 888 : i32}, tf.devices = ["/job:worker/replica:0/task:0/device:CPU:0", "/job:worker/replica:0/task:0/device:TPU_SYSTEM:0", "/job:worker/replica:0/task:0/device:TPU:0"]} {
  func.func @main(%arg0: tensor<1x3xf32>) -> tensor<1x3xf32> {
    %result = "tf_device.cluster_func"(%arg0) {_replication_info = "cluster0", func = @add_one, num_cores_per_replica = 1, step_marker_location = "", input_sharding_configuration = [""], output_sharding_configuration = [""], use_spmd_for_xla_partitioning = false} : (tensor<1x3xf32>) -> tensor<1x3xf32>
    func.return %result : tensor<1x3xf32>
  }
  func.func @add_one(%arg0: tensor<1x3xf32>) -> tensor<1x3xf32> {
    %one = "tf.Const"() <{value = dense<1.0> : tensor<1x3xf32>}> : () -> tensor<1x3xf32>
    %sum = "tf.AddV2"(%arg0, %one) : (tensor<1x3xf32>, tensor<1x3xf32>) -> tensor<1x3xf32>
    func.return %sum : tensor<1x3xf32>
  }
}
