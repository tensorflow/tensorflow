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

module attributes {tf.versions = {producer = 888 : i32}, tf.devices = ["/job:localhost/replica:0/task:0/device:CPU:0", "/job:localhost/replica:0/task:0/device:TPU_SYSTEM:0", "/job:localhost/replica:0/task:0/device:TPU:0", "/job:localhost/replica:0/task:0/device:TPU:1", "/job:localhost/replica:0/task:0/device:TPU:2", "/job:localhost/replica:0/task:0/device:TPU:3", "/job:localhost/replica:0/task:0/device:TPU:4", "/job:localhost/replica:0/task:0/device:TPU:5", "/job:localhost/replica:0/task:0/device:TPU:6", "/job:localhost/replica:0/task:0/device:TPU:7"]} {
  func.func @main(%arg0: tensor<*xf32> {tf.device = "/job:localhost/replica:0/task:0/device:CPU:0"}) {
    "tf_device.cluster_func"(%arg0) <{func = @empty_func}> {_dynamic_arg_index = [], _replication_info = "cluster", _xla_compile_device_type = "TPU", allow_soft_placement = false, computation_shape = [], device = "", device_assignment = [0, 0, 0, 0, 0, 0, 0, 1], host_compute_core = [], input_sharding_configuration = ["{devices=[2,1]0,1}"], num_cores_per_replica = 2 : i64, output_sharding_configuration = [""], padding_map = [], step_marker_location = "STEP_MARK_AT_ENTRY", topology = "\0A\04\02\02\01\02\10\01\18\08\22 \00\00\00\00\00\00\00\01\01\00\00\00\01\00\00\01\00\01\00\00\00\01\00\01\01\01\00\00\01\01\00\01", tpu_compile_options_proto = "", use_spmd_for_xla_partitioning = true, use_tpu = true} : (tensor<*xf32>) -> (tensor<*xf32>)
    func.return
  }
  func.func @empty_func(%arg0: tensor<*xf32>) -> tensor<*xf32> {
    func.return %arg0 : tensor<*xf32>
  }
}
