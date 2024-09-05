/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/compiler/mlir/tfrt/translate/tfrt_compile_options.h"

#include <ostream>
#include <string>
#include <vector>

namespace tensorflow {

std::ostream& operator<<(std::ostream& os,
                         TfrtDeviceInfraTarget device_target) {
  switch (device_target) {
    case TfrtDeviceInfraTarget::kCpu:
      return os << "Cpu";
    case TfrtDeviceInfraTarget::kTpurt:
      return os << "Tpurt";
    case TfrtDeviceInfraTarget::kTfFallback:
      return os << "TfFallback";
    case TfrtDeviceInfraTarget::kBridgeFallback:
      return os << "BridgeFallback";
    case TfrtDeviceInfraTarget::kGpu:
      return os << "Gpu";
  }
}

std::ostream& operator<<(std::ostream& os, const TfrtCompileOptions& options) {
  return os << "{" << "variable_device = " << options.variable_device
            << ", default_device = " << options.default_device
            << ", enable_optimizer = " << options.enable_optimizer
            << ", enable_grappler = " << options.enable_grappler
            << ", force_data_format = " << options.force_data_format
            << ", device_target = " << options.device_target
            << ", tpu_fuse_ops = " << options.tpu_fuse_ops
            << ", tpu_move_resource_gather_to_host = "
            << options.tpu_move_resource_gather_to_host
            << ", tpu_gather_table_width_threshold_bytes = "
            << options.tpu_gather_table_width_threshold_bytes
            << ", use_tpu_host_allocator_for_inputs = "
            << options.use_tpu_host_allocator_for_inputs
            << ", hoist_invariant_ops = " << options.hoist_invariant_ops
            << ", enable_while_parallel_iterations = "
            << options.enable_while_parallel_iterations
            << ", cost_threshold = " << options.cost_threshold
            << ", min_num_batch_threads = " << options.min_num_batch_threads
            << ", min_max_enqueued_batches = "
            << options.min_max_enqueued_batches
            << ", batch_padding_policy = " << options.batch_padding_policy
            << ", merge_inter_dependent_streams = "
            << options.merge_inter_dependent_streams
            << ", decompose_resource_ops = " << options.decompose_resource_ops
            << ", compile_to_sync_tfrt_dialect = "
            << options.compile_to_sync_tfrt_dialect << "}";
}

}  // namespace tensorflow
