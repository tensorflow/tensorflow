/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_CORE_TFRT_GPU_KERNEL_GPU_RUNNER_H_
#define TENSORFLOW_CORE_TFRT_GPU_KERNEL_GPU_RUNNER_H_

#include <cstdint>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "llvm/ADT/SmallVector.h"
#include "xla/tsl/framework/serving_device_selector.h"
#include "tensorflow/core/framework/device.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/runtime_fallback/kernel/kernel_fallback_compat_request_state.h"
#include "tensorflow/core/tfrt/utils/fallback_tensor.h"
#include "tensorflow/core/tfrt/utils/gpu_variables_table.h"
#include "tfrt/host_context/async_value_ref.h"  // from @tf_runtime
#include "tfrt/host_context/execution_context.h"  // from @tf_runtime
#include "tfrt/support/forward_decls.h"  // from @tf_runtime

namespace tensorflow {
namespace gpu {

constexpr char kGpuRunnerResourceName[] = "GpuRunnerResource";

struct GpuRunInputs {
  std::vector<tfrt_stub::FallbackTensor> args;
  int num_outputs;
  std::vector<int64_t> resource_indices;
  std::vector<int64_t> used_output_indices;
  std::string func_name;
  Device* cpu_device;
  absl::flat_hash_map<int, Device*> gpu_devices;
  const tfd::KernelFallbackCompatRequestState* fallback_request_state;
  tfrt::HostContext* host_ctx;
};

class GpuRunner {
 public:
  explicit GpuRunner(tsl::ServingDeviceSelector* serving_device_selector)
      : serving_device_selector_(serving_device_selector) {}

  // This compiles the given program and runs the given input tensors in
  // `run_inputs`, and returns the output tensor AsyncValues.
  absl::StatusOr<
      llvm::SmallVector<tfrt::AsyncValueRef<tfrt_stub::FallbackTensor>>>
  Run(GpuRunInputs run_inputs);

 private:
  tsl::ServingDeviceSelector* serving_device_selector_;
  tfrt::gpu::GpuVariablesTable vars_table_;
};

}  // namespace gpu
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_TFRT_GPU_KERNEL_GPU_RUNNER_H_
