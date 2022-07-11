/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/runtime_fallback/kernel/kernel_fallback_utils.h"

#include "absl/container/inlined_vector.h"
#include "tensorflow/core/framework/device.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/runtime_fallback/kernel/kernel_fallback_compat_request_state.h"
#include "tensorflow/core/tfrt/fallback/op_kernel_runner.h"

namespace tensorflow {
namespace tfd {

void SetUpParams(const tfrt_stub::OpKernelRunner& runner,
                 const KernelFallbackCompatRequestState& fallback_request_state,
                 tensorflow::Device* device,
                 tfrt_stub::OpKernelRunState& run_state) {
  auto& params = run_state.params;
  params.inputs = &run_state.input_tf_tensor_values;
  params.device = device;
  params.op_kernel = runner.op_kernel();
  // Still use original device's resource_manager.
  params.resource_manager = runner.resource_manager();
  params.input_alloc_attrs = &runner.input_alloc_attrs();
  params.output_attr_array = runner.output_alloc_attrs().data();
  params.step_container = fallback_request_state.step_container();
  // Following two parameters are used to support executing tf.data via
  // fallback.
  params.function_library = runner.function_library_runtime();
  params.runner = fallback_request_state.runner();
  params.collective_executor = fallback_request_state.collective_executor();
  params.rendezvous = fallback_request_state.rendezvous();
  params.session_metadata = &fallback_request_state.session_metadata();
  params.cancellation_manager = fallback_request_state.cancellation_manager();
}

// Return the device to be used for the fallback kernel execution. The device is
// guaranteed to be alive during the graph execution.
tensorflow::Device* GetDeviceFromFallbackState(
    const KernelFallbackCompatRequestState& fallback_request_state,
    const tfrt_stub::OpKernelRunner& kernel_runner) {
  // Return the user-specified the custom device instead, (eg. to use a custom
  // thread pool).
  //
  // The device handling is similar to TF1 code in the below link:
  // http://cs/?q=f:common_runtime%2Fexecutor.cc:692%20package:piper&rcl=351575626
  auto* device = kernel_runner.device();
  if (auto* custom_device = fallback_request_state.custom_device(device)) {
    return custom_device;
  }
  return device;
}

}  // namespace tfd
}  // namespace tensorflow
