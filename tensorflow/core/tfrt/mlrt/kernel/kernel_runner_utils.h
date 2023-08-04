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
#ifndef TENSORFLOW_CORE_TFRT_MLRT_KERNEL_KERNEL_RUNNER_UTILS_H_
#define TENSORFLOW_CORE_TFRT_MLRT_KERNEL_KERNEL_RUNNER_UTILS_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/optimization.h"
#include "absl/cleanup/cleanup.h"
#include "tensorflow/core/profiler/lib/traceme.h"
#include "tensorflow/core/runtime_fallback/kernel/kernel_fallback_compat_request_state.h"
#include "tensorflow/core/runtime_fallback/kernel/kernel_fallback_utils.h"
#include "tensorflow/core/tfrt/fallback/op_kernel_runner.h"
#include "tensorflow/core/tfrt/mlrt/interpreter/context.h"
#include "tensorflow/core/tfrt/mlrt/interpreter/register_span.h"
#include "tensorflow/core/tfrt/mlrt/kernel/context.h"

namespace tensorflow {
namespace tf_mlrt {

void LaunchAsyncOpKernel(const tfrt_stub::OpKernelRunner& kernel_runner,
                         const tfrt_stub::OpKernelRunState& run_state,
                         const OpKernelContext::Params& params,
                         mlrt::RegisterSpan results,
                         std::shared_ptr<tensorflow::DeviceBase> custom_device);

inline void SetUpParams(const tfrt_stub::OpKernelRunner& kernel_runner,
                        absl::Span<const TensorValue> input_tf_tensor_values,
                        OpKernelContext::Params& params) {
  params.inputs = input_tf_tensor_values;
  params.op_kernel = kernel_runner.op_kernel();
  params.input_alloc_attrs = kernel_runner.input_alloc_attrs();
  params.output_attr_array = kernel_runner.output_alloc_attrs().data();
}

template <bool IsAsync, typename Frame>
void ExecuteKernelRunner(
    Frame& frame, Context& context,
    const tfd::KernelFallbackCompatRequestState& fallback_request_state,
    const tfrt_stub::OpKernelRunner& kernel_runner) {
  tsl::profiler::TraceMe trace_me([&]() -> std::string {
    return tsl::profiler::TraceMeOp(
        kernel_runner.op_kernel()->name_view(),
        kernel_runner.op_kernel()->type_string_view());
  });

  auto args = frame.args();
  auto last_uses = frame.last_uses();

  auto& run_state = context.run_state();
  auto& tensor_buffers = run_state.tensor_buffers;

  auto clean_up_inputs = absl::MakeCleanup([&]() {
    for (const auto* buffer : tensor_buffers) {
      DCHECK(buffer);
      buffer->Unref();
    }
    tensor_buffers.clear();
  });

  // Prepare the input tensors.
  auto& input_tf_tensor_values = run_state.input_tf_tensor_values;
  input_tf_tensor_values.resize(args.size());
  for (int i = 0; i < args.size(); ++i) {
    auto& fallback_tensor = args[i];
    // If the argument is immutable or it is the last use in the current scope,
    // we can just keep the reference without copying that invovles expensive
    // atomic reference counting. And if it is the last use, it can enable
    // buffer forwarding optimization in many tensorflow OpKernels.
    if (!fallback_tensor.is_immutable() && !last_uses[i]) {
      if (const auto* buffer = fallback_tensor.buffer()) {
        buffer->Ref();
        tensor_buffers.push_back(buffer);
      }
    }
    input_tf_tensor_values[i].tensor = &fallback_tensor.tensor();
  }

  auto& params = context.params();
  SetUpParams(kernel_runner, input_tf_tensor_values, params);

  auto results = frame.results();

  if constexpr (!IsAsync) {
    tensorflow::DeviceBase* device = nullptr;
    if constexpr (Frame::kUseCustomDevice) {
      // If the kernel is using custom device, save the current device and
      // change to the custom device.
      device = params.device;
      params.device = frame.device().get();
    }

    auto& op_kernel_context = context.op_kernel_context();
    op_kernel_context.ResetOutputs(results.size());

    kernel_runner.Run(&op_kernel_context);

    if constexpr (Frame::kUseCustomDevice) {
      // We need to restore the device as params will be reused by kernels
      // invoked later.
      params.device = device;
    }

    if (ABSL_PREDICT_FALSE(!op_kernel_context.status().ok())) {
      frame.execution_context().Fail(op_kernel_context.status());
      return;
    }

    for (int i = 0; i < op_kernel_context.num_outputs(); ++i) {
      DCHECK(op_kernel_context.mutable_output(i));
      results[i].template Emplace<tensorflow::tfrt_stub::FallbackTensor>(
          std::move(*op_kernel_context.mutable_output(i)));
    }
  } else {
    std::shared_ptr<tensorflow::DeviceBase> custom_device = nullptr;
    if constexpr (Frame::kUseCustomDevice) {
      custom_device = frame.device();
    }

    LaunchAsyncOpKernel(kernel_runner, run_state, params, results,
                        std::move(custom_device));
  }

  auto reg_span = args.reg_span();
  for (int i = 0; i < last_uses.size(); ++i) {
    if (last_uses[i]) {
      reg_span[i].template Destroy<tensorflow::tfrt_stub::FallbackTensor>();
    }
  }
}

}  // namespace tf_mlrt
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_TFRT_MLRT_KERNEL_KERNEL_RUNNER_UTILS_H_
