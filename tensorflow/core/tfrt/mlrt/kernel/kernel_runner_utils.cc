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
#include "tensorflow/core/tfrt/mlrt/kernel/kernel_runner_utils.h"

#include <memory>
#include <utility>
#include <vector>

#include "tensorflow/core/framework/device_base.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/tfrt/fallback/op_kernel_runner.h"
#include "tensorflow/core/tfrt/mlrt/interpreter/future.h"
#include "tensorflow/core/tfrt/mlrt/interpreter/register_span.h"

namespace tensorflow {
namespace tf_mlrt {

void LaunchAsyncOpKernel(
    const tfrt_stub::OpKernelRunner& kernel_runner,
    const tfrt_stub::OpKernelRunState& run_state,
    const OpKernelContext::Params& params, mlrt::RegisterSpan results,
    std::shared_ptr<tensorflow::DeviceBase> custom_device) {
  struct AsyncState {
    explicit AsyncState(const tfrt_stub::OpKernelRunState& rs,
                        const OpKernelContext::Params& params, int num_outputs,
                        std::shared_ptr<tensorflow::DeviceBase> device)
        : run_state(rs.input_tf_tensor_values, params, device.get()),
          context(&run_state.params, num_outputs),
          custom_device(std::move(device)) {}

    tfrt_stub::OpKernelRunState run_state;
    OpKernelContext context;

    std::vector<mlrt::Promise> results;
    std::shared_ptr<tensorflow::DeviceBase> custom_device;
  };

  DCHECK_EQ(results.size(), kernel_runner.op_kernel()->num_outputs());
  auto async_state = std::make_shared<AsyncState>(
      run_state, params, results.size(), std::move(custom_device));

  async_state->results.reserve(results.size());
  for (int i = 0; i < results.size(); ++i) {
    auto promise =
        mlrt::Promise::Allocate<tensorflow::tfrt_stub::FallbackTensor>();

    results[i].Set(promise.GetFuture());
    async_state->results.push_back(std::move(promise));
  }

  auto* op_kernel_context_ptr = &async_state->context;

  auto done_callback = [async_state = std::move(async_state)]() {
    auto& op_kernel_context = async_state->context;

    if (!op_kernel_context.status().ok()) {
      for (auto& result : async_state->results) {
        std::move(result).SetError(op_kernel_context.status());
      }
      return;
    }

    for (int i = 0; i < op_kernel_context.num_outputs(); ++i) {
      DCHECK(op_kernel_context.mutable_output(i));
      std::move(async_state->results[i])
          .Set<tensorflow::tfrt_stub::FallbackTensor>(
              std::move(*op_kernel_context.mutable_output(i)));
    }
  };

  kernel_runner.RunAsync(op_kernel_context_ptr, std::move(done_callback));
}

}  // namespace tf_mlrt
}  // namespace tensorflow
