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
#ifndef TENSORFLOW_COMPILER_MLIR_TFRT_TRANSLATE_MLRT_TEST_UTILS_H_
#define TENSORFLOW_COMPILER_MLIR_TFRT_TRANSLATE_MLRT_TEST_UTILS_H_

#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/tfrt/graph_executor/sync_resource_state.h"
#include "tensorflow/core/tfrt/mlrt/attribute/attribute.h"
#include "tensorflow/core/tfrt/mlrt/bytecode/bytecode.h"
#include "tensorflow/core/tfrt/mlrt/bytecode/kernel.h"
#include "tensorflow/core/tfrt/mlrt/interpreter/context.h"
#include "tensorflow/core/tfrt/mlrt/interpreter/interpreter_testutil.h"
#include "tensorflow/core/tfrt/mlrt/interpreter/value.h"
#include "tensorflow/core/tfrt/stubs/tfrt_native_lowering_stub.h"
#include "tensorflow/core/tfrt/utils/tensor_util.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"
#include "tfrt/host_context/concurrent_work_queue.h"  // from @tf_runtime
#include "tfrt/host_context/execution_context.h"  // from @tf_runtime
#include "tfrt/host_context/host_allocator.h"  // from @tf_runtime
#include "tfrt/host_context/host_context.h"  // from @tf_runtime
#include "tfrt/support/string_util.h"  // from @tf_runtime
#include "tfrt/tensor/dense_host_tensor.h"  // from @tf_runtime
#include "tfrt/tensor/dense_tensor_utils.h"  // from @tf_runtime

namespace mlrt {
namespace testing {

absl::StatusOr<std::string> EncodeAttribute(const tensorflow::AttrValue& attr);

absl::Status EncodeAttributes(AttributeTable& attributes,
                              const tensorflow::AttrValueMap& attr_map);

absl::StatusOr<std::pair<mlrt::bc::Kernel, mlrt::bc::Vector<mlrt::bc::String>>>
CreateKernelAndAttrs(int num_inputs, int num_outputs,
                     mlrt::ExecutionContext& exec_ctx, mlrt::bc::Buffer* buffer,
                     const tensorflow::AttrValueMap& attrs = {});

template <typename T>
absl::Status TestMlrtKernel(
    absl::string_view kernel_name, absl::Span<mlrt::Value> regs,
    tfrt::HostContext* host, int num_inputs, int num_outputs,
    absl::Span<const tensorflow::Tensor> expected_outputs,
    mlrt::KernelRegistry* registry, bool approx_equal = false,
    const tensorflow::AttrValueMap& attrs = {}) {
  mlrt::ExecutionContext execution_context(nullptr);

  mlrt::bc::Buffer buffer;
  TF_ASSIGN_OR_RETURN(auto kernel_and_attrs,
                      CreateKernelAndAttrs(num_inputs, num_outputs,
                                           execution_context, &buffer, attrs));

  tensorflow::tfrt_stub::SyncResourceState sync_resource_state;
  tfrt::AddSyncContext(execution_context, *host, &sync_resource_state);

  auto kernel_fn = registry->Get(kernel_name);
  mlrt::KernelFrame::State state(regs, kernel_and_attrs.second,
                                 &execution_context);
  mlrt::KernelFrame frame(&state);
  frame.set_kernel(kernel_and_attrs.first);

  kernel_fn(frame);

  TF_RETURN_IF_ERROR(execution_context.status());

  for (int i = 0, j = num_inputs; i < expected_outputs.size(); ++i, ++j) {
    const auto& expected_output = expected_outputs[i];
    auto expected_dht = tfrt::ConvertTfTensorToDHT(expected_output);
    if (!expected_dht) {
      return absl::InternalError(tfrt::StrCat(expected_dht.takeError()));
    }

    if (!approx_equal) {
      if (!tfrt::TensorEqual<T>(regs[j].Get<tfrt::DenseHostTensor>(),
                                *expected_dht)) {
        return absl::InternalError(
            absl::StrCat("wrong result for ", kernel_name));
      }
    } else {
      if (!tfrt::TensorApproxEqual<T>(regs[j].Get<tfrt::DenseHostTensor>(),
                                      *expected_dht)) {
        return absl::InternalError(
            absl::StrCat("wrong result for ", kernel_name));
      }
    }
  }

  return absl::OkStatus();
}

}  // namespace testing
}  // namespace mlrt

#endif  // TENSORFLOW_COMPILER_MLIR_TFRT_TRANSLATE_MLRT_TEST_UTILS_H_
