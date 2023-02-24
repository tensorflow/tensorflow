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
#include "tensorflow/core/tfrt/graph_executor/synchronous_graph_executor.h"

#include <memory>
#include <string>
#include <utility>

#include "learning/brain/experimental/tfrt/mlrt/application/tensorflow/kernel/kernel.h"
#include "learning/brain/experimental/tfrt/native_lowering/kernels/math_kernels.h"
#include "learning/brain/experimental/tfrt/native_lowering/kernels/sync_fallback_kernels.h"
#include "absl/status/statusor.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/tfrt/fallback/fallback_state.h"
#include "tensorflow/core/tfrt/graph_executor/graph_executor.h"
#include "tensorflow/core/tfrt/runtime/runtime.h"
#include "tensorflow/core/tfrt/utils/error_util.h"

namespace tensorflow {
namespace tfrt_stub {
namespace {

tensorflow::tfrt_stub::GraphExecutionOptions GetGraphExecutionOptions(
    tensorflow::tfrt_stub::Runtime* runtime) {
  ::tensorflow::tfrt_stub::GraphExecutionOptions options(runtime);
  auto& compile_options = options.compile_options;
  compile_options.variable_device = tensorflow::DeviceNameUtils::FullName(
      /*job=*/"localhost", /*replica=*/0,
      /*task=*/0, /*type=*/"CPU", /*id=*/0);
  compile_options.enable_grappler = true;
  compile_options.hoist_invariant_ops = true;
  compile_options.sink_in_invariant_ops = false;
  compile_options.cost_threshold = 1024;
  compile_options.compile_to_sync_tfrt_dialect = true;
  return options;
}

}  // namespace

absl::StatusOr<std::unique_ptr<SynchronousGraphExecutor>>
SynchronousGraphExecutor::Create(
    const GraphDef& graph,
    std::unique_ptr<mlrt::KernelRegistry> kernel_registry) {
  auto runtime =
      tensorflow::tfrt_stub::Runtime::Create(/*num_inter_op_threads=*/1);
  tensorflow::tfrt_stub::GraphExecutionOptions graph_execution_options =
      GetGraphExecutionOptions(runtime.get());
  const auto& fdef_lib = graph.library();
  auto session_options = tensorflow::tfrt_stub::CreateDefaultSessionOptions(
      graph_execution_options);
  tensorflow::StatusOr<std::unique_ptr<tensorflow::tfrt_stub::FallbackState>>
      fallback_state = tensorflow::tfrt_stub::FallbackState::Create(
          session_options, fdef_lib);
  if (!fallback_state.ok()) {
    return absl::InternalError(fallback_state.status().ToString());
  }

  // Register infra and standard math kernels
  tensorflow::tf_mlrt::RegisterTfMlrtKernels(*kernel_registry);
  tfrt::cpu::RegisterMlrtMathKernels(kernel_registry.get());
  tfrt::cpu::RegisterMlrtFallbackCompatKernels(kernel_registry.get());

  tensorflow::StatusOr<std::unique_ptr<tensorflow::tfrt_stub::GraphExecutor>>
      graph_executor = tensorflow::tfrt_stub::GraphExecutor::Create(
          graph_execution_options, *(*fallback_state),
          /*tpu_model_resource=*/nullptr, std::move(graph),
          std::move(kernel_registry));
  if (!graph_executor.ok()) {
    return absl::InternalError(graph_executor.status().ToString());
  }

  return absl::WrapUnique(new SynchronousGraphExecutor(
      std::move(runtime), std::move(*fallback_state),
      std::move(*graph_executor)));
}

absl::Status SynchronousGraphExecutor::Run(
    const std::string& graph_name, absl::Span<mlrt::Value> input_values,
    absl::Span<const std::string> input_names,
    absl::Span<const tensorflow::DataType> input_dtypes,
    absl::Span<const std::string> output_tensor_names,
    absl::Span<const std::string> target_tensor_names,
    absl::Span<mlrt::Value> outputs) {
  return tfrt::AbslStatusFromTfStatus(graph_executor_->RunWithSyncInterpreter(
      graph_name, input_values, input_names, input_dtypes, output_tensor_names,
      target_tensor_names, outputs));
}

}  // namespace tfrt_stub
}  // namespace tensorflow
