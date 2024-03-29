/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/tfrt/ifrt/tf_xla_host_callback.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/cleanup/cleanup.h"
#include "absl/container/fixed_array.h"
#include "absl/log/check.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "tensorflow/c/eager/abstract_tensor_handle.h"
#include "tensorflow/c/eager/immediate_execution_context.h"
#include "tensorflow/c/eager/immediate_execution_operation.h"
#include "tensorflow/compiler/tf2xla/literal_util.h"
#include "tensorflow/compiler/tf2xla/type_util.h"
#include "xla/literal.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/eager/context.h"
#include "tensorflow/core/common_runtime/eager/tensor_handle.h"
#include "tensorflow/core/framework/device.h"
#include "tensorflow/core/framework/device_factory.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/protobuf/config.pb.h"
#include "tensorflow/core/tfrt/runtime/step_id.h"
#include "tsl/platform/casts.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/refcount.h"
#include "tsl/platform/statusor.h"
#include "tsl/profiler/lib/traceme.h"

namespace tensorflow {
namespace ifrt_serving {

using RefCountHandle = ::tsl::core::RefCountPtr<tensorflow::TensorHandle>;

absl::Status TfXlaHostCallback::Call(
    tensorflow::tfrt_stub::StepId client_step_id,
    absl::Span<const std::unique_ptr<xla::LiteralBase>> operands,
    absl::Span<const std::unique_ptr<xla::MutableLiteralBase>> results) {
  tsl::profiler::TraceMe trace_me("TpuLoadProgram");

  tensorflow::ImmediateOpPtr op(ctx_->CreateOperation());
  TF_RETURN_IF_ERROR(
      op->Reset(entry_function_name_.c_str(), /*raw_device_name=*/nullptr));

  // Propagate the client step id, if any.
  if (client_step_id.valid()) {
    op->SetStepId(client_step_id.id);
  }

  // Wrap each execution with StartStep/EndStep. This ensures that per-step
  // TF resources like TensorArray are always cleaned up.
  ctx_->StartStep();
  absl::Cleanup cleanup_step = [this]() { ctx_->EndStep(); };

  // Prepare inputs.
  for (int i = 0; i < operands.size(); ++i) {
    TF_ASSIGN_OR_RETURN(const auto dtype,
                        tensorflow::EncodePrimitiveTypeAsDataType(
                            operands[i]->shape().element_type()));
    tensorflow::Tensor t;
    TF_RETURN_IF_ERROR(
        tensorflow::LiteralToHostTensor(*operands[i], dtype, &t));

    RefCountHandle handle(tensorflow::down_cast<tensorflow::TensorHandle*>(
        ctx_->CreateLocalHandleFromTFTensor(t, /*d_name=*/nullptr)));
    TF_RETURN_IF_ERROR(op->AddInput(handle.get()));
  }

  // Execute the function and block until completion.
  int num_outputs = results.size();
  absl::FixedArray<tensorflow::AbstractTensorHandle*> output_raw_handles(
      num_outputs);
  TF_RETURN_IF_ERROR(
      op->Execute(absl::MakeSpan(output_raw_handles), &num_outputs));

  std::vector<RefCountHandle> output_handles;
  output_handles.reserve(num_outputs);
  for (auto* output_raw_handle : output_raw_handles) {
    output_handles.emplace_back(
        tensorflow::down_cast<tensorflow::TensorHandle*>(output_raw_handle));
  }

  // Copy the output tensors.
  if (results.size() != num_outputs) {
    return absl::InternalError(
        absl::StrCat("TF host callback invocation expected ", results.size(),
                     " results, instead got ", num_outputs));
  }
  for (int i = 0; i < num_outputs; ++i) {
    const tensorflow::Tensor* tensor;
    TF_RETURN_IF_ERROR(output_handles[i]->Tensor(&tensor));

    xla::BorrowingLiteral literal;
    TF_RETURN_IF_ERROR(
        tensorflow::HostTensorToBorrowingLiteral(*tensor, &literal));
    TF_RETURN_IF_ERROR(results[i]->CopyFrom(literal));
  }

  return absl::OkStatus();
}

absl::StatusOr<std::unique_ptr<TfXlaHostCallback>>
TfXlaHostCallback::CreateCallback(
    const tensorflow::ConfigProto& session_config,
    absl::Span<const tensorflow::FunctionDef> functions,
    absl::string_view entry_function_name,
    std::shared_ptr<tensorflow::StaticDeviceMgr> device_mgr) {
  tensorflow::SessionOptions options;
  options.config = session_config;
  // Explicitly disable non-CPU devices to avoid triggering TPU device
  // initialization inside TF.
  options.config.add_device_filters("/device:CPU:*");

  DCHECK(device_mgr != nullptr);

  // Create a new synchronous TF Eager context. Using sync mode simplifies the
  // error semantics and host callbacks cannot use asynchronous execution anyway
  // because they have to write results to specified buffers before the call
  // returns.
  tensorflow::EagerContextPtr ctx(new tensorflow::EagerContext(
      options,
      tensorflow::ContextDevicePlacementPolicy::DEVICE_PLACEMENT_SILENT,
      /*async=*/false, device_mgr.get(),
      /*device_mgr_owned=*/false,
      /*rendezvous=*/nullptr,
      /*cluster_flr=*/nullptr,
      /*collective_executor_mgr=*/nullptr,
      /*run_eager_op_as_function=*/true));

  for (const tensorflow::FunctionDef& function : functions) {
    TF_RETURN_IF_ERROR(ctx->AddFunctionDef(function));
  }

  return absl::WrapUnique(
      new TfXlaHostCallback(entry_function_name, device_mgr, std::move(ctx)));
}

absl::StatusOr<std::shared_ptr<tensorflow::StaticDeviceMgr>>
CreateTfStaticDeviceMgr() {
  tensorflow::SessionOptions options;
  // Explicitly disable non-CPU devices to avoid triggering TPU device
  // initialization inside TF.
  options.config.add_device_filters("/device:CPU:*");

  // Share the same TF devices across all host callbacks in a single XLA
  // computation. This makes it possible to share states (e.g., TF resources)
  // across host callbacks in a single XLA computation.
  std::vector<std::unique_ptr<tensorflow::Device>> devices;
  TF_RETURN_IF_ERROR(tensorflow::DeviceFactory::AddCpuDevices(
      options, "/job:localhost/replica:0/task:0", &devices));
  return std::make_shared<tensorflow::StaticDeviceMgr>(std::move(devices));
}

}  // namespace ifrt_serving
}  // namespace tensorflow
