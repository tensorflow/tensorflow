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

#include "tensorflow/core/tfrt/ifrt/tf_host_callback.h"

#include <cstddef>
#include <cstring>
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
#include "tensorflow/compiler/mlir/tfrt/transforms/ifrt/ifrt_types.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/eager/context.h"
#include "tensorflow/core/common_runtime/eager/tensor_handle.h"
#include "tensorflow/core/framework/device.h"
#include "tensorflow/core/framework/device_factory.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/protobuf/config.pb.h"
#include "tsl/platform/casts.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/refcount.h"
#include "tsl/profiler/lib/traceme.h"

namespace tensorflow {
namespace ifrt_serving {
namespace {
using RefCountHandle = ::tsl::core::RefCountPtr<tensorflow::TensorHandle>;

size_t GetSizeInBytes(const tensorflow::Tensor& tensor) {
  return tensor.shape().num_elements() * DataTypeSize(tensor.dtype());
}

// Returns a tensor of the specified type and shape. The tensor's data is filled
// from `src`.
tensorflow::Tensor GetTensor(const DtypeAndShape& dtype_and_shape, void* src) {
  DCHECK(DataTypeCanUseMemcpy(dtype_and_shape.dtype));
  tensorflow::Tensor t(dtype_and_shape.dtype, dtype_and_shape.shape);
  std::memcpy(t.data(), src, GetSizeInBytes(t));
  return t;
}

// Fills the buffer pointed by `dst` by data from the given tensor.
void CopyToBuffer(void* dst, const tensorflow::Tensor& tensor) {
  DCHECK(DataTypeCanUseMemcpy(tensor.dtype()));
  std::memcpy(dst, tensor.data(), GetSizeInBytes(tensor));
}
}  // namespace

absl::Status TfHostCallback::Call(void** inputs, void** outputs) {
  tsl::profiler::TraceMe trace_me("TfHostCallback::Call");

  tensorflow::ImmediateOpPtr op(ctx_->CreateOperation());
  TF_RETURN_IF_ERROR(
      op->Reset(entry_function_name_.c_str(), /*raw_device_name=*/nullptr));

  // Wrap each execution with StartStep/EndStep. This ensures that per-step
  // TF resources like TensorArray are always cleaned up.
  ctx_->StartStep();
  absl::Cleanup cleanup_step = [this]() { ctx_->EndStep(); };

  // Prepare inputs.
  for (int i = 0; i < operand_type_and_shapes_.size(); ++i) {
    tensorflow::Tensor t = GetTensor(operand_type_and_shapes_[i], inputs[i]);
    RefCountHandle handle(tensorflow::down_cast<tensorflow::TensorHandle*>(
        ctx_->CreateLocalHandleFromTFTensor(t, /*d_name=*/nullptr)));
    TF_RETURN_IF_ERROR(op->AddInput(handle.get()));
  }

  // Execute the function and block until completion.
  int num_outputs = result_type_and_shapes_.size();
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
  if (result_type_and_shapes_.size() != num_outputs) {
    return absl::InternalError(absl::StrCat(
        "TF host callback invocation expected ", result_type_and_shapes_.size(),
        " results, instead got ", num_outputs));
  }
  for (int i = 0; i < num_outputs; ++i) {
    const tensorflow::Tensor* tensor;
    TF_RETURN_IF_ERROR(output_handles[i]->Tensor(&tensor));
    CopyToBuffer(outputs[i], *tensor);
  }

  return absl::OkStatus();
}

absl::StatusOr<std::unique_ptr<TfHostCallback>> TfHostCallback::Create(
    absl::Span<const tensorflow::FunctionDef> functions,
    absl::string_view entry_function_name,
    absl::Span<const DtypeAndShape> operand_type_and_shapes,
    absl::Span<const DtypeAndShape> result_type_and_shapes,
    tensorflow::DeviceMgr* device_mgr) {
  tensorflow::SessionOptions options;
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
      /*async=*/false, device_mgr,
      /*device_mgr_owned=*/false,
      /*rendezvous=*/nullptr,
      /*cluster_flr=*/nullptr,
      /*collective_executor_mgr=*/nullptr,
      /*run_eager_op_as_function=*/true));

  for (const tensorflow::FunctionDef& function : functions) {
    TF_RETURN_IF_ERROR(ctx->AddFunctionDef(function));
  }

  return absl::WrapUnique(
      new TfHostCallback(entry_function_name, operand_type_and_shapes,
                         result_type_and_shapes, std::move(ctx)));
}

absl::StatusOr<std::unique_ptr<tensorflow::StaticDeviceMgr>>
CreateTfStaticDeviceMgr() {
  // Share the same TF devices across all host callbacks in a single
  // computation. This makes it possible to share states (e.g., TF resources)
  // across host callbacks in a single computation.
  std::vector<std::unique_ptr<tensorflow::Device>> devices;
  TF_RETURN_IF_ERROR(tensorflow::DeviceFactory::AddCpuDevices(
      tensorflow::SessionOptions(), "/job:localhost/replica:0/task:0",
      &devices));
  return std::make_unique<tensorflow::StaticDeviceMgr>(std::move(devices));
}

}  // namespace ifrt_serving
}  // namespace tensorflow
