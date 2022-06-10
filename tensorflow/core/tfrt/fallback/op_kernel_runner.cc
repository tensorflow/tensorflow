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
#include "tensorflow/core/tfrt/fallback/op_kernel_runner.h"

#include "tensorflow/core/platform/errors.h"

namespace tensorflow {
namespace tfrt_stub {
namespace {

Status CheckOpDefCompatibility(const tensorflow::OpDef& op_def) {
  auto check_arg_def = [&](const auto& arg_def) {
    if (arg_def.is_ref())
      return tensorflow::errors::Internal(
          "TFRT kernel fallback error: Unsupported ref args in ",
          op_def.name());
    return OkStatus();
  };

  for (const auto& arg_def : op_def.input_arg())
    TF_RETURN_IF_ERROR(check_arg_def(arg_def));
  for (const auto& arg_def : op_def.output_arg())
    TF_RETURN_IF_ERROR(check_arg_def(arg_def));

  return OkStatus();
}

// Create a tensorflow::NodeDef from the tensorflow::OpDef and the attributes.
StatusOr<tensorflow::NodeDef> BuildNodeDef(
    const tensorflow::OpDef& op_def, int num_args,
    const std::function<Status(tensorflow::AttrValueMap*)>& attr_builder) {
  tensorflow::NodeDef node_def;
  node_def.set_name(op_def.name());
  node_def.set_op(op_def.name());
  for (int i = 0; i < num_args; ++i) {
    node_def.add_input("dummy_input");
  }

  auto* attr_value_map = node_def.mutable_attr();
  TF_RETURN_IF_ERROR(attr_builder(attr_value_map));

  // For any attr-value pairs that exist in the op def (from op registry)
  // but not in `attr_value_map`, fill them into `attr_value_map`, so that we
  // can run a TFE_Op without having to specify all the default attr values
  // (e.g. for matmul, the `transpose_a` attr defaults to false).
  for (const auto& attr_def : op_def.attr()) {
    if (attr_def.has_default_value()) {
      // Insertion will fail if this attribute already has a value.
      attr_value_map->insert({attr_def.name(), attr_def.default_value()});
    }
  }
  return node_def;
}

tensorflow::Status CreateOpKernel(
    tensorflow::FunctionLibraryRuntime* flr, tensorflow::NodeDef ndef,
    std::unique_ptr<tensorflow::OpKernel>* result) {
  std::shared_ptr<const tensorflow::NodeProperties> props;
  TF_RETURN_IF_ERROR(tensorflow::NodeProperties::CreateFromNodeDef(
      ndef, flr->GetFunctionLibraryDefinition(), &props));
  tensorflow::OpKernel* k = nullptr;
  TF_RETURN_IF_ERROR(flr->CreateKernel(props, &k));
  result->reset(k);
  return OkStatus();
}

}  // namespace

StatusOr<OpKernelRunner> OpKernelRunner::Create(
    absl::string_view op_name, absl::string_view device_name, int num_args,
    const std::function<Status(tensorflow::AttrValueMap*)>& attr_builder,
    const tensorflow::DeviceMgr& device_manager,
    const tensorflow::ProcessFunctionLibraryRuntime&
        process_function_library_runtime) {
  tensorflow::Device* device = nullptr;
  Status s = device_manager.LookupDevice(device_name, &device);

  // Fall back to host device if it fails to find the specified device.
  if (!s.ok()) {
    LOG(WARNING) << "Failed to find device " << device_name
                 << " when creating OpKernel: " << op_name << ". Error: " << s;
    LOG(WARNING) << "Fallback to host device instead";
    device = device_manager.HostCPU();
  }

  return Create(op_name, num_args, attr_builder,
                process_function_library_runtime, device);
}

StatusOr<OpKernelRunner> OpKernelRunner::Create(
    absl::string_view op_name, int num_args,
    const std::function<Status(tensorflow::AttrValueMap*)>& attr_builder,
    const tensorflow::ProcessFunctionLibraryRuntime&
        process_function_library_runtime,
    tensorflow::Device* device) {
  const OpDef* op_def = nullptr;
  TF_RETURN_IF_ERROR(tensorflow::OpRegistry::Global()->LookUpOpDef(
      std::string(op_name), &op_def));
  TF_RETURN_IF_ERROR(CheckOpDefCompatibility(*op_def));
  VLOG(1) << "KernelFallbackExecuteCompat creating op from OpDef: "
          << op_def->DebugString();

  TF_ASSIGN_OR_RETURN(auto node_def,
                      BuildNodeDef(*op_def, num_args, attr_builder));

  VLOG(1) << "KernelFallbackExecuteCompat created NodeDef: "
          << node_def.DebugString();

  tensorflow::FunctionLibraryRuntime* function_library_runtime = nullptr;

  function_library_runtime =
      process_function_library_runtime.GetFLR(device->name());

  std::unique_ptr<OpKernel> op_kernel;
  TF_RETURN_IF_ERROR(CreateOpKernel(function_library_runtime,
                                    std::move(node_def), &op_kernel));
  return OpKernelRunner(device, function_library_runtime, std::move(op_kernel));
}

OpKernelRunner::OpKernelRunner(
    tensorflow::Device* device,
    tensorflow::FunctionLibraryRuntime* function_library_runtime,
    std::unique_ptr<tensorflow::OpKernel> op_kernel)
    : device_(device),
      function_library_runtime_(function_library_runtime),
      resource_manager_(device->resource_manager()),
      op_kernel_(std::move(op_kernel)),
      is_async_(op_kernel_->AsAsync() != nullptr) {
  DCHECK(device_);
  DCHECK(function_library_runtime_);

  const auto& input_memory_types = op_kernel_->input_memory_types();
  input_alloc_attrs_.resize(op_kernel_->num_inputs());
  for (size_t i = 0, e = op_kernel_->num_inputs(); i < e; ++i) {
    input_alloc_attrs_[i].set_on_host(input_memory_types[i] ==
                                      tensorflow::HOST_MEMORY);
  }
  const auto& output_memory_types = op_kernel_->output_memory_types();
  output_alloc_attrs_.resize(op_kernel_->num_outputs());
  for (size_t i = 0, e = output_alloc_attrs_.size(); i < e; ++i) {
    output_alloc_attrs_[i].set_on_host(output_memory_types[i] ==
                                       tensorflow::HOST_MEMORY);
  }
}

void OpKernelRunner::Run(OpKernelContext* context) const {
  DVLOG(1) << "KernelFallbackExecuteCompat Running Op: "
           << op_kernel_->def().DebugString()
           << ", on Device: " << context->device()->name();

  static_cast<tensorflow::Device*>(context->device())
      ->Compute(op_kernel_.get(), context);
}

void OpKernelRunner::RunAsync(OpKernelContext* context,
                              AsyncOpKernel::DoneCallback done_callback) const {
  DVLOG(1) << "KernelFallbackExecuteCompat Running Async Op: "
           << op_kernel_->def().DebugString()
           << ", on Device: " << context->device()->name();

  AsyncOpKernel* async = op_kernel_->AsAsync();
  DCHECK(async);

  static_cast<tensorflow::Device*>(context->device())
      ->ComputeAsync(async, context, std::move(done_callback));
}

}  // namespace tfrt_stub
}  // namespace tensorflow
