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
#include "tensorflow/core/runtime_fallback/kernel/op_kernel_runner.h"

#include "tensorflow/core/platform/errors.h"

namespace tensorflow {
namespace tfd {
namespace {

llvm::Error CheckOpDefCompatibility(const tensorflow::OpDef& op_def) {
  auto check_arg_def = [&](const auto& arg_def) -> llvm::Error {
    if (arg_def.is_ref())
      return tfrt::MakeStringError(
          "TFRT kernel fallback error: Unsupported ref args in ",
          op_def.name());
    return llvm::Error::success();
  };

  for (const auto& arg_def : op_def.input_arg())
    if (auto error = check_arg_def(arg_def)) return error;
  for (const auto& arg_def : op_def.output_arg())
    if (auto error = check_arg_def(arg_def)) return error;

  return llvm::Error::success();
}

// Create a tensorflow::NodeDef from the tensorflow::OpDef and the attributes.
tfrt::StatusOr<tensorflow::NodeDef> BuildNodeDef(
    const tensorflow::OpDef& op_def, int num_args,
    const std::function<llvm::Error(tensorflow::AttrValueMap*)>& attr_builder) {
  tensorflow::NodeDef node_def;
  node_def.set_name(op_def.name());
  node_def.set_op(op_def.name());
  for (int i = 0; i < num_args; ++i) {
    node_def.add_input("dummy_input");
  }

  auto* attr_value_map = node_def.mutable_attr();
  if (auto error = attr_builder(attr_value_map)) {
    return tensorflow::errors::InvalidArgument(tfrt::StrCat(error));
  }

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
  return Status::OK();
}

}  // namespace

tfrt::StatusOr<OpKernelRunner> OpKernelRunner::Create(
    absl::string_view op_name, absl::string_view device_name, int num_args,
    const std::function<llvm::Error(tensorflow::AttrValueMap*)>& attr_builder,
    const KernelFallbackCompatRequestState& fallback_request_state) {
  const OpDef* op_def = nullptr;
  TF_RETURN_IF_ERROR(tensorflow::OpDefForOp(std::string(op_name), &op_def));
  if (auto error = CheckOpDefCompatibility(*op_def)) {
    return tensorflow::errors::Internal(tfrt::StrCat(error));
  }
  VLOG(1) << "KernelFallbackExecuteCompat creating op from OpDef: "
          << op_def->DebugString();

  TF_ASSIGN_OR_RETURN(auto node_def,
                      BuildNodeDef(*op_def, num_args, attr_builder));

  VLOG(1) << "KernelFallbackExecuteCompat created NodeDef: "
          << node_def.DebugString();

  tensorflow::Device* device = nullptr;
  tensorflow::FunctionLibraryRuntime* function_library_runtime = nullptr;

  // TODO(b/176451036): For device names that are not in tensorflow format, we
  // handle it specially. This is a workaround as the compiler lowering does not
  // use tensorflow format in some cases. Ideally, we should always use device
  // name in tensorflow format in fallback code.
  Status s = fallback_request_state.device_manager().LookupDevice(device_name,
                                                                  &device);

  // Fall back to host device if it fails to find the specified device.
  if (!s.ok()) {
    LOG(ERROR) << "Failed to find device " << device_name
               << " when creating OpKernel: " << op_name << ". Error: " << s;
    LOG(ERROR) << "Fallback to host device instead";
    device = fallback_request_state.device_manager().HostCPU();
  }

  function_library_runtime =
      fallback_request_state.process_function_library_runtime().GetFLR(
          device->name());

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

void OpKernelRunner::RunAsync(OpKernelContext* context,
                              AsyncOpKernel::DoneCallback done_callback) const {
  DVLOG(1) << "KernelFallbackExecuteCompat Running Async Op: "
           << op_kernel_->def().DebugString()
           << ", on Device: " << device_->name();

  AsyncOpKernel* async = op_kernel_->AsAsync();
  DCHECK(async);

  async->ComputeAsync(context, std::move(done_callback));
}

OpKernelRunnerCache::OpKernelRunnerCache() {}

tfrt::StatusOr<OpKernelRunner*> OpKernelRunnerCache::GetOrCreate(
    tfrt::Location loc, absl::string_view op_name,
    absl::string_view device_name, int num_args,
    const std::function<llvm::Error(tensorflow::AttrValueMap*)>& attr_builder,
    const KernelFallbackCompatRequestState& fallback_request_state) {
  OpLocationKey key(loc);
  {
    tf_shared_lock lock(mu_);
    auto it = map_.find(key);
    if (it != map_.end()) {
      DCHECK_EQ(it->second->op_kernel()->name(), op_name);
      return it->second.get();
    }
  }

  mutex_lock lock(mu_);

  auto it = map_.find(key);
  if (it != map_.end()) {
    DCHECK_EQ(it->second->op_kernel()->name(), op_name);
    return it->second.get();
  }

  VLOG(1) << "KernelFallbackExecuteCompat creating op " << op_name
          << " at location " << loc.data << " on device " << device_name;

  TF_ASSIGN_OR_RETURN(auto runner, OpKernelRunner::Create(
                                       op_name, device_name, num_args,
                                       attr_builder, fallback_request_state));

  auto runner_uptr = std::make_unique<OpKernelRunner>(std::move(runner));

  auto* runner_ptr = runner_uptr.get();
  auto r = map_.emplace(key, std::move(runner_uptr)).second;
  DCHECK(r);

  return runner_ptr;
}

}  // namespace tfd
}  // namespace tensorflow
