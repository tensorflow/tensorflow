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

#include "tensorflow/core/common_runtime/next_pluggable_device/c_plugin_op_kernel.h"

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "tensorflow/c/c_api.h"
#include "tensorflow/c/experimental/next_pluggable_device/c_api.h"
#include "tensorflow/c/kernels.h"
#include "tensorflow/c/kernels_experimental.h"
#include "tensorflow/c/tf_buffer.h"
#include "tensorflow/c/tf_buffer_internal.h"
#include "tensorflow/c/tf_datatype.h"
#include "tensorflow/c/tf_status.h"
#include "tensorflow/c/tf_status_helper.h"
#include "tensorflow/c/tf_tensor_helper.h"
#include "tensorflow/core/common_runtime/next_pluggable_device/c_plugin_variable.h"
#include "tensorflow/core/common_runtime/next_pluggable_device/plugin_coordination_service_agent.h"
#include "tensorflow/core/common_runtime/next_pluggable_device/plugin_coordination_service_agent_helper.h"
#include "tensorflow/core/common_runtime/next_pluggable_device/plugin_variable.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/resource_handle.h"
#include "tensorflow/core/framework/resource_handle.pb.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/protobuf/config.pb.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/logging.h"  // IWYU pragma: keep
#include "tsl/platform/mutex.h"

constexpr int kInvalidLineNumber = -1;

namespace tensorflow {

// ------------------  CPluginOpKernelConstruction  ----------------------------
Status CPluginOpKernelConstruction::GetBoolAttr(std::string_view attr_name,
                                                bool* value) const {
  TF_StatusPtr c_status_ptr(TF_NewStatus());
  TF_Status* status = c_status_ptr.get();
  unsigned char bool_as_char;
  TF_OpKernelConstruction_GetAttrBool(ctx_, attr_name.data(), &bool_as_char,
                                      status);
  *value = static_cast<bool>(bool_as_char);
  return StatusFromTF_Status(status);
}

Status CPluginOpKernelConstruction::GetInt32Attr(std::string_view attr_name,
                                                 int32_t* value) const {
  TF_StatusPtr c_status_ptr(TF_NewStatus());
  TF_Status* status = c_status_ptr.get();
  TF_OpKernelConstruction_GetAttrInt32(ctx_, attr_name.data(), value, status);
  return StatusFromTF_Status(status);
}

Status CPluginOpKernelConstruction::GetInt32AttrList(
    std::string_view attr_name, std::vector<int32_t>* value) const {
  TF_StatusPtr c_status_ptr(TF_NewStatus());
  TF_Status* status = c_status_ptr.get();
  int32_t list_size;
  int32_t total_size;  // total_size is undefined for int32 attribute.
  TF_OpKernelConstruction_GetAttrSize(ctx_, attr_name.data(), &list_size,
                                      &total_size, status);
  TF_RETURN_IF_ERROR(StatusFromTF_Status(status));

  value->resize(list_size);

  TF_OpKernelConstruction_GetAttrInt32List(
      ctx_, attr_name.data(), value->data(), /*max_vals=*/list_size, status);
  return StatusFromTF_Status(status);
}

Status CPluginOpKernelConstruction::GetInt64Attr(std::string_view attr_name,
                                                 int64_t* value) const {
  TF_StatusPtr c_status_ptr(TF_NewStatus());
  TF_Status* status = c_status_ptr.get();
  TF_OpKernelConstruction_GetAttrInt64(ctx_, attr_name.data(), value, status);
  return StatusFromTF_Status(status);
}

Status CPluginOpKernelConstruction::GetStringAttr(std::string_view attr_name,
                                                  std::string* value) const {
  TF_StatusPtr c_status_ptr(TF_NewStatus());
  TF_Status* status = c_status_ptr.get();
  int list_size = 0, attr_string_size = 0;  // list_size is not used.
  TF_OpKernelConstruction_GetAttrSize(ctx_, attr_name.data(), &list_size,
                                      &attr_string_size, status);
  TF_RETURN_IF_ERROR(StatusFromTF_Status(status));
  value->resize(attr_string_size);
  TF_OpKernelConstruction_GetAttrString(ctx_, attr_name.data(), value->data(),
                                        /*max_length=*/attr_string_size,
                                        status);
  return StatusFromTF_Status(status);
}

Status CPluginOpKernelConstruction::GetFunctionAttr(
    std::string_view attr_name, NameAttrList* function) const {
  TF_StatusPtr c_status_ptr(TF_NewStatus());
  TF_Status* status = c_status_ptr.get();
  TF_Buffer* serialized_function =
      TF_OpKernelConstruction_GetAttrFunction(ctx_, attr_name.data(), status);
  TF_RETURN_IF_ERROR(StatusFromTF_Status(status));
  TF_RETURN_IF_ERROR(BufferToMessage(serialized_function, function));
  TF_DeleteBuffer(serialized_function);
  return absl::OkStatus();
}

void CPluginOpKernelConstruction::CtxFailure(const Status& status) {
  CtxFailure(/*file=*/"", /*line=*/kInvalidLineNumber, status);
}

void CPluginOpKernelConstruction::CtxFailure(const char* file, int line,
                                             const Status& status) {
  TF_StatusPtr c_status_ptr(TF_NewStatus());
  tsl::Set_TF_Status_from_Status(c_status_ptr.get(), status);
  if (line != kInvalidLineNumber) {
    LOG(WARNING) << "Plugin OP_REQUIRES failed at " << file << ": " << line
                 << ": " << status;
  }
  TF_OpKernelConstruction_Failure(ctx_, c_status_ptr.get());
}

// -------------------  CPluginOpKernelContext  -------------------------------
std::string_view CPluginOpKernelContext::GetResourceMgrDefaultContainerName() {
  TF_StringView default_container_name =
      TF_GetResourceMgrDefaultContainerName(ctx_);
  return {default_container_name.data, default_container_name.len};
}

Status CPluginOpKernelContext::LookupOrCreateResource(
    std::string_view container_name, std::string_view plugin_resource_name,
    void** result_plugin_resource, void* (*create_func)(void*),
    void* create_func_args, void (*delete_func)(void*)) {
  TF_StatusPtr c_status_ptr(TF_NewStatus());
  TF_Status* status = c_status_ptr.get();
  TF_LookupOrCreatePluginResource(
      ctx_,
      /*container_name=*/container_name.data(),
      /*plugin_resource_name=*/plugin_resource_name.data(),
      /*result_plugin_resource=*/result_plugin_resource,
      /*create_func=*/create_func,
      /*create_func_args=*/create_func_args,
      /*delete_func=*/delete_func, /*status=*/status);
  return StatusFromTF_Status(status);
}

std::unique_ptr<PluginCoordinationServiceAgent>
CPluginOpKernelContext::GetPluginCoordinationServiceAgent() const {
  auto* agent = TF_GetCoordinationServiceAgent(ctx_);
  return CreatePluginCoordinationServiceAgent(agent);
}

Status CPluginOpKernelContext::CreatePluginVariable(
    int index, PluginVariable** variable) const {
  TF_StatusPtr c_status_ptr(TF_NewStatus());
  TF_VariableInfo* c_var_info =
      TF_CreateVariableInfoFromContext(ctx_, index, c_status_ptr.get());
  if (TF_GetCode(c_status_ptr.get()) != TF_OK) {
    return StatusFromTF_Status(c_status_ptr.get());
  }
  *variable = new CPluginVariable(c_var_info);
  return absl::OkStatus();
}

Status CPluginOpKernelContext::AllocateTempForPluginVariable(
    PluginVariable* variable) {
  TF_StatusPtr c_status_ptr(TF_NewStatus());
  CPluginVariable* c_plugin_variable =
      reinterpret_cast<CPluginVariable*>(variable);
  TF_AllocateTempForVariableInfo(ctx_, c_plugin_variable->var_info_,
                                 c_status_ptr.get());
  absl::Status status = StatusFromTF_Status(c_status_ptr.get());
  if (status.ok()) {
    // Invalidate the cached tensor since we allocated a new one.
    c_plugin_variable->tensor_obtained_ = false;
  }
  return status;
}

Status CPluginOpKernelContext::GetInput(int index, Tensor* tensor) const {
  TF_StatusPtr c_status_ptr(TF_NewStatus());
  TF_Tensor* c_tensor;
  TF_GetInput(ctx_, index, &c_tensor, c_status_ptr.get());
  TF_TensorPtr c_tensor_ptr(c_tensor);
  if (TF_GetCode(c_status_ptr.get()) != TF_OK) {
    return StatusFromTF_Status(c_status_ptr.get());
  }
  return TF_TensorToTensor(c_tensor, tensor);
}

Status CPluginOpKernelContext::GetInput(const char* name,
                                        const Tensor** tensor) {
  TF_StatusPtr c_status_ptr(TF_NewStatus());
  TF_Tensor* c_tensor;
  TF_GetInputByName(ctx_, name, &c_tensor, c_status_ptr.get());
  TF_TensorPtr c_tensor_ptr(c_tensor);
  Tensor tensor_tmp;
  absl::Status status = TF_TensorToTensor(c_tensor, &tensor_tmp);
  if (status.ok()) {
    tsl::mutex_lock lock(mu_);
    obtained_tensors_.push_back(std::move(tensor_tmp));
    *tensor = &(obtained_tensors_.back());
  }
  return status;
}

Status CPluginOpKernelContext::GetInputRange(std::string_view name,
                                             std::pair<int, int>* range) const {
  TF_StatusPtr c_status_ptr(TF_NewStatus());
  TF_InputRange_Args args;
  args.status = c_status_ptr.get();
  TF_InputRange(ctx_, name.data(), &args);
  TF_RETURN_IF_ERROR(StatusFromTF_Status(args.status));
  range->first = args.start;
  range->second = args.stop;
  return absl::OkStatus();
}

DataType CPluginOpKernelContext::GetInputDataType(int index) const {
  return static_cast<DataType>(TF_InputDatatype(ctx_, index));
}

std::string_view CPluginOpKernelContext::GetOpKernelRequestedInput(
    int index) const {
  TF_StringView requested_input = TF_GetOpKernelRequestedInput(ctx_, index);
  return {requested_input.data, requested_input.len};
}

std::string_view CPluginOpKernelContext::GetOpKernelName() const {
  TF_StringView op_kernel_name = TF_GetOpKernelName(ctx_);
  return {op_kernel_name.data, op_kernel_name.len};
}

std::string_view CPluginOpKernelContext::GetDeviceName() const {
  TF_StringView device_name = TF_GetDeviceName(ctx_);
  return {device_name.data, device_name.len};
}

Status CPluginOpKernelContext::GetConfigProto(
    const ConfigProto** config_proto) const {
  TF_BufferPtr serialized_config_proto_ptr(TF_NewBuffer());
  TF_StatusPtr c_status_ptr(TF_NewStatus());
  TF_GetSerializedConfigProto(ctx_, serialized_config_proto_ptr.get(),
                              c_status_ptr.get());
  TF_RETURN_IF_ERROR(StatusFromTF_Status(c_status_ptr.get()));
  ConfigProto* config_proto_ptr = new ConfigProto();
  Status status =
      BufferToMessage(serialized_config_proto_ptr.get(), config_proto_ptr);
  *config_proto = config_proto_ptr;
  return status;
}

Status CPluginOpKernelContext::GetFunctionLibraryDefinition(
    const FunctionLibraryDefinition** flib_def) const {
  TF_BufferPtr serialized_function_library_ptr(TF_NewBuffer());
  TF_StatusPtr c_status_ptr(TF_NewStatus());

  TF_GetSerializedFunctionDefLibrary(
      ctx_, serialized_function_library_ptr.get(), c_status_ptr.get());
  TF_RETURN_IF_ERROR(StatusFromTF_Status(c_status_ptr.get()));
  FunctionDefLibrary fdef_lib;
  TF_RETURN_IF_ERROR(
      BufferToMessage(serialized_function_library_ptr.get(), &fdef_lib));
  auto flib_def_ptr =
      new FunctionLibraryDefinition(OpRegistry::Global(), fdef_lib);
  *flib_def = flib_def_ptr;
  return absl::OkStatus();
}

Status CPluginOpKernelContext::GetResourceHandle(
    int index, const ResourceHandle** handle) const {
  TF_BufferPtr serialized_resource_handle_ptr(TF_NewBuffer());
  TF_StatusPtr c_status_ptr(TF_NewStatus());

  TF_GetSerializedResourceHandleProto(
      ctx_, index, serialized_resource_handle_ptr.get(), c_status_ptr.get());
  TF_RETURN_IF_ERROR(StatusFromTF_Status(c_status_ptr.get()));

  ResourceHandleProto handle_proto;
  TF_RETURN_IF_ERROR(
      BufferToMessage(serialized_resource_handle_ptr.get(), &handle_proto));
  const ResourceHandle* handle_ptr = new ResourceHandle(handle_proto);

  *handle = handle_ptr;
  return absl::OkStatus();
}

Status CPluginOpKernelContext::AllocateOutput(int index,
                                              const TensorShape& shape,
                                              Tensor** out) {
  TF_StatusPtr c_status_ptr(TF_NewStatus());
  const auto num_dims = shape.dims();
  int64_t* dim_array = new int64_t[num_dims];
  for (int i = 0; i < num_dims; ++i) {
    dim_array[i] = shape.dim_size(i);
  }
  // Note: dtype and len in TF_AllocateOutput are dummy.
  TF_TensorPtr c_tensor_ptr(
      TF_AllocateOutput(ctx_, index, /*dtype=*/TF_FLOAT, /*dims=*/dim_array,
                        /*num_dims=*/num_dims, /*len=*/0, c_status_ptr.get()));
  delete[] dim_array;
  TF_RETURN_IF_ERROR(StatusFromTF_Status(c_status_ptr.get()));
  return TF_TensorToTensor(c_tensor_ptr.get(), *out);
}

Status CPluginOpKernelContext::SetOutput(int index, const Tensor& tensor) {
  TF_StatusPtr c_status_ptr(TF_NewStatus());
  TF_TensorPtr c_tensor_ptr;
  Status status;
  c_tensor_ptr.reset(TF_TensorFromTensor(tensor, &status));
  TF_RETURN_IF_ERROR(status);
  TF_SetOutput(ctx_, index, c_tensor_ptr.get(), c_status_ptr.get());
  return StatusFromTF_Status(c_status_ptr.get());
}

void CPluginOpKernelContext::CtxFailure(const Status& status) {
  CtxFailure(/*file=*/"", /*line=*/kInvalidLineNumber, status);
}

void CPluginOpKernelContext::CtxFailure(const char* file, int line,
                                        const Status& status) {
  TF_StatusPtr c_status_ptr(TF_NewStatus());
  tsl::Set_TF_Status_from_Status(c_status_ptr.get(), status);
  if (line != kInvalidLineNumber) {
    LOG(WARNING) << "Plugin OP_REQUIRES failed at " << file << ": " << line
                 << ": " << status;
  }
  TF_OpKernelContext_Failure(ctx_, c_status_ptr.get());
}

}  // namespace tensorflow
