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

#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_NEXT_PLUGGABLE_DEVICE_DIRECT_PLUGIN_OP_KERNEL_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_NEXT_PLUGGABLE_DEVICE_DIRECT_PLUGIN_OP_KERNEL_H_

#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "tensorflow/core/common_runtime/next_pluggable_device/plugin_coordination_service_agent_helper.h"
#include "tensorflow/core/common_runtime/next_pluggable_device/plugin_op_kernel.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"

namespace tensorflow {

class DirectPluginOpKernelConstruction : public PluginOpKernelConstruction {
 public:
  explicit DirectPluginOpKernelConstruction(void* ctx)
      : ctx_(reinterpret_cast<OpKernelConstruction*>(ctx)) {}

  Status GetBoolAttr(std::string_view attr_name, bool* value) const override;
  Status GetInt32Attr(std::string_view attr_name, int* value) const override;
  Status GetInt32AttrList(std::string_view attr_name,
                          std::vector<int32_t>* value) const override;
  Status GetInt64Attr(std::string_view attr_name,
                      int64_t* value) const override;
  Status GetStringAttr(std::string_view attr_name,
                       std::string* value) const override;
  Status GetFunctionAttr(std::string_view attr_name,
                         NameAttrList* function) const override;

  void CtxFailure(const Status& status) override { ctx_->CtxFailure(status); }

  void CtxFailure(const char* file, int line, const Status& status) override {
    ctx_->CtxFailure(file, line, status);
  }

  void* GetContext() const override { return ctx_; }

 private:
  OpKernelConstruction* ctx_;  // not owned.
};

class DirectPluginOpKernelContext : public PluginOpKernelContext {
 public:
  explicit DirectPluginOpKernelContext(OpKernelContext* ctx) : ctx_(ctx) {}

  std::string_view GetResourceMgrDefaultContainerName() override;

  Status LookupOrCreateResource(std::string_view container_name,
                                std::string_view plugin_resource_name,
                                void** result_plugin_resource,
                                void* (*create_func)(void*),
                                void* create_func_args,
                                void (*delete_func)(void*)) override;

  std::unique_ptr<PluginCoordinationServiceAgent>
  GetPluginCoordinationServiceAgent() const override {
    return CreatePluginCoordinationServiceAgent(
        ctx_->coordination_service_agent());
  }

  Status CreatePluginVariable(int index,
                              PluginVariable** variable) const override;

  Status AllocateTempForPluginVariable(PluginVariable* variable) override;

  int NumInputs() const override { return ctx_->num_inputs(); }

  absl::Status GetInput(int index, const Tensor** tensor) const override;

  absl::Status GetInput(const char* name, const Tensor** tensor) const override;

  Status GetInputRange(std::string_view name,
                       std::pair<int, int>* range) const override;

  DataType GetInputDataType(int index) const override {
    return ctx_->input_dtype(index);
  }

  std::string_view GetOpKernelRequestedInput(int index) const override {
    return ctx_->op_kernel().requested_input(index);
  }

  std::string_view GetOpKernelName() const override {
    return ctx_->op_kernel().name();
  }

  uint64_t GetFrameId() const override { return ctx_->frame_iter().frame_id; }

  int64_t GetIterId() const override { return ctx_->frame_iter().iter_id; }

  int64_t GetStepId() const override { return ctx_->step_id(); }

  int GetDeviceId() const override;

  std::string_view GetDeviceName() const override;

  std::string GetSessionName() const override {
    return ctx_->session_metadata() ? ctx_->session_metadata()->name() : "";
  }

  Status GetConfigProto(const ConfigProto** config_proto) const override {
    *config_proto = ctx_->function_library()->config_proto();
    return absl::OkStatus();
  }

  void MaybeDeleteConfigProto(const ConfigProto* config_proto) const override {
    // We don't need to specifically delete ConfigProto since it is obtained
    // from FunctionLibraryRuntime in `ctx_`.
  }

  Status GetFunctionLibraryDefinition(
      const FunctionLibraryDefinition** flib_def) const override {
    *flib_def = ctx_->function_library()->GetFunctionLibraryDefinition();
    return absl::OkStatus();
  }

  void MaybeDeleteFunctionLibraryDefinition(
      const FunctionLibraryDefinition* flib_def) const override {
    // We don't need to specifically delete FunctionLibraryDefinition since it
    // is obtained from FunctionLibraryRuntime in `ctx_`.
  }

  Status GetResourceHandle(int index,
                           const ResourceHandle** handle) const override {
    *handle = &HandleFromInput(ctx_, index);
    return absl::OkStatus();
  }

  void MaybeDeleteResourceHandle(const ResourceHandle* handle) const override {
    // We don't need to specifically delete ResourceHandle since it is obtained
    // from `ctx_`.
  }

  int GetGraphDefVersion() const override {
    return ctx_->function_library()->graph_def_version();
  }

  Status AllocateOutput(int index, const TensorShape& shape,
                        Tensor** out) override {
    return ctx_->allocate_output(index, shape, out);
  }

  Status SetOutput(int index, const Tensor& tensor) override {
    ctx_->set_output(index, tensor);
    return absl::OkStatus();
  }

  void CtxFailure(const Status& status) override { ctx_->CtxFailure(status); }

  void CtxFailure(const char* file, int line, const Status& status) override {
    LOG(WARNING) << "Plugin OP_REQUIRES failed at " << file << ": " << line
                 << ": " << status;
    ctx_->CtxFailure(file, line, status);
  }

  void* GetContext() const override { return ctx_; }

 private:
  OpKernelContext* ctx_;  // not owned.
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_NEXT_PLUGGABLE_DEVICE_DIRECT_PLUGIN_OP_KERNEL_H_
