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

#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_NEXT_PLUGGABLE_DEVICE_C_PLUGIN_OP_KERNEL_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_NEXT_PLUGGABLE_DEVICE_C_PLUGIN_OP_KERNEL_H_

#include <memory>
#include <string>
#include <string_view>
#include <utility>

#include "tensorflow/c/kernels.h"
#include "tensorflow/compiler/xla/pjrt/pjrt_client.h"
#include "tensorflow/core/common_runtime/next_pluggable_device/plugin_coordination_service_agent.h"
#include "tensorflow/core/common_runtime/next_pluggable_device/plugin_op_kernel.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/protobuf/config.pb.h"

namespace tensorflow {

class CPluginOpKernelConstruction : public PluginOpKernelConstruction {
 public:
  explicit CPluginOpKernelConstruction(void* ctx)
      : ctx_(reinterpret_cast<TF_OpKernelConstruction*>(ctx)) {}

  Status GetBoolAttr(std::string_view attr_name, bool* value) const override;
  Status GetInt32Attr(std::string_view attr_name, int* value) const override;
  Status GetInt64Attr(std::string_view attr_name,
                      int64_t* value) const override;
  Status GetStringAttr(std::string_view attr_name,
                       std::string* value) const override;
  Status GetFunctionAttr(std::string_view attr_name,
                         NameAttrList* function) const override;

  void CtxFailure(const Status& status) override;
  void CtxFailure(const char* file, int line, const Status& status) override;

  void* GetContext() const override { return ctx_; }

 private:
  TF_OpKernelConstruction* ctx_;  // not owned.
};

class CPluginOpKernelContext : public PluginOpKernelContext {
 public:
  explicit CPluginOpKernelContext(void* ctx)
      : ctx_(reinterpret_cast<TF_OpKernelContext*>(ctx)) {}

  std::string_view GetResourceMgrDefaultContainerName() override;

  Status LookupOrCreateResource(std::string_view container_name,
                                std::string_view plugin_resource_name,
                                void** result_plugin_resource,
                                void* (*create_func)(void*),
                                void* create_func_args,
                                void (*delete_func)(void*)) override;

  PluginCoordinationServiceAgent* GetPluginCoordinationServiceAgent()
      const override;

  int NumInputs() const override { return TF_NumInputs(ctx_); }

  Status GetInput(int index, Tensor* tensor) const override;

  Status GetInputRange(std::string_view name,
                       std::pair<int, int>* range) const override;

  std::string_view GetOpKernelName() const override;

  uint64_t GetFrameId() const override { return TF_GetFrameId(ctx_); }

  int64_t GetIterId() const override { return TF_GetIterId(ctx_); }

  std::string GetSessionName() const override {
    // TODO(haoyuzhang): Implement with ctx_->session_metadata() if needed.
    return "";
  }

  Status GetConfigProto(const ConfigProto** config_proto) const override;

  // Note: this function is only meant to clear up `config_proto` created by the
  // above `COpKernelContextWrapper::GetConfigProto()`.
  void MaybeDeleteConfigProto(const ConfigProto* config_proto) override {
    delete config_proto;
  }

  Status GetFunctionLibraryDefinition(
      const FunctionLibraryDefinition** flib_def) const override;

  // Note: this function is only meant to clear up `flib_def` created by the
  // above `COpKernelContextWrapper::GetFunctionLibraryDefinition()`.
  void MaybeDeleteFunctionLibraryDefinition(
      const FunctionLibraryDefinition* flib_def) const override {
    delete flib_def;
  }

  int GetGraphDefVersion() const override {
    return TF_GetGraphDefVersion(ctx_);
  }

  Status AllocateOutput(int index, const TensorShape& shape,
                        Tensor** out) override;

  Status SetOutput(int index, const Tensor& tensor) override;

  void CtxFailure(const Status& status) override;
  void CtxFailure(const char* file, int line, const Status& status) override;

  void* GetContext() const override { return ctx_; }

 private:
  TF_OpKernelContext* ctx_;  // not owned.
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_NEXT_PLUGGABLE_DEVICE_C_PLUGIN_OP_KERNEL_H_
