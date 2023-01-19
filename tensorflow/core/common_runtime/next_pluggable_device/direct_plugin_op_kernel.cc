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

#include "tensorflow/core/common_runtime/next_pluggable_device/direct_plugin_op_kernel.h"

#include <memory>
#include <string>
#include <string_view>
#include <utility>

#include "tensorflow/compiler/xla/pjrt/pjrt_client.h"
#include "tensorflow/core/common_runtime/next_pluggable_device/plugin_resource.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/tfrt/common/pjrt_util.h"

namespace tensorflow {

Status DirectPluginOpKernelConstruction::GetBoolAttr(std::string_view attr_name,
                                                     bool* value) const {
  return ctx_->GetAttr(attr_name, value);
}

Status DirectPluginOpKernelConstruction::GetInt32Attr(
    std::string_view attr_name, int* value) const {
  return ctx_->GetAttr(attr_name, value);
}

Status DirectPluginOpKernelConstruction::GetInt64Attr(
    std::string_view attr_name, int64_t* value) const {
  return ctx_->GetAttr(attr_name, value);
}

Status DirectPluginOpKernelConstruction::GetStringAttr(
    std::string_view attr_name, std::string* value) const {
  return ctx_->GetAttr(attr_name, value);
}

Status DirectPluginOpKernelConstruction::GetFunctionAttr(
    std::string_view attr_name, NameAttrList* function) const {
  return ctx_->GetAttr(attr_name, function);
}

std::string_view
DirectPluginOpKernelContext::GetResourceMgrDefaultContainerName() {
  CHECK(ctx_->resource_manager() != nullptr);  // Crash OK.
  return ctx_->resource_manager()->default_container();
}

Status DirectPluginOpKernelContext::LookupOrCreateResource(
    std::string_view container_name, std::string_view plugin_resource_name,
    void** result_plugin_resource, void* (*create_func)(void*),
    void* create_func_args, void (*delete_func)(void*)) {
  auto* resource_mgr = ctx_->resource_manager();
  tensorflow::core::RefCountPtr<tensorflow::PluginResource>
      tf_plugin_resource_ptr;
  tensorflow::PluginResource* tf_plugin_resource = nullptr;

  TF_RETURN_IF_ERROR(resource_mgr->LookupOrCreate<tensorflow::PluginResource>(
      std::string(container_name), std::string(plugin_resource_name),
      &tf_plugin_resource,
      [plugin_resource_name, create_func, create_func_args,
       delete_func](tensorflow::PluginResource** new_resource) {
        void* opaque_plugin_resource = create_func(create_func_args);
        *new_resource = new tensorflow::PluginResource(
            opaque_plugin_resource, plugin_resource_name, delete_func);
        return tensorflow::OkStatus();
      }));
  tf_plugin_resource_ptr.reset(tf_plugin_resource);
  *result_plugin_resource = tf_plugin_resource_ptr->GetOpaquePluginResource();
  return OkStatus();
}

Status DirectPluginOpKernelContext::GetInput(int index, Tensor* tensor) const {
  *tensor = ctx_->input(index);
  return OkStatus();
}

Status DirectPluginOpKernelContext::GetInput(const char* name,
                                             const Tensor** tensor) {
  return ctx_->input(name, tensor);
}

Status DirectPluginOpKernelContext::GetInputRange(
    std::string_view name, std::pair<int, int>* range) const {
  return ctx_->op_kernel().InputRange(name, &range->first, &range->second);
}

int DirectPluginOpKernelContext::GetDeviceId() const {
  const auto* device = ctx_->device();
  CHECK(device->parsed_name().has_id);  // Crash OK.
  return device->parsed_name().id;
}

}  // namespace tensorflow
