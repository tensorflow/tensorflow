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

#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_NEXT_PLUGGABLE_DEVICE_PLUGIN_RESOURCE_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_NEXT_PLUGGABLE_DEVICE_PLUGIN_RESOURCE_H_

#include <string>
#include <string_view>

#include "tensorflow/core/framework/resource_base.h"

namespace tensorflow {

// A wrapper class for plugin to create resources to the ResourceMgr managed by
// TensorFlow. The main motivation is to make resources in plugin have the same
// lifetime as TensorFlow ResourceMgr.
//
// Usage:
// Plugin uses a TensorFlow C API `TF_CreatePluginResource()`,
// to register the `PluginResource` to the ResourceMgr managed by TensorFlow.
// `PluginResource` holds a opaque pointer and a deleter function. The deleter
// will be called at `PluginResource`'s destruction.
class PluginResource : public ResourceBase {
 public:
  PluginResource(void* plugin_resource, std::string_view plugin_resource_name,
                 void (*delete_func)(void* plugin_resource))
      : resource_(plugin_resource),
        resource_name_(plugin_resource_name),
        delete_func_(delete_func) {}
  ~PluginResource() override;

  void* GetOpaquePluginResource() { return resource_; }

  std::string DebugString() const override { return resource_name_; }

 private:
  void* resource_;
  std::string resource_name_;
  void (*delete_func_)(void* plugin_resource);
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_NEXT_PLUGGABLE_DEVICE_PLUGIN_RESOURCE_H_
