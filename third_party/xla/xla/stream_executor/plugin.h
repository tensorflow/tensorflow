/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#ifndef XLA_STREAM_EXECUTOR_PLUGIN_H_
#define XLA_STREAM_EXECUTOR_PLUGIN_H_

namespace stream_executor {

// A plugin ID is a unique identifier for each registered plugin type.
typedef void* PluginId;

// Helper macro to define a plugin ID. To be used only inside plugin
// implementation files. Works by "reserving" an address/value (guaranteed to be
// unique) inside a process space.
#define PLUGIN_REGISTRY_DEFINE_PLUGIN_ID(ID_VAR_NAME) \
  namespace {                                         \
  int plugin_id_value;                                \
  }                                                   \
  const PluginId ID_VAR_NAME = &plugin_id_value;

// kNullPlugin denotes an invalid plugin identifier.
extern const PluginId kNullPlugin;

// Enumeration to list the supported types of plugins / support libraries.
enum class PluginKind {
  kInvalid,
  kBlas,
  kDnn,
  kFft,
};

}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_PLUGIN_H_
