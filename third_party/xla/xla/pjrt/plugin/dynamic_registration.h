/* Copyright 2025 The OpenXLA Authors.

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

#ifndef XLA_PJRT_PLUGIN_DYNAMIC_REGISTRATION_H_
#define XLA_PJRT_PLUGIN_DYNAMIC_REGISTRATION_H_

#include "absl/strings/string_view.h"
#include "xla/pjrt/pjrt_api.h"  // IWYU pragma: keep

bool RegisterDynamicPjrtPlugin(absl::string_view plugin_name,
                               absl::string_view library_env_name);

// Registers a dynamic PJRT plugin.
//
// The plugin is loaded from the library path specified by the environment
// variable `library_env_name`.
//
// Example:
//
//   #include
//   "third_party/tensorflow/compiler/xla/pjrt/plugin/dynamic_registration.h"
//
//   REGISTER_DYNAMIC_PJRT_PLUGIN("my_plugin", "MY_PJRT_PLUGIN_LIBRARY_PATH");
//   // this will register a plugin named "my_plugin" that is loaded from the
//   // path in the environment variable "MY_PJRT_PLUGIN_LIBRARY_PATH".
#define REGISTER_DYNAMIC_PJRT_PLUGIN(plugin_name, library_env_name)      \
  [[maybe_unused]] static bool already_registered_##plugin_name =        \
      [](auto plugin_name) {                                             \
        return RegisterDynamicPjrtPlugin(plugin_name, library_env_name); \
      }(plugin_name);

#endif  // XLA_PJRT_PLUGIN_DYNAMIC_REGISTRATION_H_
