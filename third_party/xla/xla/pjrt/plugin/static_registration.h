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

#ifndef XLA_PJRT_PLUGIN_STATIC_REGISTRATION_H_
#define XLA_PJRT_PLUGIN_STATIC_REGISTRATION_H_

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/pjrt/pjrt_api.h"  // IWYU pragma: keep

absl::Status RegisterStaticPjrtPlugin(absl::string_view plugin_name,
                                      const PJRT_Api* plugin_api);

// Registers a static PJRT plugin.
//
// Example:
//
//   #include
//   "third_party/tensorflow/compiler/xla/pjrt/plugin/static_registration.h"
//
//   REGISTER_PJRT_PLUGIN("my_plugin", GetMyPluginPjrtApi);
//   // this will register a plugin named "my_plugin" that is loaded from the
//   // static function GetMyPluginPjrtApi (which returns a PJRT_Api*).
#define REGISTER_PJRT_PLUGIN(plugin_name, get_plugin_fn)          \
  [[maybe_unused]] static bool already_registered_##plugin_name = \
      [](auto plugin_name, const PJRT_Api* plugin_api) -> bool {  \
    QCHECK_OK(RegisterStaticPjrtPlugin(plugin_name, plugin_api)); \
    return true;                                                  \
  }(plugin_name, get_plugin_fn);

#endif  // XLA_PJRT_PLUGIN_STATIC_REGISTRATION_H_
