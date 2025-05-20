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

#define REGISTER_PJRT_PLUGIN(plugin_name, get_plugin_api)        \
  static bool already_registered = []() {                        \
    pjrt::SetPjrtApi(plugin_name, get_plugin_api).IgnoreError(); \
    return true;                                                 \
  }();

#endif  // XLA_PJRT_PLUGIN_STATIC_REGISTRATION_H_
