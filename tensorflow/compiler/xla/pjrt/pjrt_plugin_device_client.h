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
#ifndef TENSORFLOW_COMPILER_XLA_PJRT_PJRT_PLUGIN_DEVICE_CLIENT_H_
#define TENSORFLOW_COMPILER_XLA_PJRT_PJRT_PLUGIN_DEVICE_CLIENT_H_

#include <memory>

#include "tensorflow/compiler/xla/statusor.h"

namespace xla {

class PjRtClient;

// Not implemented by default. It is the responsibility of the plugin device
// author to provide an implementation of this function. It is recommended to
// implement this in //tensorflow/compiler/plugin:plugin
StatusOr<std::unique_ptr<PjRtClient>> GetTfrtPluginDeviceClient();

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_PJRT_PJRT_PLUGIN_DEVICE_CLIENT_H_
