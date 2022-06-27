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

#ifndef TENSORFLOW_COMPILER_XLA_PJRT_C_PJRT_C_API_HELPERS_H_
#define TENSORFLOW_COMPILER_XLA_PJRT_C_PJRT_C_API_HELPERS_H_

#include <functional>
#include <memory>

#include "tensorflow/compiler/xla/pjrt/c/pjrt_c_api.h"

namespace pjrt {

using PJRT_ClientDeleter = std::function<void(PJRT_Client *)>;

// Pass in an API pointer; recieve a custom deleter for smart pointers.
// The lifetime of the Api pointed to must be longer than the client.
PJRT_ClientDeleter MakeClientDeleter(const PJRT_Api *api);

}  // namespace pjrt

#endif  // TENSORFLOW_COMPILER_XLA_PJRT_C_PJRT_C_API_HELPERS_H_
