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
#include "tensorflow/compiler/xla/status.h"

namespace pjrt {

using PJRT_ClientDeleter = std::function<void(PJRT_Client*)>;

// Pass in an API pointer; receive a custom deleter for smart pointers.
// The lifetime of the Api pointed to must be longer than the client.
PJRT_ClientDeleter MakeClientDeleter(const PJRT_Api* api);

using PJRT_ErrorDeleter = std::function<void(PJRT_Error*)>;

// Pass in an API pointer; receive a custom deleter for smart pointers.
// The lifetime of the Api pointed to must be longer than the error.
PJRT_ErrorDeleter MakeErrorDeleter(const PJRT_Api* api);

using PJRT_BufferDeleter = std::function<void(PJRT_Buffer*)>;

// Pass in an API pointer; receive a custom deleter for smart pointers.
// The lifetime of the Api pointed to must be longer than the buffer.
PJRT_BufferDeleter MakeBufferDeleter(const PJRT_Api* api);

using PJRT_ExecutableDeleter = std::function<void(PJRT_Executable*)>;

// Creates a custom deleter for smart pointers.
// Pass in pointer `api` to the PJRT C API.
// The lifetime of the Api pointed to must be longer than the executable.
PJRT_ExecutableDeleter MakeExecutableDeleter(const PJRT_Api* api);

// Fatal error logging if status is not success. This terminates the process
// and frees the PJRT_Error passed in.
void LogFatalIfPjrtError(PJRT_Error* error, const PJRT_Api* api);

absl::string_view GetPjrtErrorMessage(const PJRT_Error* error,
                                      const PJRT_Api* api);

xla::Status PjrtErrorToStatus(const PJRT_Error* error, const PJRT_Api* api);

tensorflow::error::Code PjrtErrorToStatusCode(const PJRT_Error* error,
                                              const PJRT_Api* api);

PJRT_Error_Code StatusCodeToPjrtErrorCode(tensorflow::error::Code code);

using PJRT_EventDeleter = std::function<void(PJRT_Event*)>;

// Pass in an API pointer; receive a custom deleter for smart pointers.
// The lifetime of the Api pointed to must be longer than the event.
PJRT_EventDeleter MakeEventDeleter(const PJRT_Api* api);

}  // namespace pjrt

#endif  // TENSORFLOW_COMPILER_XLA_PJRT_C_PJRT_C_API_HELPERS_H_
