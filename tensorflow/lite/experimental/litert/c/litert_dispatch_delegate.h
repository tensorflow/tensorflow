// Copyright 2024 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_LITERT_C_LITERT_DISPATCH_DELEGATE_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_LITERT_C_LITERT_DISPATCH_DELEGATE_H_

#include <stddef.h>

#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/experimental/litert/vendors/c/litert_dispatch.h"

typedef struct LiteRtDispatchDelegateOptions LiteRtDispatchDelegateOptions;
typedef struct LiteRtEnvironmentT* LiteRtEnvironment;

// Returns DispatchDelegateOptions populated with default values.
LiteRtDispatchDelegateOptions* LiteRtCreateDefaultDispatchDelegateOptions(
    LiteRtEnvironment environment);

TfLiteStatus LiteRtAddDispatchDelegateOption(
    LiteRtDispatchDelegateOptions* options, LiteRtDispatchOption option);

void LiteRtDestroyDispatchDelegateOptions(
    LiteRtDispatchDelegateOptions* options);

// Create a delegate that uses the Dispatch API for execution. Takes ownership
// of the passed `options`. Must outlive the TFL interpreter.
TfLiteOpaqueDelegate* LiteRtCreateDispatchDelegate(
    LiteRtEnvironment environment, LiteRtDispatchDelegateOptions* options);

// Do any needed cleanup and delete 'delegate'.
void LiteRtDestroyDispatchDelegate(TfLiteOpaqueDelegate* delegate);

//
// Common option helpers
//

// Alloc base is the address of the first byte of flatbuffer model in memory. It
// is used by ops to find the start of npu byte code appended to the file.
TfLiteStatus LiteRtDispatchDelegateAddAllocBaseOption(
    LiteRtDispatchDelegateOptions* options, const void* alloc_base);

// Alloc fd is the file descriptor for an mmapped flatbuffer. It is used by ops
// to find the start of npu byte code appended to the file.
TfLiteStatus LiteRtDispatchDelegateAddAllocFdOption(
    LiteRtDispatchDelegateOptions* options, int alloc_fd);

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LITERT_C_LITERT_DISPATCH_DELEGATE_H_
