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

#ifndef TENSORFLOW_COMPILER_XLA_STREAM_EXECUTOR_TPU_PJRT_API_H_
#define TENSORFLOW_COMPILER_XLA_STREAM_EXECUTOR_TPU_PJRT_API_H_

#include "absl/strings/string_view.h"
#include "tensorflow/compiler/xla/pjrt/c/pjrt_c_api.h"
#include "tensorflow/compiler/xla/status.h"
#include "tensorflow/compiler/xla/statusor.h"

namespace stream_executor {
namespace tpu {

// Gets and sets the global map for PJRT_Api*. Not thread safe. `device_type` is
// case insensitive.
xla::StatusOr<const PJRT_Api*> PjrtApi(absl::string_view device_type);
xla::Status SetPjrtApi(absl::string_view device_type, const PJRT_Api* api);

// Loads a PJRT plugin. This method dlopen the plugin library, gets `PJRT_Api*`
// and calls `SetPjrtApi`. Currently this method only supports libtpu.
// TODO(b/261137756): support other PJRT plugins.
xla::Status LoadPjrtPlugin(absl::string_view device_type,
                           absl::string_view library_path);

}  // namespace tpu
}  // namespace stream_executor

#endif  // TENSORFLOW_COMPILER_XLA_STREAM_EXECUTOR_TPU_PJRT_API_H_
