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

#ifndef TENSORFLOW_C_TF_BUFFER_INTERNAL_H_
#define TENSORFLOW_C_TF_BUFFER_INTERNAL_H_

#include <memory>

#include "tensorflow/c/tf_buffer.h"
#include "tensorflow/core/platform/protobuf.h"  // IWYU pragma: keep
#include "tensorflow/core/platform/status.h"

namespace tensorflow {

absl::Status MessageToBuffer(const tensorflow::protobuf::MessageLite& in,
                             TF_Buffer* out);

absl::Status BufferToMessage(const TF_Buffer* in,
                             tensorflow::protobuf::MessageLite* out);

namespace internal {

struct TF_BufferDeleter {
  void operator()(TF_Buffer* buf) const { TF_DeleteBuffer(buf); }
};

}  // namespace internal

using TF_BufferPtr = std::unique_ptr<TF_Buffer, internal::TF_BufferDeleter>;

}  // namespace tensorflow

#endif  // TENSORFLOW_C_TF_BUFFER_INTERNAL_H_
