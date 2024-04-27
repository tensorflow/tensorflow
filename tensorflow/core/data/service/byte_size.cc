/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/data/service/byte_size.h"

#include <cstddef>
#include <string>

#include "absl/strings/str_cat.h"

namespace tensorflow {
namespace data {

size_t ByteSize::ToUnsignedBytes() const { return bytes_; }
double ByteSize::ToDoubleBytes() const { return static_cast<double>(bytes_); }
double ByteSize::ToDoubleKB() const { return *this / ByteSize::KB(1); }
double ByteSize::ToDoubleMB() const { return *this / ByteSize::MB(1); }
double ByteSize::ToDoubleGB() const { return *this / ByteSize::GB(1); }
double ByteSize::ToDoubleTB() const { return *this / ByteSize::TB(1); }

std::string ByteSize::DebugString() const {
  if (*this < ByteSize::KB(1)) {
    return absl::StrCat(ToUnsignedBytes(), "B");
  }
  if (*this < ByteSize::MB(1)) {
    return absl::StrCat(ToDoubleKB(), "KB");
  }
  if (*this < ByteSize::GB(1)) {
    return absl::StrCat(ToDoubleMB(), "MB");
  }
  if (*this < ByteSize::TB(1)) {
    return absl::StrCat(ToDoubleGB(), "GB");
  }
  return absl::StrCat(ToDoubleTB(), "TB");
}

}  // namespace data
}  // namespace tensorflow
