/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/printer.h"

#include <cstring>
#include <string>
#include <utility>

#include "absl/strings/cord.h"
#include "absl/strings/string_view.h"

namespace xla {

void StringPrinter::Append(const absl::AlphaNum& a) {
  absl::StrAppend(&result_, a);
}

std::string StringPrinter::ToString() && { return std::move(result_); }

void CordPrinter::AppendImpl(const absl::AlphaNum& a) {
  if (buffer_.capacity() <= a.size()) {
    AppendBuffer();
    result_.Append(a.Piece());
    return;
  }
  if (buffer_.capacity() < a.size() + buffer_.length()) {
    AppendBuffer();
  }
  auto dst = buffer_.available_up_to(a.size());
  std::memcpy(dst.data(), a.data(), a.size());
  buffer_.IncreaseLengthBy(a.size());
}

void CordPrinter::AppendBuffer() {
  if (buffer_.length() == 0) return;
  result_.Append(std::move(buffer_));
  constexpr size_t kCordBufferSize = 64 << 10;
  buffer_ =
      absl::CordBuffer::CreateWithCustomLimit(kCordBufferSize, kCordBufferSize);
}

void CordPrinter::Append(const absl::AlphaNum& a) { AppendImpl(a); }

absl::Cord CordPrinter::ToCord() && {
  if (buffer_.length() > 0) result_.Append(std::move(buffer_));
  return std::move(result_);
}

}  // namespace xla
