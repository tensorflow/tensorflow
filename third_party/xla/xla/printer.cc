/* Copyright 2023 The OpenXLA Authors.

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

#include "xla/printer.h"

#include <cstring>
#include <string>
#include <utility>

#include "absl/strings/cord.h"
#include "absl/strings/cord_buffer.h"
#include "absl/strings/str_cat.h"
#include "tsl/platform/logging.h"

namespace xla {

void StringPrinter::Append(const absl::AlphaNum& a) {
  absl::StrAppend(&result_, a);
}

std::string StringPrinter::ToString() && { return std::move(result_); }

void CordPrinter::AppendImpl(const absl::AlphaNum& a) {
  // CordBuffer methods all contain branches so not cheap, caching the values.
  const size_t capacity = buffer_.capacity();
  size_t length = buffer_.length();
  if (capacity <= a.size()) {
    if (length > 0) AppendBuffer();
    result_.Append(a.Piece());
    return;
  }
  if (capacity < a.size() + length) {
    // Because capacity > a.size(), length > 0.
    AppendBuffer();
    DCHECK_EQ(buffer_.length(), 0);
    length = 0;
  }
  DCHECK_LE(a.size(), buffer_.available().size());
  std::memcpy(buffer_.data() + length, a.data(), a.size());
  buffer_.IncreaseLengthBy(a.size());
}

void CordPrinter::AppendBuffer() {
  DCHECK_GT(buffer_.length(), 0);
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
