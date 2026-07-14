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

#include <cstdint>
#include <cstring>
#include <string>
#include <utility>

#include "absl/log/check.h"
#include "absl/strings/cord.h"
#include "absl/strings/cord_buffer.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "highwayhash/hh_types.h"
#include "xla/tsl/platform/logging.h"
#include "tsl/platform/fingerprint.h"

namespace xla {

void Printer::AppendInt64List(absl::Span<const int64_t> list,
                              bool leading_comma) {
  if (leading_comma) {
    Append(",");
  }
  Append("{");
  AppendJoin(this, list, ",");
  Append("}");
}

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

namespace {
// Generated using openssl rand.
static constexpr highwayhash::HHKey kDefaultKey = {
    0x9e0433b546e065d2ull,
    0x0e7ecad49e703760ull,
    0x83d29f20dae229b0ull,
    0x40c1ce3ff9d19a42ull,
};
}  // namespace

HighwayHashPrinter::HighwayHashPrinter()
    : Printer(true), hasher_(kDefaultKey) {}

void HighwayHashPrinter::Append(const absl::AlphaNum& a) {
  hasher_.Append(a.data(), a.size());
}

void HighwayHashPrinter::AppendInt64List(absl::Span<const int64_t> list,
                                         bool _ /*leading_comma*/) {
  // Instead of separators, prefix with the length. This is fine since
  // there's no way for the caller to distinguish between the two.
  const uint64_t num = list.size();
  hasher_.Append(reinterpret_cast<const char*>(&num), sizeof(num));
  hasher_.Append(reinterpret_cast<const char*>(list.data()),
                 list.size() * sizeof(list[0]));
}

uint64_t HighwayHashPrinter::ToFingerprint() {
  highwayhash::HHResult64 result;
  hasher_.Finalize(&result);
  return result;
}

::tsl::Fprint128 HighwayHashPrinter::ToFingerprint128() {
  highwayhash::HHResult128 result;
  hasher_.Finalize(&result);
  return ::tsl::Fprint128{result[0], result[1]};
}

}  // namespace xla
