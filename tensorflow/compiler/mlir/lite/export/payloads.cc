/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/mlir/lite/export/payloads.h"

#include <cstdint>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/AsmState.h"  // from @llvm-project

namespace mlir::TFL::fb_export {
namespace {

// Zero bytes after the last payload, so that SIMD kernels (e.g. XNNPack on
// Hexagon) reading past the end of a memory mapped buffer stay in bounds.
constexpr uint64_t kTailPadding = 128;

void PadTo(llvm::raw_pwrite_stream& stream, uint64_t file_start,
           uint64_t alignment) {
  const uint64_t offset = stream.tell() - file_start;
  stream.write_zeros(llvm::alignTo(offset, alignment) - offset);
}

}  // namespace

int PayloadTable::Add(uint64_t fingerprint, uint64_t size,
                      PayloadWriter write) {
  payloads_.push_back({fingerprint, size, std::move(write), {}});
  return payloads_.size() - 1;
}

void PayloadTable::TieBlob(int id, AsmResourceBlob* blob) {
  if (blob != nullptr) payloads_[id].blobs.push_back(blob);
}

absl::Status PayloadTable::WriteTo(llvm::raw_pwrite_stream& stream,
                                   uint64_t file_start) {
  uint64_t total_size = kTailPadding;
  absl::flat_hash_map<AsmResourceBlob*, int> pending_writes;
  for (const Payload& payload : payloads_) {
    total_size += payload.size + kBufferAlignment;
    for (AsmResourceBlob* blob : payload.blobs) ++pending_writes[blob];
  }
  stream.reserveExtraSpace(total_size);

  locations_.assign(payloads_.size(), {});
  absl::flat_hash_map<std::pair<uint64_t, uint64_t>, PayloadLocation> written;
  for (int id = 0; id < payloads_.size(); ++id) {
    const Payload& payload = payloads_[id];
    const std::pair<uint64_t, uint64_t> key(payload.fingerprint, payload.size);
    if (auto it = written.find(key); it != written.end()) {
      locations_[id] = it->second;
    } else {
      PadTo(stream, file_start, kBufferAlignment);
      PayloadLocation& location = locations_[id];
      location.offset = stream.tell() - file_start;
      absl::Status status = payload.write([&](absl::string_view chunk) {
        stream.write(chunk.data(), chunk.size());
        location.size += chunk.size();
        return absl::OkStatus();
      });
      if (!status.ok()) return status;
      if (location.size != payload.size) {
        return absl::InternalError(
            absl::StrCat("payload ", id, " has ", location.size,
                         " bytes instead of ", payload.size));
      }
      written[key] = location;
    }
    for (AsmResourceBlob* blob : payload.blobs) {
      if (--pending_writes[blob] == 0) *blob = AsmResourceBlob();
    }
  }

  stream.write_zeros(kTailPadding);
  PadTo(stream, file_start, kBufferAlignment);
  return absl::OkStatus();
}

}  // namespace mlir::TFL::fb_export
