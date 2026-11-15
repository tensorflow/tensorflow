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

#ifndef TENSORFLOW_COMPILER_MLIR_LITE_EXPORT_PAYLOADS_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_EXPORT_PAYLOADS_H_

#include <cstdint>
#include <vector>

#include "absl/functional/any_invocable.h"
#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/AsmState.h"  // from @llvm-project

namespace mlir::TFL::fb_export {

// Alignment of TFLite buffers, which payloads start at.
inline constexpr uint64_t kBufferAlignment = 16;

// Receives the bytes of a payload, possibly in several chunks.
using ByteSink = absl::FunctionRef<absl::Status(absl::string_view)>;

// Streams the bytes of a payload into a sink.
using PayloadWriter = absl::AnyInvocable<absl::Status(ByteSink) const>;

// Where a payload ended up, relative to the start of the model file.
struct PayloadLocation {
  uint64_t offset = 0;
  uint64_t size = 0;
};

// Bytes stored after the flatbuffer in the model file and referenced by offset
// and size (constant buffers, large custom options). Keeping them out of the
// flatbuffer lifts its 2GB limit and lets them be streamed straight from the
// MLIR attributes to the file.
class PayloadTable {
 public:
  // Adds a payload of `size` bytes and returns its id. Payloads with equal
  // `fingerprint` and size must have equal bytes; they are written once and
  // share a location.
  int Add(uint64_t fingerprint, uint64_t size, PayloadWriter write);

  // Ties `blob` to payload `id`, typically because `id` reads from it. To keep
  // the peak memory low, a blob is released once all payloads tied to it have
  // been written.
  void TieBlob(int id, AsmResourceBlob* blob);

  // Appends all payloads to `stream`, in the order they were added, and
  // records their locations relative to `file_start`. Fails if a payload
  // doesn't have the size it was added with.
  absl::Status WriteTo(llvm::raw_pwrite_stream& stream, uint64_t file_start);

  // Location of payload `id`. Valid after `WriteTo`.
  const PayloadLocation& location(int id) const { return locations_[id]; }

 private:
  struct Payload {
    uint64_t fingerprint;
    uint64_t size;
    PayloadWriter write;
    llvm::SmallVector<AsmResourceBlob*, 1> blobs;
  };

  std::vector<Payload> payloads_;
  std::vector<PayloadLocation> locations_;
};

}  // namespace mlir::TFL::fb_export

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_EXPORT_PAYLOADS_H_
