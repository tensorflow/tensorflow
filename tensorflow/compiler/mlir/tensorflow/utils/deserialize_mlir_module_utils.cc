/* Copyright 2025 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/mlir/tensorflow/utils/deserialize_mlir_module_utils.h"

#include <climits>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/OwningOpRef.h"  // from @llvm-project
#include "mlir/Parser/Parser.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/utils/error_util.h"
#include "xla/status_macros.h"
#include "tensorflow/core/lib/io/inputstream_interface.h"
#include "tensorflow/core/lib/io/zlib_compression_options.h"
#include "tensorflow/core/lib/io/zlib_inputstream.h"
#include "tensorflow/core/platform/tstring.h"

namespace tensorflow {
namespace {
// Wrap memory buffer into InputStreamInterface
class MemoryInputStream : public tensorflow::io::InputStreamInterface {
 public:
  explicit MemoryInputStream(const char* buffer, size_t length)
      : buf_(buffer), len_(length), pos_(0) {}

  ~MemoryInputStream() override = default;

  absl::Status ReadNBytes(int64_t bytes_to_read, tstring* result) override {
    result->clear();
    if (bytes_to_read < 0) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Can't read a negative number of bytes: ", bytes_to_read));
    }
    absl::Status status = absl::OkStatus();
    int64_t bytes = bytes_to_read;
    if (pos_ + bytes_to_read > len_) {
      bytes = len_ - pos_;
      status = absl::OutOfRangeError("Reached end of file");
    }
    if (bytes > 0) {
      result->resize(bytes);
      memcpy(&(*result)[0], &buf_[pos_], bytes);
      pos_ += bytes;
    }
    return status;
  }

  int64_t Tell() const override { return pos_; }

  absl::Status Reset() override {
    pos_ = 0;
    return absl::OkStatus();
  }

 private:
  const char* buf_;  // Not owned.
  int64_t len_;
  int64_t pos_ = 0;  // Tracks where we are in the file.
};
}  // namespace

absl::Status DeserializeMlirModule(
    llvm::StringRef serialized_mlir_module, mlir::MLIRContext* mlir_context,
    mlir::OwningOpRef<mlir::ModuleOp>* mlir_module) {
  TF_RET_CHECK(!serialized_mlir_module.empty())
      << "unexpected empty serialized MLIR module string";
  TF_RET_CHECK(mlir_module) << "unexpected null MLIR module pointer";

  // Make sure we catch any error reported by MLIR and forward it to the TF
  // error reporting system.
  mlir::StatusScopedDiagnosticHandler error_handler(mlir_context);

  // Look for the GZIP magic number to check if this is a compressed bytecode.
  if (serialized_mlir_module.starts_with("\x1f\x8b")) {
    // Try to uncompress the and parse the bytecode.
    auto input_stream = std::make_unique<MemoryInputStream>(
        serialized_mlir_module.data(), serialized_mlir_module.size());
    io::ZlibCompressionOptions options = io::ZlibCompressionOptions::GZIP();
    auto zlib_stream = std::make_unique<tensorflow::io::ZlibInputStream>(
        input_stream.get(), options.input_buffer_size,
        options.output_buffer_size, options);
    tstring uncompressed_bytecode;
    absl::Status s = zlib_stream->ReadNBytes(/*bytes_to_read=*/INT_MAX,
                                             &uncompressed_bytecode);
    // OK status means the decompression is successful.
    // OutOfRange error means the decompression is successful but end of input
    // was reached before *bytes_to_read* bytes were read.
    if (!s.ok() && !absl::IsOutOfRange(s)) {
      // Failed to uncompress the bytecode and it is not the end of the input.
      return error_handler.Combine(absl::InvalidArgumentError(
          absl::StrCat("Failed to uncompress MLIR module", s.message())));
    }
    // Parse the uncompressed bytecode.
    auto uncompressed_bytecode_str =
        std::string(uncompressed_bytecode.data(), uncompressed_bytecode.size());
    *mlir_module = mlir::parseSourceString<mlir::ModuleOp>(
        uncompressed_bytecode_str, mlir_context);
    if (!*mlir_module) {
      // Uncompressing was successful but the parsed MLIR module is invalid.
      return error_handler.Combine(absl::InvalidArgumentError(
          "Failed to parse MLIR module after uncompressing"));
    }
  } else {
    *mlir_module = mlir::parseSourceString<mlir::ModuleOp>(
        serialized_mlir_module, mlir_context);
    if (!*mlir_module) {
      return error_handler.Combine(
          absl::InvalidArgumentError("could not parse MLIR module"));
    }
  }
  return absl::OkStatus();
}

}  // namespace tensorflow
