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

#include <iostream>
#include <memory>
#include <string>

#include "absl/status/status.h"
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
    int64_t bytes = bytes_to_read;
    absl::Status s = absl::OkStatus();
    if (pos_ + bytes_to_read > len_) {
      bytes = len_ - pos_;
      s = absl::OutOfRangeError("Reached end of file");
    }
    if (bytes > 0) {
      result->resize(bytes);
      memcpy(&(*result)[0], &buf_[pos_], bytes);
      pos_ += bytes;
    }
    return s;
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
  LOG(INFO) << "[debugsa] (loginfo) DeserializeMlirModule";

  std::unique_ptr<MemoryInputStream> input_stream(new MemoryInputStream(
      serialized_mlir_module.data(), serialized_mlir_module.size()));
  io::ZlibCompressionOptions options = io::ZlibCompressionOptions::GZIP();
  std::unique_ptr<tensorflow::io::ZlibInputStream> zlib_stream(
      new tensorflow::io::ZlibInputStream(input_stream.get(),
                                          options.input_buffer_size,
                                          options.output_buffer_size, options));
  tstring uncompressed_bytecode;
  absl::Status s = zlib_stream->ReadNBytes(serialized_mlir_module.size(),
                                           &uncompressed_bytecode);
  if (s.ok() || absl::IsOutOfRange(s)) {
    // Parse as compressed bytecode.
    *mlir_module = mlir::parseSourceString<mlir::ModuleOp>(
        std::string(uncompressed_bytecode.data(), uncompressed_bytecode.size()),
        mlir_context);
    if (*mlir_module) {
      return absl::OkStatus();
    }
    LOG(ERROR) << "Failed to parse MLIR module after uncompressing. Trying to "
                  "parse as text instead.";
  } else {
    LOG(ERROR) << "Failed to decompress MLIR module: " << s
               << "trying to parse as text instead.";
  }
  *mlir_module = mlir::parseSourceString<mlir::ModuleOp>(
      serialized_mlir_module.data(), mlir_context);

  if (!*mlir_module)
    return error_handler.Combine(
        absl::InvalidArgumentError("could not parse MLIR module"));
  return absl::OkStatus();
}

}  // namespace tensorflow
