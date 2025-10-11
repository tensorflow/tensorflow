/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/mlir/tensorflow/utils/serialize_mlir_module_utils.h"

#include <string>
#include <utility>

#include "llvm/Support/raw_ostream.h"
#include "mlir/Bytecode/BytecodeWriter.h"  // from @llvm-project
#include "mlir/IR/OperationSupport.h"  // from @llvm-project
#include "tensorflow/compiler/jit/flags.h"
#include "xla/tsl/lib/io/zlib_compression_options.h"
#include "xla/tsl/lib/io/zlib_outputbuffer.h"

namespace tensorflow {

absl::StatusOr<std::string> SerializeMlirModuleToCompressedBytecode(
    mlir::ModuleOp module_op) {
  LOG(INFO) << "[debugsa] (loginfo) SerializeMlirModuleToBytecode compressed "
               "with zliboutputbuffer";

  class WritableStringFile : public tsl::WritableFile {
   public:
    explicit WritableStringFile(std::string* data) : data_(data) {};
    ~WritableStringFile() override = default;

    absl::Status Append(absl::string_view data) override {
      absl::StrAppend(data_, data);
      return absl::OkStatus();
    }

    absl::Status Close() override { return absl::OkStatus(); }
    absl::Status Flush() override { return absl::OkStatus(); }
    absl::Status Sync() override { return absl::OkStatus(); }

   private:
    std::string* data_;
  };

  std::string bytecode;
  llvm::raw_string_ostream os(bytecode);
  mlir::BytecodeWriterConfig config;
  auto _ = mlir::writeBytecodeToFile(module_op, os, config);

  std::string compressed_bytecode;
  WritableStringFile f(&compressed_bytecode);

  tsl::io::ZlibCompressionOptions options =
      tsl::io::ZlibCompressionOptions::GZIP();
  tsl::io::ZlibOutputBuffer buffer(&f, options.input_buffer_size,
                                   options.output_buffer_size, options);
  TF_RETURN_IF_ERROR(buffer.Init());
  TF_RETURN_IF_ERROR(buffer.Append(bytecode));
  TF_RETURN_IF_ERROR(buffer.Close());
  return compressed_bytecode;
}

std::string SerializeMlirModule(mlir::ModuleOp module_op) {
  std::string serialized_mlir_module;
  llvm::raw_string_ostream os(serialized_mlir_module);
  mlir::OpPrintingFlags print_flags;
  if (GetMlirCommonFlags()->tf_mlir_enable_debug_info_serialization) {
    print_flags.enableDebugInfo();
  }
  module_op.print(os, print_flags);
  return std::move(os.str());
}

}  // namespace tensorflow
