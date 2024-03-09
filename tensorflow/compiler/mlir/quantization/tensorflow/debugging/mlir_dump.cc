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
#include "tensorflow/compiler/mlir/quantization/tensorflow/debugging/mlir_dump.h"

#include <cstdint>
#include <cstdlib>
#include <memory>
#include <string>
#include <utility>

#include "absl/log/log.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/OperationSupport.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "tsl/platform/env.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/file_system.h"
#include "tsl/platform/path.h"
#include "tsl/platform/status.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/stringpiece.h"

namespace tensorflow {
namespace quantization {
namespace {

// Retrieve the MLIR dump directory. The directory is read from the environment
// variable `TF_QUANT_MLIR_DUMP_PREFIX`. However, if a special value "sponge" is
// set to `TF_QUANT_MLIR_DUMP_PREFIX`, it uses the directory set in
// `TEST_UNDECLARED_OUTPUT_DIRS`. Returns `absl::FailedPreconditionError` if
// either:
//   1. `TF_QUANT_MLIR_DUMP_PREFIX` is not set (empty), or
//   2. `TEST_UNDECLARED_OUTPUT_DIRS` is not set (empty) when
//      `TF_QUANT_MLIR_DUMP_PREFIX = "sponge"`.
absl::StatusOr<std::string> GetMlirDumpDir() {
  auto dump_dir = std::string(
      absl::NullSafeStringView(std::getenv("TF_QUANT_MLIR_DUMP_PREFIX")));
  if (dump_dir.empty()) {
    return absl::FailedPreconditionError(
        "Environment variable not set: TF_QUANT_MLIR_DUMP_PREFIX, "
        "IR dump file for TF quantization is not created.");
  }

  if (absl::EqualsIgnoreCase(dump_dir, "sponge")) {
    if (!tsl::io::GetTestUndeclaredOutputsDir(&dump_dir)) {
      return absl::FailedPreconditionError(
          "Environment variable TF_QUANT_MLIR_DUMP_PREFIX=sponge but "
          "TEST_UNDECLARED_OUTPUT_DIRS not set.");
    }
  }

  return dump_dir;
}

// A simple wrapper of tsl::WritableFile so that mlir Pass infra can use it.
class WritableFileWrapper : public llvm::raw_ostream {
 public:
  ~WritableFileWrapper() override { flush(); }
  static absl::StatusOr<std::unique_ptr<WritableFileWrapper>> Create(
      const std::string& filepath) {
    std::unique_ptr<tsl::WritableFile> file;
    TF_RETURN_IF_ERROR(tsl::Env::Default()->NewWritableFile(filepath, &file));
    return absl::WrapUnique(new WritableFileWrapper(std::move(file)));
  }

 private:
  explicit WritableFileWrapper(std::unique_ptr<tsl::WritableFile> file)
      : file_(std::move(file)) {
    SetBuffered();
  }

  uint64_t current_pos() const override {
    int64_t position;
    if (file_->Tell(&position).ok()) {
      return position;
    } else {
      return -1;
    }
  }

  void write_impl(const char* ptr, size_t size) override {
    if (file_ && !file_->Append(tsl::StringPiece(ptr, size)).ok()) {
      file_ = nullptr;
    }
  }

  std::unique_ptr<tsl::WritableFile> file_;
};

// Creates a new file to dump the intermediate MLIRs by prefixing the
// `dump_file_name` with the value of the TF_QUANT_MLIR_DUMP_PREFIX env
// variable. Returns absl::FailedPreconditionError if the env variable is not
// set or set to an empty string.
absl::StatusOr<std::unique_ptr<llvm::raw_ostream>> CreateMlirDumpFile(
    const absl::string_view dump_file_name) {
  const absl::StatusOr<std::string> dump_dir = GetMlirDumpDir();
  if (!dump_dir.ok()) {
    return dump_dir.status();
  }

  auto* env = tsl::Env::Default();
  TF_RETURN_IF_ERROR(env->RecursivelyCreateDir(*dump_dir));

  const std::string dump_file_path =
      tsl::io::JoinPath(*dump_dir, dump_file_name);
  TF_ASSIGN_OR_RETURN(std::unique_ptr<llvm::raw_ostream> file,
                      WritableFileWrapper::Create(dump_file_path));

  LOG(INFO) << "IR dump file created: " << dump_file_path;
  return file;
}

class PrinterConfig : public mlir::PassManager::IRPrinterConfig {
 public:
  explicit PrinterConfig(
      absl::string_view dump_file_prefix, bool print_module_scope = false,
      bool print_after_only_on_change = true,
      mlir::OpPrintingFlags op_printing_flags = mlir::OpPrintingFlags())
      : mlir::PassManager::IRPrinterConfig(
            print_module_scope, print_after_only_on_change,
            /*printAfterOnlyOnFailure=*/false, op_printing_flags),
        mlir_pass_count_(1),
        dump_file_prefix_(dump_file_prefix) {}

  void printBeforeIfEnabled(mlir::Pass* pass, mlir::Operation* op,
                            PrintCallbackFn print_callback) override {
    Dump(pass, print_callback, /*is_before=*/true);
  }

  void printAfterIfEnabled(mlir::Pass* pass, mlir::Operation* op,
                           PrintCallbackFn print_callback) override {
    Dump(pass, print_callback, /*is_before=*/false);
  }

 private:
  int64_t mlir_pass_count_;
  absl::string_view dump_file_prefix_;
  // Map from pass ptr to dump files and pass number.
  //
  // Each pass has unique and stable pointer, even for passes with the same
  // name. E.g. a PassManager could have multiple Canonicalizer passes.
  // We use this property to uniquely determine a Pass in a PassManager.
  //
  // If multiple consecutive func passes are applied to a Module. PassManager
  // will iterate over the func in the outer loop and apply the passes in the
  // inner loop. This may cause passes to run out-of-order. But the 1st runs of
  // each pass are still in-order. So we use pass_to_number_map_ to keep track
  // of the number for each pass.
  llvm::DenseMap<mlir::Pass*, std::unique_ptr<llvm::raw_ostream>>
      pass_to_dump_file_before_map_;
  llvm::DenseMap<mlir::Pass*, std::unique_ptr<llvm::raw_ostream>>
      pass_to_dump_file_after_map_;
  llvm::DenseMap<mlir::Pass*, int64_t> pass_to_number_map_;

  // Get the unique number for each pass.
  int64_t GetPassNumber(mlir::Pass* pass) {
    if (!pass_to_number_map_.contains(pass)) {
      pass_to_number_map_[pass] = mlir_pass_count_++;
    }
    return pass_to_number_map_[pass];
  }

  void Dump(mlir::Pass* pass, PrintCallbackFn print_callback, bool is_before) {
    auto& pass_to_dump_file_map = is_before ? pass_to_dump_file_before_map_
                                            : pass_to_dump_file_after_map_;
    if (!pass_to_dump_file_map.contains(pass)) {
      std::string filename = llvm::formatv(
          "{0}_{1,0+4}_{2}_{3}.mlir", dump_file_prefix_, GetPassNumber(pass),
          pass->getName().str(), is_before ? "before" : "after");
      absl::StatusOr<std::unique_ptr<llvm::raw_ostream>> dump_file =
          CreateMlirDumpFile(filename);
      if (!dump_file.ok()) {
        LOG(WARNING) << "Failed to dump MLIR module to " << filename;
        return;
      }
      pass_to_dump_file_map[pass] = std::move(*dump_file);
    }

    return print_callback(*(pass_to_dump_file_map[pass]));
  }
};

}  // namespace

void EnableIrPrinting(mlir::PassManager& pm,
                      absl::string_view file_name_prefix) {
  mlir::OpPrintingFlags flag{};
  flag.useLocalScope().elideLargeElementsAttrs().enableDebugInfo();

  // IR printing requires multithreading disabled.
  // Even if multithreading is already disabled, if we are executing within a
  // pass-manager,  disableMultithreading throws assertion fail. Below if
  // statement ensures that disableMultithreading will not be executed if
  // multithreading is already disabled.
  if (pm.getContext()->isMultithreadingEnabled()) {
    pm.getContext()->disableMultithreading();
  }

  // The configuration uses the default parameter values for
  // `PassManager::enableIRPrinting`, except for the `printModuleScope`
  // parameter, which is true by default. It is set to false to avoid the dump
  // file size becoming too large when the passes are running on a large model.
  pm.enableIRPrinting(std::make_unique<PrinterConfig>(
      file_name_prefix, /*print_module_scope=*/false,
      /*print_after_only_on_change=*/true, flag));
}

// TODO(b/259374854): Create tests for MaybeEnableIrPrinting.
absl::Status MaybeEnableIrPrinting(mlir::PassManager& pm,
                                   absl::string_view file_name_prefix) {
  if (!VLOG_IS_ON(1)) {
    LOG(INFO) << "Verbosity level too low to enable IR printing.";
    return absl::OkStatus();
  }

  EnableIrPrinting(pm, file_name_prefix);

  LOG(INFO) << "IR dump for TensorFlow quantization pipeline enabled.";
  return absl::OkStatus();
}

}  // namespace quantization
}  // namespace tensorflow
