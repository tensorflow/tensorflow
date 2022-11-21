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

#include <cstdlib>
#include <memory>
#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/OperationSupport.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "tensorflow/tsl/platform/env.h"
#include "tensorflow/tsl/platform/path.h"
#include "tensorflow/tsl/platform/status.h"

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

// Creates a new file to dump the intermediate MLIRs by prefixing the
// `dump_file_name` with the value of the TF_QUANT_MLIR_DUMP_PREFIX env
// variable. Returns absl::FailedPreconditionError if the env variable is not
// set or set to an empty string.
absl::StatusOr<std::unique_ptr<llvm::raw_fd_ostream>> CreateMlirDumpFile(
    const absl::string_view dump_file_name) {
  const absl::StatusOr<std::string> dump_dir = GetMlirDumpDir();
  if (!dump_dir.ok()) {
    return dump_dir.status();
  }

  auto *env = tsl::Env::Default();
  const tsl::Status status = env->RecursivelyCreateDir(*dump_dir);
  if (!status.ok()) {
    return tsl::ToAbslStatus(status);
  }

  std::error_code ec{};  // NOLINT: Required to create llvm::raw_fd_ostream
  const std::string dump_file_path =
      tsl::io::JoinPath(*dump_dir, dump_file_name);
  auto dump_file = std::make_unique<llvm::raw_fd_ostream>(dump_file_path, ec);
  if (ec) {
    return absl::InternalError(absl::StrFormat(
        "Unable to open file: %s, error: %s", dump_file_path, ec.message()));
  }

  LOG(INFO) << "IR dump file created: " << dump_file_path;
  return dump_file;
}

}  // namespace

void EnableIrPrinting(llvm::raw_ostream &out_stream, mlir::PassManager &pm) {
  mlir::OpPrintingFlags flag{};
  flag.useLocalScope().elideLargeElementsAttrs().enableDebugInfo();

  // IR printing requires multithreading disabled.
  pm.getContext()->disableMultithreading();

  // The configuration uses the default parameter values for
  // `PassManager::enableIRPrinting`, except for the `printModuleScope`
  // parameter, which is true by default. It is set to false to avoid the dump
  // file size becoming too large when the passes are running on a large model.
  pm.enableIRPrinting(
      /*shouldPrintBeforePass=*/[](mlir::Pass *,
                                   mlir::Operation *) { return true; },
      /*shouldPrintAfterPass=*/
      [](mlir::Pass *, mlir::Operation *) { return true; },
      /*printModuleScope=*/false, /*printAfterOnlyOnChange=*/true,
      /*printAfterOnlyOnFailure=*/false, out_stream, flag);

  LOG(INFO) << "IR dump for TensorFlow quantization pipeline enabled.";
}

// TODO(b/259374854): Create tests for MaybeEnableIrPrinting.
absl::StatusOr<std::unique_ptr<llvm::raw_ostream>> MaybeEnableIrPrinting(
    mlir::PassManager &pm, const absl::string_view name) {
  if (!VLOG_IS_ON(1)) {
    LOG(INFO) << "Verbosity level too low to enable IR printing.";
    return nullptr;
  }

  absl::StatusOr<std::unique_ptr<llvm::raw_fd_ostream>> dump_file =
      CreateMlirDumpFile(/*dump_file_name=*/absl::StrCat(name, ".mlir"));
  if (absl::IsFailedPrecondition(dump_file.status())) {
    // Requirements for enabling IR dump are not met. IR printing will not be
    // enabled.
    LOG(WARNING) << dump_file.status();
    return nullptr;
  } else if (!dump_file.ok()) {
    return dump_file.status();
  }

  EnableIrPrinting(**dump_file, pm);

  return dump_file;
}

}  // namespace quantization
}  // namespace tensorflow
