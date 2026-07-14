/* Copyright 2025 The OpenXLA Authors.

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

#ifndef XLA_PYTHON_IFRT_IR_TRANSFORMS_DEBUG_H_
#define XLA_PYTHON_IFRT_IR_TRANSFORMS_DEBUG_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/PassManager.h"
#include "xla/tsl/platform/file_system.h"

namespace xla {
namespace ifrt {

// Initializes the pass manager with default options that make debugging easier.
void InitPassManager(mlir::PassManager& pm, llvm::StringRef pm_name,
                     std::string dump_dir = "", std::string dump_pass_re = "",
                     std::string dump_func_re = ".*",
                     bool enable_timing = false);

// Diagnostic handler that converts MLIR errors to `absl::Status`.
class StatusScopedDiagnosticHandler : public mlir::SourceMgrDiagnosticHandler {
 public:
  explicit StatusScopedDiagnosticHandler(mlir::MLIRContext* context);

  // CHECK-fails if there's any error that is not consumed.
  ~StatusScopedDiagnosticHandler();

  // Returns Status corresponding to the diagnostics reported. This consumes the
  // diagnostics reported and returns a Status of type Unknown. It is required
  // to consume the error status, if there is one, before destroying the object.
  absl::Status ConsumeStatus();

 private:
  void Handle(mlir::Diagnostic& diag);

  std::string diag_str_;
  llvm::raw_string_ostream diag_stream_;
  llvm::SourceMgr source_mgr_;
};

// A simple wrapper that allows using `WritableFile` as `llvm::raw_ostream`.
class AppendOnlyFileRawStream : public llvm::raw_ostream {
 public:
  explicit AppendOnlyFileRawStream(std::unique_ptr<tsl::WritableFile> file)
      : file_(std::move(file)), pos_(0) {
    SetUnbuffered();
  }

  ~AppendOnlyFileRawStream() override {
    if (file_) {
      file_->Close().IgnoreError();
    }
  }

 private:
  void write_impl(const char* Ptr, size_t Size) override {
    if (!file_) {
      return;
    }

    if (file_->Append(absl::string_view(Ptr, Size)).ok()) {
      pos_ += Size;
    } else {
      if (!file_->Close().ok()) {
        absl::string_view filename;
        if (!file_->Name(&filename).ok()) {
          filename = "unknown";
        }
        LOG(ERROR) << "Failed to close file: " << filename;
        file_ = nullptr;
      }
    }
  }

  uint64_t current_pos() const override { return pos_; }

  std::unique_ptr<tsl::WritableFile> file_;
  uint64_t pos_;
};

}  // namespace ifrt
}  // namespace xla

#endif  // XLA_PYTHON_IFRT_IR_TRANSFORMS_DEBUG_H_
