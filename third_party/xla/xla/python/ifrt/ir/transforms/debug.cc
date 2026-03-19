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

#include "xla/python/ifrt/ir/transforms/debug.h"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <string>
#include <utility>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/escaping.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/string_view.h"
#include "absl/strings/strip.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassInstrumentation.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/file_system.h"
#include "tsl/platform/path.h"
#include "tsl/platform/random.h"
#include "tsl/platform/regexp.h"

namespace xla {
namespace ifrt {

namespace {

// Reproducer stream that emits a reproducer to the given `llvm::raw_ostream`.
class ReproducerStream : public mlir::ReproducerStream {
 public:
  ReproducerStream(std::string name, std::unique_ptr<llvm::raw_ostream> os)
      : name_(std::move(name)), os_(std::move(os)) {}

  llvm::StringRef description() override { return name_; }

  llvm::raw_ostream& os() override { return *os_; }

 private:
  std::string name_;
  std::unique_ptr<llvm::raw_ostream> os_;
};

// Returns a function that builds a reproducer stream, or nullptr if the MLIR
// reproducer will not be enabled.
mlir::ReproducerStreamFactory GetReproducerStreamFactory() {
  absl::string_view directory;
  if (const char* const dir = getenv("MLIR_CRASH_REPRODUCER_DIRECTORY")) {
    directory = dir;
  } else if (const char* const dir = getenv("TEST_UNDECLARED_OUTPUTS_DIR")) {
    directory = dir;
  } else {
    LOG_FIRST_N(INFO, 1)
        << "MLIR crash reproducer is not enabled; set "
           "'MLIR_CRASH_REPRODUCER_DIRECTORY' environment variable to enable";
    return nullptr;
  }

  std::string path = tsl::io::JoinPath(
      directory,
      absl::StrCat("ifrt_ir_mlir_repro_", tsl::random::New64(), ".mlir"));

  return [path = std::move(path)](
             std::string& error) -> std::unique_ptr<mlir::ReproducerStream> {
    std::unique_ptr<tsl::WritableFile> f;
    if (const absl::Status status =
            tsl::Env::Default()->NewWritableFile(path, &f);
        !status.ok()) {
      error = status.ToString();
      absl::StrAppend(&error, "; failed to create '", path,
                      "' for writing an MLIR reproducer");
      return nullptr;
    }
    return std::make_unique<ReproducerStream>(
        path, std::make_unique<AppendOnlyFileRawStream>(std::move(f)));
  };
}

// Escapes the string to be suitable to be used as a filename.
std::string EscapeFilename(absl::string_view filename) {
#if defined(PLATFORM_GOOGLE)
  std::string escaped = strings::EscapeFileName(filename);
  ReplaceCharacters(&escaped, "%", '+');
  return escaped;
#else
  return absl::StrReplaceAll(filename, {{"/", "~"}});
#endif
}

// Pass instrumentation that dumps programs based on the criteria specified by
// `--ifrt_ir_mlir_dump_*` flags.
//
// While `mlir::PassManager::enableIRPrinting` provides a similar functionality,
// it is cumbersome to manually copy printed IRs and run them with `mlir_opt`.
// Also, long programs are often truncated during printing. Instead, this
// instrumentation dumps programs to external directories for convenience.
class DumpInstrumentation : public mlir::PassInstrumentation {
 public:
  explicit DumpInstrumentation(std::string pm_name, std::string dump_dir,
                               std::string dump_pass_re,
                               std::string dump_func_re)
      : id_(tsl::random::New64()),
        pm_name_(std::move(pm_name)),
        dump_dir_(std::move(dump_dir)),
        dump_pass_re_(std::move(dump_pass_re)),
        dump_func_re_(std::move(dump_func_re)) {}

  DumpInstrumentation(const DumpInstrumentation& other) = delete;
  DumpInstrumentation& operator=(const DumpInstrumentation& other) = delete;

  void runBeforePass(mlir::Pass* pass, mlir::Operation* op) override {
    // Always print before the first pass.
    if (!printed_) {
      Dump("before_all", op);
      printed_ = true;
    }

    if (RE2::FullMatch(pass->getName(), dump_pass_re_)) {
      Dump(absl::StrCat(absl::string_view(pass->getName()), "_before"), op);
    }
  }

  void runAfterPass(mlir::Pass* pass, mlir::Operation* op) override {
    if (RE2::FullMatch(pass->getName(), dump_pass_re_)) {
      Dump(absl::StrCat(absl::string_view(pass->getName()), "_after"), op);
    }
  }

 private:
  // Dumps the given op. `name` is used as part of the filename to help
  // distinguish dumps at different passes.
  void Dump(absl::string_view name, mlir::Operation* op) {
    // Find names of all func ops with public visibility and check whether at
    // least one of them matches `--ifrt_ir_mlir_dump_func_re`.
    llvm::SmallVector<absl::string_view> func_names;
    bool match = false;
    op->walk([&](mlir::func::FuncOp func) {
      if (func.isPublic()) {
        const absl::string_view name = func.getSymName();
        if (name.empty()) {
          return;
        }

        func_names.push_back(name);
        if (RE2::FullMatch(name, dump_func_re_)) {
          match = true;
        }
      }
    });
    if (!func_names.empty() && !match) {
      return;
    }

    // Sort the function names for determinism.
    llvm::sort(func_names);

    // Build a filename such that it contains function names and pass names for
    // easy disambiguation.

    std::string filename;
    absl::StrAppend(&filename, id_);
    if (!pm_name_.empty()) {
      absl::StrAppend(&filename, ".", pm_name_);
    }
    if (!func_names.empty()) {
      absl::StrAppend(&filename, ".", absl::StrJoin(func_names, "-"));
    }
    absl::StrAppend(&filename, ".", name, ".mlir");

    const std::string path =
        tsl::io::JoinPath(dump_dir_, EscapeFilename(filename));

    // Create the file for dumping. Failures are logged instead of being
    // propagated to the client because they are non-fatal.
    std::unique_ptr<tsl::WritableFile> f;
    if (const absl::Status status =
            tsl::Env::Default()->NewWritableFile(path, &f);
        !status.ok()) {
      LOG(ERROR) << "unable to create '" << path
                 << "' for dumping IFRT IR programs: " << status;
      return;
    }

    AppendOnlyFileRawStream os(std::move(f));
    op->print(os);
  }

  // A unique id assigned to each pass manager.
  const uint64_t id_;

  const std::string pm_name_;
  const std::string dump_dir_;
  const std::string dump_pass_re_;
  const std::string dump_func_re_;

  bool printed_ = false;
};

}  // namespace

void InitPassManager(mlir::PassManager& pm, llvm::StringRef pm_name,
                     std::string dump_dir, std::string dump_pass_re,
                     std::string dump_func_re, bool enable_timing) {
  mlir::registerPassManagerCLOptions();
  CHECK(mlir::succeeded(mlir::applyPassManagerCLOptions(pm)))
      << "could not initialize MLIR pass manager CL options";

  // Set a default crash reproducer for easier debugging.
  if (auto reproducer_stream_factory = GetReproducerStreamFactory()) {
    pm.enableCrashReproducerGeneration(std::move(reproducer_stream_factory));
  }

  if (dump_dir == "sponge") {
    if (const char* const dir = getenv("TEST_UNDECLARED_OUTPUTS_DIR")) {
      dump_dir = dir;
    } else {
      LOG(ERROR)
          << "dump dir is set to `sponge` outside of a test; ignoring value.";
      dump_dir = "";
    }
  }
  if (!dump_dir.empty()) {
    pm.addInstrumentation(std::make_unique<DumpInstrumentation>(
        std::string(pm_name), std::move(dump_dir), std::move(dump_pass_re),
        std::move(dump_func_re)));
  }

  if (VLOG_IS_ON(3)) {
    pm.getContext()->disableMultithreading();
    pm.enableIRPrinting();
  }

  // Enable pass timing. Note: MLIR expects `mlir::PassManager::enableTiming` to
  // be called after all instrumentations are added.
  if (enable_timing) {
    pm.enableTiming();
  }
}

StatusScopedDiagnosticHandler::StatusScopedDiagnosticHandler(
    mlir::MLIRContext* context)
    : mlir::SourceMgrDiagnosticHandler(source_mgr_, context, diag_stream_),
      diag_stream_(diag_str_) {
  setHandler([this](mlir::Diagnostic& diag) { Handle(diag); });
}

StatusScopedDiagnosticHandler::~StatusScopedDiagnosticHandler() {
  CHECK(diag_str_.empty())
      << "unhandled error in StatusScopedDiagnosticHandler";
}

absl::Status StatusScopedDiagnosticHandler::ConsumeStatus() {
  absl::Status status = absl::UnknownError(diag_str_);
  diag_str_.clear();
  return status;
}

void StatusScopedDiagnosticHandler::Handle(mlir::Diagnostic& diag) {
  const size_t diag_str_size = diag_str_.size();

  // Emit the diagnostic and flush the stream.
  emitDiagnostic(diag);
  diag_stream_.flush();

  const auto curr_diag_str = absl::string_view(diag_str_).substr(diag_str_size);
  bool consumed = false;

  // Emit LOG/VLOG according to the severity.
  switch (diag.getSeverity()) {
    case mlir::DiagnosticSeverity::Note:
      LOG(INFO) << curr_diag_str;
      consumed = true;
      break;
    case mlir::DiagnosticSeverity::Warning:
      LOG(WARNING) << curr_diag_str;
      consumed = true;
      break;
    case mlir::DiagnosticSeverity::Error:
      // No need to log errors because they will be explicitly handled by
      // `ConsumeStatus()`.
      break;
    case mlir::DiagnosticSeverity::Remark:
      VLOG(1) << curr_diag_str;
      consumed = true;
      break;
  }

  if (consumed) {
    diag_str_.resize(diag_str_size);
  }
}

}  // namespace ifrt
}  // namespace xla
