/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/mlir/lite/debug/debug.h"

#include <stddef.h>
#include <stdint.h>

#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/OperationSupport.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassInstrumentation.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Support/FileUtilities.h"  // from @llvm-project
#include "mlir/Transforms/ViewOpGraph.h"  // from @llvm-project
#include "re2/re2.h"  // IWYU pragma: keep
#include "tensorflow/compiler/mlir/lite/debug/debug_options.pb.h"
#include "tensorflow/compiler/mlir/lite/metrics/error_collector_inst.h"
#include "xla/tsl/lib/io/buffered_file.h"
#include "tensorflow/core/platform/logging.h"
#include "tsl/platform/env.h"
#include "tsl/platform/file_system.h"
#include "tsl/platform/path.h"
#include "tsl/platform/stringpiece.h"

// IWYU pragma: no_include "util/regexp/re2/re2.h"

namespace tensorflow {
namespace {

// Simple raw_ostream that prints to a file.
struct WritableFileRawStream : public llvm::raw_ostream {
  explicit WritableFileRawStream(std::unique_ptr<tsl::WritableFile> file)
      : file(std::move(file)) {
    SetUnbuffered();
  }
  ~WritableFileRawStream() override = default;

  uint64_t current_pos() const override {
    int64_t position;
    if (file->Tell(&position).ok()) {
      return position;
    } else {
      // MLIR uses os.tell() to determine whether something was written by
      // a subroutine or not, so it's important we have a working current_pos().
      LOG(WARNING)
          << "Couldn't query file position. Stream might be malformed.\n";
      return -1;
    }
  }

  void write_impl(const char* ptr, size_t size) override {
    // Write the file if it is still valid. If the write fails, null out the
    // file to avoid encountering another error.
    if (file && !file->Append(absl::string_view(ptr, size)).ok()) {
      file = nullptr;
    }
  }

  // The file being written to.
  std::unique_ptr<tsl::WritableFile> file;
};

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
mlir::ReproducerStreamFactory GetReproducerStreamFactory(
    absl::string_view dump_dir) {
  std::string path = tsl::io::JoinPath(dump_dir, "tfl_mlir_crash_repro.mlir");

  return [path = std::move(path)](
             std::string& error) -> std::unique_ptr<mlir::ReproducerStream> {
    std::unique_ptr<tsl::WritableFile> file;
    if (auto status = tsl::Env::Default()->NewWritableFile(path, &file);
        !status.ok()) {
      error = status.ToString();
      absl::StrAppend(&error, "; failed to open '", path,
                      "' for writing an MLIR reproducer");
      return nullptr;
    }
    file = std::make_unique<tsl::BufferedWritableFile>(std::move(file));

    return std::make_unique<ReproducerStream>(
        path, std::make_unique<WritableFileRawStream>(std::move(file)));
  };
}

// Removes unwanted characters for readability and to eliminate issues when
// saving a file.
std::string Sanitize(absl::string_view string) {
  static const auto& kUnwantedChars = *new absl::flat_hash_set<char>{
      '<', '>', ':', '\"', '/', '\\', '|', '?', '*', ' ', '(', ')'};

  std::string sanitized;
  sanitized.reserve(string.size());

  bool skip = false;
  for (const char& c : string) {
    if (auto it = kUnwantedChars.find(c); it != kUnwantedChars.end()) {
      skip = true;
      continue;
    }
    if (skip) {
      skip = false;
      sanitized.push_back('_');
    }
    sanitized.push_back(c);
  }

  return sanitized;
}

// Pass instrumentation that dumps MLIR based on the criteria specified by
// `ir_dump_*` debug options.
//
// While `mlir::PassManager::enableIRPrinting` provides a similar functionality,
// it is cumbersome to manually copy printed IRs and run them with `tf-opt`.
// Also, long MLIR dumps are often truncated during printing. Instead, this
// instrumentation dumps MLIR to external directories for convenience.
class DumpInstrumentation : public mlir::PassInstrumentation {
 public:
  explicit DumpInstrumentation(absl::string_view dump_dir,
                               absl::string_view dump_pass_regex,
                               absl::string_view dump_func_regex)
      : dump_dir_(dump_dir),
        dump_pass_re_(std::make_unique<RE2>(dump_pass_regex)),
        dump_func_re_(std::make_unique<RE2>(dump_func_regex)) {}

  DumpInstrumentation(const DumpInstrumentation& other) = delete;
  DumpInstrumentation& operator=(const DumpInstrumentation& other) = delete;

  void runBeforePass(mlir::Pass* pass, mlir::Operation* op) override {
    // Always print before the first pass.
    if (!printed_) {
      Dump("before_all", op);
      printed_ = true;
    }

    if (RE2::FullMatch(pass->getName(), *dump_pass_re_)) {
      Dump(absl::StrCat(absl::string_view(pass->getName()), "_before"), op,
           absl::StrCat(absl::Hex(pass_counter_, absl::kZeroPad8)));
    }
  }

  void runAfterPass(mlir::Pass* pass, mlir::Operation* op) override {
    if (RE2::FullMatch(pass->getName(), *dump_pass_re_)) {
      Dump(absl::StrCat(absl::string_view(pass->getName()), "_after"), op,
           absl::StrCat(absl::Hex(pass_counter_++, absl::kZeroPad8)));
    }
  }

 private:
  // Dumps the given op. `name` is used as part of the filename to help
  // distinguish dumps at different passes.
  void Dump(absl::string_view name, mlir::Operation* op,
            std::string prefix = "") {
    static constexpr char kFiletypeSuffix[] = "mlir";
    // Find names of all func ops with public visibility and check whether at
    // least one of them matches `dump_func_re_`.
    llvm::SmallVector<absl::string_view> func_names;
    bool match = false;
    op->walk([&](mlir::func::FuncOp func) {
      if (func.isPublic()) {
        const absl::string_view name = func.getSymName();
        if (name.empty()) {
          return;
        }

        func_names.push_back(name);
        if (RE2::FullMatch(name, *dump_func_re_)) {
          match = true;
        }
      }
    });
    if (!func_names.empty() && !match) {
      return;
    }

    // Sort the function names for determinism.
    llvm::sort(func_names);

    std::string joined_func_names = Sanitize(absl::StrJoin(func_names, "-"));
    std::string sanitized_name = Sanitize(name);

    std::vector<absl::string_view> name_parts;
    if (!prefix.empty()) {
      name_parts.emplace_back(prefix);
    }
    if (!joined_func_names.empty()) {
      name_parts.emplace_back(joined_func_names);
    }
    name_parts.emplace_back(sanitized_name);
    name_parts.emplace_back(kFiletypeSuffix);

    // Build a filename such that it contains function names and pass names for
    // easy disambiguation.
    const std::string filename = tsl::io::JoinPath(
        dump_dir_, absl::StrJoin(name_parts.begin(), name_parts.end(), "."));

    // Open the file for dumping. Failures are logged instead of being
    // propagated to the client because they are non-fatal.

    std::unique_ptr<tsl::WritableFile> file;
    if (auto status = tsl::Env::Default()->NewWritableFile(filename, &file);
        !status.ok()) {
      LOG(ERROR) << "Unable to open '" << filename
                 << "' for dumping TFLite MLIR output: " << status;
      return;
    }
    file = std::make_unique<tsl::BufferedWritableFile>(std::move(file));

    WritableFileRawStream os(std::move(file));
    op->print(os);
  }

  const std::string dump_dir_;
  const std::unique_ptr<RE2> dump_pass_re_;
  const std::unique_ptr<RE2> dump_func_re_;

  // Counter used for pass name prefix to signify sequence
  int pass_counter_ = 0;

  bool printed_ = false;
};

std::function<bool(mlir::Pass*, mlir::Operation*)> CreatePrintIRFun(
    const std::string& pass_regex) {
  std::function<bool(mlir::Pass*, mlir::Operation*)> fun;
  if (pass_regex.empty()) {
    return fun;
  }
  return [pr = pass_regex](mlir::Pass* p, mlir::Operation*) {
    static const RE2* const re = new RE2(pr);
    if (RE2::FullMatch(p->getName(), *re)) {
      return true;
    }
    return false;
  };
}

}  // namespace

void InitPassManager(mlir::PassManager& pm,
                     const converter::DebugOptions& options,
                     llvm::raw_ostream& out) {
  std::string dump_dir = options.ir_dump_dir();

  bool dump_to_dir = !dump_dir.empty();
  bool print_to_stdout =
      !options.print_ir_before().empty() || !options.print_ir_after().empty();

  if (dump_to_dir || print_to_stdout) {
    // Necessary for maintaining sequence of passes when dumping MLIR to files
    // or stdout.
    pm.getContext()->disableMultithreading();
  }

  if (dump_to_dir) {
    dump_dir = tsl::io::JoinPath(
        dump_dir, absl::FormatTime("%E4Y%m%d_%H%M%E6S", absl::Now(),
                                   absl::LocalTimeZone()));

    // Get a valid file path to dump with.
    tsl::Env* env = tsl::Env::Default();
    if (auto status = env->RecursivelyCreateDir(dump_dir); !status.ok()) {
      LOG(WARNING) << "Failed to create '" << dump_dir
                   << "' directory for dumping: " << status;
      return;
    }

    // Set a default crash reproducer for easier debugging.
    if (auto reproducer_stream_factory = GetReproducerStreamFactory(dump_dir)) {
      pm.enableCrashReproducerGeneration(std::move(reproducer_stream_factory));
    }

    pm.addInstrumentation(std::make_unique<DumpInstrumentation>(
        dump_dir, options.ir_dump_pass_regex(), options.ir_dump_func_regex()));
  }

  if (print_to_stdout) {
    std::function<bool(mlir::Pass*, mlir::Operation*)>
        should_print_ir_before_pass(
            CreatePrintIRFun(options.print_ir_before()));
    std::function<bool(mlir::Pass*, mlir::Operation*)>
        should_print_ir_after_pass(CreatePrintIRFun(options.print_ir_after()));

    mlir::OpPrintingFlags opPrintingFlags = mlir::OpPrintingFlags();

    if (options.has_elide_elementsattrs_if_larger()) {
      opPrintingFlags.elideLargeElementsAttrs(
          options.elide_elementsattrs_if_larger());
    }

    pm.enableIRPrinting(should_print_ir_before_pass, should_print_ir_after_pass,
                        options.print_ir_module_scope(),
                        /*printAfterOnlyOnChange=*/true,
                        /*printAfterOnlyOnFailure=*/false, out,
                        opPrintingFlags);
  }

  // Enable pass timing. Note: MLIR expects `mlir::PassManager::enableTiming` to
  // be called after all instrumentations are added.
  if (options.enable_timing()) {
    pm.enableTiming();
  }

  pm.addInstrumentation(
      std::make_unique<mlir::TFL::ErrorCollectorInstrumentation>(
          pm.getContext()));
}

}  // namespace tensorflow
