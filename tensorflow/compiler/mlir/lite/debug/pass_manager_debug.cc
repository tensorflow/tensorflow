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

#include "tensorflow/compiler/mlir/lite/debug/pass_manager_debug.h"

#include <stddef.h>
#include <stdint.h>
#if defined(__linux__)
#include <unistd.h>
#endif

#include <fstream>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Regex.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/OperationSupport.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassInstrumentation.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/debug/debug_options.pb.h"
#include "tensorflow/compiler/mlir/lite/metrics/error_collector_inst.h"
#include "tensorflow/compiler/mlir/lite/utils/file_utils.h"

// Derived from debug.cc; keep the two in sync. Intentional differences: files
// are written with the local file system API, write errors are logged instead
// of being reported by TSL, each print-IR regex is owned by its closure, and
// regexes are `llvm::Regex` (POSIX extended syntax) instead of RE2.
// TODO: b/584436432 - Share the implementation with debug.cc once the
// TensorFlow path no longer needs TSL file systems.

namespace mlir::TFL {
namespace {

// A regex that must match a whole string, like `RE2::FullMatch`.
class FullMatchRegex {
 public:
  explicit FullMatchRegex(absl::string_view pattern)
      : regex_(absl::StrCat("^(", pattern, ")$")) {}

  bool Matches(llvm::StringRef str) const { return regex_.match(str); }

 private:
  llvm::Regex regex_;
};

// Logs and clears a pending error on `os`, which LLVM would otherwise treat as
// fatal when `os` is destroyed.
void CloseAndLogErrors(llvm::raw_fd_ostream& os, absl::string_view path) {
  os.close();
  if (os.has_error()) {
    LOG(ERROR) << "Failed to write '" << path << "': " << os.error().message();
    os.clear_error();
  }
}

// Reproducer stream that emits a reproducer to a file.
class ReproducerStream : public mlir::ReproducerStream {
 public:
  ReproducerStream(std::string name, std::unique_ptr<llvm::raw_fd_ostream> os)
      : name_(std::move(name)), os_(std::move(os)) {}

  ~ReproducerStream() override { CloseAndLogErrors(*os_, name_); }

  llvm::StringRef description() override { return name_; }

  llvm::raw_ostream& os() override { return *os_; }

 private:
  std::string name_;
  std::unique_ptr<llvm::raw_fd_ostream> os_;
};

// Returns a function that builds a reproducer stream, or nullptr if the MLIR
// reproducer will not be enabled.
mlir::ReproducerStreamFactory GetReproducerStreamFactory(
    absl::string_view dump_dir) {
  std::string path = JoinPath(dump_dir, "tfl_mlir_crash_repro.mlir");

  return [path = std::move(path)](
             std::string& error) -> std::unique_ptr<mlir::ReproducerStream> {
    absl::StatusOr<std::unique_ptr<llvm::raw_fd_ostream>> file =
        OpenFileForWriting(path);
    if (!file.ok()) {
      error = file.status().ToString();
      absl::StrAppend(&error, "; failed to open '", path,
                      "' for writing an MLIR reproducer");
      return nullptr;
    }
    return std::make_unique<ReproducerStream>(path, *std::move(file));
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
// it is cumbersome to manually copy printed IRs and run them with `litert-opt`.
// Also, long MLIR dumps are often truncated during printing. Instead, this
// instrumentation dumps MLIR to external directories for convenience.
class DumpInstrumentation : public mlir::PassInstrumentation {
 public:
  explicit DumpInstrumentation(
      absl::string_view dump_dir, absl::string_view dump_pass_regex,
      absl::string_view dump_func_regex,
      mlir::OpPrintingFlags flags = mlir::OpPrintingFlags())
      : dump_dir_(dump_dir),
        dump_pass_re_(dump_pass_regex),
        dump_func_re_(dump_func_regex),
        op_printing_flags_(flags) {}

  DumpInstrumentation(const DumpInstrumentation& other) = delete;
  DumpInstrumentation& operator=(const DumpInstrumentation& other) = delete;

  void runBeforePass(mlir::Pass* pass, mlir::Operation* op) override {
    // Always print before the first pass.
    if (!printed_) {
      Dump("before_all", op);
      printed_ = true;
    }

    if (dump_pass_re_.Matches(pass->getName())) {
      Dump(absl::StrCat(absl::string_view(pass->getName()), "_before"), op,
           absl::StrCat(absl::Hex(pass_counter_, absl::kZeroPad8)));
    }
  }

  void runAfterPass(mlir::Pass* pass, mlir::Operation* op) override {
    if (dump_pass_re_.Matches(pass->getName())) {
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
        if (dump_func_re_.Matches(name)) {
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
    const std::string filename = JoinPath(
        dump_dir_, absl::StrJoin(name_parts.begin(), name_parts.end(), "."));

    // Open the file for dumping. Failures are logged instead of being
    // propagated to the client because they are non-fatal.
    absl::StatusOr<std::unique_ptr<llvm::raw_fd_ostream>> file =
        OpenFileForWriting(filename);
    if (!file.ok()) {
      LOG(ERROR) << "Unable to open '" << filename
                 << "' for dumping TFLite MLIR output: " << file.status();
      return;
    }
    op->print(**file, op_printing_flags_);
    CloseAndLogErrors(**file, filename);
  }

  const std::string dump_dir_;
  const FullMatchRegex dump_pass_re_;
  const FullMatchRegex dump_func_re_;
  mlir::OpPrintingFlags op_printing_flags_;

  // Counter used for pass name prefix to signify sequence
  int pass_counter_ = 0;

  bool printed_ = false;
};

double GetCurrentRssMb() {
#if defined(__linux__)
  std::ifstream statm("/proc/self/statm");
  if (!statm.is_open()) return 0.0;
  int64_t pages = 0;
  int64_t rss_pages = 0;
  if (statm >> pages >> rss_pages) {
    int64_t page_size = sysconf(_SC_PAGESIZE);
    return static_cast<double>(rss_pages * page_size) / (1024.0 * 1024.0);
  }
#endif
  return 0.0;
}

class RssLoggingInstrumentation : public mlir::PassInstrumentation {
 public:
  void runBeforePass(mlir::Pass* pass, mlir::Operation* op) override {
    pass_start_rss_[pass] = GetCurrentRssMb();
  }

  void runAfterPass(mlir::Pass* pass, mlir::Operation* op) override {
    double end_rss = GetCurrentRssMb();
    auto node = pass_start_rss_.extract(pass);
    double start_rss = node.empty() ? 0.0 : node.mapped();
    double delta = end_rss - start_rss;
    LOG(INFO) << "[MLIR RSS] After pass '" << pass->getName().str() << "' on '"
              << op->getName().getStringRef().str() << "': RSS = " << end_rss
              << " MB (Delta: " << (delta >= 0 ? "+" : "") << delta << " MB)";
  }

 private:
  absl::flat_hash_map<mlir::Pass*, double> pass_start_rss_;
};

std::function<bool(mlir::Pass*, mlir::Operation*)> CreatePrintIRFun(
    const std::string& pass_regex) {
  std::function<bool(mlir::Pass*, mlir::Operation*)> fun;
  if (pass_regex.empty()) {
    return fun;
  }
  // Each closure owns its regex, so that the before and after regexes (and
  // those of later conversions) do not share one.
  auto re = std::make_shared<const FullMatchRegex>(pass_regex);
  return [re = std::move(re)](mlir::Pass* p, mlir::Operation*) {
    return re->Matches(p->getName());
  };
}

}  // namespace

void InitPassManager(mlir::PassManager& pm,
                     const tensorflow::converter::DebugOptions& options,
                     llvm::raw_ostream& out) {
  std::string dump_dir = options.ir_dump_dir();

  bool dump_to_dir = !dump_dir.empty();
  bool print_to_stdout =
      !options.print_ir_before().empty() || !options.print_ir_after().empty();

  if (dump_to_dir || print_to_stdout || options.log_rss()) {
    // Necessary for maintaining sequence of passes when dumping MLIR to files,
    // stdout, or logging RSS memory.
    pm.getContext()->disableMultithreading();
  }

  mlir::OpPrintingFlags opPrintingFlags = mlir::OpPrintingFlags();
  if (options.has_elide_elementsattrs_if_larger()) {
    opPrintingFlags.elideLargeElementsAttrs(
        options.elide_elementsattrs_if_larger());
    opPrintingFlags.elideLargeResourceString(
        options.elide_elementsattrs_if_larger());
  }

  if (dump_to_dir) {
    dump_dir =
        JoinPath(dump_dir, absl::FormatTime("%E4Y%m%d_%H%M%E6S", absl::Now(),
                                            absl::LocalTimeZone()));

    // Get a valid file path to dump with.
    if (auto status = CreateDirectories(dump_dir); !status.ok()) {
      LOG(WARNING) << "Failed to create '" << dump_dir
                   << "' directory for dumping: " << status;
      return;
    }

    // Set a default crash reproducer for easier debugging.
    if (auto reproducer_stream_factory = GetReproducerStreamFactory(dump_dir)) {
      pm.enableCrashReproducerGeneration(std::move(reproducer_stream_factory));
    }

    pm.addInstrumentation(std::make_unique<DumpInstrumentation>(
        dump_dir, options.ir_dump_pass_regex(), options.ir_dump_func_regex(),
        opPrintingFlags));
  }

  if (print_to_stdout) {
    std::function<bool(mlir::Pass*, mlir::Operation*)>
        should_print_ir_before_pass(
            CreatePrintIRFun(options.print_ir_before()));
    std::function<bool(mlir::Pass*, mlir::Operation*)>
        should_print_ir_after_pass(CreatePrintIRFun(options.print_ir_after()));

    pm.enableIRPrinting(should_print_ir_before_pass, should_print_ir_after_pass,
                        options.print_ir_module_scope(),
                        /*printAfterOnlyOnChange=*/true,
                        /*printAfterOnlyOnFailure=*/false, out,
                        opPrintingFlags);
  }

  if (options.log_rss()) {
    pm.addInstrumentation(std::make_unique<RssLoggingInstrumentation>());
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

}  // namespace mlir::TFL
