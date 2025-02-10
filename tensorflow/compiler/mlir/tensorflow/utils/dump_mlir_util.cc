/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/mlir/tensorflow/utils/dump_mlir_util.h"

#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <utility>

#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "xla/tsl/lib/io/buffered_file.h"
#include "tensorflow/core/platform/crash_analysis.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/path.h"

using llvm::raw_ostream;

namespace tensorflow {
namespace {

struct NameCounts {
  mutex counts_mutex;
  llvm::StringMap<int64_t> counts;
};

std::string MakeUniqueFilename(string name) {
  static NameCounts& instance = *new NameCounts;

  // Remove illegal characters from `name`.
  for (int i = 0, e = name.size(); i < e; ++i) {
    char ch = name[i];
    if (ch == '/' || ch == '[' || ch == ']' || ch == '*' || ch == '?' ||
        ch == '\\') {
      name[i] = '_';
    }
  }

  int count;
  {
    mutex_lock lock(instance.counts_mutex);
    count = instance.counts[name]++;
  }

  std::string filename = name;
  if (count > 0) {
    filename = llvm::formatv("{0}_{1}", filename, count).str();
  }
  filename = llvm::Twine(filename).concat(".mlir").str();
  return filename;
}

// Simple raw_ostream that prints to stderr.
struct LogInfoRawStream : public llvm::raw_ostream {
  LogInfoRawStream() { SetUnbuffered(); }
  ~LogInfoRawStream() override = default;
  uint64_t current_pos() const override { return 0; }

  void write_impl(const char* ptr, size_t size) override {
    fprintf(stderr, "%.*s", static_cast<int>(size), ptr);
  }
};

// Simple raw_ostream that prints to a file.
struct WritableFileRawStream : public llvm::raw_ostream {
  explicit WritableFileRawStream(std::unique_ptr<WritableFile> file)
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
  std::unique_ptr<WritableFile> file;
};

struct CrashReproducerStream : public mlir::ReproducerStream {
  CrashReproducerStream(llvm::StringRef name,
                        std::unique_ptr<llvm::raw_ostream> file)
      : name(name), ostream(std::move(file)) {}

  llvm::StringRef description() override { return name; }
  raw_ostream& os() override { return *ostream; }

 private:
  std::string name;
  std::unique_ptr<llvm::raw_ostream> ostream;
};

// MLIR crash reproducer which reports failures to the crash analysis system.
struct CrashAnalysisCrashReproducerStream : public mlir::ReproducerStream {
 public:
  CrashAnalysisCrashReproducerStream()
      : internal_str(""), string_stream(internal_str) {}

  ~CrashAnalysisCrashReproducerStream() override {
    crash_analysis::ReportEvent(
        "mlir_crash_reproducer.mlir",
        "Pass pipeline failure; crash reproducer attached",
        string_stream.str());
  }

  llvm::StringRef description() override { return "mlir_crash_reproducer"; }
  raw_ostream& os() override { return string_stream; }

 private:
  std::string internal_str;
  llvm::raw_string_ostream string_stream;
};

}  // namespace

absl::Status CreateFileForDumping(llvm::StringRef name,
                                  std::unique_ptr<raw_ostream>* os,
                                  std::string* filepath,
                                  llvm::StringRef dirname) {
  std::string dir;
  if (!dirname.empty())
    dir = std::string(dirname);
  else
    dir = GetDumpDirFromEnvVar();

  if (dir.empty()) {
    return absl::Status(absl::StatusCode::kInvalidArgument,
                        "(TF_DUMP_GRAPH_PREFIX not specified)");
  }

  if (dir == kCrashReproducerStdErr) {
    *os = std::make_unique<LogInfoRawStream>();
    *filepath =
        llvm::formatv("(stderr; requested filename: '{0}')", name).str();
    return absl::Status();
  }

  // Get a valid file path to dump with.
  Env* env = Env::Default();
  absl::Status status = env->RecursivelyCreateDir(dir);
  if (!status.ok()) {
    LOG(WARNING) << "Failed to create '" << dir
                 << "' directory for dumping: " << status;
    return absl::Status(absl::StatusCode::kUnavailable, "(unavailable)");
  }
  *filepath = io::JoinPath(dir, MakeUniqueFilename(std::string(name)));

  // Try to open the file and generate a raw_ostream.
  std::unique_ptr<WritableFile> file;
  status = env->NewWritableFile(*filepath, &file);
  if (!status.ok()) {
    LOG(WARNING) << "Failed to create file '" << filepath << "': " << status;
    return absl::Status(absl::StatusCode::kUnavailable, "(unavailable)");
  }
  file = std::make_unique<tsl::BufferedWritableFile>(std::move(file));
  *os = std::make_unique<WritableFileRawStream>(std::move(file));
  return absl::Status();
}

// Prints the pass pipeline of `pass_manager` to `os`.
void PrintPassPipeline(const mlir::PassManager& pass_manager,
                       mlir::Operation* op, llvm::raw_ostream& os) {
  std::string str;
  llvm::raw_string_ostream passOS(str);
  llvm::interleaveComma(
      pass_manager.getPasses(), passOS,
      [&](mlir::Pass& pass) { pass.printAsTextualPipeline(passOS); });
  os << "{-# external_resources: { mlir_reproducer: { pipeline: "
        "\"builtin.module("
     << passOS.str() << ")\", ";
  os << "disable_threading: true, ";
  os << "verify_each: true } } #-}";
  os << "\n\n";
}

std::string DumpMlirOpToFile(llvm::StringRef name, mlir::Operation* op,
                             llvm::StringRef dirname,
                             const mlir::PassManager* pass_manager) {
  std::unique_ptr<raw_ostream> os;
  std::string filepath;
  absl::Status result = CreateFileForDumping(name, &os, &filepath, dirname);
  if (!result.ok()) return std::string(result.message());

  LOG(INFO) << "Dumping MLIR operation '" << op->getName().getStringRef().str()
            << "' to '" << filepath << "'";
  if (pass_manager) PrintPassPipeline(*pass_manager, op, *os);
  op->print(*os, mlir::OpPrintingFlags().useLocalScope());
  return filepath;
}

std::string GetDumpDirFromEnvVar() {
  const char* prefix_env = getenv("TF_DUMP_GRAPH_PREFIX");
  if (!prefix_env) {
    LOG(WARNING)
        << "Failed to dump MLIR module because dump location is not "
        << "specified through TF_DUMP_GRAPH_PREFIX environment variable.";
    return "";
  }

  std::string result = prefix_env;

  if (absl::EqualsIgnoreCase(result, "sponge") &&
      !io::GetTestUndeclaredOutputsDir(&result)) {
    LOG(WARNING) << "TF_DUMP_GRAPH_PREFIX=sponge but "
                    "TEST_UNDECLARED_OUTPUT_DIRS is not set";
    return "";
  }
  return result;
}

std::string DumpRawStringToFile(llvm::StringRef name, llvm::StringRef content,
                                llvm::StringRef dirname) {
  std::unique_ptr<raw_ostream> os;
  std::string filepath;
  absl::Status result = CreateFileForDumping(name, &os, &filepath, dirname);
  if (!result.ok()) return std::string(result.message());

  (*os) << content;
  LOG(INFO) << "Outputted requested string to '" << filepath << "'";
  return filepath;
}

void SetCrashReproducer(mlir::PassManager& pm, llvm::StringRef dir_path) {
  std::string path = dir_path.str();
  if (path.empty() || path == kCrashReproducerCrashAnalysis) {
    if (getenv("MLIR_CRASH_REPRODUCER_DIRECTORY"))
      path = getenv("MLIR_CRASH_REPRODUCER_DIRECTORY");
    else if (getenv("TEST_UNDECLARED_OUTPUTS_DIR"))
      path = "sponge";
  }
  if (path.empty()) {
    LOG_FIRST_N(INFO, 1) << "disabling MLIR crash reproducer, set env var "
                            "`MLIR_CRASH_REPRODUCER_DIRECTORY` to enable.";
    return;
  }

  // Output dirs "sponge" (case-insensitive) have a special meaning: Dump into
  // the directory specified by the environment variable
  // TEST_UNDECLARED_OUTPUTS_DIR.
  string lower_path = absl::AsciiStrToLower(path);
  if (lower_path == "sponge") {
    if (!tensorflow::io::GetTestUndeclaredOutputsDir(&path)) {
      LOG(ERROR) << "MLIR crash reproducer is set to '" << dir_path.str()
                 << "', but environment variable TEST_UNDECLARED_OUTPUTS_DIR "
                    "is not set, so cannot dump anywhere.";
      return;
    }
  }

  // kCrashReproducerStdErr and kCrashReproducerCrashAnalysis settings do not
  // require explicit file creation.
  if (path != kCrashReproducerStdErr && path != kCrashReproducerCrashAnalysis) {
    auto* env = tensorflow::Env::Default();
    auto status = env->RecursivelyCreateDir(path);
    if (!status.ok()) {
      LOG(WARNING) << "cannot create directory '" << path
                   << "': " << status.message();
      return;
    }

    path += "/mlir_reproducer_";

    if (!tensorflow::Env::Default()->CreateUniqueFileName(&path, ".mlir")) {
      LOG(WARNING) << "cannot create unique filename, won't enable MLIR crash "
                      "reproducer.";
      return;
    }
  }

  mlir::ReproducerStreamFactory factory =
      [path](std::string& error) -> std::unique_ptr<mlir::ReproducerStream> {
    if (path == kCrashReproducerStdErr)
      return std::make_unique<CrashReproducerStream>(
          "(stderr)", std::make_unique<LogInfoRawStream>());
    if (path == kCrashReproducerCrashAnalysis) {
      return std::make_unique<CrashAnalysisCrashReproducerStream>();
    }

    // Try to open the file and generate a raw_ostream.
    std::unique_ptr<WritableFile> file;
    absl::Status status =
        tensorflow::Env::Default()->NewWritableFile(path, &file);
    file = std::make_unique<tsl::BufferedWritableFile>(std::move(file));

    if (!status.ok()) {
      error = absl::StrCat("Failed to create file '", path,
                           "': ", status.message());
      return nullptr;
    }
    return std::make_unique<CrashReproducerStream>(
        path, std::make_unique<WritableFileRawStream>(std::move(file)));
  };
  pm.enableCrashReproducerGeneration(factory, /*genLocalReproducer=*/false);
}

void applyTensorflowAndCLOptions(mlir::PassManager& pm,
                                 llvm::StringRef dir_path) {
  mlir::registerPassManagerCLOptions();
  if (!mlir::succeeded(mlir::applyPassManagerCLOptions(pm))) {
    LOG(ERROR) << "cannot apply MLIR pass manager CL options";
    return;
  }
  SetCrashReproducer(pm, dir_path);
}

}  // namespace tensorflow
