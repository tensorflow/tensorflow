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

#include "tensorflow/compiler/mlir/tensorflow/utils/dump_graph.h"

#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "mlir/Analysis/Verifier.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/import_model.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/error_util.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/path.h"

namespace tensorflow {

namespace {

// TODO(jpienaar): With this refactoring, I think these utility functions could
// be used to dedup the ~3 places where we have pretty much this same structure.
struct NameCounts {
  mutex counts_mutex;
  std::unordered_map<string, int> counts;
};

string MakeUniqueFilename(const string& name, const string& suffix) {
  static NameCounts& instance = *new NameCounts;
  string filename = name;

  // Remove illegal characters from `name`.
  for (int i = 0; i < filename.size(); ++i) {
    char ch = filename[i];
    if (ch == '/' || ch == '[' || ch == ']' || ch == '*' || ch == '?' ||
        ch == '\\') {
      filename[i] = '_';
    }
  }

  int count;
  {
    mutex_lock lock(instance.counts_mutex);
    count = instance.counts[filename]++;
  }

  if (count > 0) {
    absl::StrAppend(&filename, "_", count);
  }
  absl::StrAppend(&filename, suffix);
  return filename;
}

Status GetDirAndFilepath(Env* env, const string& dirname, const string& name,
                         const string& suffix, string* dir, string* filepath) {
  if (!dirname.empty()) {
    *dir = dirname;
  } else {
    const char* prefix = getenv("TF_DUMP_GRAPH_PREFIX");
    if (prefix != nullptr) *dir = prefix;
  }
  if (dir->empty()) {
    LOG(WARNING)
        << "Failed to dump " << name << " because dump location is not "
        << " specified through either TF_DUMP_GRAPH_PREFIX environment "
        << "variable or function argument.";
    return errors::InvalidArgument("TF_DUMP_GRAPH_PREFIX not specified");
  }

  if (absl::EqualsIgnoreCase(*dir, "sponge") ||
      absl::EqualsIgnoreCase(*dir, "test_undeclared_outputs_dir")) {
    if (!io::GetTestUndeclaredOutputsDir(dir)) {
      LOG(WARNING) << "TF_DUMP_GRAPH_PREFIX=sponge, but "
                      "TEST_UNDECLARED_OUTPUT_DIRS is not set, dumping to log";
      *dir = "-";
    }
  }

  *filepath = "NULL";
  if (*dir == "-") {
    *filepath = "LOG(INFO)";
  } else {
    TF_RETURN_WITH_CONTEXT_IF_ERROR(env->RecursivelyCreateDir(*dir),
                                    "Failed to create ", *dir);
  }
  *filepath = io::JoinPath(*dir, MakeUniqueFilename(name, suffix));
  return Status::OK();
}

string WriteTextToUniqueFile(Env* env, const string& name, const string& str,
                             const string& dirname) {
  string dir;
  string filepath;
  Status status =
      GetDirAndFilepath(env, dirname, name, ".mlir", &dir, &filepath);
  if (!status.ok()) {
    LOG(WARNING) << "Failed to get dir and filepath: " << status;
    return "(unavailable)";
  }
  if (dir == "-") {
    LOG(INFO) << str;
  } else {
    status = WriteStringToFile(env, filepath, str);
    if (!status.ok()) {
      LOG(WARNING) << "Failed to dump to file: " << filepath << " : " << status;
      return "(unavailable)";
    }
  }
  LOG(INFO) << "Dumped to " << filepath;
  return filepath;
}

}  // anonymous namespace

string DumpTextualIRToFile(const MlirDumpConfig& config, const Graph& graph,
                           const FunctionLibraryDefinition* flib_def) {
  mlir::MLIRContext context;
  mlir::OwningModuleRef module;
  if (flib_def) {
    flib_def = &graph.flib_def();
  }
  auto convert = [&]() -> Status {
    mlir::StatusScopedDiagnosticHandler status_handler(&context);
    // TODO(jpienaar): Both the graph debug info and import config should be
    // specifiable.
    TF_ASSIGN_OR_RETURN(
        module, ConvertGraphToMlir(graph, GraphDebugInfo(),
                                   flib_def ? *flib_def : graph.flib_def(),
                                   GraphImportConfig(), &context));
    if (VLOG_IS_ON(1) && failed(mlir::verify(*module))) {
      LOG(ERROR) << "Failed to verify module";
      module->dump();
    }
    if (config.run_standard_pipeline) {
      mlir::PassManager pm(&context);
      mlir::TF::StandardPipelineOptions pipeline_options;
      mlir::TF::CreateTFStandardPipeline(pm, pipeline_options);
      (void)pm.run(module.get());
    }
    return status_handler.ConsumeStatus();
  };

  if (convert().ok()) {
    string str;
    llvm::raw_string_ostream os(str);
    module->print(os, config.op_printing_flags);
    return WriteTextToUniqueFile(Env::Default(), config.name, os.str(),
                                 config.dirname);
  }
  return "(unable)";
}

}  // namespace tensorflow
