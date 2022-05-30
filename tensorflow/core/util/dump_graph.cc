/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

// Helper functions for dumping Graphs, GraphDefs, and FunctionDefs to files for
// debugging.

#include "tensorflow/core/util/dump_graph.h"

#include <memory>
#include <unordered_map>

#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "tensorflow/core/lib/strings/proto_serialization.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/file_system.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/path.h"
#include "tensorflow/core/platform/strcat.h"

namespace tensorflow {

namespace {
using strings::StrCat;

struct NameCounts {
  mutex counts_mutex;
  std::unordered_map<string, int> counts;
};

string MakeUniqueFilename(string name, const string& suffix = ".pbtxt") {
  static NameCounts& instance = *new NameCounts;

  // Remove illegal characters from `name`.
  for (int i = 0; i < name.size(); ++i) {
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

  string filename = name;
  if (count > 0) {
    absl::StrAppend(&filename, "_", count);
  }
  absl::StrAppend(&filename, suffix);
  return filename;
}

struct GraphDumperConfig {
  mutex mu;

  // The dumper and suffix configured.
  struct Config {
    bool IsSet() const { return dumper != nullptr; }
    std::function<Status(const Graph& graph,
                         const FunctionLibraryDefinition* flib_def,
                         WritableFile*)>
        dumper = nullptr;
    string suffix = ".pbtxt";
  } config TF_GUARDED_BY(mu);

  // Returns whether a custom dumper is set.
  bool IsSet() TF_LOCKS_EXCLUDED(mu) {
    mutex_lock lock(mu);
    return config.IsSet();
  }
};

GraphDumperConfig& GetGraphDumperConfig() {
  static GraphDumperConfig config;
  return config;
}

// WritableFile that simply prints to stderr.
class StderrWritableFile : public WritableFile {
 public:
  StderrWritableFile() {}

  Status Append(StringPiece data) override {
    fprintf(stderr, "%.*s", static_cast<int>(data.size()), data.data());
    return Status::OK();
  }

  Status Close() override { return Status::OK(); }

  Status Flush() override {
    fflush(stderr);
    return Status::OK();
  }

  Status Name(StringPiece* result) const override {
    *result = "stderr";
    return Status::OK();
  }

  Status Sync() override { return Status::OK(); }

  Status Tell(int64_t* position) override {
    return errors::Unimplemented("Stream not seekable");
  }
};

Status CreateWritableFile(Env* env, const string& dirname, const string& name,
                          const string& suffix, string* filepath,
                          std::unique_ptr<WritableFile>* file) {
  string dir;
  if (!dirname.empty()) {
    dir = dirname;
  } else {
    const char* prefix = getenv("TF_DUMP_GRAPH_PREFIX");
    if (prefix != nullptr) dir = prefix;
  }
  if (dir.empty()) {
    LOG(WARNING)
        << "Failed to dump " << name << " because dump location is not "
        << " specified through either TF_DUMP_GRAPH_PREFIX environment "
        << "variable or function argument.";
    return errors::InvalidArgument("TF_DUMP_GRAPH_PREFIX not specified");
  }

  if (absl::EqualsIgnoreCase(dir, "sponge") ||
      absl::EqualsIgnoreCase(dir, "test_undeclared_outputs_dir")) {
    if (!io::GetTestUndeclaredOutputsDir(&dir)) {
      LOG(WARNING) << "TF_DUMP_GRAPH_PREFIX=sponge, but "
                      "TEST_UNDECLARED_OUTPUT_DIRS is not set, dumping to log";
      dir = "-";
    }
  }

  *filepath = "NULL";
  if (dir == "-") {
    *file = std::make_unique<StderrWritableFile>();
    *filepath = "(stderr)";
    return Status::OK();
  }

  TF_RETURN_IF_ERROR(env->RecursivelyCreateDir(dir));
  *filepath = io::JoinPath(dir, MakeUniqueFilename(name, suffix));
  return env->NewWritableFile(*filepath, file);
}

Status WriteTextProtoToUniqueFile(const tensorflow::protobuf::Message& proto,
                                  WritableFile* file) {
  string s;
  if (!::tensorflow::protobuf::TextFormat::PrintToString(proto, &s)) {
    return errors::FailedPrecondition("Unable to convert proto to text.");
  }
  TF_RETURN_IF_ERROR(file->Append(s));
  StringPiece name;
  TF_RETURN_IF_ERROR(file->Name(&name));
  VLOG(5) << name;
  VLOG(5) << s;
  return file->Close();
}

Status WriteTextProtoToUniqueFile(
    const tensorflow::protobuf::MessageLite& proto, WritableFile* file) {
  string s;
  if (!SerializeToStringDeterministic(proto, &s)) {
    return errors::Internal("Failed to serialize proto to string.");
  }
  StringPiece name;
  TF_RETURN_IF_ERROR(file->Name(&name));
  VLOG(5) << name;
  VLOG(5) << s;
  TF_RETURN_IF_ERROR(file->Append(s));
  return file->Close();
}

}  // anonymous namespace

void SetGraphDumper(
    std::function<Status(const Graph& graph,
                         const FunctionLibraryDefinition* flib_def,
                         WritableFile*)>
        dumper,
    string suffix) {
  GraphDumperConfig& dumper_config = GetGraphDumperConfig();
  mutex_lock lock(dumper_config.mu);
  dumper_config.config.dumper = dumper;
  dumper_config.config.suffix = suffix;
}

string DumpGraphDefToFile(const string& name, GraphDef const& graph_def,
                          const string& dirname) {
  string filepath;
  std::unique_ptr<WritableFile> file;
  Status status = CreateWritableFile(Env::Default(), dirname, name, ".pbtxt",
                                     &filepath, &file);
  if (!status.ok()) {
    return StrCat("(failed to create writable file: ", status.ToString(), ")");
  }

  status = WriteTextProtoToUniqueFile(graph_def, file.get());
  if (!status.ok()) {
    return StrCat("(failed to dump Graph to '", filepath,
                  "': ", status.ToString(), ")");
  }
  LOG(INFO) << "Dumped Graph to " << filepath;
  return filepath;
}

string DumpCostGraphDefToFile(const string& name, CostGraphDef const& graph_def,
                              const string& dirname) {
  string filepath;
  std::unique_ptr<WritableFile> file;
  Status status = CreateWritableFile(Env::Default(), dirname, name, ".pbtxt",
                                     &filepath, &file);
  if (!status.ok()) {
    return StrCat("(failed to create writable file: ", status.ToString(), ")");
  }

  status = WriteTextProtoToUniqueFile(graph_def, file.get());
  if (!status.ok()) {
    return StrCat("(failed to dump Graph to '", filepath,
                  "': ", status.ToString(), ")");
  }
  LOG(INFO) << "Dumped Graph to " << filepath;
  return filepath;
}

string DumpGraphToFile(const string& name, Graph const& graph,
                       const FunctionLibraryDefinition* flib_def,
                       const string& dirname) {
  auto& dumper_config = GetGraphDumperConfig();
  if (dumper_config.IsSet()) {
    GraphDumperConfig::Config config;
    {
      mutex_lock lock(dumper_config.mu);
      config = dumper_config.config;
    }
    if (config.IsSet()) {
      string filepath;
      std::unique_ptr<WritableFile> file;
      Status status = CreateWritableFile(Env::Default(), dirname, name,
                                         config.suffix, &filepath, &file);
      if (!status.ok()) {
        return StrCat("(failed to create writable file: ", status.ToString(),
                      ")");
      }
      status = config.dumper(graph, flib_def, file.get());
      if (!status.ok()) {
        return StrCat("(failed to dump Graph to '", filepath,
                      "': ", status.ToString(), ")");
      }
      LOG(INFO) << "Dumped Graph to " << filepath;
      return filepath;
    }
  }

  GraphDef graph_def;
  graph.ToGraphDef(&graph_def);
  if (flib_def) {
    *graph_def.mutable_library() = flib_def->ToProto();
  }
  return DumpGraphDefToFile(name, graph_def, dirname);
}

string DumpFunctionDefToFile(const string& name, FunctionDef const& fdef,
                             const string& dirname) {
  string filepath;
  std::unique_ptr<WritableFile> file;
  Status status = CreateWritableFile(Env::Default(), dirname, name, ".pbtxt",
                                     &filepath, &file);
  if (!status.ok()) {
    return StrCat("(failed to create writable file: ", status.ToString(), ")");
  }

  status = WriteTextProtoToUniqueFile(fdef, file.get());
  if (!status.ok()) {
    return StrCat("(failed to dump FunctionDef to '", filepath,
                  "': ", status.ToString(), ")");
  }
  LOG(INFO) << "Dumped FunctionDef to " << filepath;
  return filepath;
}

}  // namespace tensorflow
