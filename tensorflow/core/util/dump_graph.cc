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

#include <functional>
#include <memory>
#include <unordered_map>

#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/util/env_var.h"
#include "tensorflow/core/lib/strings/proto_serialization.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/file_system.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/path.h"

namespace tensorflow {

namespace {
using strings::StrCat;

struct NameCounts {
  mutex counts_mutex;
  std::unordered_map<std::string, int> counts;
};

std::string MakeUniqueFilename(std::string name,
                               const std::string& suffix = ".pbtxt") {
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

  std::string filename = name;
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
    std::function<absl::Status(const Graph& graph,
                               const FunctionLibraryDefinition* flib_def,
                               WritableFile*)>
        dumper = nullptr;
    std::string suffix = ".pbtxt";
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

std::string GetDumpGraphFormatLowerCase() {
  std::string fmt;
  absl::Status status =
      tsl::ReadStringFromEnvVar("TF_DUMP_GRAPH_FMT", "TXT", &fmt);
  if (!status.ok()) {
    LOG(WARNING) << "Failed to read TF_DUMP_GRAPH_FMT: " << status;
    return "txt";
  }
  fmt = absl::AsciiStrToLower(fmt);
  return fmt;
}

std::string GetDumpGraphSuffix() {
  std::string fmt = GetDumpGraphFormatLowerCase();
  if (fmt == "txt") {
    return ".pbtxt";
  } else if (fmt == "bin") {
    return ".pb";
  } else {
    return ".pbtxt";
  }
}

// WritableFile that simply prints to stderr.
class StderrWritableFile : public WritableFile {
 public:
  StderrWritableFile() = default;

  absl::Status Append(absl::string_view data) override {
    fprintf(stderr, "%.*s", static_cast<int>(data.size()), data.data());
    return absl::OkStatus();
  }

  absl::Status Close() override { return absl::OkStatus(); }

  absl::Status Flush() override {
    fflush(stderr);
    return absl::OkStatus();
  }

  absl::Status Name(absl::string_view* result) const override {
    *result = "stderr";
    return absl::OkStatus();
  }

  absl::Status Sync() override { return absl::OkStatus(); }

  absl::Status Tell(int64_t* position) override {
    return errors::Unimplemented("Stream not seekable");
  }
};

absl::Status CreateWritableFile(Env* env, const std::string& dirname,
                                const std::string& name,
                                const std::string& suffix,
                                std::string* filepath,
                                std::unique_ptr<WritableFile>* file) {
  std::string dir;
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
    return absl::OkStatus();
  }

  TF_RETURN_IF_ERROR(env->RecursivelyCreateDir(dir));
  *filepath = io::JoinPath(dir, MakeUniqueFilename(name, suffix));
  return env->NewWritableFile(*filepath, file);
}

absl::Status WriteProtoToUniqueFile(const tensorflow::protobuf::Message& proto,
                                    WritableFile* file) {
  std::string s;
  std::string format = GetDumpGraphFormatLowerCase();
  if (format == "txt" &&
      !::tensorflow::protobuf::TextFormat::PrintToString(proto, &s)) {
    return absl::FailedPreconditionError("Unable to convert proto to text.");
  } else if (format == "bin" && !SerializeToStringDeterministic(proto, &s)) {
    return absl::FailedPreconditionError(
        "Failed to serialize proto to string.");
  } else if (format != "txt" && format != "bin") {
    return absl::FailedPreconditionError(
        absl ::StrCat("Unknown format: ", format));
  }
  TF_RETURN_IF_ERROR(file->Append(s));
  absl::string_view name;
  TF_RETURN_IF_ERROR(file->Name(&name));
  VLOG(5) << name;
  VLOG(5) << s;
  return file->Close();
}

absl::Status WriteProtoToUniqueFile(
    const tensorflow::protobuf::MessageLite& proto, WritableFile* file) {
  std::string s;
  if (!SerializeToStringDeterministic(proto, &s)) {
    return errors::Internal("Failed to serialize proto to string.");
  }
  absl::string_view name;
  TF_RETURN_IF_ERROR(file->Name(&name));
  VLOG(5) << name;
  VLOG(5) << s;
  TF_RETURN_IF_ERROR(file->Append(s));
  return file->Close();
}

}  // anonymous namespace

std::string DumpToFile(const std::string& name, const std::string& dirname,
                       const std::string& suffix, absl::string_view type_name,
                       std::function<absl::Status(WritableFile*)> dumper) {
  std::string filepath;
  std::unique_ptr<WritableFile> file;
  absl::Status status = CreateWritableFile(Env::Default(), dirname, name,
                                           suffix, &filepath, &file);
  if (!status.ok()) {
    return absl::StrCat("(failed to create writable file: ", status.ToString(),
                        ")");
  }

  status = dumper(file.get());
  if (!status.ok()) {
    return absl::StrCat("(failed to dump ", type_name, " to '", filepath,
                        "': ", status.ToString(), ")");
  }
  LOG(INFO) << "Dumped " << type_name << " to " << filepath;
  return filepath;
}

void SetGraphDumper(
    std::function<absl::Status(const Graph& graph,
                               const FunctionLibraryDefinition* flib_def,
                               WritableFile*)>
        dumper,
    std::string suffix) {
  GraphDumperConfig& dumper_config = GetGraphDumperConfig();
  mutex_lock lock(dumper_config.mu);
  dumper_config.config.dumper = dumper;
  dumper_config.config.suffix = suffix;
}

std::string DumpGraphDefToFile(const std::string& name,
                               GraphDef const& graph_def,
                               const std::string& dirname) {
  return DumpToFile(name, dirname, GetDumpGraphSuffix(), "Graph",
                    [&](WritableFile* file) {
                      return WriteProtoToUniqueFile(graph_def, file);
                    });
}

std::string DumpCostGraphDefToFile(const std::string& name,
                                   CostGraphDef const& graph_def,
                                   const std::string& dirname) {
  return DumpToFile(name, dirname, GetDumpGraphSuffix(), "Graph",
                    [&](WritableFile* file) {
                      return WriteProtoToUniqueFile(graph_def, file);
                    });
}

std::string DumpGraphToFile(const std::string& name, Graph const& graph,
                            const FunctionLibraryDefinition* flib_def,
                            const std::string& dirname) {
  auto& dumper_config = GetGraphDumperConfig();
  if (dumper_config.IsSet()) {
    GraphDumperConfig::Config config;
    {
      mutex_lock lock(dumper_config.mu);
      config = dumper_config.config;
    }
    if (config.IsSet()) {
      return DumpToFile(name, dirname, config.suffix, "Graph",
                        [&](WritableFile* file) {
                          return config.dumper(graph, flib_def, file);
                        });
    }
  }

  GraphDef graph_def;
  graph.ToGraphDef(&graph_def);
  if (flib_def) {
    *graph_def.mutable_library() = flib_def->ToProto();
  }
  return DumpGraphDefToFile(name, graph_def, dirname);
}

std::string DumpFunctionDefToFile(const std::string& name,
                                  FunctionDef const& fdef,
                                  const std::string& dirname) {
  return DumpToFile(
      name, dirname, GetDumpGraphSuffix(), "FunctionDef",
      [&](WritableFile* file) { return WriteProtoToUniqueFile(fdef, file); });
}

std::string DumpProtoToFile(const std::string& name,
                            tensorflow::protobuf::Message const& proto,
                            const std::string& dirname) {
  return DumpToFile(
      name, dirname, GetDumpGraphSuffix(), proto.GetTypeName(),
      [&](WritableFile* file) { return WriteProtoToUniqueFile(proto, file); });
}

}  // namespace tensorflow
