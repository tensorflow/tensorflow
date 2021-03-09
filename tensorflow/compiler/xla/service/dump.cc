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

#include "tensorflow/compiler/xla/service/dump.h"

#include "absl/strings/ascii.h"
#include "tensorflow/compiler/xla/service/hlo_graph_dumper.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_proto_util.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/proto_serialization.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/path.h"
#include "tensorflow/core/platform/regexp.h"

namespace xla {

namespace {

using absl::StrCat;
using absl::StrFormat;
using absl::string_view;

struct CanonicalDebugOptions {
  explicit CanonicalDebugOptions(const DebugOptions& opts)
      : dump_to(opts.xla_dump_to()),
        dump_as_text(opts.xla_dump_hlo_as_text()),
        dump_as_proto(opts.xla_dump_hlo_as_proto()),
        dump_as_dot(opts.xla_dump_hlo_as_dot()),
        dump_as_html(opts.xla_dump_hlo_as_html()),
        dump_as_url(opts.xla_dump_hlo_as_url()),
        dump_fusion_visualization(opts.xla_dump_fusion_visualization()),
        dump_snapshots(opts.xla_dump_hlo_snapshots()),
        dump_include_timestamp(opts.xla_dump_include_timestamp()),
        dump_max_hlo_modules(opts.xla_dump_max_hlo_modules()),
        dump_module_metadata(opts.xla_dump_module_metadata()) {
    // This constructor examines the values in `opts` and turns on other flags
    // based on what we think is the user's intent.  To reduce confusion about
    // what was a user-specified value versus an extrapolated value, within this
    // function we treat this struct's members as write-only, and read only from
    // `opts`.

    // Did the user specify an explicit format for dumping?
    bool output_format_other_than_url_specified =
        opts.xla_dump_hlo_as_text() || opts.xla_dump_hlo_as_proto() ||
        opts.xla_dump_hlo_as_dot() || opts.xla_dump_hlo_as_html() ||
        opts.xla_dump_hlo_snapshots();
    bool output_format_specified =
        output_format_other_than_url_specified || opts.xla_dump_hlo_as_url();

    // If we haven't specified an output format, default to dumping as text.
    if (!output_format_specified) {
      dump_as_text = true;
    }

    // If dump_to is empty, default to dumping to stdout, so long as some dump
    // format other than dump-as-url was specified.  If the user only specified
    // --xla_dump_hlo_as_url, then don't dump to stdout, that is likely noise
    // they don't want.
    if (opts.xla_dump_to().empty() && output_format_other_than_url_specified) {
      dump_to = "-";
    }

    // If we specified a regular expression restricting which modules to dump,
    // respect that.
    //
    // If we didn't specify which modules to dump but we passed some other flag
    // which implies dumping modules, dump all modules.
    //
    // Otherwise, don't dump any HLO modules.
    if (!opts.xla_dump_hlo_module_re().empty()) {
      // RE2 object is not copyable, and we can't capture "by move", so we
      // resort to this hack.
      string pattern = opts.xla_dump_hlo_module_re();
      should_dump_module = [pattern](string_view module_name) {
        return RE2::PartialMatch(module_name, pattern);
      };
    } else if (!opts.xla_dump_hlo_pass_re().empty() ||
               !opts.xla_dump_to().empty() || output_format_specified) {
      should_dump_module = [](string_view) { return true; };
    } else {
      should_dump_module = [](string_view) { return false; };
    }

    // Initialize should_dump_pass.  This one is easy: We only dump per-pass
    // data if the user asked for it explicitly.
    if (!opts.xla_dump_hlo_pass_re().empty()) {
      string pattern = opts.xla_dump_hlo_pass_re();
      should_dump_pass = [pattern](string_view pass_name) {
        return RE2::PartialMatch(pass_name, pattern);
      };
    } else {
      should_dump_pass = [](string_view) { return false; };
    }

    // Output dirs "sponge" and "test_undeclared_outputs_dir" (case-insensitive)
    // have a special meaning: Dump into the directory specified by the
    // environment variable TEST_UNDECLARED_OUTPUTS_DIR.
    string dump_to_lower = absl::AsciiStrToLower(opts.xla_dump_to());
    if (dump_to_lower == "sponge" ||
        dump_to_lower == "test_undeclared_outputs_dir") {
      if (!tensorflow::io::GetTestUndeclaredOutputsDir(&dump_to)) {
        LOG(ERROR) << "--xla_dump_to=" << opts.xla_dump_to()
                   << ", but environment variable TEST_UNDECLARED_OUTPUTS_DIR "
                      "is not set, so cannot dump anywhere.";
        should_dump_module = [](string_view) { return false; };
        should_dump_pass = [](string_view) { return false; };
      }
    }
  }

  bool dumping_to_stdout() const { return dump_to == "-"; }

  string dump_to;
  std::function<bool(string_view module_name)> should_dump_module;
  std::function<bool(string_view pass_name)> should_dump_pass;

  // dump_ir isn't present here because this file is mostly concerned with
  // dumping HLO.
  bool dump_as_text;
  bool dump_as_proto;
  bool dump_as_dot;
  bool dump_as_html;
  bool dump_as_url;
  bool dump_fusion_visualization;
  bool dump_snapshots;
  bool dump_include_timestamp;
  int64 dump_max_hlo_modules;
  bool dump_module_metadata;
};

absl::optional<std::string> DumpToFileInDirImpl(
    string_view filename, string_view contents,
    const CanonicalDebugOptions& opts) {
  if (opts.dumping_to_stdout()) {
    LOG(ERROR) << "Refusing to write " << filename
               << " to stdout.  Pass --xla_dump_to=<path> to write to a file.";
    return absl::nullopt;
  }

  if (opts.dump_to.empty()) {
    return absl::nullopt;
  }

  const string& dir = opts.dump_to;
  VLOG(1) << "Dumping " << filename << " to " << dir;

  tensorflow::Env* env = tensorflow::Env::Default();
  // Two threads can race to observe the absence of the dump directory and
  // simultaneously try to create it, causing the "losing" thread to get a
  // "directory already exists" error.  We can work around this by checking
  // again whether the dir exists.
  if (!env->IsDirectory(dir).ok()) {
    auto status = env->RecursivelyCreateDir(dir);
    if (!status.ok() && !env->IsDirectory(dir).ok()) {
      LOG(ERROR) << "Could not create directory " << dir
                 << " for dumping XLA debug data: " << status;
      return absl::nullopt;
    }
  }

  // Make sure we are not going to dump more modules than the user has asked.
  if (opts.dump_max_hlo_modules > 0) {
    std::vector<string> matches;
    auto pattern = tensorflow::io::JoinPath(dir, "*module_*.0000.*");
    auto status = env->GetMatchingPaths(pattern, &matches);
    if (!status.ok()) {
      LOG(ERROR) << "Could not get matching paths for pattern " << pattern
                 << ": " << status;
    }
    if (matches.size() > opts.dump_max_hlo_modules) {
      LOG(ERROR) << "Have already dumped " << matches.size()
                 << " modules, more than the limit of "
                 << opts.dump_max_hlo_modules;
      return absl::nullopt;
    }
  }

  string file_path =
      tensorflow::io::JoinPath(dir, SanitizeFileName(string(filename)));
  auto status = tensorflow::WriteStringToFile(env, file_path, contents);
  if (!status.ok()) {
    LOG(ERROR) << "Could not write XLA debug data to " << file_path << ": "
               << status;
  }

  return file_path;
}

absl::optional<std::string> DumpToFileInDirOrStdoutImpl(
    string_view filename, string_view contents,
    const CanonicalDebugOptions& opts) {
  // Dump to stdout if that's called for.
  if (opts.dumping_to_stdout()) {
    std::cout << "*** Begin " << filename << " ***\n"
              << contents << "\n*** End " << filename << " ***" << std::endl;
    return absl::nullopt;
  }

  // Otherwise, dump to a file.
  return DumpToFileInDirImpl(filename, contents, opts);
}

// Returns full file paths of all dumps of the module.
std::vector<std::string> DumpHloModuleImpl(const HloModule& module,
                                           const BufferAssignment* buffer_assn,
                                           const HloExecutionProfile* profile,
                                           string_view prefix,
                                           string_view suffix,
                                           const CanonicalDebugOptions& opts) {
  string filename = FilenameFor(module, prefix, suffix);

  std::vector<absl::optional<std::string>> file_paths;

  if (opts.dump_as_text) {
    file_paths.push_back(DumpToFileInDirOrStdoutImpl(
        StrCat(filename, ".txt"),
        module.ToString(HloPrintOptions().set_print_backend_config(true)),
        opts));
    if (buffer_assn) {
      file_paths.push_back(DumpToFileInDirOrStdoutImpl(
          StrCat(filename, "-buffer-assignment.txt"), buffer_assn->ToString(),
          opts));
    }
  }

  if (opts.dump_as_proto) {
    HloProto module_proto =
        buffer_assn ? MakeHloProto(module, *buffer_assn) : MakeHloProto(module);
    string pb;
    if (!tensorflow::SerializeToStringDeterministic(module_proto, &pb)) {
      pb = "Failed to serialize HLO module proto.";
    }
    file_paths.push_back(
        DumpToFileInDirImpl(StrCat(filename, ".hlo.pb"), pb, opts));
  }

  auto render_graph = [&](RenderedGraphFormat format) {
    StatusOr<string> rendered_graph = RenderGraph(
        *module.entry_computation(),
        /*label=*/filename, module.config().debug_options(), format, profile);
    if (rendered_graph.ok()) {
      return std::move(rendered_graph).ValueOrDie();
    }
    return StrFormat("Error rendering graph: %s",
                     rendered_graph.status().ToString());
  };

  if (opts.dump_as_dot) {
    file_paths.push_back(
        DumpToFileInDirImpl(StrFormat("%s.dot", filename),
                            render_graph(RenderedGraphFormat::kDot), opts));
  }

  if (opts.dump_as_html) {
    file_paths.push_back(
        DumpToFileInDirImpl(StrFormat("%s.html", filename),
                            render_graph(RenderedGraphFormat::kHtml), opts));
  }

  if (opts.dump_fusion_visualization) {
    for (const HloComputation* computation :
         module.MakeNonfusionComputations()) {
      StatusOr<string> rendered_graph = RenderGraph(
          *computation,
          /*label=*/absl::StrCat(filename, "_", computation->name()),
          module.config().debug_options(),
          RenderedGraphFormat::kFusionVisualization, profile);
      file_paths.push_back(DumpToFileInDirImpl(
          StrFormat("%s_%s_fusion_visualization.html", filename,
                    computation->name()),
          rendered_graph.ok() ? *rendered_graph
                              : StrFormat("Error rendering graph: %s",
                                          rendered_graph.status().ToString()),
          opts));
    }
  }

  // Special case for rendering graphs as URLs.  We'll dump them to a file
  // because why not, but we always log them to stdout as well.
  if (opts.dump_as_url) {
    string url = render_graph(RenderedGraphFormat::kUrl);
    std::cout << filename << " --> " << url << std::endl;
    if (!opts.dumping_to_stdout()) {
      file_paths.push_back(
          DumpToFileInDirImpl(StrFormat("%s.url", filename), url, opts));
    }
  }

  std::vector<std::string> dumped_file_paths;
  for (const absl::optional<std::string>& path : file_paths) {
    if (path.has_value()) {
      dumped_file_paths.push_back(*path);
    }
  }
  return dumped_file_paths;
}

void DumpHloModuleMetadata(const HloModuleMetadataProto& metadata,
                           const CanonicalDebugOptions& opts,
                           absl::flat_hash_set<int64>* dumped_module_ids) {
  // Return if metadata for this module has already been dumped.
  if (!dumped_module_ids->insert(metadata.canonical_module_id()).second) {
    return;
  }
  std::string filename = absl::StrFormat("module_%04d.metadata.textproto",
                                         metadata.canonical_module_id());
  std::string content;
  if (tensorflow::protobuf::TextFormat::PrintToString(metadata, &content)) {
    DumpToFileInDirImpl(filename, content, opts);
  } else {
    LOG(ERROR) << "Failed to convert HloModuleMetadataProto to text.";
  }
}

static tensorflow::mutex mu(tensorflow::LINKER_INITIALIZED);

// Maps a module's unique ID to a counter indicating how many times we've dumped
// this module during the compilation pipeline.  This lets us keep the filenames
// ordered nicely.
//
// Entries added here leak forever; we have no way to GC them when a module
// dies.  But we only add an entry if dumping is enabled for this module, and
// dumping a module leaks buffer space in stdout or bytes on disk *way* faster
// than this hashtable leaks memory.
static auto& module_id_to_step_number TF_GUARDED_BY(mu) =
    *new absl::flat_hash_map<int64, int64>();

// Maps a module's unique ID to a timestamp indicating when we've first dumped
// this module during the compilation pipeline and when we first started
// compiling this module.  This lets us keep the filenames ordered nicely.
//
// Entries added here leak forever; we have no way to GC them when a module
// dies.  But we only add an entry if dumping is enabled for this module, and
// dumping a module leaks buffer space in stdout or bytes on disk *way* faster
// than this hashtable leaks memory.
static auto& module_id_to_timestamp TF_GUARDED_BY(mu) =
    *new absl::flat_hash_map<int64, uint64>();

int64 StepNumberForModule(const HloModule& module) {
  tensorflow::mutex_lock lock(mu);
  return module_id_to_step_number[module.unique_id()]++;
}

// Get a timestamp which we can use as a filename prefix specific to this
// module.
string TimestampFor(const HloModule& module,
                    const DebugOptions& debug_options) {
  if (!debug_options.xla_dump_include_timestamp()) {
    return "";
  }
  tensorflow::mutex_lock lock(mu);
  auto timestamp_emplace = module_id_to_timestamp.try_emplace(
      module.unique_id(), tensorflow::Env::Default()->NowMicros());
  return std::to_string(timestamp_emplace.first->second);
}

string TimestampFor(const HloModule& module) {
  return TimestampFor(module, module.config().debug_options());
}

}  // namespace

string FilenameFor(const HloModule& module, string_view prefix,
                   string_view suffix) {
  return StrFormat("%s%smodule_%04d.%s", prefix, prefix.empty() ? "" : ".",
                   module.unique_id(), suffix);
}

void DumpToFileInDir(const HloModule& module, string_view file_prefix,
                     string_view file_suffix, string_view contents) {
  DumpToFileInDirImpl(FilenameFor(module, file_prefix, file_suffix), contents,
                      CanonicalDebugOptions(module.config().debug_options()));
}

void DumpToFileInDirOrStdout(const HloModule& module, string_view file_prefix,
                             string_view file_suffix, string_view contents) {
  DumpToFileInDirOrStdoutImpl(
      FilenameFor(module, file_prefix, file_suffix), contents,
      CanonicalDebugOptions(module.config().debug_options()));
}

void DumpExecutionOptions(const ExecutionOptions& execution_options,
                          const DebugOptions& debug_options) {
  CanonicalDebugOptions opts(debug_options);
  tensorflow::Env* env = tensorflow::Env::Default();
  const string& dir = opts.dump_to;
  if (!env->IsDirectory(dir).ok()) {
    auto status = env->RecursivelyCreateDir(dir);
    if (!status.ok()) {
      LOG(ERROR) << "Could not create directory " << dir
                 << " for dumping XLA execution options: " << status;
      return;
    }
  }
  if (env->IsDirectory(dir).ok()) {
    string filename = tensorflow::io::JoinPath(dir, "execution_options");
    Status status;
    if (opts.dump_as_text) {
      status = tensorflow::WriteTextProto(env, absl::StrCat(filename, ".txt"),
                                          execution_options);
    } else {
      status = tensorflow::WriteBinaryProto(env, absl::StrCat(filename, ".pb"),
                                            execution_options);
    }
    if (!status.ok()) {
      LOG(ERROR) << "Could not write XLA debug data to " << filename << ": "
                 << status;
    }
  }
}

void DumpHloModuleIfEnabled(const HloModule& module, string_view name,
                            const DebugOptions& debug_options) {
  CanonicalDebugOptions opts(debug_options);
  if (opts.should_dump_module(module.name())) {
    DumpHloModuleImpl(module, /*buffer_assn=*/nullptr, /*profile=*/nullptr,
                      TimestampFor(module, debug_options), name, opts);
  }
}

void DumpHloModuleIfEnabled(const HloModule& module, string_view name) {
  DumpHloModuleIfEnabled(module, name, module.config().debug_options());
}

void DumpHloModuleIfEnabled(const HloModule& module,
                            const BufferAssignment& buffer_assn,
                            string_view name) {
  CanonicalDebugOptions opts(module.config().debug_options());
  if (opts.should_dump_module(module.name())) {
    DumpHloModuleImpl(module, &buffer_assn, /*profile=*/nullptr,
                      TimestampFor(module), name, opts);
  }
}

void DumpHloModuleIfEnabled(const HloModule& module,
                            const HloExecutionProfile& profile,
                            string_view name) {
  CanonicalDebugOptions opts(module.config().debug_options());
  if (opts.should_dump_module(module.name())) {
    DumpHloModuleImpl(module, /*buffer_assn=*/nullptr, &profile,
                      TimestampFor(module), name, opts);
  }
}

bool DumpingEnabledForHloModule(string_view hlo_module_name,
                                const DebugOptions& opts) {
  return CanonicalDebugOptions(opts).should_dump_module(hlo_module_name);
}

bool DumpingToStdout(const DebugOptions& opts) {
  return CanonicalDebugOptions(opts).dumping_to_stdout();
}

std::vector<std::string> DumpHloModuleBetweenPassesIfEnabled(
    string_view pipeline_name, string_view before_pass_name,
    string_view after_pass_name, const HloModule& module) {
  CanonicalDebugOptions opts(module.config().debug_options());
  if (!opts.should_dump_module(module.name())) {
    return {};
  }

  if (!opts.should_dump_pass(before_pass_name) &&
      !opts.should_dump_pass(after_pass_name)) {
    return {};
  }

  int64 step_number = StepNumberForModule(module);
  std::string timestamp = TimestampFor(module);

  string filename_suffix =
      StrFormat("%04d.%s.after_%s.before_%s", step_number, pipeline_name,
                after_pass_name, before_pass_name);
  return DumpHloModuleImpl(module, /*buffer_assn=*/nullptr, /*profile=*/nullptr,
                           timestamp, filename_suffix, opts);
}

void DumpHloModuleDuringPassIfEnabled(string_view pass_name,
                                      string_view step_name,
                                      const HloModule& module) {
  CanonicalDebugOptions opts(module.config().debug_options());
  if (!opts.should_dump_module(module.name()) ||
      !opts.should_dump_pass(pass_name)) {
    return;
  }

  int64 step_number = StepNumberForModule(module);
  std::string timestamp = TimestampFor(module);

  string filename_suffix =
      StrFormat("%04d.%s.%s", step_number, pass_name, step_name);
  DumpHloModuleImpl(module, /*buffer_assn=*/nullptr, /*profile=*/nullptr,
                    timestamp, filename_suffix, opts);
}

void DumpHloSnapshotIfEnabled(const HloModule& module,
                              const HloSnapshot& snapshot) {
  CanonicalDebugOptions opts(module.config().debug_options());
  if (!opts.should_dump_module(module.name()) || !opts.dump_snapshots) {
    return;
  }
  int64 execution_count;
  uint64 timestamp;
  {
    static auto& module_id_to_execution_count TF_GUARDED_BY(mu) =
        *new absl::flat_hash_map<int64, int64>();
    tensorflow::mutex_lock lock(mu);
    execution_count = module_id_to_execution_count[module.unique_id()]++;
    auto timestamp_emplace = module_id_to_timestamp.try_emplace(
        module.unique_id(), tensorflow::Env::Default()->NowMicros());
    timestamp = timestamp_emplace.first->second;
  }
  string filename =
      StrCat(FilenameFor(module, std::to_string(timestamp),
                         StrFormat("execution_%04d", execution_count)),
             ".hlo_snapshot.pb");
  if (opts.dumping_to_stdout()) {
    LOG(ERROR) << "Refusing to write HLO snapshot proto for " << filename
               << " to stdout.  Pass --xla_dump_to=<path> to write to a file.";
    return;
  }
  string pb;
  if (!tensorflow::SerializeToStringDeterministic(snapshot, &pb)) {
    LOG(ERROR) << "Failed to serialize HLO snapshot proto " << filename;
  }
  DumpToFileInDirImpl(filename, pb, opts);
}

void DumpHloSnapshotIfEnabled(const HloSnapshot& snapshot,
                              const DebugOptions& opts) {
  CanonicalDebugOptions canonical_opts(opts);
  string name = snapshot.hlo().hlo_module().name();
  if (!canonical_opts.should_dump_module(name) ||
      !canonical_opts.dump_snapshots) {
    return;
  }

  // We don't have a unique id for an HloSnapshot, so in this overload we just
  // have to use its name.
  int64 execution_count;
  {
    static auto& module_name_to_execution_count TF_GUARDED_BY(mu) =
        *new absl::flat_hash_map<string, int64>();
    tensorflow::mutex_lock lock(mu);
    execution_count = module_name_to_execution_count[name]++;
  }
  string filename = StrFormat("module_%s.execution_%04d.hlo_snapshot.pb", name,
                              execution_count);
  if (canonical_opts.dumping_to_stdout()) {
    LOG(ERROR) << "Refusing to write HLO snapshot proto for " << filename
               << " to stdout.  Pass --xla_dump_to=<path> to write to a file.";
    return;
  }
  string pb;
  if (!tensorflow::SerializeToStringDeterministic(snapshot, &pb)) {
    LOG(ERROR) << "Failed to serialize HLO snapshot proto " << filename;
  }
  DumpToFileInDirImpl(filename, pb, canonical_opts);
}

void DumpHloModuleMetadataIfEnabled(const std::vector<HloModule*>& modules) {
  absl::flat_hash_set<int64> dumped_module_ids;
  for (const HloModule* module : modules) {
    CanonicalDebugOptions opts(module->config().debug_options());
    if (!opts.dump_module_metadata) {
      continue;
    }
    DumpHloModuleMetadata(module->metadata().proto(), opts, &dumped_module_ids);
    const absl::optional<HloModuleMetadataProto>& prepartitioning_metadata =
        module->metadata().prepartitioning_metadata();
    if (prepartitioning_metadata.has_value()) {
      DumpHloModuleMetadata(*prepartitioning_metadata, opts,
                            &dumped_module_ids);
    }
  }
}

}  // namespace xla
