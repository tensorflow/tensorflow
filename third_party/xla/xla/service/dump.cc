/* Copyright 2019 The OpenXLA Authors.

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

#include "xla/service/dump.h"

#include <functional>
#include <memory>
#include <queue>
#include <utility>

#include "absl/functional/any_invocable.h"
#include "absl/strings/ascii.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/OperationSupport.h"  // from @llvm-project
#include "mlir/Support/FileUtilities.h"  // from @llvm-project
#include "mlir/Transforms/LocationSnapshot.h"  // from @llvm-project
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/hlo_graph_dumper.h"
#include "xla/service/hlo_proto_util.h"
#include "xla/status.h"
#include "xla/statusor.h"
#include "xla/util.h"
#include "tsl/lib/io/zlib_compression_options.h"
#include "tsl/lib/io/zlib_outputbuffer.h"
#include "tsl/lib/strings/proto_serialization.h"
#include "tsl/platform/env.h"
#include "tsl/platform/path.h"
#include "tsl/platform/regexp.h"
#include "tsl/platform/status.h"

namespace xla {

std::string RenderGraph(absl::string_view label, const HloModule& module,
                        RenderedGraphFormat format,
                        bool show_fusion_subcomputations) {
  HloRenderOptions hlo_render_options;
  hlo_render_options.show_fusion_subcomputations = show_fusion_subcomputations;
  absl::StatusOr<std::string> rendered_graph =
      RenderGraph(*module.entry_computation(), label,
                  module.config().debug_options(), format, hlo_render_options);
  if (rendered_graph.ok()) {
    return std::move(rendered_graph).value();
  }
  return absl::StrFormat("Error rendering graph: %s",
                         rendered_graph.status().ToString());
}

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
        dump_module_metadata(opts.xla_dump_module_metadata()),
        dump_compress_protos(opts.xla_dump_compress_protos()),
        dump_hlo_metadata(!opts.xla_dump_disable_metadata()),
        dump_as_long_text(opts.xla_dump_hlo_as_long_text()),
        dump_mlir_pretty_form(opts.xla_dump_enable_mlir_pretty_form()) {
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

    // Disable dumping if specified by the user.
    if (!opts.xla_enable_dumping()) {
      dump_to = "";
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
      std::string pattern = opts.xla_dump_hlo_module_re();
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
      std::string pattern = opts.xla_dump_hlo_pass_re();
      should_dump_pass = [pattern](string_view pass_name) {
        return RE2::PartialMatch(pass_name, pattern);
      };
    } else {
      should_dump_pass = [](string_view) { return false; };
    }

    // Initialize should_dump_pipeline. If the option was not specified, dump
    // all pipelines. Otherwise dump only those pipelines that user asked for
    // explicitly.
    if (!opts.xla_dump_hlo_pipeline_re().empty()) {
      std::string pattern = opts.xla_dump_hlo_pipeline_re();
      should_dump_pipeline = [pattern](string_view pipeline_name) {
        return RE2::PartialMatch(pipeline_name, pattern);
      };
    } else {
      should_dump_pipeline = [](string_view) { return true; };
    }

    // Output dirs "sponge" and "test_undeclared_outputs_dir" (case-insensitive)
    // have a special meaning: Dump into the directory specified by the
    // environment variable TEST_UNDECLARED_OUTPUTS_DIR.
    std::string dump_to_lower = absl::AsciiStrToLower(dump_to);
    if (dump_to_lower == "sponge" ||
        dump_to_lower == "test_undeclared_outputs_dir") {
      if (!tsl::io::GetTestUndeclaredOutputsDir(&dump_to)) {
        LOG(ERROR) << "--xla_dump_to=" << opts.xla_dump_to()
                   << ", but environment variable TEST_UNDECLARED_OUTPUTS_DIR "
                      "is not set, so cannot dump anywhere.";
        should_dump_module = [](string_view) { return false; };
        should_dump_pass = [](string_view) { return false; };
        should_dump_pipeline = [](string_view) { return false; };
      }
    }
  }

  bool dumping_to_stdout() const { return dump_to == "-"; }

  std::string dump_to;
  std::function<bool(string_view module_name)> should_dump_module;
  std::function<bool(string_view pass_name)> should_dump_pass;
  std::function<bool(string_view pipeline_name)> should_dump_pipeline;

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
  int64_t dump_max_hlo_modules;
  bool dump_module_metadata;
  bool dump_compress_protos;
  bool dump_hlo_metadata;
  bool dump_as_long_text;
  bool dump_mlir_pretty_form;
};

// Helper class to hold a list of functions that produces data to be written to
// a file in multiple stages, so that we can lower the peak memory usage.
// Ideally we should migrate this whole file to use an I/O stream style API.
class DataProducer {
 public:
  void Append(std::function<std::string()> produce_func) {
    produce_funcs_.push(std::move(produce_func));
  }

  std::function<std::string()> Next() {
    if (produce_funcs_.empty()) {
      return nullptr;
    }
    auto next = std::move(produce_funcs_.front());
    produce_funcs_.pop();
    return next;
  }

 private:
  std::queue<std::function<std::string()>> produce_funcs_;
};

static Status WriteStringToFile(tsl::Env* env, const std::string& fname,
                                DataProducer& data_producer, bool compressed) {
  std::unique_ptr<tsl::WritableFile> file;
  TF_RETURN_IF_ERROR(env->NewWritableFile(fname, &file));
  if (compressed) {
    auto gz_opts = tsl::io::ZlibCompressionOptions::GZIP();
    tsl::io::ZlibOutputBuffer gz_file(file.get(), gz_opts.input_buffer_size,
                                      gz_opts.output_buffer_size, gz_opts);
    TF_RETURN_IF_ERROR(gz_file.Init());
    while (auto next_producer = data_producer.Next()) {
      TF_RETURN_IF_ERROR(gz_file.Append(next_producer()));
    }
    return gz_file.Close();
  } else {
    while (auto next_producer = data_producer.Next()) {
      TF_RETURN_IF_ERROR(file->Append(next_producer()));
    }
    return file->Close();
  }
}

static Status WriteStringToFile(tsl::Env* env, const std::string& fname,
                                absl::string_view data, bool compressed) {
  if (!compressed) {
    return tsl::WriteStringToFile(env, fname, data);
  }
  std::unique_ptr<tsl::WritableFile> file;
  TF_RETURN_IF_ERROR(env->NewWritableFile(fname, &file));
  auto gz_opts = tsl::io::ZlibCompressionOptions::GZIP();
  tsl::io::ZlibOutputBuffer gz_file(file.get(), gz_opts.input_buffer_size,
                                    gz_opts.output_buffer_size, gz_opts);
  TF_RETURN_IF_ERROR(gz_file.Init());
  TF_RETURN_IF_ERROR(gz_file.Append(data));
  return gz_file.Close();
}

static std::optional<std::string> GetDumpFilePath(
    string_view filename, const CanonicalDebugOptions& opts) {
  if (opts.dumping_to_stdout()) {
    LOG(ERROR) << "Refusing to write " << filename
               << " to stdout. Pass --xla_dump_to=<path> to write to a file.";
    return std::nullopt;
  }

  if (opts.dump_to.empty()) {
    return std::nullopt;
  }

  const std::string& dir = opts.dump_to;
  VLOG(1) << "Dumping " << filename << " to " << dir;

  tsl::Env* env = tsl::Env::Default();
  // Two threads can race to observe the absence of the dump directory and
  // simultaneously try to create it, causing the "losing" thread to get a
  // "directory already exists" error.  We can work around this by checking
  // again whether the dir exists.
  if (!env->IsDirectory(dir).ok()) {
    auto status = env->RecursivelyCreateDir(dir);
    if (!status.ok() && !env->IsDirectory(dir).ok()) {
      LOG(ERROR) << "Could not create directory " << dir
                 << " for dumping XLA debug data: " << status;
      return std::nullopt;
    }
  }

  // Make sure we are not going to dump more modules than the user has asked.
  if (opts.dump_max_hlo_modules > 0) {
    std::vector<std::string> matches;
    auto pattern = tsl::io::JoinPath(dir, "*module_*.*");
    auto status = env->GetMatchingPaths(pattern, &matches);
    if (!status.ok()) {
      LOG(ERROR) << "Could not get matching paths for pattern " << pattern
                 << ": " << status;
    }
    static const LazyRE2 module_id_regex = {R"(.*module_(\d+)\..*)"};
    absl::flat_hash_set<int64_t> dumped_module_ids;
    for (const std::string& match : matches) {
      int64_t dumped_module_id;
      if (RE2::FullMatch(match, *module_id_regex, &dumped_module_id)) {
        dumped_module_ids.insert(dumped_module_id);
      }
    }
    if (dumped_module_ids.size() >= opts.dump_max_hlo_modules) {
      int64_t module_id;
      if (RE2::FullMatch(filename, *module_id_regex, &module_id) &&
          !dumped_module_ids.contains(module_id)) {
        LOG(ERROR) << "Have already dumped " << dumped_module_ids.size()
                   << " modules, more than the limit of "
                   << opts.dump_max_hlo_modules;
        return std::nullopt;
      }
    }
  }

  return tsl::io::JoinPath(dir, SanitizeFileName(std::string(filename)));
}

static std::optional<std::string> DumpToFileInDirImpl(
    string_view filename, string_view contents,
    const CanonicalDebugOptions& opts, bool compress = false) {
  auto file_path = GetDumpFilePath(filename, opts);
  if (!file_path) return std::nullopt;

  auto status =
      WriteStringToFile(tsl::Env::Default(), *file_path, contents, compress);
  if (!status.ok()) {
    LOG(ERROR) << "Could not write XLA debug data to " << *file_path << ": "
               << status;
    return std::nullopt;
  }

  return file_path;
}

static std::optional<std::string> DumpToFileInDirImpl(
    string_view filename, DataProducer& data_producer,
    const CanonicalDebugOptions& opts, bool compress = false) {
  auto file_path = GetDumpFilePath(filename, opts);
  if (!file_path) return std::nullopt;

  auto status = WriteStringToFile(tsl::Env::Default(), *file_path,
                                  data_producer, compress);
  if (!status.ok()) {
    LOG(ERROR) << "Could not write XLA debug data to " << *file_path << ": "
               << status;
    return std::nullopt;
  }

  return file_path;
}

static absl::Mutex stdout_dump_mutex(absl::kConstInit);

static std::optional<std::string> DumpToFileInDirOrStdoutImpl(
    string_view filename, string_view contents,
    const CanonicalDebugOptions& opts) {
  // Dump to stdout if that's called for.
  if (opts.dumping_to_stdout()) {
    absl::MutexLock lock(&stdout_dump_mutex);
    std::cout << "*** Begin " << filename << " ***\n"
              << contents << "\n*** End " << filename << " ***" << std::endl;
    return std::nullopt;
  }

  // Otherwise, dump to a file.
  return DumpToFileInDirImpl(filename, contents, opts);
}

static std::optional<std::string> DumpToFileInDirOrStdoutImpl(
    string_view filename, DataProducer& data_producer,
    const CanonicalDebugOptions& opts) {
  // Dump to stdout if that's called for.
  if (opts.dumping_to_stdout()) {
    absl::MutexLock lock(&stdout_dump_mutex);
    std::cout << "*** Begin " << filename << " ***\n";
    while (auto next_producer = data_producer.Next()) {
      std::cout << next_producer();
    }
    std::cout << "\n*** End " << filename << " ***" << std::endl;
    return std::nullopt;
  }

  // Otherwise, dump to a file.
  return DumpToFileInDirImpl(filename, data_producer, opts);
}

// Returns whether the computation is trivial enough not to warrant dumping.
// Currently skips instructions where the root instruction has only parameters
// as operands AND is not a fusion.
static bool IsTrivial(const HloComputation& computation) {
  const HloInstruction* root = computation.root_instruction();
  return absl::c_all_of(root->operands(),
                        [&](const HloInstruction* op) {
                          return op->opcode() == HloOpcode::kParameter;
                        }) &&
         root->opcode() != HloOpcode::kFusion;
}

// Returns full file paths of all dumps of the module.
static std::vector<std::string> DumpHloModuleImpl(
    const HloModule& module, const BufferAssignment* buffer_assn,
    string_view prefix, string_view suffix, const CanonicalDebugOptions& opts) {
  std::string filename = FilenameFor(module, prefix, suffix);

  std::vector<std::optional<std::string>> file_paths;

  if (opts.dump_as_text) {
    auto print_options = opts.dump_as_long_text
                             ? HloPrintOptions::Default()
                             : HloPrintOptions::ShortParsable();
    print_options.set_print_large_constants(false);
    print_options.set_print_control_dependencies(true);
    print_options.set_print_operand_index_annotation_interval(5);
    print_options.set_print_backend_config(true);
    print_options.set_print_metadata(opts.dump_hlo_metadata);
    print_options.set_print_name_after_closing_brace(true);
    file_paths.push_back(DumpToFileInDirOrStdoutImpl(
        StrCat(filename, ".txt"), module.ToString(print_options), opts));
    if (buffer_assn) {
      DataProducer data_producer;
      data_producer.Append([&] { return buffer_assn->ToString(); });
      data_producer.Append([&] { return "\n\n"; });
      data_producer.Append(
          [&] { return buffer_assn->hlo_live_range().ToString(); });
      file_paths.push_back(DumpToFileInDirOrStdoutImpl(
          StrCat(filename, "-buffer-assignment.txt"), data_producer, opts));
    }
  }

  if (opts.dump_as_proto) {
    HloProto module_proto =
        buffer_assn ? MakeHloProto(module, *buffer_assn) : MakeHloProto(module);
    std::string pb;
    if (!tsl::SerializeToStringDeterministic(module_proto, &pb)) {
      pb = "Failed to serialize HLO module proto.";
    }
    file_paths.push_back(DumpToFileInDirImpl(
        StrCat(filename, opts.dump_compress_protos ? ".hlo.pb.gz" : ".hlo.pb"),
        pb, opts, opts.dump_compress_protos));
  }

  if (opts.dump_as_dot) {
    file_paths.push_back(DumpToFileInDirImpl(
        StrFormat("%s.dot", filename),
        RenderGraph(filename, module, RenderedGraphFormat::kDot), opts));
  }

  if (opts.dump_as_html) {
    file_paths.push_back(DumpToFileInDirImpl(
        StrFormat("%s.html", filename),
        RenderGraph(filename, module, RenderedGraphFormat::kHtml), opts));
    if (absl::StrContains(filename, kAfterOptimizationsDumpName)) {
      file_paths.push_back(DumpToFileInDirImpl(
          StrFormat("%s.top_level.html", filename),
          RenderGraph(filename, module, RenderedGraphFormat::kHtml, false),
          opts));
    }
  }

  if (opts.dump_fusion_visualization) {
    for (const HloComputation* computation :
         module.MakeNonfusionComputations()) {
      if (IsTrivial(*computation)) {
        VLOG(1) << "Skipping computation " << computation->name()
                << " as trivial";
        continue;
      }

      absl::StatusOr<std::string> rendered_graph =
          WrapFusionExplorer(*computation);
      if (!rendered_graph.ok()) {
        VLOG(1) << "Skipping fusion visualization"
                << " for computation " << computation->name()
                << " due to: " << rendered_graph.status();
        continue;
      }
      file_paths.push_back(DumpToFileInDirImpl(
          FilenameFor(module, computation->name(), "_fusion.html"),
          *rendered_graph, opts));
    }
  }

  // Special case for rendering graphs as URLs.  We'll dump them to a file
  // because why not, but we always log them to stdout as well.
  if (opts.dump_as_url) {
    std::string url = RenderGraph(filename, module, RenderedGraphFormat::kUrl);
    std::cout << filename << " --> " << url << std::endl;
    if (!opts.dumping_to_stdout()) {
      file_paths.push_back(
          DumpToFileInDirImpl(StrFormat("%s.url", filename), url, opts));
    }
  }

  std::vector<std::string> dumped_file_paths;
  for (const std::optional<std::string>& path : file_paths) {
    if (path.has_value()) {
      dumped_file_paths.push_back(*path);
    }
  }
  if (!dumped_file_paths.empty()) {
    LOG_FIRST_N(INFO, 1) << "HloModule dump enabled with path prefix: "
                         << prefix << ", suffix: " << suffix;
  }
  return dumped_file_paths;
}

static void DumpHloModuleMetadata(
    const HloModuleMetadataProto& metadata, const CanonicalDebugOptions& opts,
    absl::flat_hash_set<int64_t>* dumped_module_ids) {
  // Return if metadata for this module has already been dumped.
  if (!dumped_module_ids->insert(metadata.canonical_module_id()).second) {
    return;
  }
  std::string filename = absl::StrFormat("module_%04d.metadata.textproto",
                                         metadata.canonical_module_id());
  std::string content;
  if (tsl::protobuf::TextFormat::PrintToString(metadata, &content)) {
    DumpToFileInDirImpl(filename, content, opts);
  } else {
    LOG(ERROR) << "Failed to convert HloModuleMetadataProto to text.";
  }
}

static absl::Mutex mu(absl::kConstInit);

// Maps a module's unique ID to a counter indicating how many times we've dumped
// this module during the compilation pipeline.  This lets us keep the filenames
// ordered nicely.
//
// Entries added here leak forever; we have no way to GC them when a module
// dies.  But we only add an entry if dumping is enabled for this module, and
// dumping a module leaks buffer space in stdout or bytes on disk *way* faster
// than this hashtable leaks memory.
static auto& module_id_to_step_number ABSL_GUARDED_BY(mu) =
    *new absl::flat_hash_map<int64_t, int64_t>();

// Maps a module's unique ID to a timestamp indicating when we've first dumped
// this module during the compilation pipeline and when we first started
// compiling this module.  This lets us keep the filenames ordered nicely.
//
// Entries added here leak forever; we have no way to GC them when a module
// dies.  But we only add an entry if dumping is enabled for this module, and
// dumping a module leaks buffer space in stdout or bytes on disk *way* faster
// than this hashtable leaks memory.
static auto& module_id_to_timestamp ABSL_GUARDED_BY(mu) =
    *new absl::flat_hash_map<int64_t, uint64_t>();

int64_t StepNumberForModule(const HloModule& module) {
  absl::MutexLock lock(&mu);
  return module_id_to_step_number[module.unique_id()]++;
}

}  // namespace

// Get a timestamp which we can use as a filename prefix specific to this
// module.
std::string TimestampFor(const HloModule& module) {
  if (!module.config().debug_options().xla_dump_include_timestamp()) {
    return "";
  }
  absl::MutexLock lock(&mu);
  auto timestamp_emplace = module_id_to_timestamp.try_emplace(
      module.unique_id(), tsl::Env::Default()->NowMicros());
  return std::to_string(timestamp_emplace.first->second);
}

std::string FilenameFor(int unique_id, string_view module_name,
                        string_view prefix, string_view suffix) {
  std::string filename;
  if (!prefix.empty()) {
    absl::StrAppend(&filename, prefix, ".");
  }
  absl::StrAppendFormat(&filename, "module_%04d", unique_id);
  if (!module_name.empty()) {
    absl::StrAppend(&filename, ".", module_name);
  }
  absl::StrAppend(&filename, ".", suffix);
  // Skip the module name if the resulting length is too long.
  if (!module_name.empty() && filename.size() > 255) {
    return FilenameFor(unique_id, "", prefix, suffix);
  }
  return filename;
}

std::string FilenameFor(const HloModule& module, string_view prefix,
                        string_view suffix) {
  return FilenameFor(module.unique_id(), module.name(), prefix, suffix);
}

void DumpToFileInDir(const HloModule& module, string_view file_prefix,
                     string_view file_suffix, string_view contents) {
  DumpToFileInDir(module.config().debug_options(),
                  FilenameFor(module, file_prefix, file_suffix), contents);
}

void DumpToFileInDir(const DebugOptions& debug_options,
                     absl::string_view filename, absl::string_view contents) {
  DumpToFileInDirImpl(filename, contents, CanonicalDebugOptions(debug_options));
}

void DumpToFileInDirOrStdout(const HloModule& module, string_view file_prefix,
                             string_view file_suffix, string_view contents) {
  DumpToFileInDirOrStdoutImpl(
      FilenameFor(module, file_prefix, file_suffix), contents,
      CanonicalDebugOptions(module.config().debug_options()));
}

void DumpToFileInDirOrStdout(const DebugOptions& debug_options, int unique_id,
                             string_view module_name, string_view file_prefix,
                             string_view file_suffix, string_view contents) {
  DumpToFileInDirOrStdoutImpl(
      FilenameFor(unique_id, module_name, file_prefix, file_suffix), contents,
      CanonicalDebugOptions(debug_options));
}

void DumpToFileInDirOrStdout(const HloModule& module, string_view file_prefix,
                             mlir::Operation* op) {
  CanonicalDebugOptions opts(module.config().debug_options());
  if (opts.dumping_to_stdout()) return op->dump();

  mlir::OpPrintingFlags print_flags = mlir::OpPrintingFlags();
  // Enable debug info so that it is easier to see the corresponding HLO node.
  if (file_prefix == "lmhlo") {
    print_flags.enableDebugInfo(/*enable=*/true,
                                /*prettyForm=*/opts.dump_mlir_pretty_form);
  }
  std::string content;
  llvm::raw_string_ostream string_stream(content);
  op->print(string_stream, print_flags);
  DumpToFileInDirOrStdoutImpl(FilenameFor(module, file_prefix, "mlir"), content,
                              opts);
}

void DumpProtobufToFile(const tsl::protobuf::Message& proto,
                        const DebugOptions& debug_options,
                        absl::string_view filename,
                        absl::AnyInvocable<absl::StatusOr<std::string>(
                            tsl::Env*, const tsl::protobuf::Message&)>
                            text_formatter) {
  CanonicalDebugOptions opts(debug_options);
  tsl::Env* env = tsl::Env::Default();
  const std::string& dir = opts.dump_to;
  if (!env->IsDirectory(dir).ok()) {
    auto status = env->RecursivelyCreateDir(dir);
    if (!status.ok()) {
      LOG(ERROR) << "Could not create directory " << dir
                 << " for dumping: " << status;
      return;
    }
  }
  if (!env->IsDirectory(dir).ok()) {
    return;
  }
  const std::string path = tsl::io::JoinPath(dir, filename);
  Status status;
  if (opts.dump_as_text) {
    if (text_formatter) {
      auto written_proto = text_formatter(env, proto);
      if (!written_proto.status().ok()) {
        LOG(ERROR) << "Failure with custom proto text formatting function. "
                   << "Could not write XLA data to " << filename << ": "
                   << written_proto.status();
        return;
      }
      status = tsl::WriteStringToFile(env, absl::StrCat(path, ".txt"),
                                      written_proto.value());
    } else {
      status = tsl::WriteTextProto(env, absl::StrCat(path, ".txt"), proto);
    }
  } else {
    status = tsl::WriteBinaryProto(env, absl::StrCat(path, ".pb"), proto);
  }
  if (!status.ok()) {
    LOG(ERROR) << "Could not write XLA data to " << filename << ": " << status;
  }
}

void DumpPerModuleProtobufToFile(const HloModule& module,
                                 const tsl::protobuf::Message& proto,
                                 const DebugOptions& debug_options,
                                 absl::string_view name,
                                 absl::AnyInvocable<absl::StatusOr<std::string>(
                                     tsl::Env*, const tsl::protobuf::Message&)>
                                     text_formatter) {
  const std::string filename = FilenameFor(module, TimestampFor(module), name);
  DumpProtobufToFile(proto, debug_options, filename, std::move(text_formatter));
}

void DumpHloModuleIfEnabled(const HloModule& module, string_view name) {
  CanonicalDebugOptions opts(module.config().debug_options());
  if (opts.should_dump_module(module.name())) {
    DumpHloModuleImpl(module, /*buffer_assn=*/nullptr, TimestampFor(module),
                      name, opts);
  }
}

void DumpHloModuleIfEnabled(const HloModule& module,
                            const BufferAssignment& buffer_assn,
                            string_view name) {
  CanonicalDebugOptions opts(module.config().debug_options());
  if (opts.should_dump_module(module.name())) {
    DumpHloModuleImpl(module, &buffer_assn, TimestampFor(module), name, opts);
  }
}

bool DumpingEnabledForHloModule(string_view hlo_module_name,
                                const DebugOptions& opts) {
  return CanonicalDebugOptions(opts).should_dump_module(hlo_module_name);
}

bool DumpingEnabledForHloPass(string_view hlo_pass_name,
                              const DebugOptions& opts) {
  return CanonicalDebugOptions(opts).should_dump_pass(hlo_pass_name);
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

  if (!opts.should_dump_pipeline(pipeline_name)) {
    return {};
  }

  int64_t step_number = StepNumberForModule(module);
  std::string timestamp = TimestampFor(module);

  std::string filename_suffix =
      StrFormat("%04d.%s.after_%s.before_%s", step_number, pipeline_name,
                after_pass_name, before_pass_name);
  return DumpHloModuleImpl(module, /*buffer_assn=*/nullptr, timestamp,
                           filename_suffix, opts);
}

void DumpHloModuleDuringPassIfEnabled(string_view pass_name,
                                      string_view step_name,
                                      const HloModule& module) {
  CanonicalDebugOptions opts(module.config().debug_options());
  if (!opts.should_dump_module(module.name()) ||
      !opts.should_dump_pass(pass_name)) {
    return;
  }

  int64_t step_number = StepNumberForModule(module);
  std::string timestamp = TimestampFor(module);

  std::string filename_suffix =
      StrFormat("%04d.%s.%s", step_number, pass_name, step_name);
  DumpHloModuleImpl(module, /*buffer_assn=*/nullptr, timestamp, filename_suffix,
                    opts);
}

void DumpHloSnapshotIfEnabled(const HloModule& module,
                              const HloSnapshot& snapshot) {
  CanonicalDebugOptions opts(module.config().debug_options());
  if (!opts.should_dump_module(module.name()) || !opts.dump_snapshots) {
    return;
  }
  int64_t execution_count;
  uint64_t timestamp;
  {
    static auto& module_id_to_execution_count ABSL_GUARDED_BY(mu) =
        *new absl::flat_hash_map<int64_t, int64_t>();
    absl::MutexLock lock(&mu);
    execution_count = module_id_to_execution_count[module.unique_id()]++;
    auto timestamp_emplace = module_id_to_timestamp.try_emplace(
        module.unique_id(), tsl::Env::Default()->NowMicros());
    timestamp = timestamp_emplace.first->second;
  }
  std::string filename =
      StrCat(FilenameFor(module, std::to_string(timestamp),
                         StrFormat("execution_%04d", execution_count)),
             ".hlo_snapshot.pb");
  if (opts.dumping_to_stdout()) {
    LOG(ERROR) << "Refusing to write HLO snapshot proto for " << filename
               << " to stdout. Pass --xla_dump_to=<path> to write to a file.";
    return;
  }
  std::string pb;
  if (!tsl::SerializeToStringDeterministic(snapshot, &pb)) {
    LOG(ERROR) << "Failed to serialize HLO snapshot proto " << filename;
  }
  DumpToFileInDirImpl(filename, pb, opts);
}

void DumpHloSnapshotIfEnabled(const HloSnapshot& snapshot,
                              const DebugOptions& opts) {
  CanonicalDebugOptions canonical_opts(opts);
  std::string name = snapshot.hlo().hlo_module().name();
  if (!canonical_opts.should_dump_module(name) ||
      !canonical_opts.dump_snapshots) {
    return;
  }

  // We don't have a unique id for an HloSnapshot, so in this overload we just
  // have to use its name.
  int64_t execution_count;
  {
    static auto& module_name_to_execution_count ABSL_GUARDED_BY(mu) =
        *new absl::flat_hash_map<std::string, int64_t>();
    absl::MutexLock lock(&mu);
    execution_count = module_name_to_execution_count[name]++;
  }
  std::string filename = StrFormat("module_%s.execution_%04d.hlo_snapshot.pb",
                                   name, execution_count);
  if (canonical_opts.dumping_to_stdout()) {
    LOG(ERROR) << "Refusing to write HLO snapshot proto for " << filename
               << " to stdout. Pass --xla_dump_to=<path> to write to a file.";
    return;
  }
  std::string pb;
  if (!tsl::SerializeToStringDeterministic(snapshot, &pb)) {
    LOG(ERROR) << "Failed to serialize HLO snapshot proto " << filename;
  }
  DumpToFileInDirImpl(filename, pb, canonical_opts);
}

void DumpHloModuleMetadataIfEnabled(const std::vector<HloModule*>& modules) {
  absl::flat_hash_set<int64_t> dumped_module_ids;
  for (const HloModule* module : modules) {
    CanonicalDebugOptions opts(module->config().debug_options());
    if (!opts.dump_module_metadata) {
      continue;
    }
    DumpHloModuleMetadata(module->metadata().proto(), opts, &dumped_module_ids);
    const std::optional<HloModuleMetadataProto>& prepartitioning_metadata =
        module->metadata().prepartitioning_metadata();
    if (prepartitioning_metadata.has_value()) {
      DumpHloModuleMetadata(*prepartitioning_metadata, opts,
                            &dumped_module_ids);
    }
  }
}

}  // namespace xla
