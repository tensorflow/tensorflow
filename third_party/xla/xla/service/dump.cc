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

#include <algorithm>
#include <cstdint>
#include <functional>
#include <iostream>
#include <memory>
#include <optional>
#include <queue>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/const_init.h"
#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/functional/any_invocable.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Transforms/LocationSnapshot.h"
#include "google/protobuf/descriptor.h"
#include "google/protobuf/text_format.h"
#include "riegeli/bytes/writer.h"
#include "xla/debug_options_flags.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/printer.h"
#include "xla/runtime/large_hlo_snapshot_serialization/serialization.h"
#include "xla/service/dump_options.h"
#include "xla/service/hlo.pb.h"
#include "xla/service/hlo_graph_dumper.h"
#include "xla/service/hlo_proto_util.h"
#include "xla/service/riegeli_file_writer_factory.h"
#include "xla/tsl/lib/io/zlib_compression_options.h"
#include "xla/tsl/lib/io/zlib_outputbuffer.h"
#include "xla/tsl/lib/strings/proto_serialization.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/file_system.h"
#include "xla/tsl/platform/file_system_helper.h"
#include "xla/tsl/platform/threadpool.h"
#include "xla/util.h"
#include "xla/util/split_proto/split_hlo_writer.h"
#include "tsl/platform/cpu_info.h"
#include "tsl/platform/path.h"
#include "tsl/platform/platform.h"
#include "tsl/platform/protobuf.h"
#include "tsl/profiler/lib/scoped_annotation.h"

// BuildData isn't available in OSS.
#if !TSL_IS_IN_OSS
#include "absl/base/builddata.h"
#endif  // TSL_IS_IN_OSS

namespace xla {

static absl::Mutex mu(absl::kConstInit);

// Set of module unique IDs that have been dumped. Used to enforce
// xla_dump_max_hlo_modules limit.
static auto& dumped_module_ids ABSL_GUARDED_BY(mu) =
    *new absl::flat_hash_set<int64_t>();

static absl::Mutex pending_async_dumps_mu(absl::kConstInit);
static int pending_async_dumps ABSL_GUARDED_BY(pending_async_dumps_mu) = 0;

static int64_t StepNumberForModule(const HloModule& module) {
  return module.NextDumpStepNumber();
}

absl::Status CreateDirIfNeeded(absl::string_view dir, tsl::Env* env) {
  if (!env->IsDirectory(dir).ok()) {
    absl::Status status = env->RecursivelyCreateDir(dir);
    // Two threads can race to observe the absence of the dump directory and
    // simultaneously try to create it, causing the "losing" thread to get a
    // "directory already exists" error.  We can work around this by checking
    // again whether the dir exists.
    if (!status.ok()) {
      status = env->IsDirectory(dir);
      if (!status.ok()) {
        LOG(ERROR) << "Could not create directory: " << dir
                   << ". Error: " << status;
        return status;
      }
    }
  }
  return absl::OkStatus();
}

std::string RenderGraph(absl::string_view label, const HloModule& module,
                        RenderedGraphFormat format,
                        bool show_fusion_subcomputations,
                        const DebugOptions* dump_options) {
  HloRenderOptions hlo_render_options;
  hlo_render_options.show_fusion_subcomputations = show_fusion_subcomputations;
  const DebugOptions& opts =
      dump_options ? *dump_options : module.config().debug_options();
  absl::StatusOr<std::string> rendered_graph = RenderGraph(
      *module.entry_computation(), label, opts, format, hlo_render_options);
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

static DumpOptions GetDumpOptions(string_view module_name,
                                  const DebugOptions& debug_options) {
  return DumpOptions::Build(debug_options, module_name);
}

static DumpOptions GetDumpOptions(const HloModule& module,
                                  const DebugOptions* override_opts = nullptr) {
  const DebugOptions& debug_options =
      override_opts ? *override_opts : module.config().debug_options();
  return DumpOptions::Build(debug_options, module.name());
}

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

static absl::Status WriteStringToFile(tsl::Env* env, const std::string& fname,
                                      DataProducer& data_producer,
                                      bool compressed) {
  std::unique_ptr<tsl::WritableFile> file;
  RETURN_IF_ERROR(env->NewWritableFile(fname, &file));
  if (compressed) {
    auto gz_opts = tsl::io::ZlibCompressionOptions::GZIP();
    tsl::io::ZlibOutputBuffer gz_file(file.get(), gz_opts.input_buffer_size,
                                      gz_opts.output_buffer_size, gz_opts);
    RETURN_IF_ERROR(gz_file.Init());
    while (auto next_producer = data_producer.Next()) {
      RETURN_IF_ERROR(gz_file.Append(next_producer()));
    }
    return gz_file.Close();
  }
  while (auto next_producer = data_producer.Next()) {
    RETURN_IF_ERROR(file->Append(next_producer()));
  }
  return file->Close();
}

static absl::Status WriteStringToFile(tsl::Env* env, const std::string& fname,
                                      absl::string_view data, bool compressed) {
  if (!compressed) {
    return tsl::WriteStringToFile(env, fname, data);
  }
  std::unique_ptr<tsl::WritableFile> file;
  RETURN_IF_ERROR(env->NewWritableFile(fname, &file));
  auto gz_opts = tsl::io::ZlibCompressionOptions::GZIP();
  tsl::io::ZlibOutputBuffer gz_file(file.get(), gz_opts.input_buffer_size,
                                    gz_opts.output_buffer_size, gz_opts);
  RETURN_IF_ERROR(gz_file.Init());
  RETURN_IF_ERROR(gz_file.Append(data));
  return gz_file.Close();
}

static std::optional<std::string> GetDumpFilePath(string_view filename,
                                                  const DumpOptions& opts) {
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
  if (!CreateDirIfNeeded(dir, env).ok()) {
    return std::nullopt;
  }

  // Make sure we are not going to dump more modules than the user has asked.
  if (opts.dump_max_hlo_modules > 0) {
    absl::MutexLock lock(mu);
    if (dumped_module_ids.size() >= opts.dump_max_hlo_modules) {
      LOG(ERROR) << "Have already dumped " << dumped_module_ids.size()
                 << " modules, more than the limit of "
                 << opts.dump_max_hlo_modules;
      return std::nullopt;
    }
  }

  return tsl::io::JoinPath(dir, SanitizeFileName(std::string(filename)));
}

static std::optional<std::string> DumpToFileInDirImpl(string_view filename,
                                                      string_view contents,
                                                      const DumpOptions& opts,
                                                      bool compress = false) {
  auto file_path = GetDumpFilePath(filename, opts);
  if (!file_path) {
    return std::nullopt;
  }

  auto status =
      WriteStringToFile(tsl::Env::Default(), *file_path, contents, compress);
  if (!status.ok()) {
    LOG(ERROR) << "Could not write XLA debug data to " << *file_path << ": "
               << status;
    return std::nullopt;
  }

  return file_path;
}

static std::optional<std::string> DumpHloModuleRiegeli(
    string_view filename, HloProto proto, const DumpOptions& opts) {
  auto file_path = GetDumpFilePath(filename, opts);
  if (!file_path) {
    return std::nullopt;
  }
  std::unique_ptr<riegeli::Writer> writer = CreateRiegeliFileWriter(*file_path);
  if (writer == nullptr) {
    return std::nullopt;
  }

  absl::Status status = WriteSplitHloProto(std::move(proto), std::move(writer));
  if (!status.ok()) {
    LOG(ERROR) << "Could not write XLA debug data to " << *file_path << ": "
               << status;
    return std::nullopt;
  }
  return file_path;
}

static std::optional<std::string> DumpToFileInDirImpl(
    string_view filename, DataProducer& data_producer, const DumpOptions& opts,
    bool compress = false) {
  auto file_path = GetDumpFilePath(filename, opts);
  if (!file_path) {
    return std::nullopt;
  }

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
    string_view filename, string_view contents, const DumpOptions& opts) {
  // Dump to stdout if that's called for.
  if (opts.dumping_to_stdout()) {
    absl::MutexLock lock(stdout_dump_mutex);
    std::cout << "*** Begin " << filename << " ***\n"
              << contents << "\n*** End " << filename << " ***" << std::endl;
    return std::nullopt;
  }

  // Otherwise, dump to a file.
  return DumpToFileInDirImpl(filename, contents, opts);
}

static std::optional<std::string> DumpToFileInDirOrStdoutImpl(
    string_view filename, DataProducer& data_producer,
    const DumpOptions& opts) {
  // Dump to stdout if that's called for.
  if (opts.dumping_to_stdout()) {
    absl::MutexLock lock(stdout_dump_mutex);
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
std::vector<std::string> GetDumpFilenames(const HloModule& module,
                                          const BufferAssignment* buffer_assn,
                                          absl::string_view prefix,
                                          absl::string_view suffix,
                                          const DumpOptions& opts,
                                          const DebugOptions& debug_options) {
  std::vector<std::string> filenames;
  std::string filename = FilenameFor(module, prefix, suffix);
  std::string dir = opts.dump_to;
  auto get_abs_path = [&](string_view rel_fn) {
    return tsl::io::JoinPath(dir, SanitizeFileName(std::string(rel_fn)));
  };

  if (opts.dump_as_text) {
    filenames.push_back(get_abs_path(StrCat(filename, ".txt")));
    if (buffer_assn) {
      filenames.push_back(
          get_abs_path(StrCat(filename, "-buffer-assignment.txt")));
      if (debug_options.xla_dump_buffer_assignment_analysis()) {
        filenames.push_back(
            get_abs_path(StrCat(filename, "-buffer-assignment-values.txt")));
        filenames.push_back(get_abs_path(StrCat(filename, "-live-range.txt")));
      }
      filenames.push_back(
          get_abs_path(StrCat(filename, "-memory-usage-report.txt")));
    }
  }

  if (opts.dump_as_proto) {
    std::string proto_suffix =
        opts.dump_compress_protos ? ".hlo.pb.gz" : ".hlo.pb";
    filenames.push_back(get_abs_path(StrCat(filename, proto_suffix)));
    if (buffer_assn) {
      std::string report_suffix = opts.dump_compress_protos
                                      ? "-memory-usage-report.pb.gz"
                                      : "-memory-usage-report.pb";
      filenames.push_back(get_abs_path(StrCat(filename, report_suffix)));
    }
  }

  if (opts.dump_as_riegeli) {
    filenames.push_back(get_abs_path(StrCat(filename, ".riegeli")));
  }

  if (opts.dump_as_dot) {
    filenames.push_back(get_abs_path(StrFormat("%s.dot", filename)));
  }

  if (opts.dump_as_html) {
    filenames.push_back(get_abs_path(StrFormat("%s.html", filename)));
    if (absl::StrContains(filename, kAfterOptimizationsDumpName)) {
      filenames.push_back(
          get_abs_path(StrFormat("%s.top_level.html", filename)));
    }
  }

  if (opts.dump_fusion_visualization) {
    for (const HloComputation* computation :
         module.MakeNonfusionComputations()) {
      if (IsTrivial(*computation)) {
        continue;
      }
      std::string fusion_filename =
          FilenameFor(module, computation->name(), "_fusion.html");
      filenames.push_back(get_abs_path(fusion_filename));
    }
  }

  if (opts.dump_fdo_profiles) {
    filenames.push_back(get_abs_path(StrFormat("%s.fdo_profile", filename)));
  }

  if (opts.dump_as_url) {
    if (!opts.dumping_to_stdout()) {
      filenames.push_back(get_abs_path(StrFormat("%s.url", filename)));
    }
  }
  return filenames;
}

tsl::thread::ThreadPool* GetDumpThreadPool(const DebugOptions& debug_options) {
  static auto* const thread_pool = [](const DebugOptions& opts) {
    int num_threads = 4;
    if (opts.has_xla_async_hlo_dump_max_threads()) {
      num_threads = opts.xla_async_hlo_dump_max_threads();
    } else {
      num_threads = std::max(4, std::min(32, tsl::port::MaxParallelism() / 4));
    }
    num_threads = std::max(1, num_threads);
    return new tsl::thread::ThreadPool(tsl::Env::Default(), "hlo_async_dump",
                                       num_threads);
  }(debug_options);
  return thread_pool;
}

void IncrementPendingAsyncDumps() {
  absl::MutexLock lock(&pending_async_dumps_mu);
  pending_async_dumps++;
}

void DecrementPendingAsyncDumps() {
  absl::MutexLock lock(&pending_async_dumps_mu);
  pending_async_dumps--;
}

int GetPendingAsyncDumps() {
  absl::MutexLock lock(&pending_async_dumps_mu);
  return pending_async_dumps;
}

void WaitForPendingAsyncDumpsLessThan(int limit) {
  absl::MutexLock lock(&pending_async_dumps_mu);
  pending_async_dumps_mu.Await(absl::Condition(
      +[](int* limit_ptr) ABSL_SHARED_LOCKS_REQUIRED(pending_async_dumps_mu) {
        return pending_async_dumps < *limit_ptr;
      },
      &limit));
}

static std::vector<std::string> DumpHloModuleImpl(
    const HloModule& module, const BufferAssignment* buffer_assn,
    string_view prefix, string_view suffix, const DumpOptions& opts,
    const DebugOptions& debug_options) {
  tsl::profiler::ScopedAnnotation annotation([&] {
    return absl::StrFormat("XlaDumpHloModule:#module=%s,program_id=%d#",
                           module.name(), module.unique_id());
  });

  if (!opts.should_dump_module(module.name())) {
    return {};
  }

  if (opts.dump_max_hlo_modules > 0) {
    absl::MutexLock lock(&mu);
    dumped_module_ids.insert(module.unique_id());
  }

  std::vector<std::string> written_paths;
  std::string filename = FilenameFor(module, prefix, suffix);

  HloPrintOptions print_options = debug_options.xla_dump_hlo_as_long_text()
                                      ? HloPrintOptions::Default()
                                      : HloPrintOptions::ShortParsable();
  print_options.set_print_large_constants(
      debug_options.xla_dump_large_constants());
  print_options.set_print_metadata(!debug_options.xla_dump_disable_metadata());
  print_options.set_syntax_sugar_async_ops(
      debug_options.xla_syntax_sugar_async_ops());
  print_options.set_print_inline_stack_frames(
      debug_options.xla_hlo_print_inline_stack_frames());
  print_options.set_compact_gte(debug_options.xla_dump_compact_gte());

  if (opts.dump_as_text) {
    auto path = DumpToFileInDirOrStdoutImpl(
        StrCat(filename, ".txt"), module.ToString(print_options), opts);
    bool printed = opts.dumping_to_stdout() || path.has_value();
    if (path.has_value()) {
      written_paths.push_back(*path);
    }

    if (printed && buffer_assn) {
      auto ba_path = DumpToFileInDirOrStdoutImpl(
          StrCat(filename, "-buffer-assignment.txt"), buffer_assn->ToString(),
          opts);
      if (ba_path.has_value()) {
        written_paths.push_back(*ba_path);
      }

      if (debug_options.xla_dump_buffer_assignment_analysis()) {
        auto bav_path = DumpToFileInDirOrStdoutImpl(
            StrCat(filename, "-buffer-assignment-values.txt"),
            buffer_assn->ValuesToString(), opts);
        if (bav_path.has_value()) {
          written_paths.push_back(*bav_path);
        }

        std::string live_range_str =
            buffer_assn->HasHloLiveRange()
                ? buffer_assn->hlo_live_range().ToString()
                : "HloLiveRange not available (finalized or constructed from "
                  "proto)";
        auto lr_path = DumpToFileInDirOrStdoutImpl(
            StrCat(filename, "-live-range.txt"), live_range_str, opts);
        if (lr_path.has_value()) {
          written_paths.push_back(*lr_path);
        }
      }

      auto mr_path = DumpToFileInDirOrStdoutImpl(
          StrCat(filename, "-memory-usage-report.txt"),
          buffer_assn->MemoryUsageReport(), opts);
      if (mr_path.has_value()) {
        written_paths.push_back(*mr_path);
      }
    }
  }

  HloProto module_proto;
  bool target_proto_created = false;
  auto init_module_proto = [&]() {
    if (target_proto_created) return;
    module_proto =
        buffer_assn ? MakeHloProto(module, *buffer_assn) : MakeHloProto(module);
    target_proto_created = true;
  };

  if (opts.dump_as_proto) {
    init_module_proto();
    std::string pb;
    if (!tsl::SerializeToStringDeterministic(module_proto, &pb)) {
      pb = "Failed to serialize HLO module proto.";
    }
    std::string proto_suffix =
        opts.dump_compress_protos ? ".hlo.pb.gz" : ".hlo.pb";
    auto proto_path = DumpToFileInDirImpl(StrCat(filename, proto_suffix), pb,
                                          opts, opts.dump_compress_protos);
    if (proto_path.has_value()) {
      written_paths.push_back(*proto_path);
    }

    if (buffer_assn) {
      MemoryUsageReportProto memory_report_proto =
          buffer_assn->GetMemoryUsageReportProto();
      std::string memory_report_pb;
      if (!tsl::SerializeToStringDeterministic(memory_report_proto,
                                               &memory_report_pb)) {
        memory_report_pb = "Failed to serialize memory usage report proto.";
      }
      std::string report_suffix = opts.dump_compress_protos
                                      ? "-memory-usage-report.pb.gz"
                                      : "-memory-usage-report.pb";
      auto report_path =
          DumpToFileInDirImpl(StrCat(filename, report_suffix), memory_report_pb,
                              opts, opts.dump_compress_protos);
      if (report_path.has_value()) {
        written_paths.push_back(*report_path);
      }
    }
  }

  if (opts.dump_as_riegeli) {
    init_module_proto();
    auto path = DumpHloModuleRiegeli(StrCat(filename, ".riegeli"),
                                     std::move(module_proto), opts);
    if (path.has_value()) {
      written_paths.push_back(*path);
    }
  }

  if (opts.dump_as_dot) {
    auto path = DumpToFileInDirImpl(
        StrCat(filename, ".dot"),
        RenderGraph(filename, module, RenderedGraphFormat::kDot,
                    /*show_fusion_subcomputations=*/true, &debug_options),
        opts);
    if (path.has_value()) {
      written_paths.push_back(*path);
    }
  }

  if (opts.dump_as_html) {
    auto path = DumpToFileInDirImpl(
        StrCat(filename, ".html"),
        RenderGraph(filename, module, RenderedGraphFormat::kHtml,
                    /*show_fusion_subcomputations=*/true, &debug_options),
        opts);
    if (path.has_value()) {
      written_paths.push_back(*path);
    }
    if (absl::StrContains(filename, kAfterOptimizationsDumpName)) {
      auto tl_path = DumpToFileInDirImpl(
          StrCat(filename, ".top_level.html"),
          RenderGraph(filename, module, RenderedGraphFormat::kHtml,
                      /*show_fusion_subcomputations=*/false, &debug_options),
          opts);
      if (tl_path.has_value()) {
        written_paths.push_back(*tl_path);
      }
    }
  }

  if (opts.dump_fusion_visualization) {
    for (const HloComputation* computation :
         module.MakeNonfusionComputations()) {
      if (IsTrivial(*computation)) {
        continue;
      }
      absl::StatusOr<std::string> rendered_graph =
          WrapFusionExplorer(*computation);
      if (!rendered_graph.ok()) {
        continue;
      }
      auto path = DumpToFileInDirImpl(
          FilenameFor(module, computation->name(), "_fusion.pyz"),
          *rendered_graph, opts);
      if (path.has_value()) {
        written_paths.push_back(*path);
      }
    }
  }

  if (opts.dump_fdo_profiles) {
    auto path =
        DumpToFileInDirImpl(StrCat(filename, ".fdo_profile"),
                            std::string(module.config().fdo_profile()), opts);
    if (path.has_value()) {
      written_paths.push_back(*path);
    }
  }

  if (opts.dump_as_url) {
    std::string url =
        RenderGraph(filename, module, RenderedGraphFormat::kUrl,
                    /*show_fusion_subcomputations=*/true, &debug_options);
    auto path =
        DumpToFileInDirOrStdoutImpl(StrCat(filename, ".url"), url, opts);
    if (path.has_value()) {
      std::cout << *path << " --> " << url << std::endl;
      written_paths.push_back(*path);
    }
  }

  if (!written_paths.empty()) {
    LOG_FIRST_N(INFO, 1) << "HloModule dump enabled with path prefix: "
                         << prefix << ", suffix: " << suffix;
  }

  return written_paths;
}

static void DumpHloModuleMetadata(
    const HloModuleMetadataProto& metadata, const DumpOptions& opts,
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

static std::vector<std::string> DumpHloModuleIfEnabledImpl(
    const HloModule& module, const BufferAssignment* buffer_assn,
    const DebugOptions* maybe_dump_options, string_view name) {
  const DebugOptions& dump_options = maybe_dump_options
                                         ? *maybe_dump_options
                                         : module.config().debug_options();
  DumpOptions opts = GetDumpOptions(module, maybe_dump_options);
  if (opts.should_dump_module(module.name())) {
    std::vector<std::string> filepaths = DumpHloModuleImpl(
        module, /*buffer_assn=*/buffer_assn,
        TimestampFor(module, &dump_options), name, opts, dump_options);
    std::optional<std::string> maybe_debug_options_filepath =
        DumpNonDefaultDebugOptions(module, kNonDefaultDebugOptionsDumpSuffix,
                                   maybe_dump_options);
    if (maybe_debug_options_filepath.has_value()) {
      filepaths.push_back(*maybe_debug_options_filepath);
    }
    return filepaths;
  }
  return {};
}

}  // namespace

// Get a timestamp which we can use as a filename prefix specific to this
// module.
std::string TimestampFor(const HloModule& module,
                         const DebugOptions* debug_options_override) {
  const DebugOptions& opts = debug_options_override
                                 ? *debug_options_override
                                 : module.config().debug_options();
  if (!opts.xla_dump_include_timestamp()) {
    return "";
  }
  return std::to_string(module.GetDumpTimestamp());
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

#if !TSL_IS_IN_OSS
  absl::string_view cl_number = BuildData::Changelist();
  if (!cl_number.empty()) {
    absl::StrAppend(&filename, ".cl_", cl_number);
  }
#endif  // !TSL_IS_IN_OSS

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
  DumpToFileInDirImpl(FilenameFor(module, file_prefix, file_suffix), contents,
                      GetDumpOptions(module));
}

void DumpToFileInDir(const DebugOptions& debug_options,
                     absl::string_view filename, absl::string_view contents) {
  DumpToFileInDirImpl(filename, contents, DumpOptions::Build(debug_options));
}

void DumpToFileInDirOrStdout(const HloModule& module, string_view file_prefix,
                             string_view file_suffix, string_view contents) {
  DumpToFileInDirOrStdoutImpl(FilenameFor(module, file_prefix, file_suffix),
                              contents, GetDumpOptions(module));
}

void DumpToFileInDirOrStdout(const DebugOptions& debug_options, int unique_id,
                             string_view module_name, string_view file_prefix,
                             string_view file_suffix, string_view contents) {
  DumpToFileInDirOrStdoutImpl(
      FilenameFor(unique_id, module_name, file_prefix, file_suffix), contents,
      GetDumpOptions(module_name, debug_options));
}

void DumpToFileInDirOrStdout(const HloModule& module, string_view file_prefix,
                             mlir::Operation* op) {
  DumpOptions opts = GetDumpOptions(module);
  if (opts.dumping_to_stdout()) {
    return op->dump();
  }

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
                            text_formatter,
                        const DumpOptions* override_opts) {
  DumpOptions opts =
      override_opts ? *override_opts : DumpOptions::Build(debug_options);
  tsl::Env* env = tsl::Env::Default();
  const std::string& dir = opts.dump_to;
  if (dir.empty()) {
    return;
  }
  if (!CreateDirIfNeeded(dir, env).ok()) {
    return;
  }
  const std::string path = tsl::io::JoinPath(dir, filename);
  absl::Status status;
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
  DumpOptions opts = GetDumpOptions(module.name(), debug_options);
  DumpProtobufToFile(proto, debug_options, filename, std::move(text_formatter),
                     &opts);
}

void DumpPerExecutionProtobufToFile(
    const HloModule& module, const tsl::protobuf::Message& proto,
    const DebugOptions& debug_options, absl::string_view name,
    absl::AnyInvocable<
        absl::StatusOr<std::string>(tsl::Env*, const tsl::protobuf::Message&)>
        text_formatter) {
  int64_t execution_count = module.NextDumpExecutionCount();

  const std::string filename = FilenameFor(
      module, name, absl::StrFormat("execution_%04d", execution_count));
  DumpOptions opts = GetDumpOptions(module.name(), debug_options);
  DumpProtobufToFile(proto, debug_options, filename, std::move(text_formatter),
                     &opts);
}

std::string GetRepeatedValueAsString(
    const tsl::protobuf::Reflection* reflection,
    const DebugOptions& debug_options,
    const tsl::protobuf::FieldDescriptor* field, int index) {
  switch (field->type()) {
    case tsl::protobuf::FieldDescriptor::TYPE_INT32:
      return std::to_string(
          reflection->GetRepeatedInt32(debug_options, field, index));
    case tsl::protobuf::FieldDescriptor::TYPE_INT64:
      return std::to_string(
          reflection->GetRepeatedInt64(debug_options, field, index));
    case tsl::protobuf::FieldDescriptor::TYPE_UINT32:
      return std::to_string(
          reflection->GetRepeatedUInt32(debug_options, field, index));
    case tsl::protobuf::FieldDescriptor::TYPE_UINT64:
      return std::to_string(
          reflection->GetRepeatedUInt64(debug_options, field, index));
    case tsl::protobuf::FieldDescriptor::TYPE_DOUBLE:
      return std::to_string(
          reflection->GetRepeatedDouble(debug_options, field, index));
    case tsl::protobuf::FieldDescriptor::TYPE_FLOAT:
      return std::to_string(
          reflection->GetRepeatedFloat(debug_options, field, index));
    case tsl::protobuf::FieldDescriptor::TYPE_BOOL:
      return reflection->GetRepeatedBool(debug_options, field, index) ? "true"
                                                                      : "false";
    case tsl::protobuf::FieldDescriptor::TYPE_ENUM:
      return std::string(
          reflection->GetRepeatedEnum(debug_options, field, index)->name());
    case tsl::protobuf::FieldDescriptor::TYPE_STRING:
      return "\"" + reflection->GetRepeatedString(debug_options, field, index) +
             "\"";
    case tsl::protobuf::FieldDescriptor::TYPE_MESSAGE: {
      tsl::protobuf::TextFormat::Printer tsl_printer;
      tsl_printer.SetInitialIndentLevel(1);
      std::string result;
      tsl_printer.PrintToString(
          reflection->GetRepeatedMessage(debug_options, field, index), &result);
      return "{\n" + result + "}";
    }
    default:
      return "Unsupported field type";
  }
}

std::string GetValueAsString(const tsl::protobuf::Reflection* reflection,
                             const DebugOptions& debug_options,
                             const tsl::protobuf::FieldDescriptor* field) {
  // Based on the field type, get the value and convert it to a string
  switch (field->type()) {
    case tsl::protobuf::FieldDescriptor::TYPE_INT32:
      return std::to_string(reflection->GetInt32(debug_options, field));
    case tsl::protobuf::FieldDescriptor::TYPE_INT64:
      return std::to_string(reflection->GetInt64(debug_options, field));
    case tsl::protobuf::FieldDescriptor::TYPE_UINT32:
      return std::to_string(reflection->GetUInt32(debug_options, field));
    case tsl::protobuf::FieldDescriptor::TYPE_UINT64:
      return std::to_string(reflection->GetUInt64(debug_options, field));
    case tsl::protobuf::FieldDescriptor::TYPE_DOUBLE:
      return std::to_string(reflection->GetDouble(debug_options, field));
    case tsl::protobuf::FieldDescriptor::TYPE_FLOAT:
      return std::to_string(reflection->GetFloat(debug_options, field));
    case tsl::protobuf::FieldDescriptor::TYPE_BOOL:
      return reflection->GetBool(debug_options, field) ? "true" : "false";
    case tsl::protobuf::FieldDescriptor::TYPE_ENUM:
      return std::string(reflection->GetEnum(debug_options, field)->name());
    case tsl::protobuf::FieldDescriptor::TYPE_STRING:
      return "\"" + reflection->GetString(debug_options, field) + "\"";
    case tsl::protobuf::FieldDescriptor::TYPE_MESSAGE: {
      tsl::protobuf::TextFormat::Printer tsl_printer;
      tsl_printer.SetSingleLineMode(false);
      std::string result;
      tsl_printer.PrintToString(reflection->GetMessage(debug_options, field),
                                &result);
      return "{\n" + result + "}";
    }
    default:
      return "Unsupported field type";
  }
}

std::string GetNonDefaultDebugOptions(const DebugOptions& debug_options) {
  // Create a default DebugOptions to compare against
  DebugOptions default_options = DefaultDebugOptionsIgnoringFlags();
  std::string non_default_options;

  // Use protobuf reflection to compare fields
  const tsl::protobuf::Descriptor* descriptor = debug_options.GetDescriptor();
  const tsl::protobuf::Reflection* reflection = debug_options.GetReflection();

  // Iterate through all fields
  for (int i = 0; i < descriptor->field_count(); i++) {
    const tsl::protobuf::FieldDescriptor* field = descriptor->field(i);

    if (field->is_repeated()) {
      // Handle repeated fields by comparing the values
      int repeated_count = reflection->FieldSize(debug_options, field);
      int default_count = reflection->FieldSize(default_options, field);

      // Only process if the repeated field has values
      if (repeated_count > 0) {
        std::vector<std::string> debug_values(repeated_count);
        std::vector<std::string> default_values(default_count);

        // Collect all values from debug_options
        for (int j = 0; j < repeated_count; j++) {
          debug_values[j] =
              GetRepeatedValueAsString(reflection, debug_options, field, j);
        }

        // Collect all values from default_options
        for (int j = 0; j < default_count; j++) {
          default_values[j] =
              GetRepeatedValueAsString(reflection, default_options, field, j);
        }

        // Sort both vectors for comparison
        std::sort(debug_values.begin(), debug_values.end());
        std::sort(default_values.begin(), default_values.end());

        // Compare the sorted vectors
        if (debug_values != default_values) {
          // Values differ, append all debug values to output
          for (const auto& value : debug_values) {
            absl::StrAppend(&non_default_options, field->name(), ": ", value,
                            "\n");
          }
        }
      }
      continue;
    }

    if (GetValueAsString(reflection, debug_options, field) !=
        GetValueAsString(reflection, default_options, field)) {
      absl::StrAppend(&non_default_options, field->name(), ": ",
                      GetValueAsString(reflection, debug_options, field), "\n");
    }
  }

  return non_default_options;
}

std::optional<std::string> DumpNonDefaultDebugOptions(
    const HloModule& module, absl::string_view suffix,
    const DebugOptions* dump_options) {
  // Substance of the which-options-have-non-default-values logic always based
  // on the options embedded in the HloModule
  const DebugOptions& debug_options = module.config().debug_options();
  std::string filename = FilenameFor(module, "", suffix);
  std::string nonDefaultDebugOptions = GetNonDefaultDebugOptions(debug_options);
  // Options steering where the dump is actually written to can be overridden
  DumpOptions opts = GetDumpOptions(module, dump_options);
  return DumpToFileInDirImpl(filename, nonDefaultDebugOptions, opts);
}

std::vector<std::string> DumpHloModuleIfEnabled(
    const HloModule& module, string_view name,
    const DebugOptions* dump_options) {
  return DumpHloModuleIfEnabledImpl(module, /*buffer_assn=*/nullptr,
                                    dump_options, name);
}

std::vector<std::string> DumpHloModuleIfEnabled(
    const HloModule& module, const BufferAssignment& buffer_assn,
    string_view name) {
  return DumpHloModuleIfEnabledImpl(module, &buffer_assn,
                                    /*maybe_dump_options=*/nullptr, name);
}

std::vector<std::string> DumpHloModuleProtoIfEnabled(
    const HloModuleProto& module_proto, absl::string_view name) {
  auto config = xla::HloModule::CreateModuleConfigFromProto(
      module_proto, xla::GetDebugOptionsFromFlags());
  if (!config.ok()) {
    LOG(ERROR) << "Failed to create module config: " << config.status();
    return {};
  }

  auto module_or = xla::HloModule::CreateFromProto(module_proto, *config);
  if (!module_or.ok()) {
    LOG(ERROR) << "Failed to create module from proto: " << module_or.status();
    return {};
  }
  auto module = std::move(*module_or);

  DumpOptions opts = GetDumpOptions(*module);
  if (opts.should_dump_module(module->name())) {
    return DumpHloModuleImpl(*module, /*buffer_assn=*/nullptr,
                             TimestampFor(*module), name, opts,
                             module->config().debug_options());
  }
  return {};
}

void DumpHloConfigIfEnabled(const HloModule& module) {
  if (!module.config().debug_options().xla_dump_full_hlo_config()) {
    return;
  }

  DumpOptions opts = GetDumpOptions(module);
  if (opts.dumping_to_stdout()) {
    VLOG(2) << "Refusing to write HLO config proto for " << module.name()
            << " to stdout. Pass --xla_dump_to=<path> to write to a file.";
    return;
  }
  std::string config_str;
  if (tsl::protobuf::TextFormat::PrintToString(module.config().ToProto(),
                                               &config_str)) {
    std::string filename = FilenameFor(module, "", "config.pbtxt");
    DumpToFileInDirImpl(filename, config_str, opts);
  } else {
    VLOG(1) << "Failed to convert HloModuleConfig to text. Module: "
            << module.name();
  }
}

bool DumpingEnabledForHloModule(string_view hlo_module_name,
                                const DebugOptions& opts) {
  return DumpOptions::Build(opts).should_dump_module(hlo_module_name);
}

bool DumpingEnabledForHloPass(string_view hlo_pass_name,
                              const DebugOptions& opts) {
  return DumpOptions::Build(opts).should_dump_pass(hlo_pass_name);
}

bool DumpingEnabledForEmitter(string_view emitter_name,
                              const DebugOptions& opts) {
  return DumpOptions::Build(opts).should_dump_emitter(emitter_name);
}

bool DumpingToStdout(const DebugOptions& opts) {
  return DumpOptions::Build(opts).dumping_to_stdout();
}

static bool ShouldDumpHloModuleBetweenPasses(absl::string_view pipeline_name,
                                             absl::string_view before_pass_name,
                                             absl::string_view after_pass_name,
                                             const HloModule& module) {
  DumpOptions opts = GetDumpOptions(module);
  if (!opts.should_dump_module(module.name())) {
    return false;
  }

  if (!opts.should_dump_pass(before_pass_name) &&
      !opts.should_dump_pass(after_pass_name)) {
    return false;
  }

  if (!opts.should_dump_pipeline(pipeline_name)) {
    return false;
  }
  return true;
}

static std::vector<std::string> GetDumpFilenamesBetweenPasses(
    absl::string_view pipeline_name, absl::string_view before_pass_name,
    absl::string_view after_pass_name, const HloModule& module,
    int64_t step_number) {
  if (!ShouldDumpHloModuleBetweenPasses(pipeline_name, before_pass_name,
                                        after_pass_name, module)) {
    return {};
  }
  DumpOptions opts = GetDumpOptions(module);
  std::string timestamp = TimestampFor(module);
  std::string filename_suffix =
      StrFormat("%04d.%s.after_%s.before_%s", step_number, pipeline_name,
                after_pass_name, before_pass_name);
  return GetDumpFilenames(module, /*buffer_assn=*/nullptr, timestamp,
                          filename_suffix, opts,
                          module.config().debug_options());
}

std::vector<std::string> DumpHloModuleBetweenPassesIfEnabled(
    absl::string_view pipeline_name, absl::string_view before_pass_name,
    absl::string_view after_pass_name, const HloModule& module) {
  if (!ShouldDumpHloModuleBetweenPasses(pipeline_name, before_pass_name,
                                        after_pass_name, module)) {
    return {};
  }

  const DebugOptions& debug_options = module.config().debug_options();
  int64_t step_number = StepNumberForModule(module);

  std::string timestamp = TimestampFor(module);
  std::string filename_suffix =
      StrFormat("%04d.%s.after_%s.before_%s", step_number, pipeline_name,
                after_pass_name, before_pass_name);
  DumpOptions opts = GetDumpOptions(module);

  if (debug_options.xla_async_hlo_dump()) {
    std::vector<std::string> filenames = GetDumpFilenamesBetweenPasses(
        pipeline_name, before_pass_name, after_pass_name, module, step_number);
    if (filenames.empty()) {
      return {};
    }

    int limit = debug_options.xla_async_hlo_dump_max_pending();
    if (limit <= 0) {
      limit = 20;
    }
    WaitForPendingAsyncDumpsLessThan(limit);

    auto clone = module.Clone("");
    clone->set_unique_id(module.unique_id());
    clone->metadata()->set_canonical_module_id(
        module.metadata().proto().canonical_module_id());

    IncrementPendingAsyncDumps();
    GetDumpThreadPool(debug_options)
        ->AsExecutor()
        ->Execute([clone = std::move(clone), timestamp = std::move(timestamp),
                   filename_suffix = std::move(filename_suffix),
                   opts = std::move(opts), debug_options]() {
          DumpHloModuleImpl(*clone, /*buffer_assn=*/nullptr, timestamp,
                            filename_suffix, opts, debug_options);
          DecrementPendingAsyncDumps();
        });
    return filenames;
  } else {
    return DumpHloModuleImpl(module, /*buffer_assn=*/nullptr, timestamp,
                             filename_suffix, opts, debug_options);
  }
}

void DumpHloModuleDuringPassIfEnabled(string_view pass_name,
                                      string_view step_name,
                                      const HloModule& module) {
  DumpOptions opts = GetDumpOptions(module);
  if (!opts.should_dump_module(module.name()) ||
      !opts.should_dump_pass(pass_name)) {
    return;
  }

  int64_t step_number = StepNumberForModule(module);
  std::string timestamp = TimestampFor(module);

  std::string filename_suffix =
      StrFormat("%04d.%s.%s", step_number, pass_name, step_name);
  DumpHloModuleImpl(module, /*buffer_assn=*/nullptr, timestamp, filename_suffix,
                    opts, module.config().debug_options());
}

void DumpHloSnapshotIfEnabled(const HloModule& module,
                              const HloSnapshot& snapshot) {
  DumpOptions opts = GetDumpOptions(module);
  if (!opts.should_dump_module(module.name()) || !opts.dump_snapshots) {
    return;
  }
  int64_t execution_count = module.NextDumpExecutionCount();
  uint64_t timestamp = module.GetDumpTimestamp();
  if (opts.dump_max_hlo_modules > 0) {
    absl::MutexLock lock(mu);
    dumped_module_ids.insert(module.unique_id());
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
  std::string name = snapshot.hlo().hlo_module().name();
  DumpOptions canonical_opts = GetDumpOptions(name, opts);
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
    absl::MutexLock lock(mu);
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

void DumpHloUnoptimizedSnapshotIfEnabled(
    const HloUnoptimizedSnapshot& hlo_snapshot, const DebugOptions& opts) {
  std::string name = hlo_snapshot.hlo_module().name();
  DumpOptions canonical_opts = GetDumpOptions(name, opts);
  if (!canonical_opts.dump_unoptimized_snapshots) {
    return;
  }

  if (hlo_snapshot.partitions_size() == 0) {
    LOG(ERROR) << "Refusing to write unoptimized HLO snapshot proto for module "
               << name << ": no partitions input found.";
    return;
  }
  int64_t execution_count;
  {
    static absl::Mutex mu(absl::kConstInit);
    static auto& module_id_to_execution_count ABSL_GUARDED_BY(mu) =
        *new absl::flat_hash_map<int64_t, int64_t>();
    absl::MutexLock lock(mu);
    execution_count =
        module_id_to_execution_count[hlo_snapshot.hlo_module().id()]++;
  }
  std::string filename = FilenameFor(
      hlo_snapshot.hlo_module().id(), hlo_snapshot.hlo_module().name(), "",
      absl::StrFormat("execution_%04d.hlo_unoptimized_snapshot",
                      execution_count));
  // We use a custom proto binary serialization for HloUnoptimizedSnapshot to
  // bypass the 2GiB proto size limitation.
  if (canonical_opts.dump_as_proto && !canonical_opts.dump_as_text) {
    tsl::Env* env = tsl::Env::Default();
    const std::string& dir = canonical_opts.dump_to;
    if (dir.empty()) {
      return;
    }
    if (!CreateDirIfNeeded(dir, env).ok()) {
      return;
    }
    const std::string path = tsl::io::JoinPath(dir, filename);

    std::unique_ptr<tsl::WritableFile> file;
    absl::Status s = env->NewWritableFile(absl::StrCat(path, ".pb"), &file);
    if (!s.ok()) {
      LOG(ERROR) << "Could not create file " << filename << ": " << s;
      return;
    }
    tsl::WritableFileCopyingOutputStream output_stream(file.get());
    // TODO - b/457711066: Add missing include once capybara can re-write the
    // dependency correctly.
    tsl::protobuf::io::CopyingOutputStreamAdaptor adaptor(&output_stream);
    if (!SerializeHloUnoptimizedSnapshot(hlo_snapshot, &adaptor).ok()) {
      LOG(ERROR) << "Failed to serialize HLO unoptimized snapshot proto";
    }
    adaptor.Flush();
    if (!file->Close().ok()) {
      LOG(ERROR) << "Failed to close HLO unoptimized snapshot proto file";
    }
  } else {
    DumpProtobufToFile(hlo_snapshot, opts, filename, nullptr, &canonical_opts);
  }
}

void DumpHloModuleMetadataIfEnabled(HloModule* module) {
  absl::flat_hash_set<int64_t> dumped_module_ids;
  DumpOptions opts = GetDumpOptions(*module);
  if (!module->config().debug_options().xla_dump_module_metadata()) {
    return;
  }
  DumpHloModuleMetadata(module->metadata()->proto(), opts, &dumped_module_ids);
  const std::optional<HloModuleMetadataProto>& prepartitioning_metadata =
      module->metadata()->prepartitioning_metadata();
  if (prepartitioning_metadata.has_value()) {
    DumpHloModuleMetadata(*prepartitioning_metadata, opts, &dumped_module_ids);
  }
}

absl::Status DumpProtoToDirectory(const tsl::protobuf::Message& message,
                                  absl::string_view directory,
                                  absl::string_view file_name,
                                  std::string* full_path) {
  tsl::Env* env = tsl::Env::Default();
  RETURN_IF_ERROR(env->RecursivelyCreateDir(directory));
  RETURN_IF_ERROR(CreateDirIfNeeded(directory, env));
  std::string safe_file_name = SanitizeFileName(std::string(file_name)) + ".pb";
  std::string full_path_impl;
  if (!full_path) {
    full_path = &full_path_impl;
  }
  *full_path = tsl::io::JoinPath(directory, safe_file_name);
  return tsl::WriteBinaryProto(env, *full_path, message);
}

void WaitForAllAsyncDumps() {
  absl::MutexLock lock(&pending_async_dumps_mu);
  pending_async_dumps_mu.Await(absl::Condition(
      +[](void*) ABSL_SHARED_LOCKS_REQUIRED(pending_async_dumps_mu) {
        return pending_async_dumps == 0;
      },
      nullptr));
}

}  // namespace xla
