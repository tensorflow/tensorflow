/* Copyright 2026 The OpenXLA Authors.

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

// Tool to perform offline comparison of XLA tensor numerics or recover tensor
// summaries from a single XLA job.
//
// This tool has two modes:
// 1. Comparison mode: If `--baseline_dump_dir` is provided, the tool compares
//    tensor numerics between a baseline and a target XLA job.
// 2. Single run mode: If `--baseline_dump_dir` is NOT provided, the tool
//    recovers and summarizes tensors from a single XLA job (the target).
//
// == Log Generation ==
// This tool consumes logs dumped from an XLA job. To generate the necessary
// logs, the XLA job must be compiled and run with the following flags:
//
// 1.  `--xla_tpu_enable_comparison_mode_for_module=<your_module_name>`:
//     Enables dumping for the specified HLO module.
// 2.  `--xla_tpu_dump_logs_to_dir=<output_directory>`: Specifies the directory
//     where tensor logs will be written.
// 3.  `--xla_tpu_dump_launch_info_to_dir=<output_directory>`: Specifies the
//     directory where launch information will be written.
// 4.  [Optional]
//     `--xla_tpu_comparison_log_all_ops_for_module=<your_module_name>`: Enables
//     logging of all HLO ops for the specified HLO module. Alternatively, one
//     can also use `set_xla_metadata` and set `xla_log_for_comparison=True` in
//     Jax.
//
// The directories specified in flags 2 and 3 should be provided to this tool
// using `--target_dump_dir` (for target job) and `--baseline_dump_dir` (for
// baseline job).
//
// == Usage ==
//
// Comparison mode:
//   Provide `--target_dump_dir` and `--baseline_dump_dir`. The tool outputs a
//   riegeli file containing ComparisonResultProto to `--output_dir`.
//
// Single run mode:
//   Provide `--target_dump_dir` but NOT `--baseline_dump_dir`. The tool
//   outputs a riegeli file containing RecoveredTensorSummaryProto to
//   `--output_dir`.
#include <algorithm>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/cleanup/cleanup.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/flags/flag.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "re2/re2.h"
#include "riegeli/base/maker.h"
#include "riegeli/base/types.h"
#include "riegeli/bytes/fd_reader.h"
#include "riegeli/bytes/fd_writer.h"
#include "riegeli/bytes/joining_reader.h"
#include "riegeli/records/record_position.h"
#include "riegeli/records/record_reader.h"
#include "riegeli/records/record_writer.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/tools/comparison/comparison_hlo_dumper.h"
#include "xla/hlo/tools/comparison/comparison_result.pb.h"
#include "xla/hlo/tools/comparison/comparison_result_utils.h"
#include "xla/hlo/tools/comparison/offline_utils.h"
#include "xla/hlo/tools/comparison/original_tensor_summary_utils.h"
#include "xla/hlo/tools/comparison/tensor_summary_util.h"
#include "xla/hlo/tools/comparison/xla_job_comparator.h"
#include "xla/hlo/tools/comparison/xla_job_recoverer.h"
#include "xla/hlo/tools/hlo_diff/hlo_gumgraph_diff.h"
#include "xla/literal.h"
#include "xla/runtime/device_id.h"
#include "xla/service/computation_placer.h"
#include "xla/tools/debug_event.pb.h"
#include "xla/tsl/platform/env.h"
#include "xla/xla.pb.h"
#include "tsl/platform/init_main.h"
#include "tsl/platform/path.h"
#include "tsl/platform/random.h"

ABSL_FLAG(std::string, hlo_module_name, "",
          "Name of the HLO module to compare.");
ABSL_FLAG(std::string, baseline_hlo_module_name, "",
          "Name of the baseline HLO module, if different from target.");
ABSL_FLAG(std::vector<std::string>, target_dump_dir, {},
          "Directories for target dump files.");
ABSL_FLAG(std::vector<std::string>, baseline_dump_dir, {},
          "Directories for baseline dump files.");
ABSL_FLAG(std::optional<int64_t>, target_launch_barrier_id, std::nullopt,
          "Target launch barrier ID.");
ABSL_FLAG(std::optional<int64_t>, baseline_launch_barrier_id, std::nullopt,
          "Baseline launch barrier ID.");
ABSL_FLAG(std::string, output_dir, "",
          "Directory to write comparison results.");
ABSL_FLAG(std::optional<std::string>, temp_file_dir, std::nullopt,
          "Directory to write temporary files.");
ABSL_FLAG(bool, regenerate_output_files, false,
          "If true, regenerate intermediate files (recovered summaries and "
          "comparison results), even if they already exist.");
ABSL_FLAG(bool, generate_hlo_html_dump, true,
          "If true, generate HLO HTML dumps with diff score overlays.");
ABSL_FLAG(::absl::Duration, progress_reporter_log_interval,
          ::absl::Milliseconds(100),
          "The minimum interval between log messages for ProgressReporter.");

namespace xla::numerics::comparison {
namespace {

using ::xla::LogData;
using ::xla::LogHloOutputKind;

std::vector<::xla::comparison::DimSplitSpec> ConvertDimSplitSpecs(
    absl::Span<const ::xla::LogHloOutputMetadata::DimSplitSpecProto>
        dim_split_spec) {
  std::vector<::xla::comparison::DimSplitSpec> split_spec;
  split_spec.reserve(dim_split_spec.size());
  for (const auto& dim_split_spec_proto : dim_split_spec) {
    split_spec.push_back(::xla::comparison::DimSplitSpec{
        dim_split_spec_proto.dim_index(), dim_split_spec_proto.block_count()});
  }
  return split_spec;
}

// Proto conversion utilities
TensorKeyProto ProtoFromTensorKey(const TensorKey& tk) {
  TensorKeyProto p;
  p.set_instruction_name(tk.instruction_name);
  for (int64_t i : tk.shape_index) {
    p.add_shape_index(i);
  }
  return p;
}

AbsoluteScopedTensorKeyProto ProtoFromScopedTensorKey(
    const AbsoluteScopedTensorKey& key) {
  AbsoluteScopedTensorKeyProto p;
  *p.mutable_tensor_key() = ProtoFromTensorKey(key.tensor_key);
  for (const auto& si : key.scope_instructions) {
    *p.add_scope_instructions() = ProtoFromScopeInstruction(si);
  }
  return p;
}

TensorSummaryProto CreateTensorSummaryProto(
    const xla::comparison::FloatSummary& float_summary) {
  TensorSummaryProto p;
  for (const auto& s : float_summary.split_spec) {
    auto* sp = p.add_split_spec();
    sp->set_dim_index(s.dim_index);
    sp->set_block_count(s.block_count);
  }
  for (const auto& bs : float_summary.block_summaries) {
    auto* bp = p.add_block_summaries();
    for (int64_t i : bs.block_indices) {
      bp->add_block_indices(i);
    }
    bp->set_min(bs.min);
    bp->set_max(bs.max);
    bp->set_mean(bs.mean);
    bp->set_stddev(bs.stddev);
    bp->set_count(bs.count);
  }
  return p;
}

void PrintCreationMetrics(
    const XlaJobComparator::CreationMetrics& creation_metrics) {
  const auto& comp_metrics = creation_metrics.comparator_metrics;

  absl::FPrintF(stderr, "\n====== Creation Metrics ======\n");

  int64_t total_pairs = comp_metrics.unchanged_tensor_pair_count +
                        comp_metrics.changed_tensor_pair_count;
  absl::FPrintF(
      stderr,
      "HLO Diff: %d pairs (%d unchanged, %d changed) were found by HLO diff "
      "between baseline (%d tensors) and target (%d tensors) original "
      "modules.\n",
      total_pairs, comp_metrics.unchanged_tensor_pair_count,
      comp_metrics.changed_tensor_pair_count,
      comp_metrics.baseline_tensor_count, comp_metrics.target_tensor_count);
}

void PrintProcessingMetrics(
    absl::Span<const XlaJobComparator::ProcessingMetrics>
        processing_metrics_vec) {
  for (int i = 0; i < processing_metrics_vec.size(); ++i) {
    const auto& metrics = processing_metrics_vec[i];
    absl::FPrintF(stderr, "\n====== Processing Metrics for replica %d ======\n",
                  i);

    absl::FPrintF(stderr,
                  "Comparator: received %d baseline and %d target tensor "
                  "summaries for comparison.\n",
                  metrics.comparator_metrics.received_baseline_tensor_summaries,
                  metrics.comparator_metrics.received_target_tensor_summaries);
    double baseline_untranslatable_perc =
        metrics.comparator_metrics.received_baseline_tensor_summaries == 0
            ? 0.0
            : 100.0 *
                  metrics.comparator_metrics
                      .untranslatable_baseline_tensor_summaries /
                  metrics.comparator_metrics.received_baseline_tensor_summaries;
    double target_untranslatable_perc =
        metrics.comparator_metrics.received_target_tensor_summaries == 0
            ? 0.0
            : 100.0 *
                  metrics.comparator_metrics
                      .untranslatable_target_tensor_summaries /
                  metrics.comparator_metrics.received_target_tensor_summaries;

    absl::FPrintF(
        stderr,
        "Comparator: %d (%.2f%%) baseline summaries and %d (%.2f%%) "
        "target summaries are untranslatable (no counterpart found by "
        "HLO diff).\n",
        metrics.comparator_metrics.untranslatable_baseline_tensor_summaries,
        baseline_untranslatable_perc,
        metrics.comparator_metrics.untranslatable_target_tensor_summaries,
        target_untranslatable_perc);

    int64_t baseline_translatable =
        metrics.comparator_metrics.received_baseline_tensor_summaries -
        metrics.comparator_metrics.untranslatable_baseline_tensor_summaries;
    int64_t target_translatable =
        metrics.comparator_metrics.received_target_tensor_summaries -
        metrics.comparator_metrics.untranslatable_target_tensor_summaries;
    double baseline_compared_perc =
        baseline_translatable == 0
            ? 0.0
            : 100.0 * metrics.comparator_metrics.compared_pairs_count /
                  baseline_translatable;
    double target_compared_perc =
        target_translatable == 0
            ? 0.0
            : 100.0 * metrics.comparator_metrics.compared_pairs_count /
                  target_translatable;

    absl::FPrintF(
        stderr,
        "Comparator: %d pairs of tensor summaries compared. This is %.2f%% of "
        "translatable baseline summaries and %.2f%% of translatable target "
        "summaries.\n",
        metrics.comparator_metrics.compared_pairs_count, baseline_compared_perc,
        target_compared_perc);
  }
}

void PrintComparisonSummary(int replica_id,
                            absl::string_view comparison_results_path) {
  riegeli::RecordReader reader(
      riegeli::Maker<riegeli::FdReader>(comparison_results_path));
  ComparisonResultProto result;
  int64_t total_compared = 0;
  int64_t comparison_issues = 0;
  std::vector<double> diff_scores;
  double max_diff_score = -1.0;
  ComparisonResultProto max_diff_result;
  while (reader.ReadRecord(result)) {
    if (result.baseline_tensor_summaries_size() == 0 &&
        result.target_tensor_summaries_size() == 0) {
      continue;
    }
    total_compared++;
    double score = result.diff_score();
    if (score == -1.0) {
      comparison_issues++;
    } else {
      diff_scores.push_back(score);
      if (score > max_diff_score) {
        max_diff_score = score;
        max_diff_result = result;
      }
    }
  }
  CHECK(reader.Close()) << reader.status();

  if (total_compared == 0) {
    return;
  }

  absl::FPrintF(stderr, "\n====== Comparison Summary for replica %d ======\n",
                replica_id);
  absl::FPrintF(stderr, "Compared %d pairs of tensor summaries.\n",
                total_compared);
  if (!diff_scores.empty()) {
    absl::FPrintF(stderr, "Max diff score among mismatching pairs: %f\n",
                  max_diff_score);
    absl::FPrintF(stderr, "  Baseline tensor key: %s\n",
                  absl::StrCat(ScopedTensorKey::FromProto(
                                   max_diff_result.baseline_tensor_key())
                                   .ToString()));
    absl::FPrintF(stderr, "  Target tensor key: %s\n",
                  absl::StrCat(ScopedTensorKey::FromProto(
                                   max_diff_result.target_tensor_key())
                                   .ToString()));
  } else {
    absl::FPrintF(stderr, "Max diff score among mismatching pairs: N/A\n");
  }

  absl::FPrintF(stderr, "Uncomparable pairs: %10d (%6.2f%%)\n",
                comparison_issues, 100.0 * comparison_issues / total_compared);

  int64_t comparable_pairs_count = total_compared - comparison_issues;
  absl::FPrintF(
      stderr,
      "Among %d comparable pairs, accumulated diff score distribution:\n",
      comparable_pairs_count);

  if (diff_scores.empty()) {
    return;
  }

  std::vector<double> buckets = {0, 1e-2, 1e-1, 1, 2, 10, 100};
  std::vector<int64_t> counts(buckets.size(), 0);
  for (double score : diff_scores) {
    for (int i = 0; i < buckets.size(); ++i) {
      if (score <= buckets[i]) {
        counts[i]++;
        break;
      }
    }
  }

  int64_t accumulated_count = 0;
  for (int i = 0; i < buckets.size(); ++i) {
    if (counts[i] == 0) {
      continue;
    }
    accumulated_count += counts[i];
    absl::FPrintF(stderr, "  %-18s: %10d (%6.2f%%)\n",
                  absl::StrFormat("<=%g", buckets[i]), accumulated_count,
                  100.0 * accumulated_count / comparable_pairs_count);
  }
}

// OSS Replacement for RiegeliTpuLogReader
class RiegeliShardReader : public riegeli::JoiningReader<riegeli::FdReader<>> {
 public:
  RiegeliShardReader(absl::string_view log_dir,
                     absl::string_view filename_prefix)
      : log_dir_(log_dir), filename_prefix_(filename_prefix), next_id_(0) {}

 protected:
  bool OpenShardImpl() override {
    const std::string filepath = tsl::io::JoinPath(
        log_dir_,
        absl::StrFormat("%s%05d.riegeli", filename_prefix_, next_id_));
    if (absl::IsNotFound(tsl::Env::Default()->FileExists(filepath))) {
      return false;
    }
    shard() = riegeli::FdReader(filepath);
    if (!shard().ok()) {
      LOG(ERROR) << "Unable to open Riegeli file: " << filepath
                 << ", status: " << shard().status();
      return false;
    }
    ++next_id_;
    return true;
  }

 private:
  std::string log_dir_;
  std::string filename_prefix_;
  int64_t next_id_;
};

class SimpleRiegeliLogReader {
 public:
  SimpleRiegeliLogReader(absl::string_view log_dir,
                         absl::string_view filename_prefix)
      : reader_(RiegeliShardReader(log_dir, filename_prefix)) {}

  bool Read(std::string* data) { return reader_.ReadRecord(*data); }

  riegeli::RecordPosition Pos() const { return reader_.pos(); }

 private:
  riegeli::RecordReader<RiegeliShardReader> reader_;
};

struct LogReaderInfo {
  std::unique_ptr<SimpleRiegeliLogReader> reader;
  ComparisonVariant variant;
  int64_t device_id;
  bool active = true;
  std::string path;
  int64_t total_log_size_bytes = 0;
};

bool ParseShardedRiegeliPath(absl::string_view path, std::string* prefix,
                             int* shard_num) {
  static constexpr LazyRE2 kShardedRiegeliRegex = {"(.*)(\\d{5})\\.riegeli$"};
  return RE2::FullMatch(path, *kShardedRiegeliRegex, prefix, shard_num);
}

void AddLogReaders(ComparisonVariant variant,
                   absl::Span<const std::string> log_files,
                   absl::Span<const int64_t> device_ids,
                   std::vector<LogReaderInfo>& log_readers) {
  absl::flat_hash_map<std::string, std::vector<std::pair<int, int>>>
      sharded_files;  // prefix -> list of <shard_num, index_in_log_files>
  for (int i = 0; i < log_files.size(); ++i) {
    std::string prefix;
    int shard_num;
    if (absl::EndsWith(log_files[i], ".riegeli") &&
        ParseShardedRiegeliPath(log_files[i], &prefix, &shard_num)) {
      sharded_files[prefix].emplace_back(shard_num, i);
    }
  }

  for (auto& [prefix, shards] : sharded_files) {  // NOLINT
    std::sort(shards.begin(), shards.end());
    std::vector<std::string> paths;
    paths.reserve(shards.size());
    int64_t total_size_bytes = 0;
    for (const auto& p : shards) {
      paths.push_back(log_files[p.second]);
      uint64_t file_size;
      absl::Status status =
          tsl::Env::Default()->GetFileSize(log_files[p.second], &file_size);
      if (!status.ok()) {
        LOG(ERROR) << "Failed to get file size for " << log_files[p.second]
                   << ": " << status;
      } else {
        total_size_bytes += file_size;
      }
    }
    int first_idx = shards[0].second;
    log_readers.push_back(LogReaderInfo{
        std::make_unique<SimpleRiegeliLogReader>(tsl::io::Dirname(prefix),
                                                 tsl::io::Basename(prefix)),
        variant, device_ids[first_idx], true, paths[0], total_size_bytes});
  }
}

void ProcessLogData(LogReaderInfo& r_info, const LogData& log_data,
                    const std::function<absl::Status(
                        ComparisonVariant, const AbsoluteScopedTensorKey&,
                        XlaJobRecoverer::DeviceTensorSummary)>& process_fn) {
  if (log_data.hlo_output_metadata().kind() ==
      LogHloOutputKind::LOG_HLO_OUTPUT_BLOCK_SUMMARY) {
    AbsoluteScopedTensorKey key =
        GetAbsoluteScopedTensorKey(log_data.hlo_output_metadata());
    const auto& specs = log_data.hlo_output_metadata().dim_split_spec();
    std::vector<::xla::LogHloOutputMetadata::DimSplitSpecProto>
        dim_split_spec_vec(specs.begin(), specs.end());
    auto literal_or = Literal::CreateFromProto(log_data.literal());
    if (!literal_or.ok()) {
      LOG(ERROR) << "Failed to create literal from proto in " << r_info.path
                 << ": " << literal_or.status();
      return;
    }
    auto float_summary_or = ::xla::comparison::GetFloatSummary(
        *std::move(literal_or), ConvertDimSplitSpecs(dim_split_spec_vec));
    if (!float_summary_or.ok()) {
      LOG(ERROR) << "Failed to get float summary from " << r_info.path << ": "
                 << float_summary_or.status();
      return;
    }
    XlaJobRecoverer::DeviceTensorSummary summary{
        GlobalDeviceId(r_info.device_id), *std::move(float_summary_or)};
    absl::Status process_status =
        process_fn(r_info.variant, key, std::move(summary));
    if (!process_status.ok()) {
      LOG(ERROR) << "Failed to process device tensor summary from "
                 << r_info.path << ": " << process_status;
    }
  }
}

void ProcessLogFiles(ComparisonVariant variant,
                     std::vector<LogReaderInfo>& log_readers,
                     const std::function<absl::Status(
                         ComparisonVariant, const AbsoluteScopedTensorKey&,
                         XlaJobRecoverer::DeviceTensorSummary)>& process_fn) {
  if (log_readers.empty()) {
    return;
  }
  // Here we just use the first reader's total size as the total size for all
  // readers because the logs should contain the same number of entries for
  // all devices due to SPMD.
  int64_t total_size_per_reader = log_readers[0].total_log_size_bytes;
  int active_readers = log_readers.size();
  ProgressReporter progress_reporter(
      absl::StrFormat("Processing %s log files", ToString(variant)),
      total_size_per_reader, /*use_percent=*/true,
      absl::GetFlag(FLAGS_progress_reporter_log_interval));
  xla::LogData log_data;
  while (active_readers > 0) {
    for (int i = 0; i < log_readers.size(); ++i) {
      LogReaderInfo& r_info = log_readers[i];
      if (!r_info.active) {
        continue;
      }
      if (i == 0) {
        progress_reporter.Report(r_info.reader->Pos().numeric());
      }
      std::string raw_data;
      bool read_success = r_info.reader->Read(&raw_data);
      if (read_success) {
        if (!log_data.ParseFromString(raw_data)) {
          LOG(ERROR) << "Failed to parse LogData from string in "
                     << r_info.path;
          r_info.active = false;
          active_readers--;
          continue;
        }
        ProcessLogData(r_info, log_data, process_fn);
      } else {
        r_info.active = false;
        active_readers--;
      }
    }
  }
}

struct RecoveryRunResults {
  std::vector<std::string> output_filenames;
};

// Recovers tensor summaries for a single job (baseline or target) and writes
// them to riegeli files.
// Note: this function moves `run_data->device_assignment`, leaving it null.
absl::StatusOr<RecoveryRunResults> RecoverJob(
    ComparisonVariant variant, RunData* run_data, absl::string_view output_dir,
    absl::string_view temp_file_base_path) {
  int replica_count = run_data->device_assignment->replica_count();

  // Create list of output filenames
  std::vector<std::string> output_filenames;
  output_filenames.reserve(replica_count);
  for (int i = 0; i < replica_count; ++i) {
    std::string output_filename = tsl::io::JoinPath(
        output_dir,
        absl::StrFormat("recovered_summaries.%c-%s_%d.R_%d.riegeli",
                        variant == ComparisonVariant::kBaseline ? 'B' : 'T',
                        run_data->module_name, run_data->launch_barrier_id, i));
    output_filenames.push_back(output_filename);
  }

  // If not regenerating, check if all output files exist.
  if (!absl::GetFlag(FLAGS_regenerate_output_files)) {
    bool all_exist = true;
    for (const std::string& filename : output_filenames) {
      uint64_t file_size;
      if (!tsl::Env::Default()->GetFileSize(filename, &file_size).ok()) {
        all_exist = false;
        break;
      }
    }
    if (all_exist) {
      absl::FPrintF(
          stderr,
          "Skipping recovery for %s because output files already exist and "
          "--regenerate_output_files=false.\n",
          ToString(variant));
      return RecoveryRunResults{std::move(output_filenames)};
    }
  }

  absl::FPrintF(stderr, "\n====== %s Recovery Stats ======\n",
                ToString(variant));

  // Setup output writers.
  std::vector<std::unique_ptr<riegeli::RecordWriterBase>> writers(
      replica_count);
  for (int i = 0; i < replica_count; ++i) {
    writers[i] = riegeli::Maker<riegeli::RecordWriter>(
        riegeli::Maker<riegeli::FdWriter>(output_filenames[i]),
        riegeli::RecordWriterBase::Options().set_transpose(true));
  }
  absl::Mutex writer_mutex;

  // Setup recoverer and callback.
  struct RecoveryStats {
    int64_t recovered_tensors_count = 0;
    absl::flat_hash_set<std::string> instructions_with_recovered_tensors;
  };
  std::vector<RecoveryStats> recovery_stats(replica_count);
  XlaJobRecoverer::OriginalTensorSummaryCallbackGetter callback_getter =
      [&](int replica_id) {
        return [&, replica_id](
                   const AbsoluteScopedTensorKey& original_tensor_key,
                   std::shared_ptr<
                       const tensor_transformation::TensorTransformation>
                       pending_transformation,
                   const OriginalTensorSummary& original_tensor_summary)
                   -> absl::Status {
          RecoveredTensorSummaryProto proto = CreateRecoveredTensorSummaryProto(
              original_tensor_key, pending_transformation,
              original_tensor_summary);
          absl::MutexLock lock(writer_mutex);
          writers[replica_id]->WriteRecord(proto);
          recovery_stats[replica_id].recovered_tensors_count++;
          recovery_stats[replica_id].instructions_with_recovered_tensors.insert(
              original_tensor_key.tensor_key.instruction_name);
          return absl::OkStatus();
        };
      };

  std::string sequenced_file_base_path = tsl::io::JoinPath(
      output_dir,
      absl::StrFormat("sequenced_summaries.%s_%d", run_data->module_name,
                      run_data->launch_barrier_id));
  auto recoverer_or = XlaJobRecoverer::Create(
      std::move(run_data->device_assignment), run_data->original_module.get(),
      run_data->optimized_module.get(), std::move(callback_getter),
      temp_file_base_path, sequenced_file_base_path, variant);
  if (!recoverer_or.ok()) {
    return recoverer_or.status();
  }
  auto recoverer_pair = *std::move(recoverer_or);
  auto& recoverer = recoverer_pair.first;
  auto& creation_metrics = recoverer_pair.second;

  // Setup log readers.
  std::vector<LogReaderInfo> log_readers;
  AddLogReaders(variant, run_data->log_files,
                run_data->device_ids_for_log_files, log_readers);

  // Process log files.
  ProcessLogFiles(
      variant, log_readers,
      [&](ComparisonVariant, const AbsoluteScopedTensorKey& key,
          XlaJobRecoverer::DeviceTensorSummary summary) -> absl::Status {
        return recoverer->ProcessDeviceTensorSummary(key, std::move(summary));
      });

  auto prop_metrics_or = recoverer->Finish();
  if (!prop_metrics_or.ok()) {
    return prop_metrics_or.status();
  }
  const auto& prop_metrics = *prop_metrics_or;

  // Print recovery stats here
  int64_t total_instructions = run_data->original_module->instruction_count();
  int64_t total_recovered_tensors = 0;
  absl::flat_hash_set<std::string> total_instructions_with_recovered_tensors;
  for (int i = 0; i < replica_count; ++i) {
    total_recovered_tensors += recovery_stats[i].recovered_tensors_count;
    total_instructions_with_recovered_tensors.insert(
        recovery_stats[i].instructions_with_recovered_tensors.begin(),
        recovery_stats[i].instructions_with_recovered_tensors.end());
    absl::FPrintF(
        stderr,
        "Replica %d: recovered %d tensor summaries for %d / %d (%.2f%%) HLO "
        "instructions, among which %d are are recoverable by original "
        "values.\n",
        i, recovery_stats[i].recovered_tensors_count,
        recovery_stats[i].instructions_with_recovered_tensors.size(),
        total_instructions,
        total_instructions == 0
            ? 0.0
            : 100.0 *
                  recovery_stats[i].instructions_with_recovered_tensors.size() /
                  total_instructions,
        creation_metrics.recoverable_tensor_keys.size());
    if (i < prop_metrics.size()) {
      absl::FPrintF(
          stderr,
          "  Propagator: %d tensors recovered from runtime, %d derived by "
          "propagator, and %d unrecoverable call chains skipped.\n",
          prop_metrics[i].recovered_from_runtime_count,
          prop_metrics[i].total_propagated_tensor_count -
              prop_metrics[i].recovered_from_runtime_count,
          prop_metrics[i].skipped_unrecoverable_tensor_summaries);
    }
  }
  absl::FPrintF(stderr,
                "Total: recovered %d tensor summaries for %d / %d (%.2f%%) HLO "
                "instructions across all replicas.\n",
                total_recovered_tensors,
                total_instructions_with_recovered_tensors.size(),
                total_instructions,
                total_instructions == 0
                    ? 0.0
                    : 100.0 * total_instructions_with_recovered_tensors.size() /
                          total_instructions);

  // Close writers.
  for (int i = 0; i < replica_count; ++i) {
    CHECK(writers[i]->Close()) << writers[i]->status();
  }

  return RecoveryRunResults{std::move(output_filenames)};
}

void Run() {
  std::string target_module_name = absl::GetFlag(FLAGS_hlo_module_name);
  std::vector<std::string> target_dump_dirs =
      absl::GetFlag(FLAGS_target_dump_dir);
  std::string output_dir = absl::GetFlag(FLAGS_output_dir);

  QCHECK(!target_module_name.empty()) << "--hlo_module_name must be provided.";
  QCHECK(!target_dump_dirs.empty()) << "--target_dump_dir must be provided.";
  QCHECK(!output_dir.empty()) << "--output_dir must be provided.";

  std::vector<std::string> baseline_dump_dirs =
      absl::GetFlag(FLAGS_baseline_dump_dir);
  bool single_run_mode = baseline_dump_dirs.empty();

  // Create output directory if it doesn't exist.
  absl::Status status = tsl::Env::Default()->RecursivelyCreateDir(output_dir);
  CHECK(status.ok() || absl::IsAlreadyExists(status)) << status;

  std::string temp_file_base_dir;
  if (absl::GetFlag(FLAGS_temp_file_dir).has_value()) {
    temp_file_base_dir = *absl::GetFlag(FLAGS_temp_file_dir);
  } else {
    temp_file_base_dir = "/tmp/xla_numerics_comparison_tool/";
  }

  std::string temp_file_dir =
      tsl::io::JoinPath(temp_file_base_dir, absl::StrCat(tsl::random::New64()));

  status = tsl::Env::Default()->RecursivelyCreateDir(temp_file_dir);
  if (!status.ok() && !absl::IsAlreadyExists(status)) {
    LOG(QFATAL) << "Failed to create temporary directory '" << temp_file_dir
                << "': " << status;
  }

  absl::Cleanup temp_dir_cleanup = [&temp_file_dir] {
    int64_t undeleted_files = 0;
    int64_t undeleted_dirs = 0;
    absl::Status delete_status = tsl::Env::Default()->DeleteRecursively(
        temp_file_dir, &undeleted_files, &undeleted_dirs);
    if (!delete_status.ok()) {
      LOG(ERROR) << "Failed to delete temporary directory " << temp_file_dir
                 << ": " << delete_status;
    }
  };

  std::string temp_file_base_path =
      tsl::io::JoinPath(temp_file_dir, "comparison_temp");

  if (single_run_mode) {
    absl::FPrintF(stderr, "\n====== Running in Single Run Mode ======\n");
    auto target_data_or =
        LoadRunData(target_dump_dirs, target_module_name,
                    absl::GetFlag(FLAGS_target_launch_barrier_id));
    if (!target_data_or.ok()) {
      LOG(FATAL) << "Failed to load target data: " << target_data_or.status();
    }
    RunData target_data = *std::move(target_data_or);
    const int replica_count = target_data.device_assignment->replica_count();
    if (replica_count == 0) {
      LOG(FATAL) << "Replica count is 0.";
    }
    HloModule* target_original_module = target_data.original_module.get();

    auto target_results_or =
        RecoverJob(ComparisonVariant::kTarget, &target_data, output_dir,
                   temp_file_base_path);
    if (!target_results_or.ok()) {
      LOG(FATAL) << "Failed to recover target data: "
                 << target_results_or.status();
    }
    RecoveryRunResults target_results = *std::move(target_results_or);

    absl::FPrintF(stderr, "\n====== Recovered Summaries Files ======\n");
    for (const auto& filename : target_results.output_filenames) {
      absl::FPrintF(stderr, "%s\n", filename);
    }

    for (int i = 0; i < replica_count; ++i) {
      if (absl::GetFlag(FLAGS_generate_hlo_html_dump)) {
        GenerateSingleHloHtmlDump(i, *target_original_module, output_dir,
                                  target_results.output_filenames[i]);
      }
    }
  } else {
    absl::FPrintF(stderr, "\n====== Running in Comparison Mode ======\n");
    std::string baseline_module_name =
        absl::GetFlag(FLAGS_baseline_hlo_module_name);
    if (baseline_module_name.empty()) {
      baseline_module_name = target_module_name;
    }
    QCHECK(!baseline_dump_dirs.empty())
        << "--baseline_dump_dir must be provided.";

    auto baseline_data_or =
        LoadRunData(baseline_dump_dirs, baseline_module_name,
                    absl::GetFlag(FLAGS_baseline_launch_barrier_id));
    if (!baseline_data_or.ok()) {
      LOG(FATAL) << "Failed to load baseline data: "
                 << baseline_data_or.status();
    }
    RunData baseline_data = *std::move(baseline_data_or);

    auto target_data_or =
        LoadRunData(target_dump_dirs, target_module_name,
                    absl::GetFlag(FLAGS_target_launch_barrier_id));
    if (!target_data_or.ok()) {
      LOG(FATAL) << "Failed to load target data: " << target_data_or.status();
    }
    RunData target_data = *std::move(target_data_or);

    if (baseline_data.device_assignment->replica_count() !=
        target_data.device_assignment->replica_count()) {
      LOG(FATAL) << "Replica count mismatch: baseline="
                 << baseline_data.device_assignment->replica_count()
                 << ", target="
                 << target_data.device_assignment->replica_count();
    }
    int replica_count = baseline_data.device_assignment->replica_count();
    if (replica_count == 0) {
      LOG(FATAL) << "Replica count is 0.";
    }

    HloModule* baseline_original_module = baseline_data.original_module.get();
    HloModule* target_original_module = target_data.original_module.get();
    const int baseline_launch_barrier_id = baseline_data.launch_barrier_id;
    const int target_launch_barrier_id = target_data.launch_barrier_id;
    const std::string baseline_module_name_for_me = baseline_data.module_name;
    const std::string target_module_name_for_me = target_data.module_name;

    auto baseline_results_or =
        RecoverJob(ComparisonVariant::kBaseline, &baseline_data, output_dir,
                   temp_file_base_path);
    if (!baseline_results_or.ok()) {
      LOG(FATAL) << "Failed to recover baseline data: "
                 << baseline_results_or.status();
    }
    RecoveryRunResults baseline_results = *std::move(baseline_results_or);

    auto target_results_or =
        RecoverJob(ComparisonVariant::kTarget, &target_data, output_dir,
                   temp_file_base_path);
    if (!target_results_or.ok()) {
      LOG(FATAL) << "Failed to recover target data: "
                 << target_results_or.status();
    }
    RecoveryRunResults target_results = *std::move(target_results_or);

    std::vector<std::string> comparison_output_filenames;
    comparison_output_filenames.reserve(replica_count);
    for (int i = 0; i < replica_count; ++i) {
      std::string output_filename = tsl::io::JoinPath(
          output_dir,
          absl::StrFormat("comparison_results.B-%s_%d.T-%s_%d.R_%d.riegeli",
                          baseline_module_name_for_me,
                          baseline_launch_barrier_id, target_module_name_for_me,
                          target_launch_barrier_id, i));
      comparison_output_filenames.push_back(output_filename);
    }

    bool comparison_files_exist = false;
    if (!absl::GetFlag(FLAGS_regenerate_output_files)) {
      bool all_exist = true;
      for (const std::string& filename : comparison_output_filenames) {
        uint64_t file_size;
        if (!tsl::Env::Default()->GetFileSize(filename, &file_size).ok()) {
          all_exist = false;
          break;
        }
      }
      comparison_files_exist = all_exist;
    }

    if (comparison_files_exist) {
      absl::FPrintF(
          stderr,
          "Skipping comparison because comparison result "
          "files already exist and --regenerate_output_files=false.\n");
      absl::FPrintF(stderr, "\n====== Comparison Result Files ======\n");
      for (const auto& filename : comparison_output_filenames) {
        absl::FPrintF(stderr, "%s\n", filename);
      }
      auto diff_results = hlo_diff::ComputeDiff(*baseline_original_module,
                                                *target_original_module);
      if (!diff_results.ok()) {
        LOG(FATAL) << "Failed to compute diff: " << diff_results.status();
      }
      for (int i = 0; i < replica_count; ++i) {
        const auto& comparison_output_filename = comparison_output_filenames[i];
        PrintComparisonSummary(i, comparison_output_filename);
        if (absl::GetFlag(FLAGS_generate_hlo_html_dump)) {
          GenerateHloHtmlDumps(i, *baseline_original_module,
                               *target_original_module, output_dir,
                               comparison_output_filename);
        }
      }
    } else {
      // Setup output writers.
      std::vector<std::unique_ptr<riegeli::RecordWriterBase>> writers(
          replica_count);
      for (int i = 0; i < replica_count; ++i) {
        writers[i] = riegeli::Maker<riegeli::RecordWriter>(
            riegeli::Maker<riegeli::FdWriter>(comparison_output_filenames[i]),
            riegeli::RecordWriterBase::Options().set_transpose(true));
      }
      absl::Mutex writer_mutex;

      // Setup comparator and callback.
      auto comparison_callback =
          [&](int replica_id,
              std::shared_ptr<const tensor_transformation::TensorTransformation>
                  pending_transformation,
              AbsoluteScopedTensorKey baseline_tensor_key,
              OriginalTensorSummary const* baseline_tensor_summary,
              AbsoluteScopedTensorKey target_tensor_key,
              OriginalTensorSummary const* target_tensor_summary)
          -> absl::Status {
        ComparisonResultProto result;
        result.set_replica_id(replica_id);
        if (pending_transformation != nullptr) {
          ToProto(pending_transformation.get(),
                  result.mutable_pending_transformation());
        }
        *result.mutable_baseline_tensor_key() =
            ProtoFromScopedTensorKey(baseline_tensor_key);
        if (baseline_tensor_summary != nullptr) {
          for (const auto& summary : baseline_tensor_summary->summaries) {
            *result.add_baseline_tensor_summaries() =
                CreateTensorSummaryProto(summary);
          }
        }
        *result.mutable_target_tensor_key() =
            ProtoFromScopedTensorKey(target_tensor_key);
        if (target_tensor_summary != nullptr) {
          for (const auto& summary : target_tensor_summary->summaries) {
            *result.add_target_tensor_summaries() =
                CreateTensorSummaryProto(summary);
          }
        }
        result.set_diff_score(ComputeDiffScore(result));

        absl::MutexLock lock(writer_mutex);
        writers[replica_id]->WriteRecord(result);
        return absl::OkStatus();
      };

      CHECK_GT(baseline_results.output_filenames.size(), 0);
      CHECK_GT(target_results.output_filenames.size(), 0);
      auto comparator_or = XlaJobComparator::Create(
          replica_count, baseline_original_module, target_original_module,
          baseline_results.output_filenames.front(),
          target_results.output_filenames.front(),
          std::move(comparison_callback));

      if (!comparator_or.ok()) {
        LOG(FATAL) << "Failed to create XlaJobComparator: "
                   << comparator_or.status();
      }
      auto& [comparator, creation_metrics, diff_results] =
          comparator_or.value();

      for (int i = 0; i < replica_count; ++i) {
        CHECK_LT(i, baseline_results.output_filenames.size());
        CHECK_LT(i, target_results.output_filenames.size());
        riegeli::RecordReader baseline_reader(riegeli::Maker<riegeli::FdReader>(
            baseline_results.output_filenames[i]));
        riegeli::RecordReader target_reader(riegeli::Maker<riegeli::FdReader>(
            target_results.output_filenames[i]));

        RecoveredTensorSummaryProto proto;
        bool baseline_has_more = true;
        bool target_has_more = true;
        std::optional<riegeli::Position> baseline_size = baseline_reader.Size();
        std::optional<riegeli::Position> target_size = target_reader.Size();
        bool use_bytes_as_progress =
            baseline_size.has_value() && target_size.has_value();
        ProgressReporter comparison_progress_reporter(
            absl::StrFormat("Comparing replica %d", i),
            use_bytes_as_progress ? (*baseline_size + *target_size) : 0,
            /*use_percent=*/use_bytes_as_progress,
            absl::GetFlag(FLAGS_progress_reporter_log_interval));
        while (baseline_has_more || target_has_more) {
          if (use_bytes_as_progress) {
            comparison_progress_reporter.Report(
                baseline_reader.pos().numeric() +
                target_reader.pos().numeric());
          }
          if (baseline_has_more) {
            if (baseline_reader.ReadRecord(proto)) {
              auto summary_or = RecoveredTensorSummaryFromProto(proto);
              CHECK_OK(summary_or);
              CHECK_OK(comparator.ProcessOriginalTensorSummary(
                  ComparisonVariant::kBaseline, i,
                  summary_or->original_tensor_key,
                  summary_or->pending_transformation,
                  summary_or->original_tensor_summary));
              if (!use_bytes_as_progress) {
                comparison_progress_reporter.Report();
              }
            } else {
              baseline_has_more = false;
            }
          }
          if (target_has_more) {
            if (target_reader.ReadRecord(proto)) {
              auto summary_or = RecoveredTensorSummaryFromProto(proto);
              CHECK_OK(summary_or);
              CHECK_OK(comparator.ProcessOriginalTensorSummary(
                  ComparisonVariant::kTarget, i,
                  summary_or->original_tensor_key,
                  summary_or->pending_transformation,
                  summary_or->original_tensor_summary));
              if (!use_bytes_as_progress) {
                comparison_progress_reporter.Report();
              }
            } else {
              target_has_more = false;
            }
          }
        }
        CHECK(baseline_reader.Close()) << baseline_reader.status();
        CHECK(target_reader.Close()) << target_reader.status();
      }

      CHECK_OK(comparator.FinishComparison());

      // Close writers.
      for (int i = 0; i < replica_count; ++i) {
        CHECK(writers[i]->Close()) << writers[i]->status();
      }

      PrintCreationMetrics(creation_metrics);

      std::vector<XlaJobComparator::ProcessingMetrics> processing_metrics_vec =
          comparator.GetProcessingMetrics();
      PrintProcessingMetrics(processing_metrics_vec);

      absl::FPrintF(stderr, "\n====== Comparison Result Files ======\n");
      for (const auto& filename : comparison_output_filenames) {
        absl::FPrintF(stderr, "%s\n", filename);
      }

      for (int i = 0; i < replica_count; ++i) {
        PrintComparisonSummary(i, comparison_output_filenames[i]);
        if (absl::GetFlag(FLAGS_generate_hlo_html_dump)) {
          GenerateHloHtmlDumps(i, *baseline_original_module,
                               *target_original_module, output_dir,
                               comparison_output_filenames[i]);
        }
      }
    }
  }
}

}  // namespace
}  // namespace xla::numerics::comparison

int main(int argc, char* argv[]) {
  tsl::port::InitMain(argv[0], &argc, &argv);
  xla::numerics::comparison::Run();
  return 0;
}
