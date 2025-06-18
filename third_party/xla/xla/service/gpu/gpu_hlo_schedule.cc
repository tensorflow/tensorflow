/* Copyright 2017 The OpenXLA Authors.

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

#include "xla/service/gpu/gpu_hlo_schedule.h"

#include <stdbool.h>

#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_input_output_alias_config.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_schedule.h"
#include "xla/hlo/pass/hlo_pass_pipeline.h"
#include "xla/hlo/transforms/collectives/async_collective_creator.h"
#include "xla/hlo/transforms/simplifiers/hlo_memory_scheduler.h"
#include "xla/hlo/utils/hlo_query.h"
#include "xla/service/buffer_value.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/flag_utils.h"
#include "xla/service/gpu/gpu_latency_hiding_scheduler.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/gpu/model/analytical_latency_estimator.h"
#include "xla/service/gpu/model/sol_latency_estimator.h"
#include "xla/service/gpu/transforms/async_collective_annotator.h"
#include "xla/service/gpu/transforms/collectives/collective_ops_utils.h"
#include "xla/service/gpu/transforms/pgle_accuracy_checker.h"
#include "xla/service/gpu/transforms/scheduling_instruction_annotator.h"
#include "xla/service/hlo_module_config.h"
#include "xla/service/latency_hiding_scheduler.h"
#include "xla/service/legalize_scheduling_annotations.h"
#include "xla/service/p2p_schedule_preparation.h"
#include "xla/service/profile_guided_latency_estimator.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "tsl/platform/path.h"
#include "tsl/platform/protobuf.h"
#include "tsl/profiler/lib/traceme.h"

namespace xla {
namespace gpu {

using tensorflow::profiler::ProfiledInstructionsProto;

namespace {

bool ShouldScheduleAsEarlyAsPossible(const HloInstruction& instr) {
  switch (instr.opcode()) {
    case HloOpcode::kAllReduceStart:
    case HloOpcode::kCollectivePermuteStart:
      return !IsGPUSyncCollective(instr);
    case HloOpcode::kAsyncStart:
      // Start async ops as early as possible to allow more concurrency.
      return true;
    case HloOpcode::kCustomCall:
      return static_cast<const HloCustomCallInstruction&>(instr)
                 .custom_call_schedule() ==
             CustomCallSchedule::SCHEDULE_EARLIEST;
    default:
      return false;
  }
}

bool ShouldScheduleSuccessor(const HloInstruction& sussessor,
                             const HloPredicate& is_scheduled) {
  return ShouldScheduleAsEarlyAsPossible(sussessor) &&
         absl::c_all_of(sussessor.operands(), is_scheduled) &&
         absl::c_all_of(sussessor.control_predecessors(), is_scheduled);
}

bool ShouldScheduleAsLateAsPossible(const HloInstruction& instr) {
  switch (instr.opcode()) {
    case HloOpcode::kAllReduceDone:
    case HloOpcode::kCollectivePermuteDone:
      return ShouldScheduleAsEarlyAsPossible(*instr.operand(0));
    case HloOpcode::kAsyncDone:
      // Schedule as many other ops as possible before blocking on the
      // completion of async ops.
      return true;
    case HloOpcode::kCustomCall:
      return static_cast<const HloCustomCallInstruction&>(instr)
                 .custom_call_schedule() == CustomCallSchedule::SCHEDULE_LATEST;
    default:
      return false;
  }
}

bool ShouldSchedulePredecessor(const HloInstruction& predecessor,
                               const HloPredicate& is_scheduled) {
  return ShouldScheduleAsLateAsPossible(predecessor) &&
         absl::c_all_of(predecessor.users(), is_scheduled) &&
         absl::c_all_of(predecessor.control_successors(), is_scheduled);
}

// Schedules certain ops as early or late as possible. This supports a
// custom-call use case, where a logical operation is lowered into two HLOs
// (e.g., PerformX and PerformXDone). We utilize this mechanism to either hide
// host latencies between the pair of the custom-calls or more accurately
// identify the def-use relationship of the two calls (typically PerformX is
// scheduled right after all of its producers have been scheduled and
// PerformXDone is scheduled right before its first consumer.)
HloInstructionSequence PostprocessorToScheduleAsEarlyOrLateAsPossible(
    const HloInstructionSequence& input) {
  std::vector<HloInstruction*> earliest_scheduled;
  {
    absl::flat_hash_set<HloInstruction*> scheduled;
    auto is_scheduled = [&](const HloInstruction* instr) -> bool {
      return scheduled.contains(instr);
    };
    auto add_to_schedule = [&](HloInstruction* instr) {
      earliest_scheduled.push_back(instr);
      scheduled.insert(instr);
    };

    for (HloInstruction* instr : input.instructions()) {
      if (is_scheduled(instr)) continue;

      add_to_schedule(instr);

      // Schedule any successor that should be scheduled as early as possible if
      // all of its producers and control_predecessors have been scheduled.
      for (HloInstruction* user : instr->users()) {
        if (is_scheduled(user)) continue;

        if (ShouldScheduleSuccessor(*user, is_scheduled)) {
          add_to_schedule(user);
        }
      }
      for (HloInstruction* successor : instr->control_successors()) {
        if (is_scheduled(successor)) continue;

        if (ShouldScheduleSuccessor(*successor, is_scheduled)) {
          add_to_schedule(successor);
        }
      }
    }
  }

  std::deque<HloInstruction*> latest_scheduled;
  {
    absl::flat_hash_set<HloInstruction*> scheduled;
    auto is_scheduled = [&](const HloInstruction* instr) -> bool {
      return scheduled.contains(instr);
    };
    auto add_to_schedule = [&](HloInstruction* instr) {
      latest_scheduled.push_front(instr);
      scheduled.insert(instr);
    };
    for (auto it = earliest_scheduled.rbegin(); it != earliest_scheduled.rend();
         it++) {
      if (is_scheduled(*it)) continue;

      add_to_schedule(*it);

      // Schedule any predecessor that should be scheduled as late as possible
      // if all of its users and control_successors have been scheduled.
      for (HloInstruction* operand : (*it)->operands()) {
        if (is_scheduled(operand)) continue;

        if (ShouldSchedulePredecessor(*operand, is_scheduled)) {
          add_to_schedule(operand);
        }
      }
      for (HloInstruction* predecessor : (*it)->control_predecessors()) {
        if (is_scheduled(predecessor)) continue;

        if (ShouldSchedulePredecessor(*predecessor, is_scheduled)) {
          add_to_schedule(predecessor);
        }
      }
    }
  }

  HloInstructionSequence result;
  absl::c_for_each(latest_scheduled,
                   [&](HloInstruction* i) { result.push_back(i); });

  // Schedule post-processing can't introduce new instructions.
  CHECK(input.instructions().size() == result.size())
      << "schedule as early or late post-processing changed schedule size from "
      << input.instructions().size() << " to " << result.size();

  return result;
}

// Post process to move start/done for synchronous collectives next to each
// other.
HloInstructionSequence PostprocessorToScheduleSyncCollectives(
    const HloInstructionSequence& input) {
  HloInstructionSequence result;

  // Returns true if `inst` is a synchronous version of async collective start
  // operation (marked with `is_sync` attribute).
  auto is_sync_start = [](const HloInstruction* instr) {
    return hlo_query::IsAsyncCollectiveStartOp(instr,
                                               /*include_send_recv=*/true) &&
           IsGPUSyncCollective(*instr);
  };

  for (HloInstruction* instr : input.instructions()) {
    // Skip synchronous start instruction as it will be scheduled later when
    // we'll process corresponding done instruction.
    if (is_sync_start(instr)) continue;

    // Find a start instruction corresponding to done and schedule it right
    // before a done if it's a synchronous version.
    if (hlo_query::IsAsyncCollectiveDoneOp(instr, true)) {
      HloInstruction* start = instr->mutable_operand(0);
      if (is_sync_start(start)) result.push_back(start);
    }

    result.push_back(instr);
  }

  // Schedule post-processing can't introduce new instructions.
  CHECK(input.instructions().size() == result.size())
      << "sync collectives post-processing changed schedule size from "
      << input.instructions().size() << " to " << result.size();

  return result;
}

SchedulerConfig MakeGPUSchedulerConfig(uint64_t memory_limit,
                                       int64_t overlap_limit) {
  SchedulerConfig config;
  config.all_reduce_overlap_limit = 1;
  config.collective_broadcast_overlap_limit = 1;
  config.collective_permute_overlap_limit = 1;
  config.use_real_cost_model = false;
  config.aggressive_scheduling_policies = true;
  config.schedule_send_recvs = true;
  config.memory_limit = memory_limit;
  config.parallel_collective_overlap_limit = overlap_limit;

  CHECK(config.collective_broadcast_overlap_limit <=
        config.parallel_collective_overlap_limit);
  CHECK(config.all_to_all_overlap_limit <=
        config.parallel_collective_overlap_limit);
  CHECK(config.all_gather_overlap_limit <=
        config.parallel_collective_overlap_limit);
  CHECK(config.all_reduce_overlap_limit <=
        config.parallel_collective_overlap_limit);
  CHECK(config.reduce_scatter_overlap_limit <=
        config.parallel_collective_overlap_limit);

  return config;
}

namespace {
ProfiledInstructionsProto FilterWithFingerprint(
    const ProfiledInstructionsProto& profile, absl::string_view fingerprint) {
  ProfiledInstructionsProto result;
  bool merge_remat_clones = false;
  for (const auto& cost : profile.costs()) {
    std::string new_cost_name = cost.name();
    absl::string_view cost_sep = "::";
    if (absl::StrContains(cost.name(), cost_sep)) {
      std::vector<absl::string_view> split_names =
          absl::StrSplit(cost.name(), cost_sep);
      if (split_names.size() != 2 || split_names[0] != fingerprint) {
        continue;
      }
      new_cost_name = split_names[1];
    }

    // Check if we see instructions that have ".rematX" suffix. These are clones
    // of original instructions created by HLO rematerialization pass. We will
    // average the costs of the remat clones and the original instruction and
    // use that as the new cost of the original one.
    merge_remat_clones |= absl::StrContains(new_cost_name, ".remat");
    auto* new_cost = result.add_costs();
    new_cost->set_cost_us(cost.cost_us());
    new_cost->set_name(new_cost_name);
  }

  if (!merge_remat_clones) {
    return result;
  }

  auto strip_remat_suffix = [](absl::string_view name) -> absl::string_view {
    absl::string_view suffix = ".remat";
    size_t index = name.rfind(suffix);
    if (index == std::string::npos) {
      return name;
    }
    auto after_suffix = name.substr(index + suffix.size());
    // Everything after ".remat" should be a digit or empty. If yes, strip the
    // .rematN suffix.
    int64_t numeric_suffix;
    if (after_suffix.empty() ||
        absl::SimpleAtoi(after_suffix, &numeric_suffix)) {
      return name.substr(0, index);
    }
    return name;
  };

  struct Data {
    double accumulated_cost = 0.0;
    int64_t count = 0;
  };
  absl::flat_hash_map<absl::string_view, Data> costs;
  for (const auto& cost : result.costs()) {
    Data& data = costs[strip_remat_suffix(cost.name())];
    data.accumulated_cost += cost.cost_us();
    data.count++;
  }

  tensorflow::profiler::ProfiledInstructionsProto merged_result;
  for (const auto& [name, data] : costs) {
    auto* new_cost = merged_result.add_costs();
    double average = data.accumulated_cost / data.count;
    new_cost->set_cost_us(average);
    new_cost->set_name(std::string(name));
  }

  return merged_result;
}

std::optional<ProfiledInstructionsProto> ProfileFromConfig(
    const HloModuleConfig& config) {
  if (config.fdo_profile().empty()) {
    return std::nullopt;
  }
  ProfiledInstructionsProto profile;
  absl::string_view from_config = config.fdo_profile();
  LOG(INFO) << "Attempting to parse as a binary proto.";
  if (profile.ParseFromArray(from_config.data(), from_config.size())) {
    LOG(INFO) << "Using PGLE profile from fdo_profile (binary)";
    return profile;
  }
  LOG(INFO) << "Not a binary proto, attempt to parse it as a text proto.";
  profile.Clear();
  if (tsl::protobuf::TextFormat::ParseFromString(
          std::string(from_config),  // NOLINT copybara XLA Linux ARM64 breaks
                                     // without this explicit conversion.
          &profile)) {
    LOG(INFO) << "Using PGLE profile from fdo_profile (text)";
    return profile;
  }
  LOG(ERROR) << "Unable to parse fdo_profile: not a valid text or binary "
                "ProfiledInstructionsProto";
  return std::nullopt;
}

std::optional<ProfiledInstructionsProto> ProfileFromPath(
    const HloModuleConfig& config, const std::string& path,
    const bool as_text) {
  tsl::Env* env = tsl::Env::Default();
  if (env->FileExists(path).ok()) {
    ProfiledInstructionsProto profile;
    absl::Status s = as_text ? tsl::ReadTextProto(env, path, &profile)
                             : tsl::ReadBinaryProto(env, path, &profile);
    if (s.ok()) {
      LOG(INFO) << "Using PGLE profile from " << path;
      return profile;
    }
    LOG(ERROR) << "Tried but failed to parse PGLE proto from "
               << (as_text ? "text" : "binary") << " file '" << path
               << "'. Error message: " << s.message();
  } else {
    LOG(ERROR) << "PGLE profile file does not exist: " << path;
  }
  return std::nullopt;
}

std::optional<ProfiledInstructionsProto> ReadProfileFromSources(
    const HloModuleConfig& config, absl::string_view fingerprint) {
  if (auto profile = ProfileFromConfig(config); profile) {
    return profile;
  }

  const std::string& path =
      config.debug_options().xla_gpu_pgle_profile_file_or_directory_path();
  if (path.empty()) {
    return std::nullopt;
  }

  ProfiledInstructionsProto profile;
  tsl::Env* env = tsl::Env::Default();

  // If its a directory, use fingerprint to look for the profile for this
  // specific module.
  if (env->IsDirectory(path).ok()) {
    std::string file_name = absl::StrCat(path, "/", fingerprint);
    if (auto profile = ProfileFromPath(config, file_name + ".pbtxt", true);
        profile) {
      return profile;
    }
    if (auto profile = ProfileFromPath(config, file_name + ".pb", false);
        profile) {
      return profile;
    }
  }

  // Trie path as a file inferring the file type based on the extension.
  auto extension = tsl::io::Extension(path);
  if (extension == "pbtxt") {
    return ProfileFromPath(config, path, true);
  } else if (extension == "pb") {
    return ProfileFromPath(config, path, false);
  }
  return std::nullopt;
}

}  // namespace

std::optional<ProfiledInstructionsProto> ReadPGLEProfile(
    const HloModuleConfig& config, absl::string_view fingerprint) {
  auto profile = ReadProfileFromSources(config, fingerprint);
  if (profile.has_value()) {
    return FilterWithFingerprint(profile.value(), fingerprint);
  }
  return std::nullopt;
}

bool HasValidPGLEProfile(const HloModule& module,
                         absl::string_view fingerprint) {
  return ReadPGLEProfile(module.config(), fingerprint).has_value();
}

// Runs P2P schedule preparation prior any scheduling.
absl::Status RunP2PSchedulePreparation(HloModule* module) {
  if (!module->config().debug_options().xla_gpu_enable_pipelined_p2p()) {
    return absl::OkStatus();
  }
  HloPassPipeline prepare_pipeline("p2p-schedule-preparation");
  prepare_pipeline.AddPass<P2PSchedulePreparation>();
  return prepare_pipeline.Run(module).status();
}

// Adds fingerprint to the module before.
//
// Returns said fingerprint.
std::string TagWithFingerprint(HloModule* module) {
  std::string fingerprint = module->GetFingerprint128(
      HloPrintOptions::Canonical().set_print_backend_config(true));
  FrontendAttributes attributes;
  (*attributes.mutable_map())[std::string(kFingerprintBeforeLHS)] = fingerprint;
  module->add_frontend_attributes(attributes);
  VLOG(1) << "Fingerprint before LHS for module " << module->name() << "("
          << module->unique_id() << ") = " << fingerprint;
  return fingerprint;
}

// Returns latency estimator, key abstraction used by LHS which returns how much
// each instruction takes. If we return a PGO based estimator then we will
// additionally add fail-fast/warn checks to the pipeline which act in the
// absence of instruction in the profile. See `PGLEAccuracyChecker` for details.
std::unique_ptr<LatencyEstimator> GetLatencyEstimator(
    const HloModule& module, int pointer_size,
    const se::DeviceDescription& gpu_device_info, absl::string_view fingerprint,
    const SchedulerConfig& config) {
  const DebugOptions& options = module.config().debug_options();

  auto gpu_latency_estimator =
      std::make_unique<GpuLatencyEstimator>(pointer_size);

  std::optional<tensorflow::profiler::ProfiledInstructionsProto> profile =
      ReadPGLEProfile(module.config(), fingerprint);

  if (profile.has_value()) {
    auto aggregator = std::make_unique<GPUProfileStatisticsAggregator>();
    auto pg_latency_estimator = std::make_unique<ProfileGuidedLatencyEstimator>(
        config, std::move(gpu_latency_estimator), profile.value(),
        std::move(aggregator));
    LOG(INFO) << "Found profile, using profile guided latency estimator";
    VLOG(1) << "Profile:\n" << profile->DebugString();
    return pg_latency_estimator;
  }

  if (options.xla_gpu_enable_analytical_latency_estimator()) {
    LOG(INFO) << "Using analytical latency estimator";
    return std::make_unique<AnalyticalLatencyEstimator>(
        config, std::move(gpu_latency_estimator), gpu_device_info,
        ShapeSizeBytesFunction(pointer_size), module.entry_computation());
  }

  if (options.xla_gpu_enable_analytical_sol_latency_estimator()) {
    LOG(INFO) << "Using Speed-of-Light (SoL) analytical latency estimator";
    return std::make_unique<SolLatencyEstimator>(
        config, std::move(gpu_latency_estimator), gpu_device_info,
        ShapeSizeBytesFunction(pointer_size), module.entry_computation());
  }
  return gpu_latency_estimator;
}

// Accuracy checker is only applied to PGO based latency estimators with
// strictness level set to WARN or ERROR.
bool NeedAccuracyChecker(const DebugOptions& options,
                         const LatencyEstimator& latency_estimator) {
  if (typeid(latency_estimator) !=
      typeid(const ProfileGuidedLatencyEstimator)) {
    return false;
  }
  DebugOptions::PGLEStrictnessLevel level =
      options.xla_gpu_pgle_accuracy_checker();
  return level == DebugOptions::PGLE_STRICTNESS_LEVEL_WARN ||
         level == DebugOptions::PGLE_STRICTNESS_LEVEL_ERROR;
}

// For now, only allow cublas gemm custom calls and triton gemm fusions to
// be overlapped as the compute ops in the annotated scheduling groups.
LegalizeSchedulingAnnotations::Config SchedulingAnnotationsConfig() {
  LegalizeSchedulingAnnotations::Config annotation_config;
  annotation_config.keep_sync_annotation = [](const HloInstruction* hlo) {
    if (hlo == nullptr) {
      return false;
    }
    if (hlo->IsCustomCall("__cublas$gemm")) {
      return true;
    }
    if (hlo->opcode() == HloOpcode::kFusion && hlo->has_backend_config() &&
        hlo->backend_config<GpuBackendConfig>().ok()) {
      GpuBackendConfig gpu_config =
          hlo->backend_config<GpuBackendConfig>().value();
      return gpu_config.has_fusion_backend_config() &&
             gpu_config.fusion_backend_config().kind() == kTritonGemmFusionKind;
    }
    return false;
  };
  return annotation_config;
}

// Adds necessary passes to perform latency hiding estimations for the
// `pipeline`.
absl::Status RunLatencyHidingSchedulerPasses(
    HloModule* module, int pointer_size, absl::string_view fingerprint,
    uint64_t memory_limit, const se::DeviceDescription& gpu_device_info) {
  tsl::profiler::TraceMe traceme("RunLatencyHidingSchedulerPasses");
  HloPassPipeline pipeline("latency-hiding-scheduler");
  const DebugOptions& options = module->config().debug_options();
  pipeline.AddPass<LegalizeSchedulingAnnotations>(
      SchedulingAnnotationsConfig());

  SchedulerConfig config = MakeGPUSchedulerConfig(
      memory_limit,
      options.xla_gpu_experimental_parallel_collective_overlap_limit());

  auto shape_size_in_bytes = ShapeSizeBytesFunction(pointer_size);

  std::unique_ptr<LatencyEstimator> estimator = GetLatencyEstimator(
      *module, pointer_size, gpu_device_info, fingerprint, config);

  if (NeedAccuracyChecker(options, *estimator)) {
    pipeline.AddPass<PGLEAccuracyChecker>(
        dynamic_cast<ProfileGuidedLatencyEstimator&>(*estimator));
  }

  auto async_tracker = std::make_unique<GpuAsyncTracker>(config);
  auto scheduler_core = std::make_unique<DefaultSchedulerCore>(
      shape_size_in_bytes, async_tracker.get(), estimator.get(), config,
      /*target_scheduling_rule=*/nullptr,
      /*early_target_scheduling_rule=*/nullptr,
      /*post_processing_fn=*/nullptr,
      /*scheduling_instruction_crosses_overlap_limit=*/
      GpuScheduleCrossesOverlapLimit);

  pipeline.AddPass<LatencyHidingScheduler>(
      std::move(estimator), std::move(async_tracker), std::move(scheduler_core),
      shape_size_in_bytes);
  pipeline.AddPass<SchedulingInstructionAnnotator>();

  return pipeline.Run(module).status();
}

// Compute the device memory limit to be used by passes like scheduler and
// HLO rematerialization.
uint64_t GetSchedulerMemoryLimit(const HloModule& module,
                                 const se::DeviceDescription& gpu_device_info,
                                 int pointer_size) {
  // There is a "base" value which is either specified in HloModuleConfig
  // (this value should take into account the fact that we need to leave some
  // memory free for allocations that happen outside of XLA's allocator) or
  // obtained from GPU device info (we scale down this value to leave some
  // space for these outside XLA's allocator allocation).
  //
  // From that base value, subtract any input and output sizes (assuming they
  // are live throughout the execution) and then apply a slop factor.
  const uint64_t base_limit =
      module.config().device_memory_size() != 0
          ? module.config().device_memory_size()
          : gpu_device_info.device_memory_size() * 80 / 100;

  // Create size function that only counts device memory
  auto get_device_shape_size =
      gpu::ShapeSizeBytesFunction(pointer_size,
                                  /*memory_space=*/Layout::kDefaultMemorySpace);

  // Find the total size of inputs and outputs.
  uint64_t total_io_size = 0;
  for (HloInstruction* param :
       module.entry_computation()->parameter_instructions()) {
    ShapeUtil::ForEachSubshape(
        param->shape(),
        [&](const Shape& subshape, const ShapeIndex& /*index*/) {
          total_io_size += get_device_shape_size(subshape);
        });
  }
  ShapeUtil::ForEachSubshape(
      module.result_shape(),
      [&](const Shape& subshape, const ShapeIndex& /*index*/) {
        total_io_size += get_device_shape_size(subshape);
      });

  // If any inputs and outputs are aliased, do not double count them.
  module.input_output_alias_config().ForEachAlias(
      [&](const ShapeIndex& output_index,
          const HloInputOutputAliasConfig::Alias&) {
        const Shape& subshape =
            ShapeUtil::GetSubshape(module.result_shape(), output_index);
        total_io_size -= get_device_shape_size(subshape);
      });

  if (total_io_size > base_limit) {
    LOG(ERROR) << "The byte size of input/output arguments (" << total_io_size
               << ") exceeds the base limit (" << base_limit
               << "). This indicates an error in the calculation!";
    return 0;
  }

  return (base_limit - total_io_size) *
         module.config().debug_options().xla_gpu_memory_limit_slop_factor() /
         100;
}

bool IsLHSEnabled(const HloModule& module, absl::string_view fingerprint) {
  bool enable_lhs =
      module.config()
          .debug_options()
          .xla_gpu_enable_latency_hiding_scheduler() ||
      IsPassEnabledAtOptimizationEffort<LatencyHidingScheduler>(module);
  if (!enable_lhs && HasValidPGLEProfile(module, fingerprint)) {
    LOG(WARNING)
        << "Profile data detected but "
           "`xla_gpu_enable_latency_hiding_scheduler` unset. To use it "
           "compiler will run Latency Hiding Scheduler anyway.";
    enable_lhs = true;
  }
  return enable_lhs;
}

}  // end namespace

absl::Status RunAsyncCollectivesConversionPasses(HloModule* module) {
  HloPassPipeline pipeline("async-collective-conversion");

  // Convert all collectives to their async form, and then annotate the ones
  // that actually need to run asynchronously with a GPU specific backend
  // config.
  AsyncCollectiveCreator::CollectiveCreatorConfig config;
  config.convert_all_gather = HloPredicateTrue;
  config.convert_all_reduce = HloPredicateTrue;
  config.convert_all_to_all = HloPredicateTrue;
  config.convert_collective_broadcast = HloPredicateTrue;
  config.convert_collective_permute = HloPredicateTrue;
  config.convert_ragged_all_to_all = HloPredicateTrue;
  config.convert_reduce_scatter = HloPredicateTrue;
  pipeline.AddPass<AsyncCollectiveCreator>(std::move(config));

  absl::flat_hash_set<DebugOptions::CollectiveOpType> disabled_async_ops;
  for (auto collective_op_type :
       module->config().debug_options().xla_gpu_disable_async_collectives()) {
    if (collective_op_type == DebugOptions::ALLCOLLECTIVES) {
      for (int64_t i = DebugOptions::ALLREDUCE;
           i < DebugOptions::ALLCOLLECTIVES; i++) {
        disabled_async_ops.insert(
            static_cast<DebugOptions::CollectiveOpType>(i));
      }
      break;
    }
    disabled_async_ops.insert(
        static_cast<DebugOptions::CollectiveOpType>(collective_op_type));
  }
  auto convert_to_async = [&disabled_async_ops](const HloInstruction* inst) {
    switch (inst->opcode()) {
      case HloOpcode::kAllReduceStart:
        return !disabled_async_ops.contains(DebugOptions::ALLREDUCE);
      case HloOpcode::kCollectivePermuteStart:
        return !disabled_async_ops.contains(DebugOptions::COLLECTIVEPERMUTE);
      case HloOpcode::kAllGatherStart:
        return !disabled_async_ops.contains(DebugOptions::ALLGATHER);
      case HloOpcode::kAsyncStart: {
        auto async_inst = Cast<HloAsyncInstruction>(inst);
        switch (async_inst->async_wrapped_opcode()) {
          case HloOpcode::kCollectiveBroadcast:
            return !disabled_async_ops.contains(
                DebugOptions::COLLECTIVEBROADCAST);
          case HloOpcode::kReduceScatter:
            return !disabled_async_ops.contains(DebugOptions::REDUCESCATTER);
          case HloOpcode::kAllToAll:
            return !disabled_async_ops.contains(DebugOptions::ALLTOALL);
          case HloOpcode::kRaggedAllToAll:
            return !disabled_async_ops.contains(DebugOptions::RAGGEDALLTOALL);
          default:
            return false;
        }
      }
      default:
        return false;
    }
  };
  pipeline.AddPass<AsyncCollectiveAnnotator>(convert_to_async);

  return pipeline.Run(module).status();
}

absl::StatusOr<ScheduleMetadata> ScheduleGpuModule(
    HloModule* module, int64_t pointer_size,
    const se::DeviceDescription& gpu_device_info) {
  tsl::profiler::TraceMe traceme("ScheduleGpuModule");

  // Tag the module with its 128 bit fingerprint. The fingerprint should include
  // instruction name with ids.
  std::string fingerprint = TagWithFingerprint(module);
  uint64_t memory_limit =
      GetSchedulerMemoryLimit(*module, gpu_device_info, pointer_size);

  // Module already has a schedule, do nothing.
  if (module->has_schedule()) {
    VLOG(1) << "Module already has a schedule, do nothing.";
    return ScheduleMetadata{memory_limit};
  }

  // Run the scheduler which minimizes peak memory usage.
  // We need to run it anyway because LHS relies on it.
  // See `xla::LatencyHidingScheduler::Run`.
  TF_RETURN_IF_ERROR(RunP2PSchedulePreparation(module));
  TF_ASSIGN_OR_RETURN(
      HloSchedule schedule,
      ScheduleGpuModuleWithMemoryScheduler(module, pointer_size));
  TF_RETURN_IF_ERROR(module->set_schedule(std::move(schedule)));

  bool enable_latency_hiding_scheduler = IsLHSEnabled(*module, fingerprint);

  // Run Latency Hiding Scheduler (LHS). It maximizes the compute-communication
  // overlap, potentially at the cost of memory usage.
  if (enable_latency_hiding_scheduler) {
    TF_RETURN_IF_ERROR(RunLatencyHidingSchedulerPasses(
        module, pointer_size, fingerprint, memory_limit, gpu_device_info));
  }

  return ScheduleMetadata{memory_limit};
}

absl::StatusOr<HloSchedule> ScheduleGpuModuleWithMemoryScheduler(
    const HloModule* module, int64_t pointer_size, int64_t* peak_memory_bytes) {
  BufferValue::SizeFunction size_func =
      [pointer_size](const BufferValue& buffer) -> int64_t {
    const Shape& shape = buffer.shape();
    if (shape.has_layout() &&
        shape.layout().memory_space() == Layout::kHostMemorySpace) {
      return static_cast<int64_t>(0);
    }
    return ShapeUtil::ByteSizeOf(shape, pointer_size);
  };
  return ScheduleModule(module,
                        DefaultMemoryScheduler(size_func, PostProcessSchedule),
                        /*execution_threads=*/{}, peak_memory_bytes);
}

HloInstructionSequence PostProcessSchedule(
    const HloInstructionSequence& input) {
  HloInstructionSequence result = PostprocessorToScheduleSyncCollectives(input);
  return PostprocessorToScheduleAsEarlyOrLateAsPossible(result);
}

}  // namespace gpu
}  // namespace xla
