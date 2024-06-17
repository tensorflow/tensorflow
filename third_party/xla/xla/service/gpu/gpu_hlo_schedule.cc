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

#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/match.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_input_output_alias_config.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_schedule.h"
#include "xla/hlo/utils/hlo_query.h"
#include "xla/service/buffer_value.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/cublas_cudnn.h"
#include "xla/service/gpu/gpu_latency_hiding_scheduler.h"
#include "xla/service/gpu/gpu_schedule_postprocessing.h"
#include "xla/service/gpu/model/analytical_latency_estimator.h"
#include "xla/service/hlo_memory_scheduler.h"
#include "xla/service/hlo_pass_pipeline.h"
#include "xla/service/latency_hiding_scheduler.h"
#include "xla/service/p2p_schedule_preparation.h"
#include "xla/service/profile_guided_latency_estimator.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/statusor.h"
#include "xla/stream_executor/device_description.h"
#include "xla/util.h"
#include "tsl/platform/env.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/path.h"
#include "tsl/platform/protobuf.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace gpu {

namespace {

// A threshold for which we consider AR to be costly perf-wise.
static constexpr int64_t kCostlyAllReduceThreshold = 30 * 1024 * 1024;

// Multiplier which we apply to expand the base cost for the costly AR.
static constexpr int64_t kCostlyAllReduceMultiplier = 4;

bool IsNopInstruction(const HloInstruction& hlo) {
  HloOpcode op = hlo.opcode();
  return op == HloOpcode::kGetTupleElement || op == HloOpcode::kBitcast ||
         op == HloOpcode::kConstant || op == HloOpcode::kParameter ||
         hlo.IsEffectiveBitcast();
}

bool ShouldScheduleAsEarlyAsPossible(const HloInstruction& instr) {
  switch (instr.opcode()) {
    case HloOpcode::kAllReduceStart:
    case HloOpcode::kCollectivePermuteStart:
      return !IsSyncCollective(&instr);
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
           IsSyncCollective(instr);
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

absl::StatusOr<HloSchedule> ScheduleGpuModuleWithMemoryScheduler(
    const HloModule* module, int64_t pointer_size) {
  return ScheduleModule(
      module,
      [pointer_size](const BufferValue& buffer) {
        return ShapeUtil::ByteSizeOf(buffer.shape(), pointer_size);
      },
      ComputationSchedulerToModuleScheduler(DefaultMemoryScheduler,
                                            PostProcessSchedule));
}

// Latency hiding scheduler support.

SchedulerConfig GetSchedulerConfig(int64_t memory_limit) {
  SchedulerConfig config;
  config.all_reduce_overlap_limit = 1;
  config.collective_broadcast_overlap_limit = 1;
  config.collective_permute_overlap_limit = 1;
  config.use_real_cost_model = false;
  config.aggressive_scheduling_policies = true;
  config.schedule_send_recvs = true;
  config.memory_limit = memory_limit;
  return config;
}

class GpuLatencyEstimator : public ApproximateLatencyEstimator {
 public:
  explicit GpuLatencyEstimator(
      int64_t pointer_size,
      GetCanonicalAsyncOpFunc func = GpuGetCanonicalAsyncOp)
      : ApproximateLatencyEstimator(func), pointer_size_(pointer_size) {}
  TimeCost NodeCost(const HloInstruction* instr) const override {
    if (IsNopInstruction(*instr)) {
      return 0.0;
    }
    // Consider cublas/cuddn/softmax custom calls as medium cost. Since the
    // latency between async-start and async-done is 5000 and cost of each
    // custom call is 1000, the LHS will try to schedule approximately 5 of
    // these in between each start/end pair.
    if (instr->opcode() == HloOpcode::kCustomCall) {
      if (IsCublasGemm(*instr) || IsCustomCallToDnnConvolution(*instr)) {
        return ApproximateLatencyEstimator::kMediumCost;
      }
      // consider other custom calls as medium cost for now. Keeping the case
      // explicitly separate for further tuning.
      return ApproximateLatencyEstimator::kMediumCost;
    }
    return ApproximateLatencyEstimator::NodeCost(instr);
  }

  LatencyEstimator::TimeCost GetLatencyBetween(
      const HloGraphNode& from, const HloGraphNode& target) const override {
    if (IsAsyncPair(from, target)) {
      if (from.GetInstr().opcode() == HloOpcode::kRecv) {
        // Recv -> RecvDone has a low latency.
        return ApproximateLatencyEstimator::kLowLatency;
      } else if (from.GetInstr().opcode() == HloOpcode::kSend) {
        // Send -> SendDone has a very high latency.
        return ApproximateLatencyEstimator::kHighLatency * 10;
      }

      bool enable_approx_collectives =
          from.GetInstr()
              .GetModule()
              ->config()
              .debug_options()
              .xla_gpu_enable_approx_costly_collectives();
      bool is_all_reduce =
          from.GetInstr().opcode() == HloOpcode::kAllReduceStart;
      bool collective_size_exceeds_threshold =
          GetSizeOfShape(from.GetInstr().shape(), pointer_size_) >
          kCostlyAllReduceThreshold;
      if (enable_approx_collectives && is_all_reduce &&
          collective_size_exceeds_threshold) {
        return ApproximateLatencyEstimator::kHighLatency *
               kCostlyAllReduceMultiplier;
      }

      return ApproximateLatencyEstimator::kHighLatency;
    }
    // Every other instruction we consider synchronous, which means the
    // latency between each of them is always one unit.
    return ApproximateLatencyEstimator::kLowLatency;
  }

 private:
  int64_t pointer_size_;
};

tensorflow::profiler::ProfiledInstructionsProto GetProfileForFingerprint(
    tensorflow::profiler::ProfiledInstructionsProto& profile,
    const std::string& fingerprint) {
  tensorflow::profiler::ProfiledInstructionsProto result;
  bool merge_remat_clones = false;
  for (const auto& cost : profile.costs()) {
    absl::string_view cost_name = cost.name();
    std::string new_cost_name = cost.name();
    absl::string_view cost_sep = "::";
    if (absl::StrContains(cost_name, cost_sep)) {
      std::vector<std::string> split_names =
          absl::StrSplit(cost_name, cost_sep);
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

  // Map from stripped name -> pair<accumulated cost, count>
  absl::flat_hash_map<absl::string_view, std::pair<double, int64_t>> costs;
  for (const auto& cost : result.costs()) {
    std::pair<double, int64_t>& data = costs[strip_remat_suffix(cost.name())];
    data.first += cost.cost_us();
    data.second++;
  }

  tensorflow::profiler::ProfiledInstructionsProto merged_result;
  for (const auto& cost : costs) {
    auto* new_cost = merged_result.add_costs();
    double average = cost.second.first / cost.second.second;
    new_cost->set_cost_us(average);
    new_cost->set_name(std::string(cost.first));
  }

  return merged_result;
}

std::optional<tensorflow::profiler::ProfiledInstructionsProto> ReadPGLEProfile(
    const HloModule* module, const std::string& fingerprint) {
  tensorflow::profiler::ProfiledInstructionsProto profile;

  absl::string_view fdo_profile = module->config().fdo_profile();
  // First attempt to read the profile from `fdo_profile` in ModuleConfig
  if (!fdo_profile.empty()) {
    // Attempt to parse it as a binary proto.
    if (tsl::ParseProtoUnlimited(&profile, fdo_profile.data(),
                                 fdo_profile.size())) {
      LOG(INFO) << "Using PGLE profile for module from fdo_profile (binary)";
      return GetProfileForFingerprint(profile, fingerprint);
    }
    // If not a binary proto, attempt to parse it as a text proto.
    profile.Clear();
    if (tsl::protobuf::TextFormat::ParseFromString(std::string(fdo_profile),
                                                   &profile)) {
      LOG(INFO) << "Using PGLE profile for module from fdo_profile (text)";
      return GetProfileForFingerprint(profile, fingerprint);
    }
    LOG(ERROR) << "Unable to prase FDO profile: not a valid text or binary "
                  "ProfiledInstructionsProto";
  }

  const std::string& pgle_profile_file_or_dir_path =
      module->config()
          .debug_options()
          .xla_gpu_pgle_profile_file_or_directory_path();
  if (pgle_profile_file_or_dir_path.empty()) {
    return std::nullopt;
  }
  tsl::Env* env = tsl::Env::Default();
  auto read_text_or_binary_profile = [&profile, env, &fingerprint](
                                         const std::string& text_path,
                                         const std::string& binary_path)
      -> std::optional<tensorflow::profiler::ProfiledInstructionsProto> {
    if (env->FileExists(text_path).ok()) {
      absl::Status s = tsl::ReadTextProto(env, text_path, &profile);
      if (s.ok()) {
        LOG(INFO) << "Using PGLE profile from " << text_path;
        return GetProfileForFingerprint(profile, fingerprint);
      } else {
        LOG(ERROR) << "Unable to read PGLE text proto from " << text_path
                   << ": " << s.message();
      }
      profile.Clear();
    }
    if (env->FileExists(binary_path).ok()) {
      absl::Status s = tsl::ReadBinaryProto(env, binary_path, &profile);
      if (s.ok()) {
        LOG(INFO) << "Using PGLE profile from " << binary_path;
        return GetProfileForFingerprint(profile, fingerprint);
      } else {
        LOG(ERROR) << "Unable to read PGLE binary proto from " << binary_path
                   << ": " << s.message();
      }
      profile.Clear();
    }
    return std::nullopt;
  };

  // If its a directory, use fingerprint to look for the profile for this
  // specific module.
  if (env->IsDirectory(pgle_profile_file_or_dir_path).ok()) {
    std::string pgle_profile_path_prefix =
        pgle_profile_file_or_dir_path + "/" + fingerprint;
    return read_text_or_binary_profile(pgle_profile_path_prefix + ".pbtxt",
                                       pgle_profile_path_prefix + ".pb");
  }

  // The pgle_profile_file_or_dir is a file. Attempt to read the profile as text
  // proto or binary proto. Attempt to infer the file type based on the
  // extension.
  auto extension = tsl::io::Extension(pgle_profile_file_or_dir_path);
  if (extension == "pbtxt") {
    return read_text_or_binary_profile(pgle_profile_file_or_dir_path, "");
  } else if (extension == "pb") {
    return read_text_or_binary_profile("", pgle_profile_file_or_dir_path);
  } else {
    return read_text_or_binary_profile(pgle_profile_file_or_dir_path,
                                       pgle_profile_file_or_dir_path);
  }
}
}  // end namespace

absl::Status IsProfileApplicable(
    const HloModule* module,
    const tensorflow::profiler::ProfiledInstructionsProto& profile) {
  absl::flat_hash_set<absl::string_view> all_instruction_names;
  for (HloComputation* comp : module->MakeNonfusionComputations()) {
    for (HloInstruction* instr : comp->instructions()) {
      all_instruction_names.insert(instr->name());
    }
  }

  std::vector<std::string> missing_costs_names;
  for (const auto& cost : profile.costs()) {
    if (!all_instruction_names.contains(cost.name())) {
      missing_costs_names.push_back(cost.name());
    }
  }
  std::vector<std::string> missing_latency_names;
  for (const auto& latency : profile.latencies()) {
    if (!all_instruction_names.contains(latency.source())) {
      missing_latency_names.push_back(latency.source());
    }

    if (!all_instruction_names.contains(latency.target())) {
      missing_latency_names.push_back(latency.target());
    }
  }
  if (!(missing_costs_names.empty() && missing_latency_names.empty())) {
    return absl::InvalidArgumentError(
        absl::StrFormat("\nMissing costs: %s;\nMissing latencies: %s",
                        absl::StrJoin(missing_costs_names, ", "),
                        absl::StrJoin(missing_latency_names, ", ")));
  }

  return absl::OkStatus();
}

int64_t GetSizeOfShape(const Shape& shape, int pointer_size) {
  int64_t size = ShapeUtil::ByteSizeOf(shape, pointer_size);
  if (shape.IsTuple() || shape.is_static()) {
    return size;
  }
  // Each dynamic dimension size is represented as a S32.
  int64_t metadata_size = sizeof(int32_t) * shape.dimensions_size();
  return size + metadata_size;
}

static int64_t GetSchedulerMemoryLimit(
    const HloModule* module, const se::DeviceDescription& gpu_device_info,
    int pointer_size);

absl::StatusOr<ScheduleMetadata> ScheduleGpuModule(
    HloModule* module, int64_t pointer_size,
    const se::DeviceDescription& gpu_device_info) {
  int64_t memory_limit =
      GetSchedulerMemoryLimit(module, gpu_device_info, pointer_size);
  if (module->has_schedule()) {
    return ScheduleMetadata{memory_limit};
  }

  HloPassPipeline prepare_pipeline("p2p-schedule-preparation");
  prepare_pipeline.AddPass<P2PSchedulePreparation>();
  TF_RETURN_IF_ERROR(prepare_pipeline.Run(module).status());

  TF_ASSIGN_OR_RETURN(
      HloSchedule schedule,
      ScheduleGpuModuleWithMemoryScheduler(module, pointer_size));
  TF_RETURN_IF_ERROR(module->set_schedule(std::move(schedule)));

  // Tag the module with its 128 bit fingerprint. The fingerprint should include
  // instruction name with ids.
  std::string fingerprint = module->GetFingerprint128(
      HloPrintOptions::Canonical().set_print_backend_config(true));
  FrontendAttributes attributes;
  (*attributes.mutable_map())[std::string(kFingerprintBeforeLHS)] = fingerprint;
  module->add_frontend_attributes(attributes);
  VLOG(1) << "Fingerprint before LHS for module " << module->name() << "("
          << module->unique_id() << ") = " << fingerprint;

  const bool enable_latency_hiding_scheduler =
      module->config()
          .debug_options()
          .xla_gpu_enable_latency_hiding_scheduler();

  if (!enable_latency_hiding_scheduler) {
    return ScheduleMetadata{memory_limit};
  }

  SchedulerConfig config = GetSchedulerConfig(memory_limit);
  auto gpu_latency_estimator =
      std::make_unique<GpuLatencyEstimator>(pointer_size);

  std::unique_ptr<LatencyEstimator> latency_estimator;
  std::optional<tensorflow::profiler::ProfiledInstructionsProto> profile =
      ReadPGLEProfile(module, fingerprint);

  const bool enable_analytical_latency_estimator =
      module->config()
          .debug_options()
          .xla_gpu_enable_analytical_latency_estimator();
  if (profile.has_value()) {
    latency_estimator = std::make_unique<ProfileGuidedLatencyEstimator>(
        config, std::move(gpu_latency_estimator), profile.value());
    LOG(INFO) << "Found profile, using profile guided latency estimator";
    absl::Status s = IsProfileApplicable(module, profile.value());
    if (!s.ok()) {
      LOG(INFO) << "PGLE profile may not applicable to the module, but will "
                   "still be used : "
                << s.message();
    }
  } else if (enable_analytical_latency_estimator) {
    latency_estimator = std::make_unique<AnalyticalLatencyEstimator>(
        config, std::move(gpu_latency_estimator), gpu_device_info,
        [input_pointer_size = pointer_size](const Shape& shape) {
          return GetSizeOfShape(shape, input_pointer_size);
        },
        module->entry_computation());
    LOG(INFO) << "Using analytical latency estimator";
  } else {
    latency_estimator = std::move(gpu_latency_estimator);
  }

  auto async_tracker = [&]() -> std::unique_ptr<AsyncTracker> {
    return module->config()
                   .debug_options()
                   .xla_gpu_lhs_enable_gpu_async_tracker()
               ? std::make_unique<GpuAsyncTracker>(config)
               : std::make_unique<GpuAsyncTrackerBase>(config);
  }();

  auto shape_size_in_bytes = [pointer_size](const Shape& shape) {
    return GetSizeOfShape(shape, pointer_size);
  };
  HloPassPipeline pipeline("latency-hiding-scheduler");
  auto scheduler_core = std::make_unique<DefaultSchedulerCore>(
      shape_size_in_bytes, async_tracker.get(), latency_estimator.get(),
      config);
  pipeline.AddPass<LatencyHidingScheduler>(
      std::move(latency_estimator), std::move(async_tracker),
      std::move(scheduler_core), shape_size_in_bytes);

  TF_RETURN_IF_ERROR(pipeline.Run(module).status());

  HloPassPipeline postprocessing_pipeline("gpu-schedule-postprocessing");
  postprocessing_pipeline.AddPass<GpuSchedulePostprocessing>();
  TF_RETURN_IF_ERROR(postprocessing_pipeline.Run(module).status());

  return ScheduleMetadata{memory_limit};
}

HloInstructionSequence PostProcessSchedule(
    const HloInstructionSequence& input) {
  HloInstructionSequence result = PostprocessorToScheduleSyncCollectives(input);
  return PostprocessorToScheduleAsEarlyOrLateAsPossible(result);
}

// Compute the device memory limit to be used by passes like scheduler and
// HLO rematerialization.
static int64_t GetSchedulerMemoryLimit(
    const HloModule* module, const se::DeviceDescription& gpu_device_info,
    int pointer_size) {
  // There is a "base" value which is either specified in HloModuleConfig (this
  // value should take into account the fact that we need to leave some memory
  // free for allocations that happen outside of XLA's allocator) or
  // obtained from GPU device info (we scale down this value to leave some space
  // for these outside XLA's allocator allocation).
  //
  // From that base value, subtract any input and output sizes (assuming they
  // are live throughout the execution) and then apply a slop factor.
  const int64_t base_limit =
      module->config().device_memory_size() != 0
          ? module->config().device_memory_size()
          : gpu_device_info.device_memory_size() * 80 / 100;

  // Find the total size of inputs and outputs.
  int64_t total_io_size = 0;
  for (HloInstruction* param :
       module->entry_computation()->parameter_instructions()) {
    ShapeUtil::ForEachSubshape(
        param->shape(),
        [&](const Shape& subshape, const ShapeIndex& /*index*/) {
          total_io_size += GetSizeOfShape(subshape, pointer_size);
        });
  }
  ShapeUtil::ForEachSubshape(
      module->result_shape(),
      [&](const Shape& subshape, const ShapeIndex& /*index*/) {
        total_io_size += GetSizeOfShape(subshape, pointer_size);
      });

  // If any inputs and outputs are aliased, do not double count them.
  module->input_output_alias_config().ForEachAlias(
      [&](const ShapeIndex& output_index,
          const HloInputOutputAliasConfig::Alias&) {
        const Shape& subshape =
            ShapeUtil::GetSubshape(module->result_shape(), output_index);
        total_io_size -= GetSizeOfShape(subshape, pointer_size);
      });

  int64_t limit =
      (base_limit - total_io_size) *
      module->config().debug_options().xla_gpu_memory_limit_slop_factor() / 100;
  return limit;
}

}  // namespace gpu
}  // namespace xla
