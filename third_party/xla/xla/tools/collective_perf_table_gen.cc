/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/tools/collective_perf_table_gen.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/hash/hash.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/strings/substitute.h"
#include "absl/time/time.h"
#include "google/protobuf/text_format.h"
#include "xla/backends/gpu/cost_model/hlo_op_profile.pb.h"
#include "xla/backends/gpu/cost_model/hlo_op_profiles.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/replica_group.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/hlo/utils/hlo_query.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/plugin/xla_gpu/xla_gpu_client_options.h"
#include "xla/service/backend.h"
#include "xla/service/gpu/transforms/collectives/collective_ops_utils.h"
#include "xla/service/hlo_module_config.h"
#include "xla/tools/multihost_hlo_runner/create_client.h"
#include "xla/tools/multihost_hlo_runner/functional_hlo_runner.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {

namespace {

// Assume we're always issuing transfers for f32 data type. It should not
// necessarily change the derating curve properties, yet the effects of compute
// for e.g. all reduce are yet to be measured.
constexpr uint8_t kBytesPerElem = 4;

// Mem fraction dedicated to XLA buffers.
constexpr double kGpuMemFraction = 0.8;

// # of profiling runs.
constexpr uint8_t kNumProfilingRuns = 20;

struct StaticSpec {
  CollectivePerfTableGen::CollectiveType collective_type;
  // For collective hlo instructions, the `replica_groups` is used to define
  // the replica_groups attribute of collective hlo instructions.

  // For collective permute hlo instructions, the `replica_groups` is used to
  // define the number of devices participating in the collective permute.
  IotaReplicaGroupList replica_groups;

  // Defines the source-target pairs of a collective permute hlo instruction.
  // It is only used for collective permute hlo instructions, and is
  // std::nullopt for others.
  std::optional<CollectivePermuteCostModelType> collective_permute_pattern;
  int64_t tensor_size_bytes;
};

struct ExplicitSpec {
  std::unique_ptr<HloModule> module;
};

struct ProfilingResult {
  std::string device_info;
  HloInstructionProto hlo_proto;
  std::vector<HloInstructionProto> operands;
  std::string fingerprint;
  int64_t clock_cycles;
  int64_t flops;
  int64_t network_throughput;

  struct Hash {
    size_t operator()(const ProfilingResult& profiling_result) const {
      return absl::HashOf(profiling_result.device_info,
                          profiling_result.fingerprint);
    }
  };

  struct Eq {
    bool operator()(const ProfilingResult& lhs,
                    const ProfilingResult& rhs) const {
      return lhs.device_info == rhs.device_info &&
             lhs.fingerprint == rhs.fingerprint;
    }
  };
};

int64_t GetMedianRuntimeNs(const std::vector<ExecutionProfile>& profiles) {
  std::vector<int64_t> runtimes;
  runtimes.reserve(profiles.size());
  for (const ExecutionProfile& profile : profiles) {
    runtimes.push_back(profile.compute_time_ns());
  }
  std::sort(runtimes.begin(), runtimes.end());
  size_t mid = runtimes.size() / 2;
  if (runtimes.size() % 2 == 1) {
    return runtimes[mid];
  }
  return runtimes[mid - 1] + (runtimes[mid] - runtimes[mid - 1]) / 2;
}

int64_t GetInputDim(CollectivePerfTableGen::CollectiveType type,
                    int64_t tensor_size_bytes,
                    IotaReplicaGroupList replica_groups) {
  int64_t dim_size = -1;
  CHECK_EQ(tensor_size_bytes % kBytesPerElem, 0);
  switch (type) {
    case CollectivePerfTableGen::CollectiveType::ALL_REDUCE:
    case CollectivePerfTableGen::CollectiveType::REDUCE_SCATTER:
    case CollectivePerfTableGen::CollectiveType::ALL_TO_ALL:
    case CollectivePerfTableGen::CollectiveType::COLLECTIVE_PERMUTE:
      dim_size = tensor_size_bytes / kBytesPerElem;
      break;
    case CollectivePerfTableGen::CollectiveType::ALL_GATHER:
      dim_size = tensor_size_bytes /
                 (kBytesPerElem * replica_groups.num_devices_per_group());
      break;
    default:
      LOG(FATAL) << "Unsupported collective type.";
  }
  return dim_size;
}

int64_t GetOutputDim(CollectivePerfTableGen::CollectiveType type,
                     int64_t tensor_size_bytes,
                     IotaReplicaGroupList replica_groups) {
  int64_t dim_size = -1;
  CHECK_EQ(tensor_size_bytes % kBytesPerElem, 0);
  switch (type) {
    case CollectivePerfTableGen::CollectiveType::ALL_REDUCE:
    case CollectivePerfTableGen::CollectiveType::ALL_GATHER:
    case CollectivePerfTableGen::CollectiveType::ALL_TO_ALL:
    case CollectivePerfTableGen::CollectiveType::COLLECTIVE_PERMUTE:
      dim_size = tensor_size_bytes / kBytesPerElem;
      break;
    case CollectivePerfTableGen::CollectiveType::REDUCE_SCATTER:
      dim_size = tensor_size_bytes /
                 (kBytesPerElem * replica_groups.num_devices_per_group());
      break;
    default:
      LOG(FATAL) << "Unsupported collective type.";
  }
  return dim_size;
}

std::string GetHlo(
    CollectivePerfTableGen::CollectiveType type, int64_t input_dim,
    int64_t output_dim, const IotaReplicaGroupList& replica_groups,
    std::optional<CollectivePermuteCostModelType> collective_permute_pattern) {
  CHECK_EQ(kBytesPerElem, 4);
  CHECK(type != CollectivePerfTableGen::CollectiveType::COLLECTIVE_PERMUTE ||
        collective_permute_pattern.has_value());

  std::string hlo;
  switch (type) {
    case CollectivePerfTableGen::CollectiveType::ALL_REDUCE:
      hlo = absl::Substitute(R"(
        HloModule m

        add {
          a = f32[] parameter(0)
          b = f32[] parameter(1)
          ROOT res = add(a, b)
        }

        ENTRY e {
          p0 = $0[$1] parameter(0)
          ROOT _ = $0[$2] all-reduce(p0), replica_groups=$3,
            to_apply=add, use_global_device_ids=true, channel_id=1
        }
      )",
                             "f32", input_dim, output_dim,
                             replica_groups.ToString());
      break;
    case CollectivePerfTableGen::CollectiveType::ALL_GATHER:
      hlo = absl::Substitute(R"(
        HloModule m

        ENTRY e {
          p0 = $0[$1] parameter(0)
          ROOT _ = $0[$2] all-gather(p0), replica_groups=$3,
            use_global_device_ids=true, channel_id=1, dimensions={0}
        }
      )",
                             "f32", input_dim, output_dim,
                             replica_groups.ToString());
      break;
    case CollectivePerfTableGen::CollectiveType::REDUCE_SCATTER:
      hlo = absl::Substitute(R"(
        HloModule m

        add {
          a = f32[] parameter(0)
          b = f32[] parameter(1)
          ROOT res = add(a, b)
        }

        ENTRY e {
          p0 = $0[$1] parameter(0)
          ROOT _ = $0[$2] reduce-scatter(p0), replica_groups=$3,
            to_apply=add, use_global_device_ids=true, channel_id=1,
            dimensions={0}
        }
      )",
                             "f32", input_dim, output_dim,
                             replica_groups.ToString());
      break;
    case CollectivePerfTableGen::CollectiveType::ALL_TO_ALL:
      hlo = absl::Substitute(R"(
        HloModule m

        ENTRY e {
          p0 = $0[$1] parameter(0)
          ROOT _ = $0[$2] all-to-all(p0), replica_groups=$3, channel_id=1,
          dimensions={0}
        }
      )",
                             "f32", input_dim, output_dim,
                             replica_groups.ToString());
      break;
    case CollectivePerfTableGen::CollectiveType::COLLECTIVE_PERMUTE: {
      int num_devices = replica_groups.num_devices_per_group() *
                        replica_groups.num_replica_groups();
      std::string source_target_pairs =
          BuildSourceTargetPairs(*collective_permute_pattern, num_devices);
      hlo = absl::Substitute(
          R"(
        HloModule collective-permute-while-loop-microbenchmark, num_partitions=$2

        while_cond {
          iter = (f32[$0], u32[]) parameter(0)
          i = u32[] get-tuple-element(iter), index=1
          ub = u32[] constant(100)
          ROOT compare = pred[] compare(i, ub), direction=LT
        }

        while_body {
          iter = (f32[$0], u32[]) parameter(0)
          i = u32[] get-tuple-element(iter), index=1
          arg = f32[$0] get-tuple-element(iter), index=0
          collective-permute = f32[$0] collective-permute(arg), channel_id=1, source_target_pairs=$1
          c1 = u32[] constant(1)
          i_next = u32[] add(i, c1)
          ROOT out = (f32[$0], u32[]) tuple(collective-permute, i_next)
        }

        ENTRY main {
          arg = f32[$0] parameter(0)
          c0 = u32[] constant(0)
          cp_first_iter = f32[$0] collective-permute(arg), channel_id=1, source_target_pairs=$1
          init = (f32[$0], u32[]) tuple(cp_first_iter, c0)
          while = (f32[$0], u32[]) while(init), condition=while_cond, body=while_body
          ROOT result = f32[$0] get-tuple-element(while), index=0
        }
      )",
          input_dim, source_target_pairs, num_devices);
      break;
    }
    default:
      LOG(FATAL) << "Unsupported collective type.";
  }
  return hlo;
}

std::unique_ptr<HloModule> CreateCollectiveModule(const StaticSpec& spec) {
  int64_t input_dim = GetInputDim(spec.collective_type, spec.tensor_size_bytes,
                                  spec.replica_groups);

  int64_t output_dim = GetOutputDim(
      spec.collective_type, spec.tensor_size_bytes, spec.replica_groups);

  std::string hlo =
      GetHlo(spec.collective_type, input_dim, output_dim, spec.replica_groups,
             spec.collective_permute_pattern);

  HloModuleConfig config;
  config.set_num_partitions(spec.replica_groups.num_devices_per_group() *
                            spec.replica_groups.num_replica_groups());
  auto parsed = ParseAndReturnUnverifiedModule(hlo, config);
  CHECK_OK(parsed.status());
  return std::move(*parsed);
}

ExplicitSpec GetExplicitSpec(const StaticSpec& spec) {
  std::unique_ptr<HloModule> module = CreateCollectiveModule(spec);
  return ExplicitSpec{std::move(module)};
}

uint64_t GetNetworkThroughputBytesPerSec(absl::Duration runtime,
                                         int64_t tensor_size_bytes) {
  CHECK_NE(runtime, absl::ZeroDuration());
  return tensor_size_bytes * 1e9 / absl::ToInt64Nanoseconds(runtime);
}

IotaReplicaGroupList GetCollectiveDeviceList(
    absl::string_view collective_device_list_unparsed) {
  auto device_list_or_status =
      xla::ParseCollectiveDeviceListBase(collective_device_list_unparsed);
  if (device_list_or_status.ok()) {
    std::unique_ptr<xla::CollectiveDeviceListBase> list =
        std::move(device_list_or_status.value());
    if (auto* iota = dynamic_cast<xla::IotaReplicaGroupList*>(list.get())) {
      return *iota;
    }
    if (auto iota = list->MaybeConvertToIotaReplicaGroupList()) {
      return *iota;
    }
  }

  IotaReplicaGroupListProto proto;
  if (tsl::protobuf::TextFormat::ParseFromString(
          collective_device_list_unparsed, &proto)) {
    return IotaReplicaGroupList::FromProto(proto);
  }
  LOG(FATAL) << "Failed to parse collective device list: "
             << collective_device_list_unparsed;
}

}  // namespace

// Generates source_target_pairs based on pattern.
std::string BuildSourceTargetPairs(CollectivePermuteCostModelType pattern,
                                   int num_devices) {
  std::vector<std::pair<int, int>> pairs_vec;
  switch (pattern) {
    case CollectivePermuteCostModelType::kIntraPartitionOneWay:
      // Pattern: {0->n/2, 1->n/2+1, ...}
      // Requires even number of devices.
      CHECK_EQ(num_devices % 2, 0);
      for (int i = 0; i < num_devices / 2; ++i) {
        pairs_vec.push_back({i, i + num_devices / 2});
      }
      break;
    case CollectivePermuteCostModelType::kIntraPartitionTwoWayAllMutual:
      // Pattern: {0->1, 1->0, 2->3, 3->2, ...}
      // Requires even number of devices.
      CHECK_EQ(num_devices % 2, 0);
      for (int i = 0; i < num_devices / 2; ++i) {
        pairs_vec.push_back({2 * i, 2 * i + 1});
        pairs_vec.push_back({2 * i + 1, 2 * i});
      }
      break;
    case CollectivePermuteCostModelType::kIntraPartitionTwoWayHasNonMutual:
      // Ring pattern: {0->1, 1->2, ..., n-1->0}
      for (int i = 0; i < num_devices; ++i) {
        pairs_vec.push_back({i, (i + 1) % num_devices});
      }
      break;
    default:
      LOG(FATAL) << "Unsupported collective permute pattern.";
  }

  return absl::StrCat(
      "{",
      absl::StrJoin(pairs_vec, ",",
                    [](std::string* out, const std::pair<int, int>& pair) {
                      absl::StrAppend(out, "{", pair.first, ",", pair.second,
                                      "}");
                    }),
      "}");
}

/*static*/ std::unique_ptr<CollectivePerfTableGen>
CollectivePerfTableGen::Create(CollectivePerfTableGen::Config config) {
  return std::unique_ptr<CollectivePerfTableGen>(
      new CollectivePerfTableGen(config));
}

PjRtEnvironment& CollectivePerfTableGen::GetPjRtEnv() {
  if (pjrt_env_ == nullptr) {
    GpuClientOptions gpu_opts;
    gpu_opts.num_nodes = config_.num_nodes;
    gpu_opts.node_id = config_.task_id;
    gpu_opts.allocator_config.memory_fraction = kGpuMemFraction;
    absl::StatusOr<PjRtEnvironment> pjrt_env = GetPjRtEnvironmentForGpu(
        config_.coordinator_address, gpu_opts, config_.connection_timeout);
    CHECK_OK(pjrt_env);
    CHECK_NE(pjrt_env->client.get(), nullptr);
    pjrt_env_ = std::make_unique<PjRtEnvironment>(*std::move(pjrt_env));
  }
  CHECK_NE(pjrt_env_, nullptr);
  return *pjrt_env_;
}

Backend& CollectivePerfTableGen::GetBackend() {
  if (backend_ == nullptr) {
    backend_ = Backend::CreateDefaultBackend().value();
  }
  return *backend_;
}

std::unique_ptr<PjRtLoadedExecutable> CollectivePerfTableGen::Compile(
    std::unique_ptr<HloModule> module) {
  DebugOptions debug_opts;
  FunctionalHloRunner::RawCompileOptions opts;
  opts.num_partitions = module->config().num_partitions();
  opts.spmd_mode = FunctionalHloRunner::SpmdMode::kUseSpmdPartitioning;
  auto compile_opts = FunctionalHloRunner::CreateCompileOptions(
      *GetPjRtEnv().client, opts, config_.task_id, config_.num_nodes);
  CHECK_OK(compile_opts);
  auto executable = FunctionalHloRunner::Compile(
      *GetPjRtEnv().client, module.get(), debug_opts,
      /*preproc_options=*/{}, *compile_opts);
  CHECK_OK(executable);
  return std::move(*executable);
}

std::vector<ExecutionProfile> CollectivePerfTableGen::Run(
    PjRtLoadedExecutable& executable) {
  FunctionalHloRunner::RunningOptions run_opts;
  run_opts.module_argument_mode =
      FunctionalHloRunner::ModuleArgumentMode::kUninitialized;
  run_opts.num_repeats = kNumProfilingRuns;
  std::vector<ExecutionProfile> profiles;
  run_opts.execution_profiles = &profiles;
  CHECK_OK(FunctionalHloRunner::Run(*GetPjRtEnv().client, &executable,
                                    /*arguments=*/{}, run_opts));
  return profiles;
}

CollectivePerfTableGen::ProfilingData CollectivePerfTableGen::Profile(
    std::unique_ptr<HloModule> module) {
  auto executable = Compile(std::move(module));
  VLOG(1) << "Compiled module: "
          << executable->GetHloModules().value()[0]->ToString();

  // We do not profile dry runs or on more than one tasks.
  if (config_.dry_run) {
    return {};
  }

  if (config_.task_id == 0) {
    std::vector<ExecutionProfile> profiles = Run(*executable);
    return {
        /*runtime=*/absl::Nanoseconds(GetMedianRuntimeNs(std::move(profiles)))};
  }
  Run(*executable);
  return {};
}

DeviceHloInstructionProfiles CollectivePerfTableGen::ComputeTable() {
  std::vector<StaticSpec> static_specs;

  auto inc = [](int64_t i, const StepSpec& spec) {
    if (spec.step > 0) {
      return i + spec.step;
    }
    if (spec.factor > 0) {
      return i * spec.factor;
    }
    LOG(FATAL) << "Either factor or step should be set in the spec.";
    return i;
  };

  StepSpec tsize_spec = config_.tensor_size_bytes_spec;
  for (int64_t tensor_size = tsize_spec.start; tensor_size <= tsize_spec.stop;
       tensor_size = inc(tensor_size, tsize_spec)) {
    for (CollectiveType collective_type : config_.collective_types) {
      for (absl::string_view replica_groups_raw : config_.replica_groups_list) {
        CHECK(collective_type != CollectiveType::UNSPECIFIED);
        IotaReplicaGroupList replica_groups =
            GetCollectiveDeviceList(replica_groups_raw);
        int num_devices = replica_groups.num_devices_per_group() *
                          replica_groups.num_replica_groups();

        if (collective_type != CollectiveType::COLLECTIVE_PERMUTE) {
          static_specs.push_back(
              {collective_type, replica_groups, std::nullopt, tensor_size});
          continue;
        }
        for (CollectivePermuteCostModelType pattern :
             config_.collective_permute_patterns) {
          // Skip patterns that require an even number of devices if n is odd.
          if (num_devices % 2 != 0 &&
              (pattern ==
                   CollectivePermuteCostModelType::kIntraPartitionOneWay ||
               pattern == CollectivePermuteCostModelType::
                              kIntraPartitionTwoWayAllMutual)) {
            continue;
          }
          static_specs.push_back(
              {collective_type, replica_groups, pattern, tensor_size});
        }
      }
    }
  }
  std::vector<ExplicitSpec> specs;
  specs.reserve(static_specs.size());
  for (auto& spec : static_specs) {
    specs.emplace_back(GetExplicitSpec(spec));
  }

  HloInstructionProfileList profile_list;

  for (auto& static_spec : static_specs) {
    ExplicitSpec spec = GetExplicitSpec(static_spec);
    std::string fingerprint = spec.module->GetFingerprint128();
    HloInstruction* instr =
        static_spec.collective_type == CollectiveType::COLLECTIVE_PERMUTE
            ? spec.module->GetComputationWithName("while_body")
                  ->GetInstructionWithName("collective-permute")
            : spec.module->entry_computation()->root_instruction();
    CHECK(hlo_query::IsCollectiveCommunicationOp(instr->opcode()));

    HloInstructionProfile entry;
    *entry.mutable_instruction() = instr->ToProto();
    entry.set_fingerprint(fingerprint);
    ProfilingData profiled_data;
    if (!config_.dry_run) {
      profiled_data = Profile(std::move(spec.module));
    }
    if (profiled_data.runtime == absl::ZeroDuration()) {
      VLOG(1) << "Size: " << static_spec.tensor_size_bytes << " too small.";
      continue;
    }
    if (static_spec.collective_type == CollectiveType::COLLECTIVE_PERMUTE) {
      // collective permute template hlo runs in a 100x while loop to
      // average the variance. So we would need to divide the runtime
      // by 100 to get the runtime of a single iteration.
      profiled_data.runtime /= 100;
    }
    entry.set_network_throughput_bytes_per_sec(GetNetworkThroughputBytesPerSec(
        profiled_data.runtime, static_spec.tensor_size_bytes));

    *profile_list.add_entries() = entry;
  }

  DeviceHloInstructionProfiles profiles;
  if (profile_list.entries_size() == 0) {
    return profiles;
  }

  std::string device_key = HloOpProfiles::GetProfileName(
      /*device_info=*/GetBackend()
          .stream_executors()[0]
          ->GetDeviceDescription());
  profiles.mutable_entries()->insert({device_key, profile_list});
  return profiles;
}

absl::Status CollectivePerfTableGen::Dump(
    const DeviceHloInstructionProfiles& table) {
  if (config_.task_id != 0) {
    return absl::OkStatus();
  }
  if (config_.output == CollectivePerfTableGen::Config::kStdout) {
    LOG(INFO) << table.DebugString();
    return absl::OkStatus();
  }

  DeviceHloInstructionProfiles file;
  if (tsl::Env::Default()->FileExists(config_.output).ok()) {
    TF_RETURN_IF_ERROR(
        tsl::ReadTextOrBinaryProto(tsl::Env::Default(), config_.output, &file));
  }

  for (const auto& [sm_ver, entries] : table.entries()) {
    if (file.entries().contains(sm_ver)) {
      file.mutable_entries()->at(sm_ver).MergeFrom(entries);
    } else {
      file.MergeFrom(table);
    }

    if (absl::StrContains(config_.output, ".pbtxt")) {
      TF_RETURN_IF_ERROR(
          tsl::WriteTextProto(tsl::Env::Default(), config_.output, file));
      continue;
    }
    if (absl::StrContains(config_.output, ".pb")) {
      TF_RETURN_IF_ERROR(
          tsl::WriteBinaryProto(tsl::Env::Default(), config_.output, file));
      continue;
    }
    return absl::InvalidArgumentError(
        absl::StrCat("Unsupported file: ", config_.output,
                     ". Expecting .pb or .pbtxt suffix."));
  }
  return absl::OkStatus();
}

DeviceHloInstructionProfiles CollectivePerfTableGen::Merge(
    const std::vector<std::string>& files) {
  DeviceHloInstructionProfiles result;

  absl::flat_hash_set<ProfilingResult, ProfilingResult::Hash,
                      ProfilingResult::Eq>
      profiling_results;
  uint64_t profiling_results_counter = 0;
  for (const std::string& profile_path : files) {
    // Read file.
    DeviceHloInstructionProfiles partial_profile;
    CHECK_OK(tsl::Env::Default()->FileExists(profile_path));
    if (!tsl::ReadTextOrBinaryProto(tsl::Env::Default(), profile_path,
                                    &partial_profile)
             .ok()) {
      LOG(WARNING) << "Cannot read :" << profile_path;
      continue;
    }

    for (auto& [device_descriptor, data] : partial_profile.entries()) {
      for (const HloInstructionProfile& profile : data.entries()) {
        std::string fingerprint = profile.fingerprint();
        if (fingerprint.empty()) {
          fingerprint =
              absl::StrCat("no-fingerprint#", profiling_results_counter);
        }

        ProfilingResult profiling_result{
            device_descriptor,
            profile.instruction(),
            {
                profile.operands().begin(),
                profile.operands().end(),
            },
            fingerprint,
            profile.clock_cycles(),
            profile.flops(),
            profile.network_throughput_bytes_per_sec(),
        };
        profiling_results.insert(profiling_result);
        profiling_results_counter++;
      }
    }
  }
  LOG(INFO) << "Merging and deduplication entries count. Before "
            << profiling_results_counter << ", after "
            << profiling_results.size() << ".";

  for (const ProfilingResult& profiling_result : profiling_results) {
    std::string device_descriptor = profiling_result.device_info;
    if (!result.mutable_entries()->contains(device_descriptor)) {
      result.mutable_entries()->insert({device_descriptor, {}});
    }

    HloInstructionProfile profile_proto;
    *profile_proto.mutable_instruction() =
        std::move(profiling_result.hlo_proto);
    for (auto op : profiling_result.operands) {
      *profile_proto.add_operands() = std::move(op);
    }
    profile_proto.set_flops(profiling_result.flops);
    profile_proto.set_clock_cycles(profiling_result.clock_cycles);
    if (!absl::StartsWith(profiling_result.fingerprint, "no-fingerprint#")) {
      profile_proto.set_fingerprint(profiling_result.fingerprint);
    }
    profile_proto.set_network_throughput_bytes_per_sec(
        profiling_result.network_throughput);

    *result.mutable_entries()->at(device_descriptor).add_entries() =
        std::move(profile_proto);
  }

  return result;
}

DeviceHloInstructionProfiles CollectivePerfTableGen::Merge(
    absl::string_view merge_path) {
  std::vector<std::string> file_paths;
  std::vector<std::string> filenames;
  CHECK_OK(
      tsl::Env::Default()->GetChildren(std::string(merge_path), &filenames));
  for (const std::string& fname : filenames) {
    std::string file_path = absl::StrCat(merge_path, "/", fname);
    file_paths.push_back(file_path);
  }
  return Merge(file_paths);
}

}  // namespace xla::gpu
