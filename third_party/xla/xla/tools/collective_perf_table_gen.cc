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

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/substitute.h"
#include "absl/time/time.h"
#include "xla/hlo/ir/collective_device_list.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/hlo/utils/hlo_query.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/plugin/xla_gpu/xla_gpu_client_options.h"
#include "xla/service/gpu/model/hlo_op_profile.pb.h"
#include "xla/service/gpu/model/hlo_op_profiler.h"
#include "xla/service/gpu/model/hlo_op_profiles.h"
#include "xla/tools/multihost_hlo_runner/create_client.h"
#include "xla/tools/multihost_hlo_runner/functional_hlo_runner.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"
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
constexpr uint8_t kNumProfilingRuns = 5;

struct StaticSpec {
  CollectivePerfTableGen::CollectiveType collective_type;
  IotaReplicaGroupList replica_groups;
  int64_t tensor_size_bytes;
};

struct ExplicitSpec {
  std::unique_ptr<HloModule> module;
};

int64_t GetInputDim(CollectivePerfTableGen::CollectiveType type,
                    int64_t tensor_size_bytes,
                    IotaReplicaGroupList replica_groups) {
  int64_t dim_size = -1;
  CHECK_EQ(tensor_size_bytes % kBytesPerElem, 0);
  switch (type) {
    case CollectivePerfTableGen::CollectiveType::ALL_REDUCE:
      dim_size = tensor_size_bytes / kBytesPerElem;
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
      dim_size = tensor_size_bytes / kBytesPerElem;
      break;
    default:
      LOG(FATAL) << "Unsupported collective type.";
  }
  return dim_size;
}

std::string GetHlo(CollectivePerfTableGen::CollectiveType type,
                   int64_t input_dim, int64_t output_dim,
                   const IotaReplicaGroupList& replica_groups) {
  std::string hlo;
  switch (type) {
    case CollectivePerfTableGen::CollectiveType::ALL_REDUCE:
      CHECK_EQ(kBytesPerElem, 4);
      hlo = absl::Substitute(R"(
        HloModule m

        add {
          a = f32[] parameter(0)
          b = f32[] parameter(1)
          ROOT res = add(a, b)
        }

        ENTRY e {
          p0 = $0[$1] parameter(0)
          ROOT _ = $0[$2] $3(p0), replica_groups=$4,
            to_apply=add, use_global_device_ids=true, channel_id=1
        }
      )",
                             "f32", input_dim, output_dim, "all-reduce",
                             replica_groups.ToString());
      break;
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
      GetHlo(spec.collective_type, input_dim, output_dim, spec.replica_groups);

  auto parsed = ParseAndReturnUnverifiedModule(hlo);
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

}  // namespace

/*static*/ std::unique_ptr<CollectivePerfTableGen>
CollectivePerfTableGen::Create(CollectivePerfTableGen::Config config) {
  GpuClientOptions gpu_opts;
  gpu_opts.num_nodes = config.num_nodes;
  gpu_opts.node_id = config.task_id;
  gpu_opts.allocator_config.memory_fraction = kGpuMemFraction;

  auto pjrt_env = GetPjRtEnvironmentForGpu(config.coordinator_address, gpu_opts,
                                           config.connection_timeout);
  CHECK_OK(pjrt_env);
  CHECK_NE(pjrt_env->client.get(), nullptr);

  return std::unique_ptr<CollectivePerfTableGen>(
      new CollectivePerfTableGen(config, std::move(*pjrt_env)));
}

std::unique_ptr<PjRtLoadedExecutable> CollectivePerfTableGen::Compile(
    std::unique_ptr<HloModule> module) {
  DebugOptions debug_opts;
  FunctionalHloRunner::RawCompileOptions opts;
  opts.num_partitions = 8;
  opts.spmd_mode = FunctionalHloRunner::SpmdMode::kUseSpmdPartitioning;
  auto compile_opts = FunctionalHloRunner::CreateCompileOptions(
      *pjrt_env_.client, opts, config_.task_id, config_.num_nodes);
  CHECK_OK(compile_opts);
  auto executable =
      FunctionalHloRunner::Compile(*pjrt_env_.client, module.get(), debug_opts,
                                   /*preproc_options=*/{}, *compile_opts);
  CHECK_OK(executable);
  return std::move(*executable);
}

void CollectivePerfTableGen::Run(PjRtLoadedExecutable& executable) {
  FunctionalHloRunner::RunningOptions run_opts;
  run_opts.module_argument_mode =
      FunctionalHloRunner::ModuleArgumentMode::kUninitialized;
  CHECK_OK(FunctionalHloRunner::Run(*pjrt_env_.client, &executable,
                                    /*arguments=*/{}, run_opts));
}

CollectivePerfTableGen::ProfilingData CollectivePerfTableGen::Profile(
    std::unique_ptr<HloModule> module) {
  auto executable = Compile(std::move(module));
  VLOG(1) << "Compiled module: "
          << executable->GetHloModules().value()[0]->ToString();

  if (config_.dry_run) {
    return {};
  }

  std::unique_ptr<HloOpProfiler::KernelTracer> tracer =
      HloOpProfiler::GetKernelTracer();
  for (int i = 0; i < kNumProfilingRuns; ++i) {
    Run(*executable);
  }
  return {
      /*runtime=*/absl::Nanoseconds(std::move(*tracer).getMedianKernelTimeNs()),
  };
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
      for (const IotaReplicaGroupList& replica_groups :
           config_.replica_groups_list) {
        CHECK(collective_type != CollectiveType::UNSPECIFIED);

        StaticSpec spec{collective_type, replica_groups, tensor_size};
        static_specs.push_back(spec);
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
        spec.module->entry_computation()->root_instruction();
    CHECK(hlo_query::IsCollectiveCommunicationOp(instr->opcode()));

    HloInstructionProfile entry;
    *entry.mutable_instruction() = instr->ToProto();
    entry.set_fingerprint(fingerprint);
    ProfilingData profiled_data;
    if (!config_.dry_run) {
      profiled_data = Profile(std::move(spec.module));
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
      /*device_info=*/backend_->stream_executors()[0]->GetDeviceDescription());
  profiles.mutable_entries()->insert({device_key, profile_list});
  return profiles;
}

absl::Status CollectivePerfTableGen::Dump(
    const DeviceHloInstructionProfiles& table) {
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

}  // namespace xla::gpu
