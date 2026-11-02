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

// Tests for building the HloModule of a `GpuExecutable` from its serialized
// module on demand (see `GpuExecutable::GetOrCreateHloModule`). They don't need
// a GPU, so they live in a separate target that also runs with sanitizers (e.g.
// tsan).

#include <cstdint>
#include <memory>
#include <string>
#include <thread>  // NOLINT(build/c++11)
#include <utility>
#include <variant>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "xla/backends/gpu/runtime/gpu_executable.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/thunk_executor.h"
#include "xla/debug_options_flags.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_input_output_alias_config.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/literal_util.h"
#include "xla/service/device_assignment.h"
#include "xla/service/gpu/gpu_executable.pb.h"
#include "xla/service/gpu_topology.h"
#include "xla/service/hlo.pb.h"
#include "xla/service/hlo_module_config.h"
#include "xla/service/xla_debug_info_manager.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tsl/lib/monitoring/cell_reader.h"
#include "xla/tsl/util/proto/proto_matchers.h"
#include "xla/xla.pb.h"

namespace xla::gpu {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;
using ::testing::HasSubstr;
using ::tsl::monitoring::testing::CellReader;
using ::tsl::proto_testing::EqualsProto;

constexpr char kMaterializationCounter[] =
    "/xla/gpu/hlo_module_materializations";

// Returns a module config with non-default runtime metadata.
HloModuleConfig MakeConfig() {
  HloModuleConfig config;
  DebugOptions compile_time_options = GetDebugOptionsFromFlags();
  compile_time_options.set_xla_gpu_nccl_collective_max_nchannels(7);
  config.set_debug_options(compile_time_options);
  config.set_partition_size(4);
  DeviceAssignment device_assignment(/*replica_count=*/1,
                                     /*computation_count=*/2);
  device_assignment(0, 0) = 1;
  device_assignment(0, 1) = 0;
  config.set_static_device_assignment(device_assignment);
  return config;
}

// Returns params for an executable with a module that has a single constant
// and the given config.
GpuExecutable::Params MakeParamsWithModule(const HloModuleConfig& config) {
  HloComputation::Builder builder("test_computation");
  HloInstruction* root = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0(1)));
  auto module = std::make_unique<HloModule>("test_module", config);
  module->AddEntryComputation(builder.Build(root));

  GpuExecutable::Params params;
  params.executable = std::make_unique<ThunkExecutor>(ThunkSequence{});
  params.device_description.set_gpu_compute_capability(
      se::GpuComputeCapability{se::CudaComputeCapability::Volta()});
  params.enable_debug_info_manager = false;
  params.module_name = module->name();
  params.hlo_module = std::move(module);
  params.debug_options = GetDebugOptionsFromFlags();
  params.gpu_topology =
      GpuTopology("test_platform", /*num_partitions=*/1,
                  /*num_hosts_per_partition=*/1, /*num_devices_per_host=*/1);
  params.buffer_allocations_debug_summary = "dummy";
  return params;
}

// Returns params for an executable that has no module but keeps the serialized
// module instead. That's the state that loaded executables will be in once
// loading stops building the module.
GpuExecutable::Params MakeParamsWithSerializedModule(
    const HloModuleConfig& config) {
  GpuExecutable::Params params = MakeParamsWithModule(config);
  HloModuleProtoWithConfig proto =
      std::get<std::shared_ptr<HloModule>>(params.hlo_module)
          ->ToProtoWithConfig();
  proto.mutable_hlo_module()->set_pjrt_id(HloModule::GetNextUniqueModuleId());
  params.hlo_module = std::move(proto);
  // `Create` can't compute this without building the module. The module has no
  // collectives, so it would give true.
  params.collective_use_minimal_resource = true;
  return params;
}

// Modifies the serialized module in `params`.
template <typename Fn>
void ModifySerializedModule(GpuExecutable::Params& params, Fn fn) {
  fn(std::get<HloModuleProtoWithConfig>(params.hlo_module));
}

TEST(GpuExecutableHloModuleTest, ReturnsExistingModule) {
  CellReader<int64_t> counter(kMaterializationCounter);
  GpuExecutable::Params params = MakeParamsWithModule(MakeConfig());
  params.debug_options
      .set_xla_gpu_experimental_fail_on_lazy_hlo_module_materialization(true);
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<GpuExecutable> executable,
                       GpuExecutable::Create(std::move(params)));

  // Nothing is built, so neither the fail flag nor the counter apply.
  EXPECT_THAT(executable->GetOrCreateHloModule("existing"),
              IsOkAndHolds(executable->shared_module()));
  EXPECT_EQ(counter.Delta("existing"), 0);
}

TEST(GpuExecutableHloModuleTest, FailsWithoutAnyModule) {
  GpuExecutable::Params params = MakeParamsWithModule(MakeConfig());
  params.hlo_module = std::monostate();
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<GpuExecutable> executable,
                       GpuExecutable::Create(std::move(params)));

  EXPECT_FALSE(executable->has_module());
  EXPECT_THAT(executable->GetOrCreateHloModule("test"),
              StatusIs(absl::StatusCode::kNotFound));
}

TEST(GpuExecutableHloModuleTest, NullModuleMeansNoModule) {
  GpuExecutable::Params params = MakeParamsWithModule(MakeConfig());
  params.hlo_module = nullptr;
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<GpuExecutable> executable,
                       GpuExecutable::Create(std::move(params)));

  EXPECT_FALSE(executable->has_module());
  EXPECT_THAT(executable->GetOrCreateHloModule("test"),
              StatusIs(absl::StatusCode::kNotFound));
}

TEST(GpuExecutableHloModuleTest, CreateRequiresPjrtId) {
  GpuExecutable::Params params = MakeParamsWithSerializedModule(MakeConfig());
  ModifySerializedModule(params, [](HloModuleProtoWithConfig& proto) {
    proto.mutable_hlo_module()->clear_pjrt_id();
  });

  EXPECT_THAT(
      GpuExecutable::Create(std::move(params)),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("pjrt_id")));
}

TEST(GpuExecutableHloModuleTest, BuildsModuleOnce) {
  CellReader<int64_t> counter(kMaterializationCounter);
  HloModuleConfig config = MakeConfig();
  GpuExecutable::Params params = MakeParamsWithSerializedModule(config);
  const int64_t unique_id =
      std::get<HloModuleProtoWithConfig>(params.hlo_module)
          .hlo_module()
          .pjrt_id();
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<GpuExecutable> executable,
                       GpuExecutable::Create(std::move(params)));
  ASSERT_FALSE(executable->has_module());

  // The runtime metadata comes from the serialized module without building
  // it.
  EXPECT_EQ(executable->runtime_debug_options()
                .xla_gpu_nccl_collective_max_nchannels(),
            7);
  ASSERT_TRUE(executable->static_device_assignment().has_value());
  EXPECT_EQ(executable->static_device_assignment()->ToString(),
            config.static_device_assignment().ToString());
  EXPECT_EQ(counter.Delta("first_reason"), 0);

  ASSERT_OK_AND_ASSIGN(std::shared_ptr<HloModule> first,
                       executable->GetOrCreateHloModule("first_reason"));
  ASSERT_OK_AND_ASSIGN(std::shared_ptr<HloModule> second,
                       executable->GetOrCreateHloModule("second_reason"));

  ASSERT_NE(first, nullptr);
  EXPECT_EQ(first, second);
  EXPECT_EQ(first->unique_id(), unique_id);
  EXPECT_EQ(first->name(), executable->name());
  EXPECT_THAT(first->config().debug_options(),
              EqualsProto(executable->runtime_debug_options()));
  EXPECT_EQ(counter.Delta("first_reason"), 1);
  EXPECT_EQ(counter.Delta("second_reason"), 0);
  // The built module becomes the module of the executable.
  EXPECT_TRUE(executable->has_module());
  EXPECT_EQ(executable->shared_module(), first);
}

TEST(GpuExecutableHloModuleTest, ModuleMetadataFromSerializedModule) {
  GpuExecutable::Params params = MakeParamsWithSerializedModule(MakeConfig());
  // Makes sure that reading the metadata doesn't build the module.
  params.debug_options
      .set_xla_gpu_experimental_fail_on_lazy_hlo_module_materialization(true);
  ModifySerializedModule(params, [](HloModuleProtoWithConfig& proto) {
    HloBufferDonorProto::BufferDonorEntryProto* entry =
        proto.mutable_hlo_module()->mutable_buffer_donor()->add_entries();
    entry->set_parameter_number(0);
    entry->add_parameter_shape_index(1);
  });
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<GpuExecutable> executable,
                       GpuExecutable::Create(std::move(params)));

  const HloModuleConfig& config = executable->module_config();
  EXPECT_EQ(config.partition_size(), 4);
  ASSERT_TRUE(config.has_static_device_assignment());
  EXPECT_EQ(config.static_device_assignment().ToString(),
            MakeConfig().static_device_assignment().ToString());
  // The same debug options as the module would get when it's built.
  EXPECT_THAT(config.debug_options(),
              EqualsProto(executable->runtime_debug_options()));
  EXPECT_TRUE(executable->buffer_donor_config().ParameterIsBufferDonor(
      /*param_number=*/0, /*param_index=*/{1}));
  EXPECT_FALSE(executable->has_module());
}

TEST(GpuExecutableHloModuleTest, ModuleMetadataFromBuiltModule) {
  GpuExecutable::Params params = MakeParamsWithSerializedModule(MakeConfig());
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<GpuExecutable> executable,
                       GpuExecutable::Create(std::move(params)));
  const HloModuleConfig& config_before_build = executable->module_config();
  const int64_t partition_size = config_before_build.partition_size();
  const DebugOptions debug_options = config_before_build.debug_options();

  ASSERT_OK_AND_ASSIGN(std::shared_ptr<HloModule> module,
                       executable->GetOrCreateHloModule("test"));

  // Once the module is built, the metadata is the module's.
  EXPECT_EQ(&executable->module_config(), &module->config());
  EXPECT_EQ(&executable->input_output_alias_config(),
            &module->input_output_alias_config());
  EXPECT_EQ(&executable->buffer_donor_config(), &module->buffer_donor_config());
  // And it has the same values as before the build.
  EXPECT_EQ(module->config().partition_size(), partition_size);
  EXPECT_THAT(module->config().debug_options(), EqualsProto(debug_options));
}

TEST(GpuExecutableHloModuleTest, ToProtoWritesSerializedModuleWhenNotBuilt) {
  GpuExecutable::Params params = MakeParamsWithSerializedModule(MakeConfig());
  // Makes sure that `ToProto` doesn't build the module.
  params.debug_options
      .set_xla_gpu_experimental_fail_on_lazy_hlo_module_materialization(true);
  const HloModuleProtoWithConfig serialized_module =
      std::get<HloModuleProtoWithConfig>(params.hlo_module);
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<GpuExecutable> executable,
                       GpuExecutable::Create(std::move(params)));

  ASSERT_OK_AND_ASSIGN(GpuExecutableProto proto, executable->ToProto());
  EXPECT_FALSE(executable->has_module());
  EXPECT_THAT(proto.hlo_module_with_config(), EqualsProto(serialized_module));
  // The config that `FromProto` reads the runtime metadata from is preserved.
  const HloModuleConfigProto& config = proto.hlo_module_with_config().config();
  EXPECT_EQ(config.debug_options().xla_gpu_nccl_collective_max_nchannels(), 7);
  EXPECT_EQ(config.partition_size(), 4);
  EXPECT_TRUE(config.has_static_device_assignment());
}

TEST(GpuExecutableHloModuleTest, DestroysExecutableWithBuiltModule) {
  // Only modules that were there at construction are registered with the
  // XlaDebugInfoManager, so destroying the executable must not unregister the
  // built module (which would crash).
  GpuExecutable::Params params = MakeParamsWithSerializedModule(MakeConfig());
  params.enable_debug_info_manager = true;
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<GpuExecutable> executable,
                       GpuExecutable::Create(std::move(params)));
  ASSERT_OK_AND_ASSIGN(std::shared_ptr<HloModule> module,
                       executable->GetOrCreateHloModule("test"));
  EXPECT_FALSE(XlaDebugInfoManager::Get()->TracksModule(module->unique_id()));

  executable.reset();
}

TEST(GpuExecutableHloModuleTest, RegistersAndUnregistersExistingModule) {
  GpuExecutable::Params params = MakeParamsWithModule(MakeConfig());
  params.enable_debug_info_manager = true;
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<GpuExecutable> executable,
                       GpuExecutable::Create(std::move(params)));
  const int64_t unique_id = executable->module().unique_id();
  EXPECT_TRUE(XlaDebugInfoManager::Get()->TracksModule(unique_id));

  executable.reset();
  EXPECT_FALSE(XlaDebugInfoManager::Get()->TracksModule(unique_id));
}

TEST(GpuExecutableHloModuleTest, ConcurrentCallersBuildOnce) {
  constexpr int kNumThreads = 8;
  CellReader<int64_t> counter(kMaterializationCounter);
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<GpuExecutable> executable,
      GpuExecutable::Create(MakeParamsWithSerializedModule(MakeConfig())));

  std::vector<HloModule*> results(kNumThreads, nullptr);
  std::vector<std::thread> threads;
  threads.reserve(kNumThreads);
  for (int i = 0; i < kNumThreads; ++i) {
    threads.emplace_back([&, i] {
      absl::StatusOr<std::shared_ptr<HloModule>> module =
          executable->GetOrCreateHloModule("concurrent");
      if (module.ok()) {
        results[i] = module->get();
      }
    });
  }
  for (std::thread& thread : threads) {
    thread.join();
  }

  ASSERT_NE(results[0], nullptr);
  for (HloModule* result : results) {
    EXPECT_EQ(result, results[0]);
  }
  EXPECT_EQ(&executable->module(), results[0]);
  EXPECT_EQ(counter.Delta("concurrent"), 1);
}

TEST(GpuExecutableHloModuleTest, FailFlagAtLoadTime) {
  CellReader<int64_t> counter(kMaterializationCounter);
  GpuExecutable::Params params = MakeParamsWithSerializedModule(MakeConfig());
  params.debug_options
      .set_xla_gpu_experimental_fail_on_lazy_hlo_module_materialization(true);
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<GpuExecutable> executable,
                       GpuExecutable::Create(std::move(params)));

  EXPECT_THAT(
      executable->GetOrCreateHloModule("forbidden"),
      StatusIs(
          absl::StatusCode::kFailedPrecondition,
          HasSubstr(
              "xla_gpu_experimental_fail_on_lazy_hlo_module_materialization")));
  // The error is sticky.
  EXPECT_THAT(executable->GetOrCreateHloModule("forbidden"),
              StatusIs(absl::StatusCode::kFailedPrecondition));
  // The attempt is still counted, so that we can see which code paths need the
  // module.
  EXPECT_EQ(counter.Delta("forbidden"), 1);
}

TEST(GpuExecutableHloModuleTest, FailFlagAtCompileTime) {
  HloModuleConfig config = MakeConfig();
  DebugOptions compile_time_options = config.debug_options();
  compile_time_options
      .set_xla_gpu_experimental_fail_on_lazy_hlo_module_materialization(true);
  config.set_debug_options(compile_time_options);
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<GpuExecutable> executable,
      GpuExecutable::Create(MakeParamsWithSerializedModule(config)));

  EXPECT_THAT(executable->GetOrCreateHloModule("forbidden"),
              StatusIs(absl::StatusCode::kFailedPrecondition));
}

TEST(GpuExecutableHloModuleTest, ErrorIsSticky) {
  CellReader<int64_t> counter(kMaterializationCounter);
  GpuExecutable::Params params = MakeParamsWithSerializedModule(MakeConfig());
  ModifySerializedModule(params, [](HloModuleProtoWithConfig& proto) {
    proto.mutable_hlo_module()->clear_host_program_shape();
  });
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<GpuExecutable> executable,
                       GpuExecutable::Create(std::move(params)));

  absl::StatusOr<std::shared_ptr<HloModule>> first =
      executable->GetOrCreateHloModule("broken");
  ASSERT_FALSE(first.ok());
  EXPECT_EQ(executable->GetOrCreateHloModule("broken").status(),
            first.status());
  EXPECT_EQ(counter.Delta("broken"), 1);
}

TEST(GpuExecutableHloModuleTest, CreateRejectsNegativePjrtId) {
  GpuExecutable::Params params = MakeParamsWithSerializedModule(MakeConfig());
  // A negative pjrt_id would make `HloModule::CreateFromProto` pick a fresh
  // unique id.
  ModifySerializedModule(params, [](HloModuleProtoWithConfig& proto) {
    proto.mutable_hlo_module()->set_pjrt_id(-1);
  });

  EXPECT_THAT(
      GpuExecutable::Create(std::move(params)),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("pjrt_id")));
}

TEST(GpuExecutableHloModuleTest, ComputesFingerprintFromModule) {
  GpuExecutable::Params params = MakeParamsWithModule(MakeConfig());
  const std::string expected_fingerprint =
      std::get<std::shared_ptr<HloModule>>(params.hlo_module)
          ->GetFingerprint128();
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<GpuExecutable> executable,
                       GpuExecutable::Create(std::move(params)));

  EXPECT_EQ(executable->fingerprint(), expected_fingerprint);
}

TEST(GpuExecutableHloModuleTest, KeepsStoredFingerprintWithoutBuildingModule) {
  GpuExecutable::Params params = MakeParamsWithSerializedModule(MakeConfig());
  params.debug_options
      .set_xla_gpu_experimental_fail_on_lazy_hlo_module_materialization(true);
  params.fingerprint = "stored_fingerprint";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<GpuExecutable> executable,
                       GpuExecutable::Create(std::move(params)));

  EXPECT_EQ(executable->fingerprint(), "stored_fingerprint");
  EXPECT_FALSE(executable->has_module());
}

TEST(GpuExecutableHloModuleTest,
     SerializedModuleWithoutFingerprintDoesNotBuildModule) {
  GpuExecutable::Params params = MakeParamsWithSerializedModule(MakeConfig());
  params.debug_options
      .set_xla_gpu_experimental_fail_on_lazy_hlo_module_materialization(true);
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<GpuExecutable> executable,
                       GpuExecutable::Create(std::move(params)));

  // The fingerprint is unknown, and we don't build the module just for that.
  EXPECT_TRUE(executable->fingerprint().empty());
  EXPECT_FALSE(executable->has_module());
}

TEST(GpuExecutableHloModuleTest,
     SerializedModuleRequiresCollectiveUseMinimalResource) {
  GpuExecutable::Params params = MakeParamsWithSerializedModule(MakeConfig());
  params.collective_use_minimal_resource.reset();

  EXPECT_THAT(GpuExecutable::Create(std::move(params)),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("collective_use_minimal_resource")));
}

TEST(GpuExecutableHloModuleTest,
     ComputesCollectiveUseMinimalResourceFromModule) {
  // The module has no collectives, so collectives can use minimal resources.
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<GpuExecutable> executable,
      GpuExecutable::Create(MakeParamsWithModule(MakeConfig())));
  ASSERT_OK_AND_ASSIGN(GpuExecutableProto proto, executable->ToProto());
  EXPECT_TRUE(proto.collective_use_minimal_resource());
}

}  // namespace
}  // namespace xla::gpu
