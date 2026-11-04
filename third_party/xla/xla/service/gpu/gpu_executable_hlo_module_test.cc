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
// module on demand (see `Executable::GetOrCreateHloModule`). They don't need a
// GPU, so
// they live in a separate target that also runs with sanitizers (e.g. tsan).

#include <cstdint>
#include <memory>
#include <string>
#include <thread>  // NOLINT(build/c++11)
#include <utility>
#include <variant>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/base/log_severity.h"
#include "absl/log/scoped_mock_log.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/thunk_executor.h"
#include "xla/debug_options_flags.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/literal_util.h"
#include "xla/service/device_assignment.h"
#include "xla/service/gpu/gpu_executable.h"
#include "xla/service/gpu/gpu_executable.pb.h"
#include "xla/service/gpu_topology.h"
#include "xla/service/hlo.pb.h"
#include "xla/service/hlo_module_config.h"
#include "xla/service/hlo_proto_util.h"
#include "xla/service/xla_debug_info_manager.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tsl/lib/monitoring/cell_reader.h"
#include "xla/tsl/profiler/backends/cpu/annotation_stack.h"
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
  EXPECT_EQ(executable->partition_size(), 4);
  ASSERT_NE(executable->static_device_assignment(), nullptr);
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

TEST(GpuExecutableHloModuleTest, RegistersModuleWithDebugInfoManager) {
  GpuExecutable::Params params = MakeParamsWithModule(MakeConfig());
  params.enable_debug_info_manager = true;
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<GpuExecutable> executable,
                       GpuExecutable::Create(std::move(params)));
  const int unique_id = executable->module().unique_id();

  EXPECT_TRUE(XlaDebugInfoManager::Get()->TracksModule(unique_id));
  executable.reset();
  EXPECT_FALSE(XlaDebugInfoManager::Get()->TracksModule(unique_id));
}

// Enables the annotation stack, which makes `MaybePrepareProfiling` consider
// the profiler active, for the lifetime of this object.
class ScopedEnableAnnotations {
 public:
  ScopedEnableAnnotations() { tsl::profiler::AnnotationStack::Enable(true); }
  ~ScopedEnableAnnotations() { tsl::profiler::AnnotationStack::Enable(false); }
};

TEST(GpuExecutableHloModuleTest, PreparesProfilingOnlyIfProfiling) {
  CellReader<int64_t> counter(kMaterializationCounter);
  GpuExecutable::Params params = MakeParamsWithSerializedModule(MakeConfig());
  params.enable_debug_info_manager = true;
  const int unique_id = std::get<HloModuleProtoWithConfig>(params.hlo_module)
                            .hlo_module()
                            .pjrt_id();
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<GpuExecutable> executable,
                       GpuExecutable::Create(std::move(params)));
  XlaDebugInfoManager* debug_info_manager = XlaDebugInfoManager::Get();

  // Without a profiler the module isn't built and not registered.
  ASSERT_FALSE(tsl::profiler::AnnotationStack::IsEnabled());
  executable->MaybePrepareProfiling();
  EXPECT_EQ(counter.Delta("profiler"), 0);
  EXPECT_FALSE(debug_info_manager->TracksModule(unique_id));

  // With a profiler the module is built once and registered.
  {
    ScopedEnableAnnotations enable_annotations;
    executable->MaybePrepareProfiling();
    executable->MaybePrepareProfiling();
  }
  EXPECT_EQ(counter.Delta("profiler"), 1);
  EXPECT_TRUE(debug_info_manager->TracksModule(unique_id));

  executable.reset();
  EXPECT_FALSE(debug_info_manager->TracksModule(unique_id));
}

TEST(GpuExecutableHloModuleTest, ConcurrentProfilingPreparationBuildsOnce) {
  constexpr int kNumThreads = 8;
  CellReader<int64_t> counter(kMaterializationCounter);
  GpuExecutable::Params params = MakeParamsWithSerializedModule(MakeConfig());
  params.enable_debug_info_manager = true;
  const int unique_id = std::get<HloModuleProtoWithConfig>(params.hlo_module)
                            .hlo_module()
                            .pjrt_id();
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<GpuExecutable> executable,
                       GpuExecutable::Create(std::move(params)));

  {
    ScopedEnableAnnotations enable_annotations;
    std::vector<std::thread> threads;
    threads.reserve(kNumThreads);
    for (int i = 0; i < kNumThreads; ++i) {
      threads.emplace_back([&] { executable->MaybePrepareProfiling(); });
    }
    for (std::thread& thread : threads) {
      thread.join();
    }
  }

  EXPECT_EQ(counter.Delta("profiler"), 1);
  EXPECT_TRUE(XlaDebugInfoManager::Get()->TracksModule(unique_id));
}

TEST(GpuExecutableHloModuleTest, ProfilingPreparationFailsOnce) {
  CellReader<int64_t> counter(kMaterializationCounter);
  GpuExecutable::Params params = MakeParamsWithSerializedModule(MakeConfig());
  params.enable_debug_info_manager = true;
  params.debug_options
      .set_xla_gpu_experimental_fail_on_lazy_hlo_module_materialization(true);
  const int unique_id = std::get<HloModuleProtoWithConfig>(params.hlo_module)
                            .hlo_module()
                            .pjrt_id();
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<GpuExecutable> executable,
                       GpuExecutable::Create(std::move(params)));

  {
    absl::ScopedMockLog log(absl::MockLogDefault::kIgnoreUnexpected);
    EXPECT_CALL(log,
                Log(absl::LogSeverity::kWarning, ::testing::_,
                    HasSubstr("won't contain per-instruction annotations")))
        .Times(1);
    log.StartCapturingLogs();
    ScopedEnableAnnotations enable_annotations;
    executable->MaybePrepareProfiling();
    executable->MaybePrepareProfiling();
  }

  EXPECT_EQ(counter.Delta("profiler"), 1);
  EXPECT_FALSE(XlaDebugInfoManager::Get()->TracksModule(unique_id));
}

TEST(GpuExecutableHloModuleTest, PreparesProfilingAfterOtherMaterialization) {
  CellReader<int64_t> counter(kMaterializationCounter);
  GpuExecutable::Params params = MakeParamsWithSerializedModule(MakeConfig());
  params.enable_debug_info_manager = true;
  const int unique_id = std::get<HloModuleProtoWithConfig>(params.hlo_module)
                            .hlo_module()
                            .pjrt_id();
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<GpuExecutable> executable,
                       GpuExecutable::Create(std::move(params)));
  XlaDebugInfoManager* debug_info_manager = XlaDebugInfoManager::Get();

  // A module built for another reason isn't registered.
  ASSERT_OK(executable->GetOrCreateHloModule("other").status());
  ASSERT_TRUE(executable->has_module());
  EXPECT_FALSE(debug_info_manager->TracksModule(unique_id));

  // Profiling uses that module instead of building another one, and registers
  // it once.
  {
    ScopedEnableAnnotations enable_annotations;
    executable->MaybePrepareProfiling();
    executable->MaybePrepareProfiling();
  }
  EXPECT_EQ(counter.Delta("other"), 1);
  EXPECT_EQ(counter.Delta("profiler"), 0);
  EXPECT_TRUE(debug_info_manager->TracksModule(unique_id));

  executable.reset();
  EXPECT_FALSE(debug_info_manager->TracksModule(unique_id));
}

TEST(GpuExecutableHloModuleTest, ProfilingPreparationIsNoOpWithModule) {
  CellReader<int64_t> counter(kMaterializationCounter);
  GpuExecutable::Params params = MakeParamsWithModule(MakeConfig());
  params.enable_debug_info_manager = true;
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<GpuExecutable> executable,
                       GpuExecutable::Create(std::move(params)));

  {
    ScopedEnableAnnotations enable_annotations;
    executable->MaybePrepareProfiling();
  }

  EXPECT_EQ(counter.Delta("profiler"), 0);
  // The module was registered at construction, as before.
  EXPECT_TRUE(XlaDebugInfoManager::Get()->TracksModule(
      executable->module().unique_id()));
}

// Returns the debug info that XlaDebugInfoManager reports for `module_id` in a
// trace during which `executable` is destroyed.
absl::StatusOr<HloProto> TraceAndDestroy(
    std::unique_ptr<GpuExecutable> executable, int module_id) {
  XlaDebugInfoManager* debug_info_manager = XlaDebugInfoManager::Get();
  debug_info_manager->StartTracing();
  executable.reset();
  std::vector<std::unique_ptr<HloProto>> debug_info;
  debug_info_manager->StopTracing(&debug_info);
  for (const std::unique_ptr<HloProto>& hlo_proto : debug_info) {
    if (hlo_proto->hlo_module().id() == module_id) {
      return *hlo_proto;
    }
  }
  return absl::NotFoundError("No debug info for the module.");
}

TEST(GpuExecutableHloModuleTest,
     SerializedModuleDebugInfoMatchesDeserializedModule) {
  // A module with things that the serialization changes: identical backend
  // configs (which are deduplicated), several computations, metadata and
  // instruction ids that aren't dense.
  constexpr absl::string_view kHlo = R"(
    HloModule debug_info_module, entry_computation_layout={(f32[16]{0})->f32[16]{0}}

    fused_add {
      p0 = f32[16] parameter(0)
      p1 = f32[16] parameter(1)
      ROOT add = f32[16] add(p0, p1), metadata={op_name="jit(f)/add" source_file="f.py" source_line=3}
    }

    fused_multiply {
      p0 = f32[16] parameter(0)
      p1 = f32[16] parameter(1)
      ROOT multiply = f32[16] multiply(p0, p1), metadata={op_name="jit(f)/mul" source_file="f.py" source_line=4}
    }

    while_condition {
      state = (s32[], f32[16]) parameter(0)
      i = s32[] get-tuple-element(state), index=0
      limit = s32[] constant(4)
      ROOT less_than = pred[] compare(i, limit), direction=LT
    }

    while_body {
      state = (s32[], f32[16]) parameter(0)
      i = s32[] get-tuple-element(state), index=0
      one = s32[] constant(1)
      next_i = s32[] add(i, one)
      x = f32[16] get-tuple-element(state), index=1
      fusion.1 = f32[16] fusion(x, x), kind=kLoop, calls=fused_add, backend_config={"operation_queue_id":"0","wait_on_operation_queues":[],"fusion_backend_config":{"kind":"__loop_fusion"}}
      fusion.2 = f32[16] fusion(fusion.1, x), kind=kLoop, calls=fused_multiply, backend_config={"operation_queue_id":"0","wait_on_operation_queues":[],"fusion_backend_config":{"kind":"__loop_fusion"}}
      ROOT next_state = (s32[], f32[16]) tuple(next_i, fusion.2)
    }

    ENTRY main {
      input = f32[16] parameter(0)
      zero = s32[] constant(0)
      init = (s32[], f32[16]) tuple(zero, input)
      loop = (s32[], f32[16]) while(init), condition=while_condition, body=while_body, backend_config={"known_trip_count":{"n":"4"}}
      ROOT result = f32[16] get-tuple-element(loop), index=1
    }
  )";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnUnverifiedModule(kHlo));

  // Serialize the module like `GpuExecutable::ToProto` does.
  HloModuleProtoWithConfig serialized_module = module->ToProtoWithConfig(
      HloProtoOptions{/*deduplicate_backend_config=*/true});
  const int unique_id = HloModule::GetNextUniqueModuleId();
  serialized_module.mutable_hlo_module()->set_pjrt_id(unique_id);

  // This is what XlaDebugInfoManager would get if loading built the module.
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> deserialized_module,
                       HloModule::CreateFromProtoWithConfig(serialized_module));
  ASSERT_EQ(deserialized_module->unique_id(), unique_id);
  const HloProto expected = MakeHloProto(*deserialized_module);

  GpuExecutable::Params params = MakeParamsWithModule(MakeConfig());
  params.module_name = module->name();
  params.hlo_module = std::move(serialized_module);
  // `Create` can't compute this without building the module. The module has no
  // collectives, so it would give true.
  params.collective_use_minimal_resource = true;
  params.enable_debug_info_manager = true;
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<GpuExecutable> executable,
                       GpuExecutable::Create(std::move(params)));
  {
    ScopedEnableAnnotations enable_annotations;
    executable->MaybePrepareProfiling();
  }

  ASSERT_OK_AND_ASSIGN(HloProto debug_info,
                       TraceAndDestroy(std::move(executable), unique_id));
  EXPECT_THAT(debug_info.hlo_module(), EqualsProto(expected.hlo_module()));
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
