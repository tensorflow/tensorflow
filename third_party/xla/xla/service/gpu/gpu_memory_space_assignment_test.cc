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

#include "xla/service/gpu/gpu_memory_space_assignment.h"

#include <memory>

#include <gtest/gtest.h>
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/analysis/alias_info.h"
#include "xla/hlo/analysis/hlo_alias_analysis.h"
#include "xla/hlo/analysis/hlo_ordering.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/service/hlo_module_config.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla.pb.h"

namespace xla::gpu {
namespace {

class GpuMemorySpaceAssignmentTest : public HloHardwareIndependentTestBase {};

TEST_F(GpuMemorySpaceAssignmentTest, TestDefaultColorAssignment) {
  absl::string_view kHloModule = R"(
    HloModule m

    ENTRY main {
      ROOT parameter0 = f32[] parameter(0)
    }
  )";

  HloModuleConfig config;
  auto colorer = CreateColorer(DebugOptions());

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kHloModule, config));
  AliasInfo alias_info;
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloAliasAnalysis> alias_analysis,
                          HloAliasAnalysis::Run(module.get(), &alias_info));
  DependencyHloOrdering ordering(module.get());
  TF_EXPECT_OK(colorer(alias_analysis.get(), ordering));

  EXPECT_EQ(alias_analysis->buffers().size(), 1);
  EXPECT_EQ(alias_analysis->buffers()[0].values().size(), 1);
  EXPECT_EQ(alias_analysis->buffers()[0].values()[0]->has_color(), true);
  EXPECT_EQ(alias_analysis->buffers()[0].values()[0]->color(),
            (int)MemorySpaceColor::kDefault);
}

struct CollectiveMemorySpaceAssignmentTestParams {
  bool use_nccl_user_buffers;
  bool use_nccl_symmetric_buffers;
};

class GpuCollectiveMemorySpaceAssignmentTest
    : public GpuMemorySpaceAssignmentTest,
      public ::testing::WithParamInterface<
          CollectiveMemorySpaceAssignmentTestParams> {
 public:
  bool UseNcclUserBuffers() const { return GetParam().use_nccl_user_buffers; }
  bool UseNcclSymmetricBuffers() const {
    return GetParam().use_nccl_symmetric_buffers;
  }
};

TEST_P(GpuCollectiveMemorySpaceAssignmentTest,
       TestCollectiveMemorySpaceAssignment) {
  absl::string_view kHloModule = R"(
    HloModule m, replica_count=2

    region_0.2 {
      Arg_0.3 = f32[] parameter(0)
      Arg_1.4 = f32[] parameter(1)
      ROOT add.5 = f32[] add(Arg_0.3, Arg_1.4)
    }

    ENTRY main {
      Arg_0.1 = f32[8]{0} parameter(0)
      ROOT all-reduce.6 = f32[8]{0} all-reduce(Arg_0.1), replica_groups={}, to_apply=region_0.2
    }
  )";

  HloModuleConfig config;
  DebugOptions debug_options;
  debug_options.set_xla_gpu_enable_nccl_user_buffers(UseNcclUserBuffers());
  debug_options.set_xla_gpu_experimental_enable_nccl_symmetric_buffers(
      UseNcclSymmetricBuffers());
  auto colorer = CreateColorer(debug_options);

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kHloModule, config));
  AliasInfo alias_info;
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloAliasAnalysis> alias_analysis,
                          HloAliasAnalysis::Run(module.get(), &alias_info));
  DependencyHloOrdering ordering(module.get());
  TF_EXPECT_OK(colorer(alias_analysis.get(), ordering));

  const int kExpectedBuffersCount = 5;
  const int kExpectedDefaultBuffersCount = 3;
  EXPECT_EQ(alias_analysis->buffers().size(), kExpectedBuffersCount);

  int expected_number_default_buffers =
      (UseNcclUserBuffers() || UseNcclSymmetricBuffers())
          ? kExpectedDefaultBuffersCount
          : kExpectedBuffersCount;

  // Temporary buffers.
  for (int i = 0; i < expected_number_default_buffers; ++i) {
    EXPECT_EQ(alias_analysis->buffers()[i].values().size(), 1);
    EXPECT_EQ(alias_analysis->buffers()[i].values()[0]->has_color(), true);
    EXPECT_EQ(alias_analysis->buffers()[i].values()[0]->color(),
              (int)MemorySpaceColor::kDefault);
  }

  for (int i = expected_number_default_buffers; i < kExpectedBuffersCount;
       ++i) {
    EXPECT_EQ(alias_analysis->buffers()[i].values().size(), 1);
    EXPECT_EQ(alias_analysis->buffers()[i].values()[0]->has_color(), true);
    EXPECT_EQ(alias_analysis->buffers()[i].values()[0]->color(),
              (int)MemorySpaceColor::kCollective);
  }
}

INSTANTIATE_TEST_SUITE_P(
    GpuCollectiveMemorySpaceAssignmentTestSuiteInstantiation,
    GpuCollectiveMemorySpaceAssignmentTest,
    ::testing::ValuesIn<CollectiveMemorySpaceAssignmentTestParams>(
        {{false, false}, {true, false}, {false, true}, {true, true}}),
    [](const ::testing::TestParamInfo<
        GpuCollectiveMemorySpaceAssignmentTest::ParamType>& info) {
      return absl::StrCat(info.param.use_nccl_user_buffers
                              ? "with_nccl_user_buffers"
                              : "without_nccl_user_buffers",
                          "_",
                          info.param.use_nccl_symmetric_buffers
                              ? "with_nccl_symmetric_buffers"
                              : "without_nccl_symmetric_buffers");
    });

struct MosaicMemorySpaceAssignmentTestParams {
  bool use_nvshmem;
  bool mosaic_contains_nvshmem;
};

class GpuMosaicMemorySpaceAssignmentTest
    : public GpuMemorySpaceAssignmentTest,
      public ::testing::WithParamInterface<
          MosaicMemorySpaceAssignmentTestParams> {
 public:
  bool UseNvshmem() const { return GetParam().use_nvshmem; }

  bool MosaicContainsNvshmem() const {
    return GetParam().mosaic_contains_nvshmem;
  }
};

TEST_P(GpuMosaicMemorySpaceAssignmentTest, TestMosaicMemorySpaceAssignment) {
  const absl::string_view kMosaicModule = R"(
    HloModule m

    ENTRY main {
      ROOT %custom-call.9 = (f16[8], f16[8]) custom-call(), custom_call_target="mosaic_gpu_v2"
    }
  )";

  const absl::string_view kMosaicNvshmemModule = R"(
    HloModule m

    ENTRY main {
      ROOT %custom-call.9 = (f16[8], f16[8]) custom-call(), custom_call_target="mosaic_gpu_v2", backend_config={module="nvshmem"}
    }
  )";

  const absl::string_view kHloModule =
      MosaicContainsNvshmem() ? kMosaicNvshmemModule : kMosaicModule;

  HloModuleConfig config;
  DebugOptions debug_options;
  debug_options.set_xla_gpu_experimental_enable_nvshmem(UseNvshmem());
  auto colorer = CreateColorer(debug_options);

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kHloModule, config));
  AliasInfo alias_info;
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloAliasAnalysis> alias_analysis,
                          HloAliasAnalysis::Run(module.get(), &alias_info));
  DependencyHloOrdering ordering(module.get());
  TF_EXPECT_OK(colorer(alias_analysis.get(), ordering));

  EXPECT_EQ(alias_analysis->buffers().size(), 3);

  const int kExpectedBuffersCount = 3;
  for (int i = 0; i < kExpectedBuffersCount; ++i) {
    EXPECT_EQ(alias_analysis->buffers()[i].values().size(), 1);
    if (MosaicContainsNvshmem()) {
      EXPECT_EQ(alias_analysis->buffers()[i].values()[0]->has_color(), true);
      EXPECT_EQ(alias_analysis->buffers()[i].values()[0]->color(),
                (int)(MosaicContainsNvshmem()
                          ? ((UseNvshmem() && !alias_analysis->buffers()[i]
                                                   .values()[0]
                                                   ->defining_position()
                                                   .shape()
                                                   .IsTuple())
                                 ? MemorySpaceColor::kCollective
                                 : MemorySpaceColor::kDefault)
                          : MemorySpaceColor::kDefault));
    }
  }
}

INSTANTIATE_TEST_SUITE_P(
    GpuMosaicMemorySpaceAssignmentTestSuiteInstantiation,
    GpuMosaicMemorySpaceAssignmentTest,
    ::testing::ValuesIn<MosaicMemorySpaceAssignmentTestParams>(
        {{false, false}, {true, false}, {false, true}, {true, true}}),
    [](const ::testing::TestParamInfo<
        GpuMosaicMemorySpaceAssignmentTest::ParamType>& info) {
      return absl::StrCat(
          info.param.use_nvshmem ? "with_nvshmem" : "without_nvshmem", "_",
          info.param.mosaic_contains_nvshmem ? "contains_nvshmem"
                                             : "does_not_contain_nvshmem");
    });

TEST_F(GpuMemorySpaceAssignmentTest, TestNvshmemMemorySpaceAssignment) {
  absl::string_view kHloModule = R"(
    HloModule m

    apply_op {
      x = f32[] parameter(0)
      y = f32[] parameter(1)
      ROOT apply_op = f32[] add(x, y)
    }

    ENTRY main {
      parameter0 = f32[] parameter(0)
      all-reduce = f32[] all-reduce-start(parameter0), to_apply=apply_op, backend_config={"collective_backend_config":{"backend":"NVSHMEM"}}
      ROOT all-reduce-done = f32[] all-reduce-done(all-reduce)
    }
  )";

  HloModuleConfig config;
  auto debug_options = DebugOptions();
  debug_options.set_xla_gpu_experimental_enable_nvshmem(true);
  auto colorer = CreateColorer(debug_options);

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kHloModule, config));
  AliasInfo alias_info;
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloAliasAnalysis> alias_analysis,
                          HloAliasAnalysis::Run(module.get(), &alias_info));
  DependencyHloOrdering ordering(module.get());
  TF_EXPECT_OK(colorer(alias_analysis.get(), ordering));

  const int kExpectedBuffersCount = 5;
  EXPECT_EQ(alias_analysis->buffers().size(), kExpectedBuffersCount);

  const int kExpectedDefaultBuffersCount = 3;

  for (int i = 0; i < kExpectedDefaultBuffersCount; ++i) {
    EXPECT_EQ(alias_analysis->buffers()[i].values().size(), 1);
    EXPECT_EQ(alias_analysis->buffers()[i].values()[0]->has_color(), true);
    EXPECT_EQ(alias_analysis->buffers()[i].values()[0]->color(),
              (int)MemorySpaceColor::kDefault);
  }

  for (int i = kExpectedDefaultBuffersCount; i < kExpectedBuffersCount; ++i) {
    EXPECT_EQ(alias_analysis->buffers()[i].values().size(), 1);
    EXPECT_EQ(alias_analysis->buffers()[i].values()[0]->has_color(), true);
    EXPECT_EQ(alias_analysis->buffers()[i].values()[0]->color(),
              (int)MemorySpaceColor::kCollective);
  }
}
}  // namespace
}  // namespace xla::gpu
