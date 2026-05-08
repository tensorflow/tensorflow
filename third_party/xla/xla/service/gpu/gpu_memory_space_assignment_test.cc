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
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/analysis/alias_info.h"
#include "xla/hlo/analysis/hlo_alias_analysis.h"
#include "xla/hlo/analysis/hlo_ordering.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/verified_hlo_module.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu_topology.h"
#include "xla/service/hlo_module_config.h"
#include "xla/service/hlo_value.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla.pb.h"

namespace xla::gpu {
namespace {

using ::testing::Eq;
using ::testing::IsTrue;
using ::testing::NotNull;
using ::testing::SizeIs;

class GpuMemorySpaceAssignmentTest : public HloHardwareIndependentTestBase {
 public:
  GpuMemorySpaceAssignmentTest()
      : single_device_gpu_topology_(/*platform_version=*/"_",
                                    /*num_partitions=*/1,
                                    /*num_hosts_per_partition=*/1,
                                    /*num_devices_per_host=*/1),
        multi_host_gpu_topology_(/*platform_version=*/"_",
                                 /*num_partitions=*/1,
                                 /*num_hosts_per_partition=*/2,
                                 /*num_devices_per_host=*/1) {}

 public:
  const GpuTopology& multi_host_gpu_topology() const {
    return multi_host_gpu_topology_;
  }
  const GpuTopology& single_device_gpu_topology() const {
    return single_device_gpu_topology_;
  }

 private:
  GpuTopology single_device_gpu_topology_;
  GpuTopology multi_host_gpu_topology_;
};

TEST_F(GpuMemorySpaceAssignmentTest, TestDefaultColorAssignment) {
  absl::string_view kHloModule = R"(
    HloModule m

    ENTRY main {
      ROOT parameter0 = f32[] parameter(0)
    }
  )";

  HloModuleConfig config = GetModuleConfigForTest();
  BufferAssigner::Colorer colorer =
      CreateColorer(config.debug_options(), single_device_gpu_topology());

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
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

  HloModuleConfig config = GetModuleConfigForTest();
  DebugOptions debug_options = config.debug_options();
  debug_options.set_xla_gpu_enable_nccl_user_buffers(UseNcclUserBuffers());
  debug_options.set_xla_gpu_experimental_enable_nccl_symmetric_buffers(
      UseNcclSymmetricBuffers());
  config.set_debug_options(debug_options);
  BufferAssigner::Colorer colorer =
      CreateColorer(config.debug_options(), single_device_gpu_topology());

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
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

  HloModuleConfig config = GetModuleConfigForTest();
  DebugOptions debug_options = config.debug_options();
  debug_options.set_xla_gpu_experimental_enable_nvshmem(UseNvshmem());
  config.set_debug_options(debug_options);
  BufferAssigner::Colorer colorer =
      CreateColorer(config.debug_options(), single_device_gpu_topology());

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
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

  HloModuleConfig config = GetModuleConfigForTest();
  auto debug_options = config.debug_options();
  debug_options.set_xla_gpu_experimental_enable_nvshmem(true);
  config.set_debug_options(debug_options);
  BufferAssigner::Colorer colorer =
      CreateColorer(config.debug_options(), single_device_gpu_topology());

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
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

TEST_F(GpuMemorySpaceAssignmentTest, TestMultimemMosaicMemorySpaceAssignment) {
  constexpr absl::string_view kHloModule = R"(
    HloModule m

    ENTRY main {
      ROOT %custom-call.9 = (f16[8], f16[8]) custom-call(), custom_call_target="mosaic_gpu_v2", backend_config={"xla_multimem_parameters"}
    }
  )";

  HloModuleConfig config = GetModuleConfigForTest();
  BufferAssigner::Colorer colorer =
      CreateColorer(config.debug_options(), single_device_gpu_topology());

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(kHloModule, config));
  AliasInfo alias_info;
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloAliasAnalysis> alias_analysis,
                          HloAliasAnalysis::Run(module.get(), &alias_info));
  const DependencyHloOrdering ordering(module.get());
  TF_EXPECT_OK(colorer(alias_analysis.get(), ordering));

  const int kExpectedBuffersCount = 3;
  ASSERT_THAT(alias_analysis->buffers(), SizeIs(kExpectedBuffersCount));

  for (int i = 0; i < kExpectedBuffersCount; ++i) {
    ASSERT_THAT(alias_analysis->buffers()[i].values(), SizeIs(1));
    const HloValue* value = alias_analysis->buffers()[i].values()[0];
    ASSERT_THAT(value, NotNull());
    ASSERT_THAT(value->has_color(), IsTrue());
    const bool is_tuple = value->defining_position().shape().IsTuple();
    const int expected_color = (int)(is_tuple ? MemorySpaceColor::kDefault
                                              : MemorySpaceColor::kCollective);
    ASSERT_THAT(value->color(), Eq(expected_color));
  }
}

TEST(ParseIndexMemorySpacePairsTest, SinglePair) {
  TF_ASSERT_OK_AND_ASSIGN(auto pairs, ParseIndexMemorySpacePairs("{0:1}"));
  ASSERT_EQ(pairs.size(), 1);
  EXPECT_EQ(pairs[0].first, 0);
  EXPECT_EQ(pairs[0].second, MemorySpaceColor::kCollective);
}

TEST(ParseIndexMemorySpacePairsTest, MultiplePairs) {
  TF_ASSERT_OK_AND_ASSIGN(auto pairs, ParseIndexMemorySpacePairs("{0:1,2:2}"));
  ASSERT_EQ(pairs.size(), 2);
  EXPECT_EQ(pairs[0].first, 0);
  EXPECT_EQ(pairs[0].second, MemorySpaceColor::kCollective);
  EXPECT_EQ(pairs[1].first, 2);
  EXPECT_EQ(pairs[1].second, MemorySpaceColor::kTempBuffer);
}

TEST(ParseIndexMemorySpacePairsTest, EmptyBraces) {
  TF_ASSERT_OK_AND_ASSIGN(auto pairs, ParseIndexMemorySpacePairs("{}"));
  EXPECT_TRUE(pairs.empty());
}

TEST(ParseIndexMemorySpacePairsTest, WhitespaceHandling) {
  TF_ASSERT_OK_AND_ASSIGN(auto pairs,
                          ParseIndexMemorySpacePairs("{ 0 : 1 , 2 : 0 }"));
  ASSERT_EQ(pairs.size(), 2);
  EXPECT_EQ(pairs[0].first, 0);
  EXPECT_EQ(pairs[0].second, MemorySpaceColor::kCollective);
  EXPECT_EQ(pairs[1].first, 2);
  EXPECT_EQ(pairs[1].second, MemorySpaceColor::kDefault);
}

TEST(ParseIndexMemorySpacePairsTest, MissingBraces) {
  EXPECT_FALSE(ParseIndexMemorySpacePairs("0:1").ok());
}

TEST(ParseIndexMemorySpacePairsTest, InvalidPairFormat) {
  EXPECT_FALSE(ParseIndexMemorySpacePairs("{0}").ok());
}

TEST(ParseIndexMemorySpacePairsTest, NonIntegerValues) {
  EXPECT_FALSE(ParseIndexMemorySpacePairs("{a:b}").ok());
}

TEST(ParseIndexMemorySpacePairsTest, InvalidMemorySpace) {
  EXPECT_FALSE(ParseIndexMemorySpacePairs("{0:99}").ok());
}

// Helper to find the HloValue color for a given instruction name.
static int FindColorByName(const HloAliasAnalysis& alias_analysis,
                           absl::string_view name) {
  for (const auto& buffer : alias_analysis.buffers()) {
    for (const HloValue* value : buffer.values()) {
      if (value->instruction()->name() == name) {
        return value->color();
      }
    }
  }
  return -1;
}

TEST_F(GpuMemorySpaceAssignmentTest, CustomCallOperandMemorySpace) {
  absl::string_view kHloModule = R"(
    HloModule m

    ENTRY main {
      p0 = f32[1024]{0} parameter(0)
      p1 = f32[1024]{0} parameter(1)
      ROOT custom-call = f32[1024]{0} custom-call(p0, p1),
        custom_call_target="my_custom_call",
        frontend_attributes={operands_memory_spaces="{0:1}"}
    }
  )";

  HloModuleConfig config = GetModuleConfigForTest();
  BufferAssigner::Colorer colorer =
      CreateColorer(config.debug_options(), single_device_gpu_topology());

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(kHloModule, config));
  AliasInfo alias_info;
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloAliasAnalysis> alias_analysis,
                          HloAliasAnalysis::Run(module.get(), &alias_info));
  DependencyHloOrdering ordering(module.get());
  TF_EXPECT_OK(colorer(alias_analysis.get(), ordering));

  // Operand 0 (p0) should be colored with memory space 1.
  EXPECT_EQ(FindColorByName(*alias_analysis, "p0"), 1);
  // Operand 1 (p1) should remain in default memory space.
  EXPECT_EQ(FindColorByName(*alias_analysis, "p1"),
            (int)MemorySpaceColor::kDefault);
}

TEST_F(GpuMemorySpaceAssignmentTest, CustomCallResultMemorySpace) {
  absl::string_view kHloModule = R"(
    HloModule m

    ENTRY main {
      p0 = f32[1024]{0} parameter(0)
      ROOT custom-call = f32[1024]{0} custom-call(p0),
        custom_call_target="my_custom_call",
        frontend_attributes={results_memory_spaces="{0:1}"}
    }
  )";

  HloModuleConfig config = GetModuleConfigForTest();
  BufferAssigner::Colorer colorer =
      CreateColorer(config.debug_options(), single_device_gpu_topology());

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(kHloModule, config));
  AliasInfo alias_info;
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloAliasAnalysis> alias_analysis,
                          HloAliasAnalysis::Run(module.get(), &alias_info));
  DependencyHloOrdering ordering(module.get());
  TF_EXPECT_OK(colorer(alias_analysis.get(), ordering));

  // The custom call result should be colored with memory space 1.
  EXPECT_EQ(FindColorByName(*alias_analysis, "custom-call"), 1);
}

TEST_F(GpuMemorySpaceAssignmentTest, CustomCallTupleResultMemorySpace) {
  absl::string_view kHloModule = R"(
    HloModule m

    ENTRY main {
      p0 = f32[1024]{0} parameter(0)
      custom-call = (f32[1024]{0}, f32[512]{0}) custom-call(p0),
        custom_call_target="my_custom_call",
        frontend_attributes={results_memory_spaces="{0:1,1:1}"}
      ROOT gte = f32[1024]{0} get-tuple-element(custom-call), index=0
    }
  )";

  HloModuleConfig config = GetModuleConfigForTest();
  BufferAssigner::Colorer colorer =
      CreateColorer(config.debug_options(), single_device_gpu_topology());

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(kHloModule, config));
  AliasInfo alias_info;
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloAliasAnalysis> alias_analysis,
                          HloAliasAnalysis::Run(module.get(), &alias_info));
  DependencyHloOrdering ordering(module.get());
  TF_EXPECT_OK(colorer(alias_analysis.get(), ordering));

  // Find values defined by the custom call at tuple indices 0 and 1.
  for (const auto& buffer : alias_analysis->buffers()) {
    for (const HloValue* value : buffer.values()) {
      if (value->instruction()->name() != "custom-call") continue;
      const ShapeIndex& idx = value->defining_index();
      if (idx.size() == 1 && (idx[0] == 0 || idx[0] == 1)) {
        EXPECT_EQ(value->color(), 1)
            << "Tuple element " << idx[0] << " should have color 1";
      }
    }
  }
}

class GpuMosaicCollectiveMemorySpaceAssignmentTest
    : public GpuMemorySpaceAssignmentTest,
      public ::testing::WithParamInterface<bool> {
 public:
  bool IsMosaicWithCollectiveMetadata() const { return GetParam(); }
};

TEST_P(GpuMosaicCollectiveMemorySpaceAssignmentTest,
       MosaicCollectiveMemorySpaceAssignment) {
  const std::string kMosaicModule =
      absl::StrCat(R"(
    HloModule m
    ENTRY main {
      ROOT %custom-call.9 = (f16[8], f16[8]) custom-call(), custom_call_target="mosaic_gpu_v2", backend_config={uses_xla_collective_metadata=)",
                   IsMosaicWithCollectiveMetadata() ? "true" : "false", R"(}
    }
  )");

  HloModuleConfig config = GetModuleConfigForTest();
  DebugOptions debug_options = config.debug_options();
  config.set_debug_options(debug_options);

  BufferAssigner::Colorer colorer =
      CreateColorer(config.debug_options(), multi_host_gpu_topology());

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(kMosaicModule, config));
  AliasInfo alias_info;
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloAliasAnalysis> alias_analysis,
                          HloAliasAnalysis::Run(module.get(), &alias_info));
  DependencyHloOrdering ordering(module.get());
  TF_EXPECT_OK(colorer(alias_analysis.get(), ordering));

  EXPECT_EQ(alias_analysis->buffers().size(), 3);

  // First buffer is a tuple-shaped value, so it should not be colored.
  for (int i = 1; i < alias_analysis->buffers().size(); ++i) {
    EXPECT_EQ(alias_analysis->buffers()[i].values().size(), 1);
    EXPECT_EQ(alias_analysis->buffers()[i].values()[0]->has_color(), true);

    int expected_color = static_cast<int>(IsMosaicWithCollectiveMetadata()
                                              ? MemorySpaceColor::kCollective
                                              : MemorySpaceColor::kDefault);
    EXPECT_EQ(alias_analysis->buffers()[i].values()[0]->color(), expected_color)
        << "Buffer " << alias_analysis->buffers()[i].ToString()
        << " should have color " << expected_color;
  }
}

INSTANTIATE_TEST_SUITE_P(
    GpuMosaicCollectiveMemorySpaceAssignmentTestSuiteInstantiation,
    GpuMosaicCollectiveMemorySpaceAssignmentTest, ::testing::Bool(),
    [](const ::testing::TestParamInfo<
        GpuMosaicCollectiveMemorySpaceAssignmentTest::ParamType>& info) {
      return info.param ? "mosaic_uses_collective_metadata"
                        : "mosaic_does_not_use_collective_metadata";
    });

}  // namespace
}  // namespace xla::gpu
