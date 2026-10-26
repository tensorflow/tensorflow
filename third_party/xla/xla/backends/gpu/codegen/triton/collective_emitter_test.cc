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

#include "xla/backends/gpu/codegen/triton/collective_emitter.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/check.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Module.h"
#include "llvm/TargetParser/Triple.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/backends/gpu/codegen/cubin_custom_kernel_compiler.h"
#include "xla/backends/gpu/codegen/emitters/mlir_kernel_emitter.h"
#include "xla/backends/gpu/codegen/fusion_emitter.h"
#include "xla/backends/gpu/codegen/fusions.h"
#include "xla/backends/gpu/codegen/kernel_compiler.h"
#include "xla/backends/gpu/codegen/triton/fusion.h"
#include "xla/backends/gpu/runtime/collective_params.h"
#include "xla/backends/gpu/target_config/target_config.h"
#include "xla/backends/gpu/transforms/collectives/collective_fusion.h"
#include "xla/backends/gpu/transforms/collectives/collective_kernel_strategy_annotator.h"
#include "xla/codegen/xtile/xtile_config.pb.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/utils/hlo_query.h"
#include "xla/primitive_util.h"
#include "xla/runtime/object_pool.h"
#include "xla/service/gpu/gpu_device_info_for_tests.h"
#include "xla/service/gpu/hlo_fusion_analysis.h"
#include "xla/service/gpu/mlir_context_pool.h"
#include "xla/service/gpu_topology.h"
#include "xla/service/hlo_creation_utils.h"
#include "xla/service/hlo_verifier.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/device_description.pb.h"
#include "xla/tsl/util/proto/proto_matchers.h"
#include "xla/util.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;
using ::testing::AllOf;
using ::testing::ElementsAre;
using ::tsl::proto_testing::EqualsProto;
using ::xla::xtile::BlockLevelFusionConfig;

MATCHER_P(HasShape, expected_shape, "") {
  return arg != nullptr && arg->shape() == expected_shape;
}
MATCHER_P(HasOpcode, expected_opcode, "") {
  return arg != nullptr && arg->opcode() == expected_opcode;
}
MATCHER_P(HasNumOperands, expected_operands, "") {
  return arg != nullptr && arg->operand_count() == expected_operands;
}

struct ModuleWithFusion {
  std::unique_ptr<HloModule> module;

  const HloFusionInstruction* FusionInstr() const {
    for (const HloComputation* comp : module->computations()) {
      if (const HloInstruction* fusion =
              hlo_query::GetFirstInstructionWithOpcode(*comp,
                                                       HloOpcode::kFusion)) {
        return Cast<HloFusionInstruction>(fusion);
      }
    }
    return nullptr;
  }
  HloFusionInstruction* MutableFusionInstr() {
    for (HloComputation* comp : module->computations()) {
      if (HloInstruction* fusion = hlo_query::GetFirstInstructionWithOpcode(
              *comp, HloOpcode::kFusion)) {
        return Cast<HloFusionInstruction>(fusion);
      }
    }
    return nullptr;
  }
};

struct ModuleWithEmitter : public ModuleWithFusion {
  std::optional<HloFusionAnalysis> analysis;
  std::unique_ptr<TritonFusion> emitter;
  llvm::LLVMContext llvm_context;
  llvm::Triple target_triple;
  std::string data_layout;

  explicit ModuleWithEmitter(std::unique_ptr<HloModule> module_arg)
      : ModuleWithFusion{std::move(module_arg)},
        target_triple(""),
        data_layout("") {}
};

class CollectiveBlockLevelConfigTest : public HloHardwareIndependentTestBase {
 public:
  CollectiveBlockLevelConfigTest()
      : device_info_{TestGpuDeviceInfo::H100SXMDeviceInfo()} {}

 protected:
  void SetUp() override {
    HloHardwareIndependentTestBase::SetUp();
    ASSERT_OK_AND_ASSIGN(GpuTopology topology,
                         GetGpuTopologyForPlatform("nvidia_h100", 1, 1, 8));
    gpu_topology_ = std::make_unique<GpuTopology>(std::move(topology));
  }

  absl::StatusOr<ModuleWithFusion> BuildModuleWithFusion(
      absl::string_view module_str, const GpuTopology& topology,
      int64_t replica_count = 2) const {
    ABSL_ASSIGN_OR_RETURN(std::unique_ptr<HloModule> module,
                     ParseAndReturnVerifiedModule(module_str, replica_count,
                                                  /*num_partitions=*/1));
    module->mutable_config()
        .mutable_debug_options()
        .add_xla_gpu_experimental_use_collective_kernels(
            xla::DebugOptions::COLLECTIVE_KERNEL_ALL_GATHER);
    module->mutable_config()
        .mutable_debug_options()
        .add_xla_gpu_experimental_use_collective_kernels(
            xla::DebugOptions::COLLECTIVE_KERNEL_ALL_TO_ALL);
    module->mutable_config()
        .mutable_debug_options()
        .add_xla_gpu_experimental_use_collective_kernels(
            xla::DebugOptions::COLLECTIVE_KERNEL_REDUCE_SCATTER);
    CollectiveKernelStrategyAnnotator annotator(topology,
                                                /*is_multimem_enabled=*/false);
    ABSL_RETURN_IF_ERROR(annotator.Run(module.get()).status());
    CollectiveFusion fusion_pass(topology);
    ABSL_ASSIGN_OR_RETURN(bool fused, fusion_pass.Run(module.get()));
    TF_RET_CHECK(fused) << "CollectiveFusion did not fuse any instruction";
    return ModuleWithFusion{std::move(module)};
  }

  absl::StatusOr<ModuleWithFusion> BuildModuleWithFusion(
      absl::string_view module_str, int64_t replica_count = 2) const {
    return BuildModuleWithFusion(module_str, *gpu_topology_, replica_count);
  }

 protected:
  static std::string GetModuleStr(const Shape& shape,
                                  absl::string_view replica_groups,
                                  HloOpcode opcode) {
    absl::string_view type_str =
        xla::primitive_util::LowercasePrimitiveTypeName(shape.element_type());
    absl::string_view reduction_kind = "add";
    if (shape.element_type() == PRED) {
      reduction_kind = "or";
    }
    if (opcode == HloOpcode::kAllToAll) {
      return absl::StrFormat(R"(
        HloModule test
        ENTRY test_computation {
          param_0 = %1$s parameter(0)
          ROOT all-to-all = %1$s all-to-all(param_0), replica_groups={%2$s}, dimensions={0}
        }
      )",
                             shape.ToString(), replica_groups);
    }
    if (opcode == HloOpcode::kReduceScatter) {
      Shape output_shape = shape;
      output_shape.set_dimensions(0, shape.dimensions(0) / 2);
      return absl::StrFormat(R"(
        HloModule test
        apply_op {
          x = %4$s[] parameter(0)
          y = %4$s[] parameter(1)
          ROOT apply_op = %4$s[] %5$s(x, y)
        }

        ENTRY test_computation {
          param_0 = %1$s parameter(0)
          ROOT reduce-scatter = %2$s reduce-scatter(param_0), to_apply=apply_op, replica_groups={%3$s}, dimensions={0}
        }
      )",
                             shape.ToString(), output_shape.ToString(),
                             replica_groups, type_str, reduction_kind);
    }
    return absl::StrFormat(R"(
      HloModule test
      apply_op {
        x = %3$s[] parameter(0)
        y = %3$s[] parameter(1)
        ROOT apply_op = %3$s[] %4$s(x, y)
      }

      async_computation {
        param_0 = %1$s parameter(0)
        ROOT all-reduce = %1$s all-reduce(param_0), to_apply=apply_op, replica_groups={%2$s}
      }

      ENTRY test_computation {
        param_0 = %1$s parameter(0)
        async-start = ((%1$s), %1$s) async-start(param_0), calls=async_computation
        ROOT async-done = %1$s async-done(async-start)
      }
    )",
                           shape.ToString(), replica_groups, type_str,
                           reduction_kind);
  }

  const se::DeviceDescription device_info_;
  std::unique_ptr<GpuTopology> gpu_topology_;
};

class CollectiveEmitterTest : public CollectiveBlockLevelConfigTest {
 public:
  absl::StatusOr<std::unique_ptr<ModuleWithEmitter>> BuildModuleWithEmitter(
      std::string module_str, const GpuTopology& gpu_topology) const {
    ABSL_ASSIGN_OR_RETURN(
        ModuleWithFusion module_with_fusion,
        BuildModuleWithFusion(std::move(module_str), gpu_topology));
    auto result = std::make_unique<ModuleWithEmitter>(
        std::move(module_with_fusion.module));
    const se::DeviceDescription& device_info =
        gpu_topology.gpu_target_config().device_description;
    result->analysis =
        HloFusionAnalysis::Create(*result->FusionInstr(), device_info);
    std::unique_ptr<FusionInterface> fusion_emitter =
        GetFusionEmitter(PreBufferAssignmentFusionInfo{*result->analysis});
    TritonFusion* triton_emitter =
        dynamic_cast<TritonFusion*>(fusion_emitter.get());
    TF_RET_CHECK(triton_emitter != nullptr);
    fusion_emitter.release();
    result->emitter = absl::WrapUnique(triton_emitter);
    return result;
  }
};

struct AllReduceBlockLevelConfigTestCase {
  std::string test_name;
  Shape shape;
  std::string expected_proto;

  // Teach gTest how to print the test case.
  [[maybe_unused]] friend void PrintTo(
      const AllReduceBlockLevelConfigTestCase& test_case, std::ostream* os) {
    *os << "{test_name: " << test_case.test_name
        << " shape: " << test_case.shape.ToString()
        << " expected_proto: " << test_case.expected_proto << "}";
  }
};

class CollectiveBlockLevelConfigParameterizedTest
    : public CollectiveBlockLevelConfigTest,
      public ::testing::WithParamInterface<AllReduceBlockLevelConfigTestCase> {
};

TEST_P(CollectiveBlockLevelConfigParameterizedTest, AllReduceBlockLevelConfig) {
  const auto& param = GetParam();
  ASSERT_OK_AND_ASSIGN(const auto module_with_fusion,
                       BuildModuleWithFusion(GetModuleStr(
                           param.shape, "{0,1}", HloOpcode::kAllReduce)));
  ASSERT_OK_AND_ASSIGN(BlockLevelFusionConfig block_level_config,
                       GetCollectiveBlockLevelFusionConfig(
                           *gpu_topology_, module_with_fusion.FusionInstr()));
  EXPECT_THAT(block_level_config, EqualsProto(param.expected_proto));
}

INSTANTIATE_TEST_SUITE_P(
    CollectiveEmitterParameterizedTestInstantiation,
    CollectiveBlockLevelConfigParameterizedTest,
    ::testing::ValuesIn<AllReduceBlockLevelConfigTestCase>(
        {{/* .test_name = */ "F32_65536",
          /* .shape = */ ShapeUtil::MakeShape(F32, {65536}),
          /* .expected_proto = */ R"pb(
            num_warps: 16
            num_ctas: 1
            num_stages: 1
            output_tiles { sizes: 2048 }
          )pb"},
         {/* .test_name= */ "F32_200_100",
          /* .shape= */ ShapeUtil::MakeShape(F32, {200, 100}),
          /* .expected_proto= */ R"pb(
            num_warps: 16
            num_ctas: 1
            num_stages: 1
            output_tiles { sizes: 32 sizes: 128 }
          )pb"},
         {/* .test_name= */ "F32_1040",
          /* .shape= */ ShapeUtil::MakeShape(F32, {1040}),
          /* .expected_proto= */ R"pb(
            num_warps: 16
            num_ctas: 1
            num_stages: 1
            output_tiles { sizes: 2048 }
          )pb"}}),
    [](const ::testing::TestParamInfo<
        CollectiveBlockLevelConfigParameterizedTest::ParamType>& info) {
      return info.param.test_name;
    });

TEST_F(CollectiveEmitterTest, AllReduceBlockLevelConfigNoReplicaGroups) {
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> module,
      ParseAndReturnVerifiedModule(
          GetModuleStr(ShapeUtil::MakeShape(F32, {65536}),
                       /*replica_groups=*/"", HloOpcode::kAllReduce),
          /*replica_count=*/2, /*num_partitions=*/1));
  const HloInstruction* all_reduce = hlo_query::GetFirstInstructionWithOpcode(
      *module->GetComputationWithName("async_computation"),
      HloOpcode::kAllReduce);
  ASSERT_NE(all_reduce, nullptr);
  std::unique_ptr<HloModule> fused_module =
      NewModuleWithFusion(all_reduce, HloInstruction::FusionKind::kLoop);
  const auto* fusion_instr = Cast<HloFusionInstruction>(
      fused_module->entry_computation()->root_instruction());
  absl::StatusOr<BlockLevelFusionConfig> block_level_config =
      GetCollectiveBlockLevelFusionConfig(*gpu_topology_, fusion_instr);
  EXPECT_THAT(block_level_config, StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST_F(CollectiveEmitterTest, AllReduceGetCollectiveUnmanagedKernelArguments) {
  ASSERT_OK_AND_ASSIGN(
      const auto module_with_fusion,
      BuildModuleWithFusion(GetModuleStr(ShapeUtil::MakeShape(F32, {65536}),
                                         "{0,1}", HloOpcode::kAllReduce)));
  ASSERT_OK_AND_ASSIGN(
      const auto unmanaged_arguments,
      GetCollectiveUnmanagedKernelArguments(module_with_fusion.FusionInstr()));
  ASSERT_EQ(unmanaged_arguments.size(), 4);
  EXPECT_EQ(unmanaged_arguments[0].dimensions().size(), 0);
  EXPECT_EQ(unmanaged_arguments[1].dimensions().size(), 0);
  // num_devices x input_shape
  ASSERT_EQ(unmanaged_arguments[2].dimensions().size(), 2);
  EXPECT_EQ(unmanaged_arguments[2].dimensions()[0], 2);  // num_devices

  ASSERT_EQ(unmanaged_arguments[3].dimensions().size(), 2);
  EXPECT_EQ(unmanaged_arguments[3].dimensions()[0], 2);      // num_devices
  EXPECT_EQ(unmanaged_arguments[3].dimensions()[1], 65536);  // input_shape[0]
}

TEST_F(CollectiveEmitterTest, AllReduceWithTritonGetLaunchConfig) {
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<ModuleWithEmitter> result_ptr,
      BuildModuleWithEmitter(GetModuleStr(ShapeUtil::MakeShape(F32, {65536}),
                                          "{0,1}", HloOpcode::kAllReduce),
                             *gpu_topology_));
  auto& result = *result_ptr;
  const TritonFusion* triton_fusion = result.emitter.get();
  ASSERT_NE(triton_fusion, nullptr);
  auto const launch_config = TritonFusion::GetLaunchConfig(&*result.analysis);
  ASSERT_NE(launch_config, std::nullopt);
  EXPECT_EQ(launch_config->launch_dimensions.num_blocks(), 32);
  EXPECT_EQ(launch_config->launch_dimensions.num_threads_per_block(), 512);
}

struct CollectiveSanityTestCase {
  Shape shape;
  HloOpcode opcode;
};

class CollectiveEmitterParameterizedTest
    : public CollectiveEmitterTest,
      public ::testing::WithParamInterface<CollectiveSanityTestCase> {};

TEST_P(CollectiveEmitterParameterizedTest,
       AllReduceWithTritonGenerateTritonKernelSanity) {
  const auto& [shape, opcode] = GetParam();
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<ModuleWithEmitter> result,
      BuildModuleWithEmitter(GetModuleStr(shape, "{0,1}", opcode),
                             *gpu_topology_));
  const TritonFusion* triton_fusion = result->emitter.get();
  ASSERT_NE(triton_fusion, nullptr);

  auto llvm_compiler =
      [&](llvm::Module& llvm_module, const se::DeviceDescription& descr,
          const DebugOptions& opts) -> absl::StatusOr<std::vector<uint8_t>> {
    return std::vector<uint8_t>{1};
  };
  DebugOptions debug_options;
  CubinCustomKernelCompiler kernel_compiler(llvm_compiler, device_info_,
                                            debug_options, *gpu_topology_);

  MlirContextPool mlir_context_pool([]() { return CreateMlirContext(); });
  ASSERT_OK_AND_ASSIGN(BorrowedMlirContext borrowed_context,
                       mlir_context_pool.GetOrCreate());

  ASSERT_OK_AND_ASSIGN(  // Check that we can generate the kernel
      TritonWrapperResult triton_kernel,
      triton_fusion
          ->GenerateTritonKernelAndWrapper(
              *result->FusionInstr(), "test-collective", device_info_,
              result->target_triple, result->data_layout,
              std::move(borrowed_context), &kernel_compiler)
          .Await());
}

INSTANTIATE_TEST_SUITE_P(
    CollectiveEmitterParameterizedTestInstantiation,
    CollectiveEmitterParameterizedTest,
    ::testing::ValuesIn<CollectiveSanityTestCase>(
        {{ShapeUtil::MakeShape(F32, {65536}), HloOpcode::kAllReduce},
         {ShapeUtil::MakeShape(BF16, {200, 100}), HloOpcode::kAllReduce},
         {ShapeUtil::MakeShape(PRED, {200, 64}), HloOpcode::kAllReduce},
         {ShapeUtil::MakeShape(F32, {131072}), HloOpcode::kAllReduce},
         {ShapeUtil::MakeShape(F32, {65536}), HloOpcode::kReduceScatter},
         {ShapeUtil::MakeShape(BF16, {200, 100}), HloOpcode::kReduceScatter},
         {ShapeUtil::MakeShape(PRED, {200, 64}), HloOpcode::kReduceScatter},
         {ShapeUtil::MakeShape(F32, {131072}), HloOpcode::kReduceScatter},
         {ShapeUtil::MakeShape(F32, {65536}), HloOpcode::kAllToAll},
         {ShapeUtil::MakeShape(BF16, {256, 100}), HloOpcode::kAllToAll},
         {ShapeUtil::MakeShape(PRED, {256, 64}), HloOpcode::kAllToAll},
         {ShapeUtil::MakeShape(F32, {131072}), HloOpcode::kAllToAll},
         {ShapeUtil::MakeShape(F32, {1048576}), HloOpcode::kAllToAll}}),
    [](const ::testing::TestParamInfo<
        CollectiveEmitterParameterizedTest::ParamType>& info) {
      std::string op_prefix = "ar_";
      if (info.param.opcode == HloOpcode::kReduceScatter) {
        op_prefix = "rs_";
      } else if (info.param.opcode == HloOpcode::kAllToAll) {
        op_prefix = "a2a_";
      }
      return op_prefix +
             primitive_util::LowercasePrimitiveTypeName(
                 info.param.shape.element_type()) +
             "__" + absl::StrJoin(info.param.shape.dimensions(), "_");
    });

struct GreedyPowerOfTwoTilesTestCase {
  std::string test_name;
  Shape output_shape;
  int32_t num_blocks{0};
  int64_t expected_used_blocks{0};
  llvm::SmallVector<int64_t> expected_tile_sizes;

  [[maybe_unused]] friend void PrintTo(
      const GreedyPowerOfTwoTilesTestCase& test_case, std::ostream* os) {
    *os << "{test_name: " << test_case.test_name
        << " output_shape: " << test_case.output_shape.ToString()
        << " num_blocks: " << test_case.num_blocks
        << " expected_used_blocks: " << test_case.expected_used_blocks << "}";
  }
};

class GreedyPowerOfTwoTilesTest
    : public ::testing::TestWithParam<GreedyPowerOfTwoTilesTestCase> {};

TEST_P(GreedyPowerOfTwoTilesTest, CorrectTilesGenerated) {
  const auto& [test_name, output_shape, num_blocks, expected_used_blocks,
               expected_tile_sizes] = GetParam();

  const auto get_blocks_used = [](const Shape& shape,
                                  llvm::ArrayRef<const int64_t> tile_sizes) {
    int64_t blocks_used = 1;
    for (int32_t i = 0; i < tile_sizes.size(); ++i) {
      blocks_used *= xla::CeilOfRatio(shape.dimensions(i), tile_sizes[i]);
    }
    return blocks_used;
  };
  llvm::SmallVector<int64_t> tile_sizes =
      GreedyPowerOfTwoTiles(output_shape, num_blocks);
  EXPECT_THAT(tile_sizes, testing::ElementsAreArray(expected_tile_sizes));
  EXPECT_EQ(get_blocks_used(output_shape, tile_sizes), expected_used_blocks);
}

INSTANTIATE_TEST_SUITE_P(
    GreedyPowerOfTwoTilesTestSuite, GreedyPowerOfTwoTilesTest,
    ::testing::ValuesIn<GreedyPowerOfTwoTilesTestCase>(
        {{
             /* .test_name = */ "2D_Pow2",
             /* .output_shape = */ ShapeUtil::MakeShape(F32, {1024, 1024}),
             /* .num_blocks = */ 16,
             /* .expected_used_blocks = */ 16,
             /* .expected_tile_sizes = */ {64, 1024},
         },
         {
             /* .test_name = */ "4D_MixedLayout",
             /* .output_shape = */
             ShapeUtil::MakeShapeWithDenseLayout(BF16, {16, 3072, 2, 8},
                                                 {1, 3, 0, 2}),
             /* .num_blocks = */ 32,
             /* .expected_used_blocks = */ 32,
             /* .expected_tile_sizes = */ {1, 4096, 1, 8},
         },
         {
             /* .test_name = */ "3D_PartialBlocks",
             /* .output_shape = */
             ShapeUtil::MakeShapeWithDescendingLayout(BF16, {2, 5, 1024}),
             /* .num_blocks = */ 32,
             /* .expected_used_blocks = */ 24,
             /* .expected_tile_sizes = */ {1, 2, 256},
         }}),
    [](const ::testing::TestParamInfo<GreedyPowerOfTwoTilesTest::ParamType>&
           info) { return info.param.test_name; });

TEST_F(CollectiveBlockLevelConfigTest, AllGatherBlockLevelConfig) {
  constexpr absl::string_view kAllGatherHloStr = R"(
    HloModule test
    ENTRY test_computation {
      param_0 = f32[32768] parameter(0)
      ROOT all-gather = f32[65536] all-gather(param_0), replica_groups={{0,1}},
        dimensions={0}
    }
  )";
  ASSERT_OK_AND_ASSIGN(ModuleWithFusion module_with_fusion,
                       BuildModuleWithFusion(kAllGatherHloStr));
  ASSERT_OK_AND_ASSIGN(BlockLevelFusionConfig block_level_config,
                       GetCollectiveBlockLevelFusionConfig(
                           *gpu_topology_, module_with_fusion.FusionInstr()));

  EXPECT_EQ(block_level_config.num_warps(), 16);
  EXPECT_EQ(block_level_config.num_ctas(), 1);
  EXPECT_EQ(block_level_config.num_stages(), 1);
  EXPECT_THAT(block_level_config.output_tiles(0).sizes(), ElementsAre(2048));
}

TEST_F(CollectiveEmitterTest, AllGatherGetCollectiveUnmanagedKernelArguments) {
  constexpr absl::string_view kAllGatherHloStr = R"(
    HloModule test
    ENTRY test_computation {
      param_0 = f32[32768] parameter(0)
      ROOT all-gather = f32[65536] all-gather(param_0), replica_groups={{0,1}},
        dimensions={0}
    }
  )";
  ASSERT_OK_AND_ASSIGN(ModuleWithFusion module_with_fusion,
                       BuildModuleWithFusion(kAllGatherHloStr));
  ASSERT_OK_AND_ASSIGN(
      const auto unmanaged_arguments,
      GetCollectiveUnmanagedKernelArguments(module_with_fusion.FusionInstr()));
  ASSERT_EQ(unmanaged_arguments.size(), 4);
  // [0]: rank (S32[])
  EXPECT_EQ(unmanaged_arguments[0].dimensions().size(), 0);
  // [1]: signal_value (S32[])
  EXPECT_EQ(unmanaged_arguments[1].dimensions().size(), 0);
  // [2]: signal_buffers (S32[num_devices, kMaxBlocksPerGrid])
  ASSERT_EQ(unmanaged_arguments[2].dimensions().size(), 2);
  EXPECT_EQ(unmanaged_arguments[2].dimensions()[0], 2);
  // [3]: remote buffers of param_0 (F32[num_devices, 32768])
  EXPECT_THAT(unmanaged_arguments[3].dimensions(), ElementsAre(2, 32768));
}

TEST_F(CollectiveBlockLevelConfigTest,
       AllGatherBlockLevelConfigClampsGatherDimToPerRankSize) {
  constexpr absl::string_view kAllGatherHloStr = R"(
    HloModule test
    ENTRY test_computation {
      param_0 = f32[1,64] parameter(0)
      ROOT all-gather = f32[2,64] all-gather(param_0), replica_groups={{0,1}},
        dimensions={0}
    }
  )";
  ASSERT_OK_AND_ASSIGN(ModuleWithFusion module_with_fusion,
                       BuildModuleWithFusion(kAllGatherHloStr));
  ASSERT_OK_AND_ASSIGN(const BlockLevelFusionConfig block_level_config,
                       GetCollectiveBlockLevelFusionConfig(
                           *gpu_topology_, module_with_fusion.FusionInstr()));
  ASSERT_EQ(block_level_config.output_tiles_size(), 1);
  EXPECT_THAT(block_level_config.output_tiles(0).sizes(), ElementsAre(1, 64));
}

TEST_F(CollectiveBlockLevelConfigTest, AllGatherBlockLevelConfigAtMaxBlocks) {
  constexpr absl::string_view kAllGatherHloStr = R"(
    HloModule test
    ENTRY test_computation {
      param_0 = f32[65536] parameter(0)
      ROOT all-gather = f32[131072] all-gather(param_0), replica_groups={{0,1}},
        dimensions={0}
    }
  )";
  ASSERT_OK_AND_ASSIGN(ModuleWithFusion module_with_fusion,
                       BuildModuleWithFusion(kAllGatherHloStr));
  ASSERT_OK_AND_ASSIGN(const BlockLevelFusionConfig block_level_config,
                       GetCollectiveBlockLevelFusionConfig(
                           *gpu_topology_, module_with_fusion.FusionInstr()));
  // 131072 elements / kAllGatherMaxBlocksPerGrid (64) blocks.
  EXPECT_THAT(block_level_config.output_tiles(0).sizes(), ElementsAre(2048));
}

TEST_F(CollectiveBlockLevelConfigTest, AllToAllBlockLevelConfig) {
  constexpr absl::string_view kAllToAllHloStr = R"(
    HloModule test
    ENTRY test_computation {
      param_0 = f32[65536] parameter(0)
      ROOT all-to-all = f32[65536] all-to-all(param_0), replica_groups={{0,1}},
        dimensions={0}
    }
  )";
  ASSERT_OK_AND_ASSIGN(ModuleWithFusion module_with_fusion,
                       BuildModuleWithFusion(kAllToAllHloStr));
  ASSERT_OK_AND_ASSIGN(BlockLevelFusionConfig block_level_config,
                       GetCollectiveBlockLevelFusionConfig(
                           *gpu_topology_, module_with_fusion.FusionInstr()));

  EXPECT_EQ(block_level_config.num_warps(), 16);
  EXPECT_EQ(block_level_config.num_ctas(), 1);
  EXPECT_EQ(block_level_config.num_stages(), 1);
  EXPECT_THAT(block_level_config.output_tiles(0).sizes(), ElementsAre(2048));
}

TEST_F(CollectiveEmitterTest, AllToAllGetCollectiveUnmanagedKernelArguments) {
  constexpr absl::string_view kAllToAllHloStr = R"(
    HloModule test
    ENTRY test_computation {
      param_0 = f32[65536] parameter(0)
      ROOT all-to-all = f32[65536] all-to-all(param_0), replica_groups={{0,1}},
        dimensions={0}
    }
  )";
  ASSERT_OK_AND_ASSIGN(ModuleWithFusion module_with_fusion,
                       BuildModuleWithFusion(kAllToAllHloStr));
  ASSERT_OK_AND_ASSIGN(
      const auto unmanaged_arguments,
      GetCollectiveUnmanagedKernelArguments(module_with_fusion.FusionInstr()));
  ASSERT_EQ(unmanaged_arguments.size(), 4);
  // [0]: rank (S32[])
  EXPECT_EQ(unmanaged_arguments[0].dimensions().size(), 0);
  // [1]: signal_value (S32[])
  EXPECT_EQ(unmanaged_arguments[1].dimensions().size(), 0);
  // [2]: signal_buffers (S32[num_devices, kMaxBlocksPerGrid])
  ASSERT_EQ(unmanaged_arguments[2].dimensions().size(), 2);
  EXPECT_EQ(unmanaged_arguments[2].dimensions()[0], 2);
  // [3]: remote buffers of param_0 (F32[num_devices, 65536])
  EXPECT_THAT(unmanaged_arguments[3].dimensions(), ElementsAre(2, 65536));
}

TEST_F(CollectiveBlockLevelConfigTest,
       AllToAllBlockLevelConfigClampsSplitDimToPerRankSize) {
  constexpr absl::string_view kAllToAllHloStr = R"(
    HloModule test
    ENTRY test_computation {
      param_0 = f32[2,64] parameter(0)
      ROOT all-to-all = f32[2,64] all-to-all(param_0), replica_groups={{0,1}},
        dimensions={0}
    }
  )";
  ASSERT_OK_AND_ASSIGN(ModuleWithFusion module_with_fusion,
                       BuildModuleWithFusion(kAllToAllHloStr));
  ASSERT_OK_AND_ASSIGN(const BlockLevelFusionConfig block_level_config,
                       GetCollectiveBlockLevelFusionConfig(
                           *gpu_topology_, module_with_fusion.FusionInstr()));
  ASSERT_EQ(block_level_config.output_tiles_size(), 1);
  EXPECT_EQ(block_level_config.num_warps(), 1);
  EXPECT_THAT(block_level_config.output_tiles(0).sizes(), ElementsAre(1, 64));
}

TEST_F(CollectiveEmitterTest, FlattenReduceScatterFusion) {
  Shape input_shape = ShapeUtil::MakeShape(F32, {2, 4, 8});
  Shape output_shape = ShapeUtil::MakeShape(F32, {1, 4, 8});
  Shape flat_input_shape = ShapeUtil::MakeShape(F32, {4, 16});
  Shape flat_output_shape = ShapeUtil::MakeShape(F32, {2, 16});
  ASSERT_OK_AND_ASSIGN(ModuleWithFusion module_with_fusion,
                       BuildModuleWithFusion(GetModuleStr(
                           input_shape, "{0,1}", HloOpcode::kReduceScatter)));

  HloComputation* entry_computation =
      module_with_fusion.module->entry_computation();
  HloInstruction* root = entry_computation->root_instruction();
  EXPECT_THAT(root, AllOf(HasOpcode(HloOpcode::kBitcast),
                          HasShape(output_shape), HasNumOperands(1)));

  HloInstruction* new_fusion_instr = root->mutable_operand(0);
  EXPECT_THAT(new_fusion_instr,
              AllOf(HasOpcode(HloOpcode::kFusion), HasShape(flat_output_shape),
                    HasNumOperands(1)));

  HloInstruction* input_bitcast = new_fusion_instr->mutable_operand(0);
  EXPECT_THAT(input_bitcast,
              AllOf(HasOpcode(HloOpcode::kBitcast), HasShape(flat_input_shape),
                    HasNumOperands(1)));
  EXPECT_THAT(input_bitcast->operand(0), HasOpcode(HloOpcode::kParameter));

  HloComputation* fused_comp =
      new_fusion_instr->fused_instructions_computation();
  EXPECT_THAT(fused_comp->parameter_instructions(),
              ElementsAre(AllOf(HasOpcode(HloOpcode::kParameter),
                                HasShape(flat_input_shape))));
  EXPECT_THAT(
      fused_comp->root_instruction(),
      AllOf(HasOpcode(HloOpcode::kReduceScatter), HasShape(flat_output_shape)));
}

TEST_F(CollectiveBlockLevelConfigTest, ReduceScatterBlockLevelConfig) {
  ASSERT_OK_AND_ASSIGN(
      ModuleWithFusion module_with_fusion,
      BuildModuleWithFusion(GetModuleStr(ShapeUtil::MakeShape(F32, {65536}),
                                         "{0,1}", HloOpcode::kReduceScatter)));
  EXPECT_FALSE(ShouldFlattenReduceScatter(Cast<HloReduceScatterInstruction>(
      GetCollectiveHero(module_with_fusion.FusionInstr()
                            ->fused_instructions_computation()))));
  ASSERT_OK_AND_ASSIGN(BlockLevelFusionConfig block_level_config,
                       GetCollectiveBlockLevelFusionConfig(
                           *gpu_topology_, module_with_fusion.FusionInstr()));

  EXPECT_EQ(block_level_config.num_warps(), 16);
  EXPECT_EQ(block_level_config.num_ctas(), 1);
  EXPECT_EQ(block_level_config.num_stages(), 1);
  EXPECT_THAT(block_level_config.output_tiles(0).sizes(), ElementsAre(2048));
}

TEST_F(CollectiveEmitterTest,
       ReduceScatterGetCollectiveUnmanagedKernelArguments) {
  ASSERT_OK_AND_ASSIGN(
      ModuleWithFusion module_with_fusion,
      BuildModuleWithFusion(GetModuleStr(ShapeUtil::MakeShape(F32, {65536}),
                                         "{0,1}", HloOpcode::kReduceScatter)));
  ASSERT_OK_AND_ASSIGN(
      const auto unmanaged_arguments,
      GetCollectiveUnmanagedKernelArguments(module_with_fusion.FusionInstr()));
  ASSERT_EQ(unmanaged_arguments.size(), 3);
  EXPECT_EQ(unmanaged_arguments[0].dimensions().size(), 0);
  ASSERT_EQ(unmanaged_arguments[1].dimensions().size(), 2);
  EXPECT_EQ(unmanaged_arguments[1].dimensions()[0], 2);
  EXPECT_THAT(unmanaged_arguments[2].dimensions(), ElementsAre(2, 65536));
}

constexpr absl::string_view kVariadicReduceScatterModuleStr = R"(
  HloModule test
  apply_op {
    x = f32[] parameter(0)
    y = f32[] parameter(1)
    ROOT apply_op = f32[] add(x, y)
  }

  ENTRY test_computation {
    param_0 = f32[2,4,8] parameter(0)
    param_1 = f32[4,2] parameter(1)
    ROOT rs = (f32[1,4,8], f32[2,2]) reduce-scatter(param_0, param_1),
      to_apply=apply_op, replica_groups={{0,1}}, dimensions={0}
  }
)";

TEST_F(CollectiveEmitterTest, FlattenVariadicReduceScatterFusion) {
  ASSERT_OK_AND_ASSIGN(ModuleWithFusion module_with_fusion,
                       BuildModuleWithFusion(kVariadicReduceScatterModuleStr));
  HloFusionInstruction* fusion_instr = module_with_fusion.MutableFusionInstr();
  HloComputation* entry_computation =
      module_with_fusion.module->entry_computation();
  EXPECT_FALSE(ShouldFlattenReduceScatter(Cast<HloReduceScatterInstruction>(
      GetCollectiveHero(fusion_instr->fused_instructions_computation()))));

  const Shape output_0 = ShapeUtil::MakeShape(F32, {1, 4, 8});
  const Shape output_1 = ShapeUtil::MakeShape(F32, {2, 2});
  HloInstruction* root = entry_computation->root_instruction();
  EXPECT_THAT(root, AllOf(HasOpcode(HloOpcode::kTuple), HasNumOperands(2)));
  EXPECT_THAT(root->operand(0),
              AllOf(HasOpcode(HloOpcode::kBitcast), HasShape(output_0)));
  EXPECT_THAT(root->operand(1),
              AllOf(HasOpcode(HloOpcode::kBitcast), HasShape(output_1)));
  EXPECT_THAT(root->operand(0)->operand(0),
              HasOpcode(HloOpcode::kGetTupleElement));
  EXPECT_EQ(root->operand(0)->operand(0)->operand(0), fusion_instr);
  EXPECT_EQ(root->operand(1)->operand(0)->operand(0), fusion_instr);

  // Operand 0: f32[2,4,8] -> f32[4,16] -> f32[2,16].
  // Operand 1: f32[4,2]   -> f32[4,2]  -> f32[2,2].
  const Shape param_0 = ShapeUtil::MakeShape(F32, {4, 16});
  const Shape param_1 = ShapeUtil::MakeShape(F32, {4, 2});
  const Shape new_output_0 = ShapeUtil::MakeShape(F32, {2, 16});
  const Shape new_output_1 = ShapeUtil::MakeShape(F32, {2, 2});
  const Shape new_fusion_shape =
      ShapeUtil::MakeTupleShape({new_output_0, new_output_1});
  EXPECT_THAT(fusion_instr, HasShape(new_fusion_shape));
  EXPECT_THAT(fusion_instr->operand(0),
              AllOf(HasOpcode(HloOpcode::kBitcast), HasShape(param_0)));
  EXPECT_THAT(fusion_instr->operand(1),
              AllOf(HasOpcode(HloOpcode::kBitcast), HasShape(param_1)));

  HloComputation* fused_comp = fusion_instr->fused_instructions_computation();
  EXPECT_THAT(fused_comp->parameter_instructions(),
              ElementsAre(HasShape(param_0), HasShape(param_1)));
  const HloInstruction* fused_root = fused_comp->root_instruction();
  EXPECT_THAT(fused_root, AllOf(HasOpcode(HloOpcode::kTuple), HasNumOperands(2),
                                HasShape(new_fusion_shape)));
  const HloInstruction* rs = GetCollectiveHero(fused_comp);
  ASSERT_NE(rs, nullptr);
  EXPECT_THAT(rs, AllOf(HasOpcode(HloOpcode::kReduceScatter), HasNumOperands(2),
                        HasShape(new_fusion_shape)));
  EXPECT_EQ(rs->operand(0), fused_comp->parameter_instruction(0));
  EXPECT_EQ(rs->operand(1), fused_comp->parameter_instruction(1));
  EXPECT_EQ(Cast<HloReduceScatterInstruction>(rs)->scatter_dimension(), 0);
  EXPECT_THAT(verifier().Run(module_with_fusion.module.get()),
              absl_testing::IsOkAndHolds(false));
}

TEST_F(CollectiveBlockLevelConfigTest, VariadicReduceScatterBlockLevelConfig) {
  ASSERT_OK_AND_ASSIGN(ModuleWithFusion module_with_fusion,
                       BuildModuleWithFusion(kVariadicReduceScatterModuleStr));
  ASSERT_OK_AND_ASSIGN(BlockLevelFusionConfig block_level_config,
                       GetCollectiveBlockLevelFusionConfig(
                           *gpu_topology_, module_with_fusion.FusionInstr()));

  EXPECT_EQ(block_level_config.num_ctas(), 1);
  EXPECT_EQ(block_level_config.num_stages(), 1);
  ASSERT_EQ(block_level_config.output_tiles_size(), 2);
  EXPECT_THAT(block_level_config.output_tiles(0).sizes(), ElementsAre(1, 16));
  EXPECT_THAT(block_level_config.output_tiles(1).sizes(), ElementsAre(1, 2));
}

TEST_F(CollectiveBlockLevelConfigTest,
       VariadicReduceScatterBlockLevelConfigTilesOutputsIndependently) {
  ASSERT_OK_AND_ASSIGN(ModuleWithFusion module_with_fusion,
                       BuildModuleWithFusion(R"(
  HloModule test
  apply_op {
    x = f32[] parameter(0)
    y = f32[] parameter(1)
    ROOT apply_op = f32[] add(x, y)
  }

  ENTRY test_computation {
    param_0 = f32[16384] parameter(0)
    param_1 = f32[4,1] parameter(1)
    ROOT rs = (f32[8192], f32[2,1]) reduce-scatter(param_0, param_1),
      to_apply=apply_op, replica_groups={{0,1}}, dimensions={0}
  }
)"));
  EXPECT_FALSE(ShouldFlattenReduceScatter(Cast<HloReduceScatterInstruction>(
      GetCollectiveHero(module_with_fusion.FusionInstr()
                            ->fused_instructions_computation()))));
  ASSERT_OK_AND_ASSIGN(BlockLevelFusionConfig block_level_config,
                       GetCollectiveBlockLevelFusionConfig(
                           *gpu_topology_, module_with_fusion.FusionInstr()));

  // 2 blocks/rank:
  // - Output 0 (f32[8192], 4096/rank) -> 2 tiles of [2048].
  // - Output 1 (f32[2,1], [1,1]/rank) -> 1 tile of [1,1].
  EXPECT_EQ(block_level_config.num_warps(), 16);
  ASSERT_EQ(block_level_config.output_tiles_size(), 2);
  EXPECT_THAT(block_level_config.output_tiles(0).sizes(), ElementsAre(2048));
  EXPECT_THAT(block_level_config.output_tiles(1).sizes(), ElementsAre(1, 1));
}

constexpr absl::string_view kVariadicReduceScatterNonPowerOfTwoModuleStr = R"(
  HloModule test
  apply_op {
    x = f32[] parameter(0)
    y = f32[] parameter(1)
    ROOT apply_op = f32[] add(x, y)
  }

  ENTRY test_computation {
    param_0 = f32[2,24576] parameter(0)
    param_1 = f32[4,2] parameter(1)
    ROOT rs = (f32[1,24576], f32[2,2]) reduce-scatter(param_0, param_1),
      to_apply=apply_op, replica_groups={{0,1}}, dimensions={0}
  }
)";

TEST_F(CollectiveEmitterTest,
       FlattenVariadicReduceScatterFusionNonPowerOfTwoInner) {
  ASSERT_OK_AND_ASSIGN(
      ModuleWithFusion module_with_fusion,
      BuildModuleWithFusion(kVariadicReduceScatterNonPowerOfTwoModuleStr));
  HloFusionInstruction* fusion_instr = module_with_fusion.MutableFusionInstr();
  HloComputation* entry_computation =
      module_with_fusion.module->entry_computation();

  // Operand 0 (12288/rank, non-pow2):
  //   f32[2,24576] -> f32[4,32,384] -> f32[2,32,384].
  // Operand 1 (2/rank, pow2):
  //   f32[4,2] -> f32[4,2] -> f32[2,2].
  const Shape param_0 = ShapeUtil::MakeShape(F32, {4, 32, 384});
  const Shape param_1 = ShapeUtil::MakeShape(F32, {4, 2});
  const Shape new_output_0 = ShapeUtil::MakeShape(F32, {2, 32, 384});
  const Shape new_output_1 = ShapeUtil::MakeShape(F32, {2, 2});
  const Shape new_fusion_shape =
      ShapeUtil::MakeTupleShape({new_output_0, new_output_1});
  EXPECT_THAT(fusion_instr, HasShape(new_fusion_shape));
  EXPECT_THAT(fusion_instr->operand(0),
              AllOf(HasOpcode(HloOpcode::kBitcast), HasShape(param_0)));
  EXPECT_THAT(fusion_instr->operand(1),
              AllOf(HasOpcode(HloOpcode::kBitcast), HasShape(param_1)));
  HloComputation* fused_comp = fusion_instr->fused_instructions_computation();
  EXPECT_THAT(fused_comp->parameter_instructions(),
              ElementsAre(HasShape(param_0), HasShape(param_1)));
  EXPECT_THAT(fused_comp->root_instruction(),
              AllOf(HasOpcode(HloOpcode::kTuple), HasShape(new_fusion_shape)));
  EXPECT_THAT(
      GetCollectiveHero(fused_comp),
      AllOf(HasOpcode(HloOpcode::kReduceScatter), HasShape(new_fusion_shape)));

  HloInstruction* root = entry_computation->root_instruction();
  EXPECT_THAT(root, AllOf(HasOpcode(HloOpcode::kTuple), HasNumOperands(2)));
  EXPECT_THAT(root->operand(0),
              AllOf(HasOpcode(HloOpcode::kBitcast),
                    HasShape(ShapeUtil::MakeShape(F32, {1, 24576}))));
  EXPECT_THAT(root->operand(1),
              AllOf(HasOpcode(HloOpcode::kBitcast),
                    HasShape(ShapeUtil::MakeShape(F32, {2, 2}))));
  EXPECT_THAT(verifier().Run(module_with_fusion.module.get()),
              absl_testing::IsOkAndHolds(false));
}

TEST_F(CollectiveBlockLevelConfigTest,
       VariadicReduceScatterBlockLevelConfigNonPowerOfTwoInner) {
  ASSERT_OK_AND_ASSIGN(
      ModuleWithFusion module_with_fusion,
      BuildModuleWithFusion(kVariadicReduceScatterNonPowerOfTwoModuleStr));
  ASSERT_OK_AND_ASSIGN(BlockLevelFusionConfig block_level_config,
                       GetCollectiveBlockLevelFusionConfig(
                           *gpu_topology_, module_with_fusion.FusionInstr()));

  // 4 blocks/rank:
  // - Output 0 (f32[2,32,384]) -> 4 tiles of [1,8,512].
  // - Output 1 (f32[2,2])      -> 2 tiles of [1,1].
  EXPECT_EQ(block_level_config.num_warps(), 16);
  ASSERT_EQ(block_level_config.output_tiles_size(), 2);
  EXPECT_THAT(block_level_config.output_tiles(0).sizes(),
              ElementsAre(1, 8, 512));
  EXPECT_THAT(block_level_config.output_tiles(1).sizes(), ElementsAre(1, 1));
}

TEST_F(CollectiveEmitterTest,
       VariadicReduceScatterGetCollectiveUnmanagedKernelArguments) {
  ASSERT_OK_AND_ASSIGN(ModuleWithFusion module_with_fusion,
                       BuildModuleWithFusion(kVariadicReduceScatterModuleStr));
  ASSERT_OK_AND_ASSIGN(
      const auto unmanaged_arguments,
      GetCollectiveUnmanagedKernelArguments(module_with_fusion.FusionInstr()));
  ASSERT_EQ(unmanaged_arguments.size(), 4);
  EXPECT_EQ(unmanaged_arguments[0].dimensions().size(), 0);
  ASSERT_EQ(unmanaged_arguments[1].dimensions().size(), 2);
  EXPECT_EQ(unmanaged_arguments[1].dimensions()[0], 2);
  EXPECT_THAT(unmanaged_arguments[2].dimensions(), ElementsAre(2, 4, 16));
  EXPECT_THAT(unmanaged_arguments[3].dimensions(), ElementsAre(2, 4, 2));
}

GpuTopology MakeGpuTopology(int32_t num_partitions,
                            int32_t num_hosts_per_partition,
                            int32_t num_devices_per_host,
                            int32_t num_devices_per_process,
                            bool is_cuda = true) {
  se::DeviceDescription device_info =
      is_cuda ? TestGpuDeviceInfo::H100SXMDeviceInfo()
              : TestGpuDeviceInfo::AMDMI210DeviceInfo();
  stream_executor::GpuTargetConfigProto target_config_proto;
  *target_config_proto.mutable_gpu_device_info() = device_info.ToProto();
  target_config_proto.set_platform_name(is_cuda ? "CUDA" : "ROCM");
  absl::StatusOr<GpuTargetConfig> target_config =
      GpuTargetConfig::FromProto(target_config_proto);
  CHECK_OK(target_config);
  return GpuTopology(
      /*platform_version=*/"", num_partitions, num_hosts_per_partition,
      num_devices_per_host, *target_config,
      /*host_target_machine_options=*/std::nullopt, num_devices_per_process);
}

class GetSymmetricMemoryTypeTest : public HloHardwareIndependentTestBase {
 protected:
  // Returns a module with an all-reduce over 8 replicas wrapped in a fusion.
  // The cross-host one-shot kernel flag is set to ALLCOLLECTIVES iff
  // `cross_host_kernel_enabled`.
  absl::StatusOr<ModuleWithFusion> BuildAllReduceFusion(
      bool cross_host_kernel_enabled) {
    constexpr absl::string_view kHlo = R"(
      HloModule test
      add {
        x = f32[] parameter(0)
        y = f32[] parameter(1)
        ROOT add = f32[] add(x, y)
      }
      ENTRY e {
        p = f32[1024] parameter(0)
        ROOT all-reduce = f32[1024] all-reduce(p),
            replica_groups={{0,1,2,3,4,5,6,7}}, to_apply=add
      }
    )";
    ABSL_ASSIGN_OR_RETURN(std::unique_ptr<HloModule> module,
                     ParseAndReturnVerifiedModule(kHlo, /*replica_count=*/8));
    DebugOptions& debug_options =
        module->mutable_config().mutable_debug_options();
    debug_options.clear_xla_gpu_unsupported_use_cross_host_one_shot_kernel();
    if (cross_host_kernel_enabled) {
      debug_options.add_xla_gpu_unsupported_use_cross_host_one_shot_kernel(
          DebugOptions::ALLCOLLECTIVES);
    }
    return ModuleWithFusion{
        NewModuleWithFusion(module->entry_computation()->root_instruction(),
                            HloInstruction::FusionKind::kLoop)};
  }
};

TEST_F(GetSymmetricMemoryTypeTest, SingleProcessNvidiaIsLoadStoreAccessible) {
  ASSERT_OK_AND_ASSIGN(
      ModuleWithFusion module_with_fusion,
      BuildAllReduceFusion(/*cross_host_kernel_enabled=*/false));
  EXPECT_THAT(
      GetSymmetricMemoryType(MakeGpuTopology(/*num_partitions=*/2,
                                             /*num_hosts_per_partition=*/1,
                                             /*num_devices_per_host=*/8,
                                             /*num_devices_per_process=*/8,
                                             /*is_cuda=*/true),
                             *module_with_fusion.FusionInstr()),
      IsOkAndHolds(SymmetricMemoryType::kLoadStoreAccessible));
}

TEST_F(GetSymmetricMemoryTypeTest, SingleProcessRocmIsXlaRendezvous) {
  ASSERT_OK_AND_ASSIGN(
      ModuleWithFusion module_with_fusion,
      BuildAllReduceFusion(/*cross_host_kernel_enabled=*/false));
  EXPECT_THAT(
      GetSymmetricMemoryType(MakeGpuTopology(/*num_partitions=*/2,
                                             /*num_hosts_per_partition=*/1,
                                             /*num_devices_per_host=*/8,
                                             /*num_devices_per_process=*/8,
                                             /*is_cuda=*/false),
                             *module_with_fusion.FusionInstr()),
      IsOkAndHolds(SymmetricMemoryType::kXlaRendezvous));
}

TEST_F(GetSymmetricMemoryTypeTest,
       CrossProcessNvidiaSharedSliceWithFlagIsLoadStoreAccessible) {
  ASSERT_OK_AND_ASSIGN(
      ModuleWithFusion module_with_fusion,
      BuildAllReduceFusion(/*cross_host_kernel_enabled=*/true));
  const GpuTopology topology = MakeGpuTopology(/*num_partitions=*/1,
                                               /*num_hosts_per_partition=*/2,
                                               /*num_devices_per_host=*/4,
                                               /*num_devices_per_process=*/4,
                                               /*is_cuda=*/true);
  const HloFusionInstruction& fusion = *module_with_fusion.FusionInstr();
  EXPECT_THAT(GetSymmetricMemoryType(topology, fusion),
              IsOkAndHolds(SymmetricMemoryType::kLoadStoreAccessible));
  EXPECT_THAT(GetSymmetricMemoryType(topology, *fusion.fused_expression_root()),
              IsOkAndHolds(SymmetricMemoryType::kLoadStoreAccessible));
}

TEST_F(GetSymmetricMemoryTypeTest,
       CrossProcessNvidiaSharedSliceWithoutFlagReturnsError) {
  ASSERT_OK_AND_ASSIGN(
      ModuleWithFusion module_with_fusion,
      BuildAllReduceFusion(/*cross_host_kernel_enabled=*/false));
  EXPECT_THAT(
      GetSymmetricMemoryType(MakeGpuTopology(/*num_partitions=*/1,
                                             /*num_hosts_per_partition=*/2,
                                             /*num_devices_per_host=*/4,
                                             /*num_devices_per_process=*/4,
                                             /*is_cuda=*/true),
                             *module_with_fusion.FusionInstr()),
      StatusIs(absl::StatusCode::kUnimplemented));
}

TEST_F(GetSymmetricMemoryTypeTest,
       CrossProcessNvidiaSeparateSlicesReturnsError) {
  ASSERT_OK_AND_ASSIGN(
      ModuleWithFusion module_with_fusion,
      BuildAllReduceFusion(/*cross_host_kernel_enabled=*/true));
  EXPECT_THAT(
      GetSymmetricMemoryType(MakeGpuTopology(/*num_partitions=*/2,
                                             /*num_hosts_per_partition=*/1,
                                             /*num_devices_per_host=*/4,
                                             /*num_devices_per_process=*/4,
                                             /*is_cuda=*/true),
                             *module_with_fusion.FusionInstr()),
      StatusIs(absl::StatusCode::kUnimplemented));
}

TEST_F(GetSymmetricMemoryTypeTest, CrossProcessRocmSharedSliceReturnsError) {
  ASSERT_OK_AND_ASSIGN(
      ModuleWithFusion module_with_fusion,
      BuildAllReduceFusion(/*cross_host_kernel_enabled=*/true));
  EXPECT_THAT(
      GetSymmetricMemoryType(MakeGpuTopology(/*num_partitions=*/1,
                                             /*num_hosts_per_partition=*/2,
                                             /*num_devices_per_host=*/4,
                                             /*num_devices_per_process=*/4,
                                             /*is_cuda=*/false),
                             *module_with_fusion.FusionInstr()),
      StatusIs(absl::StatusCode::kUnimplemented));
}

TEST_F(CollectiveEmitterTest,
       ReduceScatterBlockLevelConfigUses64BlocksForNonPowerOfTwoShape) {
  constexpr absl::string_view kReduceScatterHloStr = R"(
    HloModule test
    add {
      x = bf16[] parameter(0)
      y = bf16[] parameter(1)
      ROOT add = bf16[] add(x, y)
    }
    ENTRY test_computation {
      param_0 = bf16[128,24576] parameter(0)
      ROOT rs = bf16[8,24576] reduce-scatter(param_0),
        replica_groups={{0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}},
        dimensions={0}, to_apply=add
    }
  )";
  ASSERT_OK_AND_ASSIGN(GpuTopology topology_16,
                       GetGpuTopologyForPlatform("nvidia_h100", 1, 1, 16));
  ASSERT_OK_AND_ASSIGN(ModuleWithFusion module_with_fusion,
                       BuildModuleWithFusion(kReduceScatterHloStr, topology_16,
                                             /*replica_count=*/16));

  // Root is now a bitcast back to bf16[8,24576]; operand(0) is the 3D fusion
  // with shape bf16[16,4,3072].
  HloInstruction* root =
      module_with_fusion.module->entry_computation()->root_instruction();
  ASSERT_EQ(root->opcode(), HloOpcode::kBitcast);
  HloFusionInstruction* flat_fusion = module_with_fusion.MutableFusionInstr();
  EXPECT_EQ(flat_fusion, root->operand(0));
  EXPECT_EQ(flat_fusion->shape(), ShapeUtil::MakeShape(BF16, {16, 4, 3072}));

  ASSERT_OK_AND_ASSIGN(
      BlockLevelFusionConfig block_level_config,
      GetCollectiveBlockLevelFusionConfig(topology_16, flat_fusion));
  EXPECT_THAT(block_level_config.output_tiles(0).sizes(),
              ElementsAre(1, 1, 4096));

  HloFusionAnalysis analysis =
      HloFusionAnalysis::Create(*flat_fusion, device_info_);
  std::optional<TritonFusion::LaunchConfig> launch_config =
      TritonFusion::GetLaunchConfig(&analysis);
  ASSERT_TRUE(launch_config.has_value());
  EXPECT_EQ(launch_config->launch_dimensions.num_blocks(), 64);
  EXPECT_EQ(launch_config->launch_dimensions.num_threads_per_block(), 512);

  auto llvm_compiler =
      [&](llvm::Module& llvm_module, const se::DeviceDescription& descr,
          const DebugOptions& opts) -> absl::StatusOr<std::vector<uint8_t>> {
    return std::vector<uint8_t>{1};
  };
  DebugOptions debug_options;
  CubinCustomKernelCompiler kernel_compiler(llvm_compiler, device_info_,
                                            debug_options, topology_16);
  ObjectPool<std::unique_ptr<mlir::MLIRContext>> mlir_context_pool(
      []() { return CreateMlirContext(); });
  ASSERT_OK_AND_ASSIGN(BorrowedMlirContext borrowed_context,
                       mlir_context_pool.GetOrCreate());
  TritonFusion emitter(analysis);
  ASSERT_OK_AND_ASSIGN(
      TritonWrapperResult triton_kernel,
      emitter
          .GenerateTritonKernelAndWrapper(
              *flat_fusion, "test-reduce-scatter", device_info_,
              llvm::Triple(""), /*data_layout=*/"", std::move(borrowed_context),
              &kernel_compiler)
          .Await());
}

constexpr absl::string_view kVariadicAllGatherTwoDevicesHloStr = R"(
  HloModule test
  ENTRY test_computation {
    param_0 = f32[4,4096] parameter(0)
    param_1 = pred[2] parameter(1)
    ROOT ag = (f32[8,4096], pred[4]) all-gather(param_0, param_1),
      dimensions={0}, replica_groups={{0,1}}
  }
)";

// Every output is tiled independently for `num_blocks / num_devices` programs
// per peer: 32768 output elements -> 16 blocks -> 8 blocks per peer. The f32
// output gets 8 tiles per peer, the pred output only 2; the schedule pads the
// latter so the grid is 2 * 8 programs.
TEST_F(CollectiveBlockLevelConfigTest,
       VariadicAllGatherBlockLevelConfigTilesOutputsIndependently) {
  ASSERT_OK_AND_ASSIGN(
      ModuleWithFusion module_with_fusion,
      BuildModuleWithFusion(kVariadicAllGatherTwoDevicesHloStr));
  HloFusionInstruction* fusion = module_with_fusion.MutableFusionInstr();
  ASSERT_OK_AND_ASSIGN(
      BlockLevelFusionConfig block_level_config,
      GetCollectiveBlockLevelFusionConfig(*gpu_topology_, fusion));
  ASSERT_EQ(block_level_config.output_tiles_size(), 2);
  EXPECT_THAT(block_level_config.output_tiles(0).sizes(), ElementsAre(1, 2048));
  EXPECT_THAT(block_level_config.output_tiles(1).sizes(), ElementsAre(1));

  HloFusionAnalysis analysis = HloFusionAnalysis::Create(*fusion, device_info_);
  std::optional<TritonFusion::LaunchConfig> launch_config =
      TritonFusion::GetLaunchConfig(&analysis);
  ASSERT_TRUE(launch_config.has_value());
  EXPECT_EQ(launch_config->launch_dimensions.num_blocks(), 16);
}

// Shape from `all-gather.170` on 16 ranks.
// Total output is ~10.5 MB -> 64 blocks (4 blocks per peer).
// Each output is tiled independently:
// - Output 0 (`f32[32,8,10240]`, per-peer `[2,8,10240]`): 4 tiles of
//   `[1,4,16384]`.
// - Output 3 (`pred[32]`, per-peer `[2]`): 2 tiles of `[1]` (padded to 4 by
//   the schedule).
TEST_F(CollectiveBlockLevelConfigTest,
       VariadicAllGatherBlockLevelConfigSmallOutputsDoNotLimitGrid) {
  constexpr absl::string_view kHloStr = R"(
    HloModule test
    ENTRY test_computation {
      param_0 = f32[2,8,10240] parameter(0)
      param_1 = pred[2,128] parameter(1)
      param_2 = s32[2,256] parameter(2)
      param_3 = pred[2] parameter(3)
      param_4 = pred[2] parameter(4)
      param_5 = s32[8,2] parameter(5)
      ROOT ag = (f32[32,8,10240], pred[32,128], s32[32,256], pred[32], pred[32],
                 s32[128,2]) all-gather(param_0, param_1, param_2, param_3,
                 param_4, param_5), dimensions={0},
        replica_groups={{0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}}
    }
  )";
  ASSERT_OK_AND_ASSIGN(GpuTopology topology_16,
                       GetGpuTopologyForPlatform("nvidia_h100", 1, 1, 16));
  ASSERT_OK_AND_ASSIGN(
      ModuleWithFusion module_with_fusion,
      BuildModuleWithFusion(kHloStr, topology_16, /*replica_count=*/16));
  HloFusionInstruction* fusion = module_with_fusion.MutableFusionInstr();
  ASSERT_OK_AND_ASSIGN(
      BlockLevelFusionConfig block_level_config,
      GetCollectiveBlockLevelFusionConfig(topology_16, fusion));
  ASSERT_EQ(block_level_config.output_tiles_size(), 6);
  // 4 tiles per peer: 128 elements per thread with 512 threads, the same as
  // the single-operand all-gather of f32[2,8,10240].
  EXPECT_THAT(block_level_config.output_tiles(0).sizes(),
              ElementsAre(1, 4, 16384));
  EXPECT_THAT(block_level_config.output_tiles(1).sizes(), ElementsAre(1, 64));
  EXPECT_THAT(block_level_config.output_tiles(2).sizes(), ElementsAre(1, 128));
  // 2 tiles per peer each; padded to 4 by the schedule.
  EXPECT_THAT(block_level_config.output_tiles(3).sizes(), ElementsAre(1));
  EXPECT_THAT(block_level_config.output_tiles(4).sizes(), ElementsAre(1));
  // 4 tiles per peer.
  EXPECT_THAT(block_level_config.output_tiles(5).sizes(), ElementsAre(2, 2));

  HloFusionAnalysis analysis = HloFusionAnalysis::Create(*fusion, device_info_);
  std::optional<TritonFusion::LaunchConfig> launch_config =
      TritonFusion::GetLaunchConfig(&analysis);
  ASSERT_TRUE(launch_config.has_value());
  EXPECT_EQ(launch_config->launch_dimensions.num_blocks(), 64);
  EXPECT_EQ(launch_config->launch_dimensions.num_threads_per_block(), 512);
}

}  // namespace

}  // namespace xla::gpu
