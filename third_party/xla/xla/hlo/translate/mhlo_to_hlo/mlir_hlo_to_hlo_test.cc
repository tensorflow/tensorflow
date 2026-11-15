/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/hlo/translate/mhlo_to_hlo/mlir_hlo_to_hlo.h"

#include <array>
#include <cstdint>
#include <memory>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/check.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "benchmark/benchmark.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/translate/register.h"
#include "xla/mlir/utils/error_util.h"
#include "xla/service/hlo.pb.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/test.h"

// This file should contain tests for interfaces that can't be tested at the
// MLIR level.

namespace mlir {
namespace {

using testing::_;
using testing::AllOf;
using testing::HasSubstr;

TEST(ConvertMlirHloToHloModuleTest, PropagatesDiagnostics) {
  const std::string mlir_source = R"mlir(
func.func @main(%arg0: tensor<?xf32>, %arg1: tensor<1xindex>, %arg2: tensor<1xindex>, %arg3: tensor<1xindex>) -> tensor<?xf32> {
  %0 = shape.const_shape [14, 1] : tensor<2xindex>
  %1 = "stablehlo.real_dynamic_slice"(%arg0, %arg1, %arg2, %arg3) : (tensor<?xf32>, tensor<1xindex>, tensor<1xindex>, tensor<1xindex>) -> tensor<?xf32>
  func.return %1 : tensor<?xf32>
}
)mlir";

  mlir::DialectRegistry registry;
  xla::RegisterMlirToHloDependentDialects(registry);
  mlir::MLIRContext context(registry);
  mlir::OwningOpRef<mlir::ModuleOp> module;
  {
    mlir::BaseScopedDiagnosticHandler handler(&context);
    module = mlir::parseSourceString<mlir::ModuleOp>(mlir_source, &context);
    TF_ASSERT_OK(handler.ConsumeStatus());
  }

  ASSERT_THAT(ConvertMlirHloToHloModule(*module),
              absl_testing::StatusIs(
                  _, AllOf(HasSubstr("Unable to prepare for XLA export"),
                           HasSubstr("real_dynamic_slice"))));
}

TEST(ConvertMlirHloToHloModuleTest, ConvertsDotGeneralPrecisionConfig) {
  const std::string mlir_source = R"mlir(
func.func @main(%arg0: tensor<5x10xbf16>, %arg1: tensor<10x5xbf16>) -> tensor<5x5xbf16> {
  %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0], precision = [HIGHEST, HIGHEST] : (tensor<5x10xbf16>, tensor<10x5xbf16>) -> tensor<5x5xbf16>
  return %0 : tensor<5x5xbf16>
}
)mlir";

  mlir::DialectRegistry registry;
  xla::RegisterMlirToHloDependentDialects(registry);
  mlir::MLIRContext context(registry);
  mlir::OwningOpRef<mlir::ModuleOp> module;
  {
    mlir::BaseScopedDiagnosticHandler handler(&context);
    module = mlir::parseSourceString<mlir::ModuleOp>(mlir_source, &context);
    TF_ASSERT_OK(handler.ConsumeStatus());
  }

  TF_ASSERT_OK(ConvertMlirHloToHloModule(*module));
}
TEST(ConvertMlirHloToHloModuleTest, ConvertsConvolutionPrecisionConfig) {
  const std::string mlir_source = R"mlir(
func.func @main(%arg0: tensor<3x3x3x3xf32>, %arg1: tensor<3x3x3x3xf32>) -> tensor<3x3x3x3xf32> {
  %0 = stablehlo.convolution(%arg0, %arg1) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {pad = [[1, 1], [1, 1]]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision HIGHEST>, #stablehlo<precision HIGHEST>]} : (tensor<3x3x3x3xf32>, tensor<3x3x3x3xf32>) -> tensor<3x3x3x3xf32>
  return %0 : tensor<3x3x3x3xf32>
}
)mlir";

  mlir::DialectRegistry registry;
  xla::RegisterMlirToHloDependentDialects(registry);
  mlir::MLIRContext context(registry);
  mlir::OwningOpRef<mlir::ModuleOp> module;
  {
    mlir::BaseScopedDiagnosticHandler handler(&context);
    module = mlir::parseSourceString<mlir::ModuleOp>(mlir_source, &context);
    TF_ASSERT_OK(handler.ConsumeStatus());
  }

  TF_ASSERT_OK(ConvertMlirHloToHloModule(*module));
}
TEST(ConvertMlirHloToHloModuleTest, ConvertsReplicaGroupMeshAxes) {
  const std::string kMlirModule = R"mlir(
    module @main {
      sdy.mesh @mesh = <["a"=2, "b"=2], device_ids=[0, 1, 2, 3]>
      func.func @main(%arg0: tensor<f32>) -> tensor<f32> {
        %0 = "stablehlo.all_reduce"(%arg0) <{
          channel_handle = #stablehlo.channel_handle<handle = 1, type = 0>,
          replica_groups = #stablehlo.replica_group_mesh_axes<
            mesh = @mesh,
            axes = [#stablehlo.axis_ref<name = "a">, #stablehlo.axis_ref<name = "b">]
          >,
          use_global_device_ids
        }> ({
        ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
          %1 = "stablehlo.add"(%arg1, %arg2) : (tensor<f32>, tensor<f32>) -> tensor<f32>
          "stablehlo.return"(%1) : (tensor<f32>) -> ()
        }) : (tensor<f32>) -> tensor<f32>
        return %0 : tensor<f32>
      }
    }
  )mlir";

  mlir::DialectRegistry registry;
  xla::RegisterMlirToHloDependentDialects(registry);
  mlir::MLIRContext context(registry);

  mlir::BaseScopedDiagnosticHandler handler(&context);
  auto module = mlir::parseSourceString<mlir::ModuleOp>(kMlirModule, &context);
  TF_ASSERT_OK(handler.ConsumeStatus());
  ASSERT_TRUE(module);
  auto hlo_module = ConvertMlirHloToHloModule(*module);
  TF_EXPECT_OK(hlo_module.status());
}

TEST(ConvertMlirHloToHloModuleTest, PacksSpmdParametersShardingsForTupleArgs) {
  const std::string kMlirModule = R"mlir(
    module attributes {
      mhlo.spmd_parameters_shardings = [
        "{devices=[1,2]<=[2]}",
        "{{replicated}, {devices=[2,1]<=[2]}}"
      ]
    } {
      func.func @main(
          %arg0: tensor<2x4xf32>,
          %arg1: tuple<tensor<f32>, tensor<2x4xf32>>) -> tensor<2x4xf32> {
        return %arg0 : tensor<2x4xf32>
      }
    }
  )mlir";

  mlir::DialectRegistry registry;
  xla::RegisterMlirToHloDependentDialects(registry);
  mlir::MLIRContext context(registry);

  mlir::BaseScopedDiagnosticHandler handler(&context);
  auto module = mlir::parseSourceString<mlir::ModuleOp>(kMlirModule, &context);
  TF_ASSERT_OK(handler.ConsumeStatus());
  ASSERT_TRUE(module);

  MlirToHloConversionOptions options;
  options.use_tuple_args = true;
  auto hlo_module = ConvertMlirHloToHloModule(*module, options);
  TF_ASSERT_OK(hlo_module.status());
  ASSERT_TRUE((*hlo_module)->has_spmd_parameters_shardings());
  ASSERT_EQ((*hlo_module)->spmd_parameters_shardings().size(), 1);
  EXPECT_EQ((*hlo_module)->spmd_parameters_shardings()[0].ToString(),
            "{{devices=[1,2]<=[2]}, {replicated}, {devices=[2,1]<=[2]}}");
}

TEST(ConvertMlirHloToHloModuleTest, MemoizesRepeatedNamedShardings) {
  constexpr absl::string_view kShardingA =
      "{mesh['x'=2,'y'=2], [{'x'}, {'y'}]}";
  constexpr absl::string_view kShardingB =
      "{mesh['x'=2,'y'=2], [{'y'}, {'x'}]}";
  const std::string kMlirModule = R"mlir(
    module attributes {
      mhlo.spmd_output_sharding = "{mesh['x'=2,'y'=2], [{'x'}, {'y'}]}",
      mhlo.spmd_parameters_shardings = [
        "{mesh['x'=2,'y'=2], [{'x'}, {'y'}]}",
        "{mesh['x'=2,'y'=2], [{'x'}, {'y'}]}",
        "{mesh['x'=2,'y'=2], [{'y'}, {'x'}]}"
      ]
    } {
      func.func private @callee(
          %arg0: tensor<2x2xf32> {mhlo.sharding = "{mesh['x'=2,'y'=2], [{'x'}, {'y'}]}"})
          -> (tensor<2x2xf32> {mhlo.sharding = "{mesh['x'=2,'y'=2], [{'y'}, {'x'}]}"}) {
        %0 = stablehlo.negate %arg0 {mhlo.sharding = "{mesh['x'=2,'y'=2], [{'y'}, {'x'}]}"} : tensor<2x2xf32>
        return %0 : tensor<2x2xf32>
      }

      func.func @main(
          %arg0: tensor<2x2xf32> {mhlo.sharding = "{mesh['x'=2,'y'=2], [{'x'}, {'y'}]}"},
          %arg1: tensor<2x2xf32> {mhlo.sharding = "{mesh['x'=2,'y'=2], [{'x'}, {'y'}]}"},
          %arg2: tensor<2x2xf32> {mhlo.sharding = "{mesh['x'=2,'y'=2], [{'y'}, {'x'}]}"})
          -> (tensor<2x2xf32> {mhlo.sharding = "{mesh['x'=2,'y'=2], [{'x'}, {'y'}]}"},
              tensor<2x2xf32> {mhlo.sharding = "{mesh['x'=2,'y'=2], [{'y'}, {'x'}]}"}) {
        %c0 = stablehlo.constant {mhlo.sharding = "{mesh['x'=2,'y'=2], [{'x'}, {'y'}]}"} dense<[[1.0, 2.0], [3.0, 4.0]]> : tensor<2x2xf32>
        %c1 = stablehlo.constant {mhlo.sharding = "{mesh['x'=2,'y'=2], [{'x'}, {'y'}]}"} dense<[[5.0, 6.0], [7.0, 8.0]]> : tensor<2x2xf32>
        %add = stablehlo.add %arg0, %c0 {mhlo.sharding = "{mesh['x'=2,'y'=2], [{'x'}, {'y'}]}"} : tensor<2x2xf32>
        %neg = stablehlo.negate %add {mhlo.sharding = "{mesh['x'=2,'y'=2], [{'x'}, {'y'}]}"} : tensor<2x2xf32>
        %call0 = func.call @callee(%neg) {mhlo.sharding = "{mesh['x'=2,'y'=2], [{'y'}, {'x'}]}"} : (tensor<2x2xf32>) -> tensor<2x2xf32>
        %call1 = func.call @callee(%c1) {mhlo.sharding = "{mesh['x'=2,'y'=2], [{'y'}, {'x'}]}"} : (tensor<2x2xf32>) -> tensor<2x2xf32>
        %bad0 = stablehlo.subtract %arg1, %arg2 {mhlo.sharding = "invalid_sharding"} : tensor<2x2xf32>
        %bad1 = stablehlo.abs %bad0 {mhlo.sharding = "invalid_sharding"} : tensor<2x2xf32>
        %out0 = stablehlo.add %call0, %bad1 {mhlo.sharding = "{mesh['x'=2,'y'=2], [{'x'}, {'y'}]}"} : tensor<2x2xf32>
        return %out0, %call1 : tensor<2x2xf32>, tensor<2x2xf32>
      }
    }
  )mlir";

  mlir::DialectRegistry registry;
  xla::RegisterMlirToHloDependentDialects(registry);
  mlir::MLIRContext context(registry);

  mlir::BaseScopedDiagnosticHandler handler(&context);
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(kMlirModule, &context);
  ASSERT_THAT(handler.ConsumeStatus(), ::absl_testing::IsOk());
  ASSERT_TRUE(module);

  MlirToHloConversionOptions options;
  options.use_tuple_args = true;
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::HloModule> hlo_module,
                       ConvertMlirHloToHloModule(*module, options));

  ASSERT_TRUE(hlo_module->has_spmd_output_sharding());
  EXPECT_EQ(hlo_module->spmd_output_sharding().ToString(), kShardingA);

  ASSERT_TRUE(hlo_module->has_spmd_parameters_shardings());
  ASSERT_EQ(hlo_module->spmd_parameters_shardings().size(), 1);
  EXPECT_EQ(hlo_module->spmd_parameters_shardings()[0].ToString(),
            "{{mesh['x'=2,'y'=2], [{'x'}, {'y'}]}, "
            "{mesh['x'=2,'y'=2], [{'x'}, {'y'}]}, "
            "{mesh['x'=2,'y'=2], [{'y'}, {'x'}]}}");

  const xla::HloComputation* entry = hlo_module->entry_computation();
  ASSERT_NE(entry, nullptr);
  ASSERT_EQ(entry->num_parameters(), 1);
  ASSERT_TRUE(entry->parameter_instruction(0)->has_sharding());
  EXPECT_EQ(entry->parameter_instruction(0)->sharding().ToString(),
            "{{mesh['x'=2,'y'=2], [{'x'}, {'y'}]}, "
            "{mesh['x'=2,'y'=2], [{'x'}, {'y'}]}, "
            "{mesh['x'=2,'y'=2], [{'y'}, {'x'}]}}");
  ASSERT_TRUE(entry->root_instruction()->has_sharding());
  EXPECT_EQ(entry->root_instruction()->sharding().ToString(),
            "{{mesh['x'=2,'y'=2], [{'x'}, {'y'}]}, "
            "{mesh['x'=2,'y'=2], [{'y'}, {'x'}]}}");

  int gte_count = 0;
  int constant_count = 0;
  int add_count = 0;
  int negate_count = 0;
  int call_count = 0;
  int subtract_count = 0;
  int abs_count = 0;
  for (const xla::HloInstruction* instr : entry->instructions()) {
    switch (instr->opcode()) {
      case xla::HloOpcode::kGetTupleElement: {
        ++gte_count;
        ASSERT_TRUE(instr->has_sharding());
        const absl::string_view expected =
            instr->tuple_index() == 2 ? kShardingB : kShardingA;
        EXPECT_EQ(instr->sharding().ToString(), expected);
        break;
      }
      case xla::HloOpcode::kConstant: {
        ++constant_count;
        ASSERT_TRUE(instr->has_sharding());
        EXPECT_EQ(instr->sharding().ToString(), kShardingA);
        break;
      }
      case xla::HloOpcode::kAdd: {
        ++add_count;
        ASSERT_TRUE(instr->has_sharding());
        EXPECT_EQ(instr->sharding().ToString(), kShardingA);
        break;
      }
      case xla::HloOpcode::kNegate: {
        ++negate_count;
        ASSERT_TRUE(instr->has_sharding());
        EXPECT_EQ(instr->sharding().ToString(), kShardingA);
        break;
      }
      case xla::HloOpcode::kCall: {
        ++call_count;
        ASSERT_TRUE(instr->has_sharding());
        EXPECT_EQ(instr->sharding().ToString(), kShardingB);
        const xla::HloComputation* callee = instr->to_apply();
        ASSERT_NE(callee, nullptr);
        ASSERT_EQ(callee->num_parameters(), 1);
        ASSERT_TRUE(callee->parameter_instruction(0)->has_sharding());
        EXPECT_EQ(callee->parameter_instruction(0)->sharding().ToString(),
                  kShardingA);
        ASSERT_TRUE(callee->root_instruction()->has_sharding());
        EXPECT_EQ(callee->root_instruction()->sharding().ToString(),
                  kShardingB);
        break;
      }
      case xla::HloOpcode::kSubtract: {
        ++subtract_count;
        EXPECT_FALSE(instr->has_sharding());
        break;
      }
      case xla::HloOpcode::kAbs: {
        ++abs_count;
        EXPECT_FALSE(instr->has_sharding());
        break;
      }
      default: {
        break;
      }
    }
  }
  EXPECT_EQ(gte_count, 3);
  EXPECT_EQ(constant_count, 2);
  EXPECT_EQ(add_count, 2);
  EXPECT_EQ(negate_count, 1);
  EXPECT_EQ(call_count, 2);
  EXPECT_EQ(subtract_count, 1);
  EXPECT_EQ(abs_count, 1);

  MlirToHloConversionOptions non_tuple_options;
  non_tuple_options.use_tuple_args = false;
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::HloModule> non_tuple_hlo_module,
                       ConvertMlirHloToHloModule(*module, non_tuple_options));
  ASSERT_TRUE(non_tuple_hlo_module->has_spmd_parameters_shardings());
  ASSERT_EQ(non_tuple_hlo_module->spmd_parameters_shardings().size(), 3);
  EXPECT_EQ(non_tuple_hlo_module->spmd_parameters_shardings()[0].ToString(),
            kShardingA);
  EXPECT_EQ(non_tuple_hlo_module->spmd_parameters_shardings()[1].ToString(),
            kShardingA);
  EXPECT_EQ(non_tuple_hlo_module->spmd_parameters_shardings()[2].ToString(),
            kShardingB);
}

std::string MakeModuleSourceWithSharedNamedShardings(int num_ops) {
  static constexpr std::array<absl::string_view, 16> kShardings = {
      "{mesh['data'=64,'fsdp'=16,'tensor'=8,'context'=2], "
      "[{'data', 'fsdp'}, {'tensor'}]}",
      "{mesh['data'=64,'fsdp'=16,'tensor'=8,'context'=2], "
      "[{'data'}, {'fsdp', 'tensor'}]}",
      "{mesh['data'=64,'fsdp'=16,'tensor'=8,'context'=2], "
      "[{'data', 'context'}, {'tensor'}]}",
      "{mesh['data'=64,'fsdp'=16,'tensor'=8,'context'=2], "
      "[{'fsdp'}, {'tensor'}]}",
      "{mesh['data'=64,'fsdp'=16,'tensor'=8,'context'=2], "
      "[{'tensor'}, {'fsdp'}]}",
      "{mesh['data'=64,'fsdp'=16,'tensor'=8,'context'=2], "
      "[{'data'}, {'tensor'}]}",
      "{mesh['data'=64,'fsdp'=16,'tensor'=8,'context'=2], "
      "[{'data', 'fsdp'}, {}]}",
      "{mesh['data'=64,'fsdp'=16,'tensor'=8,'context'=2], "
      "[{}, {'fsdp', 'tensor'}]}",
      "{mesh['data'=64,'fsdp'=16,'tensor'=8,'context'=2], [{'data'}, {}]}",
      "{mesh['data'=64,'fsdp'=16,'tensor'=8,'context'=2], [{}, {'tensor'}]}",
      "{mesh['data'=64,'fsdp'=16,'tensor'=8,'context'=2], "
      "[{'fsdp', 'context'}, {'tensor'}]}",
      "{mesh['data'=64,'fsdp'=16,'tensor'=8,'context'=2], "
      "[{'data':(8)8, 'fsdp'}, {'tensor'}]}",
      "{mesh['data'=64,'fsdp'=16,'tensor'=8,'context'=2], "
      "[{'data'}, {'tensor':(2)4, 'context'}]}",
      "{mesh['data'=64,'fsdp'=16,'tensor'=8,'context'=2], "
      "[{'data', ?}, {'tensor', ?}]}",
      "{mesh['data'=64,'fsdp'=16,'tensor'=8,'context'=2], "
      "[{'fsdp'}, {}], replicated={'data', 'tensor'}}",
      "{mesh['data'=64,'fsdp'=16,'tensor'=8,'context'=2], replicated}",
  };
  constexpr int kNumArgs = 64;
  constexpr absl::string_view kTensorType = "tensor<16384x16384xf32>";

  std::string source = "func.func @main(";
  for (int i = 0; i < kNumArgs; ++i) {
    if (i > 0) {
      absl::StrAppend(&source, ", ");
    }
    absl::StrAppend(&source, "%arg", i, ": ", kTensorType,
                    " {mhlo.sharding = \"", kShardings[i % kShardings.size()],
                    "\"}");
  }
  absl::StrAppend(&source, ") -> (");
  for (int i = 0; i < kNumArgs; ++i) {
    if (i > 0) {
      absl::StrAppend(&source, ", ");
    }
    absl::StrAppend(&source, kTensorType, " {mhlo.sharding = \"",
                    kShardings[(i + 3) % kShardings.size()], "\"}");
  }
  absl::StrAppend(&source, ") {\n");
  for (int i = 0; i < num_ops; ++i) {
    std::string operand = i < kNumArgs ? absl::StrCat("%arg", i)
                                       : absl::StrCat("%v", i - kNumArgs);
    absl::StrAppend(&source, "  %v", i, " = stablehlo.negate ", operand,
                    " {mhlo.sharding = \"", kShardings[i % kShardings.size()],
                    "\"} : ", kTensorType, "\n");
  }
  absl::StrAppend(&source, "  func.return ");
  for (int i = 0; i < kNumArgs; ++i) {
    if (i > 0) {
      absl::StrAppend(&source, ", ");
    }
    int val_idx = num_ops - kNumArgs + i;
    absl::StrAppend(&source, "%v", val_idx);
  }
  absl::StrAppend(&source, " : ");
  for (int i = 0; i < kNumArgs; ++i) {
    if (i > 0) {
      absl::StrAppend(&source, ", ");
    }
    absl::StrAppend(&source, kTensorType);
  }
  absl::StrAppend(&source, "\n}\n");
  return source;
}

void BM_ExportWithSharedNamedShardings(benchmark::State& state) {
  const int num_ops = state.range(0);
  const std::string mlir_source =
      MakeModuleSourceWithSharedNamedShardings(num_ops);

  mlir::DialectRegistry registry;
  xla::RegisterMlirToHloDependentDialects(registry);
  mlir::MLIRContext context(registry);
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(mlir_source, &context);
  CHECK(module);

  MlirToHloConversionOptions options;
  {
    xla::HloProto warmup_proto;
    CHECK_OK(ConvertMlirHloToHlo(*module, &warmup_proto, options));
    int sharded_negates = 0;
    for (const xla::HloInstructionProto& instr :
         warmup_proto.hlo_module().computations(0).instructions()) {
      if (instr.opcode() == "negate" && instr.has_sharding()) {
        ++sharded_negates;
      }
    }
    CHECK_EQ(sharded_negates, num_ops);
  }

  for (auto s : state) {
    xla::HloProto hlo_proto;
    CHECK_OK(ConvertMlirHloToHlo(*module, &hlo_proto, options));
    benchmark::DoNotOptimize(hlo_proto);
  }
  state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * num_ops);
}

BENCHMARK(BM_ExportWithSharedNamedShardings)
    ->Arg(1000)
    ->Arg(4000)
    ->Arg(16000)
    ->Unit(benchmark::kMillisecond);

}  // namespace
}  // namespace mlir
