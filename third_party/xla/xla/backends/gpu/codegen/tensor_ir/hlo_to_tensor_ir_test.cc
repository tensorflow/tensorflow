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

#include "xla/backends/gpu/codegen/tensor_ir/hlo_to_tensor_ir.h"

#include <cstdint>

#include "tensor_ir/Dialect/TensorIR.h"
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Support/LLVM.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/service/llvm_ir/llvm_util.h"

namespace xla::gpu::tensor_ir {
namespace {

using ::absl_testing::IsOk;
using ::absl_testing::StatusIs;
using ::testing::HasSubstr;
using ::testing::Not;

namespace tir = ::mlir::nv_tensor_ir;

class HloToTensorIrTest : public HloHardwareIndependentTestBase {
 protected:
  HloToTensorIrTest() {
    context_.loadDialect<tir::TensorIRDialect, mlir::func::FuncDialect,
                         mlir::stablehlo::StablehloDialect>();
    module_ = llvm_ir::CreateMlirModuleOp(mlir::UnknownLoc::get(&context_));
  }

  mlir::MLIRContext context_;
  mlir::OwningOpRef<mlir::ModuleOp> module_;
};

// Returns the number of ops of type `OpTy` nested inside `graph`.
template <typename OpTy>
int64_t CountOps(tir::GraphOp graph) {
  int64_t count = 0;
  graph.walk([&](OpTy op) { ++count; });
  return count;
}

TEST_F(HloToTensorIrTest, ElementwiseAddF32) {
  constexpr absl::string_view kHloText = R"(
    HloModule test_module

    ENTRY main {
      p0 = f32[4,8]{1,0} parameter(0)
      p1 = f32[4,8]{1,0} parameter(1)
      ROOT add = f32[4,8]{1,0} add(p0, p1)
    }
  )";
  ASSERT_OK_AND_ASSIGN(auto hlo_module,
                       ParseAndReturnUnverifiedModule(kHloText));

  ASSERT_OK_AND_ASSIGN(
      tir::GraphOp graph,
      ImportAndLegalizeComputation(*hlo_module->entry_computation(), *module_));

  // The func.func has been rewritten into a single nv_tensor_ir.graph.
  EXPECT_TRUE(module_->getOps<mlir::func::FuncOp>().empty());

  auto graph_type = graph.getFunctionType();
  ASSERT_EQ(graph_type.getNumInputs(), 2);
  ASSERT_EQ(graph_type.getNumResults(), 1);
  auto arg_type =
      mlir::dyn_cast<mlir::RankedTensorType>(graph_type.getInput(0));
  ASSERT_TRUE(arg_type != nullptr);
  EXPECT_EQ(arg_type.getShape(), llvm::ArrayRef<int64_t>({4, 8}));
  EXPECT_TRUE(arg_type.getElementType().isF32());

  EXPECT_EQ(CountOps<tir::AddOp>(graph), 1);
  EXPECT_EQ(CountOps<tir::ResultsOp>(graph), 1);
}

TEST_F(HloToTensorIrTest, ElementwiseMulS32UsesSignedIntegers) {
  constexpr absl::string_view kHloText = R"(
    HloModule test_module

    ENTRY main {
      p0 = s32[4,8]{1,0} parameter(0)
      p1 = s32[4,8]{1,0} parameter(1)
      ROOT mul = s32[4,8]{1,0} multiply(p0, p1)
    }
  )";
  ASSERT_OK_AND_ASSIGN(auto hlo_module,
                       ParseAndReturnUnverifiedModule(kHloText));

  ASSERT_OK_AND_ASSIGN(
      tir::GraphOp graph,
      ImportAndLegalizeComputation(*hlo_module->entry_computation(), *module_));

  auto graph_type = graph.getFunctionType();
  ASSERT_EQ(graph_type.getNumInputs(), 2);
  ASSERT_EQ(graph_type.getNumResults(), 1);

  // The signedness bridge: StableHLO uses signless i32, nv_tensor_ir requires
  // signed si32.
  for (mlir::Type type : graph_type.getInputs()) {
    auto tensor_type = mlir::dyn_cast<mlir::RankedTensorType>(type);
    ASSERT_TRUE(tensor_type != nullptr);
    EXPECT_TRUE(tensor_type.getElementType().isSignedInteger(32))
        << "expected si32 element type, got signless/unsigned integer";
  }
  auto result_type =
      mlir::dyn_cast<mlir::RankedTensorType>(graph_type.getResult(0));
  ASSERT_TRUE(result_type != nullptr);
  EXPECT_TRUE(result_type.getElementType().isSignedInteger(32));

  // The block arguments must be converted as well.
  for (mlir::BlockArgument arg : graph.getGraphBody().front().getArguments()) {
    auto tensor_type = mlir::dyn_cast<mlir::RankedTensorType>(arg.getType());
    ASSERT_TRUE(tensor_type != nullptr);
    EXPECT_TRUE(tensor_type.getElementType().isSignedInteger(32));
  }

  EXPECT_EQ(CountOps<tir::MulOp>(graph), 1);
}

TEST_F(HloToTensorIrTest, GraphIsNamedAfterTheComputation) {
  constexpr absl::string_view kHloText = R"(
    HloModule test_module

    ENTRY tensor_ir_fusion {
      p0 = f32[4]{0} parameter(0)
      ROOT n = f32[4]{0} negate(p0)
    }
  )";
  ASSERT_OK_AND_ASSIGN(auto hlo_module,
                       ParseAndReturnUnverifiedModule(kHloText));

  ASSERT_OK_AND_ASSIGN(
      tir::GraphOp graph,
      ImportAndLegalizeComputation(*hlo_module->entry_computation(), *module_));

  EXPECT_EQ(graph.getSymName().str(), "tensor_ir_fusion");
}

TEST_F(HloToTensorIrTest, GraphNameIsSanitized) {
  // HLO names may contain '.' and '-', which are not valid in a kernel symbol.
  constexpr absl::string_view kHloText = R"(
    HloModule test_module

    ENTRY fused-computation.1 {
      p0 = f32[4]{0} parameter(0)
      ROOT n = f32[4]{0} negate(p0)
    }
  )";
  ASSERT_OK_AND_ASSIGN(auto hlo_module,
                       ParseAndReturnUnverifiedModule(kHloText));

  ASSERT_OK_AND_ASSIGN(
      tir::GraphOp graph,
      ImportAndLegalizeComputation(*hlo_module->entry_computation(), *module_));

  EXPECT_EQ(graph.getSymName().str(), "fused_computation_1");
}

TEST_F(HloToTensorIrTest, UnsupportedOpFailsAndNamesTheOp) {
  // stablehlo.xor has no nv_tensor_ir equivalent.
  constexpr absl::string_view kHloText = R"(
    HloModule test_module

    ENTRY main {
      p0 = s32[4,8]{1,0} parameter(0)
      p1 = s32[4,8]{1,0} parameter(1)
      ROOT x = s32[4,8]{1,0} xor(p0, p1)
    }
  )";
  ASSERT_OK_AND_ASSIGN(auto hlo_module,
                       ParseAndReturnUnverifiedModule(kHloText));

  auto graph =
      ImportAndLegalizeComputation(*hlo_module->entry_computation(), *module_);
  EXPECT_THAT(graph, Not(IsOk()));
  EXPECT_THAT(graph.status().message(), HasSubstr("stablehlo.xor"));
}

TEST_F(HloToTensorIrTest, TupleRootFailsCleanly) {
  constexpr absl::string_view kHloText = R"(
    HloModule test_module

    ENTRY main {
      p0 = f32[4]{0} parameter(0)
      p1 = f32[4]{0} parameter(1)
      a = f32[4]{0} add(p0, p1)
      m = f32[4]{0} multiply(p0, p1)
      ROOT t = (f32[4]{0}, f32[4]{0}) tuple(a, m)
    }
  )";
  ASSERT_OK_AND_ASSIGN(auto hlo_module,
                       ParseAndReturnUnverifiedModule(kHloText));

  auto graph =
      ImportAndLegalizeComputation(*hlo_module->entry_computation(), *module_);
  EXPECT_THAT(graph, Not(IsOk()));
  EXPECT_THAT(graph.status().message(), HasSubstr("stablehlo.tuple"));
}

TEST_F(HloToTensorIrTest, NullModuleIsRejected) {
  constexpr absl::string_view kHloText = R"(
    HloModule test_module

    ENTRY main {
      p0 = f32[4]{0} parameter(0)
      ROOT n = f32[4]{0} negate(p0)
    }
  )";
  ASSERT_OK_AND_ASSIGN(auto hlo_module,
                       ParseAndReturnUnverifiedModule(kHloText));

  EXPECT_THAT(ImportAndLegalizeComputation(*hlo_module->entry_computation(),
                                           mlir::ModuleOp()),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST_F(HloToTensorIrTest, CreatesAndOwnsTheModule) {
  constexpr absl::string_view kHloText = R"(
    HloModule test_module

    ENTRY main {
      p0 = f32[4]{0} parameter(0)
      ROOT n = f32[4]{0} negate(p0)
    }
  )";
  ASSERT_OK_AND_ASSIGN(auto hlo_module,
                       ParseAndReturnUnverifiedModule(kHloText));

  ASSERT_OK_AND_ASSIGN(mlir::OwningOpRef<mlir::ModuleOp> module,
                       ImportAndLegalizeComputation(
                           *hlo_module->entry_computation(), &context_));

  ASSERT_TRUE(module);
  llvm::SmallVector<tir::GraphOp> graphs(module->getOps<tir::GraphOp>());
  ASSERT_EQ(graphs.size(), 1);
  EXPECT_EQ(CountOps<tir::NegOp>(graphs.front()), 1);
  EXPECT_TRUE(module->getOps<mlir::func::FuncOp>().empty());
}

TEST_F(HloToTensorIrTest, NullContextIsRejected) {
  constexpr absl::string_view kHloText = R"(
    HloModule test_module

    ENTRY main {
      p0 = f32[4]{0} parameter(0)
      ROOT n = f32[4]{0} negate(p0)
    }
  )";
  ASSERT_OK_AND_ASSIGN(auto hlo_module,
                       ParseAndReturnUnverifiedModule(kHloText));

  EXPECT_THAT(ImportAndLegalizeComputation(*hlo_module->entry_computation(),
                                           /*context=*/nullptr),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

}  // namespace
}  // namespace xla::gpu::tensor_ir
