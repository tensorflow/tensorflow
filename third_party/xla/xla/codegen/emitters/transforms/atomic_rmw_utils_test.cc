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

#include "xla/codegen/emitters/transforms/atomic_rmw_utils.h"

#include <optional>
#include <string>

#include <gtest/gtest.h>
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/IR/ArithDialect.h"
#include "mlir/Dialect/Func/IR/FuncDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Value.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/Passes.h"
#include "xla/codegen/emitters/ir/xla_dialect.h"
#include "xla/codegen/emitters/ir/xla_ops.h"

namespace xla::emitters {
namespace {

class AtomicRMWUtilsTest : public ::testing::Test {
 protected:
  AtomicRMWUtilsTest() {
    context_.loadDialect<mlir::arith::ArithDialect, mlir::func::FuncDialect,
                         xla::XlaDialect>();
  }

  mlir::OwningOpRef<mlir::ModuleOp> ParseModule(absl::string_view ir) {
    return mlir::parseSourceString<mlir::ModuleOp>(
        llvm::StringRef(ir.data(), ir.size()), &context_);
  }

  AtomicRMWOp FindAtomicRMW(mlir::ModuleOp module) {
    AtomicRMWOp found;
    module.walk([&](AtomicRMWOp op) { found = op; });
    return found;
  }

  mlir::MLIRContext context_;
};

TEST_F(AtomicRMWUtilsTest, IdentityAndStoreBodiesReturnNullopt) {
  auto identity_module = ParseModule(R"mlir(
    func.func @id(%in: tensor<4xf32>, %i: index) -> tensor<4xf32> {
      %ret = xla.atomic_rmw %in[%i] : tensor<4xf32> {
      ^bb0(%current: f32):
        xla.yield %current : f32
      }
      return %ret : tensor<4xf32>
    }
  )mlir");
  ASSERT_TRUE(identity_module);
  EXPECT_EQ(GetAtomicModifierParameters(FindAtomicRMW(*identity_module)),
            std::nullopt);

  auto overwrite_module = ParseModule(R"mlir(
    func.func @store(%in: tensor<4xf32>, %i: index, %x: f32) -> tensor<4xf32> {
      %ret = xla.atomic_rmw %in[%i] : tensor<4xf32> {
      ^bb0(%current: f32):
        xla.yield %x : f32
      }
      return %ret : tensor<4xf32>
    }
  )mlir");
  ASSERT_TRUE(overwrite_module);
  EXPECT_EQ(GetAtomicModifierParameters(FindAtomicRMW(*overwrite_module)),
            std::nullopt);
}

TEST_F(AtomicRMWUtilsTest, MatchesCommutativeAndNonCommutativeOps) {
  auto commuted_add = ParseModule(R"mlir(
    func.func @add(%in: tensor<4xf32>, %i: index, %x: f32) -> tensor<4xf32> {
      %ret = xla.atomic_rmw %in[%i] : tensor<4xf32> {
      ^bb0(%current: f32):
        %v = arith.addf %x, %current : f32
        xla.yield %v : f32
      }
      return %ret : tensor<4xf32>
    }
  )mlir");
  ASSERT_TRUE(commuted_add);
  std::optional<AtomicModifierParameters> add_params =
      GetAtomicModifierParameters(FindAtomicRMW(*commuted_add));
  ASSERT_TRUE(add_params.has_value());
  EXPECT_EQ(add_params->kind, AtomicRMWKind::kFAdd);
  auto add_func = mlir::cast<mlir::func::FuncOp>(commuted_add->front());
  EXPECT_EQ(mlir::dyn_cast<mlir::Value>(add_params->modifier),
            add_func.getArgument(2));

  auto subi_module = ParseModule(R"mlir(
    func.func @sub(%in: tensor<4xi32>, %i: index, %x: i32) -> tensor<4xi32> {
      %ret = xla.atomic_rmw %in[%i] : tensor<4xi32> {
      ^bb0(%current: i32):
        %v = arith.subi %current, %x : i32
        xla.yield %v : i32
      }
      return %ret : tensor<4xi32>
    }
  )mlir");
  ASSERT_TRUE(subi_module);
  std::optional<AtomicModifierParameters> sub_params =
      GetAtomicModifierParameters(FindAtomicRMW(*subi_module));
  ASSERT_TRUE(sub_params.has_value());
  EXPECT_EQ(sub_params->kind, AtomicRMWKind::kSub);
  auto sub_func = mlir::cast<mlir::func::FuncOp>(subi_module->front());
  EXPECT_EQ(mlir::dyn_cast<mlir::Value>(sub_params->modifier),
            sub_func.getArgument(2));

  auto swapped_subi = ParseModule(R"mlir(
    func.func @sub_swapped(%in: tensor<4xi32>, %i: index, %x: i32)
        -> tensor<4xi32> {
      %ret = xla.atomic_rmw %in[%i] : tensor<4xi32> {
      ^bb0(%current: i32):
        %v = arith.subi %x, %current : i32
        xla.yield %v : i32
      }
      return %ret : tensor<4xi32>
    }
  )mlir");
  ASSERT_TRUE(swapped_subi);
  EXPECT_EQ(GetAtomicModifierParameters(FindAtomicRMW(*swapped_subi)),
            std::nullopt);

  auto duplicate_current = ParseModule(R"mlir(
    func.func @dup(%in: tensor<4xf32>, %i: index) -> tensor<4xf32> {
      %ret = xla.atomic_rmw %in[%i] : tensor<4xf32> {
      ^bb0(%current: f32):
        %v = arith.addf %current, %current : f32
        xla.yield %v : f32
      }
      return %ret : tensor<4xf32>
    }
  )mlir");
  ASSERT_TRUE(duplicate_current);
  EXPECT_EQ(GetAtomicModifierParameters(FindAtomicRMW(*duplicate_current)),
            std::nullopt);
}

TEST_F(AtomicRMWUtilsTest, IgnoresDeadOpsInBody) {
  auto module = ParseModule(R"mlir(
    func.func @with_dead_ops(%in: tensor<4xi32>, %i: index, %x: i32, %y: i32)
        -> tensor<4xi32> {
      %ret = xla.atomic_rmw %in[%i] : tensor<4xi32> {
      ^bb0(%current: i32):
        %dead0 = arith.muli %current, %y : i32
        %dead1 = arith.subi %dead0, %x : i32
        %v = arith.xori %current, %x : i32
        xla.yield %v : i32
      }
      return %ret : tensor<4xi32>
    }
  )mlir");
  ASSERT_TRUE(module);
  std::optional<AtomicModifierParameters> params =
      GetAtomicModifierParameters(FindAtomicRMW(*module));
  ASSERT_TRUE(params.has_value());
  EXPECT_EQ(params->kind, AtomicRMWKind::kXor);
  auto func = mlir::cast<mlir::func::FuncOp>(module->front());
  EXPECT_EQ(mlir::dyn_cast<mlir::Value>(params->modifier), func.getArgument(2));
}

TEST_F(AtomicRMWUtilsTest, MatchesAllAtomicRMWKinds) {
  struct Case {
    absl::string_view op_mnemonic;
    absl::string_view elem_type;
    AtomicRMWKind expected_kind;
  };
  const Case cases[] = {
      {"arith.addi", "i32", AtomicRMWKind::kAdd},
      {"arith.subi", "i32", AtomicRMWKind::kSub},
      {"arith.maxsi", "i32", AtomicRMWKind::kMaxS},
      {"arith.minsi", "i32", AtomicRMWKind::kMinS},
      {"arith.maxui", "i32", AtomicRMWKind::kMaxU},
      {"arith.minui", "i32", AtomicRMWKind::kMinU},
      {"arith.andi", "i32", AtomicRMWKind::kAnd},
      {"arith.ori", "i32", AtomicRMWKind::kOr},
      {"arith.xori", "i32", AtomicRMWKind::kXor},
      {"arith.addf", "f32", AtomicRMWKind::kFAdd},
      {"arith.subf", "f32", AtomicRMWKind::kFSub},
      {"arith.maximumf", "f32", AtomicRMWKind::kFMax},
      {"arith.minimumf", "f32", AtomicRMWKind::kFMin},
  };
  for (const Case& test_case : cases) {
    std::string ir = absl::StrFormat(
        R"mlir(
          func.func @f(%%in: tensor<4x%2$s>, %%i: index, %%x: %2$s)
              -> tensor<4x%2$s> {
            %%ret = xla.atomic_rmw %%in[%%i] : tensor<4x%2$s> {
            ^bb0(%%current: %2$s):
              %%v = %1$s %%current, %%x : %2$s
              xla.yield %%v : %2$s
            }
            return %%ret : tensor<4x%2$s>
          }
        )mlir",
        test_case.op_mnemonic, test_case.elem_type);
    auto module = ParseModule(ir);
    ASSERT_TRUE(module);
    AtomicRMWOp rmw = FindAtomicRMW(*module);
    ASSERT_TRUE(rmw);
    std::optional<AtomicModifierParameters> params =
        GetAtomicModifierParameters(rmw);
    ASSERT_TRUE(params.has_value());
    EXPECT_EQ(params->kind, test_case.expected_kind);
    auto func = mlir::cast<mlir::func::FuncOp>(module->front());
    EXPECT_EQ(mlir::dyn_cast<mlir::Value>(params->modifier),
              func.getArgument(2));
  }
}

TEST_F(AtomicRMWUtilsTest, WidenedFloatExactAndInexactConstants) {
  auto exact_module = ParseModule(R"mlir(
    func.func @exact(%in: tensor<4xbf16>, %i: index) -> tensor<4xbf16> {
      %c = arith.constant 2.5 : f32
      %ret = xla.atomic_rmw %in[%i] : tensor<4xbf16> {
      ^bb0(%current: bf16):
        %ext = arith.extf %current : bf16 to f32
        %sum = arith.addf %ext, %c : f32
        %res = arith.truncf %sum : f32 to bf16
        xla.yield %res : bf16
      }
      return %ret : tensor<4xbf16>
    }
  )mlir");
  ASSERT_TRUE(exact_module);
  AtomicRMWOp exact_op = FindAtomicRMW(*exact_module);
  std::optional<AtomicModifierParameters> exact_params =
      GetAtomicModifierParameters(exact_op);
  ASSERT_TRUE(exact_params.has_value());
  EXPECT_EQ(exact_params->kind, AtomicRMWKind::kFAdd);
  auto attr = mlir::dyn_cast<mlir::Attribute>(exact_params->modifier);
  ASSERT_TRUE(attr);
  auto float_attr = mlir::dyn_cast<mlir::FloatAttr>(attr);
  ASSERT_TRUE(float_attr);
  EXPECT_TRUE(float_attr.getType().isBF16());
  EXPECT_DOUBLE_EQ(float_attr.getValueAsDouble(), 2.5);

  mlir::OpBuilder builder(exact_op);
  mlir::Value materialized = MaterializeAtomicModifier(
      builder, exact_op.getLoc(), exact_params->modifier);
  EXPECT_TRUE(materialized.getType().isBF16());

  auto inexact_module = ParseModule(R"mlir(
    func.func @inexact(%in: tensor<4xbf16>, %i: index) -> tensor<4xbf16> {
      %c = arith.constant 0.1 : f32
      %ret = xla.atomic_rmw %in[%i] : tensor<4xbf16> {
      ^bb0(%current: bf16):
        %ext = arith.extf %current : bf16 to f32
        %sum = arith.addf %ext, %c : f32
        %res = arith.truncf %sum : f32 to bf16
        xla.yield %res : bf16
      }
      return %ret : tensor<4xbf16>
    }
  )mlir");
  ASSERT_TRUE(inexact_module);
  EXPECT_EQ(GetAtomicModifierParameters(FindAtomicRMW(*inexact_module)),
            std::nullopt);
}

TEST_F(AtomicRMWUtilsTest, WidenedVectorConstant) {
  auto vec_module = ParseModule(R"mlir(
    func.func @vec(%in: tensor<4xbf16>, %i: index) -> tensor<4xbf16> {
      %c = arith.constant dense<[1.5, -2.0]> : vector<2xf32>
      %ret = xla.atomic_rmw %in[%i] : tensor<4xbf16> {
      ^bb0(%current: vector<2xbf16>):
        %ext = arith.extf %current : vector<2xbf16> to vector<2xf32>
        %sub = arith.subf %ext, %c : vector<2xf32>
        %res = arith.truncf %sub : vector<2xf32> to vector<2xbf16>
        xla.yield %res : vector<2xbf16>
      }
      return %ret : tensor<4xbf16>
    }
  )mlir");
  ASSERT_TRUE(vec_module);
  AtomicRMWOp vec_op = FindAtomicRMW(*vec_module);
  std::optional<AtomicModifierParameters> params =
      GetAtomicModifierParameters(vec_op);
  ASSERT_TRUE(params.has_value());
  EXPECT_EQ(params->kind, AtomicRMWKind::kFSub);
  mlir::OpBuilder builder(vec_op);
  mlir::Value materialized =
      MaterializeAtomicModifier(builder, vec_op.getLoc(), params->modifier);
  auto vec_ty = mlir::dyn_cast<mlir::VectorType>(materialized.getType());
  ASSERT_TRUE(vec_ty);
  EXPECT_EQ(vec_ty.getNumElements(), 2);
  EXPECT_TRUE(vec_ty.getElementType().isBF16());
  auto const_op = materialized.getDefiningOp<mlir::arith::ConstantOp>();
  ASSERT_NE(const_op, nullptr);
  auto dense_attr =
      mlir::dyn_cast<mlir::DenseFPElementsAttr>(const_op.getValue());
  ASSERT_TRUE(dense_attr);
  llvm::SmallVector<double> values;
  for (const llvm::APFloat& elem : dense_attr.getValues<llvm::APFloat>()) {
    values.push_back(elem.convertToDouble());
  }
  ASSERT_EQ(values.size(), 2);
  EXPECT_DOUBLE_EQ(values[0], 1.5);
  EXPECT_DOUBLE_EQ(values[1], -2.0);
}

TEST_F(AtomicRMWUtilsTest, WidenedFloatValueRoundingModeAndPrecision) {
  auto extf_module = ParseModule(R"mlir(
    func.func @extf_val(%in: tensor<4xf16>, %i: index, %x: f16) -> tensor<4xf16> {
      %ext_x = arith.extf %x : f16 to f32
      %ret = xla.atomic_rmw %in[%i] : tensor<4xf16> {
      ^bb0(%current: f16):
        %ext_c = arith.extf %current : f16 to f32
        %sum = arith.addf %ext_c, %ext_x : f32
        %res = arith.truncf %sum to_nearest_even : f32 to f16
        xla.yield %res : f16
      }
      return %ret : tensor<4xf16>
    }
  )mlir");
  ASSERT_TRUE(extf_module);
  std::optional<AtomicModifierParameters> params =
      GetAtomicModifierParameters(FindAtomicRMW(*extf_module));
  ASSERT_TRUE(params.has_value());
  EXPECT_EQ(params->kind, AtomicRMWKind::kFAdd);
  auto func = mlir::cast<mlir::func::FuncOp>(extf_module->front());
  EXPECT_EQ(mlir::dyn_cast<mlir::Value>(params->modifier), func.getArgument(2));

  auto toward_zero_module = ParseModule(R"mlir(
    func.func @rz(%in: tensor<4xf16>, %i: index, %x: f16) -> tensor<4xf16> {
      %ext_x = arith.extf %x : f16 to f32
      %ret = xla.atomic_rmw %in[%i] : tensor<4xf16> {
      ^bb0(%current: f16):
        %ext_c = arith.extf %current : f16 to f32
        %sum = arith.addf %ext_c, %ext_x : f32
        %res = arith.truncf %sum toward_zero : f32 to f16
        xla.yield %res : f16
      }
      return %ret : tensor<4xf16>
    }
  )mlir");
  ASSERT_TRUE(toward_zero_module);
  EXPECT_EQ(GetAtomicModifierParameters(FindAtomicRMW(*toward_zero_module)),
            std::nullopt);

  auto double_round_module = ParseModule(R"mlir(
    func.func @bf16_tf32(%in: tensor<4xbf16>, %i: index, %x: bf16)
        -> tensor<4xbf16> {
      %ext_x = arith.extf %x : bf16 to tf32
      %ret = xla.atomic_rmw %in[%i] : tensor<4xbf16> {
      ^bb0(%current: bf16):
        %ext_c = arith.extf %current : bf16 to tf32
        %sum = arith.addf %ext_c, %ext_x : tf32
        %res = arith.truncf %sum : tf32 to bf16
        xla.yield %res : bf16
      }
      return %ret : tensor<4xbf16>
    }
  )mlir");
  ASSERT_TRUE(double_round_module);
  EXPECT_EQ(GetAtomicModifierParameters(FindAtomicRMW(*double_round_module)),
            std::nullopt);

  auto addf_downward_module = ParseModule(R"mlir(
    func.func @down(%in: tensor<4xf32>, %i: index, %x: f32) -> tensor<4xf32> {
      %ret = xla.atomic_rmw %in[%i] : tensor<4xf32> {
      ^bb0(%current: f32):
        %v = arith.addf %current, %x downward : f32
        xla.yield %v : f32
      }
      return %ret : tensor<4xf32>
    }
  )mlir");
  ASSERT_TRUE(addf_downward_module);
  EXPECT_EQ(GetAtomicModifierParameters(FindAtomicRMW(*addf_downward_module)),
            std::nullopt);

  auto signed_zero_loss_module = ParseModule(R"mlir(
    func.func @fnuz(%in: tensor<4xf4E2M1FN>, %i: index, %x: f4E2M1FN)
        -> tensor<4xf4E2M1FN> {
      %ext_x = arith.extf %x : f4E2M1FN to f8E5M2FNUZ
      %ret = xla.atomic_rmw %in[%i] : tensor<4xf4E2M1FN> {
      ^bb0(%current: f4E2M1FN):
        %ext_c = arith.extf %current : f4E2M1FN to f8E5M2FNUZ
        %m = arith.minimumf %ext_c, %ext_x : f8E5M2FNUZ
        %res = arith.truncf %m : f8E5M2FNUZ to f4E2M1FN
        xla.yield %res : f4E2M1FN
      }
      return %ret : tensor<4xf4E2M1FN>
    }
  )mlir");
  ASSERT_TRUE(signed_zero_loss_module);
  EXPECT_EQ(
      GetAtomicModifierParameters(FindAtomicRMW(*signed_zero_loss_module)),
      std::nullopt);

  auto inf_on_finite_only_module = ParseModule(R"mlir(
    func.func @inf_f4(%in: tensor<4xf4E2M1FN>, %i: index)
        -> tensor<4xf4E2M1FN> {
      %c = arith.constant dense<0x7F800000> : vector<2xf32>
      %ret = xla.atomic_rmw %in[%i] : tensor<4xf4E2M1FN> {
      ^bb0(%current: vector<2xf4E2M1FN>):
        %ext_c = arith.extf %current : vector<2xf4E2M1FN> to vector<2xf32>
        %sum = arith.addf %ext_c, %c : vector<2xf32>
        %res = arith.truncf %sum : vector<2xf32> to vector<2xf4E2M1FN>
        xla.yield %res : vector<2xf4E2M1FN>
      }
      return %ret : tensor<4xf4E2M1FN>
    }
  )mlir");
  ASSERT_TRUE(inf_on_finite_only_module);
  EXPECT_EQ(
      GetAtomicModifierParameters(FindAtomicRMW(*inf_on_finite_only_module)),
      std::nullopt);
}

TEST_F(AtomicRMWUtilsTest, LoopInvariantCodeMotionHoistsModifier) {
  auto module = ParseModule(R"mlir(
    func.func @licm(%in: tensor<4xf32>, %i: index, %x: f32, %y: f32)
        -> tensor<4xf32> {
      %ret = xla.atomic_rmw %in[%i] : tensor<4xf32> {
      ^bb0(%current: f32):
        %prod = arith.mulf %x, %y : f32
        %sum = arith.addf %current, %prod : f32
        xla.yield %sum : f32
      }
      return %ret : tensor<4xf32>
    }
  )mlir");
  ASSERT_TRUE(module);
  // Before hoisting, %prod is defined inside the region so direct matching
  // refuses it.
  EXPECT_EQ(GetAtomicModifierParameters(FindAtomicRMW(*module)), std::nullopt);

  mlir::PassManager pm(&context_);
  pm.addPass(mlir::createLoopInvariantCodeMotionPass());
  ASSERT_TRUE(mlir::succeeded(pm.run(*module)));

  std::optional<AtomicModifierParameters> params =
      GetAtomicModifierParameters(FindAtomicRMW(*module));
  ASSERT_TRUE(params.has_value());
  EXPECT_EQ(params->kind, AtomicRMWKind::kFAdd);
  auto hoisted_val = mlir::dyn_cast<mlir::Value>(params->modifier);
  ASSERT_TRUE(hoisted_val);
  EXPECT_NE(hoisted_val.getDefiningOp<mlir::arith::MulFOp>(), nullptr);
}

TEST_F(AtomicRMWUtilsTest, RejectsSideEffectingOpInBody) {
  auto impure_module = ParseModule(R"mlir(
    func.func private @side_effect(i32) -> i32
    func.func @impure_dead(%in: tensor<4xi32>, %i: index, %x: i32)
        -> tensor<4xi32> {
      %ret = xla.atomic_rmw %in[%i] : tensor<4xi32> {
      ^bb0(%current: i32):
        %call = func.call @side_effect(%current) : (i32) -> i32
        %v = arith.addi %current, %x : i32
        xla.yield %v : i32
      }
      return %ret : tensor<4xi32>
    }
  )mlir");
  ASSERT_TRUE(impure_module);
  EXPECT_EQ(GetAtomicModifierParameters(FindAtomicRMW(*impure_module)),
            std::nullopt);

  auto dead_div_module = ParseModule(R"mlir(
    func.func @trapping_dead(%in: tensor<4xi32>, %i: index, %x: i32, %y: i32)
        -> tensor<4xi32> {
      %ret = xla.atomic_rmw %in[%i] : tensor<4xi32> {
      ^bb0(%current: i32):
        %trap = arith.divsi %current, %y : i32
        %v = arith.addi %current, %x : i32
        xla.yield %v : i32
      }
      return %ret : tensor<4xi32>
    }
  )mlir");
  ASSERT_TRUE(dead_div_module);
  std::optional<AtomicModifierParameters> div_params =
      GetAtomicModifierParameters(FindAtomicRMW(*dead_div_module));
  ASSERT_TRUE(div_params.has_value());
  EXPECT_EQ(div_params->kind, AtomicRMWKind::kAdd);
}

TEST_F(AtomicRMWUtilsTest, RejectsF32WidenedToF64) {
  auto module = ParseModule(R"mlir(
    func.func @f32_in_f64(%in: tensor<4xf32>, %i: index, %x: f32)
        -> tensor<4xf32> {
      %ext_x = arith.extf %x : f32 to f64
      %ret = xla.atomic_rmw %in[%i] : tensor<4xf32> {
      ^bb0(%current: f32):
        %ext_c = arith.extf %current : f32 to f64
        %sum = arith.addf %ext_c, %ext_x : f64
        %res = arith.truncf %sum : f64 to f32
        xla.yield %res : f32
      }
      return %ret : tensor<4xf32>
    }
  )mlir");
  ASSERT_TRUE(module);
  EXPECT_EQ(GetAtomicModifierParameters(FindAtomicRMW(*module)), std::nullopt);
}

}  // namespace
}  // namespace xla::emitters
