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

#include "xla/hlo/translate/hlo_to_mhlo/mlir/passes.h"

#include <memory>
#include <string>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "xla/hlo/testlib/filecheck.h"
#include "xla/hlo/translate/hlo_to_mhlo/mlir/passes.h.inc"
#include "xla/pjrt/mlir_to_hlo.h"

namespace xla {
namespace stablehlo_ext {
namespace {

// Runs the given pass on the given MLIR input and returns whether the output
// matches the expected output (specified as a .mlir file).
absl::StatusOr<bool> TestMlirPassWithFileCheck(
    absl::string_view input, std::unique_ptr<::mlir::Pass> pass) {
  llvm::DebugFlag = true;
  mlir::MLIRContext context;
  context.allowUnregisteredDialects(true);
  ASSIGN_OR_RETURN(auto module, xla::ParseMlirModuleString(input, context));

  // Run pass
  mlir::PassManager pm(module->getContext());
  pm.addNestedPass<mlir::func::FuncOp>(std::move(pass));
  if (mlir::failed(pm.run(*module))) {
    return absl::InternalError("Failed to normalize kernel strings.");
  }

  std::string dumped;
  llvm::raw_string_ostream os(dumped);
  module->print(os);
  return xla::RunFileCheck(dumped, input);
}

TEST(StableHloCanonicalizeFromHloImportTest,
     FlattenTupleInEntryComputationCase1) {
  constexpr absl::string_view kModule = R"mlir(
    // CHECK-LABEL: func @main
    func.func @main(%arg0: tensor<1x224x224x3xf16>, %arg1: tensor<f32>) -> tensor<1x224x224x3xf16> {
      // CHECK: return %arg0 : tensor<1x224x224x3xf16>
      func.return %arg0 : tensor<1x224x224x3xf16>
    }
  )mlir";

  auto status = TestMlirPassWithFileCheck(
      kModule,
      mlir::stablehlo_ext::createStablehloCanonicalizeFromHloImportPass());
  ASSERT_OK(status);
  EXPECT_TRUE(status.value());
}

TEST(StableHloCanonicalizeFromHloImportTest,
     FlattenTupleInEntryComputationCase2) {
  constexpr absl::string_view kModule = R"mlir(
    // CHECK-LABEL: func @main
    // CHECK-SAME: %[[ARG0:.*]]: tensor<1x1xf32>, %[[ARG1:.*]]: tensor<1x8x8x16xf32>) -> (tensor<1024xf32>, tensor<1xf32>)
    func.func @main(%arg0: tensor<1x1xf32>, %arg1: tensor<1x8x8x16xf32>) -> tuple<tensor<1024xf32>, tensor<1xf32>> {
      // CHECK-NEXT: %[[RESHAPE0:.*]] = stablehlo.reshape %[[ARG0]] : (tensor<1x1xf32>) -> tensor<1xf32>
      %0 = stablehlo.reshape %arg0 : (tensor<1x1xf32>) -> tensor<1xf32>
      // CHECK-NEXT: %[[RESHAPE1:.*]] = stablehlo.reshape %[[ARG1]] : (tensor<1x8x8x16xf32>) -> tensor<1024xf32>
      %1 = stablehlo.reshape %arg1 : (tensor<1x8x8x16xf32>) -> tensor<1024xf32>
      %2 = stablehlo.tuple %1, %0 {name = "tuple.374"} : tuple<tensor<1024xf32>, tensor<1xf32>>
      // CHECK-NEXT: return %[[RESHAPE1]], %[[RESHAPE0]] : tensor<1024xf32>, tensor<1xf32>
      return %2 : tuple<tensor<1024xf32>, tensor<1xf32>>
    }
  )mlir";

  auto status = TestMlirPassWithFileCheck(
      kModule,
      mlir::stablehlo_ext::createStablehloCanonicalizeFromHloImportPass());
  ASSERT_OK(status);
  EXPECT_TRUE(status.value());
}

TEST(StableHloCanonicalizeFromHloImportTest,
     FlattenTupleInEntryComputationCase3) {
  constexpr absl::string_view kModule = R"mlir(
    // CHECK-LABEL: func @main
    // CHECK-SAME: () -> (tensor<1xf32>, tensor<1xi32>)
    func.func @main() -> tuple<tensor<1xf32>, tensor<1xi32>> {
      // CHECK-NEXT: %[[TUPLE:.*]] = "test.dummy"() : () -> tuple<tensor<1xf32>, tensor<1xi32>>
      %0 = "test.dummy"() : () -> tuple<tensor<1xf32>, tensor<1xi32>>
      // CHECK-NEXT: %[[RES0:.*]] = stablehlo.get_tuple_element %[[TUPLE]][0] : (tuple<tensor<1xf32>, tensor<1xi32>>) -> tensor<1xf32>
      // CHECK-NEXT: %[[RES1:.*]] = stablehlo.get_tuple_element %[[TUPLE]][1] : (tuple<tensor<1xf32>, tensor<1xi32>>) -> tensor<1xi32>
      // CHECK-NEXT: return %[[RES0]], %[[RES1]] : tensor<1xf32>, tensor<1xi32>
      func.return %0 : tuple<tensor<1xf32>, tensor<1xi32>>
    }
  )mlir";

  auto status = TestMlirPassWithFileCheck(
      kModule,
      mlir::stablehlo_ext::createStablehloCanonicalizeFromHloImportPass());
  ASSERT_OK(status);
  EXPECT_TRUE(status.value());
}

TEST(StableHloCanonicalizeFromHloImportTest,
     FlattenTupleInEntryComputationCase4) {
  constexpr absl::string_view kModule = R"mlir(
    // CHECK-LABEL: func @main
    func.func @main() -> tuple<> {
      %0 = "stablehlo.tuple"() {xla_shape = "()"} : () -> tuple<>
      // CHECK-NEXT: return{{$}}
      func.return %0 : tuple<>
    }
  )mlir";

  auto status = TestMlirPassWithFileCheck(
      kModule,
      mlir::stablehlo_ext::createStablehloCanonicalizeFromHloImportPass());
  ASSERT_OK(status);
  EXPECT_TRUE(status.value());
}

TEST(StableHloCanonicalizeFromHloImportTest,
     FlattenTupleInEntryComputationCase5) {
  constexpr absl::string_view kModule = R"mlir(
    // CHECK-LABEL: func @main
    // CHECK-SAME: %[[ARG0:.*]]: tensor<1024xf32>, %[[ARG1:.*]]: tensor<1xf32>) -> (tensor<1024xf32>, tensor<1xf32>)
    func.func @main(%arg0: tuple<tensor<1024xf32>, tensor<1xf32>>) -> tuple<tensor<1024xf32>, tensor<1xf32>> {
      // CHECK-NEXT: return %[[ARG0]], %[[ARG1]] : tensor<1024xf32>, tensor<1xf32>
      func.return %arg0 : tuple<tensor<1024xf32>, tensor<1xf32>>
    }
  )mlir";

  auto status = TestMlirPassWithFileCheck(
      kModule,
      mlir::stablehlo_ext::createStablehloCanonicalizeFromHloImportPass());
  ASSERT_OK(status);
  EXPECT_TRUE(status.value());
}

TEST(StableHloCanonicalizeFromHloImportTest,
     FlattenTupleInEntryComputationCase6) {
  constexpr absl::string_view kModule = R"mlir(
    // CHECK-LABEL: func @main
    // CHECK-SAME: %[[ARG:.*]]: tensor<1xi8>) -> (tensor<1xf32>, tensor<1xi32>)
    func.func @main(%arg0: tuple<tuple<tensor<1xi8>>>) -> tuple<tuple<tensor<1xf32>>, tensor<1xi32>> {
      // CHECK: %[[T0:.*]] = stablehlo.tuple %[[ARG]] : tuple<tensor<1xi8>>
      // CHECK: %[[T1:.*]] = stablehlo.tuple %[[T0]] : tuple<tuple<tensor<1xi8>>>
      // CHECK: %[[T:.*]] = "test.dummy"(%[[T1]]) : (tuple<tuple<tensor<1xi8>>>) -> tuple<tuple<tensor<1xf32>>, tensor<1xi32>>
      %0 = "test.dummy"(%arg0) : (tuple<tuple<tensor<1xi8>>>) -> tuple<tuple<tensor<1xf32>>, tensor<1xi32>>
      // CHECK: %[[GTE0:.*]] = stablehlo.get_tuple_element %[[T]][0] : (tuple<tuple<tensor<1xf32>>, tensor<1xi32>>) -> tuple<tensor<1xf32>>
      // CHECK: %[[GTE1:.*]] = stablehlo.get_tuple_element %[[T]][1] : (tuple<tuple<tensor<1xf32>>, tensor<1xi32>>) -> tensor<1xi32>
      // CHECK: %[[GTE2:.*]] = stablehlo.get_tuple_element %[[GTE0]][0] : (tuple<tensor<1xf32>>) -> tensor<1xf32>
      // CHECK: return %[[GTE2]], %[[GTE1]] : tensor<1xf32>, tensor<1xi32>
      func.return %0 : tuple<tuple<tensor<1xf32>>, tensor<1xi32>>
    }
  )mlir";

  auto status = TestMlirPassWithFileCheck(
      kModule,
      mlir::stablehlo_ext::createStablehloCanonicalizeFromHloImportPass());
  ASSERT_OK(status);
  EXPECT_TRUE(status.value());
}

TEST(StableHloCanonicalizeFromHloImportTest, FlattenTupleInCustomCall) {
  constexpr absl::string_view kModule = R"mlir(
    // CHECK-LABEL: @custom_call
    // CHECK-SAME: %[[X:.*]]: tensor<6x3xf32>
    func.func @custom_call(%x: tensor<6x3xf32>) -> (tensor<6xf32>, tensor<3xf32>) {
      // CHECK: %[[CALL:.+]]:2 = stablehlo.custom_call @f(%[[X]]) {api_version = 2 : i32} : (tensor<6x3xf32>) -> (tensor<6xf32>, tensor<3xf32>)
      %0 = "stablehlo.custom_call"(%x) {api_version = 2 : i32, call_target_name = "f"} : (tensor<6x3xf32>) -> tuple<tensor<6xf32>, tensor<3xf32>>
      %1 = "stablehlo.get_tuple_element"(%0) {index = 0 : i32} : (tuple<tensor<6xf32>, tensor<3xf32>>) -> tensor<6xf32>
      %2 = "stablehlo.get_tuple_element"(%0) {index = 1 : i32} : (tuple<tensor<6xf32>, tensor<3xf32>>) -> tensor<3xf32>
      // CHECK: return {{.*}} : tensor<6xf32>, tensor<3xf32>
      return %1, %2 : tensor<6xf32>, tensor<3xf32>
    }
  )mlir";

  auto status = TestMlirPassWithFileCheck(
      kModule,
      mlir::stablehlo_ext::createStablehloCanonicalizeFromHloImportPass());
  ASSERT_OK(status);
  EXPECT_TRUE(status.value());
}

TEST(StableHloCanonicalizeFromHloImportTest,
     FlattenTupleInCustomCallTupledOperand) {
  constexpr absl::string_view kModule = R"mlir(
    // CHECK-LABEL: @custom_call_tupled_operand
    // CHECK-SAME: %[[ARG0:.*]]: tuple<tensor<ui32>, tensor<i32>>
    func.func @custom_call_tupled_operand(%arg0: tuple<tensor<ui32>, tensor<i32>>) -> (tensor<i32>, tensor<ui32>, tensor<ui32>) {
      // CHECK-NEXT: %[[C0:.*]] = stablehlo.constant dense<1> : tensor<ui32>
      %0 = stablehlo.constant dense<1> : tensor<ui32>
      // CHECK-NEXT: %[[C1:.*]] = stablehlo.constant dense<10> : tensor<i32>
      %1 = stablehlo.constant dense<10> : tensor<i32>
      %2 = stablehlo.tuple %0, %1, %arg0 : tuple<tensor<ui32>, tensor<i32>, tuple<tensor<ui32>, tensor<i32>>>
      // CHECK-NEXT: %[[VAR1:.*]] = stablehlo.get_tuple_element %[[ARG0]][0]
      // CHECK-NEXT: %[[VAR2:.*]] = stablehlo.get_tuple_element %[[ARG0]][1]
      // CHECK-NEXT: stablehlo.custom_call @ScalarProgramDummyConstant(%[[C0]], %[[C1]], %[[VAR1]], %[[VAR2]])
      %3:2 = stablehlo.custom_call @ScalarProgramDummyConstant(%2) : (tuple<tensor<ui32>, tensor<i32>, tuple<tensor<ui32>, tensor<i32>>>) -> (tensor<ui32>, tensor<ui32>)
      return %1, %3#0, %3#1 : tensor<i32>, tensor<ui32>, tensor<ui32>
    }
  )mlir";

  auto status = TestMlirPassWithFileCheck(
      kModule,
      mlir::stablehlo_ext::createStablehloCanonicalizeFromHloImportPass());
  ASSERT_OK(status);
  EXPECT_TRUE(status.value());
}

TEST(StableHloCanonicalizeFromHloImportTest,
     FlattenTupleInCustomCallTupledResult) {
  constexpr absl::string_view kModule = R"mlir(
    // CHECK-LABEL: @custom_call_tupled_result
    // CHECK-SAME: %[[ARG0:.*]]: tensor<ui32>
    func.func @custom_call_tupled_result(%arg0: tensor<ui32>) -> (tuple<tensor<ui32>, tuple<tensor<ui32>, tensor<i32>>, tensor<i32>>) {
      // CHECK-NEXT: %[[CUSTOM_CALL:.*]]:4 = stablehlo.custom_call @ScalarProgramTupleResult(%[[ARG0]])
      %0 = stablehlo.custom_call @ScalarProgramTupleResult(%arg0) : (tensor<ui32>) -> tuple<tensor<ui32>, tuple<tensor<ui32>, tensor<i32>>, tensor<i32>>
    // CHECK-NEXT: %[[TUPLE1:.*]] = stablehlo.tuple %[[CUSTOM_CALL]]#1, %[[CUSTOM_CALL]]#2
    // CHECK-NEXT: %[[TUPLE2:.*]] = stablehlo.tuple %[[CUSTOM_CALL]]#0, %[[TUPLE1]], %[[CUSTOM_CALL]]#3
    // CHECK-NEXT: return %[[TUPLE2]]
    return %0 : tuple<tensor<ui32>, tuple<tensor<ui32>, tensor<i32>>, tensor<i32>>
    }
  )mlir";

  auto status = TestMlirPassWithFileCheck(
      kModule,
      mlir::stablehlo_ext::createStablehloCanonicalizeFromHloImportPass());
  ASSERT_OK(status);
  EXPECT_TRUE(status.value());
}

TEST(StableHloCanonicalizeFromHloImportTest, TupleUnpackingAndRepacking) {
  constexpr absl::string_view kModule = R"mlir(
    // CHECK-LABEL: func.func @get_tuple_element
    // CHECK-SAME:   ([[ARG0:%.+]]: tensor<f32>, [[ARG1:%.+]]: tensor<i32>, [[ARG2:%.+]]: tuple<tensor<f32>, tensor<f16>>)
    func.func @get_tuple_element(%arg0: tensor<f32>, %arg1: tensor<i32>, %arg2: tuple<tensor<f32>, tensor<f16>>) -> (tensor<f32>, tensor<i32>, tensor<f16>) {
      %t = stablehlo.tuple %arg0, %arg1 : tuple<tensor<f32>, tensor<i32>>
      %a = stablehlo.get_tuple_element %t[0] : (tuple<tensor<f32>, tensor<i32>>) -> tensor<f32>
      %b = stablehlo.get_tuple_element %t[1] : (tuple<tensor<f32>, tensor<i32>>) -> tensor<i32>
      %c = stablehlo.get_tuple_element %arg2[1] : (tuple<tensor<f32>, tensor<f16>>) -> tensor<f16>
      // CHECK:      [[GTE:%.+]] = stablehlo.get_tuple_element [[ARG2]][1] : (tuple<tensor<f32>, tensor<f16>>) -> tensor<f16>
      // CHECK-NEXT: return [[ARG0]], [[ARG1]], [[GTE]]
      return %a, %b, %c : tensor<f32>, tensor<i32>, tensor<f16>
    }
  )mlir";

  auto status = TestMlirPassWithFileCheck(
      kModule,
      mlir::stablehlo_ext::createStablehloCanonicalizeFromHloImportPass());
  ASSERT_OK(status);
  EXPECT_TRUE(status.value());
}

TEST(StableHloCanonicalizeFromHloImportTest, UnpackRepackSameTuple) {
  constexpr absl::string_view kModule = R"mlir(
    // CHECK-LABEL: unpack_repack_same_tuple
    // CHECK-SAME: ([[ARG0:%.*]]: tuple<tensor<i32>, !stablehlo.token, tensor<f32>>)
    func.func @unpack_repack_same_tuple(%arg0: tuple<tensor<i32>, !stablehlo.token, tensor<f32>>) -> tuple<tensor<i32>, !stablehlo.token, tensor<f32>> {
      %0 = stablehlo.get_tuple_element %arg0[0] : (tuple<tensor<i32>, !stablehlo.token, tensor<f32>>) -> tensor<i32>
      %1 = stablehlo.get_tuple_element %arg0[1] : (tuple<tensor<i32>, !stablehlo.token, tensor<f32>>) -> !stablehlo.token
      %2 = stablehlo.get_tuple_element %arg0[2] : (tuple<tensor<i32>, !stablehlo.token, tensor<f32>>) -> tensor<f32>
      %3 = stablehlo.tuple %0, %1, %2 : tuple<tensor<i32>, !stablehlo.token, tensor<f32>>
      // CHECK: return [[ARG0]]
      return %3 : tuple<tensor<i32>, !stablehlo.token, tensor<f32>>
    }
  )mlir";

  auto status = TestMlirPassWithFileCheck(
      kModule,
      mlir::stablehlo_ext::createStablehloCanonicalizeFromHloImportPass());
  ASSERT_OK(status);
  EXPECT_TRUE(status.value());
}

TEST(StableHloCanonicalizeFromHloImportTest,
     UnpackRepackSameTupleSingleElement) {
  constexpr absl::string_view kModule = R"mlir(
    // CHECK-LABEL: unpack_repack_same_tuple_single_element
    // CHECK-SAME: ([[ARG0:%.*]]: tuple<tensor<i32>>)
    func.func @unpack_repack_same_tuple_single_element(%arg0: tuple<tensor<i32>>) -> tuple<tensor<i32>> {
      %0 = stablehlo.get_tuple_element %arg0[0] : (tuple<tensor<i32>>) -> tensor<i32>
      %1 = stablehlo.tuple %0 : tuple<tensor<i32>>
      // CHECK: return [[ARG0]]
      return %1 : tuple<tensor<i32>>
    }
  )mlir";

  auto status = TestMlirPassWithFileCheck(
      kModule,
      mlir::stablehlo_ext::createStablehloCanonicalizeFromHloImportPass());
  ASSERT_OK(status);
  EXPECT_TRUE(status.value());
}

TEST(StableHloCanonicalizeFromHloImportTest,
     WhileOpConstantCaptureWithOriginalValue) {
  constexpr absl::string_view kModule = R"mlir(
    // CHECK-LABEL: while_op_constant_capture
    // CHECK-SAME: ([[ARG0:%.*]]: tensor<10xf32>)
    func.func @while_op_constant_capture(%arg0: tensor<10xf32>) -> (tensor<10xf32>) {
      %c = stablehlo.constant dense<1> : tensor<i32>
      %c_0 = stablehlo.constant dense<10> : tensor<i32>
      %c_1 = stablehlo.constant dense<0> : tensor<i32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<10xf32>
      // CHECK: stablehlo.while(%iterArg = %c_1, %iterArg_2 = %0) : tensor<i32>, tensor<10xf32> attributes {mhlo.original_value = "({\22while.192\22 {1}}, {\22while.192\22 {2}})"}
      %1:3 = stablehlo.while(%iterArg = %arg0, %iterArg_2 = %c_1, %iterArg_3 = %0) : tensor<10xf32>, tensor<i32>, tensor<10xf32> attributes {mhlo.original_value = "{({\22while.192\22 {0}}, {\22while.192\22 {1}}, {\22while.192\22 {2}})}"}
        cond {
          %2 = stablehlo.compare  LT, %iterArg_2, %c_0,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
          stablehlo.return %2 : tensor<i1>
        } do {
          %2 = stablehlo.dynamic_slice %iterArg, %iterArg_2, sizes = [1] : (tensor<10xf32>, tensor<i32>) -> tensor<1xf32>
          %3 = stablehlo.reshape %2 : (tensor<1xf32>) -> tensor<f32>
          %4 = stablehlo.sine %3 : tensor<f32>
          %5 = stablehlo.broadcast_in_dim %4, dims = [] : (tensor<f32>) -> tensor<1xf32>
          %6 = stablehlo.dynamic_update_slice %iterArg_3, %5, %iterArg_2 : (tensor<10xf32>, tensor<1xf32>, tensor<i32>) -> tensor<10xf32>
          %7 = stablehlo.add %iterArg_2, %c : tensor<i32>
          stablehlo.return %iterArg, %7, %6 : tensor<10xf32>, tensor<i32>, tensor<10xf32>
      }
      return %1#2 : tensor<10xf32>
    }
  )mlir";

  auto status = TestMlirPassWithFileCheck(
      kModule,
      mlir::stablehlo_ext::createStablehloCanonicalizeFromHloImportPass());
  ASSERT_OK(status);
  EXPECT_TRUE(status.value());
}

}  // namespace
}  // namespace stablehlo_ext
}  // namespace xla
