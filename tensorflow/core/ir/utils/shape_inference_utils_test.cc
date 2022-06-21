/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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

#include "tensorflow/core/ir/utils/shape_inference_utils.h"

#include <vector>

#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/Parser/Parser.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/ir/dialect.h"
#include "tensorflow/core/ir/ops.h"
#include "tensorflow/core/ir/tf_op_wrapper.h"
#include "tensorflow/core/platform/test.h"

using tensorflow::shape_inference::DimensionHandle;
using tensorflow::shape_inference::InferenceContext;
using tensorflow::shape_inference::ShapeHandle;

namespace mlir {
namespace tfg {
namespace {
// These operations cover the most of logics used in the
// InferReturnTypeComponentsForTFOp.
const char *const code = R"mlir(
  tfg.func @test(%arg : tensor<32x?x256x4xi32> {tfg.name = "arg"}) -> (tensor<2x2xf32>) {
    %Placeholder, %ctl = Placeholder name("placeholder") {dtype = f32, shape = #tf_type.shape<>} : () -> (tensor<f32>)
    %Const, %ctl_0 = Const name("c0") {dtype = f32, value = dense<1.000000e+00> : tensor<2x2xf32>} : () -> (tensor<2x2xf32>)
    %Const_1, %ctl_2 = Const name("c1") {dtype = f32, value = dense<2.000000e+00> : tensor<2x2xf32>} : () -> (tensor<2x2xf32>)
    %IdentityN:3, %ctl_3 = IdentityN(%Const, %Placeholder, %Const_1) name("id_n") {T = [f32, f32, f32]} : (tensor<2x2xf32>, tensor<f32>, tensor<2x2xf32>) -> (tensor<2x2xf32>, tensor<f32>, tensor<2x2xf32>)
    %Identity, %ctl_6 = Identity(%IdentityN#1) name("id1") {T = f32} : (tensor<f32>) -> (tensor<f32>)
    %Add, %ctl_7 = Add(%Const, %IdentityN#1) name("add") {T = f32} : (tensor<2x2xf32>, tensor<f32>) -> (tensor<2x2xf32>)
    %Const_1000, %ctl_9 = Const name("c1000") {dtype = i32, value = dense<1000> : tensor<i32>} : () -> (tensor<i32>)
    %Const_2, %ctl_10 = Const name("c2") {dtype = i32, value = dense<0> : tensor<i32>} : () -> (tensor<i32>)
    %Const_3, %ctl_11 = Const name("c3") {dtype = i32, value = dense<1> : tensor<i32>} : () -> (tensor<i32>)
    %Range, %ctl_range = Range(%Const_2, %Const_1000, %Const_3) name("range") {Tidx = i32} : (tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1000xi32>
    %Const_4, %ctl_12 = Const name("c4") {dtype = i32, value = dense<[32, -1, 4]> : tensor<3xi32>} : () -> (tensor<3xi32>)
    %Reshape, %ctl_13 = Reshape(%arg, %Const_4) name("reshape") {T = i32} : (tensor<32x?x256x4xi32>, tensor<3xi32>) -> tensor<32x?x4xi32>
    %Const_5, %ctl_14 = Const name("TensorListReserve/num_elements") {dtype = i32, value = dense<3> : tensor<i32>} : () -> (tensor<i32>)
    %Const_6, %ctl_15 = Const name("TensorListReserve/element_shape") {dtype = i32, value = dense<2> : tensor<2xi32>} : () -> (tensor<2xi32>)
    %TensorListReserve, %ctl_16 = TensorListReserve(%Const_6, %Const_5) name("TensorListReserve") {element_dtype = f32, shape_type = i32} : (tensor<2xi32>, tensor<i32>) -> (tensor<!tf_type.variant<tensor<2x2xf32>>>)
    %Const_7, %ctl_17 = Const name("index") {dtype = i32, value = dense<0> : tensor<i32>} : () -> (tensor<i32>)
    %Const_8, %ctl_18 = Const name("item") {dtype = f32, value = dense<[[1.000000e+00, 2.000000e+00], [3.000000e+00, 4.000000e+00]]> : tensor<2x2xf32>} : () -> (tensor<2x2xf32>)
    %TensorListSetItem, %ctl_19 = TensorListSetItem(%TensorListReserve, %Const_7, %Const_8) name("TensorListSetItem") {element_dtype = f32} : (tensor<!tf_type.variant<tensor<2x2xf32>>>, tensor<i32>, tensor<2x2xf32>) -> (tensor<!tf_type.variant<tensor<2x2xf32>>>)
    return (%Const_1) : tensor<2x2xf32>
  }
)mlir";
}  // namespace

class ShapeInferenceTest : public ::testing::Test {
 protected:
  using OpShapeInfo = SmallVector<ShapedTypeComponents>;

  ShapeInferenceTest() {
    context_.getOrLoadDialect<tfg::TFGraphDialect>();
    module_ = mlir::parseSourceString<mlir::ModuleOp>(code, &context_);
    assert(module_);
  }

  // Given the range of ops and the inferred results, verify if the inferred
  // result matches the specified result type of the operation.
  template <typename OpRange>
  void VerifyInferredShapes(OpRange &&ops,
                            SmallVector<OpShapeInfo> &inferred_result,
                            bool check_type) {
    for (auto it : llvm::zip(ops, inferred_result)) {
      Operation &op = std::get<0>(it);
      OpShapeInfo &info = std::get<1>(it);

      EXPECT_EQ(op.getNumResults() - 1, info.size());
      for (int i = 0; i < op.getNumResults() - 1; ++i) {
        ShapedType shape = op.getResultTypes()[i].cast<ShapedType>();
        EXPECT_EQ(shape.hasRank(), info[i].hasRank());
        if (shape.hasRank()) EXPECT_EQ(shape.getShape(), info[i].getDims());
        if (check_type)
          EXPECT_EQ(shape.getElementType(), info[i].getElementType());
      }
    }
  }

  ModuleOp GetModule() { return module_.get(); }
  MLIRContext *GetContext() { return &context_; }

 private:
  MLIRContext context_;
  OwningOpRef<ModuleOp> module_;
};

// In InferReturnTypeComponentsForTFOp, it requires 4 callbacks to help with the
// shape inference and result type determination. The 4 callbacks are
// `operand_as_constant_fn`, `op_result_as_shape_fn`, `result_element_type_fn`
// and `get_attr_values_fn`. The callback may have different implementations
// that depends on the dialect. In this test, we provide the callbacks for TFG
// operations.
TEST_F(ShapeInferenceTest, TestShapeAndTypeInference) {
  // `value` attr contains the tensor information.
  auto operand_as_constant_fn = [](Value operand) -> Attribute {
    return operand.getDefiningOp()->getAttr("value");
  };

  // `value` attr contains the tensor information and it's a DenseElementAttr.
  auto op_result_as_shape_fn = [](InferenceContext &ic,
                                  OpResult op_result) -> ShapeHandle {
    auto rt = op_result.getType().dyn_cast<RankedTensorType>();
    if (!rt || rt.getRank() != 1 || !rt.hasStaticShape()) return {};

    std::vector<DimensionHandle> dims(rt.getDimSize(0), ic.UnknownDim());
    auto attr =
        op_result.getDefiningOp()->getAttrOfType<DenseElementsAttr>("value");
    for (auto element : llvm::enumerate(attr.getValues<APInt>()))
      dims[element.index()] = ic.MakeDim(element.value().getSExtValue());
    return ic.MakeShape(dims);
  };

  GraphFuncOp func = GetModule().lookupSymbol<GraphFuncOp>("test");
  ASSERT_TRUE(func);
  Block &block = *func.body().begin();

  SmallVector<SmallVector<ShapedTypeComponents>> all_results;

  for (Operation &op : block.without_terminator()) {
    // InferenceContext only infers the shape,
    // `InferReturnTypeComponentsForTFOp`uses this callback to get the type
    // information.
    auto result_element_type_fn = [&](int idx) -> Type {
      return op.getResult(idx).getType().cast<ShapedType>().getElementType();
    };

    // We use TFG operation so that we don't need to provide
    // `get_attr_values_fn`. It will by default use TFG importer to get the
    // attributes.
    SmallVector<ShapedTypeComponents> results;
    EXPECT_TRUE(InferReturnTypeComponentsForTFOp(
                    op.getLoc(), &op, TFOp(&op).getNonControlOperands(),
                    /*graph_version=*/1010, operand_as_constant_fn,
                    op_result_as_shape_fn, result_element_type_fn,
                    /*get_attr_values_fn=*/nullptr, results)
                    .succeeded());
    all_results.push_back(results);
  }

  VerifyInferredShapes(func.body().begin()->without_terminator(), all_results,
                       /*check_type*/ true);

  // In general, `operand_as_constant_fn` and `op_result_as_shape_fn` may have
  // the similar capability, i.e., they access the definingOp of an operation
  // and get the type and shape information there. So in most cases
  // `operand_as_constant_fn` just covers `op_result_as_shape_fn`.
  // The following intends to test if the `op_result_as_shape_fn` works. In the
  // test case, `Reshape` needs to call `op_result_as_shape_fn` if
  // `operand_as_constant_fn` doesn't give the correct information.

  auto exclude_reshape_operand_as_constant_fn =
      [&](Value operand) -> Attribute {
    Operation *defining_op = operand.getDefiningOp();
    // As mentioned above, return a BoolAttr for Reshape on purpose to make it
    // relies on `op_result_as_shape_fn` which should be able to give correct
    // shape information.
    if (!defining_op || defining_op->getName().getStringRef() == "tfg.Reshape")
      return BoolAttr::get(GetContext(), false);
    return operand.getDefiningOp()->getAttr("value");
  };

  all_results.clear();
  for (Operation &op : block.without_terminator()) {
    auto result_element_type_fn = [&](int idx) -> Type {
      return op.getResult(idx).getType().cast<ShapedType>().getElementType();
    };

    SmallVector<ShapedTypeComponents> results;
    EXPECT_TRUE(InferReturnTypeComponentsForTFOp(
                    op.getLoc(), &op, TFOp(&op).getNonControlOperands(),
                    /*graph_version=*/1010,
                    exclude_reshape_operand_as_constant_fn,
                    op_result_as_shape_fn, result_element_type_fn,
                    /*get_attr_values_fn=*/nullptr, results)
                    .succeeded());
    all_results.push_back(results);
  }

  VerifyInferredShapes(func.body().begin()->without_terminator(), all_results,
                       /*check_type*/ true);
}

// In this case, we are testing various kinds of inference failures.
TEST_F(ShapeInferenceTest, TestInferenceFailure) {
  // The three callbacks give no capability of inferring shapes.
  auto operand_as_constant_fn = [](Value operand) -> Attribute {
    return nullptr;
  };
  auto op_result_as_shape_fn = [](InferenceContext &ic,
                                  OpResult op_result) -> ShapeHandle {
    return {};
  };
  auto result_element_type_fn = [](int idx) -> Type { return nullptr; };

  GraphFuncOp func = GetModule().lookupSymbol<GraphFuncOp>("test");
  ASSERT_TRUE(func);
  Block &block = *func.body().begin();

  SmallVector<SmallVector<ShapedTypeComponents>> all_results;

  // Return an empty attribute will result in inference failure for certain ops.
  // For example, Const op expects it can get the shape information from the
  // "value" attribute.
  auto get_empty_attr_values_fn =
      [](Operation *, llvm::StringRef, const tensorflow::OpRegistrationData *,
         bool, tensorflow::AttrValueMap *) { return ::tensorflow::OkStatus(); };

  for (Operation &op : block.without_terminator()) {
    SmallVector<ShapedTypeComponents> results;
    auto result = InferReturnTypeComponentsForTFOp(
        op.getLoc(), &op, TFOp(&op).getNonControlOperands(),
        /*graph_version=*/1010, operand_as_constant_fn, op_result_as_shape_fn,
        result_element_type_fn, get_empty_attr_values_fn, results);
    // These ops expect the existing of certain attributes for shape inference.
    if (op.getName().getStringRef() == "tfg.Const" ||
        op.getName().getStringRef() == "tfg.IdentityN" ||
        op.getName().getStringRef() == "tfg.PlaceHolder" ||
        op.getName().getStringRef() == "tfg.Range")
      EXPECT_TRUE(failed(result));
  }

  // If parsing attribute returns error, then it won't invoke the
  // InferenceContext. As a result, all the shape inference should be failed.
  auto error_attr_values_fn = [](Operation *, llvm::StringRef,
                                 const tensorflow::OpRegistrationData *, bool,
                                 tensorflow::AttrValueMap *) {
    return tensorflow::errors::Unknown("Intended error");
  };

  for (Operation &op : block.without_terminator()) {
    SmallVector<ShapedTypeComponents> results;
    EXPECT_FALSE(InferReturnTypeComponentsForTFOp(
                     op.getLoc(), &op, TFOp(&op).getNonControlOperands(),
                     /*graph_version=*/1010, operand_as_constant_fn,
                     op_result_as_shape_fn, result_element_type_fn,
                     error_attr_values_fn, results)
                     .succeeded());
  }
}

}  // namespace tfg
}  // namespace mlir
