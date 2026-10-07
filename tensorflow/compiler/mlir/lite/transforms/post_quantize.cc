/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

// This transformation pass applies some clean up steps after quantization.

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/Dialect/Quant/IR/QuantTypes.h"  // from @llvm-project
#include "mlir/IR/AsmState.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributeInterfaces.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypeInterfaces.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/DialectResourceBlobManager.h"  // from @llvm-project
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Matchers.h"  // from @llvm-project
#include "mlir/IR/OpDefinition.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/TypeUtilities.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Support/TypeID.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops_in_place_transpose.h"
#include "tensorflow/compiler/mlir/lite/quantization/common/quantization_lib/quantization_config.h"
#include "tensorflow/compiler/mlir/lite/quantization/common/quantization_lib/quantization_utils.h"
#include "tensorflow/compiler/mlir/lite/transforms/passes.h"
#include "tensorflow/compiler/mlir/lite/utils/utils.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/dynamic_shape_utils.h"

//===----------------------------------------------------------------------===//
// The post-quantize Passes.
//
namespace mlir {
namespace TFL {
namespace {
#define GEN_PASS_DEF_POSTQUANTIZEPASS
#define GEN_PASS_DEF_POSTQUANTIZEREMOVEQDQPASS
#include "tensorflow/compiler/mlir/lite/transforms/passes.h.inc"

// Applies all the clean up steps after quantization.
class PostQuantizePass : public impl::PostQuantizePassBase<PostQuantizePass> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PostQuantizePass)

  // Constructor used by the PassRegistration. This will remove the adaptor ops.
  explicit PostQuantizePass() { this->emit_quant_adaptor_ops_ = false; }

  // Constructor used by manually creating the pass.
  explicit PostQuantizePass(bool emit_quant_adaptor_ops,
                            const CustomOpMap& custom_op_map)
      : custom_op_map_(custom_op_map) {
    // Set this flag to true if the inputs and outputs are in floating point.
    // The quant adaptor ops convert them to fixed point values (i.e. quantize)
    // before feeding them to the model and convert them back to floating point
    // (i.e. dequantize) as the output.
    this->emit_quant_adaptor_ops_ = emit_quant_adaptor_ops;
  }

  void runOnOperation() override;

 private:
  CustomOpMap custom_op_map_;
};

// Cleans up unnecessary QDQ pattern for input/output ops.
class PostQuantizeRemoveQDQPass
    : public impl::PostQuantizeRemoveQDQPassBase<PostQuantizeRemoveQDQPass> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PostQuantizeRemoveQDQPass)

  void runOnOperation() override;
};

// TODO(fengliuai): migrate to use modify_io_nodes pass.
void RemoveQuantizationAdaptorOps(func::FuncOp func) {
  mlir::OpBuilder builder(func.getBody());
  auto& bb = func.front();
  auto loc = func.getLoc();

  int num_args = bb.getNumArguments();
  llvm::SmallVector<Type, 4> input_types;
  input_types.reserve(num_args);
  // Edit the block arguments and create the new input ops in place to replace
  // the old input ops and quantize ops.
  for (int i = 0; i != num_args; ++i) {
    // Previous loop iteration may invalidate the insertion point so we have to
    // reset insertion point each iteration.
    builder.setInsertionPointToStart(&bb);

    // In each iteration, a new argument is appended to the end of the list
    // and the current argument is erased, so here we always process the first
    // argument in the list.
    auto arg = bb.getArgument(0);

    auto remove_quantize_op = [&](QuantizeOp quantize_op) {
      auto quantize_output = quantize_op.getOutput();
      auto quantize_type = quantize_output.getType();
      input_types.push_back(quantize_type);
      auto new_arg = bb.addArgument(quantize_type, loc);
      quantize_output.replaceAllUsesWith(new_arg);
      quantize_op.erase();
      arg.dropAllUses();
      bb.eraseArgument(0);
    };

    // This is looking for a pattern: arg -> tfl.quantize
    if (arg.hasOneUse() && llvm::isa<QuantizeOp>(*arg.user_begin())) {
      auto quantize_op = llvm::cast<QuantizeOp>(*arg.user_begin());
      remove_quantize_op(quantize_op);
      continue;
    }

    // Make a copy of current argument and append it to the end of the list if
    // the pattern isn't found.
    Type arg_type = arg.getType();
    input_types.push_back(arg_type);
    auto new_arg = bb.addArgument(arg_type, loc);
    arg.replaceAllUsesWith(new_arg);
    arg.dropAllUses();
    bb.eraseArgument(0);
  }

  // Edit the return ops and remove the dequantize ops in place.
  auto* terminator = bb.getTerminator();
  int num_return_operands = terminator->getNumOperands();
  llvm::SmallVector<Type, 4> output_types;
  output_types.reserve(num_return_operands);
  for (int i = 0; i != num_return_operands; ++i) {
    auto returned_value = terminator->getOperand(i);
    Operation* returned_op = returned_value.getDefiningOp();
    if (returned_op && returned_op->hasOneUse() &&
        llvm::isa<DequantizeOp>(returned_op)) {
      auto dequantize_op = llvm::cast<DequantizeOp>(returned_op);
      Value dequantized_result = dequantize_op.getInput();
      output_types.push_back(dequantized_result.getType());
      terminator->setOperand(i, dequantized_result);
      returned_op->erase();
    } else {
      output_types.push_back(returned_value.getType());
    }
  }
  auto new_func_type = builder.getFunctionType(input_types, output_types);
  func.setType(new_func_type);
}

enum RemoveVolatileOpsType {
  // Remove all volatile quant-dequant ops.
  kPreserveNone,
  // Preserve volatile quant-dequants for input and output ops.
  kPreserveInputsAndOutputs,
};

// Returns a constant tensor with the given scalar/vector value and shape.
template <typename T>
std::optional<mlir::Value> GetConstTensor(PatternRewriter& rewriter,
                                          Location loc, llvm::ArrayRef<T> vec,
                                          llvm::ArrayRef<int64_t> shape) {
  int64_t num_total_elements = 1;
  for (int64_t a : shape) {
    num_total_elements *= a;
  }

  if (vec.size() != num_total_elements) {
    return std::nullopt;
  }

  auto const_type = tensorflow::GetTypeFromTFTensorShape(
      shape, rewriter.getIntegerType(sizeof(T) * 8));
  auto const_attr = DenseElementsAttr::get(const_type, vec);

  auto const_op =
      arith::ConstantOp::create(rewriter, loc, const_type, const_attr);
  return const_op.getResult();
}

template <>
std::optional<mlir::Value> GetConstTensor(PatternRewriter& rewriter,
                                          Location loc,
                                          llvm::ArrayRef<double> vec,
                                          llvm::ArrayRef<int64_t> shape) {
  int64_t num_total_elements = 1;
  for (int64_t a : shape) {
    num_total_elements *= a;
  }

  if (vec.size() != num_total_elements) {
    return std::nullopt;
  }

  llvm::SmallVector<float, 4> float_vec;
  float_vec.reserve(vec.size());
  for (double d : vec) {
    float_vec.push_back(static_cast<float>(d));
  }

  auto const_type =
      tensorflow::GetTypeFromTFTensorShape(shape, rewriter.getF32Type());
  auto const_attr =
      DenseElementsAttr::get(const_type, llvm::ArrayRef<float>(float_vec));

  auto const_op =
      arith::ConstantOp::create(rewriter, loc, const_type, const_attr);
  return const_op.getResult();
}

// Converts a dequantize op to a (scale * (input - zeropoint)). The expectation
// is that the qconst value will be constant folded to retain the original
// constant value. This is essentially a constant fold of the dequantize op,
// privided that the value, zp and scale are all constants.
std::optional<mlir::Value> ConvertDequantizeOp(
    PatternRewriter& rewriter, mlir::Operation* op,
    mlir::ShapedType output_type, mlir::Value input_value,
    llvm::ArrayRef<double> scale, llvm::ArrayRef<int64_t> zeropoint,
    int64_t dim) {
  RankedTensorType input_type =
      dyn_cast<RankedTensorType>(input_value.getType());
  if (!input_type) return std::nullopt;

  std::optional<mlir::Value> zp_val;
  if (zeropoint.size() == 1) {
    auto const_type =
        tensorflow::GetTypeFromTFTensorShape({}, rewriter.getF32Type());
    auto const_attr =
        DenseElementsAttr::get(const_type, static_cast<float>(zeropoint[0]));

    auto const_op = arith::ConstantOp::create(rewriter, op->getLoc(),
                                              const_type, const_attr);
    zp_val = const_op.getResult();
  } else {
    SmallVector<int64_t> shape;
    shape.resize(input_type.getRank(), 1);
    shape[dim] = zeropoint.size();
    zp_val = GetConstTensor(rewriter, op->getLoc(), zeropoint, shape);
  }

  std::optional<mlir::Value> scale_val;
  if (scale.size() == 1) {
    auto const_type =
        tensorflow::GetTypeFromTFTensorShape({}, rewriter.getF32Type());
    auto const_attr =
        DenseElementsAttr::get(const_type, static_cast<float>(scale[0]));

    auto const_op = arith::ConstantOp::create(rewriter, op->getLoc(),
                                              const_type, const_attr);
    scale_val = const_op.getResult();
  } else {
    SmallVector<int64_t> shape;
    shape.resize(input_type.getRank(), 1);
    shape[dim] = scale.size();
    scale_val = GetConstTensor(rewriter, op->getLoc(), scale, shape);
  }

  if (!zp_val || !scale_val) return std::nullopt;

  auto op1_cast_in =
      TFL::CastOp::create(rewriter, op->getLoc(), output_type, input_value);

  auto op2_sub_op1 = TFL::SubOp::create(
      rewriter, op->getLoc(), output_type, op1_cast_in.getResult(),
      zp_val.value(),
      /*fused_activation_function=*/rewriter.getStringAttr("NONE"));

  return TFL::MulOp::create(
             rewriter, op->getLoc(), output_type, op2_sub_op1.getResult(),
             scale_val.value(),
             /*fused_activation_function=*/rewriter.getStringAttr("NONE"))
      .getResult();
}

// Remove the back-to-back quantize and dequantize ops with volatile attribute.
template <RemoveVolatileOpsType remove_volatile_ops_type>
struct RemoveVolatileOps : public OpRewritePattern<DequantizeOp> {
  explicit RemoveVolatileOps(MLIRContext* context)
      : OpRewritePattern<DequantizeOp>(context, 1) {}

  LogicalResult matchAndRewrite(DequantizeOp op,
                                PatternRewriter& rewriter) const override {
    auto input_op = op.getInput().getDefiningOp();
    if (auto q = llvm::dyn_cast_or_null<QuantizeOp>(input_op)) {
      if (!q->getAttr(kVolatileOpAttrName)) return failure();

      if (remove_volatile_ops_type == kPreserveInputsAndOutputs) {
        // Don't remove leading and trailing QDQ for PTQ workflow, so the io
        // modifying lib can work correctly.
        if (!q.getInput().getDefiningOp()) return failure();
        if (op->hasOneUse() &&
            op->user_begin()->hasTrait<OpTrait::IsTerminator>())
          return failure();
      }
      // If the quantize op is a requantize op, it is being used in other scale
      // adjustments and should be kept. Instead, moving dequantize op before
      // the requantize op to remove the unnecessary requantize op.
      if (auto qtype = quant::QuantizedType::getQuantizedElementType(
              q.getInput().getType())) {
        rewriter.setInsertionPoint(op);
        rewriter.replaceOpWithNewOp<DequantizeOp>(op, op.getOutput().getType(),
                                                  q.getInput());
        return success();
      }

      op.replaceAllUsesWith(q.getInput());
      return success();
    } else if (auto qconst_op = llvm::dyn_cast_or_null<QConstOp>(input_op)) {
      if (!qconst_op->getAttr(kVolatileOpAttrName)) return failure();

      auto qtype =
          quant::QuantizedType::getQuantizedElementType(qconst_op.getType());
      if (!qtype) return failure();
      SmallVector<double, 1> scale;
      SmallVector<int64_t, 1> zeropoint;
      int64_t dim = 0;

      if (auto uniform_qtype =
              mlir::dyn_cast<quant::UniformQuantizedType>(qtype)) {
        scale.push_back(uniform_qtype.getScale());
        zeropoint.push_back(uniform_qtype.getZeroPoint());
      } else if (auto per_axis_qtype =
                     mlir::dyn_cast<quant::UniformQuantizedPerAxisType>(
                         qtype)) {
        scale.assign(per_axis_qtype.getScales().begin(),
                     per_axis_qtype.getScales().end());
        zeropoint.assign(per_axis_qtype.getZeroPoints().begin(),
                         per_axis_qtype.getZeroPoints().end());
        dim = per_axis_qtype.getQuantizedDimension();
      } else {
        return failure();
      }

      auto output_type = mlir::cast<mlir::ShapedType>(op.getOutput().getType());

      auto const_type = tensorflow::GetTypeFromTFTensorShape(
          output_type.getShape(), qtype.getStorageType());
      auto const_op = arith::ConstantOp::create(
          rewriter, op->getLoc(), const_type, qconst_op.getValue());

      auto new_value =
          ConvertDequantizeOp(rewriter, op, output_type, const_op.getResult(),
                              scale, zeropoint, dim);
      if (!new_value) return failure();

      op.replaceAllUsesWith(new_value.value());
      op->erase();
      return success();
    }
    return failure();
  }
};

static bool MatchElementsAttr(Value val, ElementsAttr& attr) {
  if (matchPattern(val, m_Constant(&attr))) {
    return true;
  }
  if (Operation* op = val.getDefiningOp()) {
    if (auto const_op = llvm::dyn_cast<TFL::ConstOp>(op)) {
      attr = const_op.getValue();
      return true;
    }
    if (auto qconst_op = llvm::dyn_cast<TFL::QConstOp>(op)) {
      attr = qconst_op.getValue();
      return true;
    }
    if (auto arith_const = llvm::dyn_cast<arith::ConstantOp>(op)) {
      attr = mlir::dyn_cast<ElementsAttr>(arith_const.getValue());
      return attr != nullptr;
    }
  }
  return false;
}

// Fold the constant quantized Transpose ops.
struct FoldTransposeOp : public OpRewritePattern<TransposeOp> {
  explicit FoldTransposeOp(MLIRContext* context)
      : OpRewritePattern<TransposeOp>(context, 1) {}

  // Computes the permutation of a constant `input_tensor` according to `perm`.
  // The function recursively traverses the dimensions of the output tensor in
  // a row-major order and writes the value in the output tensor into
  // `new_values`.
  void ComputePermutation(ElementsAttr input_tensor, ArrayRef<int32_t> perm,
                          ArrayRef<int64_t> output_shape, int num_dimensions,
                          int output_axis, std::vector<uint64_t>* input_indices,
                          std::vector<Attribute>* new_values) const {
    // Refer to the implementation of `Transpose` function in
    // tensorflow/lite/kernels/internal/reference/reference_ops.h
    assert(output_axis < num_dimensions);
    const int input_axis = perm[output_axis];
    for (int i = 0; i < output_shape[output_axis]; ++i) {
      // Update the input indices on `input_axis`.
      assert(input_axis < input_indices->size());
      input_indices->operator[](input_axis) = static_cast<uint64_t>(i);
      // Write the value from `input_tensor` if it is the last axis or
      // recurse into the next axis.
      const bool is_last_axis = output_axis == num_dimensions - 1;
      if (is_last_axis) {
        new_values->push_back(
            input_tensor.getValues<Attribute>()[*input_indices]);
      } else {
        ComputePermutation(input_tensor, perm, output_shape, num_dimensions,
                           output_axis + 1, input_indices, new_values);
      }
    }
  }

  LogicalResult matchAndRewrite(TransposeOp op,
                                PatternRewriter& rewriter) const override {
    Operation* def_op = op.getInput().getDefiningOp();
    auto qconst_op = llvm::dyn_cast_or_null<QConstOp>(def_op);
    if (qconst_op == nullptr) return failure();

    ElementsAttr perm_attr;
    if (!MatchElementsAttr(op.getPerm(), perm_attr)) return failure();
    auto int_perm_attr = mlir::dyn_cast<DenseIntElementsAttr>(perm_attr);
    if (!int_perm_attr) return failure();

    auto result_type = mlir::cast<RankedTensorType>(op.getOutput().getType());
    auto output_element_type = result_type.getElementType();
    if (!mlir::isa<quant::UniformQuantizedType>(output_element_type) &&
        !mlir::isa<quant::UniformQuantizedPerAxisType>(output_element_type)) {
      return failure();
    }

    ElementsAttr input_tensor = qconst_op.getValue();

    const int num_dimensions = input_tensor.getShapedType().getRank();
    ArrayRef<int64_t> input_shape = input_tensor.getShapedType().getShape();

    SmallVector<int32_t, 4> perm;
    for (const APInt& it : int_perm_attr.getValues<APInt>()) {
      perm.push_back(it.getSExtValue());
    }
    if (perm.size() != num_dimensions) return failure();

    SmallVector<int64_t, 4> output_shape;
    for (int i = 0; i < num_dimensions; ++i) {
      output_shape.push_back(input_shape[perm[i]]);
      assert(!result_type.hasStaticShape() ||
             result_type.getShape()[i] == output_shape[i]);
    }

    if (auto dense_input = mlir::dyn_cast<DenseElementsAttr>(input_tensor)) {
      std::vector<Attribute> new_values;
      new_values.reserve(input_tensor.getShapedType().getNumElements());
      std::vector<uint64_t> input_indices(num_dimensions);
      ComputePermutation(dense_input, perm, output_shape, num_dimensions,
                         /*output_axis=*/0, &input_indices, &new_values);
      RankedTensorType values_type;
      if (mlir::isa<quant::UniformQuantizedType>(output_element_type)) {
        values_type = RankedTensorType::get(
            output_shape,
            mlir::cast<quant::UniformQuantizedType>(output_element_type)
                .getStorageType());
      } else {
        values_type = RankedTensorType::get(
            output_shape,
            mlir::cast<quant::UniformQuantizedPerAxisType>(output_element_type)
                .getStorageType());
      }

      rewriter.replaceOpWithNewOp<QConstOp>(
          op, TypeAttr::get(result_type),
          DenseIntElementsAttr::get(values_type, new_values));
      return success();
    }

    if (auto dense_res =
            mlir::dyn_cast<DenseResourceElementsAttr>(input_tensor)) {
      AsmResourceBlob* blob = dense_res.getRawHandle().getBlob();
      if (!blob && dense_res.getRawHandle().getResource()) {
        blob = dense_res.getRawHandle().getResource()->getBlob();
      }
      if (!blob || blob->getData().empty()) return failure();

      uint32_t storage_bit_width = 8;
      if (auto u = mlir::dyn_cast<quant::UniformQuantizedType>(
              output_element_type)) {
        storage_bit_width = u.getStorageTypeIntegralWidth();
      } else if (auto p = mlir::dyn_cast<quant::UniformQuantizedPerAxisType>(
                     output_element_type)) {
        storage_bit_width = p.getStorageTypeIntegralWidth();
      }

      size_t num_elements = result_type.getNumElements();
      size_t out_byte_size = storage_bit_width < 8
                                 ? (num_elements * storage_bit_width + 7) / 8
                                 : num_elements * (storage_bit_width / 8);
      auto raw_output_blob = mlir::HeapAsmResourceBlob::allocate(
          out_byte_size, /*align=*/64, /*dataIsMutable=*/true);
      char* raw_output =
          const_cast<char*>(raw_output_blob.getDataAs<char>().data());
      std::memset(raw_output, 0, out_byte_size);
      const char* raw_input = blob->getData().data();

      SmallVector<int64_t, 4> perms_i64(perm.begin(), perm.end());
      if (!TransposeBuffer(raw_input, raw_output, input_shape, perms_i64,
                           storage_bit_width)) {
        return failure();
      }

      DenseResourceElementsAttr new_res_attr = DenseResourceElementsAttr::get(
          result_type, dense_res.getRawHandle().getKey(),
          std::move(raw_output_blob));
      rewriter.replaceOpWithNewOp<QConstOp>(op, TypeAttr::get(result_type),
                                            new_res_attr);
      return success();
    }

    return failure();
  }
};

// Fold constant quantized Reshape ops.
struct FoldReshapeOp : public OpRewritePattern<ReshapeOp> {
  // Does not take ownership of context, which must refer to a valid value that
  // outlives this object.
  explicit FoldReshapeOp(MLIRContext* context)
      : OpRewritePattern<ReshapeOp>(context, /*benefit=*/1) {}

  LogicalResult matchAndRewrite(ReshapeOp op,
                                PatternRewriter& rewriter) const override {
    Operation* def_op = op.getInput().getDefiningOp();
    auto qconst_op = llvm::dyn_cast_or_null<QConstOp>(def_op);
    if (qconst_op == nullptr) {
      return rewriter.notifyMatchFailure(op, "input is not a QConstOp.");
    }

    auto output_element_type = getElementTypeOrSelf(op.getType());
    if (!mlir::isa<quant::QuantizedType>(output_element_type)) {
      return rewriter.notifyMatchFailure(op, "output type is not quantized.");
    }

    // Remove identity reshape with both static result and input shape.
    auto result_type = mlir::cast<ShapedType>(op.getType());
    auto input_type = mlir::cast<ShapedType>(op.getInput().getType());

    // Constant folding
    // If the result type isn't static, tries to derive the result type from
    // the #2 operand.
    if (!result_type.hasStaticShape()) {
      ElementsAttr shape_attr;
      if (!MatchElementsAttr(op.getShape(), shape_attr)) return failure();
      auto shape_elements = mlir::dyn_cast<DenseIntElementsAttr>(shape_attr);
      if (!shape_elements) return failure();

      SmallVector<int64_t, 4> shape_data;
      for (const APInt& it : shape_elements.getValues<APInt>()) {
        shape_data.push_back(it.getSExtValue());
      }
      result_type =
          RankedTensorType::get(shape_data, input_type.getElementType());
    }

    RankedTensorType values_type;
    if (auto uniform_qtype =
            mlir::dyn_cast<quant::UniformQuantizedType>(output_element_type)) {
      values_type = RankedTensorType::get(result_type.getShape(),
                                          uniform_qtype.getStorageType());
    } else {
      values_type = RankedTensorType::get(
          result_type.getShape(),
          mlir::cast<quant::UniformQuantizedPerAxisType>(output_element_type)
              .getStorageType());
    }

    ElementsAttr value_attr = qconst_op.getValue();
    if (auto dense_elements = mlir::dyn_cast<DenseElementsAttr>(value_attr)) {
      DenseElementsAttr reshaped_elements = dense_elements.reshape(values_type);
      rewriter.replaceOpWithNewOp<QConstOp>(op, TypeAttr::get(result_type),
                                            reshaped_elements);
      return success();
    }

    if (auto dense_resource_elements =
            mlir::dyn_cast<DenseResourceElementsAttr>(value_attr)) {
      AsmResourceBlob* blob = dense_resource_elements.getRawHandle().getBlob();
      if (!blob && dense_resource_elements.getRawHandle().getResource()) {
        blob = dense_resource_elements.getRawHandle().getResource()->getBlob();
      }
      if (!blob || blob->getData().empty()) return failure();

      DenseResourceElementsAttr new_res_attr;
      if (qconst_op.getOutput().hasOneUse()) {
        new_res_attr = DenseResourceElementsAttr::get(
            result_type, dense_resource_elements.getRawHandle().getKey(),
            std::move(*blob));
      } else {
        auto new_blob = mlir::HeapAsmResourceBlob::allocate(
            blob->getData().size(), /*align=*/64, true);
        memcpy(const_cast<char*>(new_blob.getData().data()),
               blob->getData().data(), blob->getData().size());
        new_res_attr = DenseResourceElementsAttr::get(
            result_type, dense_resource_elements.getRawHandle().getKey(),
            std::move(new_blob));
      }
      rewriter.replaceOpWithNewOp<QConstOp>(op, TypeAttr::get(result_type),
                                            new_res_attr);
      return success();
    }

    return failure();
  }
};

// Removes operations with side effect (i.e. LSTM, SVDF) that have dangling
// output.
template <typename OpTy>
struct PruneUnusedOpsWithSideEffect : public OpRewritePattern<OpTy> {
 public:
  explicit PruneUnusedOpsWithSideEffect(MLIRContext* context,
                                        const CustomOpMap& custom_op_map = {})
      : OpRewritePattern<OpTy>(context), custom_op_map(custom_op_map) {}

  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter& rewriter) const override {
    if (op.getOperation()->template hasTrait<OpTrait::IsTerminator>()) {
      return failure();
    }
    for (auto result : op.getOperation()->getOpResults()) {
      if (!result.use_empty()) {
        return failure();
      }
    }
    // Remove if the custom op is in the provided map and is NoSideEffect.
    auto custom_op = llvm::isa<CustomOp>(op);
    if (custom_op) {
      auto q = llvm::cast<CustomOp>(op);
      std::string op_name = q.getCustomCode().str();
      if ((custom_op_map.find(op_name) == custom_op_map.end()) ||
          !custom_op_map.find(op_name)->second.no_side_effect)
        return failure();
    }
    rewriter.eraseOp(op);
    return success();
  }
  CustomOpMap custom_op_map;
};

#include "tensorflow/compiler/mlir/lite/transforms/generated_post_quantize.inc"

void PostQuantizePass::runOnOperation() {
  if (!enable_custom_op_no_side_effect_.empty()) {
    ParseCustomOpSpecs(enable_custom_op_no_side_effect_,
                       CustomOpUpdateOptions::kNoSideEffect, custom_op_map_);
  }

  RewritePatternSet patterns(&getContext());
  auto func = getOperation();
  auto* ctx = func.getContext();
  TFL::populateWithGenerated(patterns);
  patterns.add<FoldTrivalRequantizeOp<QuantizeOp>>(ctx);
  patterns.add<PruneUnusedOpsWithSideEffect<TFL::LSTMOp>>(ctx);
  patterns.add<PruneUnusedOpsWithSideEffect<TFL::UnidirectionalSequenceLSTMOp>>(
      ctx);
  patterns.add<PruneUnusedOpsWithSideEffect<TFL::SVDFOp>>(ctx);
  patterns.add<PruneUnusedOpsWithSideEffect<TFL::CustomOp>>(ctx,
                                                            custom_op_map_);
  (void)applyPatternsGreedily(func, std::move(patterns));

  if (!emit_quant_adaptor_ops_) {
    RemoveQuantizationAdaptorOps(getOperation());
  }

  RewritePatternSet phase_2_patterns(&getContext());
  TFL::populateWithGenerated(phase_2_patterns);
  phase_2_patterns.add<FoldTrivalRequantizeOp<QuantizeOp>,
                       RemoveVolatileOps<kPreserveInputsAndOutputs>,
                       FoldTransposeOp, FoldReshapeOp>(ctx);
  (void)applyPatternsGreedily(func, std::move(phase_2_patterns));
}

void PostQuantizeRemoveQDQPass::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  auto func = getOperation();
  auto* ctx = func.getContext();
  TFL::populateWithGenerated(patterns);
  patterns.add<RemoveVolatileOps<kPreserveNone>>(ctx);
  (void)applyPatternsGreedily(func, std::move(patterns));
}

}  // namespace

// Creates an instance of the TensorFlow Lite dialect PostQuantize pass.
std::unique_ptr<OperationPass<func::FuncOp>> CreatePostQuantizePass(
    bool emit_quant_adaptor_ops, const CustomOpMap& custom_op_map) {
  return std::make_unique<PostQuantizePass>(emit_quant_adaptor_ops,
                                            custom_op_map);
}

std::unique_ptr<OperationPass<func::FuncOp>> CreatePostQuantizePass() {
  return std::make_unique<PostQuantizePass>();
}

// Creates an instance of the TensorFlow Lite dialect PostQuantizeRemoveQDQ
// pass.
std::unique_ptr<OperationPass<func::FuncOp>> CreatePostQuantizeRemoveQDQPass() {
  return std::make_unique<PostQuantizeRemoveQDQPass>();
}

}  // namespace TFL
}  // namespace mlir
