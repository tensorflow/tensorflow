/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/legalize_hlo_conversions/reduce_window.h"

#include <cstdint>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypeInterfaces.h"  // from @llvm-project
#include "mlir/IR/IRMapping.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Matchers.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/DialectConversion.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"  // IWYU pragma: keep
#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/legalize_hlo_conversions/op_util_common.h"
#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/legalize_hlo_conversions/reduce_window_util.h"
#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/legalize_hlo_conversions/util.h"
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"

namespace mlir::odml {
namespace {

// filters, strides, padding, faf.
using TFLPoolAttrsT = std::tuple<IntegerAttr, IntegerAttr, IntegerAttr,
                                 IntegerAttr, StringAttr, StringAttr>;

bool AreDilationsSupported(const ReduceWindowView& op) {
  auto is_one = [](int64_t v) { return v == 1; };
  return llvm::all_of(op.BaseDilations(), is_one) &&
         llvm::all_of(op.WindowDilations(), is_one);
}

bool IsRankSupported(const ReduceWindowView& op) { return op.Rank() == 4; }

std::optional<std::tuple<ReduceWindowView, Layout>> GetViewIfAttrsSupported(
    mhlo::ReduceWindowOp op) {
  const ReduceWindowView view(op);

  if (!IsRankSupported(view)) {
    return std::nullopt;
  }

  if (!AreDilationsSupported(view)) {
    return std::nullopt;
  }

  auto opt_layout = view.GuessLayout();
  if (!opt_layout.has_value()) {
    return std::nullopt;
  }
  auto layout = opt_layout.value();

  const int64_t batch = layout.SpecialDim1();
  if (!view.Paddings()[batch].Trivial()) {
    return std::nullopt;
  }

  const int64_t chan = layout.SpecialDim2();
  if (!view.Paddings()[chan].Trivial()) {
    return std::nullopt;
  }

  return std::tuple(view, layout);
}

std::optional<bool> IsReduceWindowLegal(mhlo::ReduceWindowOp op) {
  return std::nullopt;
}

std::optional<bool> IsDivideLegal(mhlo::DivOp op) { return std::nullopt; }

Layout TFLNativePoolingLayout(int64_t rank) {
  return Layout(0, rank - 1, llvm::to_vector(llvm::seq<int64_t>(1, rank - 1)));
}

bool IsCstFloatZero(Value val) {
  DenseFPElementsAttr initial_value;
  return matchPattern(val, m_Constant(&initial_value)) &&
         initial_value.getNumElements() == 1 &&
         initial_value.getValues<APFloat>()[0].isZero();
}

bool IsCstIntZero(Value val) {
  DenseIntElementsAttr initial_value;
  return matchPattern(val, m_Constant(&initial_value)) &&
         initial_value.getNumElements() == 1 &&
         initial_value.getValues<APInt>()[0].isZero();
}

llvm::SmallVector<int64_t> Permute(llvm::ArrayRef<int64_t> data,
                                   llvm::ArrayRef<int64_t> perm) {
  llvm::SmallVector<int64_t> res(data.size());
  for (int i = 0; i < data.size(); ++i) {
    res[i] = data[perm[i]];
  }
  return res;
}

Value TransposeTensor(OpBuilder& b, Value tensor,
                      llvm::SmallVector<int64_t> perm) {
  const int64_t perm_size = perm.size();
  auto perm_attr_type = RankedTensorType::get({perm_size}, b.getI64Type());
  auto perm_attr = DenseIntElementsAttr::get(perm_attr_type, perm);
  return b.create<mhlo::TransposeOp>(tensor.getLoc(), tensor, perm_attr);
}

DenseIntElementsAttr BuildDenseI64(OpBuilder& b, ArrayRef<int64_t> shape,
                                   ArrayRef<int64_t> data) {
  return DenseIntElementsAttr::get(RankedTensorType::get(shape, b.getI64Type()),
                                   data);
}

DenseIntElementsAttr BuildDenseI64(OpBuilder& b, ArrayRef<int64_t> data) {
  const int64_t dim = data.size();
  return BuildDenseI64(b, {dim}, data);
}

std::optional<std::tuple<Value, Value>> GetInputAndInitIfValid(
    mhlo::ReduceWindowOp op) {
  if (op->getNumResults() != 1) {
    return std::nullopt;
  }
  if (op.getInputs().size() > 1) {
    return std::nullopt;
  }
  if (op.getInitValues().size() > 1) {
    return std::nullopt;
  }
  auto init_val = op.getInitValues().front();
  if (llvm::dyn_cast<ShapedType>(init_val.getType()).getNumElements() != 1) {
    return std::nullopt;
  }
  return std::tuple(op.getInputs().front(), op.getInitValues().front());
}

std::optional<std::string> GetTFLPadding(ArrayRef<DimPadding> paddings,
                                         ArrayRef<int64_t> window_strides,
                                         ArrayRef<int64_t> in_shape,
                                         ArrayRef<int64_t> window_dims) {
  const int64_t rank = paddings.size();
  std::string tfl_padding = "VALID";
  for (int i = 1; i < rank - 1; ++i) {
    const auto& dim_pad = paddings[i];
    if (dim_pad.Trivial()) {
      continue;
    }
    if (!IsSamePaddingOnDim(in_shape[i], 1, window_strides[i], window_dims[i],
                            dim_pad)) {
      return std::nullopt;
    }
    tfl_padding = "SAME";
  }
  return tfl_padding;
}

TFLPoolAttrsT BuildTFLPoolAttrs(OpBuilder& b, const ReduceWindowView& view,
                                StringRef padding) {
  const int32_t filter_h = view.WindowDims()[1];
  auto filter_h_attr = b.getI32IntegerAttr(filter_h);

  const int32_t filter_w = view.WindowDims()[2];
  auto filter_w_attr = b.getI32IntegerAttr(filter_w);

  const int32_t stride_h = view.WindowStrides()[1];
  auto stride_h_attr = b.getI32IntegerAttr(stride_h);

  const int32_t stride_w = view.WindowStrides()[2];
  auto stride_w_attr = b.getI32IntegerAttr(stride_w);

  auto padding_attr = b.getStringAttr(padding);
  auto faf_attr = b.getStringAttr("NONE");

  return std::tuple(filter_h_attr, filter_w_attr, stride_h_attr, stride_w_attr,
                    padding_attr, faf_attr);
}

//===------------------------------------------------------------------------===
// relayout reduce_window to channel last
//===------------------------------------------------------------------------===

class RelayoutReduceWindow : public OpRewritePattern<mhlo::ReduceWindowOp> {
 public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(mhlo::ReduceWindowOp op,
                                PatternRewriter& rewriter) const final;
};

LogicalResult RelayoutReduceWindow::matchAndRewrite(
    mhlo::ReduceWindowOp op, PatternRewriter& rewriter) const {
  //
  // check and parse attributes
  //=-----

  auto opt_view = GetViewIfAttrsSupported(op);
  if (!opt_view.has_value()) {
    return rewriter.notifyMatchFailure(
        op, "Reduce window attributes not supported.");
  }
  const auto [view, layout] = opt_view.value();

  //
  // get inputs and inits if there are only one
  //=-----

  auto opt_input_and_init = GetInputAndInitIfValid(op);
  if (!opt_input_and_init.has_value()) {
    return rewriter.notifyMatchFailure(
        op, "Reduce window has wrong number of inputs or init values.");
  }
  auto [input, init_val] = opt_input_and_init.value();

  //
  // figure out permutations for layout change
  //=-----

  const auto target_layout = TFLNativePoolingLayout(view.Rank());
  if (layout == target_layout) {
    return rewriter.notifyMatchFailure(
        op, "Reduce window does not need layout change");
  }

  llvm::SmallVector<int64_t> perm_for_inputs =
      layout.GetPermForReLayout(target_layout);

  //
  // permute layout sensitive attrs
  //=-----

  // permute paddings
  auto paddings = view.Paddings();
  llvm::SmallVector<int64_t> new_paddings(paddings.size() * 2);
  for (int i = 0; i < new_paddings.size() / 2; ++i) {
    const auto& dim_pad = paddings[perm_for_inputs[i]];
    new_paddings[2 * i] = dim_pad.Lo();
    new_paddings[2 * i + 1] = dim_pad.Hi();
  }
  const int64_t new_paddings_size = paddings.size();
  auto new_paddings_type =
      RankedTensorType::get({new_paddings_size, 2}, rewriter.getI64Type());
  auto new_paddings_attr =
      DenseIntElementsAttr::get(new_paddings_type, new_paddings);

  // permute window dims
  llvm::SmallVector<int64_t> new_window_dims =
      Permute(view.WindowDims(), perm_for_inputs);
  auto new_window_dims_attr = BuildDenseI64(rewriter, new_window_dims);

  // permute window strides
  llvm::SmallVector<int64_t> new_window_strides =
      Permute(view.WindowStrides(), perm_for_inputs);
  auto new_window_strides_attr = BuildDenseI64(rewriter, new_window_strides);

  //
  // permute params and build new op
  //=-----

  // figure out permuted result type
  llvm::SmallVector<int64_t> perm_for_outputs =
      target_layout.GetPermForReLayout(layout);
  auto cur_out_type = llvm::dyn_cast<ShapedType>(op.getResult(0).getType());
  llvm::SmallVector<int64_t> new_rw_out_shape =
      layout.PermuteShape(target_layout, cur_out_type.getShape());
  auto new_out_type = cur_out_type.clone(new_rw_out_shape);

  // transpose input and build new reduce_window
  auto new_input = TransposeTensor(rewriter, input, perm_for_inputs);
  auto new_rw = rewriter.create<mhlo::ReduceWindowOp>(
      op.getLoc(), new_out_type, new_input, init_val, new_window_dims_attr,
      new_window_strides_attr, BuildDenseI64(rewriter, view.BaseDilations()),
      BuildDenseI64(rewriter, view.WindowDilations()), new_paddings_attr);
  IRMapping ir_map;
  op.getBody().cloneInto(&new_rw.getBody(), ir_map);

  // transpose output and update ir
  auto new_output =
      TransposeTensor(rewriter, new_rw.getResult(0), perm_for_outputs);
  rewriter.replaceOp(op, new_output);

  return success();
}

//===------------------------------------------------------------------------===
// mhlo.reduce_window -> tfl.cum_sum
//===------------------------------------------------------------------------===

class LegalizeCumSum : public OpConversionPattern<mhlo::ReduceWindowOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      mhlo::ReduceWindowOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const final;
};

LogicalResult LegalizeCumSum::matchAndRewrite(
    mhlo::ReduceWindowOp op, OpAdaptor adaptor,
    ConversionPatternRewriter& rewriter) const {
  //
  // check singular params and trivial attrs
  //=-----

  auto opt_input_init = GetInputAndInitIfValid(op);
  if (!opt_input_init.has_value()) {
    return rewriter.notifyMatchFailure(op,
                                       "Must have 1 input, init and result.");
  }
  auto [input, init] = opt_input_init.value();

  if (failed(MatchBinaryReduceFunction<mhlo::AddOp>(op.getBody()))) {
    return rewriter.notifyMatchFailure(op, "Requires scalar add in region.");
  }

  if (!IsCstFloatZero(init) && !IsCstIntZero(init)) {
    return rewriter.notifyMatchFailure(op, "Requires 0 for init value.");
  }

  const ReduceWindowView view(op);

  auto trivial = [](int64_t v) { return v == 1; };
  const bool trivial_window_dilate =
      llvm::all_of(view.WindowDilations(), trivial);
  const bool trivial_base_dilate = llvm::all_of(view.BaseDilations(), trivial);
  const bool trivial_stride = llvm::all_of(view.WindowStrides(), trivial);
  if (!trivial_window_dilate || !trivial_stride || !trivial_base_dilate) {
    return rewriter.notifyMatchFailure(
        op, "Requires trivial strides and dilations attributes.");
  }

  //
  // figure out the implicit axis of reduction
  //=-----

  auto input_type = llvm::cast<ShapedType>(input.getType());
  if (view.WindowDims().size() != input_type.getRank()) {
    return rewriter.notifyMatchFailure(op, "Splat window dims not supported.");
  }
  int64_t axis = -1;
  for (auto [ind, val] : llvm::enumerate(view.WindowDims())) {
    if (val == 1) {
      continue;
    }

    if (axis != -1) {
      return rewriter.notifyMatchFailure(op, "Multiple non 1 dimensions.");
    }

    if (val != input_type.getShape()[ind]) {
      return rewriter.notifyMatchFailure(
          op, "Axis dimension requires size be same as input shape's.");
    }
    axis = ind;
  }

  if (axis == -1) {
    return rewriter.notifyMatchFailure(op, "Could not identify axis.");
  }

  const int64_t axis_size = input_type.getShape()[axis];

  //
  // validate padding is [N-1, 0] on axis and zero elsewhere
  //=-----

  for (const auto& [ind, dim_pad] : llvm::enumerate(view.Paddings())) {
    if (dim_pad.Hi() != 0) {
      return rewriter.notifyMatchFailure(op, "Has non trivial high padding.");
    }

    if (ind != axis) {
      if (!dim_pad.Trivial()) {
        return rewriter.notifyMatchFailure(
            op, "Has non trivial padding on non axis dim.");
      }
    } else {
      if (dim_pad.Lo() != axis_size - 1) {
        return rewriter.notifyMatchFailure(
            op, "Requires low padding on axis dim to be N - 1.");
      }
    }
  }

  //
  // build axis constant and tfl op
  //=-----

  auto axis_cst_attr = DenseIntElementsAttr::get(
      RankedTensorType::get({}, rewriter.getI32Type()),
      static_cast<int32_t>(axis));
  auto axis_cst =
      rewriter.create<arith::ConstantOp>(op->getLoc(), axis_cst_attr);

  auto tfl_exclusive_attr = rewriter.getBoolAttr(false);
  auto tfl_reverse_attr = rewriter.getBoolAttr(false);

  rewriter.replaceOpWithNewOp<TFL::CumsumOp>(op, op->getResultTypes()[0], input,
                                             axis_cst, tfl_exclusive_attr,
                                             tfl_reverse_attr);

  return success();
}

//===------------------------------------------------------------------------===
// mhlo.reduce_window -> tfl.max_pool
//===------------------------------------------------------------------------===

bool isFloatMinusInfinity(Value value) {
  DenseFPElementsAttr float_value;
  if (!matchPattern(value, m_Constant(&float_value))) {
    return false;
  }
  if (float_value.getNumElements() != 1) {
    return false;
  }
  APFloat element = float_value.getValues<APFloat>()[0];
  return element.isInfinity() && element.isNegative();
}

class LegalizeMaxPool : public OpConversionPattern<mhlo::ReduceWindowOp> {
 public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      mhlo::ReduceWindowOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const final;

 private:
  TFL::PadV2Op BuildExplicitPadOp(mhlo::ReduceWindowOp op, const Layout& layout,
                                  const ShapedType& input_type,
                                  const ShapedType& output_type, Value input,
                                  Value init, const ReduceWindowView& view,
                                  PatternRewriter& rewriter) const;
};

TFL::PadV2Op LegalizeMaxPool::BuildExplicitPadOp(
    mhlo::ReduceWindowOp op, const Layout& layout, const ShapedType& input_type,
    const ShapedType& output_type, Value input, Value init,
    const ReduceWindowView& view, PatternRewriter& rewriter) const {
  // The following works for rank=4 (see IsRankSupported()).
  std::vector<int64_t> shape = {layout.Rank(), layout.NumSpatials()};

  // Create attribute: padding_values
  // For an NHWC with this padding: [[a, b], [c, d], [e, f], [g, h]]
  // we want [a, b, c, d, e, f, g, h]
  llvm::SmallVector<int64_t, 8> padding_values;
  for (auto& padding : view.Paddings()) {
    padding_values.push_back(padding.Lo());
    padding_values.push_back(padding.Hi());
  }

  auto padding_dense_attr = mlir::DenseElementsAttr::get(
      mlir::RankedTensorType::get(shape, rewriter.getIntegerType(64)),
      llvm::ArrayRef<int64_t>(padding_values));

  auto padding_values_op =
      rewriter.create<arith::ConstantOp>(op.getLoc(), padding_dense_attr);

  llvm::SmallVector<int64_t, 4> pad_output_shape_vector;
  pad_output_shape_vector.push_back(input_type.getDimSize(0));
  pad_output_shape_vector.push_back(input_type.getDimSize(1) +
                                    view.Paddings()[1].Lo() +
                                    view.Paddings()[1].Hi());
  pad_output_shape_vector.push_back(input_type.getDimSize(2) +
                                    view.Paddings()[2].Lo() +
                                    view.Paddings()[2].Hi());
  pad_output_shape_vector.push_back(input_type.getDimSize(3));
  auto pad_output_type = mlir::RankedTensorType::get(
      pad_output_shape_vector, output_type.getElementType());
  return rewriter.create<TFL::PadV2Op>(op.getLoc(), pad_output_type, input,
                                       padding_values_op, init);
}

LogicalResult LegalizeMaxPool::matchAndRewrite(
    mhlo::ReduceWindowOp op, OpAdaptor adaptor,
    ConversionPatternRewriter& rewriter) const {
  //
  // parse and validate lhs reduce window
  //=-----

  const auto opt_view = GetViewIfAttrsSupported(op);
  if (!opt_view.has_value()) {
    return rewriter.notifyMatchFailure(op, "Reduce window is not valid.");
  }
  const auto [view, layout] = opt_view.value();
  if (layout != TFLNativePoolingLayout(layout.Rank())) {
    return rewriter.notifyMatchFailure(op, "Not tfl standard layout.");
  }

  // Check that the reduce-window is a max-reduce-window.
  if (failed(MatchBinaryReduceFunction<mhlo::MaxOp>(op.getBody()))) {
    return rewriter.notifyMatchFailure(op, "Must be a max pool.");
  }

  auto type = mlir::dyn_cast<ShapedType>(op.getResult(0).getType());
  if (!mlir::isa<FloatType>(type.getElementType())) {
    return rewriter.notifyMatchFailure(op, "Not a floating point pool.");
  }

  //
  // validate inputs and init
  //=-----

  auto opt_inputs_and_init = GetInputAndInitIfValid(op);
  if (!opt_inputs_and_init.has_value()) {
    return rewriter.notifyMatchFailure(op, "Too many inputs or inits.");
  }
  auto [input, init] = opt_inputs_and_init.value();
  auto input_type = llvm::dyn_cast<ShapedType>(input.getType());

  if (!isFloatMinusInfinity(init)) {
    return rewriter.notifyMatchFailure(op, "Init not minus infinity.");
  }

  //
  // build tfl
  //=-----

  auto opt_tfl_padding =
      GetTFLPadding(view.Paddings(), view.WindowStrides(),
                    input_type.getShape(), view.WindowDims());

  Value max_pool_input;
  std::string tfl_padding_attr;
  if (opt_tfl_padding.has_value()) {
    max_pool_input = input;
    tfl_padding_attr = opt_tfl_padding.value();
  } else {
    max_pool_input = BuildExplicitPadOp(op, layout, input_type, type, input,
                                        init, view, rewriter);
    tfl_padding_attr = "VALID";
  }

  auto [fh, fw, sh, sw, p, faf] =
      BuildTFLPoolAttrs(rewriter, view, tfl_padding_attr);

  rewriter.replaceOpWithNewOp<TFL::MaxPool2DOp>(op, type, max_pool_input, p, sw,
                                                sh, fw, fh, faf);

  return success();
}

//===------------------------------------------------------------------------===
// mhlo.div(mhlo.reduce_window, cst | mhlo.reduce_window) -> tfl.avg_pool
//===------------------------------------------------------------------------===

void ReplaceWithAvgPool(mhlo::DivOp op, Value rw_lhs_input,
                        const ReduceWindowView& lhs_view,
                        llvm::StringRef padding, PatternRewriter& rewriter,
                        mhlo::TransposeOp opt_final_tpose) {
  Type out_type =
      opt_final_tpose ? opt_final_tpose.getOperand().getType() : op.getType();

  auto [fh, fw, sh, sw, p, faf] =
      BuildTFLPoolAttrs(rewriter, lhs_view, padding);
  Value final_op = rewriter.create<TFL::AveragePool2DOp>(
      op->getLoc(), out_type, rw_lhs_input, fh, fw, p, sh, sw, faf);

  if (opt_final_tpose) {
    final_op = rewriter
                   .create<mhlo::TransposeOp>(final_op.getLoc(), final_op,
                                              opt_final_tpose.getPermutation())
                   .getResult();
  }

  rewriter.replaceOp(op, final_op);
}

// Walks up the op and ignore all precedding ops of type Tys.
// Returns the first producer op whose type is not in Tys.
template <typename... Tys>
Value RecursivelyWalkUp(Value op) {
  while (llvm::isa_and_nonnull<Tys...>(op.getDefiningOp())) {
    Operation* producer = op.getDefiningOp();
    op = producer->getOperand(/*idx=*/0);
  }

  return op;
}

class LegalizeAvgPool : public OpConversionPattern<mhlo::DivOp> {
 public:
  using OpConversionPattern::OpConversionPattern;
  explicit LegalizeAvgPool(MLIRContext* context)
      : OpConversionPattern<mhlo::DivOp>(context, 10) {}
  LogicalResult matchAndRewrite(
      mhlo::DivOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const final;
};

LogicalResult LegalizeAvgPool::matchAndRewrite(
    mhlo::DivOp div_op, OpAdaptor adaptor,
    ConversionPatternRewriter& rewriter) const {
  //
  // parse and validate lhs reduce window
  //=-----

  auto div_lhs = div_op.getLhs();
  // If div's input is transposed, save it to chain on the new pool op.
  mhlo::TransposeOp opt_final_tpose;
  if (auto div_lhs_op = div_lhs.getDefiningOp()) {
    opt_final_tpose = llvm::dyn_cast_or_null<mhlo::TransposeOp>(div_lhs_op);
  }

  auto rw_lhs_val = RecursivelyWalkUp<mhlo::TransposeOp>(div_lhs);
  auto rw_lhs =
      llvm::dyn_cast_or_null<mhlo::ReduceWindowOp>(rw_lhs_val.getDefiningOp());
  if (!rw_lhs) {
    return rewriter.notifyMatchFailure(
        div_op, "Could not match lhs of div on reduce window.");
  }

  const auto opt_rw_lhs_view = GetViewIfAttrsSupported(rw_lhs);
  if (!opt_rw_lhs_view.has_value()) {
    return rewriter.notifyMatchFailure(div_op, "Lhs rw is not valid.");
  }
  const auto [rw_lhs_view, rw_lhs_layout] = opt_rw_lhs_view.value();
  if (rw_lhs_layout != TFLNativePoolingLayout(rw_lhs_layout.Rank())) {
    return rewriter.notifyMatchFailure(
        div_op, "Lhs reduce window not tfl standard layout.");
  }

  // Check that the reduce-window is a sum-reduce-window.
  if (failed(MatchBinaryReduceFunction<mhlo::AddOp>(rw_lhs.getBody()))) {
    return rewriter.notifyMatchFailure(div_op,
                                       "Failed to match rw lhs binary func.");
  }

  //
  // validate inputs and init val
  //=-----

  auto opt_rw_lhs_input_and_init = GetInputAndInitIfValid(rw_lhs);
  if (!opt_rw_lhs_input_and_init.has_value()) {
    return rewriter.notifyMatchFailure(
        div_op, "Lhs reduce window has wrong number of inputs or init values.");
  }
  auto [rw_lhs_input, rw_lhs_init_val] = opt_rw_lhs_input_and_init.value();
  auto rw_lhs_input_type = llvm::dyn_cast<ShapedType>(rw_lhs_input.getType());

  auto rw_lhs_type =
      mlir::dyn_cast<RankedTensorType>(rw_lhs.getResult(0).getType());
  if (!mlir::isa<FloatType>(rw_lhs_type.getElementType())) {
    return rewriter.notifyMatchFailure(div_op,
                                       "Reduce window lhs most be float type.");
  }

  // If the init value isn't zero then it can't be an average pool.
  if (!IsCstFloatZero(rw_lhs_init_val)) {
    return rewriter.notifyMatchFailure(
        div_op, "Reduce window lhs init value is not zero.");
  }

  //
  // case 1: rhs is splat const with val == window_size
  //=-----

  auto opt_tfl_padding =
      GetTFLPadding(rw_lhs_view.Paddings(), rw_lhs_view.WindowStrides(),
                    rw_lhs_input_type.getShape(), rw_lhs_view.WindowDims());
  if (!opt_tfl_padding.has_value()) {
    return rewriter.notifyMatchFailure(div_op,
                                       "Padding must be VALID or SAME.");
  }
  const auto& tfl_padding = opt_tfl_padding.value();

  {
    DenseFPElementsAttr divisor;
    auto div_rhs = RecursivelyWalkUp<mhlo::BroadcastInDimOp, mhlo::TransposeOp>(
        div_op.getRhs());
    if (matchPattern(div_rhs, m_Constant(&divisor))) {
      if (!divisor.isSplat()) {
        return failure();
      }

      if (!divisor.getSplatValue<APFloat>().isExactlyValue(
              rw_lhs_view.WindowSize())) {
        return rewriter.notifyMatchFailure(
            div_op, "Rhs splat const is not equal to window size.");
      }

      if (tfl_padding != "VALID") {
        return rewriter.notifyMatchFailure(div_op,
                                           "Matching on rhs splat const where "
                                           "rw lhs has non-trivial padding.");
      }

      ReplaceWithAvgPool(div_op, rw_lhs_input, rw_lhs_view, tfl_padding,
                         rewriter, opt_final_tpose);
      return success();
    }
  }

  //
  // case 2: rhs is another reduce window over 1's with same config as lhs
  //=-----

  {
    Value divisor = RecursivelyWalkUp<mhlo::BroadcastInDimOp, mhlo::ReshapeOp,
                                      mhlo::TransposeOp>(div_op.getRhs());
    auto rw_rhs =
        dyn_cast_or_null<mhlo::ReduceWindowOp>(divisor.getDefiningOp());
    if (!rw_rhs) {
      return rewriter.notifyMatchFailure(
          div_op, "Rhs of div op is not a reduce window.");
    }

    const auto opt_rw_rhs_view = GetViewIfAttrsSupported(rw_rhs);
    if (!opt_rw_rhs_view.has_value()) {
      return rewriter.notifyMatchFailure(div_op, "Rhs rw is not valid.");
    }
    const auto [rw_rhs_view, rw_rhs_layout] = opt_rw_rhs_view.value();
    if (rw_rhs_layout != TFLNativePoolingLayout(rw_rhs_layout.Rank())) {
      return rewriter.notifyMatchFailure(
          div_op, "Rhs reduce window not tfl standard layout.");
    }

    // Check that RHS is a sum-reduce-window.
    if (failed(MatchBinaryReduceFunction<mhlo::AddOp>(rw_rhs.getBody()))) {
      return rewriter.notifyMatchFailure(
          div_op, "Rhs rw body function is not an add op.");
    }

    auto opt_rw_rhs_input_and_init = GetInputAndInitIfValid(rw_rhs);
    if (!opt_rw_rhs_input_and_init.has_value()) {
      return rewriter.notifyMatchFailure(
          div_op,
          "Rhs reduce window has wrong number of inputs or init values.");
    }
    auto [rw_rhs_input, rw_rhs_init_val] = opt_rw_rhs_input_and_init.value();

    if (!IsCstFloatZero(rw_rhs_init_val)) {
      return rewriter.notifyMatchFailure(div_op,
                                         "Rhs rw init vals is not zero.");
    }

    rw_rhs_input = RecursivelyWalkUp<mhlo::BroadcastInDimOp, mhlo::TransposeOp>(
        rw_rhs_input);
    DenseFPElementsAttr rhs_input_data;
    if (!matchPattern(rw_rhs_input, m_Constant(&rhs_input_data)) ||
        !rhs_input_data.isSplat() ||
        !rhs_input_data.getSplatValue<APFloat>().isExactlyValue(1.0)) {
      return rewriter.notifyMatchFailure(div_op,
                                         "Rw rhs input is not splat of 1.0.");
    }

    // Check that the two reduce window have the same window configuration.
    if (rw_lhs.getWindowDimensions() != rw_rhs.getWindowDimensions() ||
        rw_lhs.getWindowStrides() != rw_rhs.getWindowStrides() ||
        rw_lhs.getPadding() != rw_rhs.getPadding()) {
      return rewriter.notifyMatchFailure(
          div_op, "Lhs rw and Rhs rw do not have the same config.");
    }

    ReplaceWithAvgPool(div_op, rw_lhs_input, rw_lhs_view, tfl_padding, rewriter,
                       opt_final_tpose);
    return success();
  }

  return failure();
}

}  // namespace

void PopulateLegalizeReduceWindowPatterns(MLIRContext* ctx,
                                          RewritePatternSet& patterns,
                                          ConversionTarget& target) {
  patterns.add<LegalizeAvgPool, LegalizeMaxPool, LegalizeCumSum>(ctx);
  target.addDynamicallyLegalOp<mhlo::ReduceWindowOp>(IsReduceWindowLegal);
  target.addDynamicallyLegalOp<mhlo::DivOp>(IsDivideLegal);
}

void PopulatePrepareReduceWindowPatterns(MLIRContext* ctx,
                                         RewritePatternSet& patterns) {
  patterns.add<RelayoutReduceWindow>(ctx);
}

}  // namespace mlir::odml
