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
#include "tensorflow/core/transforms/constant_folding/pass.h"

#include <algorithm>
#include <numeric>
#include <string>
#include <utility>

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/Twine.h"
#include "mlir/Dialect/Traits.h"  // from @llvm-project
#include "mlir/IR/Matchers.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/FoldUtils.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/ir/dialect.h"
#include "tensorflow/core/ir/importexport/convert_types.h"
#include "tensorflow/core/ir/importexport/export.h"
#include "tensorflow/core/transforms/pass_detail.h"
#include "tensorflow/core/transforms/utils/eval_utils.h"
#include "tensorflow/core/transforms/utils/op_cat_helper.h"
#include "tensorflow/core/transforms/utils/utils.h"
#include "tensorflow/core/util/bcast.h"
#include "tensorflow/core/util/device_name_utils.h"

namespace mlir {
namespace tfg {

// TODO(chiahungduan): Some cases we may need to support ComplexType.
template <typename T>
static std::enable_if_t<std::is_integral<T>::value, ElementsAttr>
CreateElementsAttrOfTypeValues(Type element_type, ArrayRef<int64_t> shape,
                               ArrayRef<T> values) {
  auto tensor_shape = RankedTensorType::get(shape, element_type);
  SmallVector<APInt> elements;
  for (T v : values)
    elements.push_back(APInt(element_type.getIntOrFloatBitWidth(), v));
  auto const_attr = DenseElementsAttr::get(tensor_shape, elements);
  return const_attr;
}

template <typename T>
static std::enable_if_t<std::is_floating_point<T>::value, ElementsAttr>
CreateElementsAttrOfTypeValues(Type element_type, ArrayRef<int64_t> shape,
                               ArrayRef<T> values) {
  auto tensor_shape = RankedTensorType::get(shape, element_type);
  SmallVector<APFloat> elements;
  if (element_type.getIntOrFloatBitWidth() == 32)
    llvm::for_each(values, [&](float v) { elements.push_back(APFloat(v)); });
  else
    llvm::for_each(values, [&](double v) { elements.push_back(APFloat(v)); });
  auto const_attr = DenseElementsAttr::get(tensor_shape, elements);
  return const_attr;
}

static ElementsAttr CreateElementsAttrOfTypeValues(Type element_type,
                                                   ArrayRef<int64_t> shape,
                                                   ElementsAttr value_attr) {
  auto tensor_shape = RankedTensorType::get(shape, element_type);
  DenseElementsAttr const_attr;
  if (element_type.isIntOrIndex()) {
    const_attr = DenseElementsAttr::get(
        tensor_shape, llvm::to_vector(value_attr.getValues<APInt>()));
  } else {
    const_attr = DenseElementsAttr::get(
        tensor_shape, llvm::to_vector(value_attr.getValues<APFloat>()));
  }
  return const_attr;
}

static ElementsAttr ConvertShapeToAttr(ShapedType shape) {
  return CreateElementsAttrOfTypeValues(
      IntegerType::get(shape.getContext(), 32), {shape.getRank()},
      shape.getShape());
}

static Type GetDataTypeFromOp(OpBuilder &builder, Operation *op) {
  if (auto t_attr = op->getAttrOfType<TypeAttr>("T"))
    return t_attr.getValue();
  else if (auto dtype_attr = op->getAttrOfType<TypeAttr>("dtype"))
    return dtype_attr.getValue();
  else if (op->getName().stripDialect() == "LogicalOr" ||
           op->getName().stripDialect() == "LogicalAnd")
    return builder.getI1Type();
  return *(op->result_type_begin());
}

static FailureOr<TFOp> CreateConstantTensorOp(
    OpBuilder &builder, Location loc, StringRef name_prefix, Type type,
    ValueRange control_operands, Attribute tensor_value,
    ArrayRef<NamedAttribute> other_attrs = llvm::None) {
  if (type.isa<VariantType>()) return failure();
  // TODO(chiahungduan): Reuse ConstOp Like
  // OperationFolder::tryGetOrCreateConstant.
  OperationState state(loc, "tfg.Const");
  state.addTypes({type, ControlType::get(builder.getContext())});

  state.attributes = other_attrs;
  state.attributes.set(
      "dtype", TypeAttr::get(
                   tensor_value.getType().cast<ShapedType>().getElementType()));
  state.attributes.set("value", tensor_value);

  // TODO(chiahungduan): In general, the creation of the Const operation is used
  // to replace an operation but it may be ideal to check the uniqueness of the
  // name.
  if (!state.attributes.get(TFGraphDialect::getNameAttrKey())) {
    // If the new constant op has no designed name, attach one to it so that it
    // can be translated back to NodeDef while evaluation.
    state.attributes.set(
        TFGraphDialect::getNameAttrKey(),
        builder.getStringAttr(Twine(name_prefix, "Const_folded")));
  }

  state.addOperands(control_operands);
  return TFOp(builder.create(state));
}

static FailureOr<TFOp> ReplaceOpWithConstantTensor(
    OpBuilder &builder, TFOp op, ElementsAttr value,
    ArrayRef<StringRef> exclude_attrs = llvm::None) {
  // New const op has the control dependency with op's non-control operands.
  SmallVector<Value> operands_controls;
  llvm::append_range(operands_controls,
                     OperandControlRetRange(op.getNonControlOperands()));

  NamedAttrList attr_list;
  for (NamedAttribute attr : op->getAttrs()) {
    if (llvm::find_if(exclude_attrs,
                      [&](StringRef name) { return name == attr.getName(); }))
      continue;
    attr_list.append(attr);
  }
  return CreateConstantTensorOp(builder, op->getLoc(),
                                op->getName().getStringRef(), value.getType(),
                                operands_controls, value, attr_list);
}

static FailureOr<TFOp> ReplaceOpWithIdentity(OpBuilder &builder, TFOp owner,
                                             unsigned idx) {
  OperationState state(owner->getLoc(), "tfg.Identity");
  state.addTypes({owner->getOperand(idx).getType(),
                  ControlType::get(builder.getContext())});
  state.addAttribute(
      "T", TypeAttr::get(GetDataTypeFromOp(builder, owner.getOperation())));
  state.addAttribute(TFGraphDialect::getNameAttrKey(),
                     builder.getStringAttr(Twine(owner.name(), "/Identity")));

  Value kept_value = owner->getOperand(idx);
  state.addOperands(kept_value);
  Value kept_value_control_ret = LookupControlDependency(kept_value);

  for (Value control_ret : OperandControlRetRange(owner->getOperands())) {
    if (control_ret != kept_value_control_ret) state.addOperands(control_ret);
  }

  state.addOperands(owner.getControlOperands());

  return TFOp(builder.create(state));
}

static FailureOr<TFOp> ReplaceOperationWithConstant(OpBuilder &builder,
                                                    Operation *op,
                                                    double constant_value) {
  auto res = (*op->result_type_begin()).cast<ShapedType>();
  Type dtype = GetDataTypeFromOp(builder, op);
  Attribute value_attr;
  if (dtype.isIntOrIndex())
    value_attr = builder.getIntegerAttr(dtype, constant_value);
  else
    value_attr = builder.getFloatAttr(dtype, constant_value);

  auto const_attr = SplatElementsAttr::get(
      RankedTensorType::get(res.getShape(), dtype), value_attr);
  return ReplaceOpWithConstantTensor(builder, op, const_attr);
}

static FailureOr<TFOp> ReplaceOperationWithSnapshot(OpBuilder &builder, TFOp op,
                                                    int idx) {
  // TODO(chiahungduan): if (!graph_contains_assign_or_inplace_op_)

  Value replace_value = op->getOperand(idx);
  OperationState state(op->getLoc(), "tfg.Snapshot");
  state.addAttribute(
      "T", TypeAttr::get(GetDataTypeFromOp(builder, op.getOperation())));
  // Propagate the designated input through the Snapshot.
  state.addOperands(replace_value);
  // Add all other inputs as control dependencies.
  llvm::append_range(state.operands,
                     OperandControlRetRange(op.getNonControlOperands()));

  return TFOp(builder.create(state));
}

static FailureOr<TFOp> ReplaceOperationWithBroadcastTo(OpBuilder &builder,
                                                       TFOp op,
                                                       int idx_to_replace) {
  ShapedType tensor_type = (*op->result_type_begin()).cast<ShapedType>();
  ElementsAttr const_attr = ConvertShapeToAttr(tensor_type);
  SmallVector<Value> control_operands;
  for (auto &it : llvm::enumerate(op.getNonControlOperands())) {
    int idx = it.index();
    Value v = it.value();
    if (idx == idx_to_replace) continue;
    control_operands.push_back(LookupControlDependency(v));
  }
  FailureOr<TFOp> const_op = CreateConstantTensorOp(
      builder, op->getLoc(), op->getName().getStringRef(), const_attr.getType(),
      control_operands, const_attr);
  if (failed(const_op)) return failure();
  const_op->setRequestedDevice(op.device());

  OperationState state(op->getLoc(), "tfg.BroadcastTo");

  state.addAttribute(
      "T", TypeAttr::get(GetDataTypeFromOp(builder, op.getOperation())));
  state.addAttribute("Tidx", TypeAttr::get(builder.getI32Type()));

  state.addOperands(
      {op->getOperand(idx_to_replace), (*const_op)->getResult(0)});
  for (Value v : op.getNonControlOperands())
    if (v != op->getOperand(idx_to_replace)) state.addOperands(v);
  state.addTypes(op->getResultTypes());
  return TFOp(builder.create(state));
}

namespace {
// A helper class to see if an operation falls into certain category or has
// certain non-trivial properties.
class OpPropertyHelper : public OpCatHelper {
 public:
  OpPropertyHelper(MLIRContext *context,
                   ArrayRef<std::string> nodes_to_preserve,
                   bool disable_compressed_tensor_optimization)
      : OpCatHelper(context),
        nodes_to_preserve_(nodes_to_preserve.begin(), nodes_to_preserve.end()),
        disable_compressed_tensor_optimization_(
            disable_compressed_tensor_optimization) {}

  // Return true if the operation modifies the input in-place.
  bool ModifiesInputsInPlace(TFOp op);

  // Return true if this operation doesn't have any side effect.
  bool IsFreeOfSideEffect(TFOp op);

  // Return true if an operation may modify the frame info.
  bool ModifiesFrameInfo(TFOp op) {
    return IsEnter(op) || IsExit(op) || IsNextIteration(op);
  }

  // This combines the results of both MaybeFoldable() and IsFoldableUncached()
  bool IsFoldable(TFOp op);

  // Return if this is a preserved op. It checks the `name` attr.
  bool ShouldPreserveOp(TFOp op);

  // Disable compressed tensor optimization.
  bool DisableCompressedTensorOptimization();

 private:
  // Return true if this operation is safe to be folded. This filter the ops by
  // name.
  bool MaybeFoldable(TFOp op);

  // Return true if this operation is safe to be folded. This filter the ops by
  // the operation property like, it'll check the operands, attributes, .etc.
  bool IsFoldableUncached(TFOp op);

  // The list of op names which should be preserved.
  DenseSet<StringRef> nodes_to_preserve_;

  // Indicate that if we've disabled compressed tensor optimization.
  bool disable_compressed_tensor_optimization_;

  // We only fold/materialize constants smaller than 100kB.
  static constexpr int64_t kMaxConstantSize = 100 * 1024;
};
}  // namespace

bool OpPropertyHelper::ModifiesInputsInPlace(TFOp op) {
  StringRef op_name = op->getName().stripDialect();

  // Ops that modify resource variables effectively modify one of their inputs.
  if (op_name == "AssignVariableOp" || op_name == "AssignAddVariableOp" ||
      op_name == "AssignSubVariableOp" || op_name == "ResourceScatterUpdate" ||
      op_name == "ResourceScatterAdd" || op_name == "ResourceScatterSub" ||
      op_name == "ResourceScatterMul" || op_name == "ResourceScatterDiv" ||
      op_name == "ResourceScatterMin" || op_name == "ResourceScatterMax")
    return false;

  std::string lower_op_name = op_name.str();
  std::transform(lower_op_name.begin(), lower_op_name.end(),
                 lower_op_name.begin(), ::tolower);
  if (absl::StrContains(lower_op_name, "inplace")) return true;

  return op->hasAttr("in_place") || op->hasAttr("inplace");
}

bool OpPropertyHelper::IsFreeOfSideEffect(TFOp op) {
  tensorflow::OpRegistry *op_registry = tensorflow::OpRegistry::Global();
  const tensorflow::OpDef *op_def;
  tensorflow::Status status =
      op_registry->LookUpOpDef(op->getName().stripDialect().str(), &op_def);
  if (!status.ok()) return false;

  if (op_def->is_stateful()) return false;

  for (const auto &input : op_def->input_arg())
    if (input.is_ref()) return false;

  if (IsQueue(op)) return false;

  if (IsSend(op)) return false;

  return !ModifiesInputsInPlace(op);
}

// To determine if we want to evalue the value of the operation. There several
// kinds operation we don't want to evalute with the eager runtime. Those
// operations may not safe for evaluation or not worth for evaluating because of
// the evaluation cost. For example, Const op already has the constant value
// attached as attribute.
bool OpPropertyHelper::MaybeFoldable(TFOp op) {
  StringRef op_name = op->getName().stripDialect();

  if (IsConstant(op)) return false;

  // Don't fold stateful ops such as TruncatedNormal.
  if (!IsFreeOfSideEffect(op)) return false;

  // TODO(chiahungduan): Handle preserve nodes

  // Skips ops that don't benefit from folding.
  if (IsPlaceholder(op)) return false;

  if (IsFakeParam(op)) return false;

  // Skip certain control flow nodes, they can't be folded.
  if (ModifiesFrameInfo(op)) return false;

  if (op_name == "AccumulateNV2") return false;

  // Removing LoopCond nodes can screw up the partitioner.
  if (op_name == "LoopCond") return false;

  // TODO(chiahungduan): add fold_quantization_emulation arg.
  // if (!fold_quantization_emulation && IsQuantizationEmulation(op)) return
  // false;

  if (IsRestore(op) || op_name.find("Save") != StringRef::npos ||
      op_name.find("Reader") != StringRef::npos)
    return false;

  if (op_name.find("Quantized") != StringRef::npos ||
      absl::StartsWith(op_name.data(), "Sparse"))
    return false;

  // Don't fold nodes that contain TPU attributes.
  // TODO(rmlarsen): We should be able to fold many of these nodes as long as we
  // properly forward custom attributes, b/119051778.
  for (NamedAttribute attr : op->getAttrs())
    if (attr.getName().strref().find("_tpu_") != StringRef::npos) return false;

  // Don't fold ops without outputs. Note that almost all tfg op has additional
  // control output value.
  if (op->getNumResults() <= 1) return false;

  const tensorflow::OpDef *op_def = nullptr;
  tensorflow::Status status = tensorflow::OpRegistry::Global()->LookUpOpDef(
      op->getName().stripDialect().str(), &op_def);
  if (!status.ok()) {
    return false;
  }
  // Don't fold ops without outputs.
  if (op_def->output_arg_size() == 0) {
    return false;
  }

  // Don't fold DT_VARIANT outputs as this can cause problems with XLA compile.
  // TODO(rmlarsen): Only do this for XLA_* devices.
  for (const tensorflow::OpDef::ArgDef &output_arg : op_def->output_arg()) {
    if (output_arg.type() == tensorflow::DT_VARIANT) {
      return false;
    }
  }

  // Don't fold nodes that have no outgoing edges except allowlisted nodes.
  // Such nodes could be introduced by an earlier constant folding pass and are
  // preserved in case users want to fetch their values; re-processing them
  // would lead to an error of adding a duplicated node to graph.
  // TODO(chiahungduan): Op has no users and doesn't in nodes_allowlist_ can't
  // be folded.
  return true;
}

bool OpPropertyHelper::IsFoldableUncached(TFOp op) {
  ValueRange operands = op.getNonControlOperands();
  if (operands.empty()) return false;

  // We can only fold nodes if all their inputs are known statically, except in
  // the case of a merge node that propagate the first inputs that becomes
  // available, and therefore only requires a single constant input to be
  // foldable.
  bool merge_has_constant_input = false;
  bool is_merge = IsMerge(op);
  for (Value operand : operands) {
    TFOp operand_op = operand.getDefiningOp();
    if (operand_op && IsConstant(operand_op)) {
      auto dtype = operand_op->getAttrOfType<TypeAttr>("dtype");
      if (!dtype || dtype.getValue().isa<tf_type::StringType>()) return false;

      // Special case: If a Merge node has at least one constant input that
      // does not depend on a control input, we can fold it.
      merge_has_constant_input |= operand_op.getControlOperands().empty();
    } else if (!is_merge) {
      return false;
    }
  }

  if (is_merge && !merge_has_constant_input) return false;
  if (DisableCompressedTensorOptimization() &&
      (IsFill(op) || IsZerosLike(op) || IsOnesLike(op)))
    return false;

  // If we know the output shapes, make sure that the outputs are small enough
  // to materialize.
  int64_t input_size_bytes = 0;
  for (Value operand : operands) {
    auto shape = operand.getType().dyn_cast<ShapedType>();
    if (!shape || !shape.hasStaticShape()) continue;
    auto element_type = shape.getElementType();

    tensorflow::DataType dtype;
    if (!ConvertScalarTypeToDataType(element_type, &dtype).ok()) return false;
    input_size_bytes += shape.getNumElements() * DataTypeSize(dtype);
  }
  for (Value res : op->getResults().drop_back()) {
    auto shape = res.getType().dyn_cast<ShapedType>();
    if (!shape || !shape.hasStaticShape()) continue;
    auto element_type = shape.getElementType();

    tensorflow::DataType dtype;
    if (!ConvertScalarTypeToDataType(element_type, &dtype).ok()) return false;
    int64_t num_bytes = shape.getNumElements() * DataTypeSize(dtype);
    if (num_bytes > input_size_bytes && num_bytes > kMaxConstantSize)
      return false;
  }

  return true;
}

bool OpPropertyHelper::IsFoldable(TFOp op) {
  // TODO(chiahungduan): Cache foldable ops
  if (!MaybeFoldable(op)) return false;
  return IsFoldableUncached(op);
}

bool OpPropertyHelper::ShouldPreserveOp(TFOp op) {
  return nodes_to_preserve_.contains(op.name());
}

bool OpPropertyHelper::DisableCompressedTensorOptimization() {
  return disable_compressed_tensor_optimization_;
}

static bool IsValidConstShapeForMulConvPushDown(StringAttr data_format,
                                                ShapedType filter_shape,
                                                ShapedType const_shape) {
  if (const_shape.getRank() <= data_format.size() &&
      const_shape.getNumElements() == 1)
    return true;
  if (data_format == "NHWC" || data_format == "NDHWC") {
    SmallVector<int64_t> broadcast_shape;
    if (!OpTrait::util::getBroadcastedShape(
            filter_shape.getShape(), const_shape.getShape(), broadcast_shape))
      return false;

    if (filter_shape.getShape() != llvm::makeArrayRef(broadcast_shape))
      return false;

    // Only the last dimension could be larger than one, since broadcasting over
    // the last dimension (the output channel) will result in invalid filter.
    for (int dim_size : const_shape.getShape())
      if (dim_size > 1) return false;
    return true;
  } else if (data_format == "NCHW" || data_format == "NCDHW") {
    // TODO(laigd): support NCHW and NCDHW (b/111214513).
    return false;
  }
  return false;
}

namespace {
class FolderPatternBase : public RewritePattern {
 public:
  FolderPatternBase(StringRef opName, MLIRContext *context,
                    OpPropertyHelper &helper)
      : RewritePattern(opName, PatternBenefit(1), context), helper_(helper) {}
  FolderPatternBase(MatchAnyOpTypeTag tag, MLIRContext *context,
                    OpPropertyHelper &helper)
      : RewritePattern(tag, PatternBenefit(1), context), helper_(helper) {}

 protected:
  OpPropertyHelper &helper_;
};
}  // namespace

// EvaluateConstant maps the implementation of FoldGraph in
// ConstantFolding::FoldGraph in grappler/optimizers/constant_folding.cc
class EvaluateConstant : public FolderPatternBase {
 public:
  EvaluateConstant(MLIRContext *context, OpPropertyHelper &helper)
      : FolderPatternBase(MatchAnyOpTypeTag(), context, helper),
        cpu_device_(std::make_unique<util::SimpleDevice>()),
        resource_mgr_(std::make_unique<tensorflow::ResourceMgr>()) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (!helper_.IsFoldable(op)) return failure();

    SmallVector<ElementsAttr> const_operands;
    for (Value operand : TFOp(op).getNonControlOperands()) {
      Operation *defining_op = operand.getDefiningOp();
      if (defining_op && helper_.IsConstant(defining_op)) {
        const_operands.push_back(
            defining_op->getAttrOfType<ElementsAttr>("value"));
      } else {
        return failure();
      }
    }

    SmallVector<Attribute> result;
    if (failed(util::EvaluateOperation(cpu_device_.get(), resource_mgr_.get(),
                                       op, const_operands, result)))
      return failure();

    StringAttr name_attr = static_cast<TFGraphDialect *>(op->getDialect())
                               ->getNameAttrIdentifier();
    std::pair<OperandRange, OperandRange> operands = TFOp(op).splitOperands();
    SmallVector<Value> control_operands;
    llvm::append_range(control_operands,
                       OperandControlRetRange(operands.first));
    llvm::append_range(control_operands, operands.second);

    StringAttr device_attr = TFOp(op).deviceAttr();
    SmallVector<Value> const_values;
    for (auto &it : llvm::enumerate(result)) {
      Attribute attr = it.value();
      FailureOr<TFOp> const_op = CreateConstantTensorOp(
          rewriter, op->getLoc(),
          (Twine(TFOp(op).name(), "/eval-") + Twine(it.index())).str(),
          attr.getType().cast<ShapedType>(), control_operands, attr,
          NamedAttribute(name_attr, TFOp(op).nameAttr()));
      if (failed(const_op)) return failure();
      if (device_attr) (*const_op).setRequestedDevice(device_attr);
      const_values.emplace_back(*((*const_op)->result_begin()));
      // TODO(chiahungduan): Review the following comments.
      // Create an empty NodeDef to identify dead outputs (e.g. the
      // output of a
      // switch that's not selected by the switch predicate).
      // if (output_tensors[i].tensor)
      //   outputs->at(i) = NodeDef();
    }

    // Replace the control edge to one of the constant value.
    // TODO(chiahungduan): In grappler, it adds the edge to the last const
    // value and I think that's just implemention defined behavior.
    Operation *control_repl = const_values[0].getDefiningOp();
    const_values.push_back(control_repl->getResults().back());

    rewriter.replaceOp(op, const_values);

    return success();
  }

 private:
  std::unique_ptr<util::SimpleDevice> cpu_device_;
  std::unique_ptr<tensorflow::ResourceMgr> resource_mgr_;
};

// This implementation is mapped to the ShapeOp materialization in
// ConstantFolding::MaterializeShapes in grappler/optimizers/constant_folding.cc
class MaterializeShapeOp : public FolderPatternBase {
 public:
  MaterializeShapeOp(MLIRContext *context, OpPropertyHelper &helper)
      : FolderPatternBase("tfg.Shape", context, helper) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    Value input = op->getOperand(0);
    if (!input.getDefiningOp()) return failure();

    // TODO(rmlarsen): Remove this workaround for b/150861569
    // The bug involves an expression of the form Shape(ExpandDims(x)
    // with an incorrectly inferred zero-size first dimension.
    auto input_shape = input.getType().cast<ShapedType>();
    if (!input_shape.hasStaticShape()) return failure();

    if (!input_shape.getShape().empty() && input_shape.getShape()[0] == 0)
      return failure();

    ElementsAttr const_attr = ConvertShapeToAttr(input_shape);

    // Add the control edge to `input` to ensure that the constant value will
    // only be run in the cases where Shape would have been run in the original
    // graph.
    FailureOr<TFOp> const_op = CreateConstantTensorOp(
        rewriter, op->getLoc(), op->getName().getStringRef(),
        const_attr.getType(), TFOp(input.getDefiningOp()).controlRet(),
        const_attr, op->getAttrs());
    if (failed(const_op)) return failure();

    rewriter.replaceOp(op, (*const_op)->getResults());

    return success();
  }
};

// This implementation is mapped to the SizeOp materialization in
// ConstantFolding::MaterializeShapes in grappler/optimizers/constant_folding.cc
class MaterializeSizeOp : public FolderPatternBase {
 public:
  MaterializeSizeOp(MLIRContext *context, OpPropertyHelper &helper)
      : FolderPatternBase("tfg.Size", context, helper) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    Value input = op->getOperand(0);

    auto input_shape = input.getType().cast<ShapedType>();
    if (!input_shape.hasStaticShape()) return failure();

    ShapedType result_type = (*op->result_type_begin()).cast<ShapedType>();
    ElementsAttr const_attr = CreateElementsAttrOfTypeValues(
        result_type.getElementType(), {},
        ArrayRef<int64_t>(input_shape.getNumElements()));

    // Add the control edge to `input` to ensure that the constant value will
    // only be run in the cases where Size would have been run in the original
    // graph.
    FailureOr<TFOp> const_op = CreateConstantTensorOp(
        rewriter, op->getLoc(), op->getName().getStringRef(),
        const_attr.getType(), LookupControlDependency(input), const_attr,
        op->getAttrs());
    if (failed(const_op)) return failure();

    rewriter.replaceOp(op, (*const_op)->getResults());

    return success();
  }
};

// This implementation is mapped to the RankOp materialization in
// ConstantFolding::MaterializeShapes in grappler/optimizers/constant_folding.cc
class MaterializeRankOp : public FolderPatternBase {
 public:
  MaterializeRankOp(MLIRContext *context, OpPropertyHelper &helper)
      : FolderPatternBase("tfg.Rank", context, helper) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    Value input = op->getOperand(0);

    auto input_shape = input.getType().cast<ShapedType>();
    if (!input_shape.hasRank()) return failure();

    ShapedType result_type = (*op->result_type_begin()).cast<ShapedType>();
    ElementsAttr const_attr = CreateElementsAttrOfTypeValues(
        result_type.getElementType(), {}, ArrayRef<int>(input_shape.getRank()));

    // Add the control edge to `input` to ensure that the constant value will
    // only be run in the cases where Rank would have been run in the original
    // graph.
    FailureOr<TFOp> const_op = CreateConstantTensorOp(
        rewriter, op->getLoc(), op->getName().getStringRef(),
        const_attr.getType(), LookupControlDependency(input), const_attr,
        op->getAttrs());
    if (failed(const_op)) return failure();

    rewriter.replaceOp(op, (*const_op)->getResults());

    return success();
  }
};

// This implementation is mapped to the TensorArraySizeV3 materialization in
// ConstantFolding::MaterializeShapes in grappler/optimizers/constant_folding.cc
class MaterializeTensorArraySizeV3Op : public FolderPatternBase {
 public:
  MaterializeTensorArraySizeV3Op(MLIRContext *context, OpPropertyHelper &helper)
      : FolderPatternBase("tfg.TensorArraySizeV3", context, helper) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    Operation *handle_op = op->getOperand(0).getDefiningOp();
    if (!handle_op) return failure();

    auto dynamic_size = handle_op->getAttrOfType<BoolAttr>("dynamic_size");
    if (dynamic_size && dynamic_size.getValue()) return failure();

    Operation *array_size = handle_op->getOperand(0).getDefiningOp();
    if (!array_size || !helper_.IsConstant(array_size)) return failure();

    // Don't materialize 0 sizes to avoid triggering incorrect static checks.
    // A 0 sized array that can't grow isn't useful anyway.
    auto size_attr = array_size->getAttrOfType<SplatElementsAttr>("value");
    if (!size_attr || !size_attr.getElementType().isInteger(32))
      return failure();
    if (size_attr.getSplatValue<IntegerAttr>().getInt() == 0) return failure();

    SmallVector<Value> control_operands;
    control_operands.push_back(TFOp(handle_op).controlRet());
    control_operands.push_back(LookupControlDependency(op->getOperand(1)));
    FailureOr<TFOp> const_op = CreateConstantTensorOp(
        rewriter, op->getLoc(), op->getName().getStringRef(),
        size_attr.getType(), control_operands, size_attr, op->getAttrs());
    if (failed(const_op)) return failure();

    rewriter.replaceOp(op, (*const_op)->getResults());

    return failure();
  }
};

// This implementation is mapped to the ShapeN materialization in
// ConstantFolding::MaterializeShapes in grappler/optimizers/constant_folding.cc
class MaterializeShapeNOp : public FolderPatternBase {
 public:
  MaterializeShapeNOp(MLIRContext *context, OpPropertyHelper &helper)
      : FolderPatternBase("tfg.ShapeN", context, helper) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    for (const auto &it : llvm::enumerate(TFOp(op).getNonControlOperands())) {
      Value operand = op->getOperand(it.index());

      auto operand_shape = operand.getType().cast<ShapedType>();
      if (!operand_shape.hasStaticShape()) continue;

      if (op->getResults()[it.index()].use_empty()) continue;

      ElementsAttr const_attr = ConvertShapeToAttr(operand_shape);

      FailureOr<TFOp> const_op = CreateConstantTensorOp(
          rewriter, op->getLoc(), op->getName().getStringRef(),
          *(op->result_type_begin()), TFOp(op).controlRet(), const_attr);
      if (failed(const_op)) return failure();

      (*const_op).setName(Twine(TFOp(op).name(), "/-matshapes-") +
                          std::to_string(it.index()));
      if (!TFOp(op).device().empty())
        (*const_op).setRequestedDevice(TFOp(op).deviceAttr());

      // TODO(chiahungduan): Do we need to handle `direct_edges_exist` in
      // ConstantFolding::MaterializeShapes for ShapeN?

      for (OpOperand &user : op->getResult(it.index()).getUses()) {
        rewriter.startRootUpdate(user.getOwner());
        user.set((*const_op)->getResult(0));
        rewriter.finalizeRootUpdate(user.getOwner());
      }
    }

    return success();
  }
};

// This implementation is mapped to the BroadcastGradientArgsOp materialization
// in ConstantFolding::MaterializeBroadcastGradientArgs in
// grappler/optimizers/constant_folding.cc
class MaterializeBroadcastGradientArgsOp : public FolderPatternBase {
 public:
  MaterializeBroadcastGradientArgsOp(MLIRContext *context,
                                     OpPropertyHelper &helper)
      : FolderPatternBase("tfg.BroadcastGradientArgs", context, helper) {}
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    Operation *s0 = op->getOperand(0).getDefiningOp();
    Operation *s1 = op->getOperand(1).getDefiningOp();
    if (!s0 || !s1) return failure();

    if (!helper_.IsShape(s0) && !helper_.IsConstant(s0)) return failure();
    if (!helper_.IsShape(s1) && !helper_.IsConstant(s1)) return failure();

    // This operation has been optimized.
    if (op->getResult(0).use_empty() && op->getResult(1).use_empty())
      return failure();

    auto getShape = [&](Operation *op, ArrayRef<int64_t> shape) -> bool {
      ShapedType shaped_type;
      if (helper_.IsShape(op)) {
        auto type = op->getOperand(0).getType().cast<ShapedType>();
        if (!type.hasRank()) return false;
        shape = type.getShape();
      } else {
        auto attr = op->getAttrOfType<ElementsAttr>("value");
        if (!attr) return false;
        if (!attr.getElementType().isInteger(32) &&
            !attr.getElementType().isInteger(64))
          return false;
        shape = attr.getType().cast<ShapedType>().getShape();
      }
      return true;
    };

    ArrayRef<int64_t> s0_shape;
    ArrayRef<int64_t> s1_shape;
    if (!getShape(s0, s0_shape) || !getShape(s1, s1_shape)) return failure();

    // TODO(chiahungduan): The Dim::size of TensorShapeProto is supposed to be
    // greater than or equal to -1, but in Grappler, it seems that the shape
    // inference will fill some values less than -1. Check that then fix the
    // logic here.
    const int common_dims = std::min(s0_shape.size(), s1_shape.size());
    for (int i = 0; i < common_dims; ++i) {
      if (s0_shape[i] >= 0 && s1_shape[i] >= 0) continue;
      return failure();
    }
    for (int i = common_dims; i < s0_shape.size(); ++i)
      if (s0_shape[i] < 0) return failure();
    for (int i = common_dims; i < s1_shape.size(); ++i)
      if (s1_shape[i] < 0) return failure();

    // TODO(chiahungduan): Refactor the computation down below.
    tensorflow::BCast::Vec s0_vec(s0_shape.begin(), s0_shape.end());
    tensorflow::BCast::Vec s1_vec(s1_shape.begin(), s1_shape.end());
    tensorflow::BCast bcast(s0_vec, s1_vec);
    if (!bcast.IsValid()) return failure();

    tensorflow::BCast::Vec reduce_dims[2];
    reduce_dims[0] = bcast.grad_x_reduce_idx();
    reduce_dims[1] = bcast.grad_y_reduce_idx();

    auto type_attr = op->getAttr("T").dyn_cast_or_null<TypeAttr>();
    if (!type_attr) return failure();

    SmallVector<Value, 2> const_values;
    for (int j = 0; j < 2; ++j) {
      int reduction_indices = reduce_dims[j].size();
      ElementsAttr const_attr = CreateElementsAttrOfTypeValues(
          type_attr.getValue(), {reduction_indices},
          llvm::makeArrayRef<int64_t>(reduce_dims[j].data(),
                                      reduce_dims[j].size()));
      FailureOr<TFOp> const_op = CreateConstantTensorOp(
          rewriter, op->getLoc(), op->getName().getStringRef(),
          *(op->result_type_begin()), TFOp(op).controlRet(), const_attr);
      if (failed(const_op)) return failure();
      (*const_op).setName(Twine(TFOp(op).name(), "-bcastargs-") +
                          std::to_string(j));
      const_values.push_back((*const_op)->getResult(0));
    }

    for (OpOperand &user : op->getResult(0).getUses()) {
      rewriter.startRootUpdate(user.getOwner());
      user.set(const_values[0]);
      rewriter.finalizeRootUpdate(user.getOwner());
    }
    for (OpOperand &user : op->getResult(1).getUses()) {
      rewriter.startRootUpdate(user.getOwner());
      user.set(const_values[1]);
      rewriter.finalizeRootUpdate(user.getOwner());
    }

    return success();
  }
};

// This implementation is mapped to the indices of reduction ops materialization
// in ConstantFolding::MaterializeReductionIndices in
// grappler/optimizers/constant_folding.cc
class MaterializeReductionIndices : public FolderPatternBase {
 public:
  MaterializeReductionIndices(MLIRContext *context, OpPropertyHelper &helper)
      : FolderPatternBase(MatchAnyOpTypeTag(), context, helper) {}
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (!helper_.IsReduction(op)) return failure();

    Operation *indices = op->getOperand(1).getDefiningOp();
    if (!indices || helper_.IsConstant(indices)) return failure();

    auto indices_shape = indices->getResult(0).getType().cast<ShapedType>();
    if (!indices_shape.getElementType().isInteger(32) &&
        indices_shape.getElementType().isInteger(64))
      return failure();

    auto input_shape = op->getOperand(0).getType().cast<ShapedType>();
    // Unexpected graph, don't try to change it.
    if (input_shape.getRank() < 1) return failure();

    auto output_shape = op->getResult(0).getType().cast<ShapedType>();
    const int output_rank =
        output_shape.hasRank() ? output_shape.getRank() : -1;

    bool full_reduction = output_rank == 0 || indices_shape.getNumElements() ==
                                                  input_shape.getRank();

    if (!full_reduction) {
      // TODO(chiahungduan): The logic of computing `full_reduction` looks weird
      // in grappler, verify it again.
      for (Operation *user : op->getResult(0).getUsers()) {
        full_reduction = false;
        if (!helper_.IsReshape(user)) return failure();

        auto shape = user->getResult(0).getType().cast<ShapedType>();
        if (!shape.hasStaticShape() || shape.getNumElements() != 1)
          return failure();
        else
          full_reduction = true;
      }
      if (!full_reduction) return failure();
    }

    SmallVector<APInt> elements(indices_shape.getNumElements());
    for (unsigned i = 0; i < indices_shape.getNumElements(); ++i)
      elements[i] =
          APInt(indices_shape.getElementType().getIntOrFloatBitWidth(), i);
    DenseElementsAttr const_attr =
        DenseElementsAttr::get(indices_shape, elements);

    FailureOr<TFOp> const_op = CreateConstantTensorOp(
        rewriter, indices->getLoc(), indices->getName().getStringRef(),
        const_attr.getType(), TFOp(indices).controlRet(), const_attr);
    if (failed(const_op)) return failure();
    rewriter.startRootUpdate(op);
    op->setOperand(1, (*const_op)->getResults()[0]);

    if (TFOp(op).deviceAttr())
      (*const_op).setRequestedDevice(TFOp(op).deviceAttr());
    rewriter.finalizeRootUpdate(op);

    return success();
  }
};

// This implementation is mapped to the constant value materialization in
// ConstantFolding::MaterializeConstantValuedNode in
// grappler/optimizers/constant_folding.cc
class MaterializeFillNode : public FolderPatternBase {
 public:
  MaterializeFillNode(MLIRContext *context, OpPropertyHelper &helper)
      : FolderPatternBase("tfg.Fill", context, helper) {}
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (helper_.DisableCompressedTensorOptimization()) return failure();

    auto output_type = op->getResult(0).getType().cast<ShapedType>();
    if (!output_type.hasStaticShape()) return failure();

    for (Value operand : TFOp(op).getNonControlOperands()) {
      if (!operand.getDefiningOp() ||
          !helper_.IsConstant(operand.getDefiningOp()))
        return failure();
    }

    Operation *dim = op->getOperand(0).getDefiningOp();
    Operation *value = op->getOperand(1).getDefiningOp();
    if (!dim || !value) return failure();

    ElementsAttr dim_attr = dim->getAttrOfType<ElementsAttr>("value");

    ElementsAttr const_attr = CreateElementsAttrOfTypeValues(
        output_type.getElementType(),
        dim_attr.getType().cast<ShapedType>().getShape(),
        {value->getAttrOfType<ElementsAttr>("value")});

    FailureOr<TFOp> const_op = ReplaceOpWithConstantTensor(
        rewriter, op, const_attr, ArrayRef<StringRef>({"T", "index_type"}));
    if (failed(const_op)) return failure();

    rewriter.replaceOp(op, (*const_op)->getResults());

    return success();
  }
};

// This implementation is mapped to the constant value materialization in
// ConstantFolding::MaterializeConstantValuedNode in
// grappler/optimizers/constant_folding.cc
class MaterializeConstantValuedNode : public FolderPatternBase {
 public:
  MaterializeConstantValuedNode(MLIRContext *context, OpPropertyHelper &helper)
      : FolderPatternBase(MatchAnyOpTypeTag(), context, helper) {}
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (helper_.DisableCompressedTensorOptimization()) return failure();

    // TODO(chiahungduan): disable_compressed_tensor_optimization_

    // FillOp is handled in MaterializeFillNode pattern.
    if (helper_.IsFill(op)) return failure();
    if (!helper_.IsZerosLike(op) && !helper_.IsOnesLike(op)) return failure();

    // TODO(chiahungduan): If op->getOperand(0) has static shape, can we use
    // that to materialize?
    auto output_type = op->getResult(0).getType().cast<ShapedType>();
    if (!output_type.hasStaticShape()) return failure();

    int value = helper_.IsZerosLike(op) ? 0 : (helper_.IsOnesLike(op) ? 1 : -1);
    if (value < 0) return failure();

    ElementsAttr const_attr;
    if (output_type.getElementType().isIntOrIndex()) {
      const_attr = CreateElementsAttrOfTypeValues(output_type.getElementType(),
                                                  output_type.getShape(),
                                                  ArrayRef<int>(value));
    } else {
      const_attr = CreateElementsAttrOfTypeValues(output_type.getElementType(),
                                                  output_type.getShape(),
                                                  ArrayRef<double>(value));
    }

    FailureOr<TFOp> const_op =
        ReplaceOpWithConstantTensor(rewriter, op, const_attr);
    if (failed(const_op)) return failure();

    rewriter.replaceOp(op, (*const_op)->getResults());
    return failure();
  }
};

// This implementation is mapped to the output value materialization in
// ConstantFolding::MaterializeOutputValues in
// grappler/optimizers/constant_folding.cc
class MaterializeOutputValue : public FolderPatternBase {
 public:
  MaterializeOutputValue(MLIRContext *context, OpPropertyHelper &helper)
      : FolderPatternBase(MatchAnyOpTypeTag(), context, helper) {}
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    // In grappler, the shape information is stored in a separate structure and
    // this pass is used to materialize the shape inference information to the
    // node. But in MLIR, the shape inference information is stored in the
    // operation.
    return failure();
  }
};

// This implementation is mapped to the merge node folding in
// ConstantFolding::FoldMergeNode in
// grappler/optimizers/constant_folding.cc
class MergeNodeFoldingBase : public FolderPatternBase {
 protected:
  MergeNodeFoldingBase(StringRef op_name, MLIRContext *context,
                       OpPropertyHelper &helper)
      : FolderPatternBase(op_name, context, helper),
        zero_dim_i32_tensor_type_(
            RankedTensorType::get(llvm::None, IntegerType::get(context, 32))) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    // Merge nodes are special, in the sense that they execute as soon as one of
    // their input is ready. We can therefore fold a merge node iff it has at
    // least one constant input without control dependency.
    // We still need to ensure that the nodes in the fanin of the merge node are
    // scheduled. We'll therefore add a control dependency from the merge node
    // to the folded constant. We end up with:
    //  * the merge node and its inputs are preserved as is
    //  * a new constant node C1, driven by the merge node through a control
    //  dependency, initialized to the value of the folded input
    //  * a new constant node C2, driven by the merge node through a control
    //  dependency, initialized to the index of the folded input
    //  * the fanout of the merge nodes is rewired to be driven by either C1 or
    //  C2.
    if (op->use_empty()) return failure();

    int idx = 0;
    for (Value operand : TFOp(op).getNonControlOperands()) {
      Operation *operand_op = operand.getDefiningOp();
      if (!operand_op) continue;
      if (!helper_.IsConstant(operand_op)) continue;
      if (!TFOp(operand_op).getControlOperands().empty()) continue;

      FailureOr<TFOp> const_out = CreateConstantTensorOp(
          rewriter, op->getLoc(), op->getName().getStringRef(),
          *(operand_op->result_type_begin()), TFOp(op).controlRet(),
          operand_op->getAttrOfType<ElementsAttr>("value"), op->getAttrs());
      if (failed(const_out)) return failure();
      (*const_out).setName(Twine(TFOp(op).name(), "/_const"));

      FailureOr<TFOp> const_index = CreateConstantTensorOp(
          rewriter, op->getLoc(), op->getName().getStringRef(),
          rewriter.getIntegerType(32), TFOp(op).controlRet(),
          DenseElementsAttr::get(zero_dim_i32_tensor_type_, idx++));
      if (failed(const_index)) return failure();

      (*const_index).setName(Twine(TFOp(op).name(), "/_index"));
      if (!TFOp(op).device().empty())
        (*const_index).setRequestedDevice(TFOp(op).device());

      for (OpOperand &user : op->getResults()[0].getUses()) {
        rewriter.startRootUpdate(user.getOwner());
        user.set((*const_out)->getResult(0));
        rewriter.finalizeRootUpdate(user.getOwner());
      }
      for (OpOperand &user : op->getResults()[1].getUses()) {
        rewriter.startRootUpdate(user.getOwner());
        user.set((*const_index)->getResult(1));
        rewriter.finalizeRootUpdate(user.getOwner());
      }

      return success();
    }
    return failure();
  }

  RankedTensorType zero_dim_i32_tensor_type_;
};

class MergeNodeFolding : public MergeNodeFoldingBase {
 public:
  MergeNodeFolding(MLIRContext *context, OpPropertyHelper &helper)
      : MergeNodeFoldingBase("tfg.Merge", context, helper) {}
};

class RefMergeNodeFolding : public MergeNodeFoldingBase {
 public:
  RefMergeNodeFolding(MLIRContext *context, OpPropertyHelper &helper)
      : MergeNodeFoldingBase("tfg.RefMerge", context, helper) {}
};

class XlaMergeNodeFolding : public MergeNodeFoldingBase {
 public:
  XlaMergeNodeFolding(MLIRContext *context, OpPropertyHelper &helper)
      : MergeNodeFoldingBase("tfg.XlaMerge", context, helper) {}
};

// This implementation is mapped with ConstantFolding::RemoveSplitOrSplitVin in
// grappler/optimizers/constant_folding.cc
class RemoveSplitOp : public FolderPatternBase {
 public:
  RemoveSplitOp(MLIRContext *context, OpPropertyHelper &helper)
      : FolderPatternBase("tfg.Split", context, helper) {}
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    auto num_split_attr = op->getAttrOfType<IntegerAttr>("num_split");
    if (!num_split_attr || num_split_attr.getInt() != 1) return failure();
    FailureOr<TFOp> identity = ReplaceOpWithIdentity(rewriter, op, 1);
    if (failed(identity)) return failure();
    rewriter.replaceOp(op, (*identity)->getResults());
    return success();
  }
};

// This implementation is mapped with ConstantFolding::RemoveSplitOrSplitVin in
// grappler/optimizers/constant_folding.cc
class RemoveSplitVOp : public FolderPatternBase {
 public:
  RemoveSplitVOp(MLIRContext *context, OpPropertyHelper &helper)
      : FolderPatternBase("tfg.SplitV", context, helper) {}
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (!op->getAttr("num_split")) return failure();
    if (op->getAttrOfType<IntegerAttr>("num_split").getInt() == 1) {
      FailureOr<TFOp> identity = ReplaceOpWithIdentity(rewriter, op, 0);
      if (failed(identity)) return failure();
      rewriter.replaceOp(op, (*identity)->getResults());
      return success();
    }
    return failure();
  }
};

// TODO(chiahungduan): Do we still have "Shuffle" op?
// This implementation is mapped with ConstantFolding::RemoveShuffleOrTranspose
// in grappler/optimizers/constant_folding.cc
class RemoveShuffleOp : public FolderPatternBase {
 public:
  RemoveShuffleOp(MLIRContext *context, OpPropertyHelper &helper)
      : FolderPatternBase("tfg.Shuffle", context, helper) {}
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    Operation *perm_op = op->getOperand(1).getDefiningOp();
    if (!perm_op || !helper_.IsConstant(perm_op)) return failure();
    ElementsAttr perm_tensor = perm_op->getAttrOfType<ElementsAttr>("value");
    if (!perm_tensor) return failure();

    Operation *x_op = op->getOperand(0).getDefiningOp();
    if (!x_op) return failure();
    ShapedType x_shape = x_op->getResult(0).getType().cast<ShapedType>();
    if (!x_shape.hasRank()) return failure();
    if (perm_tensor.getNumElements() != x_shape.getRank()) return failure();

    for (unsigned i = 0; i < x_shape.getRank(); ++i) {
      int64_t value = perm_tensor.getElementType().isInteger(32)
                          ? perm_tensor.getValues<int32_t>()[i]
                          : perm_tensor.getValues<int64_t>()[i];
      if (value != i && x_shape.getShape()[i] != 1) return failure();
    }

    FailureOr<TFOp> identity = ReplaceOpWithIdentity(rewriter, op, 0);
    if (failed(identity)) return failure();
    rewriter.replaceOp(op, (*identity)->getResults());

    return success();
  }
};

// This implementation is mapped with ConstantFolding::RemoveShuffleOrTranspose
// in grappler/optimizers/constant_folding.cc
class RemoveTransposeOp : public FolderPatternBase {
 public:
  RemoveTransposeOp(MLIRContext *context, OpPropertyHelper &helper)
      : FolderPatternBase("tfg.Transpose", context, helper) {}
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    Operation *perm_op = op->getOperand(1).getDefiningOp();
    if (!perm_op || !helper_.IsConstant(perm_op)) return failure();
    ElementsAttr perm_tensor = perm_op->getAttrOfType<ElementsAttr>("value");
    if (!perm_tensor) return failure();

    Operation *x_op = op->getOperand(0).getDefiningOp();
    if (!x_op) return failure();
    ShapedType x_shape = x_op->getResult(0).getType().cast<ShapedType>();
    if (!x_shape.hasRank()) return failure();
    if (perm_tensor.getNumElements() != x_shape.getRank()) return failure();

    for (unsigned i = 0; i < x_shape.getRank(); ++i) {
      int64_t value = perm_tensor.getElementType().isInteger(32)
                          ? perm_tensor.getValues<int32_t>()[i]
                          : perm_tensor.getValues<int64_t>()[i];
      if (value != i && x_shape.getShape()[i] != 1) return failure();
    }

    FailureOr<TFOp> identity = ReplaceOpWithIdentity(rewriter, op, 0);
    if (failed(identity)) return failure();
    rewriter.replaceOp(op, (*identity)->getResults());

    return success();
  }
};

// This implementation is mapped with ConstantFolding::RemoveRandomShuffle
// in grappler/optimizers/constant_folding.cc
class RemoveRandomShuffleOp : public FolderPatternBase {
 public:
  RemoveRandomShuffleOp(MLIRContext *context, OpPropertyHelper &helper)
      : FolderPatternBase("tfg.RandomShuffle", context, helper) {}
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    Operation *operand_op = op->getOperand(0).getDefiningOp();
    if (!operand_op) return failure();
    auto shape = (*operand_op->result_type_begin()).dyn_cast<ShapedType>();
    if (!shape.hasRank()) return failure();
    if (shape.getRank() != 0 && shape.getShape()[0] != 1) return failure();

    FailureOr<TFOp> identity = ReplaceOpWithIdentity(rewriter, op, 0);
    if (failed(identity)) return failure();
    rewriter.replaceOp(op, (*identity)->getResults());

    return success();
  }
};

// This implementation is mapped with ConstantFolding::RemoveReverse
// in grappler/optimizers/constant_folding.cc
class RemoveReverse : public FolderPatternBase {
 public:
  RemoveReverse(MLIRContext *context, OpPropertyHelper &helper)
      : FolderPatternBase("tfg.ReverseV2", context, helper) {}
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    Operation *tensor_op = op->getOperand(0).getDefiningOp();
    if (!tensor_op) return failure();

    ShapedType tensor_type =
        (*tensor_op->result_type_begin()).cast<ShapedType>();
    if (!tensor_type.hasRank()) return failure();

    Operation *dim_op = op->getOperand(1).getDefiningOp();
    if (!dim_op || !helper_.IsConstant(dim_op)) return failure();

    auto dim_attr = dim_op->getAttrOfType<ElementsAttr>("value");
    DenseSet<int> target_axis;
    for (unsigned i = 0; i < dim_attr.getNumElements(); ++i) {
      if (dim_attr.getElementType().isInteger(32)) {
        target_axis.insert(
            (dim_attr.getValues<int32_t>()[i] + tensor_type.getRank()) %
            tensor_type.getRank());
      } else {
        target_axis.insert(
            (dim_attr.getValues<int64_t>()[i] + tensor_type.getRank()) %
            tensor_type.getRank());
      }
    }

    for (unsigned i = 0; i < tensor_type.getRank(); ++i)
      if (tensor_type.getShape()[i] != 1 && target_axis.contains(i))
        return failure();

    FailureOr<TFOp> identity = ReplaceOpWithIdentity(rewriter, op, 0);
    if (failed(identity)) return failure();
    rewriter.replaceOp(op, (*identity)->getResults());

    return success();
  }
};

// This implementation is mapped with ConstantFolding::SimplifySlice
// in grappler/optimizers/constant_folding.cc
class SimlifySliceOp : public FolderPatternBase {
 public:
  SimlifySliceOp(MLIRContext *context, OpPropertyHelper &helper)
      : FolderPatternBase("tfg.Slice", context, helper) {}
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    Operation *begin_op = op->getOperand(1).getDefiningOp();
    Operation *size_op = op->getOperand(2).getDefiningOp();
    if (!begin_op || !size_op) return failure();

    if (!helper_.IsConstant(begin_op) || !helper_.IsConstant(size_op))
      return failure();

    auto begin_attr = begin_op->getAttrOfType<ElementsAttr>("value");
    auto size_attr = size_op->getAttrOfType<ElementsAttr>("value");

    Operation *input_op = op->getOperand(0).getDefiningOp();
    ShapedType input_type = (*input_op->result_type_begin()).cast<ShapedType>();
    if (!input_type.hasRank()) return failure();

    for (unsigned i = 0; i < input_type.getRank(); ++i) {
      if (begin_attr.getElementType().isInteger(32)) {
        if (begin_attr.getValues<int32_t>()[i] != 0) return failure();
      } else {
        if (begin_attr.getValues<int64_t>()[i] != 0) return failure();
      }

      if (size_attr.getElementType().isInteger(32)) {
        if (size_attr.getValues<int32_t>()[i] != -1 &&
            size_attr.getValues<int32_t>()[i] != input_type.getShape()[i])
          return failure();
      } else {
        if (size_attr.getValues<int64_t>()[i] != -1 &&
            size_attr.getValues<int64_t>()[i] != input_type.getShape()[i])
          return failure();
      }
    }

    FailureOr<TFOp> identity = ReplaceOpWithIdentity(rewriter, op, 0);
    if (failed(identity)) return failure();
    rewriter.replaceOp(op, (*identity)->getResults());

    return success();
  }
};

// This implementation is mapped with ConstantFolding::SimplifyStridedSlice
// in grappler/optimizers/constant_folding.cc
class SimplifyStridedSlice : public FolderPatternBase {
 public:
  SimplifyStridedSlice(MLIRContext *context, OpPropertyHelper &helper)
      : FolderPatternBase("tfg.StridedSlice", context, helper) {}
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (auto attr = op->getAttrOfType<IntegerAttr>("new_axis_mask")) {
      if (attr.getInt() != 0) return failure();
    } else {
      return failure();
    }
    if (auto attr = op->getAttrOfType<IntegerAttr>("shrink_axis_mask")) {
      if (attr.getInt() != 0) return failure();
    } else {
      return failure();
    }

    Operation *input_op = op->getOperand(0).getDefiningOp();
    if (!input_op) return failure();
    ShapedType input_type = (*input_op->result_type_begin()).cast<ShapedType>();
    if (!input_type.hasStaticShape()) return failure();

    Operation *begin_op = op->getOperand(1).getDefiningOp();
    Operation *end_op = op->getOperand(2).getDefiningOp();
    Operation *strides_op = op->getOperand(3).getDefiningOp();
    if (!begin_op || !end_op || !strides_op) return failure();

    if (!helper_.IsConstant(begin_op) || !helper_.IsConstant(end_op) ||
        !helper_.IsConstant(strides_op))
      return failure();

    ElementsAttr begin_attr = begin_op->getAttrOfType<ElementsAttr>("value");
    ElementsAttr end_attr = end_op->getAttrOfType<ElementsAttr>("value");
    ElementsAttr strides_attr =
        strides_op->getAttrOfType<ElementsAttr>("value");

    if (!op->getAttr("begin_mask") || !op->getAttr("end_mask") ||
        !op->getAttr("ellipsis_mask"))
      return failure();

    int begin_mask = op->getAttrOfType<IntegerAttr>("begin_mask").getInt();
    int end_mask = op->getAttrOfType<IntegerAttr>("end_mask").getInt();
    int ellipsis_mask =
        op->getAttrOfType<IntegerAttr>("ellipsis_mask").getInt();
    DenseSet<int> expanded_ellipsis_indices;
    int ellipsis_index = -1;

    for (unsigned i = 0; i < input_type.getRank(); ++i) {
      if (ellipsis_mask & 1 << i ||
          (ellipsis_index == -1 && i >= strides_attr.getNumElements()))
        ellipsis_index = i;
      if (ellipsis_index != -1 &&
          input_type.getRank() >
              strides_attr.getNumElements() + i - ellipsis_index)
        expanded_ellipsis_indices.insert(i);
    }

    for (unsigned i = 0; i < input_type.getRank(); ++i) {
      if (expanded_ellipsis_indices.find(i) != expanded_ellipsis_indices.end())
        continue;
      int j = i;
      int expanded_ellipsis_indices_size = expanded_ellipsis_indices.size();
      if (ellipsis_index != -1 &&
          i >= ellipsis_index + expanded_ellipsis_indices_size) {
        j = i - expanded_ellipsis_indices_size;
      }
      int b = begin_attr.getElementType().isInteger(32)
                  ? begin_attr.getValues<int32_t>()[j]
                  : begin_attr.getValues<int64_t>()[j];
      int e = end_attr.getElementType().isInteger(32)
                  ? end_attr.getValues<int32_t>()[j]
                  : end_attr.getValues<int64_t>()[j];
      int s = strides_attr.getElementType().isInteger(32)
                  ? strides_attr.getValues<int32_t>()[j]
                  : strides_attr.getValues<int64_t>()[j];

      if (!(begin_mask & 1 << j || b == 0) ||
          !(end_mask & 1 << j || e == input_type.getShape()[i]) || s != 1)
        return failure();
    }

    FailureOr<TFOp> identity = ReplaceOpWithIdentity(rewriter, op, 0);
    if (failed(identity)) return failure();
    rewriter.replaceOp(op, (*identity)->getResults());

    return success();
  }
};

// This implementation is mapped with ConstantFolding::SimplifyTile
// in grappler/optimizers/constant_folding.cc
class SimplifyTileOp : public FolderPatternBase {
 public:
  SimplifyTileOp(MLIRContext *context, OpPropertyHelper &helper)
      : FolderPatternBase("tfg.Tile", context, helper) {}
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    Operation *multiples_op = op->getOperand(1).getDefiningOp();
    if (!multiples_op || !helper_.IsConstant(multiples_op)) return failure();

    ElementsAttr multiples_attr =
        multiples_op->getAttrOfType<ElementsAttr>("value");
    if (multiples_attr.getElementType().isInteger(32)) {
      if (llvm::any_of(multiples_attr.getValues<int32_t>(),
                       [](int v) { return v != 1; }))
        return failure();
    } else {
      if (llvm::any_of(multiples_attr.getValues<int64_t>(),
                       [](int64_t v) { return v != 1; }))
        return failure();
    }

    FailureOr<TFOp> identity = ReplaceOpWithIdentity(rewriter, op, 0);
    if (failed(identity)) return failure();
    rewriter.replaceOp(op, (*identity)->getResults());

    return success();
  }
};

// This implementation is mapped with ConstantFolding::SimplifyPad
// in grappler/optimizers/constant_folding.cc
class SimplifyPadOpBase : public FolderPatternBase {
 protected:
  SimplifyPadOpBase(StringRef op_name, MLIRContext *context,
                    OpPropertyHelper &helper)
      : FolderPatternBase(op_name, context, helper) {}
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    // TODO(chiahungduan): use_shape_info in ConstantFolding::SimplifyPad
    Operation *paddings = op->getOperand(1).getDefiningOp();
    if (!paddings || !helper_.IsConstant(paddings)) return failure();

    ElementsAttr paddings_attr = paddings->getAttrOfType<ElementsAttr>("value");
    if (paddings_attr.getElementType().isInteger(32)) {
      if (llvm::any_of(paddings_attr.getValues<int32_t>(),
                       [](int v) { return v != 0; }))
        return failure();
    } else {
      if (llvm::any_of(paddings_attr.getValues<int64_t>(),
                       [](int64_t v) { return v != 0; }))
        return failure();
    }

    FailureOr<TFOp> identity = ReplaceOpWithIdentity(rewriter, op, 0);
    if (failed(identity)) return failure();
    rewriter.replaceOp(op, (*identity)->getResults());

    return success();
  }
};

// This implementation is mapped with ConstantFolding::SimplifyPad
// in grappler/optimizers/constant_folding.cc
class SimplifyPadOp : public SimplifyPadOpBase {
 public:
  SimplifyPadOp(MLIRContext *context, OpPropertyHelper &helper)
      : SimplifyPadOpBase("tfg.Pad", context, helper) {}
};

// This implementation is mapped with ConstantFolding::SimplifyPad
// in grappler/optimizers/constant_folding.cc
class SimplifyPadV2Op : public SimplifyPadOpBase {
 public:
  SimplifyPadV2Op(MLIRContext *context, OpPropertyHelper &helper)
      : SimplifyPadOpBase("tfg.PadV2", context, helper) {}
};

// This implementation is mapped with ConstantFolding::SimplifySqueeze
// in grappler/optimizers/constant_folding.cc
class SimplifySqueezeOp : public FolderPatternBase {
 public:
  SimplifySqueezeOp(MLIRContext *context, OpPropertyHelper &helper)
      : FolderPatternBase("tfg.Squeeze", context, helper) {}
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    Operation *shape = op->getOperand(0).getDefiningOp();
    if (!shape) return failure();
    ShapedType shape_type = (*shape->result_type_begin()).cast<ShapedType>();
    if (!shape_type.hasRank()) return failure();
    if (llvm::any_of(shape_type.getShape(), [](int64_t s) { return s <= 1; }))
      return failure();
    FailureOr<TFOp> identity = ReplaceOpWithIdentity(rewriter, op, 0);
    if (failed(identity)) return failure();
    rewriter.replaceOp(op, (*identity)->getResults());

    return success();
  }
};

// This implementation is mapped with ConstantFolding::SimplifyPack
// in grappler/optimizers/constant_folding.cc
class SimplifyPackOp : public FolderPatternBase {
 public:
  SimplifyPackOp(MLIRContext *context, OpPropertyHelper &helper)
      : FolderPatternBase("tfg.Pack", context, helper) {}
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    // TODO(chiahungduan): check if the optimization has applied.

    if (TFOp(op).getNonControlOperands().size() != 1) return failure();

    IntegerAttr axis = op->getAttrOfType<IntegerAttr>("axis");
    ElementsAttr const_attr =
        CreateElementsAttrOfTypeValues(rewriter.getIntegerType(32), {},
                                       ArrayRef<int>(axis ? axis.getInt() : 0));
    FailureOr<TFOp> const_op = CreateConstantTensorOp(
        rewriter, op->getLoc(), op->getName().getStringRef(),
        const_attr.getType(), TFOp(op).controlRet(), const_attr);
    if (failed(const_op)) return failure();
    (*const_op).setName(Twine(TFOp(op).name(), "/_const_axis"));
    if (!TFOp(op).device().empty())
      (*const_op).setRequestedDevice(TFOp(op).deviceAttr());

    OperationState state(op->getLoc(), "tfg.ExpandDims");
    state.addTypes(op->getResultTypes());

    state.attributes = op->getAttrDictionary();
    state.attributes.erase("axis");
    state.attributes.erase("T");
    state.addAttribute("Tdim", TypeAttr::get(rewriter.getI32Type()));

    state.addOperands({op->getOperand(0), (*const_op)->getResult(0)});
    state.addOperands(TFOp(op).getControlOperands());
    Operation *expand_dims_op = rewriter.create(state);
    rewriter.replaceOp(op, expand_dims_op->getResults());
    return success();
  }
};

// This implementation is mapped with ConstantFolding::MoveConstantsPastEnter
// in grappler/optimizers/constant_folding.cc
class MoveConstantsPastEnterOpBase : public FolderPatternBase {
 protected:
  MoveConstantsPastEnterOpBase(StringRef op_name, MLIRContext *context,
                               OpPropertyHelper &helper)
      : FolderPatternBase(op_name, context, helper) {}
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    auto is_constant_attr = op->getAttrOfType<BoolAttr>("is_constant");
    if (!is_constant_attr || !is_constant_attr.getValue()) return failure();

    Operation *input = op->getOperand(0).getDefiningOp();
    if (!input || !helper_.IsConstant(input)) return failure();

    // Find non-constant nodes that consume the outputs of Enter.
    if (llvm::all_of(op->getResults()[0].getUsers(),
                     [&](Operation *op) { return helper_.IsConstant(op); }))
      return failure();

    FailureOr<TFOp> cloned_const_op = CreateConstantTensorOp(
        rewriter, op->getLoc(), op->getName().getStringRef(),
        *(input->result_type_begin()), TFOp(op).controlRet(),
        input->getAttr("value"), input->getAttrs());
    if (failed(cloned_const_op)) return failure();

    op->getResults()[0].replaceAllUsesWith((*cloned_const_op)->getResults()[0]);
    return success();
  }
};

// This implementation is mapped with ConstantFolding::MoveConstantsPastEnter
// in grappler/optimizers/constant_folding.cc
class MoveConstantsPastEnterOp : public MoveConstantsPastEnterOpBase {
 public:
  MoveConstantsPastEnterOp(MLIRContext *context, OpPropertyHelper &helper)
      : MoveConstantsPastEnterOpBase("tfg.Enter", context, helper) {}
};

// This implementation is mapped with ConstantFolding::MoveConstantsPastEnter
// in grappler/optimizers/constant_folding.cc
class MoveConstantsPastRefEnterOp : public MoveConstantsPastEnterOpBase {
 public:
  MoveConstantsPastRefEnterOp(MLIRContext *context, OpPropertyHelper &helper)
      : MoveConstantsPastEnterOpBase("tfg.RefEnter", context, helper) {}
};

// This implementation is mapped with ConstantFolding::SimplifySwitch
// in grappler/optimizers/constant_folding.cc
class SimplifySwitchOp : public FolderPatternBase {
 public:
  SimplifySwitchOp(MLIRContext *context, OpPropertyHelper &helper)
      : FolderPatternBase("tfg.Switch", context, helper),
        zero_dim_i1_tensor_type_(
            RankedTensorType::get({}, IntegerType::get(context, 1))) {}
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (op->getOperand(0) != op->getOperand(1)) return failure();

    auto hasRegularUsers = [&](Operation *op) {
      for (Value v : op->getResults()) {
        if (v.getType().isa<ControlType>()) return false;
        if (!v.use_empty()) return true;
      }
      return false;
    };

    // Check if it has optimized.
    if (llvm::all_of(op->getResult(0).getUsers(), [&](Operation *user) {
          return (!user || helper_.IsIdentity(user) ||
                  helper_.IsIdentityNSingleInput(user)) &&
                 !hasRegularUsers(user);
        }))
      return failure();

    Operation *input_0 = op->getOperand(0).getDefiningOp();
    Operation *input_1 = op->getOperand(1).getDefiningOp();
    if (!input_0 || !input_1) return failure();

    // We can't anchor control dependencies directly on the switch node: unlike
    // other nodes only one of the outputs of the switch node will be generated
    // when the switch node is executed, and we need to make sure the control
    // dependency is only triggered when the corresponding output is triggered.
    // We start by looking for an identity node connected to the output of the
    // switch node, and use it to anchor the control dependency.
    OperationState true_identity_op_state(op->getLoc(), "tfg.Identity");
    true_identity_op_state.addAttribute("T", op->getAttr("T"));
    true_identity_op_state.addOperands(TFOp(op).controlRet());
    true_identity_op_state.addTypes(
        {op->getResult(1).getType(), ControlType::get(rewriter.getContext())});
    Operation *true_identity_op = rewriter.create(true_identity_op_state);
    if (!TFOp(input_0).name().empty())
      TFOp(true_identity_op)
          .setName(Twine(TFOp(input_0).name(), "/ControlDependencyCtrl_1"));
    if (!TFOp(input_0).device().empty())
      TFOp(true_identity_op).setRequestedDevice(TFOp(input_0).device());

    OperationState false_identity_op_state(op->getLoc(), "tfg.Identity");
    false_identity_op_state.addAttribute("T", op->getAttr("T"));
    false_identity_op_state.addOperands(TFOp(op).controlRet());
    false_identity_op_state.addTypes(
        {op->getResult(0).getType(), ControlType::get(rewriter.getContext())});

    Operation *false_identity_op = rewriter.create(false_identity_op_state);
    if (!TFOp(input_1).name().empty())
      TFOp(false_identity_op)
          .setName(Twine(TFOp(input_1).name(), "/ControlDependencyCtrl_0"));
    if (!TFOp(input_1).device().empty())
      TFOp(false_identity_op).setRequestedDevice(TFOp(input_1).device());

    FailureOr<TFOp> true_op = CreateConstantTensorOp(
        rewriter, op->getLoc(), op->getName().getStringRef(),
        *(op->result_type_begin()), TFOp(true_identity_op).controlRet(),
        DenseElementsAttr::get(zero_dim_i1_tensor_type_, true));
    if (failed(true_op)) return failure();

    if (!TFOp(op).name().empty())
      (*true_op).setName(Twine(TFOp(op).name(), "/_const_true"));

    if (!TFOp(op).device().empty())
      (*true_op).setRequestedDevice(TFOp(op).device().data());

    FailureOr<TFOp> false_op = CreateConstantTensorOp(
        rewriter, op->getLoc(), op->getName().getStringRef(),
        *(op->result_type_begin()), TFOp(false_identity_op).controlRet(),
        DenseElementsAttr::get(zero_dim_i1_tensor_type_, false));
    if (failed(false_op)) return failure();

    if (!TFOp(op).name().empty())
      (*false_op).setName(Twine(TFOp(op).name(), "/_const_false"));

    if (!TFOp(op).device().empty())
      (*false_op).setRequestedDevice(TFOp(op).device().data());

    for (OpOperand &user : op->getResult(0).getUses()) {
      rewriter.startRootUpdate(user.getOwner());
      user.set((*false_op)->getResult(0));
      rewriter.finalizeRootUpdate(user.getOwner());
    }
    for (OpOperand &user : op->getResult(1).getUses()) {
      rewriter.startRootUpdate(user.getOwner());
      user.set((*true_op)->getResult(0));
      rewriter.finalizeRootUpdate(user.getOwner());
    }

    // TODO(chiahungduan): In order to user `replaceAllUsesWith` above, we set a
    // fake operand in both `true_identity_op` and `false_identity_op` and
    // update it here. See if we have better way to handle this.
    true_identity_op->setOperand(0, op->getResult(1));
    false_identity_op->setOperand(0, op->getResult(0));

    return success();
  }
  RankedTensorType zero_dim_i1_tensor_type_;
};

// This implementation is mapped with ConstantFolding::SimplifyReduction
// in grappler/optimizers/constant_folding.cc
class SimplifyReductionOp : public FolderPatternBase {
 public:
  SimplifyReductionOp(MLIRContext *context, OpPropertyHelper &helper)
      : FolderPatternBase(MatchAnyOpTypeTag(), context, helper) {}
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (!helper_.IsReduction(op)) return failure();

    Operation *reduction_indices = op->getOperand(1).getDefiningOp();
    if (!reduction_indices) return failure();
    // In grappler, they call EvaluateNode() to get the constant value. But if
    // it is a constant, then the EvaluationConstant will have folded it.
    if (!helper_.IsConstant(reduction_indices)) return failure();

    auto ReplaceReductionWithIdentity = [&]() -> Operation * {
      OperationState state(op->getLoc(), "tfg.Identity");
      if (auto T_attr = op->getAttrOfType<TypeAttr>("T"))
        state.addTypes(T_attr.getValue());
      else if (helper_.IsAny(op) || helper_.IsAll(op))
        state.addTypes(rewriter.getI1Type());
      else
        return nullptr;
      state.addTypes(ControlType::get(rewriter.getContext()));

      state.attributes = op->getAttrDictionary();
      util::EraseRegularNodeAttributes(state.attributes);
      state.addAttribute("T", TypeAttr::get(state.types[0]));

      state.addOperands(
          {op->getOperand(0), TFOp(reduction_indices).controlRet()});
      Operation *identity_op = rewriter.create(state);
      return identity_op;
    };

    ShapedType indices_type = *(reduction_indices->result_type_begin());
    if (indices_type.getNumElements() == 0) {
      Operation *identity_op = ReplaceReductionWithIdentity();
      if (!identity_op) return failure();
      rewriter.replaceOp(op, identity_op->getResults());
      return success();
    }

    // Check `IsReductionCandidateForSimplification`
    Operation *input = op->getOperand(0).getDefiningOp();
    if (!input) return failure();
    ShapedType input_type = (*input->result_type_begin()).cast<ShapedType>();
    if (!input_type.hasStaticShape()) return failure();

    ShapedType op_type = (*op->result_type_begin()).cast<ShapedType>();
    if (!op_type.hasStaticShape()) return failure();

    bool is_single_element_op =
        input_type.getNumElements() == 1 && op_type.getNumElements() == 1;

    bool keep_dims = false;
    if (auto attr = op->getAttrOfType<BoolAttr>("keep_dims"))
      keep_dims = attr.getValue();
    bool simplifiable_to_reshape =
        is_single_element_op && !keep_dims && op->hasAttr("T");
    ElementsAttr reduction_indices_attr =
        reduction_indices->getAttrOfType<ElementsAttr>("value");

    bool simplifiable_to_identity;
    if (reduction_indices_attr.getElementType().isInteger(32)) {
      for (int v : reduction_indices_attr.getValues<int32_t>()) {
        if (v < 0) v += input_type.getRank();
        if (v < 0 || v >= input_type.getRank() || input_type.getShape()[v] != 1)
          simplifiable_to_identity = false;
      }
    } else {
      for (int64_t v : reduction_indices_attr.getValues<int64_t>()) {
        if (v < 0) v += input_type.getRank();
        if (v < 0 || v >= input_type.getRank() || input_type.getShape()[v] != 1)
          simplifiable_to_identity = false;
      }
    }
    simplifiable_to_identity &= keep_dims;

    if (simplifiable_to_reshape) {
      const int new_num_dimensions = op_type.getRank();
      SmallVector<int64_t> elements(new_num_dimensions);
      std::iota(elements.begin(), elements.end(), 1);
      ElementsAttr const_attr = CreateElementsAttrOfTypeValues(
          rewriter.getIntegerType(32), {new_num_dimensions},
          llvm::makeArrayRef(elements));
      FailureOr<TFOp> const_op = CreateConstantTensorOp(
          rewriter, op->getLoc(), op->getName().getStringRef(), indices_type,
          TFOp(reduction_indices).controlRet(), const_attr);
      if (failed(const_op)) return failure();

      (*const_op).setName(Twine(TFOp(op).name(), "/_shape_const"));
      if (!TFOp(op).device().empty())
        (*const_op).setRequestedDevice(TFOp(op).deviceAttr());

      OperationState state(op->getLoc(), "tfg.Reshape");
      state.attributes = op->getAttrDictionary();
      state.attributes.erase("keep_dims");
      state.attributes.erase("Tidx");
      state.addAttribute("Tshape", TypeAttr::get(rewriter.getI32Type()));

      state.addOperands(op->getOperands());
      state.operands[1] = (*const_op)->getResult(0);
      state.addTypes(op->getResultTypes());

      Operation *reshape_op = rewriter.create(state);
      rewriter.replaceOp(op, reshape_op->getResults());
    } else if (simplifiable_to_identity) {
      Operation *identity_op = ReplaceReductionWithIdentity();
      if (!identity_op) return failure();
      rewriter.replaceOp(op, identity_op->getResults());
    } else {
      return failure();
    }

    return success();
  }
};

// This implementation is mapped with ConstantFolding::SimplifyReshapeOp
// in grappler/optimizers/constant_folding.cc
class SimplifyReshapeOp : public FolderPatternBase {
 public:
  SimplifyReshapeOp(MLIRContext *context, OpPropertyHelper &helper)
      : FolderPatternBase(MatchAnyOpTypeTag(), context, helper) {}
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (!helper_.IsReshape(op)) return failure();
    if (!TFOp(op).getControlOperands().empty()) return failure();

    Operation *shape_op = op->getOperand(1).getDefiningOp();
    if (!shape_op || !helper_.IsConstant(shape_op)) return failure();

    // TODO(chiahungduan): Check Status s = EvaluateNode(*new_shape,
    // TensorVector(), &outputs); in ConstantFolding::IsSimplifiableReshape Why
    // do we need to evalute a Const op?

    OperationState state(op->getLoc(), "tfg.Identity");
    state.addTypes(op->getResultTypes());
    state.addOperands({op->getOperand(0), TFOp(shape_op).controlRet()});

    state.attributes = op->getAttrDictionary();
    util::EraseRegularNodeAttributes(state.attributes);
    state.addAttribute("T", op->getAttrOfType<TypeAttr>("T"));

    Operation *identity = rewriter.create(state);
    rewriter.replaceOp(op, identity->getResults());

    return success();
  }
};

// This implementation is mapped with
// ConstantFolding::SimplifyArithmeticOperations in
// grappler/optimizers/constant_folding.cc
class SimplifyArithmeticOp : public FolderPatternBase {
 public:
  SimplifyArithmeticOp(MLIRContext *context, OpPropertyHelper &helper)
      : FolderPatternBase(MatchAnyOpTypeTag(), context, helper) {}
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    const bool is_mul = helper_.IsAnyMul(op) || helper_.IsLogicalAnd(op);
    const bool is_matmul = helper_.IsAnyMatMul(op);
    const bool is_add =
        helper_.IsAdd(op) || helper_.IsBiasAdd(op) || helper_.IsLogicalOr(op);
    const bool is_sub = helper_.IsSub(op);
    const bool is_any_div = helper_.IsAnyDiv(op) || helper_.IsFloorDiv(op);

    if (!is_mul && !is_matmul && !is_add && !is_sub && !is_any_div)
      return failure();

    Operation *x = op->getOperand(0).getDefiningOp();
    Operation *y = op->getOperand(1).getDefiningOp();

    if (!x || !y) return failure();

    ShapedType op_type = (*op->result_type_begin()).cast<ShapedType>();
    ShapedType x_type = (*x->result_type_begin()).cast<ShapedType>();
    ShapedType y_type = (*y->result_type_begin()).cast<ShapedType>();

    const bool y_matches_output_shape = op_type == y_type;
    const bool x_matches_output_shape = op_type == x_type;

    const bool x_is_zero = helper_.IsZeros(x);
    const bool x_is_one = helper_.IsOnes(x);

    // TODO(chiahungduan): Check if the optimizations has been applied.

    if ((is_mul && x_is_one) || (is_add && x_is_zero)) {
      if (y_matches_output_shape) {
        FailureOr<TFOp> snapshot_op =
            ReplaceOperationWithSnapshot(rewriter, op, 1);
        if (failed(snapshot_op)) return failure();
        rewriter.replaceOp(op, (*snapshot_op)->getResults());
        return success();
      } else if (x_matches_output_shape) {
        if (!(*x->result_type_begin()).cast<ShapedType>().hasStaticShape())
          return failure();
        FailureOr<TFOp> broadcast_to_op =
            ReplaceOperationWithBroadcastTo(rewriter, op, 1);
        rewriter.replaceOp(op, (*broadcast_to_op)->getResults());
      }
      return success();
    }

    if (y_matches_output_shape && (is_sub && x_is_zero)) {
      OperationState state(op->getLoc(), "tfg.Neg");
      state.addOperands({op->getOperand(0), TFOp(x).controlRet()});
      state.attributes = op->getAttrDictionary();
      state.addTypes(op->getResultTypes());
      Operation *neg = rewriter.create(state);
      rewriter.replaceOp(op, neg->getResults());
      return success();
    }

    if (y_matches_output_shape && is_any_div && x_is_one) {
      TypeAttr type_attr = op->getAttrOfType<TypeAttr>("T");
      if (!type_attr) return failure();

      if (type_attr.getValue().isa<FloatType>() ||
          type_attr.getValue().isa<ComplexType>()) {
        OperationState state(op->getLoc(), "tfg.Reciprocal");
        state.addOperands({op->getOperand(0), TFOp(x).controlRet()});
        state.attributes = op->getAttrDictionary();
        state.addTypes(op->getResultTypes());
        Operation *reciprocal_op = rewriter.create(state);
        rewriter.replaceOp(op, reciprocal_op->getResults());
        return success();
      }
    }

    const bool y_is_zero = helper_.IsZeros(y);
    const bool y_is_one = helper_.IsOnes(y);

    if (((is_mul || is_any_div) && y_is_one) ||
        ((is_add || is_sub) && y_is_zero)) {
      // x * 1 = x or x / 1 = x or x +/- 0 = x
      if (x_matches_output_shape) {
        FailureOr<TFOp> snapshot_op =
            ReplaceOperationWithSnapshot(rewriter, op, 0);
        if (failed(snapshot_op)) return failure();
        rewriter.replaceOp(op, (*snapshot_op)->getResults());
      } else if (y_matches_output_shape) {
        if (!(*x->result_type_begin()).cast<ShapedType>().hasStaticShape())
          return failure();
        FailureOr<TFOp> broadcast_to_op =
            ReplaceOperationWithBroadcastTo(rewriter, op, 0);
        if (failed(broadcast_to_op)) return failure();
        rewriter.replaceOp(op, (*broadcast_to_op)->getResults());
      }
      return success();
    }

    if (op_type.hasStaticShape() && helper_.IsLogicalOr(op) &&
        (y_is_one || x_is_one)) {
      FailureOr<TFOp> const_op = ReplaceOperationWithConstant(rewriter, op, 1);
      if (failed(const_op)) return failure();

      rewriter.replaceOp(op, (*const_op)->getResults());
      return success();
    }

    // TODO(chiahungduan): handle RewriterConfig::AGGRESSIVE
    const bool is_aggressive = false;
    bool optimize_zeros_divided_by_y = is_any_div && x_is_zero && is_aggressive;
    if ((x_is_zero || y_is_zero) &&
        (is_mul || is_matmul || optimize_zeros_divided_by_y)) {
      if (op_type.hasStaticShape()) {
        FailureOr<TFOp> const_op =
            ReplaceOperationWithConstant(rewriter, op, 0);
        if (failed(const_op)) return failure();

        rewriter.replaceOp(op, (*const_op)->getResults());

        bool is_quantized = helper_.IsQuantizedMatMul(op);
        if (is_quantized) {
          // TODO(chiahungduan): AddQuantizedMatMulMinMaxOutConstNodes
          return failure();
        }
        return success();
      }

      if ((is_mul || is_any_div) && x_is_zero) {
        if (x_matches_output_shape) {
          FailureOr<TFOp> identity = ReplaceOpWithIdentity(rewriter, op, 0);
          if (failed(identity)) return failure();
          rewriter.replaceOp(op, (*identity)->getResults());
        } else if (y_matches_output_shape) {
          if (!(*x->result_type_begin()).cast<ShapedType>().hasStaticShape())
            return failure();
          FailureOr<TFOp> broadcast_to_op =
              ReplaceOperationWithBroadcastTo(rewriter, op, 0);
          if (failed(broadcast_to_op)) return failure();
          rewriter.replaceOp(op, (*broadcast_to_op)->getResults());
        }
        return success();
      } else if (is_mul && y_is_zero) {
        if (y_matches_output_shape) {
          FailureOr<TFOp> identity = ReplaceOpWithIdentity(rewriter, op, 0);
          if (failed(identity)) return failure();
          rewriter.replaceOp(op, (*identity)->getResults());
        } else if (x_matches_output_shape) {
          if (!(*x->result_type_begin()).cast<ShapedType>().hasStaticShape())
            return failure();
          FailureOr<TFOp> broadcast_to_op =
              ReplaceOperationWithBroadcastTo(rewriter, op, 1);
          if (failed(broadcast_to_op)) return failure();
          rewriter.replaceOp(op, (*broadcast_to_op)->getResults());
        }
        return success();
      }
    }

    return failure();
  }
};

// This implementation is mapped with ConstantFolding::ReduceDivToReciprocalMul
// in grappler/optimizers/constant_folding.cc
class ReduceDivToReciprocalMul : public FolderPatternBase {
 public:
  ReduceDivToReciprocalMul(MLIRContext *context, OpPropertyHelper &helper)
      : FolderPatternBase(MatchAnyOpTypeTag(), context, helper) {}
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (op->getNumOperands() < 2 ||
        (!helper_.IsDiv(op) && !helper_.IsRealDiv(op) && !helper_.IsXdivy(op)))
      return failure();

    Operation *y = op->getOperand(1).getDefiningOp();
    if (!y || !helper_.IsConstant(y)) return failure();

    TypeAttr type_attr = op->getAttrOfType<TypeAttr>("T");
    if (!type_attr) return failure();

    if (helper_.IsDiv(op) && type_attr.getValue().isa<IntegerType>())
      return failure();

    OperationState state(op->getLoc(), "tfg.Reciprocal");
    state.addOperands(y->getResult(0));
    state.addTypes({*(y->result_type_begin()), ControlType::get(getContext())});
    state.addAttribute("T", type_attr);
    TFOp reciprocal_op = rewriter.create(state);
    reciprocal_op.setName(Twine(TFOp(op).name(), "/_recip"));
    if (!TFOp(op).device().empty())
      reciprocal_op.setAssignedDevice(TFOp(op).deviceAttr());

    StringRef new_op_name = helper_.IsXdivy(op) ? "tfg.MulNoNan" : "tfg.Mul";
    OperationState new_op_state(op->getLoc(), new_op_name);

    if (helper_.IsXdivy(op)) {
      new_op_state.addOperands(
          {reciprocal_op->getResult(0), op->getOperand(0)});
    } else {
      new_op_state.addOperands(
          {op->getOperand(0), reciprocal_op->getResult(0)});
    }

    new_op_state.attributes = op->getAttrDictionary();
    new_op_state.addTypes(op->getResultTypes());

    Operation *new_op = rewriter.create(new_op_state);
    rewriter.replaceOp(op, new_op->getResults());

    return success();
  }
};

// Consider the transformation
//
//                      +                +       = parent
//                     / \              / \
//                    C   +    -- >    X   +     = children
//                       / \              / \
//                      X   Y            C   Y   = leaves
//
// where C is constant, X is non-constant, Y may be constant or non-constant,
// and '+' denotes an associative and commutative operator like addition or
// multiplication. This optimization pushes constants down in the tree to
// canonicalize it. Moreover, in cases where the child node has a second
// constant input Y we will create a leaf node that can be folded, e.g.
//
//    Add(C1, Add(C2, X)) -> Add(X, Add(C1, C2)) -> Add(X, C1 + C2)
//
// We also handle the non-commutative cases of subtraction and division
// by rotating the tree locally, e.g.
//    Sub(C, Add(X, Y)) -> Sub(Sub(C, Y), X)
//    Mul(C, Div(X, Y)) -> Mul(X, Div(C, Y)).
class ConstantPushDown : public FolderPatternBase {
 public:
  ConstantPushDown(MLIRContext *context, OpPropertyHelper &helper)
      : FolderPatternBase(MatchAnyOpTypeTag(), context, helper) {}
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    // Get parent op type.
    const bool is_add = helper_.IsAdd(op);
    const bool is_mul = helper_.IsMul(op);
    const bool is_sub = helper_.IsSub(op);
    const bool is_div = helper_.IsDiv(op);
    if (!(is_add || is_sub || is_mul || is_div)) return failure();
    const bool is_symmetric = is_add || is_mul;

    Operation *child_op = op->getOperand(0).getDefiningOp();
    Operation *const_op = op->getOperand(1).getDefiningOp();
    if (!child_op || !const_op) return failure();
    bool left_child_is_const = helper_.IsConstant(child_op);

    if (!helper_.IsConstant(const_op)) std::swap(child_op, const_op);
    if (!helper_.IsConstant(const_op)) return failure();
    if (helper_.ShouldPreserveOp(child_op)) return failure();

    // Get child op type.
    const bool is_child_add = helper_.IsAdd(child_op);
    const bool is_child_mul = helper_.IsMul(child_op);
    const bool is_child_sub = helper_.IsSub(child_op);
    const bool is_child_div = helper_.IsDiv(child_op);
    const bool is_add_sub =
        (is_add || is_sub) && (is_child_add || is_child_sub);
    const bool is_mul_div =
        (is_mul || is_div) && (is_child_mul || is_child_div);
    if (!is_add_sub && !is_mul_div) return failure();
    const bool is_child_symmetric = is_child_add || is_child_mul;

    TypeAttr t_attr = op->getAttrOfType<TypeAttr>("T");
    if (!t_attr) return failure();

    if (!(is_symmetric && is_child_symmetric) &&
        t_attr.getValue().isIntOrIndex())
      return failure();

    Operation *left_leaf_op = child_op->getOperand(0).getDefiningOp();
    Operation *right_leaf_op = child_op->getOperand(1).getDefiningOp();

    if (!left_leaf_op || !right_leaf_op) return failure();

    const bool left_leaf_is_const = helper_.IsConstant(left_leaf_op);
    Operation *y_node = left_leaf_is_const ? left_leaf_op : right_leaf_op;

    if (!helper_.IsConstant(y_node)) {
      // If we know the shapes of the nodes being swapped, make sure we don't
      // push down a larger node and create more work by broadcasting earlier
      // in the expressions tree.
      Operation *c_op =
          op->getOperand((left_child_is_const ? 0 : 1)).getDefiningOp();
      Operation *x_op =
          op->getOperand((left_leaf_is_const ? 0 : 1)).getDefiningOp();
      if (!c_op || !x_op) return failure();
      auto c_shape = (*c_op->result_type_begin()).cast<ShapedType>();
      auto x_shape = (*x_op->result_type_begin()).cast<ShapedType>();

      if (c_shape.hasStaticShape() && x_shape.hasStaticShape() &&
          c_shape.getNumElements() > x_shape.getNumElements())
        return failure();
      if (c_shape.hasRank() && x_shape.hasRank() && c_shape.getRank() > 0) {
        for (int idx = 0; idx < std::min(c_shape.getRank(), x_shape.getRank());
             ++idx) {
          if (x_shape.getShape()[idx] >= 0 &&
              c_shape.getShape()[idx] > x_shape.getShape()[idx])
            return failure();
        }
      }
    }

    // Child input
    Operation *input_x = left_leaf_is_const
                             ? child_op->getOperand(1).getDefiningOp()
                             : child_op->getOperand(0).getDefiningOp();
    Operation *input_y = left_leaf_is_const
                             ? child_op->getOperand(0).getDefiningOp()
                             : child_op->getOperand(1).getDefiningOp();

    if (!input_x || !input_y) return failure();

    Operation *input_c = const_op;
    Operation *input_op = child_op;

    if (op->getOperand(0).getDefiningOp() == input_c)
      op->setOperand(0, input_x->getResult(0));
    else
      op->setOperand(1, input_x->getResult(0));

    if (is_symmetric && is_child_symmetric) {
      // Easy case (only commutative ops). We always write this as one of
      //   +
      //  / \
      // X   +
      //    / \
      //   C   Y
      op->setOperand(0, input_x->getResult(0));
      op->setOperand(1, input_op->getResult(0));
      child_op->setOperand(0, input_c->getResult(0));
      child_op->setOperand(1, input_y->getResult(0));
    } else {
      // More complicated case: When there are non-commutative operations like
      // subtractions or divisions involved, we may have to rotate the tree
      // and/or change op types. There are 6 non-trivial cases depending on
      // the effective generalized "sign" of each of the three terms C, Y, and
      // X. Here are the final trees we want to generate for those 6 cases:
      //
      // (CYX signs):   ++-      +--      -+-    --+     +-+      -++
      //
      //                 -        -        -      -       +        +
      //                / \      / \      / \    / \     / \      / \
      //               +   X    -   X    -   X  X   +   X   -    X   -
      //              / \      / \      / \        / \     / \      / \
      //             C   Y    C   Y    Y   C      Y   C   C   Y    Y   C
      //

      // First, let's determine the effective sign of each term in the original
      // expression
      auto is_leaf_negated = [&](const bool is_right_leaf) -> bool {
        bool leaf_negated = !is_child_symmetric && is_right_leaf;
        bool child_negated = !is_symmetric && left_child_is_const;
        return leaf_negated != child_negated;
      };

      StringRef symmetric_op = (is_add || is_sub) ? "tfg.Add" : "tfg.Mul";
      StringRef nonsymmetric_op = (is_add || is_sub) ? "tfg.Sub" : "tfg.Div";
      bool neg_c = !is_symmetric && !left_child_is_const;
      bool neg_x = is_leaf_negated(left_leaf_is_const);
      bool neg_y = is_leaf_negated(!left_leaf_is_const);

      StringRef op_name =
          (neg_x || (neg_c && neg_y)) ? nonsymmetric_op : symmetric_op;
      OperationState state(op->getLoc(), op_name);
      state.addOperands({input_op->getResult(0), input_x->getResult(0)});
      // TODO(chiahungduan): Control edge should be inherited as well
      if (!neg_x) std::swap(state.operands[0], state.operands[1]);
      state.addTypes(op->getResultTypes());
      state.attributes = op->getAttrDictionary();
      Operation *new_op = rewriter.create(state);
      rewriter.replaceOp(op, new_op->getResults());

      StringRef child_name = neg_c != neg_y ? nonsymmetric_op : symmetric_op;
      OperationState new_child_state(child_op->getLoc(), child_name);
      new_child_state.addOperands(
          {input_y->getResult(0), input_c->getResult(0)});
      // TODO(chiahungduan): Control edge should be inherited as well
      if (!neg_c)
        std::swap(new_child_state.operands[0], new_child_state.operands[1]);
      new_child_state.addTypes(child_op->getResultTypes());
      new_child_state.attributes = child_op->getAttrDictionary();
      rewriter.setInsertionPoint(child_op);
      Operation *new_child_op = rewriter.create(new_child_state);
      rewriter.replaceOp(child_op, new_child_op->getResults());
    }
    return success();
  }
};

// This implementation is mapped with
// ConstantFolding::PartialConstPropThroughIdentityN in
// grappler/optimizers/constant_folding.cc
// In grappler's constant folding, it propagates the values from IdentityN. But
// here it grabs the constant value from the definingOp which is IdentityN.
class PartialConstPropThroughIdentityN : public FolderPatternBase {
 public:
  PartialConstPropThroughIdentityN(MLIRContext *context,
                                   OpPropertyHelper &helper)
      : FolderPatternBase(MatchAnyOpTypeTag(), context, helper) {}
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    SmallVector<Value> control_operands;
    for (OpOperand &operand : op->getOpOperands()) {
      Value v = operand.get();
      if (v.getType().isa<ControlType>()) break;

      Operation *v_op = v.getDefiningOp();
      if (!v_op || !helper_.IsIdentityN(v_op) ||
          helper_.IsIdentityNSingleInput(v_op))
        continue;

      int res_index = v.cast<OpResult>().getResultNumber();
      Value value_to_forward = v_op->getOperand(res_index);
      if (!value_to_forward.getDefiningOp() ||
          !helper_.IsConstant(value_to_forward.getDefiningOp()))
        continue;
      operand.set(value_to_forward);
      if (llvm::none_of(control_operands, [&](Value v) {
            return v == TFOp(v_op).controlRet();
          })) {
        control_operands.push_back(TFOp(v_op).controlRet());
      }
    }

    if (control_operands.empty()) return failure();

    OperationState state(op->getLoc(), op->getName().getStringRef());
    state.attributes = op->getAttrDictionary();
    state.addOperands(op->getOperands());
    state.addOperands(control_operands);
    state.addTypes(op->getResultTypes());

    Operation *new_op = rewriter.create(state);
    rewriter.replaceOp(op, new_op->getResults());

    return success();
  }
};

// This implementation is mapped with
// ConstantFolding::PartialAssocOpConstFolding in
// grappler/optimizers/constant_folding.cc
class PartialAssocOpConstFolding : public FolderPatternBase {
 public:
  PartialAssocOpConstFolding(MLIRContext *context, OpPropertyHelper &helper)
      : FolderPatternBase(MatchAnyOpTypeTag(), context, helper) {}
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    // Partial constant folding for associative operators:
    // Split AddN/AccumulateNV2 to enable partial
    // folding of ops when more than one but not all inputs are constant.
    // For AddN and AccumulateNV2, we may furthermore reorder inputs, since
    // addition is commutative.
    if (!helper_.IsAggregate(op) || !helper_.IsCommutative(op))
      return failure();
    SmallVector<Value> const_inputs;
    SmallVector<Value> non_const_inputs;

    std::pair<OperandRange, OperandRange> operands = TFOp(op).splitOperands();
    int non_control_inputs_size = operands.first.size();
    int control_inputs_size = operands.second.size();

    for (Value operand : operands.first) {
      Operation *may_const_op = operand.getDefiningOp();
      if (!may_const_op) return failure();
      if (may_const_op->getName().stripDialect() == "Const")
        const_inputs.push_back(operand);
      else
        non_const_inputs.push_back(operand);
    }

    for (Value operand : TFOp(op).getControlOperands())
      non_const_inputs.push_back(operand);

    if (const_inputs.size() == non_control_inputs_size &&
        // TODO(chiahungduan): Do we have this operation tfg.AccumulateNV2?
        op->getName().stripDialect() == "AccumulateNV2") {
      OperationState state(op->getLoc(), "tfg.AddN");
      state.addTypes(op->getResultTypes());
      state.addOperands(op->getOperands());
      state.attributes = op->getAttrDictionary();
      state.attributes.erase("shape");
      Operation *add_n = rewriter.create(state);
      rewriter.replaceOp(op, add_n->getResults());
      return success();
    }

    // TODO(chiahungduan): Check if this optimization is applid
    if (const_inputs.size() <= 1 ||
        const_inputs.size() >= non_control_inputs_size)
      return failure();

    OperationState state(op->getLoc(), "tfg.AddN");
    state.addOperands(const_inputs);
    state.addTypes(op->getResultTypes());
    state.attributes = op->getAttrDictionary();
    state.attributes.erase("shape");
    state.attributes.set("N", IntegerAttr::get(rewriter.getIntegerType(32),
                                               const_inputs.size()));
    Operation *add_n = rewriter.create(state);

    OperationState new_op_state(op->getLoc(), op->getName().getStringRef());
    for (Value v : op->getOperands()) {
      if (v == const_inputs[0])
        new_op_state.addOperands(add_n->getResult(0));
      else
        new_op_state.addOperands(v);
    }
    new_op_state.addOperands(non_const_inputs);
    new_op_state.addTypes(op->getResultTypes());
    new_op_state.attributes = op->getAttrDictionary();
    new_op_state.attributes.set(
        "N", IntegerAttr::get(rewriter.getIntegerType(32),
                              non_const_inputs.size() - control_inputs_size));

    Operation *new_op = rewriter.create(new_op_state);
    rewriter.replaceOp(op, new_op->getResults());

    return success();
  }
};

// This implementation is mapped with ConstantFolding::MergeConcat in
// grappler/optimizers/constant_folding.cc
class MergeConcatOp : public FolderPatternBase {
 public:
  MergeConcatOp(MLIRContext *context, OpPropertyHelper &helper)
      : FolderPatternBase("tfg.ConcatV2", context, helper) {}
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (helper_.ShouldPreserveOp(op)) return failure();

    auto getAxis = [&](Operation *axis_op) {
      ElementsAttr axis_attr = axis_op->getAttrOfType<ElementsAttr>("value");
      return axis_attr.getElementType().isInteger(64)
                 ? static_cast<int>(axis_attr.getSplatValue<int64_t>())
                 : axis_attr.getSplatValue<int>();
    };
    std::pair<OperandRange, OperandRange> op_operands =
        TFOp(op).splitOperands();
    Operation *axis_op = op_operands.first.back().getDefiningOp();
    if (!axis_op || !helper_.IsConstant(axis_op)) return failure();
    int axis = getAxis(axis_op);

    // If all inputs are constant, don't merge and let folding take case of it.
    ValueRange non_control_operands = op_operands.first;
    if (llvm::all_of(non_control_operands.drop_back(), [&](Value v) {
          return v.getDefiningOp() && helper_.IsConstant(v.getDefiningOp());
        }))
      return failure();

    // TODO(chiahungduan): We can scan all the operands to see if there's a
    // Concat op rather just check the first operand. Not sure if it's worth.
    Operation *operand_0 = op->getOperand(0).getDefiningOp();
    if (!operand_0 || !helper_.IsConcatV2(operand_0)) return failure();

    Operation *operand_0_axis_op =
        TFOp(operand_0).getNonControlOperands().back().getDefiningOp();
    if (!operand_0_axis_op || !helper_.IsConstant(operand_0_axis_op))
      return failure();
    if (axis != getAxis(operand_0_axis_op)) return failure();

    std::string task, device;
    StringRef unique_input_tasks;
    for (Value v : op_operands.first) {
      Operation *v_op = v.getDefiningOp();
      if (!v_op || v_op == axis_op) continue;
      if (!TFOp(v_op).device().empty() &&
          tensorflow::DeviceNameUtils::SplitDeviceName(
              TFOp(v_op).device().str(), &task, &device)) {
        if (unique_input_tasks.empty())
          unique_input_tasks = task;
        else if (unique_input_tasks != task)
          return failure();
      }
    }

    OperationState state(op->getLoc(), "tfg.ConcatV2");
    // Move the operands of operand_o first to keep the input order.
    state.addOperands(
        ValueRange(TFOp(operand_0).getNonControlOperands().drop_back()));
    state.addOperands(
        ValueRange(TFOp(op).getNonControlOperands()).drop_front());

    // Copy the control operands.
    state.addOperands(TFOp(op).getControlOperands());

    state.addTypes(op->getResultTypes());

    state.attributes = op->getAttrDictionary();
    state.attributes.set("N", IntegerAttr::get(rewriter.getIntegerType(32),
                                               state.operands.size() - 1));
    Operation *concat_op = rewriter.create(state);

    rewriter.replaceOp(op, concat_op->getResults());

    // TODO(chiahungduan): There's a ReplaceOperationWithNoOp in grappler, it
    // replaces the operation which only has control users into NoOp. Let's do
    // this in another pass to convert all this kind of op.
    return success();
  }
};

// This implementation is mapped with ConstantFolding::MulConvPushDown
// in grappler/optimizers/constant_folding.cc
class MulConvPushDown : public FolderPatternBase {
 public:
  MulConvPushDown(MLIRContext *context, OpPropertyHelper &helper)
      : FolderPatternBase(MatchAnyOpTypeTag(), context, helper) {}
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    // Push down multiplication on ConvND.
    //                       *                  ConvND
    //                     /   \                /    \
    //                 ConvND  C2    -- >      X      *
    //                  / \                          / \
    //                 X  C1                       C1  C2
    //
    // where C1 and C2 are constants and X is non-constant.
    //
    // TODO(rmlarsen): Use PrepareConstantPushDown() to simplify this code.
    if (!helper_.IsAnyMul(op) || TFOp(op).getNonControlOperands().size() != 2)
      return failure();

    Operation *mul_left_child = op->getOperand(0).getDefiningOp();
    Operation *mul_right_child = op->getOperand(1).getDefiningOp();

    if (!mul_left_child || !mul_right_child) return failure();

    const bool left_child_is_constant = helper_.IsConstant(mul_left_child);
    const bool right_child_is_constant = helper_.IsConstant(mul_right_child);
    // One child must be constant, and the second must be Conv op.
    if (!left_child_is_constant && !right_child_is_constant) return failure();

    Operation *conv_node =
        left_child_is_constant ? mul_right_child : mul_left_child;
    if (!helper_.IsConv2D(conv_node) && !helper_.IsConv3D(conv_node))
      return failure();
    // Make sure that it is safe to change the value of the convolution
    // output.
    if (helper_.ShouldPreserveOp(conv_node)) return failure();

    if (TFOp(op).deviceAttr() != TFOp(mul_left_child).deviceAttr() ||
        TFOp(op).deviceAttr() != TFOp(mul_right_child).deviceAttr())
      return failure();

    // Identify the nodes to swap.
    Operation *conv_left_child = conv_node->getOperand(0).getDefiningOp();
    Operation *conv_right_child = conv_node->getOperand(1).getDefiningOp();
    if (!conv_left_child || !conv_right_child) return failure();

    const bool conv_left_is_constant = helper_.IsConstant(conv_left_child);
    const bool conv_right_is_constant = helper_.IsConstant(conv_right_child);
    if (!conv_left_is_constant && !conv_right_is_constant) {
      // At least one of the convolution inputs should be constant.
      return failure();
    }

    if (conv_left_is_constant && conv_right_is_constant) {
      // Leverage regular constant folding to handle this.
      return failure();
    }

    ShapedType mul_shape = (*op->result_type_begin()).cast<ShapedType>();
    ShapedType conv_shape =
        (*conv_node->result_type_begin()).cast<ShapedType>();
    if (mul_shape != conv_shape) return failure();

    ShapedType filter_shape =
        conv_node->getOperand(1).getType().cast<ShapedType>();
    Operation *const_node =
        left_child_is_constant ? mul_left_child : mul_right_child;

    ShapedType const_node_shape =
        (*const_node->result_type_begin()).cast<ShapedType>();
    if (!IsValidConstShapeForMulConvPushDown(
            conv_node->getAttrOfType<StringAttr>("data_format"), filter_shape,
            const_node_shape))
      return failure();

    Operation *conv_const_node =
        conv_left_is_constant ? conv_left_child : conv_right_child;
    for (OpOperand &control_operand : const_node->getOpOperands()) {
      assert(control_operand.get().getType().isa<ControlType>());
      // Make sure we don't introduce loops in the graph by removing control
      // dependencies from the conv2d node to c2.
      if (control_operand.get().getDefiningOp() == conv_node) {
        // Add a control dep from c1 to c2 to ensure c2 is in the right frame
        rewriter.startRootUpdate(conv_const_node);
        control_operand.set(TFOp(conv_const_node).controlRet());
        rewriter.finalizeRootUpdate(conv_const_node);
      }
    }

    StringRef conv_node_name = TFOp(conv_node).name();
    TFOp(conv_node).setName(TFOp(op).nameAttr());
    TFOp(op).setName(Twine(conv_node_name, "/merged_input"));

    if (conv_left_is_constant)
      conv_node->setOperand(0, op->getResult(0));
    else
      conv_node->setOperand(1, op->getResult(0));

    if (left_child_is_constant)
      op->setOperand(1, conv_const_node->getResult(0));
    else
      op->setOperand(0, conv_const_node->getResult(0));

    return success();
  }
};

// This implementation is mapped with ConstantFolding::PartialConcatConstFolding
// in grappler/optimizers/constant_folding.cc
class PartialConcatConstFolding : public FolderPatternBase {
 public:
  PartialConcatConstFolding(MLIRContext *context, OpPropertyHelper &helper)
      : FolderPatternBase(MatchAnyOpTypeTag(), context, helper) {}
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    // Partial constant folding for Concat which is not commutative, so
    // we have to preserve order and can only push consecutive runs of constant
    // inputs into sub-nodes.
    if (!helper_.IsConcat(op)) return failure();
    if (op->getName().getStringRef().rfind("_partial_split_") !=
        StringRef::npos)
      return failure();

    const int num_non_control_inputs = TFOp(op).getNonControlOperands().size();
    if (num_non_control_inputs <= 3) return failure();

    int axis_arg = -1;
    int begin = 0;
    int end = num_non_control_inputs;
    if (helper_.IsConcat(op)) {
      begin = 1;
      axis_arg = 0;
    } else if (helper_.IsConcatV2(op)) {
      end = num_non_control_inputs - 1;
      axis_arg = num_non_control_inputs - 1;
    } else {
      return failure();
    }

    // We search for consecutive runs of constant inputs in the range
    // [begin:end] and push then down into child nodes.
    SmallVector<std::pair<int, int>> constant_input_runs;
    int first = begin;
    int last = begin;
    while (last < end) {
      while (first < end &&
             !(op->getOperand(first).getDefiningOp() &&
               helper_.IsConstant(op->getOperand(first).getDefiningOp())))
        ++first;

      // Invariant: node[first] is constant || first >= end.
      last = first + 1;
      while (last < end &&
             !(op->getOperand(last).getDefiningOp() &&
               helper_.IsConstant(op->getOperand(last).getDefiningOp())))
        ++last;

      // Invariant: node[last] is not constant || last >= end
      // Discard intervals shorter than 2 elements.
      if (first < end && (last - first) > 1) {
        constant_input_runs.emplace_back(first, last);
      }
      first = last;
    }

    // Skip if all inputs are constant, and let constant folding take over.
    if (constant_input_runs.empty() || (constant_input_runs.size() == 1 &&
                                        constant_input_runs[0].first == begin &&
                                        constant_input_runs[0].second == end))
      return failure();

    DenseSet<int> inputs_to_delete;
    for (auto interval : constant_input_runs) {
      // Push the constant inputs in the interval to a child node than can be
      // constant folded.
      OperationState state(op->getLoc(), "tfg.ConcatV2");
      state.addTypes(op->getResultTypes());

      for (auto i : llvm::seq<int>(interval.first, interval.second)) {
        state.addOperands(op->getOperand(i));
        if (i != interval.first) inputs_to_delete.insert(i);
      }
      state.addOperands(op->getOperand(axis_arg));
      state.attributes = op->getAttrDictionary();
      state.attributes.set("N",
                           IntegerAttr::get(rewriter.getI32Type(),
                                            interval.second - interval.first));
      Operation *new_op = rewriter.create(state);

      // Overwrite the first constant input with the result of the added
      // child node.
      rewriter.startRootUpdate(op);
      op->setOperand(interval.first, new_op->getResult(0));
      rewriter.finalizeRootUpdate(op);
    }
    if (!inputs_to_delete.empty()) {
      OperationState state(op->getLoc(), op->getName().getStringRef());
      state.addTypes(op->getResultTypes());
      for (int i = 0; i < op->getNumOperands(); ++i)
        if (inputs_to_delete.find(i) == inputs_to_delete.end())
          state.addOperands(op->getOperand(i));

      state.attributes = op->getAttrDictionary();
      state.attributes.set("N", IntegerAttr::get(rewriter.getI32Type(),
                                                 state.operands.size() - 1));
      Operation *new_op = rewriter.create(state);
      rewriter.replaceOp(op, new_op->getResults());
    }
    return success();
  }
};

// This implements constant push-down for BiasAdd. In the following "CV" is a
// constant vector (tensor of rank 1), "V" is a (possibly) non-constant vector,
// "CM" is a matrix (tensor of rank >= 2), "M" is a (possibly)
// non-constant matrix, and "BA" is BiasAdd.
// For a valid input graph, the following 4 rewrites are legal:
//
//  1)                  +                +
//                     / \              / \
//                    BA  CV    -- >   BA  V
//                   / \              / \
//                  M   V            M   CV
//
//  2)                  +                +
//                     / \              / \
//                    BA  CM    -- >   BA  M
//                   / \              / \
//                  M   V            CM  V
//
//  3)                  BA               BA
//                     / \              / \
//                    +  CV     -- >   +   V
//                   / \              / \
//                  M   V            M  CV
//
//  4)                  BA               BA      = parent
//                     / \              / \
//                    BA  CV    -- >   BA  V     = children
//                   / \              / \
//                  M   V            M  CV       = leaves
//
// Cases 1 through 3 have additional sub-cases due to the symmetry of Add.
class ConstantPushDownBiasAdd : public FolderPatternBase {
 public:
  ConstantPushDownBiasAdd(MLIRContext *context, OpPropertyHelper &helper)
      : FolderPatternBase(MatchAnyOpTypeTag(), context, helper) {}
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (!helper_.IsBiasAdd(op)) return failure();

    Operation *add_child = op->getOperand(0).getDefiningOp();
    Operation *const_child = op->getOperand(1).getDefiningOp();

    if (!add_child || !const_child) return failure();
    if (helper_.ShouldPreserveOp(add_child)) return failure();

    // Special case for BiasAdd: Since the left argument to BiasAdd must be rank
    // >= 2 and the leaves must be vectors, we cannot swap them.
    if (helper_.IsConstant(add_child)) return failure();

    if (!helper_.IsBiasAdd(add_child) && !helper_.IsAdd(add_child))
      return failure();

    auto hasRank = [&](Value value) {
      return value.getType().cast<ShapedType>().hasRank();
    };

    if (!hasRank(op->getOperand(0)) || !hasRank(op->getOperand(1)) ||
        !hasRank(add_child->getOperand(0)) ||
        !hasRank(add_child->getOperand(0)))
      return failure();

    const int left_leaf_rank =
        add_child->getOperand(0).getType().cast<ShapedType>().getRank();
    const int right_leaf_rank =
        add_child->getOperand(1).getType().cast<ShapedType>().getRank();

    if (left_leaf_rank != 1 && right_leaf_rank != 1) return failure();

    const int vector_idx = left_leaf_rank == 1 ? 0 : 1;

    ShapedType vector_type =
        add_child->getOperand(vector_idx).getType().cast<ShapedType>();
    Type vector_d_type = vector_type.getElementType();

    ShapedType const_type =
        const_child->getResult(0).getType().cast<ShapedType>();
    const int const_rank = const_type.getRank();
    Type const_d_type = const_type.getElementType();

    int input_to_swap = -1;

    if (const_rank == 1 && const_d_type == vector_d_type) {
      // Case 1, 3, and, 4:
      input_to_swap = vector_idx;
    } else {
      return failure();
    }

    Operation *leaf_to_swap =
        add_child->getOperand(input_to_swap).getDefiningOp();
    if (!leaf_to_swap || helper_.IsConstant(leaf_to_swap)) return failure();

    rewriter.startRootUpdate(op);
    op->setOperand(1, leaf_to_swap->getResult(0));
    rewriter.finalizeRootUpdate(op);
    add_child->setOperand(input_to_swap, const_child->getResult(0));

    return success();
  }
};

// This implements constant push-down for Add. In the following "CV" is a
// constant vector (tensor of rank 1), "V" is a (possibly) non-constant vector,
// "CM" is a matrix (tensor of rank >= 2), "M" is a (possibly)
// non-constant matrix, and "BA" is BiasAdd.
// For a valid input graph, the following 4 rewrites are legal:
//
//  1)                  +                +
//                     / \              / \
//                    BA  CV    -- >   BA  V
//                   / \              / \
//                  M   V            M   CV
//
//  2)                  +                +
//                     / \              / \
//                    BA  CM    -- >   BA  M
//                   / \              / \
//                  M   V            CM  V
//
//  3)                  BA               BA
//                     / \              / \
//                    +  CV     -- >   +   V
//                   / \              / \
//                  M   V            M  CV
//
//  4)                  BA               BA      = parent
//                     / \              / \
//                    BA  CV    -- >   BA  V     = children
//                   / \              / \
//                  M   V            M  CV       = leaves
//
// Cases 1 through 3 have additional sub-cases due to the symmetry of Add.
class ConstantPushDownAdd : public FolderPatternBase {
 public:
  ConstantPushDownAdd(MLIRContext *context, OpPropertyHelper &helper)
      : FolderPatternBase(MatchAnyOpTypeTag(), context, helper) {}
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (!helper_.IsAdd(op)) return failure();

    Operation *add_child = op->getOperand(0).getDefiningOp();
    Operation *const_child = op->getOperand(1).getDefiningOp();
    if (!add_child || !const_child) return failure();

    if (!helper_.IsConstant(const_child)) std::swap(add_child, const_child);
    if (!helper_.IsConstant(const_child)) return failure();

    bool child_is_bias_add = helper_.IsBiasAdd(add_child);
    if (!child_is_bias_add && !helper_.IsAdd(add_child)) return failure();

    auto hasRank = [&](Value value) {
      return value.getType().cast<ShapedType>().hasRank();
    };

    if (!hasRank(op->getOperand(0)) || !hasRank(op->getOperand(1)) ||
        !hasRank(add_child->getOperand(0)) ||
        !hasRank(add_child->getOperand(1)))
      return failure();

    const int left_leaf_rank =
        add_child->getOperand(0).getType().cast<ShapedType>().getRank();
    const int right_leaf_rank =
        add_child->getOperand(1).getType().cast<ShapedType>().getRank();

    if (left_leaf_rank != 1 && right_leaf_rank != 1) return failure();

    const int vector_idx = left_leaf_rank == 1 ? 0 : 1;
    const int matrix_idx = 1 - vector_idx;

    ShapedType vector_type =
        add_child->getOperand(vector_idx).getType().cast<ShapedType>();
    Type vector_d_type = vector_type.getElementType();

    ShapedType matrix_type =
        add_child->getOperand(matrix_idx).getType().cast<ShapedType>();
    const int matrix_rank = matrix_type.getRank();
    Type matrix_d_type = matrix_type.getElementType();

    ShapedType const_type =
        const_child->getResult(0).getType().cast<ShapedType>();
    const int const_rank = const_type.getRank();
    Type const_d_type = const_type.getElementType();

    int input_to_swap = -1;

    if (child_is_bias_add && const_rank == matrix_rank &&
        const_d_type == matrix_d_type) {
      // Case 2:
      input_to_swap = matrix_idx;
    } else if (const_rank == 1 && const_d_type == vector_d_type) {
      // Case 1, 3, and, 4:
      input_to_swap = vector_idx;
    } else {
      return failure();
    }

    Operation *leaf_to_swap =
        add_child->getOperand(input_to_swap).getDefiningOp();
    if (!leaf_to_swap || helper_.IsConstant(leaf_to_swap)) return failure();

    rewriter.startRootUpdate(op);
    if (op->getOperand(0).getDefiningOp() == add_child)
      op->setOperand(1, leaf_to_swap->getResult(0));
    else
      op->setOperand(0, leaf_to_swap->getResult(0));
    rewriter.finalizeRootUpdate(op);
    add_child->setOperand(input_to_swap, const_child->getResult(0));

    return success();
  }
};

// This implementation is mapped with ConstantFolding::SimplifyCase in
// grappler/optimizers/constant_folding.cc
class SimplifyCaseOp : public FolderPatternBase {
 public:
  SimplifyCaseOp(MLIRContext *context, OpPropertyHelper &helper)
      : FolderPatternBase("tfg.Case", context, helper) {}
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    Operation *branch_index_op = op->getOperand(0).getDefiningOp();
    if (!branch_index_op || !branch_index_op->getAttr("value"))
      return failure();

    ElementsAttr value_attr =
        branch_index_op->getAttrOfType<ElementsAttr>("value");
    if (!value_attr) return failure();

    int output_idx = value_attr.getSplatValue<int>();
    ArrayAttr branch_attr = op->getAttrOfType<ArrayAttr>("branches");
    if (output_idx < 0 || output_idx >= branch_attr.size()) return failure();

    OperationState state(op->getLoc(), "tfg.PartitionedCall");
    state.addOperands(ValueRange(op->getOperands()).drop_front());

    state.attributes = op->getAttrDictionary();
    ArrayAttr output_shape_attr = op->getAttrOfType<ArrayAttr>("output_shapes");
    if (output_shape_attr.size() > output_idx)
      state.attributes.set("_output_shapes", output_shape_attr[output_idx]);
    state.attributes.set("f", branch_attr[output_idx]);

    state.addTypes(op->getResultTypes());

    Operation *partitioned_call_op = rewriter.create(state);
    rewriter.replaceOp(op, partitioned_call_op->getResults());

    return success();
  }
};

// This implementation is mapped with ConstantFolding::SimplifySelect in
// grappler/optimizers/constant_folding.cc
class SimplifySelectOpBase : public FolderPatternBase {
 protected:
  SimplifySelectOpBase(StringRef op_name, MLIRContext *context,
                       OpPropertyHelper &helper)
      : FolderPatternBase(op_name, context, helper) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    Operation *condition_op = op->getOperand(0).getDefiningOp();
    if (!condition_op) return failure();

    bool is_all_true = helper_.IsOnes(condition_op);
    bool is_all_false = helper_.IsZeros(condition_op);

    if (!is_all_true && !is_all_false) return failure();

    ShapedType condition_type =
        (*condition_op->result_type_begin()).cast<ShapedType>();
    Operation *t_op = op->getOperand(1).getDefiningOp();
    Operation *e_op = op->getOperand(2).getDefiningOp();
    if (!t_op || !e_op) return failure();

    ShapedType t_type = (*t_op->result_type_begin()).cast<ShapedType>();
    ShapedType e_type = (*e_op->result_type_begin()).cast<ShapedType>();

    const int live_input_idx = is_all_true ? 1 : 2;
    Value live_operand = op->getOperand(live_input_idx);
    bool predicate_is_scalar =
        !condition_type.hasRank() && condition_type.getRank() == 0;

    if (t_type.hasStaticShape() && t_type.getShape() == e_type.getShape() &&
        (condition_type.getShape() == t_type.getShape() ||
         predicate_is_scalar)) {
      OperationState state(op->getLoc(), "tfg.Identity");
      state.addTypes(op->getResultTypes());
      state.addOperands(live_operand);
      std::pair<OperandRange, OperandRange> operands = TFOp(op).splitOperands();

      for (Value operand : operands.first) {
        if (operand == live_operand) continue;
        // Add the remaining operands as control operands.
        state.addOperands(LookupControlDependency(operand));
      }
      state.addOperands(operands.second);
      state.attributes = op->getAttrDictionary();
      Operation *identity = rewriter.create(state);
      rewriter.replaceOp(op, identity->getResults());
    } else {
      FailureOr<TFOp> broadcast_to_op =
          ReplaceOperationWithBroadcastTo(rewriter, op, live_input_idx);
      if (failed(broadcast_to_op)) return failure();
      rewriter.replaceOp(op, (*broadcast_to_op)->getResults());
    }

    return success();
  }
};

class SimplifySelectOp : public SimplifySelectOpBase {
 public:
  SimplifySelectOp(MLIRContext *context, OpPropertyHelper &helper)
      : SimplifySelectOpBase("tfg.Select", context, helper) {}
};

class SimplifySelectV2Op : public SimplifySelectOpBase {
 public:
  SimplifySelectV2Op(MLIRContext *context, OpPropertyHelper &helper)
      : SimplifySelectOpBase("tfg.SelectV2", context, helper) {}
};

class ConstantFolding : public ConstantFoldingPassBase<ConstantFolding> {
 public:
  LogicalResult initialize(MLIRContext *context) override {
    helper_ = std::make_shared<OpPropertyHelper>(
        context, nodes_to_preserve_, disable_compressed_tensor_optimization_);
    RewritePatternSet patterns(context);
    populateConstantPropagationPatterns(*context, patterns);
    populateConstantFoldingPatterns(*context, patterns);
    final_patterns_ = std::move(patterns);
    return success();
  }

  void runOnOperation() override;

 private:
  void populateConstantPropagationPatterns(MLIRContext &context,
                                           ::mlir::RewritePatternSet &patterns);
  void populateConstantFoldingPatterns(MLIRContext &context,
                                       ::mlir::RewritePatternSet &patterns);

  FrozenRewritePatternSet final_patterns_;
  std::shared_ptr<OpPropertyHelper> helper_;
};

void ConstantFolding::populateConstantPropagationPatterns(
    MLIRContext &context, ::mlir::RewritePatternSet &patterns) {
  patterns
      .insert<MaterializeBroadcastGradientArgsOp, MaterializeShapeNOp,
              SimplifySwitchOp, MergeNodeFolding, RefMergeNodeFolding,
              XlaMergeNodeFolding, MoveConstantsPastEnterOp,
              MoveConstantsPastRefEnterOp, MaterializeReductionIndices,
              PartialConstPropThroughIdentityN, ConstantPushDown,
              MulConvPushDown, ConstantPushDownBiasAdd, ConstantPushDownAdd>(
          &context, *helper_);
}

void ConstantFolding::populateConstantFoldingPatterns(
    MLIRContext &context, ::mlir::RewritePatternSet &patterns) {
  // This is a No-Op in MLIR (see comments in the pattern), comment it out here
  // as a reminder that this is the mapping of MaterializeOutputValue in
  // grappler.
  // patterns.insert<MaterializeOutputValue>(&context, *helper_);
  patterns.insert<
      EvaluateConstant, PartialConcatConstFolding, PartialAssocOpConstFolding,
      SimplifyArithmeticOp, ReduceDivToReciprocalMul, SimplifyReshapeOp,
      RemoveReverse, SimplifyStridedSlice, SimplifyTileOp, SimplifySqueezeOp,
      SimlifySliceOp, RemoveTransposeOp, RemoveRandomShuffleOp, RemoveShuffleOp,
      SimplifyPackOp, SimplifyReductionOp, SimplifyPadOp, SimplifyPadV2Op,
      RemoveSplitOp, RemoveSplitVOp, MaterializeFillNode,
      MaterializeConstantValuedNode, MaterializeShapeOp, MaterializeRankOp,
      MaterializeSizeOp, MergeConcatOp, SimplifyCaseOp, SimplifySelectOp,
      SimplifySelectV2Op>(&context, *helper_);
}

void ConstantFolding::runOnOperation() {
  // TODO(chiahungduan): Set up the attributes before operation creation.
  // Because of the conveniency, in some cases we set up the device/name later
  // operation creation.

  // TODO(chiahungduan): Do the folding start from constant ops so that we don't
  // need to scan the entire nested ops in every iteration.
  getOperation()->walk([&](Region *region) {
    (void)applyPatternsAndFoldGreedily(*region, final_patterns_);
  });

  // TODO(chiahungduan): Remove dead op if meets certain conditions.
}

std::unique_ptr<Pass> CreateConstantFoldingPass() {
  return std::make_unique<ConstantFolding>();
}

}  // namespace tfg
}  // namespace mlir
