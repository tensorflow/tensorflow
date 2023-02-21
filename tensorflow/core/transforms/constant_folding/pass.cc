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
#include <iterator>
#include <numeric>
#include <optional>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>

#include "absl/container/flat_hash_set.h"
#include "absl/strings/match.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/Twine.h"
#include "mlir/Dialect/Traits.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributeInterfaces.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/ir/dialect.h"
#include "tensorflow/core/ir/importexport/convert_types.h"
#include "tensorflow/core/ir/ops.h"
#include "tensorflow/core/ir/utility.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/transforms/utils/eval_utils.h"
#include "tensorflow/core/transforms/utils/op_cat_helper.h"
#include "tensorflow/core/transforms/utils/utils.h"
#include "tensorflow/core/util/bcast.h"
#include "tensorflow/core/util/device_name_utils.h"

namespace mlir {
namespace tfg {

#define GEN_PASS_DEF_CONSTANTFOLDINGPASS
#include "tensorflow/core/transforms/passes.h.inc"

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
  if (auto t_attr = op->getAttrOfType<TypeAttr>("T")) {
    return t_attr.getValue();
  } else if (auto dtype_attr = op->getAttrOfType<TypeAttr>("dtype")) {
    return dtype_attr.getValue();
  } else if (op->getName().stripDialect() == "LogicalOr" ||
             op->getName().stripDialect() == "LogicalAnd") {
    return builder.getI1Type();
  }
  return *(op->result_type_begin());
}

static FailureOr<TFOp> CreateConstantTensorOp(
    OpBuilder &builder, Location loc, StringRef name_prefix, Type type,
    ValueRange control_operands, TypedAttr tensor_value,
    ArrayRef<NamedAttribute> other_attrs = std::nullopt) {
  if (type.isa<VariantType>()) return failure();
  // TODO(chiahungduan): Reuse ConstOp Like
  // OperationFolder::tryGetOrCreateConstant.
  OperationState state(loc, "tfg.Const");
  state.addTypes({type, ControlType::get(builder.getContext())});

  state.attributes = other_attrs;
  util::EraseRegularNodeAttributes(state.attributes);
  state.attributes.set(
      "dtype", TypeAttr::get(
                   tensor_value.getType().cast<ShapedType>().getElementType()));
  state.attributes.set("value", tensor_value);
  if (!name_prefix.empty()) {
    state.attributes.set(
        TFGraphDialect::getNameAttrKey(),
        builder.getStringAttr(Twine(name_prefix, "/const_folded")));
  }

  state.addOperands(control_operands);
  return TFOp(builder.create(state));
}

static bool IsControlAnchor(TFOp op, TFGraphDialect const *const dialect) {
  return (dialect->IsIdentity(op) || dialect->IsIdentityNSingleInput(op)) &&
         op->getResults().drop_back().use_empty();
}

// We can't anchor control dependencies directly on the switch node: unlike
// other nodes only one of the outputs of the switch node will be generated
// when the switch node is executed, and we need to make sure the control
// dependency is only triggered when the corresponding output is triggered.
// We start by looking for an identity node connected to the output of the
// switch node, and use it to anchor the control dependency.
// @param builder Builder, used for creating the anchor if necessary
// @param value   Output of a switch operation to be replaced
// @param dialect TFG dialect (passed in to avoid cost of looking it up)
static TFOp GetControlAnchorForSwitchResult(
    OpBuilder &builder, OpResult value, TFGraphDialect const *const dialect) {
  assert(builder.getContext()->getLoadedDialect<TFGraphDialect>() == dialect);
  TFOp switch_op = value.getDefiningOp();
  assert(dialect->IsSwitch(switch_op));
  // We cannot get the control edge from the parent op. We instead create a
  // control anchor i.e. an Identity op without non-control uses and get the
  // edge from there.

  // Try to find an existing control anchor
  if (auto it = llvm::find_if(
          value.getUsers(),
          [&](Operation *op) { return IsControlAnchor(op, dialect); });
      it != value.getUsers().end())
    return TFOp(*it);

  // If it doesn't exist, create a new control anchor.
  OperationState identity_op_state(value.getLoc(), "tfg.Identity");
  identity_op_state.addOperands(value);
  identity_op_state.addTypes(
      {value.getType(), ControlType::get(builder.getContext())});
  assert(switch_op->hasAttr("T"));
  identity_op_state.addAttribute("T", switch_op->getAttr("T"));
  TFOp identity_op = builder.create(identity_op_state);
  if (StringAttr device_attr = switch_op.deviceAttr())
    identity_op.setRequestedDevice(device_attr);
  identity_op.setName(Twine(switch_op.name(), "/ControlDependencyCtrl_") +
                      Twine(value.cast<OpResult>().getResultNumber()));
  return identity_op;
}

// Same as LookupControlDependency, except when value originates from a switch
// op. In such cases, we cannot add a control dependency to the parent op since
// the output does not necessarily activate when the switch op activates. We
// add a "control anchor" in the form of an identity op instead.
static Value GetControlDependency(OpBuilder &builder, Value value) {
  if (value.getType().isa<ControlType>()) return value;

  TFGraphDialect *dialect =
      builder.getContext()->getLoadedDialect<TFGraphDialect>();
  assert(dialect);
  if (OpResult result = value.dyn_cast<OpResult>();
      result && dialect->IsSwitch(result.getOwner())) {
    return GetControlAnchorForSwitchResult(builder, result, dialect)
        .controlRet();
  } else {
    return LookupControlDependency(value);
  }
}

// Add control operand to `op` if it doesn't exist.
static void AddControlOperand(Operation *op, Value control,
                              PatternRewriter &rewriter) {
  assert(control.getType().isa<ControlType>());
  if (llvm::is_contained(op->getOperands(), control)) return;
  rewriter.startRootUpdate(op);
  op->insertOperands(op->getNumOperands(), control);
  rewriter.finalizeRootUpdate(op);
}

static FailureOr<TFOp> ReplaceOpWithConstantTensor(
    OpBuilder &builder, TFOp op, ElementsAttr value,
    ArrayRef<StringRef> exclude_attrs = std::nullopt) {
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
  FailureOr<TFOp> const_op = CreateConstantTensorOp(
      builder, op->getLoc(), /*name_prefix=*/"", value.getType(),
      operands_controls, value, attr_list);
  (*const_op).setName(op.nameAttr());
  if (!op.device().empty()) (*const_op).setRequestedDevice(op.deviceAttr());
  return *const_op;
}

static FailureOr<TFOp> ReplaceOpWithIdentity(OpBuilder &builder, TFOp owner,
                                             unsigned idx) {
  OperationState state(owner->getLoc(), "tfg.Identity");
  state.addTypes({owner->getOperand(idx).getType(),
                  ControlType::get(builder.getContext())});
  state.addAttribute(
      "T", TypeAttr::get(GetDataTypeFromOp(builder, owner.getOperation())));

  Value kept_value = owner->getOperand(idx);
  state.addOperands(kept_value);
  auto [non_control_operands, control_operands] = owner.splitOperands();
  for (Value value : non_control_operands) {
    if (value != kept_value)
      state.addOperands(GetControlDependency(builder, value));
  }
  state.addOperands(control_operands);

  Operation *identity_op = builder.create(state);
  TFOp(identity_op).setName(owner.nameAttr());
  if (!owner.device().empty())
    TFOp(identity_op).setRequestedDevice(owner.deviceAttr());
  return TFOp(identity_op);
}

static FailureOr<TFOp> ReplaceOpWithNoOp(OpBuilder &builder, TFOp op) {
  OperationState state(op->getLoc(), "tfg.NoOp");
  // Op may not have non-control results
  if (TFOp(op)->getNumResults() > 1) return failure();

  state.addTypes({ControlType::get(builder.getContext())});

  for (Value value : op->getOperands()) {
    Value control = GetControlDependency(builder, value);
    if (!llvm::is_contained(state.operands, control))
      state.addOperands(control);
  }

  TFOp noop_op = builder.create(state);
  noop_op.setName(op.nameAttr());
  if (!op.device().empty()) noop_op.setRequestedDevice(op.device());
  return noop_op;
}

static FailureOr<TFOp> ReplaceOpWithConstant(OpBuilder &builder, Operation *op,
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

static FailureOr<TFOp> ReplaceOpWithSnapshot(OpBuilder &builder, TFOp op,
                                             int idx) {
  // TODO(chiahungduan): If the graph contains no ops that mutate their
  // inputs, we can use Identity instead of Snapshot.
  // if (!graph_contains_assign_or_inplace_op_)
  auto [non_control_operands, control_operands] = op.splitOperands();

  Value replace_value = op->getOperand(idx);
  OperationState state(op->getLoc(), "tfg.Snapshot");
  state.attributes = op->getAttrDictionary();
  util::EraseRegularNodeAttributes(state.attributes);
  state.addAttribute(
      "T", TypeAttr::get(GetDataTypeFromOp(builder, op.getOperation())));
  // Propagate the designated input through the Snapshot.
  state.addOperands(replace_value);
  // Add all other inputs as control dependencies.
  llvm::append_range(state.operands,
                     OperandControlRetRange(non_control_operands));
  // Append the control operands
  state.addOperands(control_operands);
  state.addTypes(op->getResultTypes());

  Operation *snapshot_op = builder.create(state);
  TFOp(snapshot_op).setName(op.nameAttr());
  if (!op.device().empty())
    TFOp(snapshot_op).setRequestedDevice(op.deviceAttr());
  return TFOp(snapshot_op);
}

static FailureOr<TFOp> ReplaceOpWithBroadcastTo(OpBuilder &builder, TFOp op,
                                                int idx_to_replace) {
  ShapedType tensor_type = (*op->result_type_begin()).cast<ShapedType>();
  if (!tensor_type.hasStaticShape()) return failure();
  ElementsAttr const_attr = ConvertShapeToAttr(tensor_type);

  // Create a vector of control operands. We should not fail beyond this point
  // since GetControlDependency may create a control anchor (a new op).
  SmallVector<Value> control_operands;
  for (auto &it : llvm::enumerate(op.getNonControlOperands())) {
    int idx = it.index();
    Value v = it.value();
    if (idx == idx_to_replace) continue;
    if (llvm::is_contained(control_operands, v)) continue;
    control_operands.push_back(GetControlDependency(builder, v));
  }
  // CreateConstantTensorOp cannot fail; it only fails for variant types and
  // const_attr is a tensor of i32.
  TFOp const_op = *CreateConstantTensorOp(
      builder, op->getLoc(),
      (Twine(op.name(), "/broadcastto_shape_") + std::to_string(idx_to_replace))
          .str(),
      const_attr.getType(), control_operands, const_attr);
  if (!op.device().empty()) const_op.setRequestedDevice(op.device());

  OperationState state(op->getLoc(), "tfg.BroadcastTo");

  state.attributes = op->getAttrDictionary();
  util::EraseRegularNodeAttributes(state.attributes);
  state.addAttribute(
      "T", TypeAttr::get(GetDataTypeFromOp(builder, op.getOperation())));
  state.addAttribute("Tidx", TypeAttr::get(builder.getI32Type()));

  state.addOperands({op->getOperand(idx_to_replace), const_op->getResult(0)});
  state.addOperands(control_operands);
  state.addTypes(op->getResultTypes());

  Operation *broadcast_to_op = builder.create(state);
  TFOp(broadcast_to_op).setName(op.nameAttr());
  if (!op.device().empty())
    TFOp(broadcast_to_op).setRequestedDevice(op.deviceAttr());
  return TFOp(broadcast_to_op);
}

namespace {
// A helper class to see if an operation falls into certain category or has
// certain non-trivial properties.
class OpPropertyHelper : public OpCatHelper {
 public:
  OpPropertyHelper(TFGraphDialect *dialect,
                   bool disable_compressed_tensor_optimization)
      : OpCatHelper(dialect),
        dialect_(dialect),
        disable_compressed_tensor_optimization_(
            disable_compressed_tensor_optimization) {}

  // Return true if the operation modifies the input in-place.
  bool ModifiesInputsInPlace(TFOp op);

  // Return true if this operation doesn't have any side effect.
  bool IsFreeOfSideEffect(TFOp op);

  // Return true if an operation may modify the frame info.
  bool ModifiesFrameInfo(TFOp op) {
    return dialect_->IsEnter(op) || dialect_->IsExit(op) ||
           dialect_->IsNextIteration(op);
  }

  // This combines the results of both MaybeFoldable() and IsFoldableUncached()
  bool IsFoldable(TFOp op);

  // Return if this is a preserved op. It checks the `name` attr.
  bool ShouldPreserveOp(TFOp op);

  // Disable compressed tensor optimization.
  bool DisableCompressedTensorOptimization();

  // Get the TFG dialect instance.
  TFGraphDialect *getDialect() { return dialect_; }

 private:
  // Return true if this operation is safe to be folded. This filter the ops by
  // name.
  bool MaybeFoldable(TFOp op);

  // Return true if this operation is safe to be folded. This filter the ops by
  // the operation property like, it'll check the operands, attributes, .etc.
  bool IsFoldableUncached(TFOp op);

  // A reference to the TFG dialect.
  TFGraphDialect *dialect_;

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
      op_name == "ResourceScatterMin" || op_name == "ResourceScatterMax") {
    return false;
  }

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

  if (dialect_->IsQueue(op)) return false;

  if (dialect_->IsSend(op)) return false;

  return !ModifiesInputsInPlace(op);
}

// To determine if we want to evalue the value of the operation. There several
// kinds operation we don't want to evalute with the eager runtime. Those
// operations may not safe for evaluation or not worth for evaluating because of
// the evaluation cost. For example, Const op already has the constant value
// attached as attribute.
bool OpPropertyHelper::MaybeFoldable(TFOp op) {
  StringRef op_name = op->getName().stripDialect();

  if (dialect_->IsConstant(op)) return false;

  // Don't fold stateful ops such as TruncatedNormal.
  if (!IsFreeOfSideEffect(op)) return false;

  // Fold fetch nodes iff it has a single fanout. Note that if a fetch node
  // has a single fanout, it would be rewritten as a constant with the same
  // node name, and therefore users are still able to fetch it. This is not
  // the case if the node has multiple fanouts, and constant folding would
  // replace the node with multiple constants (each for one fanout) with
  // new names, and as a result users would not be able to fetch the node any
  // more with the original node name.
  if (ShouldPreserveOp(op) &&
      !(llvm::any_of(  // Is a fetch node
            op->getResults().drop_back().getUsers(),
            [&](TFOp child_op) { return dialect_->IsReturn(child_op); }) &&
        op->getNumResults() == 2  // Has single non-control output
        ))
    return false;

  // Skips ops that don't benefit from folding.
  if (dialect_->IsPlaceholder(op)) return false;

  if (dialect_->IsFakeParam(op)) return false;

  // Skip certain control flow nodes, they can't be folded.
  if (ModifiesFrameInfo(op)) return false;

  if (op_name == "AccumulateNV2") return false;

  // Removing LoopCond nodes can screw up the partitioner.
  if (op_name == "LoopCond") return false;

  // TODO(chiahungduan): add fold_quantization_emulation arg.
  // if (!fold_quantization_emulation && IsQuantizationEmulation(op)) return
  // false;

  if (dialect_->IsRestore(op) || op_name.contains("Save") ||
      op_name.contains("Reader"))
    return false;

  if (op_name.contains("Quantized") ||
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
  bool is_merge = dialect_->IsMerge(op);
  for (Value operand : operands) {
    TFOp operand_op = operand.getDefiningOp();
    if (operand_op && dialect_->IsConstant(operand_op)) {
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
      (dialect_->IsFill(op) || dialect_->IsZerosLike(op) ||
       dialect_->IsOnesLike(op))) {
    return false;
  }

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
  // TODO(tlongeri): Find a better way to identify preserved ops. A node has its
  // control output returned if it is a node-to-be-preserved (in
  // LiftGraphToFunc) - *not* iff, so the following check is overly broad:
  return llvm::any_of(op.controlRet().getUsers(), [&](TFOp child_op) {
    return dialect_->IsReturn(child_op);
  });
}

bool OpPropertyHelper::DisableCompressedTensorOptimization() {
  return disable_compressed_tensor_optimization_;
}

static bool IsValidConstShapeForMulConvPushDown(StringAttr data_format,
                                                ShapedType filter_shape,
                                                ShapedType const_shape) {
  if (!filter_shape.hasStaticShape() || !const_shape.hasStaticShape())
    return false;
  if (const_shape.getRank() <= data_format.size() &&
      const_shape.getNumElements() == 1) {
    return true;
  }
  if (data_format == "NHWC" || data_format == "NDHWC") {
    SmallVector<int64_t> broadcast_shape;
    if (!OpTrait::util::getBroadcastedShape(
            filter_shape.getShape(), const_shape.getShape(), broadcast_shape)) {
      return false;
    }

    // TODO(chiahungduan): Symbolic shape equivalence is acceptable.
    if (filter_shape.getShape() != llvm::ArrayRef(broadcast_shape))
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
template <typename ConcreteType, template <typename> class... Traits>
class ConstantPatternBase : public RewritePattern,
                            public Traits<ConcreteType>... {
 public:
  using RewritePattern::RewritePattern;

  ConstantPatternBase(StringRef opName, OpPropertyHelper &helper)
      : RewritePattern(opName, PatternBenefit(1),
                       helper.getDialect()->getContext()),
        helper_(helper),
        dialect_(helper.getDialect()) {}
  ConstantPatternBase(MatchAnyOpTypeTag tag, OpPropertyHelper &helper)
      : RewritePattern(tag, PatternBenefit(1),
                       helper.getDialect()->getContext()),
        helper_(helper),
        dialect_(helper.getDialect()) {}

 protected:
  OpPropertyHelper &helper_;
  TFGraphDialect *dialect_;
};

// A base trait which can help with classifying patterns and filter patterns
// according to the classification.
template <typename ConcreteType>
struct TraitBase {
  ConcreteType *getPattern() { return static_cast<ConcreteType *>(this); }
};

// A trait indicates that the pattern will fold the root operation into a
// another operation like a constant op.
template <typename ConcreteType>
struct FolderTrait : public TraitBase<ConcreteType> {};

// A trait indicates that the pattern may propagate the constant operands to its
// users.
template <typename ConcreteType>
struct PropagationTrait : public TraitBase<ConcreteType> {};

template <typename ConcreteType>
using FolderPatternBase = ConstantPatternBase<ConcreteType, FolderTrait>;

template <typename ConcreteType>
using PropagationPatternBase =
    ConstantPatternBase<ConcreteType, PropagationTrait>;
}  // namespace

// EvaluateConstant maps the implementation of FoldGraph in
// ConstantFolding::FoldGraph in grappler/optimizers/constant_folding.cc
class EvaluateConstant : public FolderPatternBase<EvaluateConstant> {
 public:
  explicit EvaluateConstant(OpPropertyHelper &helper)
      : FolderPatternBase<EvaluateConstant>(MatchAnyOpTypeTag(), helper),
        has_folded_(BoolAttr::get(helper.getDialect()->getContext(), true)),
        folded_attr_name_(
            StringAttr::get(helper.getDialect()->getContext(), "has_folded")),
        cpu_device_(std::make_unique<util::SimpleDevice>()),
        resource_mgr_(std::make_unique<tensorflow::ResourceMgr>()) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (!helper_.IsFoldable(op)) return failure();

    // The op has been folded but it has multiple results which we can just
    // replace it with a constant op and it also has control edges which prevent
    // it from removing. Use the attr to avoid evaluating them again.
    if (op->hasAttr(folded_attr_name_)) return failure();

    // If the op has no users, don't invoke the eager runtime.
    if (llvm::all_of(op->getResults().drop_back(),
                     [](Value v) { return v.use_empty(); })) {
      return failure();
    }

    SmallVector<ElementsAttr> const_operands;
    for (Value operand : TFOp(op).getNonControlOperands()) {
      Operation *defining_op = operand.getDefiningOp();
      if (defining_op && dialect_->IsConstant(defining_op)) {
        const_operands.push_back(
            defining_op->getAttrOfType<ElementsAttr>("value"));
      } else {
        return failure();
      }
    }

    SmallVector<TypedAttr> result;
    if (failed(util::EvaluateOperation(cpu_device_.get(), resource_mgr_.get(),
                                       op, const_operands, result))) {
      return failure();
    }

    // Check if CreateConstantTensorNode ops can fail before creating any nodes
    // TODO(tlongeri): Is CreateConstantTensorNode check correct? Shouldn't it
    // always be a ShapedType?
    for (TypedAttr r : result)
      if (r && r.getType().isa<VariantType>()) return failure();

    StringAttr name_attr = static_cast<TFGraphDialect *>(op->getDialect())
                               ->getNameAttrIdentifier();
    SmallVector<Value> control_operands(
        OperandControlRetRange(op->getOperands()));

    SmallVector<TFOp> const_ops(result.size());
    for (auto &it : llvm::enumerate(result)) {
      TypedAttr attr = it.value();
      // Null values represent dead outputs. They can result from evaluating a
      // switch op.
      if (!attr) continue;
      if (op->getResult(it.index()).use_empty()) continue;
      // CreateConstantTensorOp cannot return failure, we checked failure
      // conditions above.
      TFOp const_op = *CreateConstantTensorOp(
          rewriter, op->getLoc(),
          (Twine(TFOp(op).name(), "/eval_") + Twine(it.index())).str(),
          attr.getType(), control_operands, attr,
          NamedAttribute(name_attr, TFOp(op).nameAttr()));
      if (StringAttr device_attr = TFOp(op).deviceAttr())
        const_op.setRequestedDevice(device_attr);
      const_ops[it.index()] = const_op;
    }

    // If this is single output, just replace the op.
    if (const_ops.size() == 1) {
      // Use the same node name for the replacement. Note that even this is not
      // in nodes_to_preserve, certain cases may still expect the op has the
      // same name after folding.
      TFOp const_op = const_ops[0];
      assert(const_op);
      const_op.setName(TFOp(op).nameAttr());
      rewriter.replaceOp(op, const_op->getResults());
    } else {
      for (auto &it : llvm::enumerate(const_ops)) {
        if (!it.value()) continue;
        for (OpOperand &use :
             llvm::make_early_inc_range(op->getResult(it.index()).getUses())) {
          rewriter.startRootUpdate(use.getOwner());
          use.set(it.value()->getResult(0));
          rewriter.finalizeRootUpdate(use.getOwner());
        }
      }
      // All the non-control outputs are replaced with constant ops, except for
      // dead outputs (in the case of a switch op).
      // If the op has no dead outputs and no uses of its control output, then
      // it can be removed.
      // Dead code removal for switches with dead outputs (because of a constant
      // pred) is handled in Grappler's LoopOptimizer pass.
      if (op->use_empty()) {
        rewriter.eraseOp(op);
      } else {
        // We can't remove it directly. To avoid folding it again, add an attr
        // to identity these ops. This will be removed in the end of constant
        // folding pass.
        op->setAttr(folded_attr_name_, has_folded_);
      }
    }

    return success();
  }

 private:
  BoolAttr has_folded_;
  StringAttr folded_attr_name_;
  std::unique_ptr<util::SimpleDevice> cpu_device_;
  std::unique_ptr<tensorflow::ResourceMgr> resource_mgr_;
};

// This implementation is mapped to the ShapeOp materialization in
// ConstantFolding::MaterializeShapes in grappler/optimizers/constant_folding.cc
class MaterializeShapeOp : public FolderPatternBase<MaterializeShapeOp> {
 public:
  explicit MaterializeShapeOp(OpPropertyHelper &helper)
      : FolderPatternBase<MaterializeShapeOp>("tfg.Shape", helper) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    Value input = op->getOperand(0);

    auto input_shape = input.getType().cast<ShapedType>();
    if (!input_shape.hasStaticShape()) return failure();

    // TODO(rmlarsen): Remove this workaround for b/150861569
    // The bug involves an expression of the form Shape(ExpandDims(x)
    // with an incorrectly inferred zero-size first dimension.
    if (!input_shape.getShape().empty() && input_shape.getShape()[0] == 0)
      return failure();

    Type output_dtype =
        op->getResult(0).getType().cast<ShapedType>().getElementType();
    ElementsAttr const_attr = CreateElementsAttrOfTypeValues(
        output_dtype, {input_shape.getRank()}, input_shape.getShape());

    // Add the control edge to `input` to ensure that the constant value will
    // only be run in the cases where Shape would have been run in the original
    // graph.
    TFOp const_op = *CreateConstantTensorOp(
        rewriter, op->getLoc(), /*name_prefix=*/"", const_attr.getType(),
        GetControlDependency(rewriter, input), const_attr, op->getAttrs());
    const_op.setName(TFOp(op).nameAttr());

    rewriter.replaceOp(op, const_op->getResults());

    return success();
  }
};

// This implementation is mapped to the SizeOp materialization in
// ConstantFolding::MaterializeShapes in grappler/optimizers/constant_folding.cc
class MaterializeSizeOp : public FolderPatternBase<MaterializeSizeOp> {
 public:
  explicit MaterializeSizeOp(OpPropertyHelper &helper)
      : FolderPatternBase<MaterializeSizeOp>("tfg.Size", helper) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    Value input = op->getOperand(0);

    auto input_shape = input.getType().cast<ShapedType>();
    if (!input_shape.hasStaticShape()) return failure();

    ShapedType result_type = (*op->result_type_begin()).cast<ShapedType>();
    if (!result_type.getElementType().isIntOrIndexOrFloat()) return failure();

    ElementsAttr const_attr = CreateElementsAttrOfTypeValues(
        result_type.getElementType(), {},
        ArrayRef<int64_t>(input_shape.getNumElements()));

    // Add the control edge to `input` to ensure that the constant value will
    // only be run in the cases where Size would have been run in the original
    // graph.
    TFOp const_op = *CreateConstantTensorOp(
        rewriter, op->getLoc(), /*name_prefix=*/"", const_attr.getType(),
        GetControlDependency(rewriter, input), const_attr, op->getAttrs());
    const_op.setName(TFOp(op).nameAttr());

    rewriter.replaceOp(op, const_op->getResults());

    return success();
  }
};

// This implementation is mapped to the RankOp materialization in
// ConstantFolding::MaterializeShapes in grappler/optimizers/constant_folding.cc
class MaterializeRankOp : public FolderPatternBase<MaterializeRankOp> {
 public:
  explicit MaterializeRankOp(OpPropertyHelper &helper)
      : FolderPatternBase<MaterializeRankOp>("tfg.Rank", helper) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    Value input = op->getOperand(0);

    auto input_shape = input.getType().cast<ShapedType>();
    if (!input_shape.hasRank()) return failure();

    ShapedType result_type = (*op->result_type_begin()).cast<ShapedType>();
    if (!result_type.getElementType().isIntOrIndexOrFloat()) return failure();

    ElementsAttr const_attr = CreateElementsAttrOfTypeValues(
        result_type.getElementType(), {}, ArrayRef<int>(input_shape.getRank()));

    // Add the control edge to `input` to ensure that the constant value will
    // only be run in the cases where Rank would have been run in the original
    // graph.
    TFOp const_op = *CreateConstantTensorOp(
        rewriter, op->getLoc(), /*name_prefix=*/"", const_attr.getType(),
        GetControlDependency(rewriter, input), const_attr, op->getAttrs());
    const_op.setName(TFOp(op).nameAttr());

    rewriter.replaceOp(op, const_op->getResults());

    return success();
  }
};

// This implementation is mapped to the TensorArraySizeV3 materialization in
// ConstantFolding::MaterializeShapes in grappler/optimizers/constant_folding.cc
class MaterializeTensorArraySizeV3Op
    : public FolderPatternBase<MaterializeTensorArraySizeV3Op> {
 public:
  explicit MaterializeTensorArraySizeV3Op(OpPropertyHelper &helper)
      : FolderPatternBase<MaterializeTensorArraySizeV3Op>(
            "tfg.TensorArraySizeV3", helper) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    Operation *handle_op = op->getOperand(0).getDefiningOp();
    if (!handle_op || handle_op->getNumOperands() == 0) return failure();

    auto dynamic_size = handle_op->getAttrOfType<BoolAttr>("dynamic_size");
    if (dynamic_size && dynamic_size.getValue()) return failure();

    Operation *array_size = handle_op->getOperand(0).getDefiningOp();
    if (!array_size || !dialect_->IsConstant(array_size)) return failure();

    // Don't materialize 0 sizes to avoid triggering incorrect static checks.
    // A 0 sized array that can't grow isn't useful anyway.
    auto size_attr = array_size->getAttrOfType<SplatElementsAttr>("value");
    if (!size_attr || !size_attr.getElementType().isInteger(32))
      return failure();
    if (size_attr.getSplatValue<IntegerAttr>().getInt() == 0) return failure();

    SmallVector<Value> control_operands;
    control_operands.push_back(TFOp(handle_op).controlRet());
    control_operands.push_back(
        GetControlDependency(rewriter, op->getOperand(1)));
    // CreateConstantTensorOp cannot fail; its type is tensor of i32
    TFOp const_op = *CreateConstantTensorOp(
        rewriter, op->getLoc(), /*name_prefix=*/"", size_attr.getType(),
        control_operands, size_attr, op->getAttrs());
    const_op.setName(TFOp(op).nameAttr());

    rewriter.replaceOp(op, const_op->getResults());

    return success();
  }
};

// This implementation is mapped to the ShapeN materialization in
// ConstantFolding::MaterializeShapes in grappler/optimizers/constant_folding.cc
class MaterializeShapeNOp : public FolderPatternBase<MaterializeShapeNOp> {
 public:
  explicit MaterializeShapeNOp(OpPropertyHelper &helper)
      : FolderPatternBase<MaterializeShapeNOp>("tfg.ShapeN", helper) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    for (const auto &it : llvm::enumerate(TFOp(op).getNonControlOperands())) {
      Value operand = op->getOperand(it.index());

      auto operand_shape = operand.getType().cast<ShapedType>();
      if (!operand_shape.hasStaticShape()) continue;

      if (op->getResults()[it.index()].use_empty()) continue;

      ElementsAttr const_attr = ConvertShapeToAttr(operand_shape);

      FailureOr<TFOp> const_op = CreateConstantTensorOp(
          rewriter, op->getLoc(), TFOp(op).name(), *(op->result_type_begin()),
          TFOp(op).controlRet(), const_attr);
      if (failed(const_op)) return failure();

      (*const_op).setName(Twine(TFOp(op).name(), "/matshapes_") +
                          std::to_string(it.index()));
      if (!TFOp(op).device().empty())
        (*const_op).setRequestedDevice(TFOp(op).deviceAttr());

      // TODO(chiahungduan): Do we need to handle `direct_edges_exist` in
      // ConstantFolding::MaterializeShapes for ShapeN?

      for (OpOperand &user :
           llvm::make_early_inc_range(op->getResult(it.index()).getUses())) {
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
class MaterializeBroadcastGradientArgsOp
    : public PropagationPatternBase<MaterializeBroadcastGradientArgsOp> {
 public:
  explicit MaterializeBroadcastGradientArgsOp(OpPropertyHelper &helper)
      : PropagationPatternBase<MaterializeBroadcastGradientArgsOp>(
            "tfg.BroadcastGradientArgs", helper) {}
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    Operation *s0 = op->getOperand(0).getDefiningOp();
    Operation *s1 = op->getOperand(1).getDefiningOp();
    if (!s0 || !s1) return failure();

    if (!dialect_->IsShape(s0) && !dialect_->IsConstant(s0)) return failure();
    if (!dialect_->IsShape(s1) && !dialect_->IsConstant(s1)) return failure();

    // This operation has been optimized.
    if (op->getResult(0).use_empty() || op->getResult(1).use_empty())
      return failure();

    auto get_shape = [this](Operation *op,
                            SmallVector<int64_t> &shape) -> bool {
      if (dialect_->IsShape(op)) {
        auto type = op->getOperand(0).getType().cast<ShapedType>();
        if (!type.hasRank()) return false;

        llvm::append_range(shape, type.getShape());
      } else {
        auto attr = op->getAttrOfType<ElementsAttr>("value");
        if (!attr) return false;

        Type element_type = attr.getElementType();
        if (element_type.isInteger(32)) {
          llvm::append_range(shape, attr.getValues<int32_t>());
        } else if (element_type.isInteger(64)) {
          llvm::append_range(shape, attr.getValues<int64_t>());
        } else {
          return false;
        }
      }
      return true;
    };

    SmallVector<int64_t> s0_shape;
    SmallVector<int64_t> s1_shape;
    if (!get_shape(s0, s0_shape) || !get_shape(s1, s1_shape)) return failure();

    const int common_dims = std::min(s0_shape.size(), s1_shape.size());
    for (int i = 0; i < common_dims; ++i) {
      if (s0_shape[i] >= 0 && s1_shape[i] >= 0) continue;

      // TODO(chiahungduan): Check if two dims are symbolically equal. Grappler
      // stores the symbolic shape information with dim < -1 which is not a
      // convention in TFG. Use symbolic shape information instead.

      // Return failure if two dims are symbolically unequal.
      return failure();
    }

    for (int i = common_dims; i < s0_shape.size(); ++i)
      if (s0_shape[i] < 0) return failure();
    for (int i = common_dims; i < s1_shape.size(); ++i)
      if (s1_shape[i] < 0) return failure();

    tensorflow::BCast::Vec s0_vec(s0_shape.begin(), s0_shape.end());
    tensorflow::BCast::Vec s1_vec(s1_shape.begin(), s1_shape.end());
    tensorflow::BCast bcast(s0_vec, s1_vec);
    if (!bcast.IsValid()) return failure();

    tensorflow::BCast::Vec reduce_dims[2];
    reduce_dims[0] = bcast.grad_x_reduce_idx();
    reduce_dims[1] = bcast.grad_y_reduce_idx();

    auto type_attr = op->getAttrOfType<TypeAttr>("T");
    if (!type_attr) return failure();
    if (!type_attr.getValue().isIntOrIndexOrFloat()) return failure();

    SmallVector<Value, 2> const_values;
    for (int j = 0; j < 2; ++j) {
      int reduction_indices = reduce_dims[j].size();
      ElementsAttr const_attr = CreateElementsAttrOfTypeValues(
          type_attr.getValue(), {reduction_indices},
          llvm::ArrayRef<int64_t>(reduce_dims[j].data(), reduction_indices));
      FailureOr<TFOp> const_op = CreateConstantTensorOp(
          rewriter, op->getLoc(), TFOp(op).name(), op->getResultTypes()[j],
          TFOp(op).controlRet(), const_attr);
      if (failed(const_op)) return failure();

      (*const_op).setName(Twine(TFOp(op).name(), "/bcastargs_") +
                          std::to_string(j));
      if (!TFOp(op).device().empty())
        (*const_op).setRequestedDevice(TFOp(op).deviceAttr());
      const_values.push_back((*const_op)->getResult(0));
    }

    for (OpOperand &user :
         llvm::make_early_inc_range(op->getResult(0).getUses())) {
      rewriter.startRootUpdate(user.getOwner());
      user.set(const_values[0]);
      rewriter.finalizeRootUpdate(user.getOwner());
    }
    for (OpOperand &user :
         llvm::make_early_inc_range(op->getResult(1).getUses())) {
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
class MaterializeReductionIndices
    : public PropagationPatternBase<MaterializeReductionIndices> {
 public:
  explicit MaterializeReductionIndices(OpPropertyHelper &helper)
      : PropagationPatternBase<MaterializeReductionIndices>(MatchAnyOpTypeTag(),
                                                            helper) {}
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (!dialect_->IsReduction(op)) return failure();

    Operation *indices = op->getOperand(1).getDefiningOp();
    // The reduction indices are already constant, there's nothing to do.
    if (!indices || dialect_->IsConstant(indices)) return failure();

    auto indices_shape = indices->getResult(0).getType().cast<ShapedType>();
    if (!indices_shape.hasRank()) return failure();
    if (!indices_shape.getElementType().isInteger(32) &&
        !indices_shape.getElementType().isInteger(64)) {
      return failure();
    }

    auto input_shape = op->getOperand(0).getType().cast<ShapedType>();
    // Unexpected graph, don't try to change it.
    if (!input_shape.hasRank() || input_shape.getRank() < 1) return failure();

    auto output_shape = op->getResult(0).getType().cast<ShapedType>();
    const int output_rank =
        output_shape.hasRank() ? output_shape.getRank() : -1;

    bool full_reduction = output_rank == 0 || (indices_shape.hasStaticShape() &&
                                               indices_shape.getNumElements() ==
                                                   input_shape.getRank());

    if (!full_reduction) {
      // A full reduction will generate a tensor of one of the shapes
      // [], [1], [1, 1], [1, 1, ...]. Even if we do not know the number of
      // elements in the output of the reduction, we may deduce it from reshape
      // nodes following it.
      for (Operation *user : op->getResult(0).getUsers()) {
        full_reduction = false;
        if (!dialect_->IsReshape(user)) return failure();

        auto shape = user->getResult(0).getType().cast<ShapedType>();
        if (!shape.hasStaticShape() || shape.getNumElements() != 1)
          return failure();
        else
          full_reduction = true;
      }
      if (!full_reduction) return failure();
    }

    // We know it's a full reduction. We can generate the full set of indices
    // to reduce as a constant node.
    SmallVector<int> elements(input_shape.getRank());
    std::iota(elements.begin(), elements.end(), 0);

    ElementsAttr const_attr = CreateElementsAttrOfTypeValues(
        indices_shape.getElementType(), {input_shape.getRank()},
        llvm::ArrayRef(elements));

    FailureOr<TFOp> const_op = CreateConstantTensorOp(
        rewriter, indices->getLoc(), Twine(TFOp(op).name(), "/indices").str(),
        const_attr.getType(), TFOp(indices).controlRet(), const_attr);
    if (failed(const_op)) return failure();

    if (TFOp(op).deviceAttr())
      (*const_op).setRequestedDevice(TFOp(op).deviceAttr());

    rewriter.startRootUpdate(op);
    op->setOperand(1, (*const_op)->getResults()[0]);
    rewriter.finalizeRootUpdate(op);

    return success();
  }
};

// This implementation is mapped to the constant value materialization in
// ConstantFolding::MaterializeConstantValuedNode in
// grappler/optimizers/constant_folding.cc
class MaterializeFillNode : public FolderPatternBase<MaterializeFillNode> {
 public:
  explicit MaterializeFillNode(OpPropertyHelper &helper)
      : FolderPatternBase<MaterializeFillNode>("tfg.Fill", helper) {}
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (helper_.DisableCompressedTensorOptimization()) return failure();
    // Only handles single result op. Note that another result is control ret.
    if (op->getNumResults() != 2) return failure();

    auto output_type = op->getResult(0).getType().cast<ShapedType>();
    if (!output_type.hasStaticShape()) return failure();
    if (!output_type.isIntOrIndexOrFloat()) return failure();

    Operation *dim = op->getOperand(0).getDefiningOp();
    Operation *value = op->getOperand(1).getDefiningOp();
    if (!dim || !value) return failure();
    // In grappler's constant folding, they also check if `dim` is constant.
    // Which is redundant because it's constant property is never used.
    if (!dialect_->IsConstant(value)) return failure();

    ElementsAttr const_attr = CreateElementsAttrOfTypeValues(
        output_type.getElementType(), output_type.getShape(),
        {value->getAttrOfType<ElementsAttr>("value")});

    FailureOr<TFOp> const_op = ReplaceOpWithConstantTensor(
        rewriter, op, const_attr,
        /*exclude_attrs=*/ArrayRef<StringRef>({"T", "index_type"}));
    if (failed(const_op)) return failure();

    rewriter.replaceOp(op, (*const_op)->getResults());

    return success();
  }
};

// This implementation is mapped to the constant value materialization in
// ConstantFolding::MaterializeConstantValuedNode in
// grappler/optimizers/constant_folding.cc
class MaterializeConstantValuedNode
    : public FolderPatternBase<MaterializeConstantValuedNode> {
 public:
  explicit MaterializeConstantValuedNode(OpPropertyHelper &helper)
      : FolderPatternBase<MaterializeConstantValuedNode>(MatchAnyOpTypeTag(),
                                                         helper) {}
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (helper_.DisableCompressedTensorOptimization()) return failure();
    // Only handles single result op. Note that another result is control ret.
    if (op->getNumResults() != 2) return failure();

    // FillOp is handled in MaterializeFillNode pattern.
    if (dialect_->IsFill(op)) return failure();
    const bool is_zeros_like = dialect_->IsZerosLike(op);
    if (!is_zeros_like && !dialect_->IsOnesLike(op)) return failure();

    // TODO(chiahungduan): If op->getOperand(0) has static shape, can we use
    // that to materialize?
    auto output_type = op->getResult(0).getType().cast<ShapedType>();
    if (!output_type.hasStaticShape()) return failure();

    int value = is_zeros_like ? 0 : 1;
    Type output_element_type = output_type.getElementType();
    if (!output_element_type.isIntOrIndexOrFloat()) return failure();

    ElementsAttr const_attr;
    if (output_element_type.isIntOrIndex()) {
      const_attr = SplatElementsAttr::get(
          output_type,
          APInt(output_element_type.getIntOrFloatBitWidth(), value));
    } else {
      const_attr = SplatElementsAttr::get(
          output_type,
          APFloat(output_element_type.cast<FloatType>().getFloatSemantics(),
                  value));
    }

    FailureOr<TFOp> const_op =
        ReplaceOpWithConstantTensor(rewriter, op, const_attr);
    if (failed(const_op)) return failure();

    rewriter.replaceOp(op, (*const_op)->getResults());
    return success();
  }
};

// This implementation is mapped to the output value materialization in
// ConstantFolding::MaterializeOutputValues in
// grappler/optimizers/constant_folding.cc
class MaterializeOutputValue
    : public PropagationPatternBase<MaterializeOutputValue> {
 public:
  explicit MaterializeOutputValue(OpPropertyHelper &helper)
      : PropagationPatternBase<MaterializeOutputValue>(MatchAnyOpTypeTag(),
                                                       helper) {}
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
template <typename ConcreteType>
class MergeNodeFoldingBase : public PropagationPatternBase<ConcreteType> {
 protected:
  MergeNodeFoldingBase(StringRef op_name, OpPropertyHelper &helper)
      : PropagationPatternBase<ConcreteType>(op_name, helper),
        zero_dim_i32_tensor_type_(RankedTensorType::get(
            std::nullopt,
            IntegerType::get(helper.getDialect()->getContext(), 32))) {}

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

    // The node may have been optimized.
    if (llvm::all_of(op->getResults().drop_back(),
                     [](Value v) { return v.use_empty(); })) {
      return failure();
    }

    int idx = 0;
    for (Value operand : TFOp(op).getNonControlOperands()) {
      Operation *operand_op = operand.getDefiningOp();
      if (!operand_op) continue;
      if (!this->dialect_->IsConstant(operand_op)) continue;
      if (!TFOp(operand_op).getControlOperands().empty()) continue;

      FailureOr<TFOp> const_out = CreateConstantTensorOp(
          rewriter, op->getLoc(), TFOp(op).name(),
          *(operand_op->result_type_begin()), TFOp(op).controlRet(),
          operand_op->getAttrOfType<ElementsAttr>("value"), op->getAttrs());
      if (failed(const_out)) return failure();
      (*const_out).setName(Twine(TFOp(op).name(), "/const"));
      if (!TFOp(op).device().empty())
        (*const_out).setRequestedDevice(TFOp(op).device());

      FailureOr<TFOp> const_index = CreateConstantTensorOp(
          rewriter, op->getLoc(), TFOp(op).name(), rewriter.getIntegerType(32),
          TFOp(op).controlRet(),
          DenseElementsAttr::get(zero_dim_i32_tensor_type_, idx++));
      if (failed(const_index)) return failure();

      (*const_index).setName(Twine(TFOp(op).name(), "/index"));
      if (!TFOp(op).device().empty())
        (*const_index).setRequestedDevice(TFOp(op).device());

      for (OpOperand &user :
           llvm::make_early_inc_range(op->getResults()[0].getUses())) {
        rewriter.startRootUpdate(user.getOwner());
        user.set((*const_out)->getResult(0));
        rewriter.finalizeRootUpdate(user.getOwner());
      }
      for (OpOperand &user :
           llvm::make_early_inc_range(op->getResults()[1].getUses())) {
        rewriter.startRootUpdate(user.getOwner());
        user.set((*const_index)->getResult(0));
        rewriter.finalizeRootUpdate(user.getOwner());
      }

      // Already found an avaiable input.
      return success();
    }
    return failure();
  }

  RankedTensorType zero_dim_i32_tensor_type_;
};

class MergeNodeFolding : public MergeNodeFoldingBase<MergeNodeFolding> {
 public:
  explicit MergeNodeFolding(OpPropertyHelper &helper)
      : MergeNodeFoldingBase("tfg.Merge", helper) {}
};

class RefMergeNodeFolding : public MergeNodeFoldingBase<RefMergeNodeFolding> {
 public:
  explicit RefMergeNodeFolding(OpPropertyHelper &helper)
      : MergeNodeFoldingBase("tfg.RefMerge", helper) {}
};

class XlaMergeNodeFolding : public MergeNodeFoldingBase<XlaMergeNodeFolding> {
 public:
  explicit XlaMergeNodeFolding(OpPropertyHelper &helper)
      : MergeNodeFoldingBase("tfg.XlaMerge", helper) {}
};

// This implementation is mapped with ConstantFolding::RemoveSplitOrSplitVin in
// grappler/optimizers/constant_folding.cc
class RemoveSplitOp : public FolderPatternBase<RemoveSplitOp> {
 public:
  explicit RemoveSplitOp(OpPropertyHelper &helper)
      : FolderPatternBase<RemoveSplitOp>("tfg.Split", helper) {}
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
class RemoveSplitVOp : public FolderPatternBase<RemoveSplitVOp> {
 public:
  explicit RemoveSplitVOp(OpPropertyHelper &helper)
      : FolderPatternBase<RemoveSplitVOp>("tfg.SplitV", helper) {}
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    auto num_split_attr = op->getAttrOfType<IntegerAttr>("num_split");
    if (!num_split_attr || num_split_attr.getInt() != 1) return failure();
    FailureOr<TFOp> identity = ReplaceOpWithIdentity(rewriter, op, 0);
    if (failed(identity)) return failure();
    rewriter.replaceOp(op, (*identity)->getResults());
    return success();
  }
};

// TODO(chiahungduan): Do we still have "Shuffle" op?
// This implementation is mapped with ConstantFolding::RemoveShuffleOrTranspose
// in grappler/optimizers/constant_folding.cc
class RemoveShuffleOp : public FolderPatternBase<RemoveShuffleOp> {
 public:
  explicit RemoveShuffleOp(OpPropertyHelper &helper)
      : FolderPatternBase<RemoveShuffleOp>("tfg.Shuffle", helper) {}
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    Operation *perm_op = op->getOperand(1).getDefiningOp();
    if (!perm_op || !dialect_->IsConstant(perm_op)) return failure();
    ElementsAttr perm_tensor = perm_op->getAttrOfType<ElementsAttr>("value");
    if (!perm_tensor) return failure();

    ShapedType x_shape = op->getOperand(0).getType().cast<ShapedType>();
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
class RemoveTransposeOp : public FolderPatternBase<RemoveTransposeOp> {
 public:
  explicit RemoveTransposeOp(OpPropertyHelper &helper)
      : FolderPatternBase<RemoveTransposeOp>("tfg.Transpose", helper) {}
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    Operation *perm_op = op->getOperand(1).getDefiningOp();
    if (!perm_op || !dialect_->IsConstant(perm_op)) return failure();
    ElementsAttr perm_tensor = perm_op->getAttrOfType<ElementsAttr>("value");
    if (!perm_tensor) return failure();

    ShapedType x_shape = op->getOperand(0).getType().cast<ShapedType>();
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
class RemoveRandomShuffleOp : public FolderPatternBase<RemoveRandomShuffleOp> {
 public:
  explicit RemoveRandomShuffleOp(OpPropertyHelper &helper)
      : FolderPatternBase<RemoveRandomShuffleOp>("tfg.RandomShuffle", helper) {}
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    auto shape = op->getOperand(0).getType().cast<ShapedType>();
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
class RemoveReverse : public FolderPatternBase<RemoveReverse> {
 public:
  explicit RemoveReverse(OpPropertyHelper &helper)
      : FolderPatternBase<RemoveReverse>("tfg.ReverseV2", helper) {}
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    ShapedType tensor_type = op->getOperand(0).getType().cast<ShapedType>();
    if (!tensor_type.hasRank()) return failure();

    Operation *dim_op = op->getOperand(1).getDefiningOp();
    if (!dim_op || !dialect_->IsConstant(dim_op)) return failure();

    auto dim_attr = dim_op->getAttrOfType<ElementsAttr>("value");
    DenseSet<int> target_axis;
    for (unsigned i = 0; i < dim_attr.getNumElements(); ++i) {
      // Value of axis can be negative.
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

    for (unsigned i = 0; i < tensor_type.getRank(); ++i) {
      if (tensor_type.getShape()[i] != 1 && target_axis.contains(i))
        return failure();
    }

    FailureOr<TFOp> identity = ReplaceOpWithIdentity(rewriter, op, 0);
    if (failed(identity)) return failure();
    rewriter.replaceOp(op, (*identity)->getResults());

    return success();
  }
};

// This implementation is mapped with ConstantFolding::SimplifySlice
// in grappler/optimizers/constant_folding.cc
class SimplifySliceOp : public FolderPatternBase<SimplifySliceOp> {
 public:
  explicit SimplifySliceOp(OpPropertyHelper &helper)
      : FolderPatternBase<SimplifySliceOp>("tfg.Slice", helper) {}
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    Operation *begin_op = op->getOperand(1).getDefiningOp();
    Operation *size_op = op->getOperand(2).getDefiningOp();
    if (!begin_op || !size_op) return failure();

    if (!dialect_->IsConstant(begin_op) || !dialect_->IsConstant(size_op))
      return failure();

    auto begin_attr = begin_op->getAttrOfType<ElementsAttr>("value");
    auto size_attr = size_op->getAttrOfType<ElementsAttr>("value");

    ShapedType input_type = op->getOperand(0).getType().cast<ShapedType>();
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
class SimplifyStridedSlice : public FolderPatternBase<SimplifyStridedSlice> {
 public:
  explicit SimplifyStridedSlice(OpPropertyHelper &helper)
      : FolderPatternBase<SimplifyStridedSlice>("tfg.StridedSlice", helper) {}
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    // Skip ops with new/shrink axis mask, since they involve dimension changes.
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

    auto begin_mask_attr = op->getAttrOfType<IntegerAttr>("begin_mask");
    auto end_mask_attr = op->getAttrOfType<IntegerAttr>("end_mask");
    auto ellipsis_mask_attr = op->getAttrOfType<IntegerAttr>("ellipsis_mask");
    if (!begin_mask_attr || !end_mask_attr || !ellipsis_mask_attr)
      return failure();

    ShapedType input_type = op->getOperand(0).getType().cast<ShapedType>();
    if (!input_type.hasStaticShape()) return failure();

    Operation *begin_op = op->getOperand(1).getDefiningOp();
    Operation *end_op = op->getOperand(2).getDefiningOp();
    Operation *strides_op = op->getOperand(3).getDefiningOp();
    if (!begin_op || !end_op || !strides_op) return failure();

    if (!dialect_->IsConstant(begin_op) || !dialect_->IsConstant(end_op) ||
        !dialect_->IsConstant(strides_op))
      return failure();

    ElementsAttr begin_attr = begin_op->getAttrOfType<ElementsAttr>("value");
    ElementsAttr end_attr = end_op->getAttrOfType<ElementsAttr>("value");
    ElementsAttr strides_attr =
        strides_op->getAttrOfType<ElementsAttr>("value");

    const int64_t begin_mask = begin_mask_attr.getInt();
    const int64_t end_mask = end_mask_attr.getInt();
    const int64_t ellipsis_mask = ellipsis_mask_attr.getInt();
    const int64_t num_strides_elements = strides_attr.getNumElements();

    DenseSet<int> expanded_ellipsis_indices;
    int ellipsis_index = -1;

    for (unsigned i = 0; i < input_type.getRank(); ++i) {
      if (ellipsis_mask & 1 << i ||
          (ellipsis_index == -1 && i >= num_strides_elements)) {
        ellipsis_index = i;
      }
      if (ellipsis_index != -1 &&
          input_type.getRank() > num_strides_elements + i - ellipsis_index) {
        expanded_ellipsis_indices.insert(i);
      }
    }

    for (unsigned i = 0; i < input_type.getRank(); ++i) {
      if (expanded_ellipsis_indices.contains(i)) {
        // ellipsis_mask is effective on current dimension.
        continue;
      }

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
          !(end_mask & 1 << j || e == input_type.getShape()[i]) || s != 1) {
        return failure();
      }
    }

    FailureOr<TFOp> identity = ReplaceOpWithIdentity(rewriter, op, 0);
    if (failed(identity)) return failure();
    rewriter.replaceOp(op, (*identity)->getResults());

    return success();
  }
};

// This implementation is mapped with ConstantFolding::SimplifyTile
// in grappler/optimizers/constant_folding.cc
class SimplifyTileOp : public FolderPatternBase<SimplifyTileOp> {
 public:
  explicit SimplifyTileOp(OpPropertyHelper &helper)
      : FolderPatternBase<SimplifyTileOp>("tfg.Tile", helper) {}
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    Operation *multiples_op = op->getOperand(1).getDefiningOp();
    if (!multiples_op || !dialect_->IsConstant(multiples_op)) return failure();

    ElementsAttr multiples_attr =
        multiples_op->getAttrOfType<ElementsAttr>("value");
    if (multiples_attr.getElementType().isInteger(32)) {
      if (llvm::any_of(multiples_attr.getValues<int32_t>(),
                       [](int v) { return v != 1; })) {
        return failure();
      }
    } else {
      if (llvm::any_of(multiples_attr.getValues<int64_t>(),
                       [](int64_t v) { return v != 1; })) {
        return failure();
      }
    }

    FailureOr<TFOp> identity = ReplaceOpWithIdentity(rewriter, op, 0);
    if (failed(identity)) return failure();
    rewriter.replaceOp(op, (*identity)->getResults());

    return success();
  }
};

// This implementation is mapped with ConstantFolding::SimplifyPad
// in grappler/optimizers/constant_folding.cc
template <typename ConcreteType>
class SimplifyPadOpBase : public FolderPatternBase<ConcreteType> {
 protected:
  SimplifyPadOpBase(StringRef op_name, OpPropertyHelper &helper)
      : FolderPatternBase<ConcreteType>(op_name, helper) {}
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    Operation *paddings = op->getOperand(1).getDefiningOp();
    if (!paddings || !this->dialect_->IsConstant(paddings)) return failure();

    ElementsAttr paddings_attr = paddings->getAttrOfType<ElementsAttr>("value");
    if (paddings_attr.getElementType().isInteger(32)) {
      if (llvm::any_of(paddings_attr.getValues<int32_t>(),
                       [](int v) { return v != 0; })) {
        return failure();
      }
    } else {
      if (llvm::any_of(paddings_attr.getValues<int64_t>(),
                       [](int64_t v) { return v != 0; })) {
        return failure();
      }
    }

    FailureOr<TFOp> identity = ReplaceOpWithIdentity(rewriter, op, 0);
    if (failed(identity)) return failure();
    rewriter.replaceOp(op, (*identity)->getResults());

    return success();
  }
};

// This implementation is mapped with ConstantFolding::SimplifyPad
// in grappler/optimizers/constant_folding.cc
class SimplifyPadOp : public SimplifyPadOpBase<SimplifyPadOp> {
 public:
  explicit SimplifyPadOp(OpPropertyHelper &helper)
      : SimplifyPadOpBase("tfg.Pad", helper) {}
};

// This implementation is mapped with ConstantFolding::SimplifyPad
// in grappler/optimizers/constant_folding.cc
class SimplifyPadV2Op : public SimplifyPadOpBase<SimplifyPadV2Op> {
 public:
  explicit SimplifyPadV2Op(OpPropertyHelper &helper)
      : SimplifyPadOpBase("tfg.PadV2", helper) {}
};

// This implementation is mapped with ConstantFolding::SimplifySqueeze
// in grappler/optimizers/constant_folding.cc
class SimplifySqueezeOp : public FolderPatternBase<SimplifySqueezeOp> {
 public:
  explicit SimplifySqueezeOp(OpPropertyHelper &helper)
      : FolderPatternBase<SimplifySqueezeOp>("tfg.Squeeze", helper) {}
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    auto shape_type = op->getOperand(0).getType().cast<ShapedType>();
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
// Rewrite a Pack op with a single non-control input into ExpandDims.
class SimplifyPackOp : public FolderPatternBase<SimplifyPackOp> {
 public:
  explicit SimplifyPackOp(OpPropertyHelper &helper)
      : FolderPatternBase<SimplifyPackOp>("tfg.Pack", helper) {}
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    auto [non_control_operands, control_operands] = TFOp(op).splitOperands();
    if (non_control_operands.size() != 1) return failure();

    // ExpandDims is not supported on DT_VARIANT (see ExpandDimsOp::Compute),
    // and DT_VARIANT tensor protos are converted to opaque tensors. We skip
    // such cases (even though not all opaque tensors are DT_VARIANT tensor
    // protos, e.g. there is DT_RESOURCE).
    // TODO(tlongeri): is there a reason ExpandDims does not support DT_VARIANT?
    if (ShapedType values_type =
            non_control_operands[0].getType().dyn_cast<ShapedType>();
        !values_type || values_type.getElementType().isa<VariantType>())
      return failure();

    // It's unsafe to add a control dependency on the feed node, because it
    // might have been never executed otherwiwise.
    if (non_control_operands[0].isa<BlockArgument>()) return failure();

    IntegerAttr axis = op->getAttrOfType<IntegerAttr>("axis");
    ElementsAttr const_attr = CreateElementsAttrOfTypeValues(
        rewriter.getIntegerType(32), /*shape=*/{},
        ArrayRef<int>(axis ? axis.getInt() : 0));
    // CreateConstantTensorOp cannot fail
    TFOp const_op = *CreateConstantTensorOp(
        rewriter, op->getLoc(), TFOp(op).name(), const_attr.getType(),
        GetControlDependency(rewriter, op->getOperand(0)), const_attr);

    const_op.setName(Twine(TFOp(op).name(), "/_const_axis"));
    if (!TFOp(op).device().empty())
      const_op.setRequestedDevice(TFOp(op).deviceAttr());

    OperationState state(op->getLoc(), "tfg.ExpandDims");
    state.addTypes(op->getResultTypes());

    state.attributes = op->getAttrDictionary();
    state.attributes.erase("axis");
    state.attributes.erase("N");
    state.addAttribute("Tdim", TypeAttr::get(rewriter.getI32Type()));

    state.addOperands({op->getOperand(0), const_op->getResult(0)});
    state.addOperands(control_operands);
    Operation *expand_dims_op = rewriter.create(state);
    rewriter.replaceOp(op, expand_dims_op->getResults());
    return success();
  }
};

// This implementation is mapped with ConstantFolding::MoveConstantsPastEnter
// in grappler/optimizers/constant_folding.cc
template <typename ConcreteType>
class MoveConstantsPastEnterOpBase
    : public PropagationPatternBase<ConcreteType> {
 protected:
  MoveConstantsPastEnterOpBase(StringRef op_name, OpPropertyHelper &helper)
      : PropagationPatternBase<ConcreteType>(op_name, helper) {}
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    auto is_constant_attr = op->getAttrOfType<BoolAttr>("is_constant");
    if (!is_constant_attr || !is_constant_attr.getValue()) return failure();

    Operation *input = op->getOperand(0).getDefiningOp();
    if (!input || !this->dialect_->IsConstant(input)) return failure();

    // Find non-constant nodes that consume the outputs of Enter.
    if (op->getResults()[0].use_empty()) return failure();

    FailureOr<TFOp> cloned_const_op = CreateConstantTensorOp(
        rewriter, op->getLoc(), TFOp(op).name(), *(input->result_type_begin()),
        TFOp(op).controlRet(), input->getAttr("value"), input->getAttrs());
    if (failed(cloned_const_op)) return failure();

    (*cloned_const_op).setName(Twine(TFOp(op).name(), "/_enter"));
    if (!TFOp(op).device().empty())
      (*cloned_const_op).setRequestedDevice(TFOp(op).deviceAttr());

    rewriter.startRootUpdate(op);
    op->getResults()[0].replaceAllUsesWith((*cloned_const_op)->getResults()[0]);
    rewriter.finalizeRootUpdate(op);
    return success();
  }
};

// This implementation is mapped with ConstantFolding::MoveConstantsPastEnter
// in grappler/optimizers/constant_folding.cc
class MoveConstantsPastEnterOp
    : public MoveConstantsPastEnterOpBase<MoveConstantsPastEnterOp> {
 public:
  explicit MoveConstantsPastEnterOp(OpPropertyHelper &helper)
      : MoveConstantsPastEnterOpBase("tfg.Enter", helper) {}
};

// This implementation is mapped with ConstantFolding::MoveConstantsPastEnter
// in grappler/optimizers/constant_folding.cc
class MoveConstantsPastRefEnterOp
    : public MoveConstantsPastEnterOpBase<MoveConstantsPastRefEnterOp> {
 public:
  explicit MoveConstantsPastRefEnterOp(OpPropertyHelper &helper)
      : MoveConstantsPastEnterOpBase("tfg.RefEnter", helper) {}
};

// This implementation is mapped with ConstantFolding::SimplifySwitch
// in grappler/optimizers/constant_folding.cc.
// In addition to the Grappler functionality, we remove duplicate anchors from
// the switch.
class SimplifySwitchOp : public PropagationPatternBase<SimplifySwitchOp> {
 public:
  explicit SimplifySwitchOp(OpPropertyHelper &helper)
      : PropagationPatternBase<SimplifySwitchOp>("tfg.Switch", helper),
        zero_dim_i1_tensor_type_(RankedTensorType::get(
            {}, IntegerType::get(helper.getDialect()->getContext(), 1))) {}
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    // Currently, there is no infallible protection against reapplications of
    // the pattern resulting in constant nodes with duplicate names (Grappler
    // handled this by checking names globally).
    // We could add a suffix to the node name on each application, but then we
    // could not apply the pattern on fetch/preserved nodes.
    // Removing duplicate anchors prevents the problem from manifesting in
    // certain situations (namely, when the common subgraph elimination pass
    // merges two switch ops on which the pattern had been already applied).

    bool modified = false;

    auto remove_duplicate_anchors = [&](OpResult result) {
      auto anchors = make_filter_range(result.getUsers(), [&](Operation *op) {
        return IsControlAnchor(op, dialect_);
      });

      for (Operation *anchor : make_early_inc_range(anchors)) {
        if (anchor == *anchors.begin()) continue;
        rewriter.replaceOp(anchor, (*anchors.begin())->getResults());
        modified = true;
      }
    };

    auto simplify_result = [&](OpResult result, const bool const_value,
                               const StringRef name_suffix) {
      if (result.use_empty() ||
          (result.hasOneUse() &&
           IsControlAnchor(*result.getUsers().begin(), dialect_)))
        return;

      FailureOr<TFOp> failure_or_const_op = CreateConstantTensorOp(
          rewriter, op->getLoc(), TFOp(op).name(), result.getType(),
          std::nullopt,
          DenseElementsAttr::get(zero_dim_i1_tensor_type_, const_value));
      if (failed(failure_or_const_op)) return;
      TFOp const_op = *failure_or_const_op;
      const_op.setName(TFOp(op).name() + name_suffix);
      if (StringAttr device_attr = TFOp(op).deviceAttr())
        const_op.setRequestedDevice(device_attr);

      // May create a new op - must be careful to not fail out after.
      TFOp anchor = GetControlAnchorForSwitchResult(rewriter, result, dialect_);
      const_op->insertOperands(0, anchor.controlRet());

      // Note that we can't use replaceAllUsesWith here because we don't want to
      // replace the user of control identity.
      for (OpOperand &user : llvm::make_early_inc_range(result.getUses())) {
        if (user.getOwner() == &(*anchor)) continue;

        rewriter.startRootUpdate(user.getOwner());
        user.set(const_op->getResult(0));
        rewriter.finalizeRootUpdate(user.getOwner());
      }
      modified = true;
    };

    remove_duplicate_anchors(op->getResult(0));
    remove_duplicate_anchors(op->getResult(1));

    if (op->getOperand(0) == op->getOperand(1)) {
      simplify_result(op->getResult(0), false, "/_const_false");
      simplify_result(op->getResult(1), true, "/_const_true");
    }

    return success(modified);
  }

  RankedTensorType zero_dim_i1_tensor_type_;
};

// This implementation is mapped with ConstantFolding::SimplifyReduction
// in grappler/optimizers/constant_folding.cc
class SimplifyReductionOp : public FolderPatternBase<SimplifyReductionOp> {
 public:
  explicit SimplifyReductionOp(OpPropertyHelper &helper)
      : FolderPatternBase<SimplifyReductionOp>(MatchAnyOpTypeTag(), helper) {}
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (!dialect_->IsReduction(op)) return failure();

    Operation *reduction_indices = op->getOperand(1).getDefiningOp();
    if (!reduction_indices) return failure();

    ShapedType indices_type = *(reduction_indices->result_type_begin());
    if (indices_type.hasStaticShape() && indices_type.getNumElements() == 0) {
      Operation *identity_op = ReplaceReductionWithIdentity(rewriter, op);
      if (!identity_op) return failure();

      rewriter.replaceOp(op, identity_op->getResults());
      return success();
    }

    // Check `IsReductionCandidateForSimplification`
    auto input_type = op->getOperand(0).getType().cast<ShapedType>();
    auto op_type = (*op->result_type_begin()).cast<ShapedType>();
    if (!input_type.hasStaticShape() || !op_type.hasStaticShape())
      return failure();

    bool is_single_element_op =
        (input_type.getNumElements() == 1) &&
        (op_type.hasStaticShape() && op_type.getNumElements() == 1);

    bool keep_dims = false;
    if (auto attr = op->getAttrOfType<BoolAttr>("keep_dims")) {
      keep_dims = attr.getValue();
    }
    bool simplifiable_to_reshape =
        is_single_element_op && !keep_dims && op->hasAttr("T");

    bool simplifiable_to_identity = keep_dims;
    // In grappler, they call EvaluateNode() to try to get the constant value of
    // reduction indices. But if it is a constant, then the EvaluationConstant
    // will have folded it. So we don't need to evalute the node here.
    if (dialect_->IsConstant(reduction_indices)) {
      ElementsAttr reduction_indices_attr =
          reduction_indices->getAttrOfType<ElementsAttr>("value");

      if (reduction_indices_attr.getElementType().isInteger(32)) {
        for (int v : reduction_indices_attr.getValues<int32_t>()) {
          if (v < 0) v += input_type.getRank();
          if (v < 0 || v >= input_type.getRank() ||
              input_type.getShape()[v] != 1)
            simplifiable_to_identity = false;
        }
      } else {
        for (int64_t v : reduction_indices_attr.getValues<int64_t>()) {
          if (v < 0) v += input_type.getRank();
          if (v < 0 || v >= input_type.getRank() ||
              input_type.getShape()[v] != 1)
            simplifiable_to_identity = false;
        }
      }
    }

    if (simplifiable_to_reshape) {
      Operation *reshape_op =
          ReplaceReductionWithReshape(rewriter, op, reduction_indices);
      if (!reshape_op) return failure();

      rewriter.replaceOp(op, reshape_op->getResults());
    } else if (simplifiable_to_identity) {
      Operation *identity_op = ReplaceReductionWithIdentity(rewriter, op);
      if (!identity_op) return failure();

      rewriter.replaceOp(op, identity_op->getResults());
    } else {
      return failure();
    }

    return success();
  }

 private:
  Operation *ReplaceReductionWithReshape(OpBuilder &builder, Operation *op,
                                         Operation *reduction_indices) const {
    const int new_num_dimensions =
        (*op->result_type_begin()).cast<ShapedType>().getRank();
    SmallVector<int64_t> elements(new_num_dimensions);
    std::iota(elements.begin(), elements.end(), 1);
    ElementsAttr const_attr = CreateElementsAttrOfTypeValues(
        builder.getIntegerType(32), {new_num_dimensions},
        llvm::ArrayRef(elements));
    FailureOr<TFOp> const_op = CreateConstantTensorOp(
        builder, op->getLoc(), TFOp(op).name(),
        *(reduction_indices->result_type_begin()),
        TFOp(reduction_indices).controlRet(), const_attr);
    if (failed(const_op)) return nullptr;

    (*const_op).setName(Twine(TFOp(op).name(), "/_shape_const"));
    if (!TFOp(op).device().empty())
      (*const_op).setRequestedDevice(TFOp(op).deviceAttr());

    OperationState state(op->getLoc(), "tfg.Reshape");
    state.attributes = op->getAttrDictionary();
    state.attributes.erase("keep_dims");
    state.attributes.erase("Tidx");
    state.addAttribute("Tshape", TypeAttr::get(builder.getI32Type()));

    state.addOperands(op->getOperands());
    state.operands[1] = (*const_op)->getResult(0);
    state.addTypes(op->getResultTypes());

    Operation *reshape_op = builder.create(state);
    TFOp(reshape_op).setName(TFOp(op).nameAttr());
    if (!TFOp(op).device().empty())
      TFOp(reshape_op).setRequestedDevice(TFOp(op).deviceAttr());
    return reshape_op;
  }

  Operation *ReplaceReductionWithIdentity(OpBuilder &builder,
                                          Operation *op) const {
    OperationState state(op->getLoc(), "tfg.Identity");
    Type t_attr_type;
    if (auto T_attr = op->getAttrOfType<TypeAttr>("T"))
      t_attr_type = T_attr.getValue();
    else if (dialect_->IsAny(op) || dialect_->IsAll(op))
      t_attr_type = builder.getI1Type();
    else
      return nullptr;
    state.attributes = op->getAttrDictionary();
    util::EraseRegularNodeAttributes(state.attributes);
    state.addAttribute("T", TypeAttr::get(t_attr_type));
    state.addTypes(op->getResultTypes());
    state.addOperands(
        {op->getOperand(0), GetControlDependency(builder, op->getOperand(1))});

    Operation *identity_op = builder.create(state);
    TFOp(identity_op).setName(TFOp(op).nameAttr());
    if (!TFOp(op).device().empty())
      TFOp(identity_op).setRequestedDevice(TFOp(op).deviceAttr());
    return identity_op;
  }
};

// This implementation is mapped with ConstantFolding::SimplifyReshapeOp
// in grappler/optimizers/constant_folding.cc
class SimplifyReshapeOp : public FolderPatternBase<SimplifyReshapeOp> {
 public:
  explicit SimplifyReshapeOp(OpPropertyHelper &helper)
      : FolderPatternBase<SimplifyReshapeOp>(MatchAnyOpTypeTag(), helper) {}
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (!dialect_->IsReshape(op) || !op->hasAttr("T")) return failure();

    auto input_shape = op->getOperand(0).getType().cast<ShapedType>();
    if (!input_shape.hasStaticShape()) return failure();

    Operation *shape_op = op->getOperand(1).getDefiningOp();
    if (!shape_op || !dialect_->IsConstant(shape_op)) return failure();

    auto shape_attr = shape_op->getAttrOfType<ElementsAttr>("value");
    // TODO(tlongeri): only reason for SmallVector instead of range directly is
    // that llvm::zip implementation requires copy assignment (it shouldn't)
    SmallVector<APInt> new_shape(shape_attr.getValues<APInt>());

    if (input_shape.getRank() != new_shape.size()) return failure();
    for (const auto &it : llvm::zip(input_shape.getShape(), new_shape)) {
      int64_t dim_0 = std::get<0>(it);
      int64_t dim_1 = std::get<1>(it).getSExtValue();
      if (dim_0 >= 0 && dim_1 >= 0 && dim_0 != dim_1) return failure();
    }

    OperationState state(op->getLoc(), "tfg.Identity");
    state.addTypes(op->getResultTypes());
    state.addOperands(
        {op->getOperand(0), GetControlDependency(rewriter, op->getOperand(1))});
    state.addOperands(TFOp(op).getControlOperands());

    state.attributes = op->getAttrDictionary();
    util::EraseRegularNodeAttributes(state.attributes);
    state.addAttribute("T", op->getAttrOfType<TypeAttr>("T"));

    Operation *identity_op = rewriter.create(state);
    TFOp(identity_op).setName(TFOp(op).nameAttr());
    if (!TFOp(op).device().empty())
      TFOp(identity_op).setRequestedDevice(TFOp(op).deviceAttr());
    rewriter.replaceOp(op, identity_op->getResults());

    return success();
  }
};

// This implementation is mapped with
// ConstantFolding::SimplifyArithmeticOperations in
// grappler/optimizers/constant_folding.cc
class SimplifyArithmeticOp
    : public ConstantPatternBase<SimplifyArithmeticOp, FolderTrait,
                                 PropagationTrait> {
 public:
  explicit SimplifyArithmeticOp(OpPropertyHelper &helper)
      : ConstantPatternBase(MatchAnyOpTypeTag(), helper) {}
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    const bool is_mul = dialect_->IsAnyMul(op) || dialect_->IsLogicalAnd(op);
    const bool is_matmul = dialect_->IsAnyMatMul(op);
    const bool is_add = dialect_->IsAdd(op) || dialect_->IsBiasAdd(op) ||
                        dialect_->IsLogicalOr(op);
    const bool is_sub = dialect_->IsSub(op);
    const bool is_any_div = dialect_->IsAnyDiv(op) && !dialect_->IsFloorDiv(op);

    if (!is_mul && !is_matmul && !is_add && !is_sub && !is_any_div)
      return failure();

    Operation *x = op->getOperand(0).getDefiningOp();
    Operation *y = op->getOperand(1).getDefiningOp();
    if (!x || !y) return failure();

    ShapedType op_type = (*op->result_type_begin()).cast<ShapedType>();
    ShapedType x_type = (*x->result_type_begin()).cast<ShapedType>();
    ShapedType y_type = (*y->result_type_begin()).cast<ShapedType>();

    const bool y_matches_output_shape = op_type.hasStaticShape() &&
                                        y_type.hasStaticShape() &&
                                        op_type == y_type;
    const bool x_matches_output_shape = op_type.hasStaticShape() &&
                                        x_type.hasStaticShape() &&
                                        op_type == x_type;

    const bool x_is_zero = helper_.IsZeros(x);
    const bool x_is_one = x_is_zero ? false : helper_.IsOnes(x);

    // TODO(chiahungduan): Check if the optimizations has been applied.

    if ((is_mul && x_is_one) || (is_add && x_is_zero)) {
      // 1 * y = y or 0 + y = y.
      if (y_matches_output_shape) {
        FailureOr<TFOp> snapshot_op = ReplaceOpWithSnapshot(rewriter, op, 1);
        if (failed(snapshot_op)) return failure();
        rewriter.replaceOp(op, (*snapshot_op)->getResults());
        return success();
      } else if (x_matches_output_shape) {
        FailureOr<TFOp> broadcast_to_op =
            ReplaceOpWithBroadcastTo(rewriter, op, 1);
        rewriter.replaceOp(op, (*broadcast_to_op)->getResults());
        return success();
      }
      return failure();
    }

    if (y_matches_output_shape && (is_sub && x_is_zero)) {
      // Replace 0 - y with Neg(y).
      OperationState state(op->getLoc(), "tfg.Neg");
      state.addOperands({op->getOperand(1),
                         GetControlDependency(rewriter, op->getOperand(0))});
      state.addOperands(TFOp(op).getControlOperands());
      state.attributes = op->getAttrDictionary();
      state.addTypes(op->getResultTypes());
      Operation *neg = rewriter.create(state);
      rewriter.replaceOp(op, neg->getResults());
      return success();
    }

    // Replace 1 / y with Reciprocal op.
    if (y_matches_output_shape && is_any_div && x_is_one) {
      TypeAttr type_attr = op->getAttrOfType<TypeAttr>("T");
      if (!type_attr) return failure();

      if (type_attr.getValue().isa<FloatType>() ||
          type_attr.getValue().isa<ComplexType>()) {
        OperationState state(op->getLoc(), "tfg.Reciprocal");
        state.addOperands({op->getOperand(1),
                           GetControlDependency(rewriter, op->getOperand(0))});
        state.addOperands(TFOp(op).getControlOperands());
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
        FailureOr<TFOp> snapshot_op = ReplaceOpWithSnapshot(rewriter, op, 0);
        if (failed(snapshot_op)) return failure();
        rewriter.replaceOp(op, (*snapshot_op)->getResults());
        return success();
      } else if (y_matches_output_shape) {
        FailureOr<TFOp> broadcast_to_op =
            ReplaceOpWithBroadcastTo(rewriter, op, 0);
        if (failed(broadcast_to_op)) return failure();
        rewriter.replaceOp(op, (*broadcast_to_op)->getResults());
        return success();
      }
      return failure();
    }

    // x OR true = true OR y = true.
    if (op_type.hasStaticShape() && dialect_->IsLogicalOr(op) &&
        (y_is_one || x_is_one)) {
      FailureOr<TFOp> const_op = ReplaceOpWithConstant(rewriter, op, 1);
      if (failed(const_op)) return failure();
      rewriter.replaceOp(op, (*const_op)->getResults());
      return success();
    }

    // TFG optimizer doesn't support aggrasive mode.
    const bool is_aggressive = false;
    // Note that this is always false because of `is_aggressive`. Keep it in
    // this form to alleviate the effort of comparing the logic with the same
    // logic in grappler.
    bool optimize_zeros_divided_by_y = is_any_div && x_is_zero && is_aggressive;
    if ((x_is_zero || y_is_zero) &&
        (is_mul || is_matmul || optimize_zeros_divided_by_y)) {
      if (op_type.hasStaticShape()) {
        bool is_quantized = dialect_->IsQuantizedMatMul(op);
        if (is_quantized) {
          // TODO(chiahungduan): AddQuantizedMatMulMinMaxOutConstNodes
          return failure();
        }

        FailureOr<TFOp> const_op = ReplaceOpWithConstant(rewriter, op, 0);
        if (failed(const_op)) return failure();

        rewriter.replaceOp(op, (*const_op)->getResults());
        return success();
      }

      if ((is_mul || is_any_div) && x_is_zero) {
        if (x_matches_output_shape) {
          FailureOr<TFOp> identity = ReplaceOpWithIdentity(rewriter, op, 0);
          if (failed(identity)) return failure();
          rewriter.replaceOp(op, (*identity)->getResults());
          return success();
        } else if (y_matches_output_shape) {
          FailureOr<TFOp> broadcast_to_op =
              ReplaceOpWithBroadcastTo(rewriter, op, 0);
          if (failed(broadcast_to_op)) return failure();
          rewriter.replaceOp(op, (*broadcast_to_op)->getResults());
          return success();
        }
      } else if (is_mul && y_is_zero) {
        if (y_matches_output_shape) {
          FailureOr<TFOp> identity = ReplaceOpWithIdentity(rewriter, op, 0);
          if (failed(identity)) return failure();
          rewriter.replaceOp(op, (*identity)->getResults());
          return success();
        } else if (x_matches_output_shape) {
          FailureOr<TFOp> broadcast_to_op =
              ReplaceOpWithBroadcastTo(rewriter, op, 1);
          if (failed(broadcast_to_op)) return failure();
          rewriter.replaceOp(op, (*broadcast_to_op)->getResults());
          return success();
        }
      }
    }

    return failure();
  }
};

// This implementation is mapped with ConstantFolding::ReduceDivToReciprocalMul
// in grappler/optimizers/constant_folding.cc
class ReduceDivToReciprocalMul
    : public FolderPatternBase<ReduceDivToReciprocalMul> {
 public:
  explicit ReduceDivToReciprocalMul(OpPropertyHelper &helper)
      : FolderPatternBase<ReduceDivToReciprocalMul>(MatchAnyOpTypeTag(),
                                                    helper) {}
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    // Strength reduce floating point division by a constant Div(x, const) to
    // multiplication by the reciprocal Mul(x, Reciprocal(const)). This in turn
    // will be constant folded to Mul(x, 1.0/const).
    if (!dialect_->IsDiv(op) && !dialect_->IsRealDiv(op) &&
        !dialect_->IsXdivy(op)) {
      return failure();
    }

    Operation *y = op->getOperand(1).getDefiningOp();
    if (!y || !dialect_->IsConstant(y)) return failure();

    TypeAttr type_attr = op->getAttrOfType<TypeAttr>("T");
    if (!type_attr) return failure();

    // Skip integer division.
    if (dialect_->IsDiv(op) && !(type_attr.getValue().isa<FloatType>() ||
                                 type_attr.getValue().isa<ComplexType>())) {
      return failure();
    }

    OperationState state(op->getLoc(), "tfg.Reciprocal");
    state.addOperands(y->getResult(0));
    state.addTypes({*(y->result_type_begin()), ControlType::get(getContext())});
    state.addAttribute("T", type_attr);
    TFOp reciprocal_op = rewriter.create(state);
    reciprocal_op.setName(Twine(TFOp(op).name(), "/") +
                          Twine(TFOp(y).name(), "/_recip"));
    if (!TFOp(op).device().empty())
      reciprocal_op.setRequestedDevice(TFOp(op).deviceAttr());

    StringRef new_op_name = dialect_->IsXdivy(op) ? "tfg.MulNoNan" : "tfg.Mul";
    OperationState new_op_state(op->getLoc(), new_op_name);

    if (dialect_->IsXdivy(op)) {
      new_op_state.addOperands(
          {reciprocal_op->getResult(0), op->getOperand(0)});
    } else {
      new_op_state.addOperands(
          {op->getOperand(0), reciprocal_op->getResult(0)});
    }
    new_op_state.addOperands(TFOp(op).getControlOperands());

    new_op_state.attributes = op->getAttrDictionary();
    new_op_state.addTypes(op->getResultTypes());

    Operation *new_op = rewriter.create(new_op_state);
    rewriter.replaceOp(op, new_op->getResults());

    return success();
  }
};

namespace {
template <typename ConcreteType>
using Base = ConstantPatternBase<ConcreteType, FolderTrait, PropagationTrait>;

template <typename ConcreteType>
class ConstantPushDownBase : public Base<ConcreteType> {
 protected:
  using Base<ConcreteType>::Base;

  bool IsOperandsSafeToMove(Operation *op_child, Operation *const_child) const {
    // Don't rewrite the tree if it might create cycles.
    // TODO(chiahungduan): Remove the control dependency which may create
    // cycles.
    if (llvm::any_of(
            TFOp(const_child).getControlOperands(),
            [op_child](Value v) { return v.getDefiningOp() == op_child; })) {
      return false;
    }

    // Move operands may change the result shapes, only do it when there's one
    // user for each of non control return values.
    if (llvm::any_of(op_child->getResults().drop_back(),
                     [](Value v) { return !v.hasOneUse(); })) {
      return false;
    }
    return true;
  }
};
}  // namespace

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
class ConstantPushDown : public ConstantPushDownBase<ConstantPushDown> {
 public:
  explicit ConstantPushDown(OpPropertyHelper &helper)
      : ConstantPushDownBase(MatchAnyOpTypeTag(), helper) {}
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    // Get parent op type.
    const bool is_add = dialect_->IsAdd(op);
    const bool is_mul = dialect_->IsMul(op);
    const bool is_sub = dialect_->IsSub(op);
    const bool is_div = dialect_->IsDiv(op);
    if (!(is_add || is_sub || is_mul || is_div)) return failure();
    const bool is_symmetric = is_add || is_mul;

    Operation *child_op = op->getOperand(0).getDefiningOp();
    Operation *const_op = op->getOperand(1).getDefiningOp();
    if (!child_op || !const_op) return failure();

    // Don't move nodes across devices.
    if (TFOp(op).deviceAttr() != TFOp(child_op).deviceAttr() ||
        TFOp(op).deviceAttr() != TFOp(const_op).deviceAttr()) {
      return failure();
    }

    const bool left_child_is_const = dialect_->IsConstant(child_op);

    // One of the child ops has to be constant.
    if (!dialect_->IsConstant(const_op)) std::swap(child_op, const_op);
    if (!dialect_->IsConstant(const_op)) return failure();
    if (helper_.ShouldPreserveOp(child_op)) return failure();

    if (!IsOperandsSafeToMove(child_op, const_op)) return failure();

    // Get child op type.
    const bool is_child_add = dialect_->IsAdd(child_op);
    const bool is_child_mul = dialect_->IsMul(child_op);
    const bool is_child_sub = dialect_->IsSub(child_op);
    const bool is_child_div = dialect_->IsDiv(child_op);
    const bool is_add_sub =
        (is_add || is_sub) && (is_child_add || is_child_sub);
    const bool is_mul_div =
        (is_mul || is_div) && (is_child_mul || is_child_div);
    if (!is_add_sub && !is_mul_div) return failure();

    const bool is_child_symmetric = is_child_add || is_child_mul;

    TypeAttr t_attr = op->getAttrOfType<TypeAttr>("T");
    assert(t_attr == child_op->getAttrOfType<TypeAttr>("T"));
    if (!t_attr) return failure();

    // Do not rewrite expressions of integer types with division because:
    // - They use integer division.
    // - There may be overflow. (a * b) / c != (a / c) * b if (a * b) overflows,
    // even if divisions have no remainder.
    if (t_attr.getValue().isIntOrIndex() && (is_div || is_child_div))
      return failure();

    Operation *left_leaf_op = child_op->getOperand(0).getDefiningOp();
    Operation *right_leaf_op = child_op->getOperand(1).getDefiningOp();
    // TODO(tlongeri): Is this check really necessary? Why not allow block
    // arguments?
    if (!left_leaf_op || !right_leaf_op) return failure();

    // Don't move nodes across devices.
    if (TFOp(op).deviceAttr() != TFOp(left_leaf_op).deviceAttr() ||
        TFOp(op).deviceAttr() != TFOp(right_leaf_op).deviceAttr()) {
      return failure();
    }

    const bool left_leaf_is_const = dialect_->IsConstant(left_leaf_op);
    if (left_leaf_is_const && dialect_->IsConstant(right_leaf_op))
      return failure();
    // X is never Const. Y may be Const.
    Value x_value = child_op->getOperand(left_leaf_is_const ? 1 : 0);
    Value y_value = child_op->getOperand(left_leaf_is_const ? 0 : 1);
    Operation *y_node = left_leaf_is_const ? left_leaf_op : right_leaf_op;

    if (!dialect_->IsConstant(y_node)) {
      // If we know the shapes of the nodes being swapped, make sure we don't
      // push down a larger node and create more work by broadcasting earlier
      // in the expressions tree.
      // Dimensions of X must be smaller than or equal than those of C.
      // This also avoids having to increase the size of the child op's result
      // to match the broadcast with a bigger operand.
      auto c_shape = const_op->getResult(0).getType().cast<ShapedType>();
      auto x_shape = x_value.getType().cast<ShapedType>();

      if (c_shape.hasStaticShape() && x_shape.hasStaticShape() &&
          c_shape.getNumElements() > x_shape.getNumElements()) {
        return failure();
      }
      if (c_shape.hasRank() && x_shape.hasRank() && c_shape.getRank() > 0) {
        for (auto it : llvm::zip(c_shape.getShape(), x_shape.getShape())) {
          int c_dim = std::get<0>(it);
          int x_dim = std::get<1>(it);
          if (x_dim >= 0 && c_dim > x_dim) return failure();
        }
      }
    }

    if (is_symmetric && is_child_symmetric) {
      // Easy case (only commutative ops). We always write this as one of
      //   +
      //  / \
      // X   +
      //    / \
      //   C   Y
      rewriter.startRootUpdate(op);
      op->setOperand(0, x_value);
      op->setOperand(1, child_op->getResult(0));
      rewriter.finalizeRootUpdate(op);
      rewriter.startRootUpdate(child_op);
      child_op->setOperand(0, const_op->getResult(0));
      child_op->setOperand(1, y_value);
      rewriter.finalizeRootUpdate(child_op);
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
      state.addOperands({x_value, child_op->getResult(0)});
      if (neg_x) std::swap(state.operands[0], state.operands[1]);
      state.addOperands(TFOp(op).getControlOperands());
      state.attributes = op->getAttrDictionary();
      state.addTypes(op->getResultTypes());
      Operation *new_op = rewriter.create(state);
      rewriter.replaceOp(op, new_op->getResults());

      StringRef child_name = neg_c != neg_y ? nonsymmetric_op : symmetric_op;
      OperationState new_child_state(child_op->getLoc(), child_name);
      new_child_state.addOperands({const_op->getResult(0), y_value});
      if (neg_c)
        std::swap(new_child_state.operands[0], new_child_state.operands[1]);
      new_child_state.addOperands(TFOp(child_op).getControlOperands());
      new_child_state.attributes = child_op->getAttrDictionary();
      new_child_state.addTypes(child_op->getResultTypes());
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
class PartialConstPropThroughIdentityN
    : public PropagationPatternBase<PartialConstPropThroughIdentityN> {
 public:
  explicit PartialConstPropThroughIdentityN(OpPropertyHelper &helper)
      : PropagationPatternBase<PartialConstPropThroughIdentityN>(
            MatchAnyOpTypeTag(), helper) {}
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    // In grappler's constant folding, it propagates the values from IdentityN.
    // At here, we check the operand which is defined by Identity/IdentityN.

    SmallVector<Value> control_operands;
    for (OpOperand &operand : op->getOpOperands()) {
      Value v = operand.get();
      if (v.getType().isa<ControlType>()) break;

      Operation *v_op = v.getDefiningOp();
      if (!v_op || !dialect_->IsIdentityN(v_op) ||
          dialect_->IsIdentityNSingleInput(v_op)) {
        continue;
      }

      int res_index = v.cast<OpResult>().getResultNumber();
      Value value_to_forward = v_op->getOperand(res_index);
      if (!value_to_forward.getDefiningOp() ||
          !dialect_->IsConstant(value_to_forward.getDefiningOp())) {
        continue;
      }

      rewriter.startRootUpdate(op);
      operand.set(value_to_forward);
      rewriter.finalizeRootUpdate(op);

      // Add the control dependency to the Identity/IdentityN. Note that it's
      // possible to have multiple operands defined by the same
      // Identity/IdentityN. Given the number is small and this propagation is
      // usually done on an operation one time, do a linear scan before
      // insertion.
      Value control = TFOp(v_op).controlRet();
      if (!llvm::is_contained(control_operands, control))
        control_operands.push_back(control);
    }

    // No new control operands implies that we didn't find constants that can be
    // propagated through Identity/IdentityN.
    if (control_operands.empty()) return failure();

    OperationState state(op->getLoc(), op->getName());
    state.attributes = op->getAttrDictionary();
    state.addOperands(op->getOperands());
    // Append the newly added control operands from Identity/IdentityN.
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
class PartialAssocOpConstFolding
    : public FolderPatternBase<PartialAssocOpConstFolding> {
 public:
  explicit PartialAssocOpConstFolding(OpPropertyHelper &helper)
      : FolderPatternBase<PartialAssocOpConstFolding>(MatchAnyOpTypeTag(),
                                                      helper) {}
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

    auto [non_control_operands, control_operands] = TFOp(op).splitOperands();
    int non_control_inputs_size = non_control_operands.size();
    if (non_control_inputs_size <= 2) return failure();

    if (llvm::any_of(non_control_operands, [](Value v) {
          Operation *v_op = v.getDefiningOp();
          return v_op &&
                 TFOp(v_op).name().rfind("_partial_split_") != StringRef::npos;
        })) {
      return failure();
    }

    for (Value operand : non_control_operands) {
      Operation *may_const_op = operand.getDefiningOp();
      if (may_const_op && dialect_->IsConstant(may_const_op))
        const_inputs.push_back(operand);
      else
        non_const_inputs.push_back(operand);
    }

    if (const_inputs.size() == non_control_inputs_size &&
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

    if (const_inputs.size() <= 1) return failure();

    OperationState state(op->getLoc(), "tfg.AddN");
    state.addOperands(const_inputs);
    state.addTypes(op->getResultTypes());
    state.attributes = op->getAttrDictionary();
    state.attributes.erase("shape");
    state.attributes.set("N", IntegerAttr::get(rewriter.getIntegerType(32),
                                               const_inputs.size()));
    Operation *add_n = rewriter.create(state);
    TFOp(add_n).setName(Twine(TFOp(op).name(), "/_partial_split_") +
                        std::to_string(const_inputs.size()));
    // Op inherits all the attrs of op, don't need to update the device attr.

    OperationState new_op_state(op->getLoc(), op->getName());
    // Note that in grappler, it puts the AddOp at the position of the first
    // const operand. Here we always put the AddOp at begin.
    new_op_state.addOperands(add_n->getResult(0));
    new_op_state.addOperands(non_const_inputs);
    new_op_state.addOperands(control_operands);
    new_op_state.addTypes(op->getResultTypes());
    new_op_state.attributes = op->getAttrDictionary();
    new_op_state.attributes.set("N",
                                IntegerAttr::get(rewriter.getIntegerType(32),
                                                 non_const_inputs.size() + 1));

    Operation *new_op = rewriter.create(new_op_state);
    rewriter.replaceOp(op, new_op->getResults());

    return success();
  }
};

// This implementation is mapped with ConstantFolding::MergeConcat in
// grappler/optimizers/constant_folding.cc
class MergeConcatOp : public FolderPatternBase<MergeConcatOp> {
 public:
  explicit MergeConcatOp(OpPropertyHelper &helper)
      : FolderPatternBase<MergeConcatOp>("tfg.ConcatV2", helper) {}
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (helper_.ShouldPreserveOp(op)) return failure();

    auto getAxis = [&](Operation *axis_op) {
      ElementsAttr axis_attr = axis_op->getAttrOfType<ElementsAttr>("value");
      return axis_attr.getElementType().isInteger(64)
                 ? static_cast<int>(axis_attr.getSplatValue<int64_t>())
                 : axis_attr.getSplatValue<int>();
    };

    auto [non_control_operands, control_operands] = TFOp(op).splitOperands();
    Operation *axis_op = non_control_operands.back().getDefiningOp();
    if (!axis_op || !dialect_->IsConstant(axis_op)) return failure();
    int axis = getAxis(axis_op);

    // In grappler, it checks the first user of the ConcatV2 to see if it's also
    // a ConcatV2. At here, we check the user's operand. Another difference is
    // that grappler only checks the first user and we check all the operands.
    Operation *concat_operand = nullptr;
    for (Value operand : non_control_operands) {
      Operation *defining_op = operand.getDefiningOp();
      if (defining_op && dialect_->IsConcatV2(defining_op)) {
        concat_operand = defining_op;
        break;
      }
    }
    if (!concat_operand) return failure();

    auto [concat_non_control_operands, concat_control_operands] =
        TFOp(concat_operand).splitOperands();
    Operation *concat_operand_axis_op =
        concat_non_control_operands.back().getDefiningOp();
    if (!concat_operand_axis_op ||
        !dialect_->IsConstant(concat_operand_axis_op)) {
      return failure();
    }
    if (axis != getAxis(concat_operand_axis_op)) return failure();

    // If all inputs are constant, don't merge and let EvaluateConstant take
    // case of it.
    if (llvm::all_of(concat_non_control_operands.drop_back(), [&](Value v) {
          return v.getDefiningOp() && dialect_->IsConstant(v.getDefiningOp());
        })) {
      return failure();
    }

    // Make a pass over the parent inputs to see if any of them have explicit
    // device() fields set, and if different inputs are on different tasks.  If
    // so, this concat of concats may have been carefully constructed to be a
    // two-stage concat, and we don't want to undo that here.
    std::string task, device;
    StringRef unique_input_tasks;
    for (Value v : non_control_operands) {
      Operation *v_op = v.getDefiningOp();
      if (!v_op || v_op == axis_op) continue;
      StringRef op_device = TFOp(v_op).device();
      if (!op_device.empty() && tensorflow::DeviceNameUtils::SplitDeviceName(
                                    op_device.str(), &task, &device)) {
        if (unique_input_tasks.empty())
          unique_input_tasks = task;
        else if (unique_input_tasks != task)
          return failure();
      }
    }

    OperationState state(op->getLoc(), "tfg.ConcatV2");
    for (Value operand : non_control_operands) {
      if (operand == concat_operand->getResult(0)) {
        // Inline the non-control operands of concat_operand.
        state.addOperands(ValueRange(concat_non_control_operands.drop_back()));
      } else {
        state.addOperands(operand);
      }
    }
    // Copy the control operands.
    state.addOperands(control_operands);
    state.addOperands(concat_control_operands);
    state.addTypes(op->getResultTypes());
    state.attributes = op->getAttrDictionary();
    state.attributes.set("N", IntegerAttr::get(rewriter.getIntegerType(32),
                                               state.operands.size() - 1));
    Operation *concat_op = rewriter.create(state);
    rewriter.replaceOp(op, concat_op->getResults());

    return success();
  }
};

// This implementation is mapped with ConstantFolding::MulConvPushDown
// in grappler/optimizers/constant_folding.cc
class MulConvPushDown : public ConstantPatternBase<MulConvPushDown, FolderTrait,
                                                   PropagationTrait> {
 public:
  explicit MulConvPushDown(OpPropertyHelper &helper)
      : ConstantPatternBase(MatchAnyOpTypeTag(), helper) {}
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
    if (!dialect_->IsAnyMul(op)) return failure();

    Operation *mul_left_child = op->getOperand(0).getDefiningOp();
    Operation *mul_right_child = op->getOperand(1).getDefiningOp();
    if (!mul_left_child || !mul_right_child) return failure();

    const bool left_child_is_constant = dialect_->IsConstant(mul_left_child);
    const bool right_child_is_constant = dialect_->IsConstant(mul_right_child);
    // One child must be constant, and the second must be Conv op.
    if (!left_child_is_constant && !right_child_is_constant) return failure();

    Operation *conv_node =
        left_child_is_constant ? mul_right_child : mul_left_child;
    if (!dialect_->IsConv2D(conv_node) && !dialect_->IsConv3D(conv_node))
      return failure();

    // Make sure that it is safe to change the value of the convolution
    // output.
    if (helper_.ShouldPreserveOp(conv_node)) return failure();

    if (TFOp(op).deviceAttr() != TFOp(mul_left_child).deviceAttr() ||
        TFOp(op).deviceAttr() != TFOp(mul_right_child).deviceAttr()) {
      return failure();
    }

    // Identify the nodes to swap.
    Operation *conv_left_child = conv_node->getOperand(0).getDefiningOp();
    Operation *conv_right_child = conv_node->getOperand(1).getDefiningOp();
    const bool conv_left_is_constant =
        conv_left_child && dialect_->IsConstant(conv_left_child);
    const bool conv_right_is_constant =
        conv_right_child && dialect_->IsConstant(conv_right_child);
    if (!conv_left_is_constant && !conv_right_is_constant) {
      // At least one of the convolution inputs should be constant.
      return failure();
    }

    if (conv_left_is_constant && conv_right_is_constant) {
      // Operation evaluation will handle this.
      return failure();
    }

    ShapedType mul_shape = (*op->result_type_begin()).cast<ShapedType>();
    ShapedType conv_shape =
        (*conv_node->result_type_begin()).cast<ShapedType>();
    // TODO(chiahungduan): Symbolic shape equivalence is acceptable.
    if (!mul_shape.hasStaticShape() || !conv_shape.hasStaticShape() ||
        mul_shape != conv_shape) {
      return failure();
    }

    auto filter_shape = conv_node->getOperand(1).getType().cast<ShapedType>();

    Operation *const_node =
        left_child_is_constant ? mul_left_child : mul_right_child;
    auto const_node_shape =
        (*const_node->result_type_begin()).cast<ShapedType>();
    if (!IsValidConstShapeForMulConvPushDown(
            conv_node->getAttrOfType<StringAttr>("data_format"), filter_shape,
            const_node_shape)) {
      return failure();
    }

    Operation *conv_const_node =
        conv_left_is_constant ? conv_left_child : conv_right_child;
    // Make sure we don't introduce loops in the graph by removing control
    // dependencies from the conv2d node to c2.
    if (Operation *new_const_op =
            RemoveControlOperandIfExist(rewriter, const_node, conv_node)) {
      rewriter.replaceOp(const_node, new_const_op->getResults());
      const_node = new_const_op;

      // Add a control dep from c1 to c2 to ensure c2 is in the right frame
      AddControlOperand(const_node, TFOp(conv_const_node).controlRet(),
                        rewriter);
    }

    StringRef conv_node_name = TFOp(conv_node).name();

    rewriter.startRootUpdate(conv_node);
    TFOp(conv_node).setName(TFOp(op).nameAttr());
    if (conv_left_is_constant)
      conv_node->setOperand(0, op->getResult(0));
    else
      conv_node->setOperand(1, op->getResult(0));
    rewriter.finalizeRootUpdate(conv_node);

    rewriter.startRootUpdate(op);
    TFOp(op).setName(Twine(conv_node_name, "/merged_input"));
    if (left_child_is_constant)
      op->setOperand(1, conv_const_node->getResult(0));
    else
      op->setOperand(0, conv_const_node->getResult(0));
    rewriter.finalizeRootUpdate(op);

    return success();
  }

 private:
  // Remove the control dependency from `op` to `to_remove` if any.
  Operation *RemoveControlOperandIfExist(OpBuilder &builder, Operation *op,
                                         Operation *to_remove) const {
    auto [non_control_operands, control_operands] = TFOp(op).splitOperands();
    Value control_to_remove = TFOp(to_remove).controlRet();
    SmallVector<Value> new_control_operands(control_operands);
    auto it = llvm::remove_if(
        new_control_operands,
        [control_to_remove](Value v) { return v == control_to_remove; });
    if (it == new_control_operands.end()) return nullptr;
    new_control_operands.erase(it, new_control_operands.end());

    OperationState state(op->getLoc(), op->getName());
    state.addOperands(non_control_operands);
    state.addOperands(new_control_operands);
    state.addAttributes(op->getAttrs());
    state.addTypes(op->getResultTypes());

    return builder.create(state);
  }
};

// This implementation is mapped with ConstantFolding::PartialConcatConstFolding
// in grappler/optimizers/constant_folding.cc
class PartialConcatConstFolding
    : public FolderPatternBase<PartialConcatConstFolding> {
 public:
  explicit PartialConcatConstFolding(OpPropertyHelper &helper)
      : FolderPatternBase<PartialConcatConstFolding>(MatchAnyOpTypeTag(),
                                                     helper) {}
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    // Partial constant folding for Concat which is not commutative, so
    // we have to preserve order and can only push consecutive runs of constant
    // inputs into sub-nodes.
    if (!dialect_->IsConcat(op)) return failure();
    if (TFOp(op).name().rfind("_partial_split_") != StringRef::npos) {
      return failure();
    }

    auto [non_control_operands, control_operands] = TFOp(op).splitOperands();
    const int num_non_control_inputs = non_control_operands.size();
    if (num_non_control_inputs <= 3) return failure();

    int axis_arg = -1;
    int begin = 0;
    int end = num_non_control_inputs;
    // Note that IsConcat includes both Concat and ConcatV2 so that we need to
    // check ConcatV2 first.
    if (dialect_->IsConcatV2(op)) {
      end = num_non_control_inputs - 1;
      axis_arg = num_non_control_inputs - 1;
    } else if (dialect_->IsConcat(op)) {
      begin = 1;
      axis_arg = 0;
    } else {
      return failure();
    }

    // We search for consecutive runs of constant inputs in the range
    // [begin:end] and push then down into child nodes.
    SmallVector<std::pair<int, int>> constant_input_runs;
    int first = begin;
    int last = begin;
    while (last < end) {
      while (first < end) {
        Operation *v_op = op->getOperand(first).getDefiningOp();
        if (v_op && dialect_->IsConstant(v_op)) break;
        ++first;
      }

      // Invariant: node[first] is constant || first >= end.
      last = first + 1;
      while (last < end) {
        Operation *v_op = op->getOperand(last).getDefiningOp();
        if (!v_op || !dialect_->IsConstant(v_op)) break;
        ++last;
      }

      // Invariant: node[last] is not constant || last >= end
      // Discard intervals shorter than 2 elements.
      if (first < end && (last - first) > 1)
        constant_input_runs.emplace_back(first, last);
      first = last;
    }

    // Skip if all inputs are constant, and let constant folding take over.
    if (constant_input_runs.empty() || (constant_input_runs.size() == 1 &&
                                        constant_input_runs[0].first == begin &&
                                        constant_input_runs[0].second == end)) {
      return failure();
    }

    // TODO(chiahungduan): The optimization is able to be applied multiple
    // times. Find a better way to name the new ops without having duplicate
    // name. Now we just optimize it once.
    if (llvm::any_of(non_control_operands, [](Value v) {
          Operation *v_op = v.getDefiningOp();
          return v_op &&
                 TFOp(v_op).name().rfind("_partial_split_") != StringRef::npos;
        })) {
      return failure();
    }

    DenseSet<int> inputs_to_delete;
    for (auto interval : constant_input_runs) {
      // Push the constant inputs in the interval to a child node than can be
      // constant folded.
      OperationState state(op->getLoc(), "tfg.ConcatV2");
      state.addOperands(op->getOperand(interval.first));
      for (auto i : llvm::seq<int>(interval.first + 1, interval.second)) {
        state.addOperands(op->getOperand(i));
        inputs_to_delete.insert(i);
      }
      state.addOperands(op->getOperand(axis_arg));
      state.attributes = op->getAttrDictionary();
      state.attributes.set("N",
                           IntegerAttr::get(rewriter.getI32Type(),
                                            interval.second - interval.first));
      state.addTypes(op->getResultTypes());

      Operation *new_op = rewriter.create(state);
      TFOp(new_op).setName(Twine(TFOp(op).name(), "/_partial_split_") +
                           std::to_string(interval.first));
      // Op inherits all the attrs of op, don't need to update the device attr.

      // Overwrite the first constant input with the result of the added
      // child node.
      rewriter.startRootUpdate(op);
      op->setOperand(interval.first, new_op->getResult(0));
      rewriter.finalizeRootUpdate(op);
    }

    if (!inputs_to_delete.empty()) {
      OperationState state(op->getLoc(), op->getName());
      for (auto &it : llvm::enumerate(non_control_operands)) {
        if (inputs_to_delete.contains(it.index())) continue;
        state.addOperands(it.value());
      }
      assert(state.operands.size() != non_control_operands.size());
      state.addOperands(control_operands);

      state.attributes = op->getAttrDictionary();
      state.attributes.set(
          "N", IntegerAttr::get(
                   rewriter.getI32Type(),
                   state.operands.size() - control_operands.size() - 1));
      state.addTypes(op->getResultTypes());
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
class ConstantPushDownBiasAdd
    : public ConstantPushDownBase<ConstantPushDownBiasAdd> {
 public:
  explicit ConstantPushDownBiasAdd(OpPropertyHelper &helper)
      : ConstantPushDownBase(MatchAnyOpTypeTag(), helper) {}
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (!dialect_->IsBiasAdd(op)) return failure();

    Operation *add_child = op->getOperand(0).getDefiningOp();
    if (!add_child) return failure();

    Operation *const_child = op->getOperand(1).getDefiningOp();
    if (!const_child || !dialect_->IsConstant(const_child)) return failure();

    if (helper_.ShouldPreserveOp(add_child)) return failure();

    // Special case for BiasAdd: Since the left argument to BiasAdd must be rank
    // >= 2 and the leaves must be vectors, we cannot swap them.
    if (dialect_->IsConstant(add_child)) return failure();
    if (!dialect_->IsBiasAdd(add_child) && !dialect_->IsAdd(add_child))
      return failure();

    if (!IsOperandsSafeToMove(add_child, const_child)) return failure();

    auto hasRank = [&](Value value) {
      return value.getType().cast<ShapedType>().hasRank();
    };

    if (!hasRank(op->getOperand(0)) || !hasRank(op->getOperand(1)) ||
        !hasRank(add_child->getOperand(0)) ||
        !hasRank(add_child->getOperand(1))) {
      return failure();
    }

    // Now get the ranks and types of the 3 leaf nodes.
    const int left_leaf_rank =
        add_child->getOperand(0).getType().cast<ShapedType>().getRank();
    const int right_leaf_rank =
        add_child->getOperand(1).getType().cast<ShapedType>().getRank();

    // At least one leaf must be a vector.
    if (left_leaf_rank != 1 && right_leaf_rank != 1) return failure();

    const int vector_idx = left_leaf_rank == 1 ? 0 : 1;
    auto vector_type =
        add_child->getOperand(vector_idx).getType().cast<ShapedType>();
    Type vector_d_type = vector_type.getElementType();

    auto const_type = const_child->getResultTypes()[0].cast<ShapedType>();
    const int const_rank = const_type.getRank();
    Type const_d_type = const_type.getElementType();

    if (const_rank != 1 || const_d_type != vector_d_type) return failure();

    // This is case #1, #3, and #4:
    int input_to_swap = vector_idx;

    Value leaf_to_swap = add_child->getOperand(input_to_swap);
    if (leaf_to_swap.getDefiningOp() &&
        dialect_->IsConstant(leaf_to_swap.getDefiningOp())) {
      return failure();
    }

    rewriter.startRootUpdate(op);
    op->setOperand(1, leaf_to_swap);
    rewriter.finalizeRootUpdate(op);
    rewriter.startRootUpdate(add_child);
    add_child->setOperand(input_to_swap, const_child->getResult(0));
    rewriter.finalizeRootUpdate(add_child);

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
class ConstantPushDownAdd : public ConstantPushDownBase<ConstantPushDownAdd> {
 public:
  explicit ConstantPushDownAdd(OpPropertyHelper &helper)
      : ConstantPushDownBase(MatchAnyOpTypeTag(), helper) {}
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (!dialect_->IsAdd(op)) return failure();

    Operation *add_child = op->getOperand(0).getDefiningOp();
    Operation *const_child = op->getOperand(1).getDefiningOp();
    if (!add_child || !const_child) return failure();

    if (!dialect_->IsConstant(const_child)) std::swap(add_child, const_child);
    if (!dialect_->IsConstant(const_child)) return failure();

    if (!IsOperandsSafeToMove(add_child, const_child)) return failure();

    bool child_is_bias_add = dialect_->IsBiasAdd(add_child);
    if (!child_is_bias_add && !dialect_->IsAdd(add_child)) return failure();

    auto hasRank = [&](Value value) {
      return value.getType().cast<ShapedType>().hasRank();
    };

    if (!hasRank(op->getOperand(0)) || !hasRank(op->getOperand(1)) ||
        !hasRank(add_child->getOperand(0)) ||
        !hasRank(add_child->getOperand(1))) {
      return failure();
    }

    // Now get the ranks and types of the 3 leaf nodes.
    const int left_leaf_rank =
        add_child->getOperand(0).getType().cast<ShapedType>().getRank();
    const int right_leaf_rank =
        add_child->getOperand(1).getType().cast<ShapedType>().getRank();
    // At least one leaf must be a vector.
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

    const int const_index =
        op->getOperand(0).getDefiningOp() == const_child ? 0 : 1;
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

    Value leaf_to_swap = add_child->getOperand(input_to_swap);
    if (leaf_to_swap.getDefiningOp() &&
        dialect_->IsConstant(leaf_to_swap.getDefiningOp())) {
      return failure();
    }

    rewriter.startRootUpdate(op);
    op->setOperand(const_index, leaf_to_swap);
    rewriter.finalizeRootUpdate(op);
    rewriter.startRootUpdate(add_child);
    add_child->setOperand(input_to_swap, const_child->getResult(0));
    rewriter.finalizeRootUpdate(add_child);

    return success();
  }
};

// This implementation is mapped with
// ConstantFolding::RemoveRedundantVariableUpdates in
// grappler/optimizers/constant_folding.cc
class RemoveRedundantVariableUpdates
    : public FolderPatternBase<RemoveRedundantVariableUpdates> {
 public:
  explicit RemoveRedundantVariableUpdates(OpPropertyHelper &helper)
      : FolderPatternBase<RemoveRedundantVariableUpdates>(MatchAnyOpTypeTag(),
                                                          helper) {}
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    static const auto *kVariableReadOps =
        new absl::flat_hash_set<std::string>{"AssignAddVariableOp",
                                             "AssignSubVariableOp",
                                             "AssignAdd",
                                             "AssignSub",
                                             "ScatterAdd",
                                             "ScatterSub",
                                             "ScatterMul",
                                             "ScatterDiv",
                                             "ScatterNdAdd",
                                             "ScatterNdSub",
                                             "ScatterNdMul",
                                             "ScatterNdDiv",
                                             "ResourceScatterAdd",
                                             "ResourceScatterSub",
                                             "ResourceScatterMul",
                                             "ResourceScatterDiv",
                                             "ResourceScatterNdAdd",
                                             "ResourceScatterNdSub",
                                             "ResourceScatterNdMul",
                                             "ResourceScatterNdDiv"};
    StringRef op_name = op->getName().stripDialect();
    if (kVariableReadOps == nullptr ||
        kVariableReadOps->find({op_name.data(), op_name.size()}) ==
            kVariableReadOps->end())
      return failure();
    const int value_index = op_name.contains("Scatter") ? 2 : 1;
    Operation *delta_op = op->getOpOperand(value_index).get().getDefiningOp();
    if (delta_op == nullptr) return failure();
    const bool is_add_or_sub =
        op_name.contains("Add") || op_name.contains("Sub");
    if ((is_add_or_sub && helper_.IsZeros(delta_op)) ||
        (!is_add_or_sub && helper_.IsOnes(delta_op))) {
      if (op_name.contains("Variable") || op_name.contains("Resource")) {
        FailureOr<TFOp> no_op = ReplaceOpWithNoOp(rewriter, op);
        if (failed(no_op)) return failure();
        rewriter.replaceOp(op, (*no_op)->getResults());
        return success();
      } else {
        FailureOr<TFOp> identity =
            ReplaceOpWithIdentity(rewriter, op, /*idx*/ 0);
        if (failed(identity)) return failure();
        rewriter.replaceOp(op, (*identity)->getResults());
        return success();
      }
    }
    return failure();
  }
};

// This implementation is mapped with ConstantFolding::SimplifyCase in
// grappler/optimizers/constant_folding.cc
class SimplifyCaseOp : public FolderPatternBase<SimplifyCaseOp> {
 public:
  explicit SimplifyCaseOp(OpPropertyHelper &helper)
      : FolderPatternBase<SimplifyCaseOp>("tfg.Case", helper) {}
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    Operation *branch_index_op = op->getOperand(0).getDefiningOp();
    if (!branch_index_op) return failure();

    ElementsAttr value_attr =
        branch_index_op->getAttrOfType<ElementsAttr>("value");
    if (!value_attr) return failure();

    int output_idx = value_attr.getSplatValue<int>();
    ArrayAttr branch_attr = op->getAttrOfType<ArrayAttr>("branches");
    if (output_idx < 0 || output_idx >= branch_attr.size()) return failure();

    OperationState state(op->getLoc(), "tfg.PartitionedCall");
    state.addOperands(ValueRange(op->getOperands()).drop_front());

    state.attributes = op->getAttrDictionary();
    state.attributes.erase("branches");
    // In TFG conanical form, `output_shapes` has been consolidated into op's
    // shape. Unlike grappler, we don't need to update the `output_shapes` attr
    // here.
    state.attributes.set("f", branch_attr[output_idx]);

    state.addTypes(op->getResultTypes());

    Operation *partitioned_call_op = rewriter.create(state);
    rewriter.replaceOp(op, partitioned_call_op->getResults());

    return success();
  }
};

// This implementation is mapped with ConstantFolding::SimplifySelect in
// grappler/optimizers/constant_folding.cc
template <typename ConcreteType>
class SimplifySelectOpBase : public FolderPatternBase<ConcreteType> {
 protected:
  SimplifySelectOpBase(StringRef op_name, OpPropertyHelper &helper)
      : FolderPatternBase<ConcreteType>(op_name, helper) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    Operation *condition_op = op->getOperand(0).getDefiningOp();
    if (!condition_op) return failure();

    bool is_all_true = this->helper_.IsOnes(condition_op);
    bool is_all_false = this->helper_.IsZeros(condition_op);
    if (!is_all_true && !is_all_false) return failure();

    auto condition_type = op->getOperand(0).getType().cast<ShapedType>();
    auto t_type = op->getOperand(1).getType().cast<ShapedType>();
    auto e_type = op->getOperand(2).getType().cast<ShapedType>();
    if (!condition_type.hasStaticShape() || !t_type.hasStaticShape() ||
        !e_type.hasStaticShape()) {
      return failure();
    }

    const int live_input_idx = is_all_true ? 1 : 2;
    bool predicate_is_scalar = condition_type.getRank() == 0;

    if (t_type.getShape() == e_type.getShape() &&
        (condition_type.getShape() == t_type.getShape() ||
         predicate_is_scalar)) {
      Value live_operand = op->getOperand(live_input_idx);
      OperationState state(op->getLoc(), "tfg.Identity");
      state.addTypes(op->getResultTypes());

      state.addOperands(live_operand);
      auto [non_control_operands, control_operands] = TFOp(op).splitOperands();
      for (Value operand : non_control_operands) {
        if (operand == live_operand) continue;
        // Add the remaining operands as control operands.
        state.addOperands(GetControlDependency(rewriter, operand));
      }
      // Append control operands
      state.addOperands(control_operands);

      state.attributes = op->getAttrDictionary();
      Operation *identity = rewriter.create(state);
      rewriter.replaceOp(op, identity->getResults());
    } else {
      FailureOr<TFOp> broadcast_to_op =
          ReplaceOpWithBroadcastTo(rewriter, op, live_input_idx);
      if (failed(broadcast_to_op)) return failure();
      rewriter.replaceOp(op, (*broadcast_to_op)->getResults());
    }

    return success();
  }
};

class SimplifySelectOp : public SimplifySelectOpBase<SimplifySelectOp> {
 public:
  explicit SimplifySelectOp(OpPropertyHelper &helper)
      : SimplifySelectOpBase("tfg.Select", helper) {}
};

class SimplifySelectV2Op : public SimplifySelectOpBase<SimplifySelectV2Op> {
 public:
  explicit SimplifySelectV2Op(OpPropertyHelper &helper)
      : SimplifySelectOpBase("tfg.SelectV2", helper) {}
};

namespace {

// Utilities for filtering desired patterns.
template <bool>
struct FilterPattern {
  template <class Pattern>
  using type = std::tuple<Pattern>;
};
template <>
struct FilterPattern<false> {
  template <class Pattern>
  using type = std::tuple<>;
};
template <template <class> class Pred, class... Patterns>
struct FilterPatterns {
  using type = decltype(std::tuple_cat(
      std::declval<typename FilterPattern<Pred<Patterns>::value>::template type<
          Patterns>>()...));
};

// Predicates of selecting pattern kind.
template <typename Pattern>
using FolderPatterns = std::is_base_of<FolderTrait<Pattern>, Pattern>;
template <typename Pattern>
using PropagationPatterns = std::is_base_of<PropagationTrait<Pattern>, Pattern>;
template <typename Pattern>
using AllPatterns = std::true_type;

// Registers a set of patterns.
template <typename... Patterns>
struct TargetPatterns;
template <typename... Patterns>
struct TargetPatterns<std::tuple<Patterns...>> {
  static void Register(::mlir::RewritePatternSet &patterns,
                       OpPropertyHelper &helper) {
    patterns.insert<Patterns...>(helper);
  }
};
template <template <class> class PatternsFilter>
void RegisterPatterns(::mlir::RewritePatternSet &patterns,
                      OpPropertyHelper &helper) {
  TargetPatterns<typename FilterPatterns<
      PatternsFilter, MaterializeBroadcastGradientArgsOp, MaterializeShapeNOp,
      SimplifySwitchOp, MergeNodeFolding, RefMergeNodeFolding,
      XlaMergeNodeFolding, MoveConstantsPastEnterOp,
      MoveConstantsPastRefEnterOp, MaterializeReductionIndices,
      PartialConstPropThroughIdentityN, ConstantPushDown, MulConvPushDown,
      ConstantPushDownBiasAdd, ConstantPushDownAdd, EvaluateConstant,
      PartialConcatConstFolding, PartialAssocOpConstFolding,
      SimplifyArithmeticOp, ReduceDivToReciprocalMul, SimplifyReshapeOp,
      RemoveReverse, SimplifyStridedSlice, SimplifyTileOp, SimplifySqueezeOp,
      SimplifySliceOp, RemoveTransposeOp, RemoveRandomShuffleOp,
      RemoveShuffleOp, SimplifyPackOp, SimplifyReductionOp, SimplifyPadOp,
      SimplifyPadV2Op, RemoveRedundantVariableUpdates, RemoveSplitOp,
      RemoveSplitVOp, MaterializeFillNode, MaterializeConstantValuedNode,
      MaterializeShapeOp, MaterializeRankOp, MaterializeSizeOp,
      MaterializeTensorArraySizeV3Op, MergeConcatOp, SimplifyCaseOp,
      SimplifySelectOp, SimplifySelectV2Op>::type>::Register(patterns, helper);
}
}  // namespace

class ConstantFolding : public impl::ConstantFoldingPassBase<ConstantFolding> {
 public:
  LogicalResult initialize(MLIRContext *context) override {
    helper_ = std::make_shared<OpPropertyHelper>(
        context->getOrLoadDialect<TFGraphDialect>(),
        disable_compressed_tensor_optimization_);
    RewritePatternSet patterns(context);
    populatePatterns(patterns);
    final_patterns_ = std::move(patterns);
    return success();
  }

  void runOnOperation() override;

 private:
  void populatePatterns(::mlir::RewritePatternSet &patterns) {
    switch (pattern_category_) {
      default:
        LOG(ERROR) << "unknown pattern category, will run all patterns";
        [[fallthrough]];
      case 0: {
        RegisterPatterns<AllPatterns>(patterns, *helper_);
        break;
      }
      case 1: {
        RegisterPatterns<FolderPatterns>(patterns, *helper_);
        break;
      }
      case 2: {
        RegisterPatterns<PropagationPatterns>(patterns, *helper_);
        break;
      }
    }
  }

  FrozenRewritePatternSet final_patterns_;
  std::shared_ptr<OpPropertyHelper> helper_;
};

void ConstantFolding::runOnOperation() {
  // TODO(chiahungduan): Set up the attributes before operation creation.
  // Because of the conveniency, in some cases we set up the device/name later
  // operation creation.

  GraphFuncOp func = getOperation();

  // The max iteration is the same as the max default iteration in
  // applyPatternsAndFoldGreedily.
  constexpr int max_iterations = 10;
  int iteration = 0;

  SmallVector<Operation *> ops;
  do {
    ops.clear();
    for (Operation &op : func.SingleBlock::getBody()->without_terminator()) {
      ops.push_back(&op);
    }
    bool changed = false;
    GreedyRewriteConfig config;
    config.strictMode = GreedyRewriteStrictness::ExistingAndNewOps;
    (void)applyOpPatternsAndFold(ops, final_patterns_, config, &changed);
    if (!changed) break;
  } while (iteration++ < max_iterations);

  // TODO(chiahungduan): This is used to avoid evaluating a node multiple times.
  // See more details in EvaluateConstant pattern. Maybe we can remove this by
  // checking if the user of an op is empty.
  auto has_folded = StringAttr::get(&getContext(), "has_folded");
  getOperation()->walk([&](Operation *op) { op->removeAttr(has_folded); });
}

std::unique_ptr<Pass> CreateConstantFoldingPass() {
  return std::make_unique<ConstantFolding>();
}

}  // namespace tfg
}  // namespace mlir
