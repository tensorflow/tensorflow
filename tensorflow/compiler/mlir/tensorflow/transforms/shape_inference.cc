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

#include "tensorflow/compiler/mlir/tensorflow/transforms/shape_inference.h"

#include <algorithm>
#include <cstdint>
#include <initializer_list>
#include <iterator>
#include <queue>
#include <stack>

#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/None.h"
#include "llvm/ADT/PointerUnion.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"  // from @llvm-project
#include "mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Block.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinDialect.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Diagnostics.h"  // from @llvm-project
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/OperationSupport.h"  // from @llvm-project
#include "mlir/IR/SymbolTable.h"  // from @llvm-project
#include "mlir/IR/TypeUtilities.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/IR/Visitors.h"  // from @llvm-project
#include "mlir/Interfaces/CallInterfaces.h"  // from @llvm-project
#include "mlir/Interfaces/FoldInterfaces.h"  // from @llvm-project
#include "mlir/Interfaces/InferTypeOpInterface.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassRegistry.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/FoldUtils.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_executor.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/serialize_mlir_module_utils.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/shape_inference_utils.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/translate_utils.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/types.pb.h"

#define DEBUG_TYPE "tf-shape-inference"

#define DCOMMENT(MSG) LLVM_DEBUG(llvm::dbgs() << MSG << "\n")
#define DCOMMENT_OP(OP, MSG) \
  LLVM_DEBUG(OP->print(llvm::dbgs() << MSG << " "); llvm::dbgs() << "\n")

using ::tensorflow::int64;
using tensorflow::shape_inference::DimensionHandle;
using tensorflow::shape_inference::InferenceContext;
using tensorflow::shape_inference::ShapeHandle;

namespace mlir {
namespace TF {
namespace {

// Compute a refined type between two types `lhs` and `rhs`, the result type
// is always more refined (i.e. has more static information) than `lhs`
// This method will actually merge the information contained in the
// types, it is capable of refining:
//   tensor<!tf.variant<tensor<?x8xf32>>>
// and:
//   tensor<!tf.variant<tensor<10x?xf32>>>
// into:
//   tensor<!tf.variant<tensor<10x8xf32>>>
//
// In case of inconsistencies (rank disagreement for example), it returns `lhs`.
Type TypeMeet(Type lhs, Type rhs) {
  DCOMMENT("RefineTypeWith : " << lhs << " : " << rhs);
  if (lhs == rhs) return lhs;

  auto rhs_shape_type = rhs.dyn_cast<ShapedType>();
  if (!rhs_shape_type) return lhs;
  auto lhs_shape_type = lhs.cast<ShapedType>();
  if (lhs_shape_type.hasRank() && rhs_shape_type.hasRank() &&
      lhs_shape_type.getRank() != rhs_shape_type.getRank()) {
    DCOMMENT("Unexpected rank mismatch: " << lhs << " vs " << rhs);
    return lhs;
  }

  SmallVector<int64_t> shape;
  bool refined_shape = false;
  // Build the shape of the refined type, if lhs is unranked it
  // will be directly the shape of the refined type, otherwise we merged by
  // taking the most specialized. This combines `10x?x?` and `?x?x8` into
  // `10x?x8`.
  if (!lhs_shape_type.hasRank()) {
    if (rhs_shape_type.hasRank()) {
      shape.append(rhs_shape_type.getShape().begin(),
                   rhs_shape_type.getShape().end());
      refined_shape = true;
    }
  } else if (rhs_shape_type.hasRank()) {
    for (auto shape_elts : llvm::enumerate(
             llvm::zip(lhs_shape_type.getShape(), rhs_shape_type.getShape()))) {
      if (ShapedType::isDynamic(std::get<0>(shape_elts.value())) &&
          !ShapedType::isDynamic(std::get<1>(shape_elts.value()))) {
        shape.push_back(std::get<1>(shape_elts.value()));
        refined_shape = true;
        DCOMMENT("-> refining shape element #" << shape_elts.index());
      } else {
        DCOMMENT("-> not refining shape element #" << shape_elts.index());
        shape.push_back(std::get<0>(shape_elts.value()));
      }
    }
  }

  // Some tensor have an element type wrapping a subtensor, like resource and
  // variants. In this case we may recurse on the wrapped subtype.
  // `element_type` will contain the refined inferred element type for the
  // returned type.
  auto lhs_element_type = lhs_shape_type.getElementType();
  auto rhs_element_type_with_subtype =
      rhs_shape_type.getElementType().dyn_cast<TF::TensorFlowTypeWithSubtype>();
  // Look for resource or variant element type and ensure we refine the subtype.
  // We only support a single subtype at the moment, we won't handle something
  // like:
  //   tensor<!tf.variant<tensor<10xf32>, tensor<8xf32>>
  if (rhs_element_type_with_subtype &&
      rhs_element_type_with_subtype.GetSubtypes().size() == 1) {
    auto lhs_element_type_with_subtype =
        lhs_element_type.dyn_cast<TF::TensorFlowTypeWithSubtype>();
    TensorType subtype;
    if (!lhs_element_type_with_subtype) {
      DCOMMENT(
          "Unexpected inferred `TensorFlowTypeWithSubtype` when original "
          "result isn't");
    } else if (lhs_element_type_with_subtype.GetSubtypes().size() > 1) {
      DCOMMENT(
          "Unexpected `TensorFlowTypeWithSubtype` original type with size>1");
    } else if (lhs_element_type_with_subtype.GetSubtypes().empty()) {
      subtype = rhs_element_type_with_subtype.GetSubtypes().front();
    } else {
      // Recurse on the subtypes in the variant/resource. Basically if the input
      // were:
      //   tensor<!tf.variant<tensor<?x8xf32>>>
      // and:
      //   tensor<!tf.variant<tensor<10x8xf32>>>
      // we'll try here to refine tensor<?x8xf32> with tensor<10x8xf32>.
      auto refined_subtype =
          TypeMeet(lhs_element_type_with_subtype.GetSubtypes().front(),
                   rhs_element_type_with_subtype.GetSubtypes().front())
              .cast<TensorType>();
      if (refined_subtype !=
          lhs_element_type_with_subtype.GetSubtypes().front())
        subtype = refined_subtype;
    }
    // If we managed to refine the subtype, recreate the element type itself
    // (i.e. the tf.variant or tf.resource).
    if (subtype) {
      lhs_element_type = lhs_element_type_with_subtype.clone({subtype});
    }
  }
  if (refined_shape || lhs_element_type != lhs_shape_type.getElementType()) {
    Type new_type;
    if (!lhs_shape_type.hasRank() && !rhs_shape_type.hasRank())
      new_type = UnrankedTensorType::get(lhs_element_type);
    else
      new_type = lhs_shape_type.clone(shape, lhs_element_type);
    DCOMMENT("Refined to: " << new_type);
    return new_type;
  }
  DCOMMENT("No refinement " << lhs);
  return lhs;
}

// Returns whether `original_type` type can be refined with
// `potential_refined_type` type.
bool CanRefineTypeWith(Type original_type, Type potential_refined_type) {
  return original_type != TypeMeet(original_type, potential_refined_type);
}

// Returns if the shape inference pass supports an op outside the TF dialect.
bool IsSupportedNonTFOp(Operation* op) {
  return isa<tf_device::ReturnOp, tf_device::ClusterOp, tf_device::LaunchOp,
             tf_executor::EnterOp, tf_executor::ExitOp, tf_executor::FetchOp,
             tf_executor::GraphOp, tf_executor::IslandOp,
             tf_executor::LoopCondOp, tf_executor::MergeOp,
             tf_executor::NextIterationSinkOp, tf_executor::SwitchNOp,
             tf_executor::SwitchOp, tf_executor::YieldOp>(op) ||
         isa<InferTypeOpInterface>(op);
}

// Returns whether a cast back would need to be inserted, e.g., whether the
// operation of which use is an operand allows for shape refinement without
// a cast.
bool NeedsCastBack(OpOperand& use, Dialect* tf_dialect) {
  return use.getOwner()->getDialect() != tf_dialect &&
         !IsSupportedNonTFOp(use.getOwner());
}

TensorType CreateTensorType(llvm::Optional<llvm::ArrayRef<int64_t>> shape,
                            Type element_type) {
  if (shape.hasValue())
    return RankedTensorType::get(shape.getValue(), element_type);
  return UnrankedTensorType::get(element_type);
}

// Returns true if the op creates a TensorList.
bool IsTensorListInitOp(Operation* op) {
  return isa<TensorListReserveOp>(op) || isa<EmptyTensorListOp>(op) ||
         isa<TensorListFromTensorOp>(op);
}

// Returns the `element_shape` operand of the ops that create a TensorList.
Value GetElementShapeOperand(Operation* op) {
  if (auto empty_tl = dyn_cast<EmptyTensorListOp>(op))
    return empty_tl.element_shape();
  if (auto tl_reserve = dyn_cast<TensorListReserveOp>(op))
    return tl_reserve.element_shape();
  if (auto tl_from_tensor = dyn_cast<TensorListFromTensorOp>(op))
    return tl_from_tensor.element_shape();
  llvm_unreachable("unsupported TensorList op");
}

// Utility function to create a ranked tensor type after dropping the first
// dimension from the input type.
RankedTensorType DropFirstDimension(Type type) {
  RankedTensorType ranked_type = type.dyn_cast<RankedTensorType>();
  if (!ranked_type) return {};
  llvm::ArrayRef<int64_t> dims_except_first =
      ranked_type.getShape().drop_front();
  return RankedTensorType::get(dims_except_first, ranked_type.getElementType());
}

Operation* InsertCast(OpBuilder& b, Location loc, Type dst_type, Value input) {
  Type element_type = getElementTypeOrSelf(dst_type);
  if (element_type.isa<IndexType>())
    return b.create<tensor::CastOp>(loc, dst_type, input);
  if (isa<TensorFlowDialect, BuiltinDialect>(element_type.getDialect()))
    return b.create<TF::CastOp>(loc, dst_type, input,
                                /*truncate=*/b.getBoolAttr(false));
  return nullptr;
}

// Follow the use chain of TensorList and return true iff all elements written
// to TensorList have same static shape. If all elements have same shape, assign
// it to `potential_element_type`.
//
// This can handle multiple mutations of a TensorList object and would return
// true if across all mutations the elements written have the same shape.
bool CanInferTensorListElementType(Value tensorlist,
                                   Value initial_element_shape,
                                   RankedTensorType* potential_element_type) {
  DCOMMENT("CanInferTensorListElementType " << tensorlist << " with initial "
                                            << initial_element_shape);
  // Verifies if the new element type has static shape and matches the potential
  // type passed from caller. Updates the potential_element_type, if not defined
  // yet.
  auto verify_and_update_potential_element_type =
      [&](RankedTensorType new_element_type) -> bool {
    DCOMMENT("\t\tConsidering " << new_element_type << " with old "
                                << *potential_element_type);
    if (!new_element_type || !new_element_type.hasStaticShape()) return false;
    if (!*potential_element_type) {
      DCOMMENT("\t\tUpdating potential_element_type " << new_element_type);
      *potential_element_type = new_element_type;
      return true;
    }
    return *potential_element_type == new_element_type;
  };

  std::stack<Value> worklist;
  worklist.emplace(tensorlist);

  while (!worklist.empty()) {
    tensorlist = worklist.top();
    worklist.pop();

    // TensorLists are semantically immutable. For example, TensorListSetItem
    // takes a TensorList as input and produces a TensorList as output. So to
    // traverse modifications to TensorList and verify that all elements written
    // to it have the same shape, we need to follow use-def chain of ops that
    // (conceptually) modify it i.e., ops that take an input TensorList and
    // produce an output TensorList.
    for (auto& use : tensorlist.getUses()) {
      if (auto push = llvm::dyn_cast<TensorListPushBackOp>(use.getOwner())) {
        auto element_type =
            push.tensor().getType().dyn_cast<RankedTensorType>();
        if (!verify_and_update_potential_element_type(element_type))
          return false;
        worklist.emplace(push.output_handle());
        continue;
      }
      if (auto scatter = llvm::dyn_cast<TensorListScatterIntoExistingListOp>(
              use.getOwner())) {
        // For scatter op we can get the element shape by dropping the first
        // dimension of the input tensor.
        RankedTensorType element_type =
            DropFirstDimension(scatter.tensor().getType());
        if (!verify_and_update_potential_element_type(element_type))
          return false;
        worklist.emplace(scatter.output_handle());
        continue;
      }
      if (auto set_item = llvm::dyn_cast<TensorListSetItemOp>(use.getOwner())) {
        auto element_type =
            set_item.item().getType().dyn_cast<RankedTensorType>();
        DCOMMENT("\tTensorListSetItemOp " << element_type);
        if (!verify_and_update_potential_element_type(element_type))
          return false;
        worklist.emplace(set_item.output_handle());
        continue;
      }
      if (auto pop = llvm::dyn_cast<TensorListPopBackOp>(use.getOwner())) {
        worklist.emplace(pop.output_handle());
        continue;
      }
      if (auto resize = llvm::dyn_cast<TensorListResizeOp>(use.getOwner())) {
        worklist.emplace(resize.output_handle());
        continue;
      }
      // WhileRegionOp can explicitly capture TensorList value to be used inside
      // its regions. So we check the uses of corresponding block argument in
      // each region and the use of TensorList returned using YieldOp.
      if (auto while_region = llvm::dyn_cast<WhileRegionOp>(use.getOwner())) {
        DCOMMENT("\tTL WhileRegion");
        for (auto branch : while_region.getRegions())
          worklist.emplace(branch->getArgument(use.getOperandNumber()));
        continue;
      }
      if (auto yield = llvm::dyn_cast<YieldOp>(use.getOwner())) {
        Operation* parent = yield->getParentOp();
        worklist.emplace(parent->getResult(use.getOperandNumber()));
        continue;
      }
      // TODO(jpienaar): This can be generalized.
      if (isa<IdentityOp, IdentityNOp, StopGradientOp>(use.getOwner())) {
        worklist.emplace(use.getOwner()->getResult(use.getOperandNumber()));
        continue;
      }
      // Refining the tensor list element type might change the output of
      // TensorListElementShape which is expected to be the originally assigned
      // shape to TensorList init ops. So replace it with the original element
      // shape value.
      if (auto tl_element_shape =
              dyn_cast<TensorListElementShapeOp>(use.getOwner())) {
        // If element types match, we can do a direct replacement.
        if (getElementTypeOrSelf(tl_element_shape.getResult()) ==
            getElementTypeOrSelf(initial_element_shape.getType())) {
          tl_element_shape.replaceAllUsesWith(initial_element_shape);
        } else {
          OpBuilder b(use.getOwner());
          Operation* cast_op = InsertCast(
              b, use.getOwner()->getLoc(),
              tl_element_shape.getResult().getType(), initial_element_shape);
          if (!cast_op) return false;
          tl_element_shape.replaceAllUsesWith(cast_op->getResult(0));
        }
        continue;
      }
      // Ignore ops that just consume a TensorList and do not output another
      // TensorList.
      if (isa<TensorListStackOp, TensorListGatherOp, TensorListConcatV2Op,
              TensorListLengthOp, TensorListGetItemOp>(use.getOwner()))
        continue;

      // For any other unknown users of the TensorList, we are conservative and
      // stop element shape inference.
      DCOMMENT("TensorListType infer, unknown op " << *use.getOwner());
      return false;
    }
  }
  return true;
}
}  // namespace

// Returns whether type can be further refined.
bool CanBeRefined(Type type) {
  auto shape_type = type.dyn_cast<ShapedType>();
  if (!shape_type) return false;

  // Returns whether type with subtypes can be further refined.
  auto can_refine_subtypes = [](TF::TensorFlowTypeWithSubtype tws) {
    return tws.GetSubtypes().empty() ||
           llvm::any_of(tws.GetSubtypes(), CanBeRefined);
  };
  auto type_with_subtype =
      shape_type.getElementType().dyn_cast<TF::TensorFlowTypeWithSubtype>();
  if (type_with_subtype && can_refine_subtypes(type_with_subtype)) return true;

  return !shape_type.hasStaticShape();
}

// Combination of value producer and port of value produced (e.g.,
//   <value result output>:<value in output tensor>,
// so for tf.Const -> tensor<10x20xf32>, [0,2,18] would point to a unique output
// scalar value).
struct ValuePort {
  PointerUnion<Operation*, BlockArgument> producer;
  SmallVector<unsigned int, 2> port;

  bool operator==(const ValuePort& other) const {
    return producer == other.producer && port == other.port;
  }

  // Convert output value to ValuePort.
  explicit ValuePort(Value v) {
    OpResult opr = v.dyn_cast<OpResult>();
    if (opr) {
      producer = opr.getOwner();
      port = {opr.getResultNumber()};
    } else {
      producer = v.cast<BlockArgument>();
      port = {0};
    }
  }
  ValuePort(PointerUnion<Operation*, BlockArgument> producer,
            SmallVector<unsigned int, 2> port)
      : producer(producer), port(port) {}

  raw_ostream& print(raw_ostream& os) const {
    if (auto* op = producer.dyn_cast<Operation*>())
      os << "op " << op->getName();
    if (auto ba = producer.dyn_cast<BlockArgument>())
      os << "block_arg " << ba.getArgNumber();
    os << formatv(" [{0}]", llvm::make_range(port.begin(), port.end()));
    return os;
  }
};

struct ValuePortHasher {
  std::size_t operator()(const ValuePort& other) const {
    return hash_combine(llvm::hash_value(other.producer.getOpaqueValue()),
                        hash_value(ArrayRef<unsigned int>(other.port)));
  }
};

using ValuePortResultMap =
    std::unordered_map<ValuePort, Attribute, ValuePortHasher>;
using ComputedQueryFn = function_ref<bool(ValuePort)>;
using ValueQueryFn = function_ref<Attribute(const ValuePort&)>;
using ValuePortInputs = SmallVectorImpl<ValuePort>;

// TODO(jpienaar): ComputeInputsRequiredForOutput and ComputeOutputComponent are
// intended to be switched to op interfaces once more refined.
LogicalResult ComputeInputsRequiredForOutput(ValuePort value_port,
                                             ComputedQueryFn has_been_computed,
                                             ValuePortInputs* inputs) {
  auto op = value_port.producer.dyn_cast<Operation*>();
  auto& port = value_port.port;
  if (!op) return failure();

  // No inputs required for constants.
  if (matchPattern(op, m_Constant())) return success();

  // Note: this focusses only on the trivial pack op case and this could be
  // generalized.
  if (auto pack_op = dyn_cast<TF::PackOp>(op)) {
    auto type = pack_op.getType().cast<TensorType>();
    if (!type.hasRank() || type.getRank() != 1) return failure();
    if (port.size() != 2) return failure();
    assert(port[0] == 0);
    ValuePort req(pack_op.getOperand(port[1]));
    if (!has_been_computed(req)) inputs->push_back(req);
    return success();
  }

  return failure();
}

// Computes the output produced by ValuePort using the query function of
// existing computed values.
Attribute ComputeOutputComponent(const ValuePort& value_port,
                                 ValueQueryFn values) {
  LLVM_DEBUG(value_port.print(llvm::dbgs() << "Computing output for ") << "\n");
  if (auto known = values(value_port)) return known;

  auto op = value_port.producer.dyn_cast<Operation*>();
  if (!op) return nullptr;
  auto& port = value_port.port;

  if (port.empty()) {
    LLVM_DEBUG(llvm::dbgs() << "skipping, port outside spec of " << op << "\n");
    return nullptr;
  }

  ElementsAttr attr;
  if (matchPattern(op, m_Constant(&attr))) {
    if (port.size() == 1 && port[0] == 0) return attr;
    return nullptr;
  }

  if (auto id = dyn_cast<IdentityOp>(op)) {
    if (port.size() == 1 && port[0] == 0)
      return ComputeOutputComponent(ValuePort(id.input()), values);
    return nullptr;
  }

  // Note: this focusses only on the trivial pack op case and this could be
  // generalized.
  if (auto pack_op = dyn_cast<TF::PackOp>(op)) {
    TensorType type = pack_op.getType().cast<TensorType>();
    if (!type.hasRank() || type.getRank() != 1) return nullptr;
    if (port.size() != 2 || port[0] != 0) return nullptr;
    ValuePort op_port(op->getOperand(port[1]));
    return values(op_port);
  }

  if (auto graph = dyn_cast<tf_executor::GraphOp>(op)) {
    if (port.size() == 1)
      return ComputeOutputComponent(
          ValuePort(graph.GetFetch().fetches()[port[0]]), values);
    return nullptr;
  }

  if (auto island = dyn_cast<tf_executor::IslandOp>(op)) {
    if (port.size() == 1)
      return ComputeOutputComponent(
          ValuePort(island.GetYield().fetches()[port[0]]), values);
    return nullptr;
  }

  return nullptr;
}

// Context used during ShapeInference. This class contains common information
// that is required by the individual shape inference helper functions (e.g.,
// TF Graph version, constant values computed, etc.)
class ShapeInference {
 public:
  ShapeInference(int64_t graph_version, ModuleOp module,
                 bool propagate_caller_callee_constants);

  LogicalResult ComputeInputsRequiredForOutput(ValuePort value_port,
                                               ValuePortInputs* inputs) {
    return ::mlir::TF::ComputeInputsRequiredForOutput(
        value_port,
        [this](const ValuePort& port) {
          return results_.find(port) != results_.end();
        },
        inputs);
  }

  Attribute ComputeOutputComponent(const ValuePort& value_port) {
    if (auto known_attr = results_[value_port]) return known_attr;
    auto attr = ::mlir::TF::ComputeOutputComponent(
        value_port, [this](const ValuePort& port) { return results_[port]; });
    RecordValue(value_port, attr);
    return attr;
  }

  // Returns ShapeHandle if the op result could be computed as shape.
  ShapeHandle ComputeOutputAsShape(OpResult result, InferenceContext* ic);

  void RecordValue(const ValuePort& value_port, Attribute value) {
    LLVM_DEBUG(value_port.print(llvm::dbgs() << "\trecording ")
               << value << "\n");
    results_[value_port] = value;
  }

  // Infers shape of tf.While/tf.WhileRegion. If `shape_invariant` attribute is
  // set, operand types are set as result types if associated body result types
  // match the operand type (does not change per loop iteration). If operand and
  // body result types are not the same, only handle types are propagated to
  // result types. This is necessary to not incorrectly change result shapes
  // when the While op will have a different result shape. Otherwise operand
  // shapes are propagated to result shapes.
  template <typename WhileOpTy>
  bool InferShapeForWhile(WhileOpTy op, TypeRange body_result_types);

  // Performs shape inference on the provided op and return true if the type of
  // at least one result has been changed.
  // A tf.Cast() is inserted for any uses that isn't in the TensorFlow dialect.
  // `graph_version` indicates the current GraphDef compatibility versions
  // (the versions field in graph.proto).
  bool InferShapeForSingleOperation(Operation* op);

  // Infers shape on the provided region, including nested ones, iterate until
  // fix point with a limit of max_iteration.
  // Returns a failure() on error, otherwise returns true to indicate that it
  // reached convergence, false otherwise.
  FailureOr<bool> InferShapeUntilFixPoint(Region* region,
                                          int64_t max_iterations);

  // Updates input types and refine shapes inside body of functions that are
  // attached to ControlFlow ops (If/While) or Calls. These functions include
  // Then/Else branches of IfOp and Cond/Body functions of WhileOp. Functions
  // attached to control flow share following common properties:
  //   1) They are never reused, ie. having a single use in module.
  //   2) Their input types match those of their parent ops (excluding inputs
  //      like predicate).
  // For calls, functions can be reused across multiple call sites. In this case
  // we propagate the types when all call sites have the same operand types.
  // Returns a failure() on error, otherwise returns true to indicate that it
  // reached convergence, false otherwise.
  FailureOr<bool> PropagateShapeToFunctions(ModuleOp module,
                                            TypeRange input_types,
                                            ArrayRef<FuncOp> functions,
                                            int64_t max_iteration);

  // Propagates shapes to regions given the shapes of the inputs of the regions.
  // All regions provided in `regions` are assumed to have inputs of type
  // `input_types`.
  // Returns a failure() on error, otherwise returns true to indicate that it
  // reached convergence, false otherwise.
  FailureOr<bool> PropagateShapeToRegions(TypeRange input_types,
                                          ArrayRef<Region*> regions,
                                          int64_t max_iteration);

  // Shape propagation for call/control flow ops.
  // Returns a failure() on error, otherwise returns true to indicate that it
  // reached convergence, false otherwise.
  FailureOr<bool> PropagateShapeIntoAttachedFunctions(Operation* op,
                                                      int64_t max_iteration);

  // Shape propagation for region based control flow.
  // Returns a failure() on error, otherwise returns true to indicate that it
  // reached convergence, false otherwise.
  FailureOr<bool> PropagateShapeIntoAttachedRegions(Operation* op,
                                                    int64_t max_iterations);

  // Propagates any constant operand of call_op to the called function body's
  // corresponding argument if the callee has only one use.
  //
  // TODO(b/154065712): Move this to a more general inter-procedural constant
  // folding pass.
  void PropagateConstantToCallee(CallOpInterface call_op, FuncOp func,
                                 ModuleOp module);

  // Propagates any constant return value of the callee function to the call
  // op's corresponding result.
  void PropagateConstantFromCallee(CallOpInterface call_op, FuncOp func,
                                   ModuleOp module);

  // Tries to compute the result of folding the op. This doesn't actually
  // perform constant folding, it is just computes the equivalent constants.
  // Returns whether it was able to compute constant values.
  LogicalResult TryToFold(Operation* op);

  // Makes result types match the operand types (the i-th result type will
  // match the i-th operand type). Returns true if anything is changed.
  bool RefineTypeForPassThroughOperands(Operation* op, OperandRange operands,
                                        ResultRange results);

  // Makes result type's shape match the corresponding operand's shape.
  // Returns whether any change was made.
  bool RefineShapeForPassThroughOps(Operation* op);

  // Infers shape for necessary ops that are not in the TF dialect. Returns
  // whether any result type changed.
  bool InferShapeForNonTFDialectOperation(Operation* op);

  // Infers shape for function return type and returns whether changed.
  LogicalResult InferShapeForFunctionReturnType(FuncOp func);

  // Enqueues function for processing.
  void enqueue(FuncOp fn) {
    LLVM_DEBUG(llvm::dbgs()
               << "enqueue " << fn.getName() << " ("
               << (queue_set_.count(fn) ? "already inserted" : "newly inserted")
               << ")\n");
    if (queue_set_.insert(fn).second) queue_.push(fn);
  }

  // Enqueues callers on functions.
  void EnqueueCallers(FuncOp fn);

  // Returns the function at the front of the queue.
  FuncOp front() { return queue_.front(); }

  // Returns whether work queue is empty.
  bool EmptyQueue() const { return queue_.empty(); }

  // Returns function from the front of the work queue.
  FuncOp pop_front() {
    FuncOp ret = queue_.front();
    queue_.pop();
    queue_set_.erase(ret);
    return ret;
  }

  // Returns the current size of the queue.
  std::queue<FuncOp>::size_type QueueSize() const { return queue_.size(); }

  Dialect* const tf_dialect_;

 private:
  // Returns whether the result of an operation could be updated to a new
  // inferred type. Also inserts cast operation for uses that are incompatible
  // with the new type.
  bool UpdateTypeAndInsertIncompatibleUseCasts(Type new_type, Value result);

  // Refines the type of `result` of `op` using the type
  // `potential_refined_type`. Return true if the type was changed.
  bool RefineResultType(Operation* op, Value result,
                        Type potential_refined_type);

  // Infers the shape from a (Stateful)PartionedCall operation by looking up the
  // called function and propagating the return type.
  bool InferShapeForCall(CallOpInterface call_op);

  bool InferShapeForCast(Operation* op);

  // Infers the shape IfOp outputs based on the shapes of the then and else
  // function result types.
  bool InferShapeForIf(IfOp op);

  // Infers the shape IfRegion outputs based on the shapes of the then and else
  // yields.
  bool InferShapeForIfRegion(IfRegionOp op);

  // Infers the shape of _XlaHostComputeMlir based on the host computation
  // module.  Returns true if a return type was changed.
  bool InferShapeForXlaHostComputeMlir(_XlaHostComputeMlirOp op);

  // Infers the shape of ops that create TensorList. Specifically,
  // TensorListReserveOp, EmptyTensorListOp and TensorListFromTensor ops. It
  // refines the element shape if all tensors written to the list across all
  // mutations have identical static shape.
  bool InferShapeForTensorListInitOps(Operation* op);

  bool RefineWithInferTypeOpInterface(InferTypeOpInterface infer_ti);

  // Returns all the callers of a function.
  // Note: Usage of the return value of this function may not be interleaved
  // with insertions to the callers map. This could occur if GetCallers is
  // called with two separate functions, the 2nd one incurs a resize and then
  // both first and 2nd stored callers are used.
  ArrayRef<Operation*> GetCallers(FuncOp fn);

  // Mapping between ValuePort (which corresponds to an OpResult or smaller,
  // e.g., first element of OpResult produced) to an Attribute if the ValuePort
  // corresponds to a constant value.
  ValuePortResultMap results_;

  // Map from a function to the callers of that function.
  SymbolTableCollection symbol_table_;
  SymbolUserMap symbol_users_;

  // Queue of functions being processed.
  llvm::DenseSet<FuncOp> queue_set_;
  std::queue<FuncOp> queue_;

  int64_t graph_version_;

  // TODO(b/154065712): Remove propagate_caller_callee_constants once using
  // SCCP pass instead.
  bool propagate_caller_callee_constants_;
};

ShapeInference::ShapeInference(int64_t graph_version, ModuleOp module,
                               bool propagate_caller_callee_constants)
    : tf_dialect_(module->getContext()->getLoadedDialect<TensorFlowDialect>()),
      symbol_users_(symbol_table_, module),
      graph_version_(graph_version),
      propagate_caller_callee_constants_(propagate_caller_callee_constants) {}

ArrayRef<Operation*> ShapeInference::GetCallers(FuncOp fn) {
  return symbol_users_.getUsers(fn);
}

void ShapeInference::EnqueueCallers(FuncOp fn) {
  for (auto user : GetCallers(fn)) enqueue(user->getParentOfType<FuncOp>());
}

bool ShapeInference::UpdateTypeAndInsertIncompatibleUseCasts(Type new_type,
                                                             Value result) {
  Operation* cast_op = nullptr;
  // First insert cast back for uses that need a cast and then
  // update the type.
  bool enqueue_callers = false;
  for (OpOperand& use : make_early_inc_range(result.getUses())) {
    if (isa<ReturnOp>(use.getOwner())) {
      enqueue_callers = true;
    } else if (NeedsCastBack(use, tf_dialect_)) {
      if (!cast_op) {
        Operation* op = result.getDefiningOp();
        OpBuilder b(op);
        b.setInsertionPointAfter(op);
        cast_op = InsertCast(b, op->getLoc(), result.getType(), result);
        if (!cast_op) return false;
      }
      use.set(Value(cast_op->getResult(0)));
    }
  }

  result.setType(new_type);
  if (enqueue_callers)
    EnqueueCallers(result.getDefiningOp()->getParentOfType<FuncOp>());
  return true;
}

bool ShapeInference::RefineResultType(Operation* op, Value result,
                                      Type potential_refined_type) {
  if (!CanRefineTypeWith(result.getType(), potential_refined_type))
    return false;

  return UpdateTypeAndInsertIncompatibleUseCasts(potential_refined_type,
                                                 result);
}

// Infers the shape from a (Stateful)PartionedCall operation by looking up the
// called function and propagating the return type.
bool ShapeInference::InferShapeForCall(CallOpInterface call_op) {
  FuncOp func = dyn_cast<FuncOp>(call_op.resolveCallable());
  if (!func) return false;

  DCOMMENT("Infer shape for call " << func.getName());
  Operation* op = call_op.getOperation();
  bool changed = false;
  // Map each of the results of the call to the returned type of the
  // function.
  for (auto result : zip(op->getResults(), func.getType().getResults())) {
    changed = RefineResultType(op, std::get<0>(result), std::get<1>(result)) ||
              changed;
  }
  DCOMMENT(" - call " << func.getName() << "changed ? " << changed << "\n");

  return changed;
}

bool ShapeInference::InferShapeForCast(Operation* op) {
  DCOMMENT_OP(op, "Inferring shape for ");
  Value result = op->getResult(0);
  if (!CanBeRefined(result.getType())) return false;

  Type operand_type = op->getOperand(0).getType();
  auto ranked_op_type = operand_type.dyn_cast<RankedTensorType>();
  if (!ranked_op_type) return false;
  auto ranked_res_type = result.getType().dyn_cast<RankedTensorType>();
  if (ranked_res_type &&
      ranked_op_type.getShape() == ranked_res_type.getShape())
    return false;

  // Avoid inserting a cast where no users types could be refined (e.g., where
  // there would need to be a cast inserted for every user again).
  if (llvm::all_of(result.getUses(), [this](OpOperand& use) {
        return NeedsCastBack(use, tf_dialect_);
      }))
    return false;

  auto new_type = RankedTensorType::get(
      ranked_op_type.getShape(),
      result.getType().cast<ShapedType>().getElementType());

  return UpdateTypeAndInsertIncompatibleUseCasts(new_type, op->getResult(0));
}

bool ShapeInference::InferShapeForIf(IfOp op) {
  DCOMMENT_OP(op.getOperation(), "Infer shape for if ");
  bool changed = false;
  auto then_results = op.then_function().getType().getResults();
  auto else_results = op.else_function().getType().getResults();
  for (auto it : llvm::zip(op.getResults(), then_results, else_results)) {
    // If then and else types do not match, skip refinement for that result.
    if (std::get<1>(it) != std::get<2>(it)) continue;
    changed = RefineResultType(op, std::get<0>(it), std::get<1>(it)) || changed;
  }
  return changed;
}

bool ShapeInference::InferShapeForIfRegion(IfRegionOp op) {
  bool changed = false;

  Operation* then_yield = op.then_branch().front().getTerminator();
  Operation* else_yield = op.else_branch().front().getTerminator();
  for (auto result : zip(op.getResults(), then_yield->getOperandTypes(),
                         else_yield->getOperandTypes())) {
    // If then and else types do not match, skip refinement for that result.
    if (std::get<1>(result) != std::get<2>(result)) continue;
    changed = RefineResultType(op, std::get<0>(result), std::get<1>(result)) ||
              changed;
  }
  return changed;
}

bool ShapeInference::InferShapeForXlaHostComputeMlir(
    _XlaHostComputeMlirOp host_compute_op) {
  // Extract the module and function.
  // The '_XlaHostComputeMlir` verifier verifies that `host_mlir_module`
  // attribute is well formed, so we just return in case of an error in
  // extracting the host function since it should never occur.
  StringAttr host_module =
      host_compute_op->getAttrOfType<StringAttr>("host_mlir_module");
  if (host_module.getValue().empty()) return false;

  mlir::OwningModuleRef module_for_func;
  FuncOp func = host_compute_op.GetHostFunc(&module_for_func);

  // Update/use input shapes for function.
  FunctionType func_type = func.getType();
  func.setType(FunctionType::get(func.getContext(),
                                 host_compute_op.getOperandTypes(),
                                 func_type.getResults()));

  // Run shape inference on the function.
  if (failed(PropagateShapeToRegions(host_compute_op.getOperandTypes(),
                                     {&func.getBody()}, 10)))
    return false;
  if (failed(InferShapeForFunctionReturnType(func))) return false;

  bool changed = false;
  // Use refined function return shape for XlaHostComputeMlirOp.
  for (auto result :
       zip(host_compute_op.getResults(), func.getType().getResults())) {
    changed = RefineResultType(host_compute_op, std::get<0>(result),
                               std::get<1>(result)) ||
              changed;
  }

  return changed;
}

bool ShapeInference::InferShapeForTensorListInitOps(Operation* op) {
  DCOMMENT_OP(op, "Inferring shape for TensorList ");
  Value handle = op->getResult(0);
  Value initial_element_shape = GetElementShapeOperand(op);
  RankedTensorType element_type;
  if (auto tl_from_tensor = dyn_cast<TensorListFromTensorOp>(op)) {
    // For TensorListFromTensor op we can infer element shape by dropping the
    // first dimension of input tensor.
    element_type = DropFirstDimension(tl_from_tensor.tensor().getType());
    if (!element_type || !element_type.hasStaticShape()) return false;
  }
  if (!CanInferTensorListElementType(handle, initial_element_shape,
                                     &element_type)) {
    DCOMMENT("InferShapeForListInitOps " << op << " could not infer");
    return false;
  }
  DCOMMENT("InferShapeForListInitOps " << *op << " could be inferred "
                                       << element_type);
  if (!element_type || !element_type.hasStaticShape()) return false;
  auto variant_type = VariantType::get(element_type, op->getContext());
  auto tensor_type = RankedTensorType::get({}, variant_type);
  bool changed = RefineResultType(op, handle, tensor_type);
  if (changed) DCOMMENT_OP(op, "Modified after shape inference:");
  return changed;
}

bool ShapeInference::RefineWithInferTypeOpInterface(
    InferTypeOpInterface infer_ti) {
  Operation* op = infer_ti.getOperation();
  SmallVector<Type, 4> inferred;
  LogicalResult res = infer_ti.inferReturnTypes(
      op->getContext(), op->getLoc(), op->getOperands(),
      op->getAttrDictionary(), op->getRegions(), inferred);
  if (failed(res)) {
    op->emitOpError("failed to refine type as inference failed");
    return false;
  }

  if (inferred == op->getResultTypes()) return false;

  // Map each of the results of the call to the returned type of the
  // function.
  bool changed = false;
  for (auto result : zip(op->getResults(), inferred)) {
    if (std::get<0>(result).getType() == std::get<1>(result)) continue;

    if (!UpdateTypeAndInsertIncompatibleUseCasts(std::get<1>(result),
                                                 std::get<0>(result)))
      continue;
    changed = true;
  }
  return changed;
}

ShapeHandle ShapeInference::ComputeOutputAsShape(OpResult result,
                                                 InferenceContext* ic) {
  LLVM_DEBUG(result.print(llvm::dbgs() << "\nEvaluate partially "));
  auto rt = result.getType().dyn_cast<RankedTensorType>();
  if (!rt || !rt.hasStaticShape() || rt.getRank() != 1) return {};
  int dim_size = rt.getDimSize(0);

  // Worklist to direct partial evaluation.
  SmallVector<ValuePort, 4> worklist;

  // Simple evaluator that attempts to partially evaluate the input value even
  // if unable to evaluate the complete output. Below follows a simple stack
  // based evaluation where it queries what operands/part of operands need to
  // be evaluated and attempting to partially evaluate those operands. It does
  // so by pushing the operands that need to be required on to the worklist
  // before enqueuing the operation requiering those values.
  std::vector<DimensionHandle> dims(dim_size, ic->UnknownDim());
  for (unsigned int i = 0, e = dims.size(); i != e; ++i) {
    LLVM_DEBUG(llvm::dbgs() << "\nConsidering output dim " << i << "\n");

    worklist.push_back(
        ValuePort{result.getOwner(), {result.getResultNumber(), i}});
    while (!worklist.empty()) {
      auto front = worklist.pop_back_val();
      LLVM_DEBUG(front.print(llvm::dbgs() << "\nWorklist front "));

      SmallVector<ValuePort, 4> inputs;
      auto res = ComputeInputsRequiredForOutput(front, &inputs);
      if (failed(res)) {
        // Abort if unable to find which required inputs need to be computed.
        worklist.clear();
        break;
      }

      if (!inputs.empty()) {
        // Enqueue required computation followed by its required operands in
        // stack.
        worklist.push_back(std::move(front));
        for (auto& it : inputs) worklist.push_back(std::move(it));
        continue;
      }

      auto ret = ComputeOutputComponent(front);
      if (!ret) continue;

      LLVM_DEBUG(ret.print(llvm::dbgs() << "\ncomputed result = "));

      // If worklist is empty, then this is the root query op.
      if (worklist.empty()) {
        LLVM_DEBUG(llvm::dbgs() << "[root node]\n");
        if (auto dea = ret.dyn_cast<DenseIntElementsAttr>()) {
          if (dea.getNumElements() != 1) {
            LLVM_DEBUG(llvm::dbgs() << "Unexpected number of elements\n");
            return {};
          }
          int64_t val = (*dea.getIntValues().begin()).getSExtValue();
          dims[i] = ic->MakeDim(val);
        }
      }
    }
  }
  return ic->MakeShape(dims);
}

bool ShapeInference::RefineTypeForPassThroughOperands(Operation* op,
                                                      OperandRange operands,
                                                      ResultRange results) {
  bool changed = false;
  for (auto entry : llvm::zip(operands, results)) {
    Type operand_type = std::get<0>(entry).getType();
    Value result = std::get<1>(entry);
    TensorType result_type = result.getType().cast<TensorType>();
    Type inferred_type = TypeMeet(result_type, operand_type);
    if (result_type == inferred_type) continue;

    if (!UpdateTypeAndInsertIncompatibleUseCasts(inferred_type, result))
      continue;
    changed = true;
  }
  return changed;
}

bool ShapeInference::RefineShapeForPassThroughOps(Operation* op) {
  DCOMMENT_OP(op, "Pass through op");
  bool changed = false;
  for (auto entry : llvm::zip(op->getOperands(), op->getResults())) {
    Value operand = std::get<0>(entry);
    Value result = std::get<1>(entry);
    Type inferred_type = TypeMeet(result.getType(), operand.getType());
    if (result.getType() == inferred_type) continue;
    if (!UpdateTypeAndInsertIncompatibleUseCasts(inferred_type, result))
      continue;
    changed = true;
  }
  return changed;
}

bool ShapeInference::InferShapeForNonTFDialectOperation(Operation* op) {
  if (auto graph_op = dyn_cast<tf_executor::GraphOp>(op)) {
    return RefineTypeForPassThroughOperands(
        graph_op.GetFetch(), graph_op.GetFetch().fetches(), op->getResults());
  }
  if (auto island_op = dyn_cast<tf_executor::IslandOp>(op)) {
    return RefineTypeForPassThroughOperands(
        island_op.GetYield(), island_op.GetYield().fetches(), op->getResults());
  }
  if (auto iter_sink = dyn_cast<tf_executor::NextIterationSinkOp>(op)) {
    auto iter_source = cast<tf_executor::NextIterationSourceOp>(
        iter_sink.token().getDefiningOp());
    return RefineTypeForPassThroughOperands(
        op, iter_sink.getOperands().drop_front().take_front(),
        iter_source.getResults());
  }
  if (auto launch_op = dyn_cast<tf_device::LaunchOp>(op)) {
    auto terminator = launch_op.GetBody().getTerminator();
    return RefineTypeForPassThroughOperands(op, terminator->getOperands(),
                                            op->getResults());
  }
  if (auto cluster_op = dyn_cast<tf_device::ClusterOp>(op)) {
    auto terminator = cluster_op.GetBody().getTerminator();
    return RefineTypeForPassThroughOperands(op, terminator->getOperands(),
                                            op->getResults());
  }
  if (op->hasTrait<OpTrait::SameOperandsAndResultShape>())
    return RefineShapeForPassThroughOps(op);
  if (auto call = dyn_cast<CallOpInterface>(op)) return InferShapeForCall(call);
  return false;
}

// Finds element type to be used for result from operand, with special handling
// for handle types.
Type GetElementTypeFromOperand(TensorType operand_type,
                               TensorType result_type) {
  auto operand_handle_type =
      operand_type.getElementType().dyn_cast<TensorFlowTypeWithSubtype>();
  if (!operand_handle_type) return result_type.getElementType();
  auto result_handle_type =
      result_type.getElementType().cast<TensorFlowTypeWithSubtype>();
  if (operand_handle_type.GetSubtypes().empty() ||
      !result_handle_type.GetSubtypes().empty())
    return result_type.getElementType();
  return operand_handle_type;
}

// Checks if one tensor type can refine another type for tf.While/
// tf.WhileRegion. If rank differs or static dimensions can be lost, the other
// type cannot be used for refinement.
bool CanWhileTypeBeRefinedWith(TensorType current_type,
                               TensorType potential_refined_type) {
  if (!current_type.hasRank()) return true;
  if (!potential_refined_type.hasRank()) return false;
  if (current_type.getRank() != potential_refined_type.getRank()) return false;
  for (auto dim :
       llvm::zip(current_type.getShape(), potential_refined_type.getShape())) {
    int64_t current_dim = std::get<0>(dim);
    int64_t potential_refined_dim = std::get<1>(dim);
    if (current_dim != potential_refined_dim &&
        current_dim != ShapedType::kDynamicSize)
      return false;
  }
  return true;
}

template <typename WhileOpTy>
bool ShapeInference::InferShapeForWhile(WhileOpTy op,
                                        TypeRange body_result_types) {
  if (!op.shape_invariant())
    return RefineTypeForPassThroughOperands(op, op.input(), op.output());

  bool changed = false;
  for (auto entry :
       zip(op.input().getTypes(), op.output(), body_result_types)) {
    Value result = std::get<1>(entry);
    TensorType body_result_type =
        std::get<2>(entry).template cast<TensorType>();
    auto result_type = result.getType().cast<TensorType>();

    Type potential_refined_type;
    if (CanWhileTypeBeRefinedWith(result_type, body_result_type)) {
      Type element_type =
          GetElementTypeFromOperand(body_result_type, result_type);
      potential_refined_type = CreateTensorType(
          body_result_type.hasRank() ? body_result_type.getShape()
                                     : llvm::Optional<ArrayRef<int64_t>>(),
          element_type);
    } else {
      TensorType operand_type = std::get<0>(entry).template cast<TensorType>();
      Type element_type = GetElementTypeFromOperand(operand_type, result_type);
      potential_refined_type = CreateTensorType(
          result_type.hasRank() ? result_type.getShape()
                                : llvm::Optional<ArrayRef<int64_t>>(),
          element_type);
    }
    changed |= RefineResultType(op, result, potential_refined_type);
  }
  return changed;
}

bool ShapeInference::InferShapeForSingleOperation(Operation* op) {
  LLVM_DEBUG(op->print(llvm::dbgs() << "InferShapeForSingleOperation for ");
             llvm::dbgs() << "\n");
  assert(tf_dialect_ == op->getDialect());
  // The shape function of these ops sometimes does not propagate subtypes
  // (handle shapes) for resource and variant types. We use a simple passthrough
  // to make sure they are preserved in the output.
  if (isa<TF::IdentityOp, TF::IdentityNOp, TF::StopGradientOp, TF::ZerosLikeOp>(
          op)) {
    return RefineTypeForPassThroughOperands(op, op->getOperands(),
                                            op->getResults());
  }

  // If no result for this op needs shape inference, we have a fast-path return.
  // But if the type is a resource/variant, we do not skip it because we might
  // not have the handle shapes.
  if (none_of(op->getResultTypes(), CanBeRefined)) {
    LLVM_DEBUG(llvm::dbgs() << "Skipping inference for statically shaped op '"
                            << op->getName() << "'.\n");
    return false;
  }

  // Handle call operations by looking up callee and inferring return shape as
  // needed.
  if (auto call = dyn_cast<CallOpInterface>(op)) return InferShapeForCall(call);

  // tf.Cast and tensor::Cast are only inferred if they have at least one user
  // in the TF dialect or feeding into the function return. This is necessary to
  // avoid inserting casts which cannot be refined.
  if (isa<CastOp, tensor::CastOp>(op)) return InferShapeForCast(op);

  // Handle IfOp here by inferring the shape from the else/then function
  // results. Since `output_shapes` is a derived attribute, avoid going down the
  // TF InferenceContext path as IfOp shape inference is implemented as just
  // a lookup of the output_shapes attribute.
  if (auto if_op = dyn_cast<IfOp>(op)) return InferShapeForIf(if_op);

  // Handle IfRegion operations by inferring return shape from the then and else
  // branches.
  if (auto if_region = dyn_cast<IfRegionOp>(op))
    return InferShapeForIfRegion(if_region);

  if (auto while_op = dyn_cast<WhileOp>(op))
    return InferShapeForWhile(while_op,
                              while_op.body_function().getType().getResults());

  if (auto while_region = dyn_cast<WhileRegionOp>(op))
    return InferShapeForWhile(
        while_region,
        while_region.body().front().getTerminator()->getOperandTypes());

  if (auto host_compute_op = dyn_cast<_XlaHostComputeMlirOp>(op)) {
    return InferShapeForXlaHostComputeMlir(host_compute_op);
  }

  // Handle TensorList init operations by inferring shape from TensorList write
  // operations. If we are unable to refine element shape here, proceed to use
  // the InferenceContext below to get more precise shapes.
  if (IsTensorListInitOp(op) && InferShapeForTensorListInitOps(op)) return true;

  // Return operand as a constant attribute.
  auto operand_as_constant_fn = [&](Value operand) {
    ValuePort vp(operand);
    Attribute attr = ComputeOutputComponent(vp);
    if (!attr && matchPattern(operand, m_Constant(&attr)))
      RecordValue(vp, attr);
    return attr;
  };

  // Return op result as a shape.
  auto op_result_as_shape_fn = [&](InferenceContext& context,
                                   OpResult op_result) {
    return ComputeOutputAsShape(op_result, &context);
  };

  // Return result element type at `index`.
  auto result_element_type_fn = [&](int index) {
    return op->getResult(index).getType().cast<TensorType>().getElementType();
  };

  llvm::SmallVector<ShapedTypeComponents, 4> inferred_return_shapes;
  if (failed(InferReturnTypeComponentsForTFOp(
          /*location=*/None, op, graph_version_, operand_as_constant_fn,
          op_result_as_shape_fn, result_element_type_fn,
          inferred_return_shapes)))
    return false;

  // Update the shape for each of the operation result if the InferenceContext
  // has more precise shapes recorded.
  bool changed = false;
  for (auto result : llvm::zip(op->getResults(), inferred_return_shapes)) {
    Value op_result = std::get<0>(result);
    if (!CanBeRefined(op_result.getType())) continue;

    ShapedTypeComponents inferred = std::get<1>(result);
    TensorType inferred_type;
    if (inferred.hasRank())
      inferred_type =
          RankedTensorType::get(inferred.getDims(), inferred.getElementType());
    else
      inferred_type = UnrankedTensorType::get(inferred.getElementType());

    inferred_type =
        TypeMeet(op_result.getType(), inferred_type).cast<TensorType>();
    if (op_result.getType() == inferred_type) continue;
    if (!UpdateTypeAndInsertIncompatibleUseCasts(inferred_type, op_result))
      continue;
    changed = true;
  }

  if (changed) DCOMMENT_OP(op, "Modified after shape inference:");
  return changed;
}

FailureOr<bool> ShapeInference::PropagateShapeToFunctions(
    ModuleOp module, TypeRange input_types, ArrayRef<FuncOp> functions,
    int64_t max_iteration) {
  bool any_failure = false;
  bool any_nonconvergence = false;
  // If shape propagation fails for one function, return failure, but do not
  // early exit and attempt to propagate shapes for all provided functions to
  // have a best-effort propagation.
  for (FuncOp func : functions) {
    DCOMMENT("Propating shape to " << func.getName());
    ArrayRef<Operation*> callers = GetCallers(func);
    if (!llvm::hasSingleElement(callers) &&
        !llvm::all_of(callers.drop_front(), [&](Operation* caller) {
          /// TODO(aminim): this is overly conservative as some operations
          /// (like TPUPartitionedCallOp) may have extra operands that aren't
          /// propagated to the callee.
          return isa<CallOpInterface>(caller) &&
                 std::equal(caller->getOperandTypes().begin(),
                            caller->getOperandTypes().end(),
                            callers.front()->getOperandTypes().begin());
        })) {
      if (llvm::any_of(callers, [](Operation* op) {
            return isa<IfOp, WhileOp, CaseOp>(op);
          }))
        func.emitWarning(formatv(
            "expected control flow function @{0} to have exactly 1 use, "
            "found {1}.",
            func.getName(), callers.size()));

      continue;
    }
    FunctionType func_type = func.getType();
    func.setType(FunctionType::get(func.getContext(), input_types,
                                   func_type.getResults()));

    FailureOr<bool> failure_or_converged =
        PropagateShapeToRegions(input_types, {&func.getBody()}, max_iteration);
    if (failed(failure_or_converged)) {
      any_failure = true;
      continue;
    }
    any_nonconvergence = any_nonconvergence || !failure_or_converged.getValue();
    if (failed(InferShapeForFunctionReturnType(func))) any_failure = true;
  }
  if (any_failure) return failure();
  return any_nonconvergence;
}

FailureOr<bool> ShapeInference::PropagateShapeToRegions(
    TypeRange input_types, ArrayRef<Region*> regions, int64_t max_iteration) {
  DCOMMENT("\tPropagating shapes to regions");
  bool any_failure = false;
  bool any_nonconvergence = false;
  // If shape propagation fails for one region, return failure, but do not
  // early exit and attempt to propagate shapes for all provided regions to
  // have a best-effort propagation.
  for (auto region : regions) {
    // Refine region arguments.
    Block& entry = region->front();
    assert(llvm::size(input_types) == entry.getNumArguments());
    for (auto it : llvm::zip(entry.getArguments(), input_types)) {
      BlockArgument arg = std::get<0>(it);
      Type type = std::get<1>(it);
      arg.setType(type);
    }

    // Propagate shapes into the region.
    FailureOr<bool> failure_or_converged =
        InferShapeUntilFixPoint(region, max_iteration);
    if (failed(failure_or_converged))
      any_failure = true;
    else if (!failure_or_converged.getValue())
      any_nonconvergence = true;
  }
  if (any_failure) return failure();
  return any_nonconvergence;
}

void ShapeInference::PropagateConstantToCallee(CallOpInterface call_op,
                                               FuncOp func, ModuleOp module) {
  auto callers = GetCallers(func);
  if (!llvm::hasSingleElement(callers)) return;

  OpBuilder builder(&func.front().front());
  Operation* op = call_op.getOperation();
  // If this is the only caller, and an operand is a constant, propagate
  // the constant value inside the function.
  for (auto arg : func.getArguments()) {
    auto operand = op->getOperand(arg.getArgNumber());
    if (propagate_caller_callee_constants_) {
      if (isa_and_nonnull<TF::ConstOp>(operand.getDefiningOp())) {
        arg.replaceAllUsesWith(
            builder.clone(*operand.getDefiningOp())->getResult(0));
      }
      continue;
    }

    auto known_constant = ComputeOutputComponent(ValuePort(operand));
    if (!known_constant) continue;
    LLVM_DEBUG(call_op.print(llvm::dbgs() << "Propagate to calee: ");
               known_constant.print(llvm::dbgs() << " constant ");
               llvm::dbgs() << "\n");
    RecordValue(ValuePort(arg), known_constant);
  }
}

void ShapeInference::PropagateConstantFromCallee(CallOpInterface call_op,
                                                 FuncOp func, ModuleOp module) {
  // If the return value is a constant, use the constant as the value of
  // the call return.
  Operation* op = call_op.getOperation();
  OpBuilder builder(op);
  builder.setInsertionPointAfter(op);
  for (auto retval :
       llvm::enumerate(func.front().getTerminator()->getOperands())) {
    if (propagate_caller_callee_constants_) {
      auto retval_op = retval.value().getDefiningOp();
      if (isa_and_nonnull<TF::ConstOp>(retval_op)) {
        op->getResult(retval.index())
            .replaceAllUsesWith(builder.clone(*retval_op)->getResult(0));
      }
      continue;
    }

    ValuePort vp(retval.value());
    if (auto known_constant = ComputeOutputComponent(vp)) {
      LLVM_DEBUG(known_constant.print(llvm::dbgs() << "Propagate constant ");
                 call_op.print(llvm::dbgs() << "from "); llvm::dbgs() << "\n");
      RecordValue(ValuePort(op->getResult(retval.index())), known_constant);
    }
  }
}

bool RankedAndSameRank(TensorType lhs, TensorType rhs) {
  return lhs.hasRank() && rhs.hasRank() && lhs.getRank() == rhs.getRank();
}

// Creates a compatible RankedTensorType where mismatched dimensions are
// replaced with dynamic sizes.
RankedTensorType GetCompatibleRankedTensorType(RankedTensorType lhs,
                                               RankedTensorType rhs) {
  assert(lhs.getRank() == rhs.getRank());
  llvm::SmallVector<int64_t, 4> dims;
  dims.reserve(lhs.getRank());
  for (auto dim : llvm::zip(lhs.getShape(), rhs.getShape())) {
    int64_t lhs_dim = std::get<0>(dim);
    if (lhs_dim == std::get<1>(dim)) {
      dims.push_back(lhs_dim);
    } else {
      dims.push_back(ShapedType::kDynamicSize);
    }
  }
  return RankedTensorType::get(dims, GetElementTypeFromOperand(lhs, rhs));
}

// Finds compatible types to propagate into functions/regions of a shape
// invariant tf.While/tf.WhileRegion. If operand and result types are the same,
// that type is returned. If operand and result types are of the same rank, a
// compatible type with matching dimensions is used. Otherwise functions/regions
// arguments are returned but with the handle type from the operand type.
llvm::SmallVector<Type, 4> GetWhileCompatibleTypes(
    TypeRange operand_types, TypeRange result_types,
    TypeRange region_argument_types) {
  llvm::SmallVector<Type, 4> types;
  types.reserve(operand_types.size());
  for (auto entry :
       llvm::zip(operand_types, result_types, region_argument_types)) {
    auto operand_type = std::get<0>(entry).cast<TensorType>();
    auto result_type = std::get<1>(entry).cast<TensorType>();
    if (operand_type == result_type) {
      types.push_back(operand_type);
    } else if (RankedAndSameRank(operand_type, result_type)) {
      auto potential_refined_type =
          GetCompatibleRankedTensorType(operand_type.cast<RankedTensorType>(),
                                        result_type.cast<RankedTensorType>());
      types.push_back(potential_refined_type);
    } else {
      auto region_argument_type = std::get<2>(entry).cast<TensorType>();
      Type element_type = GetElementTypeFromOperand(
          operand_type.cast<TensorType>(), region_argument_type);
      Type potential_refined_type = CreateTensorType(
          region_argument_type.hasRank() ? region_argument_type.getShape()
                                         : llvm::Optional<ArrayRef<int64_t>>(),
          element_type);
      types.push_back(potential_refined_type);
    }
  }
  return types;
}

FailureOr<bool> ShapeInference::PropagateShapeIntoAttachedFunctions(
    Operation* op, int64_t max_iteration) {
  ModuleOp module = op->getParentOfType<ModuleOp>();
  if (auto if_op = dyn_cast<TF::IfOp>(op)) {
    DCOMMENT("Propagating shapes into If");
    return PropagateShapeToFunctions(
        module, if_op.input().getTypes(),
        {if_op.then_function(), if_op.else_function()}, max_iteration);
  } else if (auto case_op = dyn_cast<TF::CaseOp>(op)) {
    SmallVector<FuncOp, 4> branches;
    case_op.get_branch_functions(branches);
    return PropagateShapeToFunctions(module, case_op.input().getTypes(),
                                     branches, max_iteration);
  } else if (auto while_op = dyn_cast<TF::WhileOp>(op)) {
    // If `shape_invariant` is set, operand shapes cannot be simply propagated
    // to result shapes as the op may have different intermediate shapes (such
    // While ops can have different result shapes from operand shapes).
    // Compatible shapes must be determined before propagating them.
    if (while_op.shape_invariant()) {
      auto compatible_types = GetWhileCompatibleTypes(
          while_op.input().getTypes(), while_op.output().getTypes(),
          while_op.body_function().getType().getInputs());
      return PropagateShapeToFunctions(
          module, compatible_types,
          {while_op.cond_function(), while_op.body_function()}, max_iteration);
    }
    return PropagateShapeToFunctions(
        module, while_op.input().getTypes(),
        {while_op.cond_function(), while_op.body_function()}, max_iteration);
  } else if (auto call_op = dyn_cast<CallOpInterface>(op)) {
    if (auto func = dyn_cast<FuncOp>(call_op.resolveCallable())) {
      PropagateConstantToCallee(call_op, func, module);
      FailureOr<bool> failure_or_converged = PropagateShapeToFunctions(
          module, call_op.getArgOperands().getTypes(), {func}, max_iteration);
      if (failed(failure_or_converged)) return failure();
      PropagateConstantFromCallee(call_op, func, module);
      return failure_or_converged;
    }
  }

  // TODO(ycao): Implement support for Call op, including function reuse.

  return true;
}

FailureOr<bool> ShapeInference::PropagateShapeIntoAttachedRegions(
    Operation* op, int64_t max_iteration) {
  if (auto while_op = dyn_cast<TF::WhileRegionOp>(op)) {
    // If `shape_invariant` is set, operand shapes cannot be simply propagated
    // to result shapes as the op may have different intermediate shapes (such
    // While ops can have different result shapes from operand shapes).
    // Compatible shapes must be determined before propagating them.
    if (while_op.shape_invariant()) {
      auto compatible_types = GetWhileCompatibleTypes(
          while_op.input().getTypes(), while_op.output().getTypes(),
          while_op.body().getArgumentTypes());
      return PropagateShapeToRegions(compatible_types,
                                     {&while_op.cond(), &while_op.body()},
                                     max_iteration);
    }
    return PropagateShapeToRegions(while_op.input().getTypes(),
                                   {&while_op.cond(), &while_op.body()},
                                   max_iteration);
  }
  return true;
}

LogicalResult ShapeInference::TryToFold(Operation* op) {
  LLVM_DEBUG(op->print(llvm::dbgs() << "TryToFold "); llvm::dbgs() << "\n");
  // If any output result is known, then the op probably has been computed
  // before.
  if (op->getNumResults() > 0 && results_[ValuePort(op->getResult(0))])
    return success();

  SmallVector<Attribute, 8> constant_operands(op->getNumOperands());
  SmallVector<OpFoldResult, 8> fold_results;

  // Check to see if any operands to the operation is constant and whether
  // the operation knows how to constant fold itself.
  bool some_unknown = false;
  for (int i = 0, e = op->getNumOperands(); i != e; ++i) {
    if (!(constant_operands[i] =
              ComputeOutputComponent(ValuePort(op->getOperand(i)))))
      some_unknown = true;
  }

  // Attempt to constant fold the operation.
  auto* abstract_op = op->getAbstractOperation();
  LogicalResult folded = failure();
  if (abstract_op) {
    folded = abstract_op->foldHook(op, constant_operands, fold_results);
  }
  // Attempt dialect fallback if op's fold hook failed.
  if (failed(folded)) {
    Dialect* dialect = op->getDialect();
    if (!dialect) return failure();
    // Only attempt TF dialect fallback if there are no unknown operands.
    if (some_unknown && dialect == tf_dialect_) return failure();
    auto* interface = dialect->getRegisteredInterface<DialectFoldInterface>();
    if (!interface) return failure();

    if (failed(interface->fold(op, constant_operands, fold_results)))
      return failure();
  }

  for (auto result : zip(op->getResults(), fold_results)) {
    auto fold_result = std::get<1>(result);
    Attribute attr = nullptr;
    if ((attr = fold_result.dyn_cast<Attribute>())) {
      RecordValue(ValuePort(std::get<0>(result)), attr);
    } else {
      auto value = fold_result.get<Value>();
      if ((attr = ComputeOutputComponent(ValuePort(value)))) {
        DCOMMENT("\t\tValue Result mapped to " << attr);
        RecordValue(ValuePort(std::get<0>(result)), attr);
      } else {
        DCOMMENT("\t\tValue result unmapped, consider value type:" << value);
        RefineResultType(op, std::get<0>(result), value.getType());
      }
    }

    if (ElementsAttr eattr = attr.dyn_cast_or_null<ElementsAttr>()) {
      if (std::get<0>(result).getType() == eattr.getType()) continue;

      (void)UpdateTypeAndInsertIncompatibleUseCasts(eattr.getType(),
                                                    std::get<0>(result));
    }
  }

  return success();
}

LogicalResult ShapeInference::InferShapeForFunctionReturnType(FuncOp func) {
  LLVM_DEBUG(llvm::dbgs() << "Inferring return type for: " << func.getName()
                          << "\n");

  // Find any return ops.
  SmallVector<ReturnOp, 4> return_ops;
  for (Block& block : func) {
    if (auto return_op = dyn_cast<ReturnOp>(block.getTerminator())) {
      return_ops.push_back(return_op);
    }
  }

  // Skip functions without a return, but don't flag as failure here.
  if (return_ops.empty()) return success();

  // Right now we only handle the case of a single return op.
  // To handle multiple return ops, we would need to look at all their shapes
  // and come up with a common shape and insert appropriate casts.
  if (return_ops.size() != 1) return failure();

  // Find the return type.
  auto return_op = return_ops.front();

  // Manually fold tf.Cast that precedes the return instruction and only differs
  // in shape refinement level.
  bool changed = false;
  for (OpOperand& arg_op : return_op.getOperation()->getOpOperands()) {
    Operation* arg_defining_op = arg_op.get().getDefiningOp();
    if (isa_and_nonnull<CastOp, tensor::CastOp>(arg_defining_op)) {
      Value input = arg_defining_op->getOperand(0);
      Value result = arg_defining_op->getResult(0);
      Type meet = TypeMeet(result.getType(), input.getType());
      if (meet == result.getType()) continue;

      LLVM_DEBUG({
        llvm::errs() << "\tfolding & updating return type ";
        result.getType().print(llvm::errs());
        input.getType().print(llvm::errs() << " to ");
        llvm::errs() << "\n";
      });

      // Shape inference should not change the element type.
      if (HasCompatibleElementTypes(input.getType(), result.getType()) &&
          meet == input.getType()) {
        arg_op.set(input);
      } else {
        OpBuilder b(return_op.getOperation());
        auto new_cast_op = InsertCast(b, return_op.getLoc(), meet, input);
        if (!new_cast_op) return failure();
        arg_op.set(new_cast_op->getResult(0));
      }
      if (result.use_empty()) arg_defining_op->erase();
      changed = true;
    }
  }

  DCOMMENT("Updating function type");
  func.setType(FunctionType::get(func.getContext(), func.getArgumentTypes(),
                                 return_op.getOperandTypes()));

  if (changed) EnqueueCallers(func);
  return success();
}

FailureOr<bool> ShapeInference::InferShapeUntilFixPoint(Region* region,
                                                        int64_t max_iteration) {
  bool changed = true;

  // TODO(aminim): we could have a more efficient traversal by guiding the
  // traversal with a worklist and reconsider only the nodes for which an
  // operand type was inferred. This would need to be careful if working on a
  // region that would not be isolated.
  for (int iteration = 0; iteration < max_iteration && changed; ++iteration) {
    changed = false;
    LLVM_DEBUG(llvm::dbgs()
               << "Shape inference, iteration " << iteration << "\n");
    auto res = region->walk([&](Operation* op) {
      DCOMMENT_OP(op, "Inferring for");
      if (auto infer_ti = dyn_cast<InferTypeOpInterface>(op)) {
        DCOMMENT("\tRefinining with type op interface");
        changed |= RefineWithInferTypeOpInterface(infer_ti);
        return WalkResult::advance();
      }

      if (op->getDialect() != tf_dialect_) {
        DCOMMENT("\tInfer non-TF dialect");
        changed |= InferShapeForNonTFDialectOperation(op);
        return WalkResult::advance();
      }

      // Before attempting inference, just try to compute the folded
      // value/shape.
      if (succeeded(TryToFold(op)) &&
          // Folding can "succeed" and yet not all types be refined. In such
          // cases we still want to give a try at `InferShapeForSingleOperation`
          none_of(op->getResultTypes(), CanBeRefined))
        return WalkResult::advance();

      // Best-effort shape inference in attached functions. Do not return
      // failure even if it doesn't get to fixed point, but propagate "real"
      // failure.
      if (failed(PropagateShapeIntoAttachedFunctions(op, max_iteration))) {
        op->emitWarning() << "unable to refine shape of attached function "
                             "arguments and bodies";
        return WalkResult::interrupt();
      }

      if (failed(PropagateShapeIntoAttachedRegions(op, max_iteration))) {
        op->emitWarning() << "unable to refine shape of attached region "
                             "arguments and bodies";
        return WalkResult::interrupt();
      }

      changed |= InferShapeForSingleOperation(op);
      return WalkResult::advance();
    });
    if (res.wasInterrupted()) return failure();
  }

  if (changed) {
    region->getParentOp()->emitWarning()
        << "shape inference did not reach stable state after " << max_iteration
        << " iterations";
  }
  return !changed;
}

static FailureOr<bool> InferShapeForFunction(ShapeInference& context,
                                             FuncOp func,
                                             int64_t max_iterations) {
  FailureOr<bool> failure_or_converged =
      context.InferShapeUntilFixPoint(&func.getBody(), max_iterations);
  if (failed(failure_or_converged) || !failure_or_converged.getValue())
    return failure_or_converged;
  // TODO(b/156276510): Verify that it is always fine to refine a function's
  // return type, as long as we do not change the argument shapes.
  if (failed(context.InferShapeForFunctionReturnType(func))) return failure();
  return true;
}

FailureOr<bool> InferShapeForFunction(FuncOp func,
                                      ArrayRef<ArrayRef<int64_t>> arg_shapes,
                                      int64_t graph_version,
                                      int64_t max_iterations) {
  ShapeInference context(graph_version, func->getParentOfType<ModuleOp>(),
                         /*propagate_caller_callee_constants=*/true);
  if (arg_shapes.empty()) {
    return InferShapeForFunction(context, func, max_iterations);
  }

  FunctionType func_type = func.getType();
  bool needs_refinement = false;
  SmallVector<Type, 4> new_arg_types;
  new_arg_types.reserve(func_type.getNumInputs());

  // Update argument types in-place using the provided arg_shapes.
  for (size_t i = 0; i < func_type.getNumInputs(); ++i) {
    ArrayRef<int64_t> shape = arg_shapes[i];
    Type element_type;
    if (auto input_ty = func_type.getInput(i).dyn_cast<RankedTensorType>()) {
      if (input_ty.getRank() != shape.size()) {
        return failure();
      }
      element_type = input_ty.getElementType();
    } else {
      auto unranked_input_ty = func_type.getInput(i).dyn_cast<TensorType>();
      if (!unranked_input_ty) {
        return failure();
      }
      element_type = unranked_input_ty.getElementType();
    }

    auto new_arg_type = RankedTensorType::get(shape, element_type);
    if (new_arg_type != func_type.getInput(i)) {
      // If the new type is more detailed, trigger shape inference.
      func.getArgument(i).setType(new_arg_type);
      needs_refinement = true;
    }
    new_arg_types.push_back(new_arg_type);
  }

  if (!needs_refinement) return true;

  FailureOr<bool> failure_or_converged =
      context.InferShapeUntilFixPoint(&func.getBody(), max_iterations);
  if (failed(failure_or_converged) || !failure_or_converged.getValue())
    return failure_or_converged;

  if (failed(context.InferShapeForFunctionReturnType(func))) return failure();
  func.setType(FunctionType::get(func.getContext(), new_arg_types,
                                 func.getType().getResults()));

  return true;
}

FailureOr<bool> InferModuleShape(ModuleOp module, int64_t max_iterations) {
  auto producer_or = tensorflow::GetTfGraphProducerVersion(module);
  if (!producer_or.ok()) {
    // TODO(jpienaar): Keeping the existing behavior for now but this could
    // be relaxed.
    LLVM_DEBUG(llvm::dbgs()
               << "Skipping inference; " << producer_or.status().ToString());
    return true;
  }
  int64_t producer = producer_or.ValueOrDie();
  // TODO(jpienaar): Clean up propagate_NextIterationSinkOp_callee_constants if
  // it is no longer needed.
  ShapeInference context(producer, module,
                         /*propagate_caller_callee_constants=*/false);
  if (auto main = module.lookupSymbol<mlir::FuncOp>("main"))
    context.enqueue(main);
  for (auto func : module.getOps<FuncOp>()) context.enqueue(func);
  // Arbitrarily upper bound the maximum number of functions that get processed
  // just to avoid pathological cases.
  auto max_iteration = context.QueueSize() * 4;
  while (!context.EmptyQueue()) {
    FuncOp func = context.front();
    FailureOr<bool> failure_or_converged =
        InferShapeForFunction(context, func, max_iterations);
    if (failed(failure_or_converged) || !failure_or_converged.getValue())
      return failure_or_converged;
    context.pop_front();

    if ((--max_iteration) == 0) {
      emitWarning(UnknownLoc::get(module.getContext()))
          << "shape inference did not reach stable state after "
          << max_iteration << " iterations";
      return false;
    }
  }
  return true;
}

}  // namespace TF
}  // namespace mlir
