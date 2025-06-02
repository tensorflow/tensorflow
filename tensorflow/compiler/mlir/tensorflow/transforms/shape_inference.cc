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
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <optional>
#include <queue>
#include <stack>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/PointerUnion.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Func/Extensions/AllExtensions.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Block.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinDialect.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Diagnostics.h"  // from @llvm-project
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Matchers.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/OperationSupport.h"  // from @llvm-project
#include "mlir/IR/OwningOpRef.h"  // from @llvm-project
#include "mlir/IR/SymbolTable.h"  // from @llvm-project
#include "mlir/IR/TypeUtilities.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/IR/Visitors.h"  // from @llvm-project
#include "mlir/Interfaces/CallInterfaces.h"  // from @llvm-project
#include "mlir/Interfaces/FoldInterfaces.h"  // from @llvm-project
#include "mlir/Interfaces/FunctionInterfaces.h"  // from @llvm-project
#include "mlir/Interfaces/InferTypeOpInterface.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassRegistry.h"  // from @llvm-project
#include "mlir/Support/DebugStringHelper.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Support/TypeID.h"  // from @llvm-project
#include "mlir/Transforms/FoldUtils.h"  // from @llvm-project
#include "stablehlo/dialect/Serialization.h"  // from @stablehlo
#include "stablehlo/dialect/Version.h"  // from @stablehlo
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_executor.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/mlir_roundtrip_flags.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/tools/parsers.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/dynamic_shape_utils.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/serialize_mlir_module_utils.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/shape_inference_utils.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/translate_utils.h"
#include "tensorflow/compiler/tf2xla/kernels/xla_call_module_loader.h"
#include "xla/hlo/translate/hlo_to_mhlo/hlo_utils.h"
#include "xla/hlo/translate/mhlo_to_hlo/type_to_shape.h"
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"
#include "xla/service/shape_inference.h"
#include "xla/shape.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/util/env_var.h"
#include "xla/window_util.h"
#include "xla/xla_data.pb.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/ir/types/dialect.h"

#define DEBUG_TYPE "tf-shape-inference"

#define DCOMMENT(MSG) LLVM_DEBUG(llvm::dbgs() << MSG << "\n")
#define DCOMMENT_OP(OP, MSG) \
  LLVM_DEBUG(OP->print(llvm::dbgs() << MSG << " "); llvm::dbgs() << "\n")

using ::int64_t;
using mlir::func::FuncOp;
using tensorflow::shape_inference::DimensionHandle;
using tensorflow::shape_inference::InferenceContext;
using tensorflow::shape_inference::ShapeHandle;

namespace mlir {
namespace TF {
namespace {

MLIRContext::Threading GetMlirContextThreading() {
  bool enable_single_thread_mlir_context = []() {
    bool result = false;
    if (auto status = tsl::ReadBoolFromEnvVar(kMLIRContextSingleThreadVar,
                                              /*default_val=*/false, &result);
        status.ok()) {
      return result;
    }
    return false;
  }();
  return enable_single_thread_mlir_context ? MLIRContext::Threading::DISABLED
                                           : MLIRContext::Threading::ENABLED;
}

// Compute a refined type between two types `lhs` and `rhs`, the result type
// is always at least as refined as (i.e. has more static information) than
// `lhs` This method will actually merge the information contained in the types,
// it is capable of refining:
//   tensor<!tf_type.variant<tensor<?x8xf32>>>
// and:
//   tensor<!tf_type.variant<tensor<10x?xf32>>>
// into:
//   tensor<!tf_type.variant<tensor<10x8xf32>>>
//
// In case of inconsistencies (rank disagreement for example), it returns `lhs`.
Type TypeMeet(Type lhs, Type rhs) {
  DCOMMENT("RefineTypeWith : " << lhs << " : " << rhs);
  if (lhs == rhs) return lhs;

  auto rhs_shape_type = mlir::dyn_cast<ShapedType>(rhs);
  if (!rhs_shape_type) return lhs;
  auto lhs_shape_type = mlir::cast<ShapedType>(lhs);
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
      mlir::dyn_cast<TF::TensorFlowTypeWithSubtype>(
          rhs_shape_type.getElementType());
  // Look for resource or variant element type and ensure we refine the subtype.
  // We only support a single subtype at the moment, we won't handle something
  // like:
  //   tensor<!tf_type.variant<tensor<10xf32>, tensor<8xf32>>
  if (rhs_element_type_with_subtype &&
      rhs_element_type_with_subtype.GetSubtypes().size() == 1) {
    auto lhs_element_type_with_subtype =
        mlir::dyn_cast<TF::TensorFlowTypeWithSubtype>(lhs_element_type);
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
      //   tensor<!tf_type.variant<tensor<?x8xf32>>>
      // and:
      //   tensor<!tf_type.variant<tensor<10x8xf32>>>
      // we'll try here to refine tensor<?x8xf32> with tensor<10x8xf32>.
      auto refined_subtype = mlir::cast<TensorType>(
          TypeMeet(lhs_element_type_with_subtype.GetSubtypes().front(),
                   rhs_element_type_with_subtype.GetSubtypes().front()));
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

TensorType CreateTensorType(std::optional<llvm::ArrayRef<int64_t>> shape,
                            Type element_type) {
  if (shape.has_value())
    return tensorflow::GetTypeFromTFTensorShape(shape.value(), element_type);
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
    return empty_tl.getElementShape();
  if (auto tl_reserve = dyn_cast<TensorListReserveOp>(op))
    return tl_reserve.getElementShape();
  if (auto tl_from_tensor = dyn_cast<TensorListFromTensorOp>(op))
    return tl_from_tensor.getElementShape();
  llvm_unreachable("unsupported TensorList op");
}

// Utility function to create a ranked tensor type after dropping the first
// dimension from the input type.
RankedTensorType DropFirstDimension(Type type) {
  RankedTensorType ranked_type = mlir::dyn_cast<RankedTensorType>(type);
  if (!ranked_type) return {};
  llvm::ArrayRef<int64_t> dims_except_first =
      ranked_type.getShape().drop_front();
  return tensorflow::GetTypeFromTFTensorShape(dims_except_first,
                                              ranked_type.getElementType());
}

Operation* InsertCast(OpBuilder& b, Location loc, Type dst_type, Value input) {
  Type element_type = getElementTypeOrSelf(dst_type);
  if (mlir::isa<IndexType>(element_type))
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

  // Track the set of values we've already visited to avoid exponential blowup.
  absl::flat_hash_set<void*> visited;
  auto add_to_worklist = [&worklist, &visited](Value v) {
    if (visited.find(v.getAsOpaquePointer()) == visited.end()) {
      worklist.emplace(v);
      visited.emplace(v.getAsOpaquePointer());
    }
  };

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
            mlir::dyn_cast<RankedTensorType>(push.getTensor().getType());
        if (!verify_and_update_potential_element_type(element_type))
          return false;
        add_to_worklist(push.getOutputHandle());
        continue;
      }
      if (auto scatter = llvm::dyn_cast<TensorListScatterIntoExistingListOp>(
              use.getOwner())) {
        // For scatter op we can get the element shape by dropping the first
        // dimension of the input tensor.
        RankedTensorType element_type =
            DropFirstDimension(scatter.getTensor().getType());
        if (!verify_and_update_potential_element_type(element_type))
          return false;
        add_to_worklist(scatter.getOutputHandle());
        continue;
      }
      if (auto set_item = llvm::dyn_cast<TensorListSetItemOp>(use.getOwner())) {
        auto element_type =
            mlir::dyn_cast<RankedTensorType>(set_item.getItem().getType());
        DCOMMENT("\tTensorListSetItemOp " << element_type);
        if (!verify_and_update_potential_element_type(element_type))
          return false;
        add_to_worklist(set_item.getOutputHandle());
        continue;
      }
      if (auto pop = llvm::dyn_cast<TensorListPopBackOp>(use.getOwner())) {
        add_to_worklist(pop.getOutputHandle());
        continue;
      }
      if (auto resize = llvm::dyn_cast<TensorListResizeOp>(use.getOwner())) {
        add_to_worklist(resize.getOutputHandle());
        continue;
      }
      // WhileRegionOp can explicitly capture TensorList value to be used inside
      // its regions. So we check the uses of corresponding block argument in
      // each region and the use of TensorList returned using YieldOp.
      if (auto while_region = llvm::dyn_cast<WhileRegionOp>(use.getOwner())) {
        DCOMMENT("\tTL WhileRegion");
        for (auto branch : while_region.getRegions())
          add_to_worklist(branch->getArgument(use.getOperandNumber()));
        continue;
      }
      if (auto yield = llvm::dyn_cast<YieldOp>(use.getOwner())) {
        Operation* parent = yield->getParentOp();
        add_to_worklist(parent->getResult(use.getOperandNumber()));
        continue;
      }
      // TODO(jpienaar): This can be generalized.
      if (isa<IdentityOp, IdentityNOp, StopGradientOp>(use.getOwner())) {
        add_to_worklist(use.getOwner()->getResult(use.getOperandNumber()));
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

// Returns the tensor type created from the `shape_attr` and `type_attr`
// attributes.
Type GetType(Attribute shape_attr, Attribute type_attr) {
  auto shape = mlir::cast<tf_type::ShapeAttr>(shape_attr);
  auto type = mlir::cast<TypeAttr>(type_attr);
  if (shape.hasRank())
    return tensorflow::GetTypeFromTFTensorShape(shape.getShape(),
                                                type.getValue());
  else
    return UnrankedTensorType::get(type.getValue());
}
}  // namespace

// Create a MLIRContext based on the threading setup in the env var.
std::unique_ptr<MLIRContext> MakeMLIRContextWithThreading() {
  return std::make_unique<MLIRContext>(GetMlirContextThreading());
}

// Returns whether type can be further refined.
bool CanBeRefined(Type type) {
  auto shape_type = mlir::dyn_cast<ShapedType>(type);
  if (!shape_type) return false;

  // Returns whether type with subtypes can be further refined.
  auto can_refine_subtypes = [](TF::TensorFlowTypeWithSubtype tws) {
    return tws.GetSubtypes().empty() ||
           llvm::any_of(tws.GetSubtypes(), CanBeRefined);
  };
  auto type_with_subtype = mlir::dyn_cast<TF::TensorFlowTypeWithSubtype>(
      shape_type.getElementType());
  if (type_with_subtype && can_refine_subtypes(type_with_subtype)) return true;

  return !shape_type.hasStaticShape();
}

// Returns a new arg type based on the shape and element type. If there are
// dynamic bounds attribute to the arg, update the bounds based on the shape
// as well.
Type GetNewArgType(Type old_arg_type, ArrayRef<int64_t> shape,
                   Type element_type, mlir::MLIRContext* context) {
  Type new_arg_type = tensorflow::GetTypeFromTFTensorShape(shape, element_type);

  if (auto input_ty = mlir::dyn_cast<RankedTensorType>(old_arg_type)) {
    ArrayRef<int64_t> bounds = hlo::encodingToBounds(input_ty.getEncoding());
    // The input type has bounded dynamic dimension.
    if (!bounds.empty()) {
      SmallVector<int64_t> new_bounds(bounds.begin(), bounds.end());
      SmallVector<int64_t> new_shape(shape.begin(), shape.end());
      // If dimension of the input type is dynamic. Update the
      // bounds of the dim with the new type if needed.
      for (int i = 0; i < input_ty.getShape().size(); i++) {
        if (hlo::isDynamicDimSize(input_ty.getShape()[i])) {
          new_bounds[i] = new_shape[i];
          new_shape[i] = ShapedType::kDynamic;
        }
      }
      new_arg_type = tensorflow::GetTypeFromTFTensorShape(
          new_shape, element_type,
          mlir::mhlo::TypeExtensionsAttr::get(context, new_bounds));
    }
  }
  return new_arg_type;
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

  ValuePort() = default;

  // Convert output value to ValuePort.
  explicit ValuePort(Value v) {
    OpResult opr = mlir::dyn_cast<OpResult>(v);
    if (opr) {
      producer = opr.getOwner();
      port = {opr.getResultNumber()};
    } else {
      producer = mlir::cast<BlockArgument>(v);
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

  bool IsValid() const { return !producer.isNull(); }
};

struct ValuePortHasher {
  std::size_t operator()(const ValuePort& other) const {
    return hash_combine(llvm::hash_value(other.producer.getOpaqueValue()),
                        hash_value(ArrayRef<unsigned int>(other.port)));
  }
};

using ValuePortResultMap =
    absl::flat_hash_map<ValuePort, Attribute, ValuePortHasher>;
using ComputedQueryFn = function_ref<bool(ValuePort)>;
using ValueQueryFn = function_ref<Attribute(const ValuePort&)>;
using ValuePortInputs = SmallVectorImpl<ValuePort>;

// Note: Following implements the rank 1 pack op case so could be
// generalized.
//
// Maps the specified component in the `port` of the given op's result to one of
// the element in the input.
ValuePort ComputeInputComponentFor(PackOp op, ArrayRef<unsigned int> port) {
  auto type = mlir::cast<TensorType>(op.getType());
  if (!type.hasRank() || type.getRank() != 1) return {};
  if (port.size() != 2) return {};
  assert(port[0] == 0);
  return ValuePort(op.getOperand(port[1]));
}

ValuePort ComputeInputComponentFor(ConcatV2Op op, ArrayRef<unsigned int> port) {
  if (port.size() != 2) return {};
  assert(port[0] == 0);

  int64_t element_idx = port[1];
  for (Value val : op.getValues()) {
    auto val_ty = mlir::cast<TensorType>(val.getType());
    if (!val_ty.hasStaticShape() || val_ty.getRank() != 1) return {};

    int64_t dim_size = val_ty.getNumElements();
    if (element_idx >= dim_size) {
      element_idx -= dim_size;
      continue;
    }

    ValuePort req(val);
    req.port.push_back(element_idx);
    return req;
  }
  return {};
}

ValuePort ComputeInputComponentFor(GatherV2Op op, ArrayRef<unsigned int> port) {
  if (port.size() != 2) return {};
  assert(port[0] == 0);

  auto params = op.getParams();
  auto params_ty = mlir::dyn_cast<RankedTensorType>(params.getType());
  if (!params_ty || !params_ty.hasStaticShape() || params_ty.getRank() != 1 ||
      op.getBatchDims() != 0) {
    return {};
  }

  DenseIntElementsAttr axis;
  if (!matchPattern(op.getAxis(), m_Constant(&axis)) ||
      axis.getNumElements() != 1 ||
      !axis.getSplatValue<llvm::APInt>().isZero()) {
    return {};
  }

  DenseIntElementsAttr indices;
  if (!matchPattern(op.getIndices(), m_Constant(&indices)) ||
      indices.getType().getRank() != 1 || port[1] >= indices.getNumElements()) {
    return {};
  }

  int64_t input_idx = indices.getValues<IntegerAttr>()[port[1]].getInt();
  if (input_idx >= params_ty.getDimSize(0)) return {};

  ValuePort req(params);
  req.port.push_back(input_idx);
  return req;
}

ValuePort ComputeInputComponentFor(Operation* op, ArrayRef<unsigned int> port) {
  if (auto pack_op = llvm::dyn_cast<PackOp>(op)) {
    return ComputeInputComponentFor(pack_op, port);
  }
  if (auto concat_op = llvm::dyn_cast<ConcatV2Op>(op)) {
    return ComputeInputComponentFor(concat_op, port);
  }
  if (auto gather_op = llvm::dyn_cast<GatherV2Op>(op)) {
    return ComputeInputComponentFor(gather_op, port);
  }
  return {};
}

// TODO(jpienaar): ComputeInputsRequiredForOutput and ComputeOutputComponent are
// intended to be switched to op interfaces once more refined.
LogicalResult ComputeInputsRequiredForOutput(ValuePort value_port,
                                             ComputedQueryFn has_been_computed,
                                             ValuePortInputs* inputs) {
  auto op = value_port.producer.dyn_cast<Operation*>();
  auto& port = value_port.port;
  if (!op) return failure();

  // No inputs required for constants and ShapeOp.
  if (matchPattern(op, m_Constant()) || isa<TF::ShapeOp>(op)) return success();

  ValuePort req = ComputeInputComponentFor(op, port);
  if (req.IsValid()) {
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
    if (port.size() == 2) {
      assert(port[0] == 0);
      DenseIntElementsAttr value;
      if (!matchPattern(op, m_Constant(&value)) ||
          value.getType().getRank() != 1 || port[1] >= value.getNumElements()) {
        return nullptr;
      }

      auto range = value.getValues<Attribute>();
      auto component_ty = RankedTensorType::get({1}, value.getElementType());
      return DenseElementsAttr::get(component_ty, range[port[1]]);
    }
    return nullptr;
  }

  if (auto id = dyn_cast<IdentityOp>(op)) {
    if (port.size() == 1 && port[0] == 0)
      return ComputeOutputComponent(ValuePort(id.getInput()), values);
    return nullptr;
  }

  if (auto shape_op = dyn_cast<TF::ShapeOp>(op)) {
    // No shape available in an unranked tensor type.
    auto operand_ty =
        mlir::dyn_cast<RankedTensorType>(shape_op.getOperand().getType());
    if (!operand_ty) return nullptr;

    // Shape op has a single output so the first element should always be zero
    // and the second element of port points to a particular element in the
    // shape result.
    if (port.size() != 2 || port[0] != 0 || port[1] >= operand_ty.getRank())
      return nullptr;

    // If the dim is dynamic, the dimension can't be inferred during
    // compilation.
    int64_t dim = operand_ty.getDimSize(port[1]);
    if (dim == ShapedType::kDynamic) return nullptr;

    // Create an elements attribute for the particular dimension.
    Type element_ty = getElementTypeOrSelf(shape_op.getType());
    APInt dim_value(element_ty.getIntOrFloatBitWidth(), dim);
    auto component_ty = RankedTensorType::get({1}, element_ty);
    return DenseElementsAttr::get(component_ty, {dim_value});
  }

  if (auto graph = dyn_cast<tf_executor::GraphOp>(op)) {
    if (port.size() == 1)
      return ComputeOutputComponent(
          ValuePort(graph.GetFetch().getFetches()[port[0]]), values);
    return nullptr;
  }

  if (auto island = dyn_cast<tf_executor::IslandOp>(op)) {
    if (port.size() == 1)
      return ComputeOutputComponent(
          ValuePort(island.GetYield().getFetches()[port[0]]), values);
    return nullptr;
  }

  ValuePort req = ComputeInputComponentFor(op, port);
  if (req.IsValid()) return values(req);

  return nullptr;
}

// Context used during ShapeInference. This class contains common information
// that is required by the individual shape inference helper functions (e.g.,
// TF Graph version, constant values computed, etc.)
class ShapeInference {
 public:
  ShapeInference(int64_t graph_version, ModuleOp module,
                 bool propagate_caller_callee_constants,
                 ArrayRef<TypeID> ops_to_skip);

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
  bool InferShapeForSingleOperation(Operation* op, int64_t max_iterations);

  // Infers shape on the provided region, including nested ones, iterate until
  // fix point with a limit of max_iteration.
  // Returns a failure() on error, otherwise returns true to indicate that it
  // reached convergence, false otherwise.
  FailureOr<bool> InferShapeUntilFixPoint(Region* region,
                                          int64_t max_iterations);

  // Updates the serialized StableHLO modules of XlaCallModule ops whose shapes
  // are refined. This function should be called after the shape refinement has
  // finished, i.e. when InferShapeUntilFixPoint will no longer be called, to
  // avoid re-serializing the modules multiple times.
  // Returns whether it was able to propagate the shape to the StableHLO modules
  // successfully.
  LogicalResult PropagateShapeToStableHloModules();

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
                                            ArrayRef<func::FuncOp> functions,
                                            int64_t max_iterations);

  // Propagates shapes to regions given the shapes of the inputs of the regions.
  // All regions provided in `regions` are assumed to have inputs of type
  // `input_types`.
  // Returns a failure() on error, otherwise returns true to indicate that it
  // reached convergence, false otherwise.
  FailureOr<bool> PropagateShapeToRegions(TypeRange input_types,
                                          ArrayRef<Region*> regions,
                                          int64_t max_iterations);

  // Shape propagation for call/control flow ops.
  // Returns a failure() on error, otherwise returns true to indicate that it
  // reached convergence, false otherwise.
  FailureOr<bool> PropagateShapeIntoAttachedFunctions(Operation* op,
                                                      int64_t max_iterations);

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

  // Forcely assign operand types to result types (the i-th operand type will
  // assign to i-th result type). Returns true if anything is changed.
  bool ForceTypeForPassThroughOperands(Operation* op, OperandRange operands,
                                       ResultRange results);

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
  LogicalResult InferShapeForFunctionReturnType(func::FuncOp func);

  // Enqueues function for processing.
  void enqueue(func::FuncOp fn) {
    LLVM_DEBUG(llvm::dbgs()
               << "enqueue " << fn.getName() << " ("
               << (queue_set_.count(fn) ? "already inserted" : "newly inserted")
               << ")\n");
    if (queue_set_.insert(fn).second) queue_.push(fn);
  }

  // Enqueues callers on functions.
  void EnqueueCallers(func::FuncOp fn);

  // Returns the function at the front of the queue.
  func::FuncOp front() { return queue_.front(); }

  // Returns whether work queue is empty.
  bool EmptyQueue() const { return queue_.empty(); }

  // Returns function from the front of the work queue.
  func::FuncOp pop_front() {
    func::FuncOp ret = queue_.front();
    queue_.pop();
    queue_set_.erase(ret);
    return ret;
  }

  // Returns the current size of the queue.
  std::queue<func::FuncOp>::size_type QueueSize() const {
    return queue_.size();
  }

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

  // Infers the shape from a (Stateful)PartitionedCall operation by looking up
  // the called function and propagating the return type.
  bool InferShapeForCall(CallOpInterface call_op);

  bool InferShapeForCast(Operation* op);

  bool InferShapeForRestore(Operation* op);

  // Infers the shape IfOp outputs based on the shapes of the then and else
  // function result types.
  bool InferShapeForIf(IfOp op);

  // Infers the shape IfRegion outputs based on the shapes of the then and else
  // yields.
  bool InferShapeForIfRegion(IfRegionOp op);

  // Infers the shape CaseOp outputs based on the shapes of branch function
  // result types.
  bool InferShapeForCase(CaseOp op);

  // Infers the shape CaseRegion outputs based on the shapes of the branch
  // yields.
  bool InferShapeForCaseRegion(CaseRegionOp op);

  // Infers the shape CaseRegion outputs based on the embedded StableHLO module.
  // Returns true if a return type was changed.
  bool InferShapeForXlaCallModule(XlaCallModuleOp op);

  // Infers the shape of _XlaHostComputeMlir based on the host computation
  // module.  Returns true if a return type was changed.
  bool InferShapeForXlaHostComputeMlir(_XlaHostComputeMlirOp op);

  // Infers the shape of function attached to XlaHostCompute.
  // Returns true if a return type was changed.
  bool InferShapeForFunctionAttachedToXlaHostCompute(XlaHostComputeOp op);

  // Infers the shape for MapDatasetOp and its associated function. Returns
  // whether either op or function type was changed.
  bool InferShapeForMapDataset(MapDatasetOp op, int64_t max_iterations);

  // Infers the shape for ReduceDatasetOp and its associated reduce function.
  // Returns whether either op or function type was changed.
  bool InferShapeForReduceDataset(ReduceDatasetOp op, int64_t max_iterations);

  // Infers the shape for TakeWhileDatasetOp and its associated predicate
  // function. Returns whether either op or function type was changed.
  bool InferShapeForTakeWhileDataset(TakeWhileDatasetOp op,
                                     int64_t max_iterations);

  // Infers shape for dataset ops that have `M` input elements and `N`
  // arguments, and also propagates the shape to the specified function (called
  // only when function exists and has single use).
  bool InferShapeForDatasetOpCommon(Operation* op, FuncOp f,
                                    int64_t max_iterations);

  // Infers the shape of ops that create TensorList. Specifically,
  // TensorListReserveOp, EmptyTensorListOp and TensorListFromTensor ops. It
  // refines the element shape if all tensors written to the list across all
  // mutations have identical static shape.
  bool InferShapeForTensorListInitOps(Operation* op);

  // Conservatively infers shape of output tenorlist and item based on
  // input tensorlist's element shape.
  bool InferShapeForTensorListPopBackOp(TensorListPopBackOp op);

  // Infers the shape of VarHandleOp based on the uses of the VarHandleOp to
  // update the subtypes of the resource type.
  bool InferShapeForVarHandleOp(VarHandleOp op);

  // Infers the output shape of XlaConvV2Op based on the input shapes
  bool InferShapeForXlaConvV2Op(XlaConvV2Op op);

  // Infers the output shape of XlaReduceWindowOp based on the input shapes.
  bool InferShapeForXlaReduceWindowOp(XlaReduceWindowOp op);

  // Infers the output shape of XlaSelectAndScatterOp based on the input shapes.
  bool InferShapeForXlaSelectAndScatterOp(XlaSelectAndScatterOp op);

  // Infers the output shape of XlaGatherOp based on the input shapes.
  bool InferShapeForXlaGatherOp(XlaGatherOp op);

  bool RefineWithInferTypeOpInterface(InferTypeOpInterface infer_ti);

  // Returns all the callers of a function.
  // Note: Usage of the return value of this function may not be interleaved
  // with insertions to the callers map. This could occur if GetCallers is
  // called with two separate functions, the 2nd one incurs a resize and then
  // both first and 2nd stored callers are used.
  ArrayRef<Operation*> GetCallers(func::FuncOp fn);

  // Mapping between ValuePort (which corresponds to an OpResult or smaller,
  // e.g., first element of OpResult produced) to an Attribute if the ValuePort
  // corresponds to a constant value.
  ValuePortResultMap results_;

  // Map from a function to the callers of that function.
  SymbolTableCollection symbol_table_;
  SymbolUserMap symbol_users_;

  // Queue of functions being processed.
  llvm::DenseSet<func::FuncOp> queue_set_;
  std::queue<func::FuncOp> queue_;

  int64_t graph_version_;

  // Op types for which shape inference should be skipped.
  llvm::SmallDenseSet<TypeID> ops_to_skip_;

  // TODO(b/154065712): Remove propagate_caller_callee_constants once using
  // SCCP pass instead.
  bool propagate_caller_callee_constants_;

  // XlaCallModule loader, which is used to deserialize the StableHLO module in
  // each `XlaCallModule` op. Uses its own MLIRContext since the loader needs to
  // load additional dialects, which is not allowed for the main context since
  // shape inference may be called from a pass.
  std::unique_ptr<MLIRContext> xla_call_module_context_;
  DenseMap<XlaCallModuleOp, std::unique_ptr<tensorflow::XlaCallModuleLoader>>
      xla_call_module_loaders_;
};

ShapeInference::ShapeInference(int64_t graph_version, ModuleOp module,
                               bool propagate_caller_callee_constants,
                               ArrayRef<TypeID> ops_to_skip)
    : tf_dialect_(module->getContext()->getLoadedDialect<TensorFlowDialect>()),
      symbol_users_(symbol_table_, module),
      graph_version_(graph_version),
      propagate_caller_callee_constants_(propagate_caller_callee_constants) {
  xla_call_module_context_ = MakeMLIRContextWithThreading();
  for (const auto& op_type : ops_to_skip) {
    ops_to_skip_.insert(op_type);
  }
  // Create symbol table for module.
  symbol_table_.getSymbolTable(module);
}

ArrayRef<Operation*> ShapeInference::GetCallers(func::FuncOp fn) {
  return symbol_users_.getUsers(fn);
}

void ShapeInference::EnqueueCallers(func::FuncOp fn) {
  for (auto user : GetCallers(fn)) {
    auto func = user->getParentOfType<func::FuncOp>();
    if (func) enqueue(func);
  }
}

bool ShapeInference::UpdateTypeAndInsertIncompatibleUseCasts(Type new_type,
                                                             Value result) {
  // No changes needed if the new type is unchanged.
  if (new_type == result.getType()) return false;

  Operation* cast_op = nullptr;
  // First insert cast back for uses that need a cast and then
  // update the type.
  bool enqueue_callers = false;
  for (OpOperand& use : make_early_inc_range(result.getUses())) {
    if (isa<func::ReturnOp>(use.getOwner())) {
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
    EnqueueCallers(result.getDefiningOp()->getParentOfType<func::FuncOp>());
  return true;
}

bool ShapeInference::RefineResultType(Operation* op, Value result,
                                      Type potential_refined_type) {
  if (!CanRefineTypeWith(result.getType(), potential_refined_type))
    return false;

  return UpdateTypeAndInsertIncompatibleUseCasts(potential_refined_type,
                                                 result);
}

// Infers the shape from a (Stateful)PartitionedCall operation by looking up
// the called function and propagating the return type.
bool ShapeInference::InferShapeForCall(CallOpInterface call_op) {
  func::FuncOp func = dyn_cast_or_null<func::FuncOp>(
      call_op.resolveCallableInTable(&symbol_table_));
  if (!func) return false;

  DCOMMENT("Infer shape for call " << func.getName());
  Operation* op = call_op.getOperation();
  bool changed = false;
  // Map each of the results of the call to the returned type of the
  // function.
  for (auto result :
       zip(op->getResults(), func.getFunctionType().getResults())) {
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

  // Avoid inserting a cast where no users types could be refined (e.g., where
  // there would need to be a cast inserted for every user again).
  if (llvm::all_of(result.getUses(), [this](OpOperand& use) {
        return NeedsCastBack(use, tf_dialect_);
      }))
    return false;

  // Combine shape information including shape info in subtypes.
  Type operand_type = op->getOperand(0).getType();
  Type result_type = result.getType();
  auto new_type = GetCastCompatibleType(operand_type, result_type);
  if (!new_type) {
    // Combine shape information when leaf element types are not the same, not
    // including shape info in subtypes.
    auto ranked_operand_type = mlir::dyn_cast<RankedTensorType>(operand_type);
    if (!ranked_operand_type) return false;
    auto ranked_res_type = mlir::dyn_cast<RankedTensorType>(result.getType());
    if (ranked_res_type &&
        ranked_operand_type.getShape() == ranked_res_type.getShape())
      return false;

    auto shaped_res_type = mlir::dyn_cast<ShapedType>(result_type);
    if (!shaped_res_type) return false;
    new_type = tensorflow::GetTypeFromTFTensorShape(
        ranked_operand_type.getShape(), shaped_res_type.getElementType());
  }
  return UpdateTypeAndInsertIncompatibleUseCasts(new_type, result);
}

bool ShapeInference::InferShapeForIf(IfOp op) {
  DCOMMENT_OP(op.getOperation(), "Infer shape for if ");
  bool changed = false;
  auto then_results =
      op.ResolveThenFunction(&symbol_table_).getFunctionType().getResults();
  auto else_results =
      op.ResolveElseFunction(&symbol_table_).getFunctionType().getResults();
  for (auto it : llvm::zip(op.getResults(), then_results, else_results)) {
    // If then and else types do not match, skip refinement for that result.
    if (std::get<1>(it) != std::get<2>(it)) continue;
    changed = RefineResultType(op, std::get<0>(it), std::get<1>(it)) || changed;
  }
  return changed;
}

bool ShapeInference::InferShapeForIfRegion(IfRegionOp op) {
  bool changed = false;

  Operation* then_yield = op.getThenBranch().front().getTerminator();
  Operation* else_yield = op.getElseBranch().front().getTerminator();
  for (auto result : zip(op.getResults(), then_yield->getOperandTypes(),
                         else_yield->getOperandTypes())) {
    // If then and else types do not match, skip refinement for that result.
    if (std::get<1>(result) != std::get<2>(result)) continue;
    changed = RefineResultType(op, std::get<0>(result), std::get<1>(result)) ||
              changed;
  }
  return changed;
}

bool ShapeInference::InferShapeForCase(CaseOp op) {
  DCOMMENT_OP(op.getOperation(), "Infer shape for case ");

  llvm::SmallVector<TypeRange> branch_result_types;
  for (int i = 0; i < op.num_branches(); ++i) {
    branch_result_types.push_back(op.ResolveBranchFunction(&symbol_table_, i)
                                      .getFunctionType()
                                      .getResults());
  }

  bool changed = false;
  for (const auto& result : op.getResults()) {
    llvm::DenseSet<Type> types;
    for (const auto& branch_result_type : branch_result_types) {
      types.insert(branch_result_type[result.getResultNumber()]);
    }
    if (types.size() == 1) {
      changed = RefineResultType(op, result, *types.begin()) || changed;
    }
  }
  return changed;
}

bool ShapeInference::InferShapeForCaseRegion(CaseRegionOp op) {
  bool changed = false;
  for (const auto& result : op.getResults()) {
    llvm::DenseSet<Type> types;
    for (auto& branch : op.getBranches()) {
      Operation* yield = branch.front().getTerminator();
      types.insert(yield->getOperandTypes()[result.getResultNumber()]);
    }
    if (types.size() == 1) {
      changed = RefineResultType(op, result, *types.begin()) || changed;
    }
  }
  return changed;
}

bool ShapeInference::InferShapeForXlaCallModule(XlaCallModuleOp op) {
  if (!llvm::any_of(op.getResultTypes(), CanBeRefined)) return false;

  tensorflow::XlaCallModuleLoader* loader;
  if (auto it = xla_call_module_loaders_.find(op);
      it != xla_call_module_loaders_.end()) {
    loader = it->second.get();
  } else {
    // Lazily parse XlaCallModule's embedded HLO module and cache the loader to
    // avoid repeatedly parsing the module.

    std::vector<std::string> disabled_checks;
    for (auto attr : op.getDisabledChecks().getAsRange<StringAttr>()) {
      disabled_checks.push_back(attr.getValue().str());
    }
    std::vector<std::string> platforms;
    for (auto attr : op.getPlatforms().getAsRange<StringAttr>()) {
      platforms.push_back(attr.getValue().str());
    }

    // It is a terrible idea to have local MLIR contexts so we need to
    // register extensions here, again.
    mlir::DialectRegistry registry;
    registry.insert<mlir::func::FuncDialect>();
    mlir::func::registerAllExtensions(registry);
    xla_call_module_context_->appendDialectRegistry(registry);

    auto l = tensorflow::XlaCallModuleLoader::Create(
        xla_call_module_context_.get(), op.getVersion(), op.getModule(),
        std::move(disabled_checks), std::move(platforms),
        /*num_invocation_args=*/op.getArgs().size(),
        op.getHasTokenInputOutput(), op.getUseShardyPartitioner());
    if (!l.ok()) {
      llvm::errs() << "Parsing error in XlaCallModule: "
                   << l.status().ToString() << "\n";
      return false;
    }

    it = xla_call_module_loaders_.insert({op, *std::move(l)}).first;
    loader = it->second.get();
  }

  // Cannot pass `op.getArgs().getTypes()` to `loader->RefineDynamicShapes`
  // because `op` and `loader` are using different MLIR contexts. See comments
  // on `xla_call_module_context_` for details.
  std::vector<xla::Shape> input_shapes;
  input_shapes.reserve(op.getArgs().size());
  for (mlir::Type type : op.getArgs().getTypes()) {
    input_shapes.push_back(xla::TypeToShape(type));
  }

  absl::Status status = loader->RefineDynamicShapes(input_shapes);
  if (!status.ok()) {
    // Do not return false here.
    //
    // RefineDynamicShapes returns ok only when it produces full static shapes.
    // It may partially succeed by producing RankedTensor shapes with dynamic
    // dimensions. Such info is still useful for the downstream. We don't need
    // to abort here.
    // TODO(b/316639984): improve RefineDynamicShapes return values to include
    // these info.
    VLOG(1) << "Failed during XlaCallModule shape refinement: " << status;
  }
  mlir::ResultRange op_results = op.getResults();
  // The main_outputs may include tokens that are not among the op_results;
  mlir::TypeRange main_output_types = loader->OutputTypes();
  int nr_main_token_outputs =
      llvm::count_if(main_output_types, tensorflow::IsTokenType);
  if (op_results.size() != main_output_types.size() - nr_main_token_outputs) {
    llvm::errs() << "XlaCallModule has " << op_results.size()
                 << " but the main function has "
                 << main_output_types.size() - nr_main_token_outputs
                 << " non-token ouputs";
    return false;
  }
  bool changed = false;
  int next_op_result = 0;
  for (auto output_type : main_output_types) {
    if (tensorflow::IsTokenType(output_type)) continue;
    auto output_type_ranked = mlir::dyn_cast<RankedTensorType>(output_type);
    if (output_type_ranked == nullptr) {
      llvm::errs() << "Unsupported XlaCallModule result type: " << output_type
                   << "\n";
      return false;
    }
    auto result = op_results[next_op_result++];

    // Build a new type object from `type` and `elem_type`. `type` is owned by
    // `xla_call_module_context_` and should not be mixed with op's context.
    auto new_type = RankedTensorType::get(
        output_type_ranked.getShape(), getElementTypeOrSelf(result.getType()));

    changed = RefineResultType(op, result, new_type) || changed;
  }

  return changed;
}

LogicalResult ShapeInference::PropagateShapeToStableHloModules() {
  for (auto& [op, loader] : xla_call_module_loaders_) {
    if (!loader->IsOutputTypeRefined()) continue;

    FailureOr<vhlo::Version> version =
        stablehlo::getPortableArtifactVersion(op.getModule());
    if (failed(version)) {
      return op.emitOpError() << "Failed to extract the VHLO version from the "
                                 "serialized StableHLO module while attempting "
                                 "to propagate shapes to the StableHLO module.";
    }

    std::string bytecode;
    llvm::raw_string_ostream os(bytecode);
    if (failed(stablehlo::serializePortableArtifact(loader->module(),
                                                    version->toString(), os))) {
      return op.emitOpError()
             << "Failed to serialize StableHLO module while attempting to "
                "propagate shapes to the StableHLO module.";
    }

    op.setModule(bytecode);
  }
  return success();
}

bool ShapeInference::InferShapeForFunctionAttachedToXlaHostCompute(
    XlaHostComputeOp op) {
  const std::string kShapeInferenceGraph = "shape_inference_graph";
  if (!op->hasAttr(kShapeInferenceGraph)) {
    return false;
  }

  ModuleOp module = op->getParentOfType<ModuleOp>();
  func::FuncOp func = module.lookupSymbol<func::FuncOp>(
      op.getShapeInferenceGraphAttr().getRootReference());

  if (func == nullptr) return false;

  std::vector<_XlaRecvAtHostOp> xla_recv_at_host_ops;
  func.walk([&](_XlaRecvAtHostOp op) { xla_recv_at_host_ops.push_back(op); });
  if (xla_recv_at_host_ops.empty()) return false;
  auto xla_recv_at_host_op = xla_recv_at_host_ops.front();

  // Copy const op into func body and replace the uses of the corresponding args
  OpBuilder builder(&func.front().front());
  for (auto arg : func.getArguments()) {
    Value operand = op.getOperand(arg.getArgNumber());
    if (isa_and_nonnull<TF::ConstOp>(operand.getDefiningOp())) {
      xla_recv_at_host_op.getResult(arg.getArgNumber())
          .replaceAllUsesWith(
              builder.clone(*operand.getDefiningOp())->getResult(0));
    }
  }

  // Update/use input shapes for function.
  FunctionType func_type = func.getFunctionType();
  func.setType(FunctionType::get(func.getContext(), op.getOperandTypes(),
                                 func_type.getResults()));

  // Run shape inference on the function.
  if (failed(
          PropagateShapeToRegions(op.getOperandTypes(), {&func.getBody()}, 10)))
    return false;
  if (failed(InferShapeForFunctionReturnType(func))) return false;

  return false;
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

  mlir::OwningOpRef<mlir::ModuleOp> module_for_func;
  func::FuncOp func = host_compute_op.GetHostFunc(&module_for_func);

  // Update/use input shapes for function.
  FunctionType func_type = func.getFunctionType();
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
       zip(host_compute_op.getResults(), func.getFunctionType().getResults())) {
    changed = RefineResultType(host_compute_op, std::get<0>(result),
                               std::get<1>(result)) ||
              changed;
  }

  return changed;
}

// Infer the shape of `Restore` and `RestoreV2` op based on the first
// `AssignVariableOp` that uses the result. This requires that the resource
// subtype inference is completed.
bool ShapeInference::InferShapeForRestore(Operation* op) {
  DCOMMENT_OP(op, "Inferring shape for Restore,RestoreV2");
  // Currently only support single output.
  if (op->getNumResults() != 1) return false;
  if (!CanBeRefined(op->getResult(0).getType())) return false;

  llvm::SmallVector<mlir::Operation*> worklist;
  llvm::append_range(worklist, op->getUsers());

  // Look for any `AssignVariableOp` that uses the result of this op.
  while (!worklist.empty()) {
    mlir::Operation* const use = worklist.pop_back_val();

    // Follow the `CastOp`/`IdentityOp`'s users to handle the `RestoreV2` ->
    // (optionally `IdentityOp`) -> `CastOp` `AssignVariableOp` case.
    if (llvm::isa<TF::CastOp, TF::IdentityOp>(use)) {
      llvm::append_range(worklist, use->getUsers());
      continue;
    }

    TF::AssignVariableOp assign_op = llvm::dyn_cast<TF::AssignVariableOp>(use);
    if (!assign_op) {
      continue;
    }
    auto subtypes = mlir::cast<TF::ResourceType>(
                        getElementTypeOrSelf(assign_op.getResource()))
                        .getSubtypes();
    if (subtypes.empty()) {
      continue;
    }
    auto subtype = mlir::dyn_cast<ShapedType>(subtypes.front());
    if (subtype == nullptr) {
      continue;
    }
    // Preserve the dtype from the restore op even if `AssignVariableOp` uses a
    // different dtype, which is possible when there's a `CastOp` between them.
    subtype = subtype.clone(
        mlir::cast<ShapedType>(op->getResult(0).getType()).getElementType());
    // Update the result type of this op with the resource's type. We only use
    // the resource subtype of the first user since shapes from all the users
    // should be equal or compatible.
    return UpdateTypeAndInsertIncompatibleUseCasts(subtype, op->getResult(0));
  }
  return false;
}

// Helper structure to capture shapes & types for Dataset input.
struct DatasetInput {
  explicit operator bool() const { return shapes && types; }

  ArrayAttr shapes;
  ArrayAttr types;
};

// Returns the input elements shapes and types for Dataset ops.
DatasetInput GetDatasetInput(Value value) {
  // TODO(haoliang): add an interface for DatasetOp to avoid the following
  // enumeration.
  // Iteratively tracing upwards if parent op is `IdentityOp` or `IdentityNOp`.
  while (
      llvm::isa_and_nonnull<IdentityOp, IdentityNOp>(value.getDefiningOp())) {
    value = value.getDefiningOp()->getOperand(
        mlir::cast<OpResult>(value).getResultNumber());
  }

  Operation* op = value.getDefiningOp();
  if (!llvm::isa_and_nonnull<BatchDatasetV2Op, MapDatasetOp, RepeatDatasetOp,
                             ParallelMapDatasetOp, ParallelMapDatasetV2Op,
                             TakeDatasetOp, TakeWhileDatasetOp>(op))
    return DatasetInput{nullptr, nullptr};

  return DatasetInput{op->getAttrOfType<ArrayAttr>("output_shapes"),
                      op->getAttrOfType<ArrayAttr>("output_types")};
}

bool ShapeInference::InferShapeForDatasetOpCommon(Operation* op, FuncOp f,
                                                  int64_t max_iterations) {
  int N = op->getNumOperands() - 1;
  int M = f.getNumArguments() - N;
  DCOMMENT_OP(op, "Inferring shape for with N = " << N << " and M = " << M);

  // Initialize with function input types.
  auto input_types = llvm::to_vector<1>(
      cast<FunctionOpInterface>(f.getOperation()).getArgumentTypes());

  DatasetInput input_elements = GetDatasetInput(op->getOperand(0));
  if (!input_elements) {
    op->emitWarning("unexpected dataset input; skipping function refinement");
    return false;
  }

  // Track if changed to skip enqueueing.
  bool changed = false;
  auto it = input_types.begin();
  // First set first M argument shapes & types.
  for (int i = 0; i < M; ++i) {
    Type t = GetType(input_elements.shapes[i], input_elements.types[i]);
    t = TypeMeet(*it, t);
    changed = changed || (t != *it);
    *it++ = t;
  }
  // Now the remaining N from operand types.
  for (auto t : llvm::drop_begin(op->getOperandTypes())) {
    auto meet = TypeMeet(*it, t);
    changed = changed || (meet != *it);
    *it++ = meet;
  }
  if (!changed) return false;

  FailureOr<bool> res = PropagateShapeToFunctions(
      op->getParentOfType<ModuleOp>(), input_types, {f}, max_iterations);
  if (failed(res)) {
    op->emitOpError("propagating shapes failed");
    return false;
  }
  return *res;
}

bool ShapeInference::InferShapeForMapDataset(MapDatasetOp op,
                                             int64_t max_iterations) {
  // MapDatasetOp's relationship with its associated function is as
  // follows: first M function params are dictated by the set
  // output shapes and types, the next N are the last Ninputs from MapDataset
  // op. The MapDataset op always has N+1 inputs.
  // TODO(jpienaar): Avoid this lookup.
  auto module = op->getParentOfType<ModuleOp>();
  auto f = module.lookupSymbol<func::FuncOp>(op.getF());
  // Skip if function is not found or more than one caller.
  if (!f || !llvm::hasSingleElement(GetCallers(f))) return false;
  return InferShapeForDatasetOpCommon(op, f, max_iterations);
}

bool ShapeInference::InferShapeForTakeWhileDataset(TakeWhileDatasetOp op,
                                                   int64_t max_iterations) {
  // TakeWhileDatasetOp's relationship with its associated function is as
  // follows: first M function params are dictated by the set
  // output shapes and types, the next N are the last Ninputs from
  // TakeWhileDataset op. The TakeWhileDataset op always has N+1 inputs.
  // TODO(jpienaar): Avoid this lookup.
  auto module = op->getParentOfType<ModuleOp>();
  auto f = module.lookupSymbol<func::FuncOp>(op.getPredicate());
  // Skip if function is not found or more than one caller.
  if (!f || !llvm::hasSingleElement(GetCallers(f))) return false;
  return InferShapeForDatasetOpCommon(op, f, max_iterations);
}

bool ShapeInference::InferShapeForReduceDataset(ReduceDatasetOp op,
                                                int64_t max_iterations) {
  // ReduceDatasetOp's relationship with its associated reduce function is
  // described as follows: The reduce function will in general have (X + Y + Z)
  // arguments, where X is the number of tensor components that represent the
  // state, Y is the number of tensor components that represent the input
  // elements, and Z is the number of tensor components that represent any
  // captured arguments. Y is determined by the output_shapes of an op that
  // defines the first operand of `op`.

  // TODO(jpienaar): Avoid this lookup.
  auto module = op->getParentOfType<ModuleOp>();
  auto f = module.lookupSymbol<func::FuncOp>(op.getF());

  // Skip if function is not found or it has more than one caller.
  if (!f || !llvm::hasSingleElement(GetCallers(f))) return false;

  DatasetInput input_elements = GetDatasetInput(op.getInputDataset());

  const int num_states = op.getOutputShapes().size();
  const int num_captured_arguments = op.getNumOperands() - 1 - num_states;

  // If input_elements is undefined, we can still infer the shapes for the
  // states and captured arguments.
  int num_input_elements;
  auto input_types = llvm::to_vector<1>(
      cast<FunctionOpInterface>(f.getOperation()).getArgumentTypes());
  if (input_elements) {
    num_input_elements = input_elements.shapes.size();
  } else {
    num_input_elements =
        input_types.size() - num_states - num_captured_arguments;
  }

  DCOMMENT_OP(op,
              "Inferring shape for ReduceDataset with #states = "
                  << num_states << " , #input_elements = " << num_input_elements
                  << " , and #captured_arguments = " << num_captured_arguments);
  if (num_states + num_input_elements + num_captured_arguments !=
      f.getNumArguments()) {
    op->emitOpError(
        "propagating shapes for ReduceDataset failed due to inconsistent "
        "number of arguments");
    return false;
  }

  // Track if changed to skip enqueueing.
  bool changed = false;
  auto it = input_types.begin();

  // Set the first num_states arguments shapes & types from the state.
  for (int i = 0; i < num_states; ++i) {
    Type t = GetType(op.getOutputShapes()[i], op.getOutputTypes()[i]);
    t = TypeMeet(*it, t);
    changed = changed || (t != *it);
    *it++ = t;
  }

  // Second set the following num_input_elements arguments from
  // repeat_dataset_op.  Skip propagating shape if input_elements is
  // undefined.
  for (int i = 0; i < num_input_elements; ++i) {
    if (input_elements) {
      Type t = GetType(input_elements.shapes[i], input_elements.types[i]);
      t = TypeMeet(*it, t);
      changed = changed || (t != *it);
      *it++ = t;
    } else {
      it++;
    }
  }

  // Last set the remaining num_captured_arguments from op.
  for (auto t : llvm::drop_begin(op.getOperandTypes(), 1 + num_states)) {
    auto meet = TypeMeet(*it, t);
    changed = changed || (meet != *it);
    *it++ = meet;
  }

  if (!changed) return false;

  FailureOr<bool> res =
      PropagateShapeToFunctions(module, input_types, {f}, max_iterations);
  if (failed(res)) {
    op->emitOpError("Propagating shapes for ReduceDataset failed");
    return false;
  }
  return *res;
}

bool ShapeInference::InferShapeForTensorListInitOps(Operation* op) {
  DCOMMENT_OP(op, "Inferring shape for TensorListInitOps.");
  Value handle = op->getResult(0);
  Value initial_element_shape = GetElementShapeOperand(op);
  RankedTensorType element_type;
  if (auto tl_from_tensor = dyn_cast<TensorListFromTensorOp>(op)) {
    // For TensorListFromTensor op we can infer element shape by dropping the
    // first dimension of input tensor.
    element_type = DropFirstDimension(tl_from_tensor.getTensor().getType());
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
  auto tensor_type = tensorflow::GetTypeFromTFTensorShape({}, variant_type);
  bool changed = RefineResultType(op, handle, tensor_type);
  if (changed) DCOMMENT_OP(op, "Modified after shape inference:");
  return changed;
}

bool ShapeInference::InferShapeForTensorListPopBackOp(TensorListPopBackOp op) {
  // The first Operand is assumed to be a TensorType around a variant with a
  // single subtype (e.g. tensor<!tf_type.variant<tensor<2xi32>>>). We will
  // copy this type to the first result, and copy the singular variant subtype
  // to the second result (tensor<2xi32>).
  DCOMMENT_OP(op, "Inferring shape for TensorListPopBackOp.");

  auto src_list_handle_t =
      mlir::dyn_cast_or_null<TensorType>(op.getOperand(0).getType());
  if (!src_list_handle_t) return false;

  // Copy of operand tensorlist type.
  TensorType dst_list_handle_t =
      src_list_handle_t.clone(src_list_handle_t.getElementType());
  auto variant_element_t =
      mlir::dyn_cast_or_null<VariantType>(dst_list_handle_t.getElementType());
  if (!variant_element_t || variant_element_t.getSubtypes().size() != 1)
    return false;

  // Underlying TensorType from variant that represents a shape signature
  // compatible with all elements in the tensorlist.
  TensorType list_handle_element_t = variant_element_t.getSubtypes()[0];

  if (!RefineResultType(op, op->getResult(0), dst_list_handle_t)) {
    DCOMMENT("InferShapeForTensorListPopBackOp could not propogate result 0"
             << op);
    return false;
  }
  if (!RefineResultType(op, op->getResult(1), list_handle_element_t)) {
    DCOMMENT("InferShapeForTensorListPopBackOp could not propogate result 1"
             << op);
    return false;
  }
  return true;
}

bool ShapeInference::InferShapeForVarHandleOp(VarHandleOp op) {
  DCOMMENT_OP(op, "Inferring shape for VarHandleOp");

  Value resource = op.getResource();
  if (!CanBeRefined(resource.getType())) return false;

  // Make sure there are only use cases from the `AssignVariableOp` and
  // `ReadVariableOp`. For other cases, we can skip to be conservative.
  for (auto& use : make_early_inc_range(resource.getUses())) {
    Operation* def = use.getOwner();
    if (!llvm::isa<AssignVariableOp>(def) && !llvm::isa<ReadVariableOp>(def)) {
      return false;
    }
  }

  bool changed = false;

  // Look for any `AssignVariableOp` and `ReadVariableOp` that uses the value of
  // this op.
  for (auto& use : make_early_inc_range(resource.getUses())) {
    Operation* def = use.getOwner();
    Value value;
    if (AssignVariableOp assign_op = dyn_cast<AssignVariableOp>(def)) {
      value = assign_op.getValue();
    } else if (ReadVariableOp read_op = dyn_cast<ReadVariableOp>(def)) {
      value = read_op.getValue();
    } else {
      llvm_unreachable("unexpected operator type");
    }

    TensorType resource_subtype = mlir::cast<TensorType>(value.getType());
    ResourceType resource_type =
        ResourceType::get({resource_subtype}, op.getContext());
    UnrankedTensorType new_resource_type =
        UnrankedTensorType::get(resource_type);

    Type refined_type = TypeMeet(resource.getType(), new_resource_type);
    if (refined_type == resource.getType()) continue;
    resource.setType(refined_type);
    changed = true;
  }

  return changed;
}

// Helper function for creating a Window proto from user-supplied data.
// Returns std::nullopt if the user-supplied data was invalid.
std::optional<xla::Window> InferWindowFromDimensions(
    llvm::SmallVector<int64_t> window_dimensions,
    llvm::SmallVector<int64_t> window_strides,
    llvm::SmallVector<std::pair<int64_t, int64_t>> padding,
    llvm::SmallVector<int64_t> lhs_dilation,
    llvm::SmallVector<int64_t> rhs_dilation) {
  const auto verify_size = [&](const size_t x, const char* x_name) {
    if (x == 0 || x == window_dimensions.size()) {
      return true;
    } else {
      llvm::errs()
          << "Window has different number of window dimensions than of "
          << x_name
          << "\nNumber of window dimensions: " << window_dimensions.size()
          << "\nNumber of " << x_name << ": " << x << "\n";
      return false;
    }
  };

  if (!(verify_size(window_dimensions.size(), "window_dimensions") &&
        verify_size(window_strides.size(), "window strides") &&
        verify_size(padding.size(), "padding entries") &&
        verify_size(lhs_dilation.size(), "lhs dilation factors") &&
        verify_size(rhs_dilation.size(), "rhs dilation factors")))
    return std::nullopt;

  xla::Window window;
  for (size_t i = 0; i < window_dimensions.size(); i++) {
    auto dim = window.add_dimensions();
    dim->set_size(window_dimensions[i]);
    if (!window_strides.empty()) {
      dim->set_stride(window_strides[i]);
    } else {
      dim->set_stride(1);
    }
    if (!padding.empty()) {
      dim->set_padding_low(padding[i].first);
      dim->set_padding_high(padding[i].second);
    } else {
      dim->set_padding_low(0);
      dim->set_padding_high(0);
    }
    if (!lhs_dilation.empty()) {
      dim->set_base_dilation(lhs_dilation[i]);
    } else {
      dim->set_base_dilation(1);
    }
    if (!rhs_dilation.empty()) {
      dim->set_window_dilation(rhs_dilation[i]);
    } else {
      dim->set_window_dilation(1);
    }
    dim->set_window_reversal(false);
  }
  return window;
}

std::optional<RankedTensorType> InferWindowOutputShape(
    const ShapedType& base_shape, const xla::Window& window,
    Type element_type) {
  if (window.dimensions_size() != base_shape.getRank()) {
    llvm::errs() << "Window has dimension " << window.dimensions_size()
                 << " but base shape has dimension " << base_shape.getRank()
                 << "\n";
    return std::nullopt;
  }

  std::vector<int64_t> output_dimensions(window.dimensions_size());
  std::vector<bool> output_is_dynamic(window.dimensions_size());
  for (int64_t i = 0; i < window.dimensions_size(); ++i) {
    const auto& dim = window.dimensions(i);
    if (dim.size() <= 0) {
      llvm::errs() << "Window " << window.DebugString()
                   << " has a non-positive dimension.\n";
      return std::nullopt;
    }
    if (dim.stride() <= 0) {
      llvm::errs() << "Window " << window.DebugString()
                   << " has a non-positive stride.\n";
      return std::nullopt;
    }
    if (dim.base_dilation() < 1) {
      llvm::errs() << "Window " << window.DebugString()
                   << " has a non-positive base area dilation factor.\n";
      return std::nullopt;
    }
    if (dim.window_dilation() < 1) {
      llvm::errs() << "Window " << window.DebugString()
                   << " has a non-positive window dilation factor.\n";
      return std::nullopt;
    }

    if (base_shape.isDynamicDim(i)) {
      output_dimensions[i] = ShapedType::kDynamic;
    } else {
      const int64_t dilated_base = xla::window_util::DilatedBound(
          base_shape.getDimSize(i), dim.base_dilation());
      const int64_t padded_dilated_base =
          dim.padding_low() + dilated_base + dim.padding_high();
      const int64_t dilated_window =
          xla::window_util::DilatedBound(dim.size(), dim.window_dilation());

      output_dimensions[i] = xla::window_util::StridedBound(
          padded_dilated_base, dilated_window, dim.stride());
    }
  }

  return tensorflow::GetTypeFromTFTensorShape(output_dimensions, element_type);
}

bool ShapeInference::InferShapeForXlaReduceWindowOp(XlaReduceWindowOp op) {
  DCOMMENT_OP(op, "Inferring shape for XlaReduceWindowOp");

  bool changed = false;

  auto input_ty = mlir::cast<ShapedType>(op.getInput().getType());
  DenseElementsAttr window_dimensions, window_strides, base_dilations,
      window_dilations, padding;
  if (input_ty.hasStaticShape() &&
      matchPattern(op.getWindowDimensions(), m_Constant(&window_dimensions)) &&
      matchPattern(op.getWindowStrides(), m_Constant(&window_strides)) &&
      matchPattern(op.getBaseDilations(), m_Constant(&base_dilations)) &&
      matchPattern(op.getWindowDilations(), m_Constant(&window_dilations)) &&
      matchPattern(op.getPadding(), m_Constant(&padding))) {
    llvm::SmallVector<int64_t> window_dimensions_vec, window_strides_vec,
        base_dilations_vec, window_dilations_vec;
    llvm::SmallVector<std::pair<int64_t, int64_t>> padding_pairs(
        padding.getNumElements() / 2);

    for (auto i = 0; i < window_dimensions.size(); ++i) {
      window_dimensions_vec.push_back(
          window_dimensions.getValues<IntegerAttr>()[i].getInt());
    }

    for (auto i = 0; i < window_strides.size(); ++i) {
      window_strides_vec.push_back(
          window_strides.getValues<IntegerAttr>()[i].getInt());
    }

    for (auto i = 0; i < base_dilations.size(); ++i) {
      base_dilations_vec.push_back(
          base_dilations.getValues<IntegerAttr>()[i].getInt());
    }

    for (auto i = 0; i < window_dilations.size(); ++i) {
      window_dilations_vec.push_back(
          window_dilations.getValues<IntegerAttr>()[i].getInt());
    }

    for (auto i = 0; i < padding_pairs.size(); ++i) {
      padding_pairs[i] = {padding.getValues<IntegerAttr>()[i * 2].getInt(),
                          padding.getValues<IntegerAttr>()[i * 2 + 1].getInt()};
    }

    auto window = InferWindowFromDimensions(
        window_dimensions_vec, window_strides_vec, padding_pairs,
        base_dilations_vec, window_dilations_vec);
    if (!window) {
      op->emitOpError("failed to create window");
    }
    auto output_shape = InferWindowOutputShape(
        input_ty, window.value(),
        mlir::cast<ShapedType>(op.getInitValue().getType()).getElementType());

    if (!output_shape) {
      op->emitOpError("failed to infer output shape");
    }

    changed = RefineResultType(op.getOperation(), op.getResult(),
                               output_shape.value());
  }

  return changed;
}

bool ShapeInference::InferShapeForXlaSelectAndScatterOp(
    XlaSelectAndScatterOp op) {
  DCOMMENT_OP(op, "Inferring shape for XlaSelectAndScatterOp");

  auto operand_shape = mlir::cast<ShapedType>(op.getOperand().getType());
  auto source_shape = mlir::cast<ShapedType>(op.getSource().getType());
  DenseElementsAttr window_dimensions, window_strides, padding;
  if (operand_shape.hasRank() && source_shape.hasRank() &&
      matchPattern(op.getWindowDimensions(), m_Constant(&window_dimensions)) &&
      matchPattern(op.getWindowStrides(), m_Constant(&window_strides)) &&
      matchPattern(op.getPadding(), m_Constant(&padding))) {
    llvm::SmallVector<int64_t> window_dimensions_vec, window_strides_vec,
        base_dilations_vec, window_dilations_vec;
    llvm::SmallVector<std::pair<int64_t, int64_t>> padding_pairs(
        padding.getNumElements() / 2);

    for (auto i = 0; i < window_dimensions.size(); ++i) {
      window_dimensions_vec.push_back(
          window_dimensions.getValues<IntegerAttr>()[i].getInt());
    }

    for (auto i = 0; i < window_strides.size(); ++i) {
      window_strides_vec.push_back(
          window_strides.getValues<IntegerAttr>()[i].getInt());
    }

    for (auto i = 0; i < padding_pairs.size(); ++i) {
      padding_pairs[i] = {padding.getValues<IntegerAttr>()[i * 2].getInt(),
                          padding.getValues<IntegerAttr>()[i * 2 + 1].getInt()};
    }

    auto window = InferWindowFromDimensions(
        window_dimensions_vec, window_strides_vec, padding_pairs, {}, {});
    if (!window) {
      op->emitOpError("failed to create window");
    }
    auto window_result_shape = InferWindowOutputShape(
        operand_shape, window.value(), operand_shape.getElementType());

    if (!window_result_shape) {
      op->emitOpError("failed to infer window result shape");
    }

    if (window_result_shape.value() != source_shape) {
      op->emitOpError(
          "Source shape does not match the shape of window-reduced operand.");
    }
  }

  return RefineResultType(op.getOperation(), op.getResult(),
                          op.getOperand().getType());
}

bool ShapeInference::InferShapeForXlaGatherOp(XlaGatherOp op) {
  xla::Shape input_shape = xla::TypeToShape(op.getOperand().getType());
  if (input_shape == xla::Shape() || input_shape.is_unbounded_dynamic())
    return false;

  xla::Shape start_indices_shape =
      xla::TypeToShape(op.getStartIndices().getType());
  if (start_indices_shape == xla::Shape()) return false;

  xla::GatherDimensionNumbers gather_dim_numbers;
  if (!gather_dim_numbers.ParseFromString(op.getDimensionNumbers().str()))
    return false;

  DenseIntElementsAttr slice_sizes_attr;
  if (DenseIntElementsAttr attr;
      matchPattern(op.getSliceSizes(), m_Constant(&attr))) {
    slice_sizes_attr = attr;
  } else if (const auto it = results_.find(ValuePort(op.getSliceSizes()));
             it != results_.end() &&
             llvm::isa_and_nonnull<DenseIntElementsAttr>(it->second)) {
    slice_sizes_attr = llvm::cast<DenseIntElementsAttr>(it->second);
  } else {
    return false;
  }

  llvm::SmallVector<int64_t> slice_sizes;
  for (const auto& attr : slice_sizes_attr.getValues<APInt>()) {
    slice_sizes.push_back(attr.getSExtValue());
  }

  auto output_shape = xla::ShapeInference::InferGatherShape(
      input_shape, start_indices_shape, gather_dim_numbers, slice_sizes);
  if (!output_shape.ok()) {
    op->emitError() << output_shape.status().message();
    return false;
  }

  auto refined_type = xla::ConvertShapeToType<RankedTensorType>(
      *output_shape, mlir::Builder(op));
  if (!refined_type.ok()) {
    op->emitError() << refined_type.status().message();
    return false;
  }

  return RefineResultType(op, op.getOutput(), *refined_type);
}

std::optional<RankedTensorType> InferXlaConvOutputShape(
    llvm::SmallVector<int64_t> input_tensor_dims,
    llvm::SmallVector<int64_t> kernel_tensor_dims,
    llvm::SmallVector<int64_t> window_strides,
    llvm::SmallVector<std::pair<int64_t, int64_t>> paddings,
    llvm::SmallVector<int64_t> lhs_dilations,
    llvm::SmallVector<int64_t> rhs_dilations, int64_t batch_group_count,
    xla::ConvolutionDimensionNumbers dnums, Type element_type) {
  auto num_spatial_dims = input_tensor_dims.size() - 2;
  std::vector<int64_t> output_dims(input_tensor_dims.size());

  auto input_batch = input_tensor_dims[dnums.input_batch_dimension()];
  auto kernel_output_feature =
      kernel_tensor_dims[dnums.kernel_output_feature_dimension()];
  output_dims[dnums.output_batch_dimension()] = input_batch / batch_group_count;
  DCOMMENT("inferrd output batch dimension is "
           << output_dims[dnums.output_batch_dimension()]);
  output_dims[dnums.output_feature_dimension()] = kernel_output_feature;
  DCOMMENT("inferrd output output_feature_dimension is "
           << output_dims[dnums.output_feature_dimension()]);

  std::vector<int64_t> input_spatial_dims;
  llvm::SmallVector<int64_t> window_spatial_dims;
  for (auto i = 0; i < num_spatial_dims; ++i) {
    input_spatial_dims.push_back(
        input_tensor_dims[dnums.input_spatial_dimensions(i)]);
    window_spatial_dims.push_back(
        kernel_tensor_dims[dnums.kernel_spatial_dimensions(i)]);
  }

  ShapedType base_shape =
      tensorflow::GetTypeFromTFTensorShape(input_spatial_dims, element_type);

  auto window =
      InferWindowFromDimensions(window_spatial_dims, window_strides, paddings,
                                lhs_dilations, rhs_dilations);

  auto output_shape =
      InferWindowOutputShape(base_shape, window.value(), element_type);

  for (auto i = 0; i < num_spatial_dims; ++i) {
    output_dims[dnums.output_spatial_dimensions(i)] =
        output_shape.value().getShape()[i];
    DCOMMENT("inferrd output spatial dimension "
             << i << " at dimension numebr "
             << dnums.output_spatial_dimensions(i) << " is "
             << output_dims[dnums.output_spatial_dimensions(i)]);
  }
  return tensorflow::GetTypeFromTFTensorShape(output_dims, element_type);
}

// TODO(hanxiongwang): The logic in this function need move to Op Verify method
// when dependecy issue of adding header file
// "third_party/tensorflow/compiler/xla/xla_data.pb.h" into
// "third_party/tensorflow/compiler/mlir/tensorflow/ir/tf_ops_n_z.cc" is
// resolved
LogicalResult PrecheckForXlaConvV2Op(XlaConvV2Op op) {
  auto input_tensor = op.getLhs();
  auto kernel_tensor = op.getRhs();
  auto window_strides = op.getWindowStrides();
  auto padding = op.getPadding();
  auto lhs_dilation = op.getLhsDilation();
  auto rhs_dilation = op.getRhsDilation();
  auto feature_group_count = op.getFeatureGroupCount();
  int64_t batch_group_count = op.getBatchGroupCount();

  auto input_args_have_static_shape = [&]() -> bool {
    return mlir::cast<TensorType>(input_tensor.getType()).hasStaticShape() &&
           mlir::cast<TensorType>(kernel_tensor.getType()).hasStaticShape() &&
           mlir::cast<TensorType>(window_strides.getType()).hasStaticShape() &&
           mlir::cast<TensorType>(padding.getType()).hasStaticShape() &&
           mlir::cast<TensorType>(lhs_dilation.getType()).hasStaticShape() &&
           mlir::cast<TensorType>(rhs_dilation.getType()).hasStaticShape() &&
           mlir::cast<TensorType>(feature_group_count.getType())
               .hasStaticShape();
  };

  // Return failure when one of the input args has not a static shape
  if (!input_args_have_static_shape()) {
    return failure();
  }

  auto input_tensor_shape =
      mlir::cast<RankedTensorType>(input_tensor.getType()).getShape();
  auto kernel_tensor_shape =
      mlir::cast<RankedTensorType>(kernel_tensor.getType()).getShape();

  if (input_tensor_shape.size() <= 2) {
    return op.emitOpError()
           << "input tensor argument is " << input_tensor_shape.size()
           << " which is invalid, since input tensor argument must has a "
           << "rank greater than 2.\n";
  }

  if (kernel_tensor_shape.size() <= 2) {
    return op.emitOpError()
           << "kernel tensor argument is " << kernel_tensor_shape.size()
           << " which is invalid, since kernel tensor argument must has a "
           << "rank greater than 2.\n";
  }

  if (input_tensor_shape.size() != kernel_tensor_shape.size()) {
    return op.emitOpError() << "both input tensor and kernel tensor must "
                            << "have same number of dimensions.\n";
  }

  DenseElementsAttr feature_group_count_attr;
  xla::ConvolutionDimensionNumbers dnums;
  dnums.ParseFromString(op.getDimensionNumbersAttr().getValue().str());
  if (dnums.input_spatial_dimensions_size() !=
      dnums.kernel_spatial_dimensions_size()) {
    return op.emitOpError() << "Both arguments to convolution must have "
                            << "same number of dimensions.\n";
  }

  if (dnums.input_spatial_dimensions_size() !=
      dnums.output_spatial_dimensions_size()) {
    return op.emitOpError() << "Both input and output of convolution must have "
                            << "same number of dimensions.\n";
  }
  if (!matchPattern(feature_group_count,
                    m_Constant(&feature_group_count_attr))) {
    return success();
  }

  auto feature_group_count_val =
      feature_group_count_attr.getValues<IntegerAttr>()[0].getInt();
  auto input_features = input_tensor_shape[dnums.input_feature_dimension()];
  auto input_batch = input_tensor_shape[dnums.input_batch_dimension()];
  auto kernel_input_features =
      kernel_tensor_shape[dnums.kernel_input_feature_dimension()];
  auto kernel_output_features =
      kernel_tensor_shape[dnums.kernel_output_feature_dimension()];

  if (feature_group_count_val <= 0) {
    return op.emitOpError()
           << "feature_group_count must be a positive number, got "
           << feature_group_count_val;
  }

  if (batch_group_count <= 0) {
    return op.emitOpError()
           << "batch_group_count must be a positive number, got "
           << batch_group_count;
  }
  if (batch_group_count > 1 && feature_group_count_val > 1) {
    return op.emitOpError()
           << "both batch_group_count " << batch_group_count
           << "and feature_group_count " << feature_group_count_val
           << " cannot be greater than 1";
  }
  if (kernel_output_features % batch_group_count != 0) {
    return op.emitOpError()
           << "Expected output feature dimension size (value "
           << kernel_output_features
           << ") to be a multiple of batch group count " << batch_group_count;
  }
  if (input_features % feature_group_count_val != 0 ||
      input_features / feature_group_count_val != kernel_input_features) {
    return op.emitOpError()
           << "Expected the size of kernel_input_features (value "
           << kernel_input_features
           << ") in rhs times feature_group_count (value "
           << feature_group_count_val
           << ") in lhs should equal the size of the z dimension (value "
           << input_features << ") in lhs.\n";
  }
  if (kernel_output_features % feature_group_count_val > 0) {
    return op.emitOpError() << "Expected output feature dimension (value "
                            << kernel_output_features << ") to be divisible by "
                            << "feature_group_count (value "
                            << feature_group_count_val << ").\n";
  }
  if (input_batch % batch_group_count != 0) {
    return op.emitOpError()
           << "Expected input batch dimension (value " << input_batch
           << " ) to be divisible by batch_group_count (value "
           << batch_group_count << "); ";
  }
  return success();
}

bool ShapeInference::InferShapeForXlaConvV2Op(XlaConvV2Op op) {
  DCOMMENT_OP(op, "Inferring shape for XlaConvV2Op");

  bool changed = false;

  if (PrecheckForXlaConvV2Op(op).failed()) {
    return changed;
  }

  auto input_tensor = op.getLhs();
  auto kernel_tensor = op.getRhs();
  auto window_strides = op.getWindowStrides();
  auto padding = op.getPadding();
  auto lhs_dilation = op.getLhsDilation();
  auto rhs_dilation = op.getRhsDilation();
  int64_t batch_group_count = op.getBatchGroupCount();

  DenseIntElementsAttr window_strides_attr, padding_attr, lhs_dilation_attr,
      rhs_dilation_attr;
  if (matchPattern(window_strides, m_Constant(&window_strides_attr)) &&
      matchPattern(padding, m_Constant(&padding_attr)) &&
      matchPattern(lhs_dilation, m_Constant(&lhs_dilation_attr)) &&
      matchPattern(rhs_dilation, m_Constant(&rhs_dilation_attr))) {
    llvm::SmallVector<int64_t> input_tensor_dims_vec, kernel_tensor_dims_vec,
        window_strides_vec, lhs_dilations_vec, rhs_dilations_vec;
    llvm::SmallVector<std::pair<int64_t, int64_t>> padding_pairs(
        padding_attr.getNumElements() / 2);
    xla::ConvolutionDimensionNumbers dnums;
    dnums.ParseFromString(op.getDimensionNumbersAttr().getValue().str());

    auto input_tensor_shape =
        mlir::cast<RankedTensorType>(input_tensor.getType());
    for (auto i = 0; i < input_tensor_shape.getShape().size(); ++i) {
      DCOMMENT("Input Tensor Shape " << i << "th is "
                                     << input_tensor_shape.getShape()[i]);
      input_tensor_dims_vec.push_back(input_tensor_shape.getShape()[i]);
    }

    auto kernel_tensor_shape =
        mlir::cast<RankedTensorType>(kernel_tensor.getType());
    for (auto i = 0; i < kernel_tensor_shape.getShape().size(); ++i) {
      DCOMMENT("Kernel tensor Shape" << i << "th is "
                                     << kernel_tensor_shape.getShape()[i]);
      kernel_tensor_dims_vec.push_back(kernel_tensor_shape.getShape()[i]);
    }

    for (const llvm::APInt& i : window_strides_attr) {
      window_strides_vec.push_back(i.getSExtValue());
    }

    for (auto i = 0; i < padding_pairs.size(); ++i) {
      padding_pairs[i] = {
          padding_attr.getValues<IntegerAttr>()[i * 2].getInt(),
          padding_attr.getValues<IntegerAttr>()[i * 2 + 1].getInt()};
    }

    for (const llvm::APInt& i : lhs_dilation_attr) {
      lhs_dilations_vec.push_back(i.getSExtValue());
    }

    for (const llvm::APInt& i : rhs_dilation_attr) {
      rhs_dilations_vec.push_back(i.getSExtValue());
    }

    Type input_tensor_element_type = input_tensor_shape.getElementType();
    Type result_element_type = op.getType().getElementType();
    Type element_type = input_tensor_element_type.getIntOrFloatBitWidth() >=
                                result_element_type.getIntOrFloatBitWidth()
                            ? input_tensor_element_type
                            : result_element_type;
    auto output_shape = InferXlaConvOutputShape(
        input_tensor_dims_vec, kernel_tensor_dims_vec, window_strides_vec,
        padding_pairs, lhs_dilations_vec, rhs_dilations_vec, batch_group_count,
        dnums, element_type);

    if (output_shape.value()) {
      changed = RefineResultType(op.getOperation(), op.getResult(),
                                 output_shape.value());
      return changed;
    }
  }
  return changed;
}

bool ShapeInference::RefineWithInferTypeOpInterface(
    InferTypeOpInterface infer_ti) {
  Operation* op = infer_ti.getOperation();
  if (none_of(op->getResultTypes(), CanBeRefined)) {
    LLVM_DEBUG(llvm::dbgs() << "Skipping inference for statically shaped op '"
                            << op->getName() << "'.\n");
    return false;
  }

  SmallVector<Type, 4> inferred;
  LogicalResult res = infer_ti.inferReturnTypes(
      op->getContext(), op->getLoc(), op->getOperands(),
      op->getAttrDictionary(), op->getPropertiesStorage(), op->getRegions(),
      inferred);
  if (failed(res)) {
    op->emitOpError("failed to refine type as inference failed");
    return false;
  }

  if (inferred == op->getResultTypes()) return false;

  // Map each of the results of the call to the returned type of the
  // function.
  bool changed = false;
  for (auto [result, inferred_type] : zip(op->getResults(), inferred)) {
    auto result_type = result.getType();
    auto new_type = inferred_type;
    if (!llvm::isa<TF::ZerosLikeOp>(op)) {
      // TODO: b/361179755 - Remove when folding issue is resolved or proper
      // shape inference is confirmed for zeros like.
      new_type = TypeMeet(inferred_type, result_type);
    }
    if (new_type == result_type) {
      continue;
    }
    if (!UpdateTypeAndInsertIncompatibleUseCasts(new_type, result)) {
      continue;
    }
    changed = true;
  }
  return changed;
}

ShapeHandle ShapeInference::ComputeOutputAsShape(OpResult result,
                                                 InferenceContext* ic) {
  LLVM_DEBUG(result.print(llvm::dbgs() << "\nEvaluate partially "));
  auto rt = mlir::dyn_cast<RankedTensorType>(result.getType());
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
        if (auto dea = mlir::dyn_cast<DenseIntElementsAttr>(ret)) {
          if (dea.getNumElements() != 1) {
            LLVM_DEBUG(llvm::dbgs() << "Unexpected number of elements\n");
            return {};
          }
          int64_t val = (*dea.getValues<APInt>().begin()).getSExtValue();
          dims[i] = ic->MakeDim(val);
        }
      }
    }
  }
  return ic->MakeShape(dims);
}

bool ShapeInference::ForceTypeForPassThroughOperands(Operation* op,
                                                     OperandRange operands,
                                                     ResultRange results) {
  bool changed = false;
  for (auto entry : llvm::zip(operands, results)) {
    Type operand_type = std::get<0>(entry).getType();
    Value result = std::get<1>(entry);
    TensorType result_type = dyn_cast<TensorType>(result.getType());
    if (result_type == operand_type) continue;

    if (!UpdateTypeAndInsertIncompatibleUseCasts(operand_type, result))
      continue;
    changed = true;
  }
  return changed;
}

bool ShapeInference::RefineTypeForPassThroughOperands(Operation* op,
                                                      OperandRange operands,
                                                      ResultRange results) {
  bool changed = false;
  for (auto entry : llvm::zip(operands, results)) {
    Type operand_type = std::get<0>(entry).getType();
    Value result = std::get<1>(entry);
    TensorType result_type = mlir::cast<TensorType>(result.getType());
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
    return ForceTypeForPassThroughOperands(graph_op.GetFetch(),
                                           graph_op.GetFetch().getFetches(),
                                           op->getResults());
  }
  if (auto island_op = dyn_cast<tf_executor::IslandOp>(op)) {
    return ForceTypeForPassThroughOperands(island_op.GetYield(),
                                           island_op.GetYield().getFetches(),
                                           op->getResults());
  }
  if (auto iter_sink = dyn_cast<tf_executor::NextIterationSinkOp>(op)) {
    auto iter_source = cast<tf_executor::NextIterationSourceOp>(
        iter_sink.getToken().getDefiningOp());
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
  if (isa<tensor::CastOp>(op)) return InferShapeForCast(op);
  return false;
}

// Finds element type to be used for result from operand, with special handling
// for handle types.
Type GetElementTypeFromOperand(TensorType operand_type,
                               TensorType result_type) {
  auto operand_handle_type =
      mlir::dyn_cast<TensorFlowTypeWithSubtype>(operand_type.getElementType());
  if (!operand_handle_type) return result_type.getElementType();
  auto result_handle_type =
      mlir::cast<TensorFlowTypeWithSubtype>(result_type.getElementType());
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
        current_dim != ShapedType::kDynamic)
      return false;
  }
  return true;
}

template <typename WhileOpTy>
bool ShapeInference::InferShapeForWhile(WhileOpTy op,
                                        TypeRange body_result_types) {
  if (!op.getShapeInvariant())
    return RefineTypeForPassThroughOperands(op, op.getInput(), op.getOutput());

  bool changed = false;
  for (auto entry :
       zip(op.getInput().getTypes(), op.getOutput(), body_result_types)) {
    Value result = std::get<1>(entry);
    TensorType body_result_type = mlir::cast<TensorType>(std::get<2>(entry));
    auto result_type = mlir::cast<TensorType>(result.getType());

    Type potential_refined_type;
    if (CanWhileTypeBeRefinedWith(result_type, body_result_type)) {
      Type element_type =
          GetElementTypeFromOperand(body_result_type, result_type);
      potential_refined_type = CreateTensorType(
          body_result_type.hasRank() ? body_result_type.getShape()
                                     : std::optional<ArrayRef<int64_t>>(),
          element_type);
    } else {
      TensorType operand_type = mlir::cast<TensorType>(std::get<0>(entry));
      Type element_type = GetElementTypeFromOperand(operand_type, result_type);
      potential_refined_type = CreateTensorType(
          result_type.hasRank() ? result_type.getShape()
                                : std::optional<ArrayRef<int64_t>>(),
          element_type);
    }
    changed |= RefineResultType(op, result, potential_refined_type);
  }
  return changed;
}

bool ShapeInference::InferShapeForSingleOperation(Operation* op,
                                                  int64_t max_iterations) {
  LLVM_DEBUG(op->print(llvm::dbgs() << "InferShapeForSingleOperation for ");
             llvm::dbgs() << "\n");
  assert(tf_dialect_ == op->getDialect());
  // The shape function of these ops sometimes does not propagate subtypes
  // (handle shapes) for resource and variant types. We use a simple passthrough
  // to make sure they are preserved in the output.
  if (isa<TF::IdentityOp, TF::IdentityNOp, TF::StopGradientOp, TF::ZerosLikeOp,
          TF::XlaShardingOp>(op)) {
    return RefineTypeForPassThroughOperands(op, op->getOperands(),
                                            op->getResults());
  }

  // The shape inference function for `ReduceDatasetOp` should always be
  // executed regardless of whether the result type can be refined.
  if (auto reduce_dataset_op = dyn_cast<ReduceDatasetOp>(op)) {
    // TODO(jpienaar): The output type of these ops need to be refined.
    return InferShapeForReduceDataset(reduce_dataset_op, max_iterations);
  }

  // If no result for this op needs shape inference, we have a fast-path return.
  // But if the type is a resource/variant, we do not skip it because we might
  // not have the handle shapes.
  if (none_of(op->getResultTypes(), CanBeRefined)) {
    if (auto host_compute_op = dyn_cast<XlaHostComputeOp>(op)) {
      LLVM_DEBUG(llvm::dbgs() << "Keep inference for statically shaped op '"
                              << op->getName() << "'.\n");
    } else {
      LLVM_DEBUG(llvm::dbgs() << "Skipping inference for statically shaped op '"
                              << op->getName() << "'.\n");
      return false;
    }
  }

  if (isa<TF::RestoreOp, TF::RestoreV2Op>(op)) return InferShapeForRestore(op);

  // Handle call operations by looking up callee and inferring return shape as
  // needed.
  if (auto call = dyn_cast<CallOpInterface>(op)) return InferShapeForCall(call);

  // tf.Cast is only inferred if it has at least one user in the TF dialect or
  // feeding into the function return. This is necessary to avoid inserting
  // casts which cannot be refined.
  if (isa<CastOp>(op)) return InferShapeForCast(op);

  // Handle IfOp here by inferring the shape from the else/then function
  // results. Since `output_shapes` is a derived attribute, avoid going down the
  // TF InferenceContext path as IfOp shape inference is implemented as just
  // a lookup of the output_shapes attribute.
  if (auto if_op = dyn_cast<IfOp>(op)) return InferShapeForIf(if_op);

  // Handle IfRegion operations by inferring return shape from the then and else
  // branches.
  if (auto if_region = dyn_cast<IfRegionOp>(op))
    return InferShapeForIfRegion(if_region);

  if (auto case_op = dyn_cast<CaseOp>(op)) return InferShapeForCase(case_op);

  if (auto case_region = dyn_cast<CaseRegionOp>(op))
    return InferShapeForCaseRegion(case_region);

  if (auto while_op = dyn_cast<WhileOp>(op))
    return InferShapeForWhile(
        while_op, while_op.body_function().getFunctionType().getResults());

  if (auto while_region = dyn_cast<WhileRegionOp>(op))
    return InferShapeForWhile(
        while_region,
        while_region.getBody().front().getTerminator()->getOperandTypes());

  if (auto xla_call_module = dyn_cast<XlaCallModuleOp>(op)) {
    return InferShapeForXlaCallModule(xla_call_module);
  }

  if (auto host_compute_op = dyn_cast<_XlaHostComputeMlirOp>(op)) {
    return InferShapeForXlaHostComputeMlir(host_compute_op);
  }

  if (auto host_compute_op = dyn_cast<XlaHostComputeOp>(op)) {
    return InferShapeForFunctionAttachedToXlaHostCompute(host_compute_op);
  }

  // TODO(jpienaar): Extract function input arg constraint interface.
  // TODO(jpienaar): Unify the shape propagation to functions using interface.
  if (auto map_dataset_op = dyn_cast<MapDatasetOp>(op)) {
    // TODO(jpienaar): The output type of these ops need to be refined.
    return InferShapeForMapDataset(map_dataset_op, max_iterations);
  }

  if (auto takewhile_dataset_op = dyn_cast<TakeWhileDatasetOp>(op)) {
    // TODO(jpienaar): The output type of these ops need to be refined.
    return InferShapeForTakeWhileDataset(takewhile_dataset_op, max_iterations);
  }

  // Handle TensorList init operations by inferring shape from TensorList write
  // operations. If we are unable to refine element shape here, proceed to use
  // the InferenceContext below to get more precise shapes.
  if (IsTensorListInitOp(op) && InferShapeForTensorListInitOps(op)) return true;

  if (auto pop_back = dyn_cast<TF::TensorListPopBackOp>(op)) {
    return InferShapeForTensorListPopBackOp(pop_back);
  }

  if (auto var_handle_op = dyn_cast<VarHandleOp>(op)) {
    return InferShapeForVarHandleOp(var_handle_op);
  }

  if (auto xla_reduce_window_op = dyn_cast<XlaReduceWindowOp>(op)) {
    return InferShapeForXlaReduceWindowOp(xla_reduce_window_op);
  }

  if (auto xla_select_and_scatter_op = dyn_cast<XlaSelectAndScatterOp>(op)) {
    return InferShapeForXlaSelectAndScatterOp(xla_select_and_scatter_op);
  }

  if (auto xla_conv_v2_op = dyn_cast<XlaConvV2Op>(op)) {
    return InferShapeForXlaConvV2Op(xla_conv_v2_op);
  }

  if (auto xla_gather_op = dyn_cast<XlaGatherOp>(op)) {
    return InferShapeForXlaGatherOp(xla_gather_op);
  }

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
    return mlir::cast<TensorType>(op->getResult(index).getType())
        .getElementType();
  };

  llvm::SmallVector<ShapedTypeComponents, 4> inferred_return_shapes;
  if (failed(InferReturnTypeComponentsForTFOp(
          /*location=*/std::nullopt, op, graph_version_, operand_as_constant_fn,
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
    if (inferred.hasRank()) {
      inferred_type =
          RankedTensorType::get(inferred.getDims(), inferred.getElementType());

    } else {
      inferred_type = UnrankedTensorType::get(inferred.getElementType());
    }
    inferred_type =
        mlir::cast<TensorType>(TypeMeet(op_result.getType(), inferred_type));
    if (op_result.getType() == inferred_type) continue;
    if (!UpdateTypeAndInsertIncompatibleUseCasts(inferred_type, op_result))
      continue;
    changed = true;
  }

  if (changed) DCOMMENT_OP(op, "Modified after shape inference:");
  return changed;
}

FailureOr<bool> ShapeInference::PropagateShapeToFunctions(
    ModuleOp module, TypeRange input_types, ArrayRef<func::FuncOp> functions,
    int64_t max_iterations) {
  bool any_failure = false;
  bool any_nonconvergence = false;
  // If shape propagation fails for one function, return failure, but do not
  // early exit and attempt to propagate shapes for all provided functions to
  // have a best-effort propagation.
  for (func::FuncOp func : functions) {
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
    FunctionType func_type = func.getFunctionType();
    func.setType(FunctionType::get(func.getContext(), input_types,
                                   func_type.getResults()));

    FailureOr<bool> failure_or_converged =
        PropagateShapeToRegions(input_types, {&func.getBody()}, max_iterations);
    if (failed(failure_or_converged)) {
      any_failure = true;
      continue;
    }
    any_nonconvergence = any_nonconvergence || !failure_or_converged.value();
    if (failed(InferShapeForFunctionReturnType(func))) any_failure = true;
  }
  if (any_failure) return failure();
  return any_nonconvergence;
}

FailureOr<bool> ShapeInference::PropagateShapeToRegions(
    TypeRange input_types, ArrayRef<Region*> regions, int64_t max_iterations) {
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
        InferShapeUntilFixPoint(region, max_iterations);
    if (failed(failure_or_converged))
      any_failure = true;
    else if (!failure_or_converged.value())
      any_nonconvergence = true;
  }
  if (any_failure) return failure();
  return any_nonconvergence;
}

void ShapeInference::PropagateConstantToCallee(CallOpInterface call_op,
                                               func::FuncOp func,
                                               ModuleOp module) {
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
                                                 func::FuncOp func,
                                                 ModuleOp module) {
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
      dims.push_back(ShapedType::kDynamic);
    }
  }
  return tensorflow::GetTypeFromTFTensorShape(
      dims, GetElementTypeFromOperand(lhs, rhs));
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
    auto operand_type = mlir::cast<TensorType>(std::get<0>(entry));
    auto result_type = mlir::cast<TensorType>(std::get<1>(entry));
    if (operand_type == result_type) {
      types.push_back(operand_type);
    } else if (RankedAndSameRank(operand_type, result_type)) {
      auto potential_refined_type = GetCompatibleRankedTensorType(
          mlir::cast<RankedTensorType>(operand_type),
          mlir::cast<RankedTensorType>(result_type));
      types.push_back(potential_refined_type);
    } else {
      auto region_argument_type = mlir::cast<TensorType>(std::get<2>(entry));
      Type element_type = GetElementTypeFromOperand(
          mlir::cast<TensorType>(operand_type), region_argument_type);
      Type potential_refined_type = CreateTensorType(
          region_argument_type.hasRank() ? region_argument_type.getShape()
                                         : std::optional<ArrayRef<int64_t>>(),
          element_type);
      types.push_back(potential_refined_type);
    }
  }
  return types;
}

FailureOr<bool> ShapeInference::PropagateShapeIntoAttachedFunctions(
    Operation* op, int64_t max_iterations) {
  ModuleOp module = op->getParentOfType<ModuleOp>();
  if (auto if_op = dyn_cast<TF::IfOp>(op)) {
    DCOMMENT("Propagating shapes into If");
    return PropagateShapeToFunctions(
        module, if_op.getInput().getTypes(),
        {if_op.ResolveThenFunction(&symbol_table_),
         if_op.ResolveElseFunction(&symbol_table_)},
        max_iterations);
  } else if (auto case_op = dyn_cast<TF::CaseOp>(op)) {
    SmallVector<func::FuncOp, 4> branches;
    case_op.get_branch_functions(branches);
    return PropagateShapeToFunctions(module, case_op.getInput().getTypes(),
                                     branches, max_iterations);
  } else if (auto while_op = dyn_cast<TF::WhileOp>(op)) {
    // If `shape_invariant` is set, operand shapes cannot be simply propagated
    // to result shapes as the op may have different intermediate shapes (such
    // While ops can have different result shapes from operand shapes).
    // Compatible shapes must be determined before propagating them.
    if (while_op.getShapeInvariant()) {
      auto compatible_types = GetWhileCompatibleTypes(
          while_op.getInput().getTypes(), while_op.getOutput().getTypes(),
          while_op.ResolveBodyFunction(&symbol_table_)
              .getFunctionType()
              .getInputs());
      return PropagateShapeToFunctions(
          module, compatible_types,
          {while_op.ResolveCondFunction(&symbol_table_),
           while_op.ResolveBodyFunction(&symbol_table_)},
          max_iterations);
    }
    return PropagateShapeToFunctions(
        module, while_op.getInput().getTypes(),
        {while_op.ResolveCondFunction(&symbol_table_),
         while_op.ResolveBodyFunction(&symbol_table_)},
        max_iterations);
  } else if (auto call_op = dyn_cast<CallOpInterface>(op)) {
    if (auto func = dyn_cast<func::FuncOp>(
            call_op.resolveCallableInTable(&symbol_table_))) {
      PropagateConstantToCallee(call_op, func, module);
      FailureOr<bool> failure_or_converged = PropagateShapeToFunctions(
          module, call_op.getArgOperands().getTypes(), {func}, max_iterations);
      if (failed(failure_or_converged)) return failure();
      PropagateConstantFromCallee(call_op, func, module);
      return failure_or_converged;
    }
  } else if (isa<TF::XlaReduceWindowOp>(op) ||
             isa<TF::XlaSelectAndScatterOp>(op) ||
             isa<TF::XlaVariadicReduceV2Op>(op) ||
             isa<TF::XlaVariadicSortOp>(op)) {
    auto propagate_shape_to = [&](mlir::SymbolRefAttr func_sym) {
      auto func = llvm::cast<mlir::func::FuncOp>(
          mlir::SymbolTable::lookupSymbolIn(module, func_sym));
      mlir::SmallVector<mlir::Type, 2> types;
      for (auto type : func.getFunctionType().getInputs()) {
        types.push_back(tensorflow::GetTypeFromTFTensorShape(
            {}, getElementTypeOrSelf(type)));
      }
      return PropagateShapeToFunctions(module, types, {func}, max_iterations);
    };

    if (auto xla_reduce_window_op = dyn_cast<TF::XlaReduceWindowOp>(op)) {
      return propagate_shape_to(xla_reduce_window_op.getComputation());
    }
    if (auto xla_select_and_scatter_op =
            dyn_cast<TF::XlaSelectAndScatterOp>(op)) {
      return propagate_shape_to(xla_select_and_scatter_op.getSelect())
                 .value() &&
             propagate_shape_to(xla_select_and_scatter_op.getScatter()).value();
    } else if (auto xla_variadic_reduce_v2_op =
                   dyn_cast<TF::XlaVariadicReduceV2Op>(op)) {
      return propagate_shape_to(xla_variadic_reduce_v2_op.getReducer());
    } else if (auto xla_variadic_sort_op =
                   dyn_cast<TF::XlaVariadicSortOp>(op)) {
      return propagate_shape_to(xla_variadic_sort_op.getComparator());
    }
  }

  // TODO(ycao): Implement support for Call op, including function reuse.

  return true;
}

FailureOr<bool> ShapeInference::PropagateShapeIntoAttachedRegions(
    Operation* op, int64_t max_iterations) {
  if (auto while_op = dyn_cast<TF::WhileRegionOp>(op)) {
    // If `shape_invariant` is set, operand shapes cannot be simply propagated
    // to result shapes as the op may have different intermediate shapes (such
    // While ops can have different result shapes from operand shapes).
    // Compatible shapes must be determined before propagating them.
    if (while_op.getShapeInvariant()) {
      auto compatible_types = GetWhileCompatibleTypes(
          while_op.getInput().getTypes(), while_op.getOutput().getTypes(),
          while_op.getBody().getArgumentTypes());
      return PropagateShapeToRegions(compatible_types,
                                     {&while_op.getCond(), &while_op.getBody()},
                                     max_iterations);
    }
    return PropagateShapeToRegions(while_op.getInput().getTypes(),
                                   {&while_op.getCond(), &while_op.getBody()},
                                   max_iterations);
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
  auto abstract_op = op->getRegisteredInfo();
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
      assert(value.getType() == std::get<0>(result).getType() &&
             "folder produced value of incorrect type");
      if ((attr = ComputeOutputComponent(ValuePort(value)))) {
        DCOMMENT("\t\tValue Result mapped to " << attr);
        RecordValue(ValuePort(std::get<0>(result)), attr);
      } else {
        DCOMMENT("\t\tValue result unmapped, consider value type:" << value);
        RefineResultType(op, std::get<0>(result), value.getType());
      }
    }

    if (ElementsAttr eattr = mlir::dyn_cast_or_null<ElementsAttr>(attr)) {
      if (std::get<0>(result).getType() == eattr.getType()) continue;

      (void)UpdateTypeAndInsertIncompatibleUseCasts(eattr.getType(),
                                                    std::get<0>(result));
    }
  }

  return success();
}

LogicalResult ShapeInference::InferShapeForFunctionReturnType(
    func::FuncOp func) {
  LLVM_DEBUG(llvm::dbgs() << "Inferring return type for: " << func.getName()
                          << "\n");

  // Find any return ops.
  SmallVector<func::ReturnOp, 4> return_ops;
  for (Block& block : func) {
    if (auto return_op = dyn_cast<func::ReturnOp>(block.getTerminator())) {
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

FailureOr<bool> ShapeInference::InferShapeUntilFixPoint(
    Region* region, int64_t max_iterations) {
  bool changed = true;

  // TODO(aminim): we could have a more efficient traversal by guiding the
  // traversal with a worklist and reconsider only the nodes for which an
  // operand type was inferred. This would need to be careful if working on a
  // region that would not be isolated.
  for (int iteration = 0; iteration < max_iterations && changed; ++iteration) {
    changed = false;
    LLVM_DEBUG(llvm::dbgs()
               << "Shape inference, iteration " << iteration << "\n");
    auto res = region->walk([&](Operation* op) {
      auto abstract_op = op->getRegisteredInfo();
      if (abstract_op && ops_to_skip_.contains(abstract_op->getTypeID())) {
        LLVM_DEBUG(llvm::dbgs()
                   << "Skipping shape inference for explicitly skipped op '"
                   << op->getName() << "'.\n");
        return WalkResult::advance();
      }

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
      if (failed(PropagateShapeIntoAttachedFunctions(op, max_iterations))) {
        op->emitWarning() << "unable to refine shape of attached function "
                             "arguments and bodies";
        return WalkResult::interrupt();
      }

      if (failed(PropagateShapeIntoAttachedRegions(op, max_iterations))) {
        op->emitWarning() << "unable to refine shape of attached region "
                             "arguments and bodies";
        return WalkResult::interrupt();
      }

      changed |= InferShapeForSingleOperation(op, max_iterations);
      return WalkResult::advance();
    });
    if (res.wasInterrupted()) return failure();
  }

  if (changed) {
    region->getParentOp()->emitWarning()
        << "shape inference did not reach stable state after " << max_iterations
        << " iterations";
  }
  return !changed;
}

static FailureOr<bool> InferShapeForFunction(ShapeInference& context,
                                             func::FuncOp func,
                                             int64_t max_iterations) {
  FailureOr<bool> failure_or_converged =
      context.InferShapeUntilFixPoint(&func.getBody(), max_iterations);
  if (failed(failure_or_converged) || !failure_or_converged.value())
    return failure_or_converged;
  // TODO(b/156276510): Verify that it is always fine to refine a function's
  // return type, as long as we do not change the argument shapes.
  if (failed(context.InferShapeForFunctionReturnType(func))) return failure();
  return true;
}

absl::StatusOr<SmallVector<SmallVector<int64_t>>> ParseArgumentShapes(
    absl::string_view input_shapes) {
  SmallVector<SmallVector<int64_t>> parsed_shapes;
  if (input_shapes.empty()) {
    return parsed_shapes;
  }

  std::vector<std::optional<std::vector<int>>> shapes;
  TF_RETURN_IF_ERROR(::tensorflow::ParseNodeShapes(input_shapes, shapes));

  for (const auto& shape : shapes) {
    if (!shape) {
      return absl::AbortedError("Missing input argument shapes");
    }
    parsed_shapes.push_back(SmallVector<int64_t>(shape->begin(), shape->end()));
  }
  return parsed_shapes;
}

FailureOr<bool> InferShapeForFunction(func::FuncOp func,
                                      ArrayRef<ArrayRef<int64_t>> arg_shapes,
                                      int64_t graph_version,
                                      int64_t max_iterations,
                                      ArrayRef<TypeID> ops_to_skip) {
  ShapeInference context(graph_version, func->getParentOfType<ModuleOp>(),
                         /*propagate_caller_callee_constants=*/true,
                         ops_to_skip);
  if (arg_shapes.empty()) {
    return InferShapeForFunction(context, func, max_iterations);
  }

  FunctionType func_type = func.getFunctionType();
  bool needs_refinement = false;
  SmallVector<Type, 4> new_arg_types;
  new_arg_types.reserve(func_type.getNumInputs());

  // Update argument types in-place using the provided arg_shapes.
  for (size_t i = 0; i < func_type.getNumInputs(); ++i) {
    ArrayRef<int64_t> shape = arg_shapes[i];
    Type element_type;
    if (auto input_ty =
            mlir::dyn_cast<RankedTensorType>(func_type.getInput(i))) {
      if (input_ty.getRank() != shape.size()) {
        return failure();
      }
      element_type = input_ty.getElementType();
    } else {
      auto unranked_input_ty =
          mlir::dyn_cast<TensorType>(func_type.getInput(i));
      if (!unranked_input_ty) {
        return failure();
      }
      element_type = unranked_input_ty.getElementType();
    }

    auto new_arg_type = GetNewArgType(func_type.getInput(i), shape,
                                      element_type, func.getContext());

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
  if (failed(failure_or_converged) || !failure_or_converged.value())
    return failure_or_converged;

  if (failed(context.InferShapeForFunctionReturnType(func))) return failure();
  func.setType(FunctionType::get(func.getContext(), new_arg_types,
                                 func.getFunctionType().getResults()));

  return true;
}

FailureOr<bool> InferModuleShape(ModuleOp module, int64_t max_iterations,
                                 ArrayRef<TypeID> ops_to_skip,
                                 ArrayRef<ArrayRef<int64_t>> input_shapes,
                                 bool enable_stablehlo_propagation) {
  auto producer_or = tensorflow::GetTfGraphProducerVersion(module);
  if (!producer_or.ok()) {
    // TODO(jpienaar): Keeping the existing behavior for now but this could
    // be relaxed.
    LLVM_DEBUG(llvm::dbgs()
               << "Skipping inference; " << producer_or.status().ToString());
    return true;
  }
  int64_t producer = producer_or.value();

  // TODO(jpienaar): Clean up propagate_NextIterationSinkOp_callee_constants if
  // it is no longer needed.
  ShapeInference context(producer, module,
                         /*propagate_caller_callee_constants=*/false,
                         ops_to_skip);
  auto main = module.lookupSymbol<mlir::func::FuncOp>("main");
  // Error if no main to refine with input shapes
  if (!main && !input_shapes.empty()) {
    return module->emitError(
        "Input shapes provided but no `main` function found.");
  }

  // Add main function to head of queue, refine input shapes if provided
  if (main) {
    if (!input_shapes.empty()) {
      FailureOr<bool> failure_or_converged =
          InferShapeForFunction(main, input_shapes, producer,
                                /*max_iterations=*/10, ops_to_skip);
      if (failed(failure_or_converged) || !failure_or_converged.value())
        return failure_or_converged;
    }
    context.enqueue(main);
  }
  for (auto func : module.getOps<func::FuncOp>()) context.enqueue(func);
  // Arbitrarily upper bound the maximum number of functions that get processed
  // just to avoid pathological cases.
  auto max_iteration = context.QueueSize() * 4;
  while (!context.EmptyQueue()) {
    func::FuncOp func = context.front();
    FailureOr<bool> failure_or_converged =
        InferShapeForFunction(context, func, max_iterations);
    if (failed(failure_or_converged) || !failure_or_converged.value())
      return failure_or_converged;
    context.pop_front();

    if ((--max_iteration) == 0) {
      emitWarning(UnknownLoc::get(module.getContext()))
          << "shape inference did not reach stable state after "
          << max_iteration << " iterations";
      return false;
    }
  }

  // Propagate shapes to StableHLO modules, if enabled and there are any
  // XlaCallModule ops whose StableHLO modules were refined.
  if (enable_stablehlo_propagation) {
    if (failed(context.PropagateShapeToStableHloModules())) return failure();
  }

  return true;
}

}  // namespace TF
}  // namespace mlir
