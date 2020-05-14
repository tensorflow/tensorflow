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

#include <cstdint>
#include <initializer_list>
#include <iterator>

#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/PointerUnion.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Block.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/Diagnostics.h"  // from @llvm-project
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/IR/Module.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/OperationSupport.h"  // from @llvm-project
#include "mlir/IR/StandardTypes.h"  // from @llvm-project
#include "mlir/IR/SymbolTable.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
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
#include "tensorflow/compiler/mlir/tensorflow/translate/export_tf_dialect_op.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/convert_tensor.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/convert_type.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/types.pb.h"

#define DEBUG_TYPE "tf-shape-inference"

using ::tensorflow::int64;
using tensorflow::shape_inference::DimensionHandle;
using tensorflow::shape_inference::InferenceContext;
using tensorflow::shape_inference::ShapeHandle;

namespace mlir {
namespace TF {
namespace {
Optional<SmallVector<Type, 4>> InferShapeForFunctionReturnType(FuncOp func) {
  // Find any return ops.
  SmallVector<ReturnOp, 4> return_ops;
  for (Block& block : func) {
    if (auto return_op = dyn_cast<ReturnOp>(block.getTerminator())) {
      return_ops.push_back(return_op);
    }
  }

  // Right now we only handle the case of a single return op.
  // To handle multiple return ops, we would need to look at all their shapes
  // and come up with a common shape and insert appropriate casts.
  if (return_ops.size() != 1) {
    return None;
  }

  // Find the return type.
  auto return_op = return_ops.front();

  // Manually fold tf.Cast that precedes the return instruction and only differs
  // in shape refinement level.
  for (OpOperand& arg_op : return_op.getOperation()->getOpOperands()) {
    Operation* arg_defining_op = arg_op.get().getDefiningOp();
    if (auto cast_op = dyn_cast_or_null<CastOp>(arg_defining_op)) {
      // Shape inference should not change the element type.
      if (cast_op.SrcT() != cast_op.DstT()) continue;
      // We only refine the result shape if the result a dynamic shape, the
      // input has static shape, and the two shapes are compatible.
      auto has_static_shape = [](const Value value) {
        auto shaped_type = value.getType().dyn_cast<ShapedType>();
        return shaped_type && shaped_type.hasStaticShape();
      };
      Value input = cast_op.x();
      Value result = cast_op.y();
      if (!has_static_shape(input) || has_static_shape(result) ||
          failed(verifyCompatibleShape(input.getType(), result.getType())))
        continue;

      arg_op.set(cast_op.x());
      if (cast_op.y().use_empty()) cast_op.erase();
    }
  }

  return llvm::to_vector<4>(return_op.getOperandTypes());
}

// Returns if the shape inference pass supports an op outside the TF dialect.
bool IsSupportedNonTFOp(Operation* op) {
  return isa<tf_executor::YieldOp>(op) || isa<tf_executor::IslandOp>(op) ||
         isa<tf_executor::FetchOp>(op) || isa<tf_executor::GraphOp>(op) ||
         isa<tf_executor::NextIterationSinkOp>(op) || isa<ReturnOp>(op) ||
         isa<tf_device::ReturnOp>(op) || isa<tf_executor::MergeOp>(op) ||
         isa<tf_executor::SwitchOp>(op) || isa<tf_executor::SwitchNOp>(op) ||
         isa<tf_executor::EnterOp>(op) || isa<tf_executor::ExitOp>(op);
}

// Inserts tf.Cast operation when changing the type of a result if the user is
// not a TF operation, as we can't guarantee that the new type will be OK.
void AddCastBackForUnsupportedNonTFUses(Operation* op, Value result,
                                        Dialect* tf_dialect, Type old_type) {
  // A tf.Cast operation is lazily created on the first uses that isn't a TF
  // operation.
  TF::CastOp cast_op;
  auto get_cast_op = [&]() {
    if (!cast_op) {
      OpBuilder b(op);
      b.setInsertionPointAfter(op);
      cast_op = b.create<TF::CastOp>(op->getLoc(), old_type, result,
                                     /*truncate=*/b.getBoolAttr(false));
    }
    return Value(cast_op);
  };
  for (OpOperand& use : make_early_inc_range(result.getUses())) {
    if (use.getOwner()->getDialect() != tf_dialect &&
        !IsSupportedNonTFOp(use.getOwner()))
      use.set(get_cast_op());
  }
}

// Extracts a PartialTensorShape from the MLIR type.
Optional<tensorflow::PartialTensorShape> GetShapeFromMlirType(Type t) {
  if (auto ranked_type = t.dyn_cast<RankedTensorType>()) {
    // Convert the MLIR shape indices (int64_t) to TensorFlow indices
    // (int64).
    ArrayRef<int64_t> shape = ranked_type.getShape();
    SmallVector<int64, 8> tf_shape(shape.begin(), shape.end());
    return tensorflow::PartialTensorShape({tf_shape.data(), tf_shape.size()});
  }
  return None;
}

// Passes the operand shapes/types to the op's results.
bool InferShapeForPassThroughOps(OperandRange pass_through_operands,
                                 Operation* op, Dialect* tf_dialect) {
  bool changed = false;
  for (auto entry : zip(pass_through_operands, op->getResults())) {
    Type operand_type = std::get<0>(entry).getType();
    Value result = std::get<1>(entry);
    if (result.getType() == operand_type) continue;
    // Pass through nodes may remove ref types, don't consider that as
    // refinement.
    // TODO(jpienaar): There could be refinement in addition to this, so
    // refine this.
    if (operand_type.cast<TensorType>()
            .getElementType()
            .isa<TF::TensorFlowRefType>() &&
        !result.getType()
             .cast<TensorType>()
             .getElementType()
             .isa<TF::TensorFlowRefType>())
      continue;
    AddCastBackForUnsupportedNonTFUses(op, result, tf_dialect,
                                       result.getType());
    result.setType(operand_type);
    changed = true;
  }
  return changed;
}

// Infers shape for necessary ops that are not in the TF dialect.
bool InferShapeForNonTFDialectOperation(Operation* op, Dialect* tf_dialect) {
  if (auto graph_op = dyn_cast<tf_executor::GraphOp>(op)) {
    return InferShapeForPassThroughOps(graph_op.GetFetch().fetches(), op,
                                       tf_dialect);
  }
  if (auto island_op = dyn_cast<tf_executor::IslandOp>(op)) {
    return InferShapeForPassThroughOps(island_op.GetYield().fetches(), op,
                                       tf_dialect);
  }
  if (auto iter_sink = dyn_cast<tf_executor::NextIterationSinkOp>(op)) {
    auto iter_source = cast<tf_executor::NextIterationSourceOp>(
        iter_sink.token().getDefiningOp());
    return InferShapeForPassThroughOps(
        iter_sink.getOperands().drop_front().take_front(), iter_source,
        tf_dialect);
  }
  // TODO(b/155227679): Use OpInterface instead of hard-coding for TensorCastOp.
  if (auto tensor_cast = dyn_cast<TensorCastOp>(op)) {
    return InferShapeForPassThroughOps(
        tensor_cast.getOperation()->getOperands(), op, tf_dialect);
  }
  return false;
}

// Gets the subtype's shape and data type for `type`. Templated to support both
// ResourceType and VariantType.
template <typename T>
std::unique_ptr<std::vector<
    std::pair<tensorflow::PartialTensorShape, tensorflow::DataType>>>
GetSubtypesHelper(Type type) {
  auto type_with_subtypes =
      type.cast<TensorType>().getElementType().dyn_cast<T>();
  if (!type_with_subtypes || type_with_subtypes.getSubtypes().empty()) {
    return nullptr;
  }
  auto shapes_and_types = absl::make_unique<std::vector<
      std::pair<tensorflow::PartialTensorShape, tensorflow::DataType>>>();
  for (auto subtype : type_with_subtypes.getSubtypes()) {
    auto shape = GetShapeFromMlirType(subtype);
    // handle_shapes_and_types requires all shapes to be known. So if any
    // subtype is unknown, clear the vector.
    if (!shape) {
      shapes_and_types = nullptr;
      break;
    }
    tensorflow::DataType dtype;
    auto status =
        tensorflow::ConvertToDataType(subtype.getElementType(), &dtype);
    assert(status.ok() && "Unknown element type");
    shapes_and_types->emplace_back(*shape, dtype);
  }
  return shapes_and_types;
}

// Gets the subtype's shape and data type for `type`.
std::unique_ptr<std::vector<
    std::pair<tensorflow::PartialTensorShape, tensorflow::DataType>>>
GetSubtypes(Type type) {
  auto subclasses = GetSubtypesHelper<TF::ResourceType>(type);
  if (subclasses) return subclasses;
  return GetSubtypesHelper<TF::VariantType>(type);
}

// Makes result types match the operand types (the i-th result type will
// match the i-th operand type). Returns true if anything is changed.
bool PassThroughOperandTypes(OperandRange operands, ResultRange results) {
  bool changed = false;
  for (auto entry : zip(operands, results)) {
    Type operand_type = std::get<0>(entry).getType();
    Type result_type = std::get<1>(entry).getType();
    if (operand_type == result_type) continue;
    // Pass through nodes may remove ref types, don't consider that as
    // refinement.
    // TODO(jpienaar): There could be refinement in addition to this, so
    // refine this.
    if (operand_type.cast<TensorType>()
            .getElementType()
            .isa<TF::TensorFlowRefType>() &&
        !result_type.cast<TensorType>()
             .getElementType()
             .isa<TF::TensorFlowRefType>())
      continue;

    std::get<1>(entry).setType(operand_type);
    changed = true;
  }
  return changed;
}

// Returns whether type can be further refined.
bool CanBeRefined(Type type) {
  auto shape_type = type.dyn_cast<ShapedType>();
  return shape_type && (!shape_type.hasStaticShape() ||
                        shape_type.getElementType().isa<TF::ResourceType>() ||
                        shape_type.getElementType().isa<TF::VariantType>());
}

// Infers the shape from a (Stateful)PartionedCall operation by looking up the
// called function and propagating the return type.
bool InferShapeForCall(Operation* op) {
  auto call_op = cast<CallOpInterface>(op);
  CallInterfaceCallable callable = call_op.getCallableForCallee();
  SymbolRefAttr sym = callable.dyn_cast<SymbolRefAttr>();
  if (!sym) return false;
  FuncOp func = dyn_cast<FuncOp>(SymbolTable::lookupNearestSymbolFrom(op, sym));
  if (!func) return false;

  bool changed = false;
  // Map each of the results of the call to the returned type of the
  // function.
  for (auto result : zip(op->getResults(), func.getType().getResults())) {
    if (std::get<0>(result).getType() == std::get<1>(result)) continue;
    // Skip already statically shaped results.
    if (!CanBeRefined(std::get<0>(result).getType())) continue;

    auto shaped_type = std::get<0>(result).getType().cast<ShapedType>();
    auto new_type = std::get<1>(result).dyn_cast<RankedTensorType>();
    if (!new_type) continue;

    // Inserts a cast back to the original type if any user is not in the
    // TF dialect.
    AddCastBackForUnsupportedNonTFUses(op, std::get<0>(result),
                                       op->getDialect(), shaped_type);
    // Finally we inferred the shape and replace the type for this result.
    std::get<0>(result).setType(new_type);
    changed = true;
  }
  return changed;
}

bool RefineWithInferTypeOpInterface(InferTypeOpInterface infer_ti,
                                    Dialect* tf_dialect) {
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

    // Inserts a cast back to the original type if any user is not in the
    // TF dialect.
    AddCastBackForUnsupportedNonTFUses(op, std::get<0>(result),
                                       op->getDialect(), std::get<1>(result));
    // Finally we inferred the shape and replace the type for this result.
    std::get<0>(result).setType(std::get<1>(result));
    changed = true;
  }
  return changed;
}

}  // namespace

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
    if (pack_op.getType().cast<TensorType>().getRank() != 1) return failure();
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
  LLVM_DEBUG(value_port.print(llvm::errs() << "\nComputing output for "));
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

  // Note: this focusses only on the trivial pack op case and this could be
  // generalized.
  if (auto pack_op = dyn_cast<TF::PackOp>(op)) {
    if (pack_op.getType().cast<TensorType>().getRank() != 1) return nullptr;
    if (port.size() != 2 || port[0] != 0) return nullptr;
    ValuePort op_port(op->getOperand(port[1]));
    return values(op_port);
  }

  return nullptr;
}

// Context used during ShapeInference. This class contains common information
// that is required by the individual shape inference helper functions (e.g.,
// TF Graph version, constant values computed, etc.)
class ShapeInference {
 public:
  ShapeInference(int64_t graph_version, MLIRContext* context);

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
    results_[value_port] = value;
  }

  // Performs shape inference on the provided op and return true if the type of
  // at least one result has been changed.
  // A tf.Cast() is inserted for any uses that isn't in the TensorFlow dialect.
  // `graph_version` indicates the current GraphDef compatibility versions
  // (the versions field in graph.proto).
  bool InferShapeForSingleOperation(Operation* op);

  // Infers shape on the provided region, including nested ones, iterate until
  // fix point with a limit of max_iteration. Returns success if fix point is
  // reached before max_iteration.
  LogicalResult InferShapeUntilFixPoint(Region* region,
                                        int64_t max_iteration = 10);

  // Updates input types and refine shapes inside body of functions that are
  // attached to ControlFlow ops (If/While). These functions include Then/Else
  // branches of IfOp and Cond/Body functions of WhileOp. These functions share
  // following common properties:
  //   1) They are never reused, ie. having a single use in module.
  //   2) Their input types match those of their parent ops (excluding inputs
  //      like predicate).
  // Returns a boolean indicating whether any change has been applied.
  LogicalResult RefineShapeForControlFlowFunc(FuncOp func,
                                              ArrayRef<Type> input_types,
                                              int64_t max_iteration);

  // Propagate the shapes to the functions named.
  LogicalResult PropagateShapeToFunctions(
      ModuleOp module, Operation::operand_type_range input_types,
      ArrayRef<StringRef> func_names, int64_t max_iteration);

  // Shape propagation for call/control flow ops.
  LogicalResult PropagateShapeIntoAttachedFunctions(Operation* op,
                                                    int64_t max_iteration);

  // Propagates any constant operand of call_op to the called function body's
  // corresponding argument if the callee has only one use.
  //
  // TODO(b/154065712): Move this to a more general inter-procedural constant
  // folding pass.
  void PropagateConstantToCallee(CallOpInterface call_op,
                                 SymbolRefAttr callee_sym, ModuleOp module);

  // Propagates any constant return value of the callee function to the call
  // op's corresponding result.
  void PropagateConstantFromCallee(CallOpInterface call_op,
                                   SymbolRefAttr callee_sym, ModuleOp module);

  // Tries to compute the result of folding the op. This doesn't actually
  // perform constant folding, it is just computes the equivalent constants.
  // Returns whether it was able to compute constant values.
  LogicalResult TryToFold(Operation* op);

 private:
  // Mapping between ValuePort (which corresponds to an OpResult or smaller,
  // e.g., first element of OpResult produded) to an Attribute if the ValuePort
  // corresponds to a constant value.
  ValuePortResultMap results_;
  int64_t graph_version_;
  Dialect* tf_dialect_;
};

ShapeInference::ShapeInference(int64_t graph_version, MLIRContext* context)
    : graph_version_(graph_version) {
  tf_dialect_ = context->getRegisteredDialect<TensorFlowDialect>();
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
      LLVM_DEBUG(front.print(llvm::errs() << "\nWorklist front "));

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
            LLVM_DEBUG(llvm::errs() << "Unexpected number of elements\n");
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

bool ShapeInference::InferShapeForSingleOperation(Operation* op) {
  assert(tf_dialect_ == op->getDialect());
  // The shape function of these ops sometimes does not propagate subtypes
  // (handle shapes) for resource and variant types. We use a simple passthrough
  // to make sure they are preserved in the output.
  if (isa<TF::IdentityOp>(op) || isa<TF::IdentityNOp>(op) ||
      isa<TF::ZerosLikeOp>(op) || isa<TF::WhileOp>(op)) {
    return PassThroughOperandTypes(op->getOperands(), op->getResults());
  }

  // If no result for this op needs shape inference, we have a fast-path return.
  // But if the type is a resource/variant, we do not skip it because we might
  // not have the handle shapes.
  if (none_of(op->getResultTypes(), CanBeRefined)) {
    LLVM_DEBUG(llvm::dbgs() << "Skipping inference for statically shaped op '"
                            << op->getName() << "'.\n");
    return false;
  }

  // Handle call operations by looking up callee and infering return shape as
  // needed.
  if (isa<PartitionedCallOp>(op) || isa<StatefulPartitionedCallOp>(op))
    return InferShapeForCall(op);

  // tf.Cast are only inferred if they have at least one user in the tf dialect.
  // This is necessary to avoid reprocessing the tf.Cast that are inserted at
  // the end of this function.
  if (isa<CastOp>(op) &&
      all_of(op->getResult(0).getUsers(), [&](Operation* user) {
        return user->getDialect() != tf_dialect_;
      })) {
    LLVM_DEBUG(llvm::dbgs() << "Skipping inference for tf.Cast with no TF "
                               "dialect operation users '"
                            << *op << "'.\n");
    return false;
  }

  StringRef op_name = op->getName().getStringRef();
  // Drop the `tf.` prefix to query TF registry.
  auto node_name =
      op_name.drop_front(TensorFlowDialect::getDialectNamespace().size() + 1);

  // Get information from the registry and check if we have a shape function for
  // this op.
  const tensorflow::OpRegistrationData* op_reg_data =
      tensorflow::OpRegistry::Global()->LookUp(node_name.data());
  if (!op_reg_data) {
    LLVM_DEBUG(llvm::dbgs() << "Skipping inference for unregistered op '"
                            << op->getName() << "'.\n");
    return false;
  }
  if (op_reg_data->shape_inference_fn == nullptr) {
    LLVM_DEBUG(llvm::dbgs()
               << "Skipping inference for op without shape function '"
               << op->getName() << "'.\n");
    return false;
  }

  // Convert the operation to a NodeDef to be able to use the InferenceContext
  // and the TensorFlow shape function.
  auto node_def_or = tensorflow::ConvertTFDialectOpToNodeDef(
      op, node_name, /*ignore_unregistered_attrs=*/true);
  if (!node_def_or.ok()) {
    LLVM_DEBUG(llvm::dbgs()
               << "Error converting op '" << *op << "' to NodeDef: "
               << node_def_or.status().error_message() << "\n");
    return false;
  }
  std::unique_ptr<tensorflow::NodeDef> node_def =
      std::move(node_def_or).ValueOrDie();

  // Collect an array with input values for constant operands and input shapes
  // for all the operands.
  std::vector<const tensorflow::Tensor*> input_tensors(op->getNumOperands());
  std::vector<tensorflow::PartialTensorShape> input_shapes(
      op->getNumOperands());
  std::vector<tensorflow::Tensor> tensors(op->getNumOperands());
  std::vector<std::unique_ptr<std::vector<
      std::pair<tensorflow::PartialTensorShape, tensorflow::DataType>>>>
      handle_shapes_and_types(op->getNumOperands());
  for (auto it : llvm::enumerate(op->getOperands())) {
    Value operand = it.value();
    size_t index = it.index();

    // If the operand is constant, then convert it to Tensor.
    ValuePort vp(operand);
    Attribute attr = ComputeOutputComponent(vp);
    if (!attr && matchPattern(operand, m_Constant(&attr)))
      RecordValue(vp, attr);
    if (attr) {
      tensorflow::Tensor* input_tensor = &tensors[index];
      auto status =
          tensorflow::ConvertToTensor(attr.cast<ElementsAttr>(), input_tensor);
      if (status.ok()) {
        input_tensors[index] = input_tensor;
      } else {
        LLVM_DEBUG(llvm::dbgs()
                   << "Error converting input " << index << " of op '" << *op
                   << "' to Tensor: " << status.error_message() << "\n");
      }
    }

    Type operand_type = operand.getType();
    if (auto shape = GetShapeFromMlirType(operand_type)) {
      input_shapes[index] = *shape;
    }
    // Collect the handle shapes and types for a resource/variant.
    handle_shapes_and_types[index] = GetSubtypes(operand_type);
  }

  // Perform the shape inference using an InferenceContext with the input
  // shapes. This object is abstracting the information that the ShapeInference
  // function operates on.
  InferenceContext c(graph_version_, *node_def, op_reg_data->op_def,
                     input_shapes, input_tensors,
                     /*input_tensors_as_shapes=*/{}, handle_shapes_and_types);
  auto status = c.Run(op_reg_data->shape_inference_fn);
  if (!status.ok()) {
    LLVM_DEBUG(llvm::dbgs() << "Shape inference error for '" << *op
                            << "': " << status.error_message() << "\n");
    return false;
  }

  // Determine if, during shape computation, the shape functions attempted to
  // query an input operand as shape where the input was not known/constant.
  bool requires_inputs =
      any_of(llvm::seq<int>(0, c.num_inputs()), [&](int input) {
        return c.requested_input_tensor_as_partial_shape(input) &&
               !input_tensors[input];
      });
  if (requires_inputs) {
    std::vector<ShapeHandle> input_tensors_as_shapes;
    for (int input : llvm::seq<int>(0, c.num_inputs())) {
      if (c.requested_input_tensor_as_partial_shape(input) &&
          !input_tensors[input]) {
        auto op_result = op->getOperand(input).dyn_cast<OpResult>();
        if (!op_result) continue;
        // Resize on first valid shape computed.
        input_tensors_as_shapes.resize(c.num_inputs());
        auto handle = ComputeOutputAsShape(op_result, &c);
        LLVM_DEBUG(llvm::dbgs() << "Requested " << input << " as shape "
                                << (handle.Handle() ? "found" : "not found"));
        if (handle.Handle()) input_tensors_as_shapes[input] = handle;
      }
    }

    // Attempt to compute the unknown operands as shapes.
    // Note: in the case where no partial outputs could be computed, this would
    // be empty.
    if (!input_tensors_as_shapes.empty()) {
      c.set_input_tensors_as_shapes(input_tensors_as_shapes);
      auto status = c.Run(op_reg_data->shape_inference_fn);
      if (!status.ok()) {
        LLVM_DEBUG(llvm::dbgs() << "Shape inference error for '" << *op
                                << "': " << status.error_message() << "\n");
        return false;
      }
    }
  }

  assert(c.num_outputs() == op->getNumResults() &&
         "inference context matches the MLIR number of results.");

  // Update the shape for each of the operation result if the InferenceContext
  // has more precise shapes recorded.
  bool changed = false;
  for (int output : llvm::seq<int>(0, c.num_outputs())) {
    // Skip already statically shaped results.
    Value result = op->getResult(output);
    if (!CanBeRefined(result.getType())) continue;
    auto shaped_type = result.getType().cast<ShapedType>();

    ShapeHandle shape_handle = c.output(output);
    LLVM_DEBUG(llvm::dbgs() << "Inferred output " << output << " : "
                            << c.DebugString(shape_handle) << "\n");
    auto get_tensor_type = [&c](const ShapeHandle& sh,
                                Type element_type) -> TensorType {
      if (!c.RankKnown(sh)) return UnrankedTensorType::get(element_type);
      // Convert the shape from TensorFlow (int64) to MLIR (int64_t).
      SmallVector<int64_t, 8> shape;
      for (int dim : llvm::seq<int>(0, c.Rank(sh)))
        shape.push_back(c.Value(c.Dim(sh, dim)));
      return RankedTensorType::get(shape, element_type);
    };
    auto new_element_type = shaped_type.getElementType();
    // Populate the handle shapes for a resource/variant.
    if (new_element_type.isa<TF::ResourceType>() ||
        new_element_type.isa<TF::VariantType>()) {
      auto handle_shapes_types = c.output_handle_shapes_and_types(output);
      if (handle_shapes_types) {
        SmallVector<TensorType, 1> subtypes;
        OpBuilder b(op);
        for (const auto& shape_n_type : *handle_shapes_types) {
          Type element_type;
          auto status =
              tensorflow::ConvertDataType(shape_n_type.dtype, b, &element_type);
          assert(status.ok() && "Unknown element type");
          subtypes.push_back(get_tensor_type(shape_n_type.shape, element_type));
        }
        if (new_element_type.isa<TF::ResourceType>()) {
          new_element_type = TF::ResourceType::get(subtypes, op->getContext());
        } else {
          new_element_type = TF::VariantType::get(subtypes, op->getContext());
        }
      }
    }
    auto new_type = get_tensor_type(shape_handle, new_element_type);
    if (result.getType() == new_type) continue;
    // Inserts a cast back to the original type if any user is not in the TF
    // dialect.
    AddCastBackForUnsupportedNonTFUses(op, result, tf_dialect_,
                                       result.getType());
    // Finally we inferred the shape and replace the type for this result.
    result.setType(new_type);
    changed = true;
  }
  if (changed)
    LLVM_DEBUG(llvm::dbgs()
               << "Modified after shape inference: '" << *op << "'\n");
  return changed;
}

LogicalResult ShapeInference::RefineShapeForControlFlowFunc(
    FuncOp func, ArrayRef<Type> input_types, int64_t max_iteration) {
  ModuleOp module = func.getParentOfType<ModuleOp>();
  auto func_uses = SymbolTable::getSymbolUses(func, &module.getBodyRegion());
  int num_uses = std::distance(func_uses->begin(), func_uses->end());
  if (num_uses != 1) {
    func.emitWarning(formatv(
        "expected control flow function {0} to have exactly 1 use, found {1}.",
        func.getName(), num_uses));
    return failure();
  }

  FunctionType func_type = func.getType();
  func.setType(FunctionType::get(input_types, func_type.getResults(),
                                 func.getContext()));

  for (auto arg_and_idx : llvm::enumerate(func.getArguments())) {
    arg_and_idx.value().setType(input_types[arg_and_idx.index()]);
  }

  auto res = InferShapeUntilFixPoint(&func.getBody(), max_iteration);
  if (failed(res)) return res;

  auto new_return_types = InferShapeForFunctionReturnType(func);
  if (new_return_types.hasValue()) {
    func.setType(FunctionType::get(input_types, new_return_types.getValue(),
                                   func.getContext()));
  }

  return success();
}

LogicalResult ShapeInference::PropagateShapeToFunctions(
    ModuleOp module, Operation::operand_type_range input_types,
    ArrayRef<StringRef> func_names, int64_t max_iteration) {
  bool all_succeeded = true;
  auto types = llvm::to_vector<4>(input_types);
  for (auto func_name : func_names) {
    FuncOp func = module.lookupSymbol<FuncOp>(func_name);
    all_succeeded =
        succeeded(RefineShapeForControlFlowFunc(func, types, max_iteration)) &&
        all_succeeded;
  }
  return success(all_succeeded);
}

void ShapeInference::PropagateConstantToCallee(CallOpInterface call_op,
                                               SymbolRefAttr callee_sym,
                                               ModuleOp module) {
  auto func = module.lookupSymbol<FuncOp>(callee_sym.getRootReference());
  auto func_uses = SymbolTable::getSymbolUses(func, &module.getBodyRegion());
  int num_uses = std::distance(func_uses->begin(), func_uses->end());
  OpBuilder builder(&func.front().front());
  Operation* op = call_op.getOperation();
  if (num_uses == 1) {
    // If this is the only caller, and an operand is a constant, propagate
    // the constant value inside the function.
    for (auto arg : func.getArguments()) {
      auto operand = op->getOperand(arg.getArgNumber());
      if (auto known_constant = ComputeOutputComponent(ValuePort(operand)))
        RecordValue(ValuePort(arg), known_constant);
    }
  }
}

void ShapeInference::PropagateConstantFromCallee(CallOpInterface call_op,
                                                 SymbolRefAttr callee_sym,
                                                 ModuleOp module) {
  auto func = module.lookupSymbol<FuncOp>(callee_sym.getRootReference());
  // If the return value is a constant, use the constant as the value of
  // the call return.
  Operation* op = call_op.getOperation();
  OpBuilder builder(op);
  builder.setInsertionPointAfter(op);
  for (auto retval :
       llvm::enumerate(func.front().getTerminator()->getOperands())) {
    ValuePort vp(retval.value());
    if (auto known_constant = ComputeOutputComponent(vp)) {
      RecordValue(ValuePort(op->getResult(retval.index())), known_constant);
    }
  }
}

LogicalResult ShapeInference::PropagateShapeIntoAttachedFunctions(
    Operation* op, int64_t max_iteration) {
  ModuleOp module = op->getParentOfType<ModuleOp>();
  if (auto if_op = dyn_cast<TF::IfOp>(op)) {
    return PropagateShapeToFunctions(
        module, drop_begin(if_op.getOperandTypes(), 1),
        {if_op.then_branch(), if_op.else_branch()}, max_iteration);
  } else if (auto while_op = dyn_cast<TF::WhileOp>(op)) {
    return PropagateShapeToFunctions(module, while_op.getOperandTypes(),
                                     {while_op.cond(), while_op.body()},
                                     max_iteration);
  } else if (auto call_op = dyn_cast<CallOpInterface>(op)) {
    CallInterfaceCallable callable = call_op.getCallableForCallee();
    if (SymbolRefAttr sym = callable.dyn_cast<SymbolRefAttr>()) {
      PropagateConstantToCallee(call_op, sym, module);
      if (failed(PropagateShapeToFunctions(
              module, call_op.getArgOperands().getTypes(),
              {sym.getRootReference()}, max_iteration))) {
        return failure();
      }
      PropagateConstantFromCallee(call_op, sym, module);
      return success();
    }
  }

  // TODO(ycao): Implement support for Call op, including function reuse.

  return success();
}

LogicalResult ShapeInference::TryToFold(Operation* op) {
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
  if (abstract_op) {
    if (failed(abstract_op->foldHook(op, constant_operands, fold_results)))
      return failure();
  } else {
    Dialect* dialect = op->getDialect();
    if (!dialect) return failure();
    // Only attempt TF dialect fallback if there are no unknown operands.
    if (some_unknown && dialect == tf_dialect_) return failure();
    SmallVector<Attribute, 8> constants;
    if (failed(dialect->constantFoldHook(op, constant_operands, constants)))
      return failure();
    fold_results.assign(constants.begin(), constants.end());
  }

  for (auto result : zip(op->getResults(), fold_results)) {
    auto fold_result = std::get<1>(result);
    Attribute attr = nullptr;
    if ((attr = fold_result.dyn_cast<Attribute>())) {
      RecordValue(ValuePort(std::get<0>(result)), attr);
    } else {
      auto value = fold_result.get<Value>();
      if ((attr = ComputeOutputComponent(ValuePort(value))))
        RecordValue(ValuePort(std::get<0>(result)), attr);
    }

    if (ElementsAttr eattr = attr.dyn_cast_or_null<ElementsAttr>()) {
      if (std::get<0>(result).getType() == eattr.getType()) continue;

      // Inserts a cast back to the original type if any user is not in the
      // TF dialect.
      Type old_type = std::get<0>(result).getType();
      std::get<0>(result).setType(eattr.getType());
      AddCastBackForUnsupportedNonTFUses(op, std::get<0>(result), tf_dialect_,
                                         old_type);
    }
  }

  return success();
}

LogicalResult ShapeInference::InferShapeUntilFixPoint(Region* region,
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
    region->walk([&](Operation* op) {
      if (auto infer_ti = dyn_cast<InferTypeOpInterface>(op)) {
        changed |= RefineWithInferTypeOpInterface(infer_ti, tf_dialect_);
        return;
      }

      if (op->getDialect() != tf_dialect_) {
        changed |= InferShapeForNonTFDialectOperation(op, tf_dialect_);
        return;
      }

      // Before attempting inference, just try to compute the folded
      // value/shape.
      if (succeeded(TryToFold(op))) return;

      // Best-effort shape inference in attached functions. Do not return
      // failure even if it doesn't get to fixed point.
      if (failed(PropagateShapeIntoAttachedFunctions(op, max_iteration))) {
        op->emitWarning() << "unable to refine shape of attached function "
                             "arguments and bodies";
      }

      changed |= InferShapeForSingleOperation(op);
    });
  }

  if (changed) {
    return region->getParentOp()->emitWarning()
           << "Shape inference did not reach stable state after "
           << max_iteration << " iterations";
  }
  return success();
}

LogicalResult InferShapeForFunction(FuncOp func,
                                    ArrayRef<ArrayRef<int64_t>> arg_shapes,
                                    int64_t graph_version) {
  ShapeInference context(graph_version, func.getContext());
  if (arg_shapes.empty()) {
    if (failed(context.InferShapeUntilFixPoint(&func.getBody())))
      return failure();
    // TODO(b/156276510): Verify that it is always fine to refine a function's
    // return type, as long as we do not change the argument shapes.
    if (auto return_types = InferShapeForFunctionReturnType(func)) {
      func.setType(FunctionType::get(func.getType().getInputs(),
                                     return_types.getValue(),
                                     func.getContext()));
    }

    return success();
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
      if (!input_ty || input_ty.getShape().size() != shape.size()) {
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

  if (!needs_refinement) {
    return success();
  }

  LogicalResult result = context.InferShapeUntilFixPoint(&func.getBody());
  if (failed(result)) {
    return failure();
  }

  auto return_types = InferShapeForFunctionReturnType(func);
  func.setType(FunctionType::get(new_arg_types,
                                 return_types.hasValue()
                                     ? return_types.getValue()
                                     : func.getType().getResults(),
                                 func.getContext()));

  return success();
}

}  // namespace TF
}  // namespace mlir
