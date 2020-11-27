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
#include <queue>

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

// Returns whether type can be further refined.
bool CanBeRefined(Type type) {
  auto shape_type = type.dyn_cast<ShapedType>();
  return shape_type &&
         (!shape_type.hasStaticShape() ||
          shape_type.getElementType().isa<TF::TensorFlowTypeWithSubtype>());
}

// Returns whether `original_type` type can be refined with
// `potential_refined_type` type.
bool CanRefineTypeWith(Type original_type, Type potential_refined_type) {
  if (original_type == potential_refined_type || !CanBeRefined(original_type))
    return false;

  auto shape_type = potential_refined_type.dyn_cast<ShapedType>();
  if (!shape_type) return false;
  if (shape_type.hasRank()) return true;

  auto element_type_with_subtype =
      shape_type.getElementType().dyn_cast<TF::TensorFlowTypeWithSubtype>();
  return element_type_with_subtype &&
         !element_type_with_subtype.GetSubtypes().empty();
}

// Returns if the shape inference pass supports an op outside the TF dialect.
bool IsSupportedNonTFOp(Operation* op) {
  return isa<tf_device::ReturnOp, tf_device::ClusterOp, tf_device::LaunchOp,
             tf_executor::EnterOp, tf_executor::ExitOp, tf_executor::FetchOp,
             tf_executor::GraphOp, tf_executor::IslandOp,
             tf_executor::LoopCondOp, tf_executor::MergeOp,
             tf_executor::NextIterationSinkOp, tf_executor::SwitchNOp,
             tf_executor::SwitchOp, tf_executor::YieldOp>(op);
}

// Returns whether a cast back would need to be inserted, e.g., whether the
// operation of which use is an operand allows for shape refinement without
// a cast.
bool NeedsCastBack(OpOperand& use, Dialect* tf_dialect) {
  return use.getOwner()->getDialect() != tf_dialect &&
         !IsSupportedNonTFOp(use.getOwner());
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
  ShapeInference(int64_t graph_version, MLIRContext* context,
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

  // Performs shape inference on the provided op and return true if the type of
  // at least one result has been changed.
  // A tf.Cast() is inserted for any uses that isn't in the TensorFlow dialect.
  // `graph_version` indicates the current GraphDef compatibility versions
  // (the versions field in graph.proto).
  bool InferShapeForSingleOperation(Operation* op);

  // Infers shape on the provided region, including nested ones, iterate until
  // fix point with a limit of max_iteration. Returns success if fix point is
  // reached before max_iteration.
  LogicalResult InferShapeUntilFixPoint(Region* region, int64_t max_iterations);

  // Updates input types and refine shapes inside body of functions that are
  // attached to ControlFlow ops (If/While). These functions include Then/Else
  // branches of IfOp and Cond/Body functions of WhileOp. These functions share
  // following common properties:
  //   1) They are never reused, ie. having a single use in module.
  //   2) Their input types match those of their parent ops (excluding inputs
  //      like predicate).
  LogicalResult PropagateShapeToFunctions(
      ModuleOp module, Operation::operand_type_range input_types,
      ArrayRef<FuncOp> functions, int64_t max_iteration);

  // Propagates shapes to regions given the shapes of the inputs of the regions.
  // All regions provided in `regions` are assumed to have inputs of type
  // `input_types`.
  LogicalResult PropagateShapeToRegions(
      Operation::operand_type_range input_types, ArrayRef<Region*> regions,
      int64_t max_iteration);

  // Shape propagation for call/control flow ops.
  LogicalResult PropagateShapeIntoAttachedFunctions(Operation* op,
                                                    int64_t max_iteration);

  // Shape propagation for region based control flow.
  LogicalResult PropagateShapeIntoAttachedRegions(Operation* op,
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
  void InferShapeForFunctionReturnType(FuncOp func);

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
  // Updates the result of an operation to a new inferred type. Also inserts
  // tf.Cast operation for uses that are incompatible with the new type.
  void UpdateTypeAndInsertIncompatibleUseCasts(Type new_type, Value result);

  // Refines the type of `result` of `op` using the type
  // `potential_refined_type`. Return true if the type was changed.
  bool RefineResultType(Operation* op, Value result,
                        Type potential_refined_type);

  // Infers the shape from a (Stateful)PartionedCall operation by looking up the
  // called function and propagating the return type.
  bool InferShapeForCall(CallOpInterface call_op);

  bool InferShapeForCast(CastOp op);

  // Infers the shape IfOp outputs based on the shapes of the then and else
  // function result types.
  bool InferShapeForIf(IfOp op);

  // Infers the shape IfRegion outputs based on the shapes of the then and else
  // yields.
  bool InferShapeForIfRegion(IfRegionOp op);

  bool RefineWithInferTypeOpInterface(InferTypeOpInterface infer_ti);

  // Returns all the callers of a function.
  // Note: Usage of the return value of this function may not be interleaved
  // with insertions to the callers map. This could occur if GetCallers is
  // called with two separate functions, the 2nd one incurs a resize and then
  // both first and 2nd stored callers are used.
  ArrayRef<FuncOp> GetCallers(FuncOp fn);

  // Mapping between ValuePort (which corresponds to an OpResult or smaller,
  // e.g., first element of OpResult produced) to an Attribute if the ValuePort
  // corresponds to a constant value.
  ValuePortResultMap results_;

  // Map from a function to the callers of that function.
  llvm::DenseMap<FuncOp, SmallVector<FuncOp, 4>> callers_of_func_;

  // Queue of functions being processed.
  llvm::DenseSet<FuncOp> queue_set_;
  std::queue<FuncOp> queue_;

  int64_t graph_version_;

  // TODO(b/154065712): Remove propagate_caller_callee_constants once using
  // SCCP pass instead.
  bool propagate_caller_callee_constants_;
};

ShapeInference::ShapeInference(int64_t graph_version, MLIRContext* context,
                               bool propagate_caller_callee_constants)
    : tf_dialect_(context->getLoadedDialect<TensorFlowDialect>()),
      graph_version_(graph_version),
      propagate_caller_callee_constants_(propagate_caller_callee_constants) {}

ArrayRef<FuncOp> ShapeInference::GetCallers(FuncOp fn) {
  auto pair = callers_of_func_.try_emplace(fn);
  if (pair.second) {
    ModuleOp module = fn.getParentOfType<ModuleOp>();
    auto uses = mlir::SymbolTable::getSymbolUses(fn.getOperation(), module);
    if (uses) {
      pair.first->second.reserve(pair.first->second.size());
      for (auto use : *uses) {
        pair.first->second.push_back(use.getUser()->getParentOfType<FuncOp>());
      }
    }
  }
  return pair.first->second;
}

void ShapeInference::EnqueueCallers(FuncOp fn) {
  for (auto user : GetCallers(fn)) enqueue(user);
}

void ShapeInference::UpdateTypeAndInsertIncompatibleUseCasts(Type new_type,
                                                             Value result) {
  // A tf.Cast operation is lazily created on the first use requires a cast.
  TF::CastOp cast_op;
  auto get_cast_op = [&]() {
    if (!cast_op) {
      Operation* op = result.getDefiningOp();
      OpBuilder b(op);
      b.setInsertionPointAfter(op);
      cast_op = b.create<TF::CastOp>(op->getLoc(), result.getType(), result,
                                     /*truncate=*/b.getBoolAttr(false));
    }
    return Value(cast_op);
  };
  // First insert cast back for uses that need a cast and then
  // update the type.
  bool enqueue_callers = false;
  for (OpOperand& use : make_early_inc_range(result.getUses())) {
    if (isa<ReturnOp>(use.getOwner()))
      enqueue_callers = true;
    else if (NeedsCastBack(use, tf_dialect_))
      use.set(get_cast_op());
  }

  result.setType(new_type);
  if (enqueue_callers)
    EnqueueCallers(result.getDefiningOp()->getParentOfType<FuncOp>());
}

bool ShapeInference::RefineResultType(Operation* op, Value result,
                                      Type potential_refined_type) {
  if (!CanRefineTypeWith(result.getType(), potential_refined_type))
    return false;

  UpdateTypeAndInsertIncompatibleUseCasts(potential_refined_type, result);
  return true;
}

// Infers the shape from a (Stateful)PartionedCall operation by looking up the
// called function and propagating the return type.
bool ShapeInference::InferShapeForCall(CallOpInterface call_op) {
  FuncOp func = dyn_cast<FuncOp>(call_op.resolveCallable());
  if (!func) return false;

  LLVM_DEBUG(llvm::dbgs() << "Infer shape for call " << func.getName());
  Operation* op = call_op.getOperation();
  bool changed = false;
  // Map each of the results of the call to the returned type of the
  // function.
  for (auto result : zip(op->getResults(), func.getType().getResults())) {
    changed = RefineResultType(op, std::get<0>(result), std::get<1>(result)) ||
              changed;
  }
  LLVM_DEBUG(llvm::dbgs() << " changed ? " << changed << "\n");

  return changed;
}

bool ShapeInference::InferShapeForCast(CastOp op) {
  DCOMMENT_OP(op.getOperation(), "Infering shape for ");
  Value result = op.getResult();
  if (!CanBeRefined(result.getType())) return false;

  Type operand_type = op.getOperand().getType();
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

  UpdateTypeAndInsertIncompatibleUseCasts(new_type, op.getResult());
  return true;
}

bool ShapeInference::InferShapeForIf(IfOp op) {
  DCOMMENT_OP(op.getOperation(), "InferShapeForIf");
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

    UpdateTypeAndInsertIncompatibleUseCasts(std::get<1>(result),
                                            std::get<0>(result));
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
  for (auto entry : zip(operands, results)) {
    Type operand_type = std::get<0>(entry).getType();
    Value result = std::get<1>(entry);
    TensorType result_type = result.getType().cast<TensorType>();
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

    UpdateTypeAndInsertIncompatibleUseCasts(operand_type, result);
    changed = true;
  }
  return changed;
}

bool ShapeInference::RefineShapeForPassThroughOps(Operation* op) {
  DCOMMENT_OP(op, "Pass through op");
  auto is_allowed_dtype = [](Type t) {
    // Skip if element type is not in standard or TF dialect.
    // TODO(jpienaar): The tf.Cast op, which is uniformly inserted at the
    // moment, cannot handle arbirary types (e.g., it can't handle quantized
    // types). This restriction can be relaxed if not only tf.Cast is used.
    return t.getDialect().getNamespace().empty() ||
           isa<TensorFlowDialect>(t.getDialect());
  };

  bool changed = false;
  for (auto entry : zip(op->getOperands(), op->getResults())) {
    TensorType operand_type = std::get<0>(entry).getType().cast<TensorType>();
    Value result = std::get<1>(entry);
    TensorType result_type = result.getType().cast<TensorType>();
    if (operand_type == result_type) continue;
    if (!operand_type.hasRank()) continue;
    if (result_type.hasRank() &&
        result_type.getShape() == operand_type.getShape())
      continue;
    if (!is_allowed_dtype(operand_type.getElementType()) ||
        !is_allowed_dtype(result_type.getElementType()))
      continue;

    auto new_type = RankedTensorType::get(operand_type.getShape(),
                                          result_type.getElementType());
    UpdateTypeAndInsertIncompatibleUseCasts(new_type, result);
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
  if (op->hasTrait<OpTrait::SameOperandsAndResultShape>()) {
    return RefineShapeForPassThroughOps(op);
  }
  if (auto call = dyn_cast<CallOpInterface>(op)) return InferShapeForCall(call);
  return false;
}

bool ShapeInference::InferShapeForSingleOperation(Operation* op) {
  LLVM_DEBUG(op->print(llvm::dbgs() << "InferShapeForSingleOperation for ");
             llvm::dbgs() << "\n");
  assert(tf_dialect_ == op->getDialect());
  // The shape function of these ops sometimes does not propagate subtypes
  // (handle shapes) for resource and variant types. We use a simple passthrough
  // to make sure they are preserved in the output.
  if (isa<TF::IdentityOp, TF::IdentityNOp, TF::ZerosLikeOp, TF::WhileOp,
          TF::WhileRegionOp>(op)) {
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

  // Handle call operations by looking up callee and infering return shape as
  // needed.
  if (auto call = dyn_cast<CallOpInterface>(op)) return InferShapeForCall(call);

  // tf.Cast are only inferred if they have at least one user in the TF dialect
  // or feeding into the function return. This is necessary to avoid inserting
  // casts which cannot be refined.
  if (auto cast_op = dyn_cast<CastOp>(op)) return InferShapeForCast(cast_op);

  // Handle IfOp here by inferring the shape from the else/then function
  // results. Since `output_shapes` is a derived attribute, avoid going down the
  // TF InferenceContext path as IfOp shape inference is implemented as just
  // a lookup of the output_shapes attribute.
  if (auto if_op = dyn_cast<IfOp>(op)) return InferShapeForIf(if_op);

  // Handle IfRegion operations by infering return shape from the then and else
  // branches.
  if (auto if_region = dyn_cast<IfRegionOp>(op))
    return InferShapeForIfRegion(if_region);

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

    if (op_result.getType() == inferred_type) continue;
    UpdateTypeAndInsertIncompatibleUseCasts(inferred_type, op_result);
    changed = true;
  }

  if (changed) DCOMMENT_OP(op, "Modified after shape inference:");
  return changed;
}

LogicalResult ShapeInference::PropagateShapeToFunctions(
    ModuleOp module, Operation::operand_type_range input_types,
    ArrayRef<FuncOp> functions, int64_t max_iteration) {
  bool all_succeeded = true;
  // If shape propagation fails for one function, return failure, but do not
  // early exit and attempt to propagate shapes for all provided functions to
  // have a best-effort propagation.
  for (FuncOp func : functions) {
    DCOMMENT("Propating shape to" << func.getName());
    auto func_uses = GetCallers(func);
    if (!llvm::hasSingleElement(func_uses)) {
      func.emitWarning(
          formatv("expected control flow function @{0} to have exactly 1 use, "
                  "found {1}.",
                  func.getName(), func_uses.size()));
      all_succeeded = false;
      continue;
    }

    FunctionType func_type = func.getType();
    func.setType(FunctionType::get(input_types, func_type.getResults(),
                                   func.getContext()));

    auto res =
        PropagateShapeToRegions(input_types, {&func.getBody()}, max_iteration);
    if (failed(res)) {
      all_succeeded = false;
      continue;
    }

    InferShapeForFunctionReturnType(func);
  }
  return success(all_succeeded);
}

LogicalResult ShapeInference::PropagateShapeToRegions(
    Operation::operand_type_range input_types, ArrayRef<Region*> regions,
    int64_t max_iteration) {
  DCOMMENT("\tPropagating shapes to regions");
  bool all_succeeded = true;
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
    all_succeeded = succeeded(InferShapeUntilFixPoint(region, max_iteration)) &&
                    all_succeeded;
  }
  return success(all_succeeded);
}

void ShapeInference::PropagateConstantToCallee(CallOpInterface call_op,
                                               FuncOp func, ModuleOp module) {
  auto func_uses = GetCallers(func);
  if (!llvm::hasSingleElement(func_uses)) return;

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

LogicalResult ShapeInference::PropagateShapeIntoAttachedFunctions(
    Operation* op, int64_t max_iteration) {
  ModuleOp module = op->getParentOfType<ModuleOp>();
  if (auto if_op = dyn_cast<TF::IfOp>(op)) {
    DCOMMENT("Propagating shapes into If");
    return PropagateShapeToFunctions(
        module, drop_begin(if_op.getOperandTypes(), 1),
        {if_op.then_function(), if_op.else_function()}, max_iteration);
  } else if (auto case_op = dyn_cast<TF::CaseOp>(op)) {
    SmallVector<FuncOp, 4> branches;
    case_op.get_branch_functions(branches);
    return PropagateShapeToFunctions(module,
                                     drop_begin(case_op.getOperandTypes(), 1),
                                     branches, max_iteration);
  } else if (auto while_op = dyn_cast<TF::WhileOp>(op)) {
    return PropagateShapeToFunctions(
        module, while_op.getOperandTypes(),
        {while_op.cond_function(), while_op.body_function()}, max_iteration);
  } else if (auto call_op = dyn_cast<CallOpInterface>(op)) {
    if (auto func = dyn_cast<FuncOp>(call_op.resolveCallable())) {
      PropagateConstantToCallee(call_op, func, module);
      if (failed(PropagateShapeToFunctions(module,
                                           call_op.getArgOperands().getTypes(),
                                           {func}, max_iteration))) {
        return failure();
      }
      PropagateConstantFromCallee(call_op, func, module);
      return success();
    }
  }

  // TODO(ycao): Implement support for Call op, including function reuse.

  return success();
}

LogicalResult ShapeInference::PropagateShapeIntoAttachedRegions(
    Operation* op, int64_t max_iteration) {
  if (auto while_op = dyn_cast<TF::WhileRegionOp>(op)) {
    return PropagateShapeToRegions(while_op.getOperandTypes(),
                                   {&while_op.cond(), &while_op.body()},
                                   max_iteration);
  }
  return success();
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
      if ((attr = ComputeOutputComponent(ValuePort(value))))
        RecordValue(ValuePort(std::get<0>(result)), attr);
    }

    if (ElementsAttr eattr = attr.dyn_cast_or_null<ElementsAttr>()) {
      if (std::get<0>(result).getType() == eattr.getType()) continue;

      UpdateTypeAndInsertIncompatibleUseCasts(eattr.getType(),
                                              std::get<0>(result));
    }
  }

  return success();
}

void ShapeInference::InferShapeForFunctionReturnType(FuncOp func) {
  LLVM_DEBUG(llvm::dbgs() << "Inferring return type for: " << func.getName()
                          << "\n");

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
  if (return_ops.size() != 1) return;

  // Find the return type.
  auto return_op = return_ops.front();

  // Manually fold tf.Cast that precedes the return instruction and only differs
  // in shape refinement level.
  bool changed = false;
  for (OpOperand& arg_op : return_op.getOperation()->getOpOperands()) {
    Operation* arg_defining_op = arg_op.get().getDefiningOp();
    if (auto cast_op = dyn_cast_or_null<CastOp>(arg_defining_op)) {
      Value input = cast_op.x();
      Value result = cast_op.y();
      if (!CanRefineTypeWith(result.getType(), input.getType())) continue;

      LLVM_DEBUG({
        llvm::errs() << "\tfolding & updating return type ";
        cast_op.getResult().getType().print(llvm::errs());
        cast_op.getOperand().getType().print(llvm::errs() << " to ");
        llvm::errs() << "\n";
      });

      // Shape inference should not change the element type.
      if (HasCompatibleElementTypes(input.getType(), result.getType())) {
        arg_op.set(input);
      } else {
        OpBuilder b(return_op.getOperation());
        auto type = RankedTensorType::get(
            input.getType().cast<TensorType>().getShape(),
            result.getType().cast<TensorType>().getElementType());
        auto new_cast_op =
            b.create<TF::CastOp>(return_op.getLoc(), type, input,
                                 /*truncate=*/b.getBoolAttr(false));
        arg_op.set(new_cast_op);
      }
      if (cast_op.y().use_empty()) cast_op.erase();
      changed = true;
    }
  }

  DCOMMENT("Updating function type");
  func.setType(FunctionType::get(
      func.getArgumentTypes(), return_op.getOperandTypes(), func.getContext()));

  if (changed) EnqueueCallers(func);
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
      DCOMMENT_OP(op, "Inferring for");
      if (auto infer_ti = dyn_cast<InferTypeOpInterface>(op)) {
        DCOMMENT("\tRefinining with type op interface");
        changed |= RefineWithInferTypeOpInterface(infer_ti);
        return;
      }

      if (op->getDialect() != tf_dialect_) {
        DCOMMENT("\tInfer non-TF dialect");
        changed |= InferShapeForNonTFDialectOperation(op);
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

      if (failed(PropagateShapeIntoAttachedRegions(op, max_iteration))) {
        op->emitWarning() << "unable to refine shape of attached region "
                             "arguments and bodies";
      }

      changed |= InferShapeForSingleOperation(op);
    });
  }

  if (changed) {
    return region->getParentOp()->emitWarning()
           << "shape inference did not reach stable state after "
           << max_iteration << " iterations";
  }
  return success();
}

static LogicalResult InferShapeForFunction(ShapeInference& context, FuncOp func,
                                           int64_t max_iterations) {
  if (failed(context.InferShapeUntilFixPoint(&func.getBody(), max_iterations)))
    return failure();
  // TODO(b/156276510): Verify that it is always fine to refine a function's
  // return type, as long as we do not change the argument shapes.
  context.InferShapeForFunctionReturnType(func);

  return success();
}

LogicalResult InferShapeForFunction(FuncOp func,
                                    ArrayRef<ArrayRef<int64_t>> arg_shapes,
                                    int64_t graph_version,
                                    int64_t max_iterations) {
  ShapeInference context(graph_version, func.getContext(),
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

  if (!needs_refinement) return success();

  if (failed(context.InferShapeUntilFixPoint(&func.getBody(), max_iterations)))
    return failure();

  context.InferShapeForFunctionReturnType(func);
  func.setType(FunctionType::get(new_arg_types, func.getType().getResults(),
                                 func.getContext()));

  return success();
}

LogicalResult InferModuleShape(ModuleOp module, int64_t max_iterations) {
  auto producer_or = tensorflow::GetTfGraphProducerVersion(module);
  if (!producer_or.ok()) {
    // TODO(jpienaar): Keeping the existing behavior for now but this could
    // be relaxed.
    LLVM_DEBUG(llvm::dbgs()
               << "Skipping inference; " << producer_or.status().ToString());
    return success();
  }
  int64_t producer = producer_or.ValueOrDie();
  ShapeInference context(producer, module.getContext(),
                         /*propagate_caller_callee_constants=*/true);
  if (auto main = module.lookupSymbol<mlir::FuncOp>("main"))
    context.enqueue(main);
  for (auto func : module.getOps<FuncOp>()) context.enqueue(func);
  // Arbitrarily upper bound the maximum number of functions that get processed
  // just to avoid pathological cases.
  auto max_iteration = context.QueueSize() * 4;
  while (!context.EmptyQueue()) {
    FuncOp func = context.front();
    auto res = InferShapeForFunction(context, func, max_iterations);
    if (failed(res)) return res;
    context.pop_front();

    if ((--max_iteration) == 0) {
      return emitWarning(UnknownLoc::get(module.getContext()))
             << "shape inference did not reach stable state after "
             << max_iteration << " iterations";
    }
  }
  return success();
}

}  // namespace TF
}  // namespace mlir
