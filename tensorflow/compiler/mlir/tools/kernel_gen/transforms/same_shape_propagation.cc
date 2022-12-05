/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

// This file contains the analysis and transformation to rewrite kernel
// functions such that they use a single set of arguments for the strides and
// sizes of operands with equal shapes.

#include <memory>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/ADT/EquivalenceClasses.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/GPU/IR/GPUDialect.h"  // from @llvm-project
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"  // from @llvm-project
#include "mlir/IR/AsmState.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tools/kernel_gen/ir/tf_framework_ops.h"
#include "tensorflow/compiler/mlir/tools/kernel_gen/transforms/passes.h"
#include "tensorflow/compiler/xla/mlir_hlo/lhlo/IR/lhlo_ops.h"

#define DEBUG_TYPE "kernel-gen-shapes"

namespace {

using mlir::ArrayRef;
using mlir::SmallVector;
using mlir::Value;

/// Represents a value or constant. Used to unify operands for operations that
/// take both ssa values and attributes.
struct ValueOrConst {
  explicit ValueOrConst(Value v) : value_or_constant(v), is_constant(false) {}
  explicit ValueOrConst(int64_t c) : value_or_constant(c), is_constant(true) {}

  Value value() const {
    assert(!is_constant);
    return value_or_constant.value;
  }

  int64_t constant() const {
    assert(is_constant);
    return value_or_constant.constant;
  }

  bool isConstant() const { return is_constant; }

 private:
  union ValueOrConstStorage {
    explicit ValueOrConstStorage(Value v) : value(v) {}
    explicit ValueOrConstStorage(size_t c) : constant(c) {}

    Value value;
    int64_t constant;
  } value_or_constant;

  bool is_constant;
};

llvm::hash_code hash_value(ValueOrConst value) {
  return value.isConstant() ? static_cast<llvm::hash_code>(value.constant())
                            : mlir::hash_value(value.value());
}

bool operator==(ValueOrConst lhs, ValueOrConst rhs) {
  if (lhs.isConstant()) {
    return rhs.isConstant() && lhs.constant() == rhs.constant();
  } else {
    return !rhs.isConstant() && lhs.value() == rhs.value();
  }
}

inline llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                     const ValueOrConst &value) {
  if (value.isConstant()) {
    os << value.constant();
  } else {
    Value val = value.value();
    mlir::AsmState asm_state(
        val.getParentRegion()->getParentOfType<mlir::func::FuncOp>());
    val.printAsOperand(os, asm_state);
  }
  return os;
}

/// Represents a shape, as either a single SSA value that represents the entire
/// shape vector or as a vector of SSA values representing scalars.
struct ShapeValue {
  explicit ShapeValue(Value vector)
      : shape({ValueOrConst{vector}}), is_vector(true) {}
  explicit ShapeValue(ValueOrConst vector) : shape({vector}), is_vector(true) {
    assert(!vector.isConstant());
  }
  template <typename T>
  explicit ShapeValue(T values)
      : shape(values.begin(), values.end()), is_vector(false) {}

  ValueOrConst vector() const {
    assert(is_vector);
    return shape.front();
  }

  ArrayRef<ValueOrConst> scalars() const {
    assert(!is_vector);
    return llvm::makeArrayRef(shape);
  }

  bool isVector() const { return is_vector; }

 private:
  SmallVector<ValueOrConst, 4> shape;
  bool is_vector;
};

llvm::hash_code hash_value(ShapeValue shape) {
  return shape.isVector() ? hash_value(shape.vector())
                          : hash_value(shape.scalars());
}

bool operator==(ShapeValue lhs, ShapeValue rhs) {
  if (lhs.isVector()) {
    return rhs.isVector() && lhs.vector() == rhs.vector();
  } else {
    return !rhs.isVector() && lhs.scalars() == rhs.scalars();
  }
}

inline llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                     const ShapeValue &shape) {
  if (shape.isVector()) {
    os << shape.vector();
    return os;
  }
  os << "[";
  bool first = true;
  for (auto scalar : shape.scalars()) {
    if (!first) {
      os << ", ";
    }
    first = false;
    os << scalar;
  }
  os << "]";
  return os;
}

}  // namespace

namespace llvm {

template <>
struct DenseMapInfo<ShapeValue> {
  static ShapeValue getEmptyKey() {
    return ShapeValue(DenseMapInfo<mlir::Value>::getEmptyKey());
  }
  static ShapeValue getTombstoneKey() {
    return ShapeValue(DenseMapInfo<mlir::Value>::getTombstoneKey());
  }
  static unsigned getHashValue(ShapeValue shape) { return hash_value(shape); }
  static bool isEqual(ShapeValue LHS, ShapeValue RHS) { return LHS == RHS; }
};

}  // namespace llvm

namespace mlir {
namespace kernel_gen {
namespace transforms {

namespace {

#define GEN_PASS_DEF_PROPAGATESHAPEKNOWLEDGETOKERNELS
#include "tensorflow/compiler/mlir/tools/kernel_gen/transforms/kernel_gen_passes.h.inc"

// A basic shape equality inference. This should be superceeded by a proper
// inference once available. Until then, we just build this out to the needs of
// the kernel generator project.
class ShapeEqualityKnowledge {
 public:
  /// Checks all operations for potential shape equality of their respective
  /// results.
  void build(func::FuncOp function) {
    function.walk([&](Operation *op) {
      if (auto reshape = dyn_cast<memref::ReshapeOp>(op)) {
        registerAssociation(ShapeValue{(Value)reshape.getShape()},
                            reshape.getResult());
        return;
      }
      if (auto cast = dyn_cast<memref::ReinterpretCastOp>(op)) {
        // Only support fully dynamic sizes for now.
        // TODO(herhut): Fix once the op has canonicalizers that break this.
        for (unsigned int p = 0, e = cast.getResultRank(); p < e; ++p) {
          if (!cast.isDynamicSize(p)) {
            return;
          }
        }
        registerAssociation(ShapeValue{cast.getSizes()}, cast.getResult());
        return;
      }
      if (auto alloc = dyn_cast<memref::AllocOp>(op)) {
        SmallVector<ValueOrConst, 4> shape;
        ShapedType type = alloc.getResult().getType().cast<ShapedType>();
        fillShapeFromAllocLike(alloc.getDynamicSizes(), type, shape);
        registerAssociation(ShapeValue{shape}, alloc.getResult());
        return;
      }
      if (auto alloc = dyn_cast<tf_framework::TFAllocOp>(op)) {
        // Construct a symbol representing the allocated shape.
        SmallVector<ValueOrConst, 4> shape;
        ShapedType type = alloc.getResult().getType().cast<ShapedType>();
        fillShapeFromAllocLike(alloc.getDynSizes(), type, shape);
        registerAssociation(ShapeValue{shape}, alloc.getResult());
        return;
      }
    });
  }

  /// Checks whether `one` and `other` are known to have the same shape and
  /// strides.
  bool haveSameShape(Value one, Value other) {
    return equal_shapes_.isEquivalent(one.getAsOpaquePointer(),
                                      other.getAsOpaquePointer());
  }

 private:
  static void fillShapeFromAllocLike(mlir::OperandRange operands,
                                     ShapedType type,
                                     SmallVectorImpl<ValueOrConst> &shape) {
    assert(type.hasRank());
    auto dynamic_sizes = operands.begin();
    for (auto extent : type.getShape()) {
      shape.push_back(ShapedType::isDynamic(extent)
                          ? ValueOrConst{*(dynamic_sizes++)}
                          : ValueOrConst{extent});
    }
  }

  /// Registers the value `value` to have the shape represented by `shape`. If
  /// `shape` has been registered before, place `value` into the same
  /// equivalence class. Otherwise register `value` as an equivalence class of
  /// its own.
  void registerAssociation(ShapeValue shape, Value value) {
    LLVM_DEBUG({ llvm::dbgs() << "Processing " << value << "\n"; });
    auto insert_symbolic = symbolic_shapes_.insert({shape, value});
    if (insert_symbolic.second) {
      LLVM_DEBUG({ llvm::dbgs() << "New symbolic shape " << shape << "\n"; });
      equal_shapes_.insert(value.getAsOpaquePointer());
      // We have seen this symbolic shape for the first time. Try to match it
      // with a vector or shape we already know and alias classes if possible.
      // This could be based on shape dialect if we weren't late in the
      // lowering.
      tryEvaluateShapeToRoot(shape, value);
    } else {
      auto rep = insert_symbolic.first->second;
      LLVM_DEBUG({ llvm::dbgs() << "Aliasing with rep " << rep << "\n"; });
      equal_shapes_.unionSets(rep.getAsOpaquePointer(),
                              value.getAsOpaquePointer());
    }
  }

  /// Follows the definition chains of the ShapeValue `shape` to identify cases
  /// where `shape` is derived from some other value's shape. In such case, the
  /// equivalence classes of that other value and `value` are unioned.
  /// This is based on pattern matching and not complete.
  void tryEvaluateShapeToRoot(ShapeValue shape, Value value) {
    // Just some pattern matching for common cases here.
    if (!shape.isVector()) {
      // Patterns that revolve around scalars.
      // Check whether the scalars are all dim operations for some other memref.
      Value candidate;
      bool all_are_dimops =
          llvm::all_of(llvm::enumerate(shape.scalars()), [&candidate](auto p) {
            ValueOrConst val = p.value();
            if (val.isConstant()) return false;
            auto dimOp = val.value().getDefiningOp<memref::DimOp>();
            if (!dimOp) return false;
            if (!candidate) candidate = dimOp.getSource();
            auto index = dimOp.getConstantIndex();
            if (!index.has_value()) return false;
            return candidate == dimOp.getSource() && p.index() == index.value();
          });
      if (all_are_dimops && candidate) {
        equal_shapes_.unionSets(candidate.getAsOpaquePointer(),
                                value.getAsOpaquePointer());
      }
    }
  }

  // These are values with identical shapes (or rather their opaque pointers).
  llvm::EquivalenceClasses<void *> equal_shapes_;
  // A map from a value that encodes a shape to a value that has this shape.
  llvm::DenseMap<ShapeValue, Value> symbolic_shapes_;
};

/// For arguments to kernels that have the same shape, use the stride and
/// shape information of the left-most argument inside of the kernel function.
/// That way, llvm can CSE index computations on same-shaped inputs.
struct PropagateShapeKnowledgeToKernels
    : public impl::PropagateShapeKnowledgeToKernelsBase<
          PropagateShapeKnowledgeToKernels> {
  void runOnOperation() override {
    ShapeEqualityKnowledge knowledge;

    knowledge.build(getOperation());

    getOperation().walk([&](gpu::LaunchFuncOp launch) {
      auto module = launch->getParentOfType<ModuleOp>();
      auto kernel = module.lookupSymbol<LLVM::LLVMFuncOp>(launch.getKernel());

      if (!kernel || kernel.isExternal()) return;

      llvm::SmallVector<std::pair<Value, int>, 4> seen_memrefs;
      // Position of the kernel argument we are currently at.
      int kernel_p = 0;
      for (auto operand : launch.getKernelOperands()) {
        auto memref = operand.getType().dyn_cast<MemRefType>();
        if (!memref) {
          // Scalar argument, advance kernel position by one.
          kernel_p++;
          continue;
        }
        for (auto previous : seen_memrefs) {
          if (!knowledge.haveSameShape(operand, previous.first)) {
            continue;
          }
          auto previous_type = previous.first.getType().cast<MemRefType>();
          // We use the first equality found and replace uses of corresponding
          // size and (potentially) stride information here.
          auto args_to_replace = memref.getRank();
          // If both memrefs have identity layouts, we can also reuse the
          // strides here, as they are the identity strides and hence fully
          // determinded by the shape.
          if (previous_type.getLayout().isIdentity() &&
              memref.getLayout().isIdentity()) {
            args_to_replace *= 2;
          }
          int previous_args_pos = previous.second;
          auto previous_args = kernel.getArguments()
                                   .drop_front(previous_args_pos + 3)
                                   .take_front(args_to_replace);
          auto current_args = kernel.getArguments()
                                  .drop_front(kernel_p + 3)
                                  .take_front(args_to_replace);
          for (auto pair : llvm::zip(previous_args, current_args)) {
            mlir::BlockArgument prev, curr;
            std::tie(prev, curr) = pair;
            curr.replaceAllUsesWith(prev);
          }
          break;
        }
        seen_memrefs.push_back({operand, kernel_p});
        // Advance base, aligned, offset, strides and sizes many arguments.
        kernel_p += memref.getRank() * 2 + 3;
      }
    });
  }
};

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
CreatePropagateShapeKnowledgeToKernels() {
  return std::make_unique<PropagateShapeKnowledgeToKernels>();
}

}  // namespace transforms
}  // namespace kernel_gen
}  // namespace mlir
