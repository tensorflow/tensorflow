/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_MLIR_TENSORFLOW_TRANSFORMS_CLUSTER_OPS_BY_POLICY_H_
#define TENSORFLOW_COMPILER_MLIR_TENSORFLOW_TRANSFORMS_CLUSTER_OPS_BY_POLICY_H_

#include <type_traits>

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/Region.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"

namespace mlir {
namespace TFDevice {

// -------------------------------------------------------------------------- //
// ValueConstraint.
// -------------------------------------------------------------------------- //

// In order to be clustered operation can require its operands to satisfy
// some constraints (e.g. reduction operation can require reduction dimension
// operand to be a constant value).
enum class ValueConstraint {
  // Operand must have statically known rank.
  kRank = 0,
  // Operand must have statically known shape (all dimensions are known at
  // compile time).
  kShape = 1,
  // Operand must have statically known value (operand must be defined by a
  // constant operation).
  kValue = 2,
};

// Returns the more restrictive constraint of `a` and `b`:
//
//    Value >> Shape >> Rank
//
// If you know the value, you always know the shape and the rank. If you know
// the shape, you always know the rank.
ValueConstraint Merge(ValueConstraint a, ValueConstraint b);

// Returns success if constraint can be resolved statically based on the value
// type, e.g. `shape` constraint can be resolved if the value is a tensor of
// statically known shape.
LogicalResult IsStaticallyResolved(Value value, ValueConstraint constraint);

raw_ostream& operator<<(raw_ostream& os, const ValueConstraint& constraint);

// -------------------------------------------------------------------------- //
// ValuesConstraintSet.
// -------------------------------------------------------------------------- //

// A set of constraints for values, that either operation results or operands.
class ValuesConstraintSet {
  using ConstraintsMap = llvm::SmallDenseMap<Value, ValueConstraint>;
  using ConstIterator = typename ConstraintsMap::const_iterator;

 public:
  ValuesConstraintSet() = default;

  // Inserts a new constraint for the `value`. If the `value` already has some
  // constraint, it will merge it with a new one, and will return a new
  // constraint value. Returned pair has a constraint value that was set for
  // a value, and a boolean flag that is true if the constraint was updated.
  std::pair<ValueConstraint, bool> Insert(Value value,
                                          ValueConstraint constraint);

  // Inserts constraints for multiple values.
  void Insert(ValueRange value, ValueConstraint constraint);

  // Walk all the constraints owned by this set.
  void Walk(llvm::function_ref<void(Value, ValueConstraint)> walk) const;

  // Returns the constraint of the value if it exists, or None otherwise.
  Optional<ValueConstraint> GetConstraint(Value value) const;
  bool HasConstraint(Value value) const;

  // Merges all constrains from the other constraints set into this one.
  void MergeAll(const ValuesConstraintSet& other);

  // Remove constraints that can be statically resolved from the type of the
  // constrained value (see `IsStaticallyResolved` defined above).
  ValuesConstraintSet& Resolve();

  // Reset all constraints.
  ValuesConstraintSet& Reset();

  // Return the number of constrained values in the set.
  size_t Size() const;

  // Returns true if the constraint set is empty.
  bool Empty() const;

  ConstIterator begin() const { return constraints_.begin(); }
  ConstIterator end() const { return constraints_.end(); }

 private:
  llvm::SmallDenseMap<Value, ValueConstraint> constraints_;
};

// -------------------------------------------------------------------------- //
// ClusteringPolicy.
// -------------------------------------------------------------------------- //

// Clustering policy specifies if the operation can be clustered (in practice it
// usually means that operation can be added to a cluster that will be later
// compiled) given the set of constraints on its results, and might propagate or
// create new constraints on the operation operands.
//
// Clustering policy must make a local decision just for a single operation. It
// is the responsibility of a clustering pass to combine all these individual
// operations constraints to form a valid cluster.
//
// Example: compilation using XLA (MHLO) lowering
//
//   %0 = "tf.Transpose"(%input, %perm)
//        : (tensor<?x?xf32>, tensor<2xi32>) -> tensor<?x?xf32>
//
//   XLAs `mhlo.transpose` operation requires permutation to be an attribute
//   (compile time value), so it means that if we want to put `tf.Transpose`
//   into a cluster that will be compiled with XLA, the `%perm` operand must
//   be a known compiled time value, e.g. result of a `tf.Const` operation.
//
class ClusteringPolicy {
 public:
  virtual ~ClusteringPolicy() = default;

  // Returns success if an operation can be clustered given the constraints on
  // the operation results. Updates operands constraits to satisfy all the
  // results constraints.
  virtual LogicalResult MatchAndUpdateConstraints(
      Operation* operation, const ValuesConstraintSet& results,
      ValuesConstraintSet& operands) const = 0;
};

// Clustering policy for a specific operation type.
template <typename OpTy>
class OpClusteringPolicy : public ClusteringPolicy {
 public:
  LogicalResult MatchAndUpdateConstraints(
      Operation* operation, const ValuesConstraintSet& results,
      ValuesConstraintSet& operands) const final {
    if (auto op = dyn_cast<OpTy>(operation))
      return MatchAndUpdateConstraints(op, results, operands);
    return failure();
  }

  virtual LogicalResult MatchAndUpdateConstraints(
      OpTy op, const ValuesConstraintSet& results,
      ValuesConstraintSet& operands) const = 0;
};

// -------------------------------------------------------------------------- //
// ClusteringPolicySet.
// -------------------------------------------------------------------------- //

// A set of clustering policies for different operations.
class ClusteringPolicySet {
 public:
  using Policies = std::vector<std::unique_ptr<ClusteringPolicy>>;

  const Policies& policies() const { return policies_; }

  // Add an instance of each of the policy types 'Ts'. Return a reference to
  // `this` for chaining insertions.
  template <typename... Ts>
  ClusteringPolicySet& Add() {
    (void)std::initializer_list<int>{0, (AddImpl<Ts>(), 0)...};
    return *this;
  }

  // ClusteringPolicySet is move only type.
  ClusteringPolicySet() = default;
  ClusteringPolicySet(const ClusteringPolicySet&) = delete;
  ClusteringPolicySet(ClusteringPolicySet&&) = default;
  ClusteringPolicySet& operator=(const ClusteringPolicySet&) = delete;
  ClusteringPolicySet& operator=(ClusteringPolicySet&&) = default;

 private:
  template <typename T, typename... Args>
  void AddImpl(Args&&... args) {
    static_assert(std::is_base_of<ClusteringPolicy, T>::value,
                  "T must implement ClusteringPolicy");
    policies_.emplace_back(std::make_unique<T>(std::forward<Args>(args)...));
  }

  std::vector<std::unique_ptr<ClusteringPolicy>> policies_;
};

// -------------------------------------------------------------------------- //
// Discovering clusters of operations based on the policy.
// -------------------------------------------------------------------------- //

// Cluster groups together operations in the single basic block based on the
// given clustering policy set. Clusters can be outlined into nested modules
// later device specific compilation (e.g. for TFRT JIT compiler).
struct Cluster {
  llvm::SmallVector<Operation*> operations;
  ValuesConstraintSet constraints;
};

// Returns clusters of operations in the given `block` based on the provided
// clustering policy. If `filter` is defined, it will be used to filter
// operations that can be considered for clustering based on the policy.
//
// TODO(ezhulenev): Additional filter function is a workaround for customizing
// clustering policies at runtime for experimentation. In the long term,
// clustering policy should be enough.
llvm::SmallVector<Cluster> FindClustersInTheBlock(
    Block* block, const ClusteringPolicySet& policies,
    std::function<bool(Operation* op)> filter = {});

// Creates a `tf_device.cluster` operation from the clustered operations.
tf_device::ClusterOp CreateClusterOp(Cluster& cluster, StringAttr policy = {});

// -------------------------------------------------------------------------- //
// Helper functions for value constraints propagations and analysis.
// -------------------------------------------------------------------------- //

// Propagates initial constraints on the values defined by the `constraints` set
// with operations in the `root` as a starting point, using user provided set of
// clustering policies.
//
// Filter predicate specifies if constraints should be propagated across the
// given operation. Operations in the root set will be also filtered using
// the `filter` predicate.
//
// Optionally resolve constraints that can be statically satisfied by the
// value type, and stop constraints propagation early.
//
// Optionally emits remarks attached to operation that failed to propagate
// results constraints to its operands (for testing purpose).
//
// Returns failure if constraints can't be propagated through some of the
// operations accepted by the filter (there is no clustering policy for an
// operation, or constraints can't be satisfied by the policy), and attaches
// error diagnostics to the operation that prevented constraints propagation.
mlir::LogicalResult PropagateValuesConstraints(
    llvm::ArrayRef<Operation*> root, std::function<bool(Operation*)> filter,
    const ClusteringPolicySet& policies, ValuesConstraintSet& constraints,
    bool resolve = false, bool emit_remarks = false);

// Propagates initial constraints on the values in the `region` to the other
// values in the same region, using user provided set of clustering policies.
mlir::LogicalResult PropagateValuesConstraints(
    mlir::Region& region, const ClusteringPolicySet& policies,
    ValuesConstraintSet& constraints, bool resolve = false,
    bool emit_remarks = false);

// Emits constraints remarks for all operations that use constrained values.
void EmitValueConstraintsRemarks(const ValuesConstraintSet& constraints);

// Emits constraints remarks for function inputs that are in the constraints
// set (entry block arguments have constraints).
void EmitInputsConstraintsRemarks(func::FuncOp func,
                                  const ValuesConstraintSet& constraints);

// Infers constraints for the values in the function body from the function
// results attributes.
//
// Example:
//   func @test(...) -> (tensor<?x?xf32> {tf.constraint = "shape"}) {
//     .....
//     %v = "some_operation"() : () -> tensor<?x?xf32>
//     return %v : tensor<?x?xf32>
//   }
LogicalResult InferFunctionBodyValuesConstraints(
    func::FuncOp func, ValuesConstraintSet& constraints);

}  // namespace TFDevice
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_TENSORFLOW_TRANSFORMS_CLUSTER_OPS_BY_POLICY_H_
