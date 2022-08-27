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

#include "tensorflow/compiler/xla/runtime/symbolic_shape.h"

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <numeric>

#include "absl/status/status.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Compiler.h"
#include "tensorflow/compiler/xla/runtime/arguments.h"
#include "tensorflow/compiler/xla/runtime/constraints.h"
#include "tensorflow/compiler/xla/runtime/logical_result.h"
#include "tensorflow/compiler/xla/runtime/types.h"

namespace xla {
namespace runtime {

using llvm::cast;
using llvm::dyn_cast;
using llvm::isa;

using llvm::ArrayRef;
using llvm::MutableArrayRef;

using SymbolicShape = SymbolicShapesResolver::SymbolicShape;
using StaticShape = SymbolicShapesResolver::StaticShape;

SymbolicShapesResolver::SymbolicShapesResolver(
    const FunctionType& signature,
    absl::Span<const ArgumentConstraint> constraints)
    : constraints_(constraints.begin(), constraints.end()) {
  for (unsigned i = 0; i < signature.num_operands(); ++i) {
    auto* type = signature.operand(i);

    // For unranked arguments we do not know any static shape information.
    if (isa<UnrankedTensorType, UnrankedMemrefType>(type)) {
      arguments_sizes_.emplace_back();
      continue;
    }

    auto emplace_sizes = [&](absl::Span<const int64_t> sizes) {
      arguments_sizes_.emplace_back(llvm::to_vector(sizes));

      // Keep track of all statically known dimension sizes.
      for (int64_t size : sizes) {
        if (size != MemrefType::kDynamicSize) seen_static_sizes_.insert(size);
      }
    };

    // Copy memref dimensions sizes from the signature type.
    if (auto* memref = dyn_cast<MemrefType>(type)) {
      emplace_sizes(memref->sizes());
      continue;
    }

    // Copy tensor dimensions sizes from the signature type.
    if (auto* tensor = dyn_cast<RankedTensorType>(type)) {
      emplace_sizes(tensor->sizes());
      continue;
    }

    // TODO(ezhulenev): Add support for `ShapedType` to allow users to enable
    // symbolic shape resolution for user-defined types.

    // All non-shaped types have statically known empty shape.
    emplace_sizes({});
  }

  // When resolving symbolic shapes we should visit arguments starting from the
  // more constrained ones, because they can change the static signature of the
  // function, and this information should be propagated to arguments with
  // dynamic shapes (e.g. all seen static sizes should be materialized in the
  // function signature).
  iteration_order_.resize(signature.num_operands());
  std::iota(iteration_order_.begin(), iteration_order_.end(), 0);

  // Make the sort stable so that dynamic shapes are computed deterministically.
  llvm::sort(iteration_order_, [&](size_t a, size_t b) {
    unsigned ca = static_cast<unsigned>(constraints[a]);
    unsigned cb = static_cast<unsigned>(constraints[b]);
    if (ca > cb) return true;
    return ca < cb ? false : a < b;
  });

  // We can safely skip arguments with a known empty symbolic shape, because
  // that's the default value we return when resolving symbolic shapes for
  // the arguments, and such shapes do not participate in the hash computation.
  llvm::erase_if(iteration_order_, [&](size_t i) {
    return arguments_sizes_[i].has_value() && arguments_sizes_[i]->empty();
  });

  // When computing a symbolic shapes hash we don't need to visit arguments with
  // a statically known shape.
  auto is_dynamic_shape_argument = [&](size_t idx) {
    return !arguments_sizes_[idx].has_value() ||
           llvm::any_of(*arguments_sizes_[idx],
                        [](int64_t d) { return d < 0; });
  };
  llvm::copy_if(iteration_order_, std::back_inserter(hash_iteration_order_),
                is_dynamic_shape_argument);
}

ArgumentConstraint SymbolicShapesResolver::constraint(size_t index) const {
  return constraints_[index];
}

size_t SymbolicShapesResolver::num_arguments() const {
  return arguments_sizes_.size();
}

bool SymbolicShapesResolver::has_argument_sizes(size_t index) const {
  return arguments_sizes_[index].has_value();
}

const StaticShape& SymbolicShapesResolver::argument_sizes(size_t index) const {
  return *arguments_sizes_[index];
}

bool SymbolicShapesResolver::seen_static_size(size_t dim) const {
  return seen_static_sizes_.contains(dim);
}

template <typename SymbolicShapes>
LLVM_ATTRIBUTE_ALWAYS_INLINE static LogicalResult ResolveImpl(
    const SymbolicShapesResolver& resolver, ArgumentsRef arguments,
    ArrayRef<size_t> iteration_order, SymbolicShapes& symbolic_shapes) {
  // The number of arguments must match the function signature.
  assert(arguments.size() == resolver.num_arguments());

  // Mapping from the runtime dimension size to the symbolic dimension.
  llvm::SmallDenseMap<int64_t, int64_t, 16> size_to_symbolic_dim;

  int64_t sym_dim = -2;  // the next symbolic dimension id

  for (size_t i : iteration_order) {
    bool has_static_sizes = resolver.has_argument_sizes(i);

    // TODO(ezhulenev): Add support for `ShapedArgument` to allow users to
    // enable symbolic shape resolution for user-defined arguments.
    //
    // At this point it's guaranteed that the argument at `i` is a shaped one,
    // because non-shaped argument are not in the `iteration_order`.
    const MemrefDesc* shaped = cast<MemrefDesc>(&arguments[i]);
    absl::Span<const int64_t> runtime_sizes = shaped->sizes();

    // Check that statically known rank matches the runtime rank.
    if (LLVM_UNLIKELY(has_static_sizes && resolver.argument_sizes(i).size() !=
                                              runtime_sizes.size()))
      return failure();

    // For shape constrained argument use runtime shape.
    if (resolver.constraint(i) == ArgumentConstraint::kShape) {
      symbolic_shapes[i].assign(runtime_sizes.begin(), runtime_sizes.end());

      // Add all runtime dimensions to the `size_to_symbolic_dim` to materialize
      // all dynamic dimensions of the same size as static dimensions.
      for (int64_t d : runtime_sizes) size_to_symbolic_dim.try_emplace(d, d);

      continue;
    }

    // Initialize symbolic shape with a statically known shape of the argument
    // if it is available, otherwise initialize it with a fully dynamic shape
    // with rank matching the runtime rank.
    if (has_static_sizes) {
      ArrayRef<int64_t> static_sizes = resolver.argument_sizes(i);
      assert(runtime_sizes.size() == static_sizes.size());
      symbolic_shapes[i].assign(static_sizes.begin(), static_sizes.end());
    } else {
      size_t rank = runtime_sizes.size();
      symbolic_shapes[i].resize(rank, MemrefType::kDynamicSize);
    }

    MutableArrayRef<int64_t> symbolic_sizes = symbolic_shapes[i];

    for (unsigned d = 0; d < runtime_sizes.size(); ++d) {
      int64_t symbolic_dim = symbolic_sizes[d];
      int64_t runtime_dim = runtime_sizes[d];

      // Skip statically known dimensions.
      if (symbolic_dim >= 0) {
        // Check that statically known dimension agrees with runtime dimension.
        if (LLVM_UNLIKELY(symbolic_dim != runtime_dim)) return failure();
        continue;
      }

      // Update unknown dimension to a static dimension.
      if (runtime_dim == 1 || resolver.seen_static_size(runtime_dim)) {
        symbolic_sizes[d] = runtime_dim;
        continue;
      }

      // Try to assign a symbolic dimension to the runtime dimension.
      auto emplaced = size_to_symbolic_dim.try_emplace(runtime_dim, sym_dim);
      symbolic_sizes[d] = emplaced.first->second;

      // Update the symbolic dimension if we assigned the previous value to the
      // runtime dimension size.
      if (emplaced.second) --sym_dim;
    }
  }

  return success();
}

absl::StatusOr<llvm::SmallVector<SymbolicShape>>
SymbolicShapesResolver::Resolve(ArgumentsRef arguments) const {
  // Prepare storage for resolving symbolic shapes.
  llvm::SmallVector<SymbolicShape> symbolic_shapes;
  symbolic_shapes.resize(arguments.size());

  if (LLVM_UNLIKELY(failed(
          ResolveImpl(*this, arguments, iteration_order_, symbolic_shapes))))
    return absl::InternalError("failed to resolve symbolic shape");

  return symbolic_shapes;
}

namespace {
// A struct to accumulate all resolved symbolic dimensions in a single vector.
// Resolved symbolic dimensions stored according to the iteration order, and not
// the argument order, however for computing the hash value it doesn't matter.
struct SymbolicShapesFingerprint {
  SymbolicShapesFingerprint() : offset(0) {}

  // Make sure that we do not copy the fingerprint.
  SymbolicShapesFingerprint(const SymbolicShapesFingerprint&) = delete;

  SymbolicShapesFingerprint& operator[](size_t i) { return *this; }

  template <typename InputIt>
  LLVM_ATTRIBUTE_ALWAYS_INLINE void assign(InputIt first, InputIt last) {
    auto rank = std::distance(first, last);
    offset = values.size();
    values.resize_for_overwrite(offset + rank);
    llvm::copy(llvm::make_range(first, last), values.begin() + offset);
  }

  LLVM_ATTRIBUTE_ALWAYS_INLINE void resize(int64_t rank, int64_t dim) {
    values.push_back(rank);
    offset = values.size();
    values.resize(offset + rank, dim);
  }

  operator MutableArrayRef<int64_t>() {  // NOLINT
    return {values.begin() + offset, values.end()};
  }

  size_t offset;
  llvm::SmallVector<int64_t, 32> values;
};
}  // namespace

absl::StatusOr<llvm::hash_code> SymbolicShapesResolver::ResolveHash(
    ArgumentsRef arguments) const {
  // Accumulate symbolic shapes into the shapes fingerprint.
  SymbolicShapesFingerprint fingerprint;

  if (LLVM_UNLIKELY(failed(
          ResolveImpl(*this, arguments, hash_iteration_order_, fingerprint))))
    return absl::InternalError("failed to resolve symbolic shape hash");

  return llvm::hash_combine_range(fingerprint.values.begin(),
                                  fingerprint.values.end());
}

/*static*/ StaticShape SymbolicShapesResolver::Normalize(
    const SymbolicShape& shape) {
  auto normalize = llvm::map_range(shape, [](int64_t dim) {
    return std::max(dim, MemrefType::kDynamicSize);
  });
  return {normalize.begin(), normalize.end()};
}

static llvm::hash_code SymbolicShapeHash(const SymbolicShape& shape) {
  return llvm::hash_combine(
      shape.size(), llvm::hash_combine_range(shape.begin(), shape.end()));
}

/*static*/ llvm::hash_code SymbolicShapesResolver::Hash(
    absl::Span<const SymbolicShape> symbolic_shapes) {
  if (LLVM_UNLIKELY(symbolic_shapes.empty())) return llvm::hash_code(0);

  llvm::hash_code hash = SymbolicShapeHash(symbolic_shapes[0]);
  for (unsigned i = 1; i < symbolic_shapes.size(); ++i)
    hash = llvm::hash_combine(hash, SymbolicShapeHash(symbolic_shapes[i]));

  return hash;
}

}  // namespace runtime
}  // namespace xla
