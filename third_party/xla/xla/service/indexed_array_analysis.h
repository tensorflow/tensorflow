/* Copyright 2018 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_INDEXED_ARRAY_ANALYSIS_H_
#define XLA_SERVICE_INDEXED_ARRAY_ANALYSIS_H_

#include <type_traits>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/literal.h"
#include "xla/service/hlo_pass_interface.h"
#include "xla/shape.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/statusor.h"

namespace xla {

// IndexedArrayAnalysis decides if an HLO instruction can be rewritten as a
// gather from another array.  It does this by mapping HLO instructions to
// instances of IndexedArrayAnalysis::Array, which can be inspected to discover
// whether said HLO is equivalent to a gather.
class IndexedArrayAnalysis {
 public:
  // IndexedArrayAnalysis maps each HLO instruction to an instance of a Array.
  // Array really just a sum type of the classes that inherit from it.  The
  // meaning of each of the subtypes is documented on the subtype declaration.
  //
  // Array instances are immutable once created.
  class Array {
   public:
    enum Kind {
      kUnknown,
      kConstant,
      kReshaped,
      kScalarIndexedConstant,
      kScalarIndexed
    };

    virtual Kind kind() const = 0;
    virtual const Shape& shape() const = 0;

    // Does a checked downcast from `Array` to `T` which must be one of its
    // subtypes.
    template <typename T>
    T* as() {
      static_assert((std::is_base_of<Array, T>::value),
                    "target type not derived from source type");
      // We skip the CHECK and hence the dynamic_cast if RTTI is disabled.
#if !defined(__GNUC__) || defined(__GXX_RTTI)
      CHECK_NE(dynamic_cast<T*>(this), nullptr);
#endif  // !defined(__GNUC__) || defined(__GXX_RTTI)

      return static_cast<T*>(this);
    }

    virtual ~Array() = default;

    Array& operator=(const Array& other) = delete;
  };

  // Represents an HLO instruction that was not analyzable by this
  // IndexedArrayAnalysis.  Instances of UnknownArray just wrap an existing
  // HloInstruction.
  class UnknownArray : public Array {
   public:
    Kind kind() const override { return kUnknown; }
    const Shape& shape() const override { return instruction().shape(); }
    const HloInstruction& instruction() const { return instruction_; }

   private:
    explicit UnknownArray(const HloInstruction* instr) : instruction_(*instr) {}

    const HloInstruction& instruction_;

    friend class IndexedArrayAnalysis;
  };

  // Represents a constant value.  This constant value may be present in the HLO
  // module being analyzed, or it could have been created on the fly by the
  // analysis.
  class ConstantArray : public Array {
   public:
    Kind kind() const override { return kConstant; }
    const Shape& shape() const override { return literal()->shape(); }
    const Literal* literal() const { return literal_; }

   private:
    explicit ConstantArray(const Literal* literal) : literal_(literal) {}
    const Literal* literal_;

    friend class IndexedArrayAnalysis;
  };

  // Represents an Array that is a reshape of another Array.
  class ReshapedArray : public Array {
   public:
    Kind kind() const override { return kReshaped; }

    // The array to reshape.
    Array* operand() const { return operand_; }

    // The output shape.
    const Shape& shape() const override { return shape_; }

   private:
    explicit ReshapedArray(Array* operand, Shape shape)
        : operand_(operand), shape_(shape) {}

    Array* operand_;
    const Shape shape_;

    friend class IndexedArrayAnalysis;
  };

  // ---------------------------------------------------------------------------
  // Indexed Array Overview
  // ---------------------------------------------------------------------------
  //
  // ScalarIndexedArray and ScalarIndexedConstantArray form the core of this
  // analysis.  ScalarIndexedConstantArray is just a specialization of
  // ScalarIndexedArray so we will only discuss ScalarIndexedArray in this
  // overview.
  //
  // A ScalarIndexedArray represents an array that can be computed by indexing
  // into a "source" array using an "indices" tensor.  A simple example is a
  // gather operation gathering 12 rows out of a [100,100] matrix -- such an
  // operation will be represented by an instance of a ScalarIndexedArray with
  // the [100,100] matrix as the "source" array and the [12]-shaped indices
  // array as the "indices" tensor.  The ScalarIndexedArray operation itself
  // will be of shape [12,100] (assuming we were gathering with axis=0).
  //
  // Gather operations are not the only operation that maps to
  // ScalarIndexedArray instances (if that were true there would be little point
  // in having a separate analysis).  We can often infer ScalarIndexedArrays for
  // other operations too.  For instance, consider:
  //
  //   %source = f32[100,100] constant
  //   %indices = s32[12] ...
  //   %gather = f32[12,100] ... gather from %source using %indices at axis 0
  //   %dot = dot(%gather, other_constant) [canonical contracting dims]
  //
  // The dot operation itself is also a ScalarIndexedArray with source =
  // dot(constant, other_constant) and indices = %indices.  A reshape of %gather
  // to [12,5,20] too is a ScalarIndexedArray with source = an appropriately
  // reshaped constant and indices = %indices.

  // Represents the result of a gather operation.  This gather operation may
  // explicitly be present in the HLO module being analyzed, or it could have
  // been created on the fly by the analysis.
  //
  // An instance of ScalarIndexedArray represents a array whose I'th element can
  // be mapped to the J'th element of the `source` array (where I and J are
  // multidimensional indices) in this way:
  //
  //   I' = remove components at positions `output_dims` from I
  //   G' = remove components not at positions `output_dims` from I
  //   T  = indices[G']
  //   J  = I' with T inserted at position `source_dim`
  //
  // For example, if source is of shape [11,13,17,19], indices is of shape
  // [23,29], output_dims is [0,2] and source_dim is 2 then the output is of
  // shape [23,11,29,13,19] and the output index [A,B,C,D,E] is mapped to the
  // input index [B,D,indices[A,C],E].
  class ScalarIndexedArray : public Array {
   public:
    Kind kind() const override { return kScalarIndexed; }
    const Shape& shape() const override { return shape_; }

    Array* source() const { return source_; }
    Array* indices() const { return indices_; }

    // `source_dim` is the dimension in the source array that is being indexed
    // over using indices from the `indices` array.  See the class documentation
    // and the overview for more details.
    int64_t source_dim() const { return source_dim_; }

    // `output_dims` are the dimensions in the output array that are being used
    // to compute an index into the `indices` array.  See the class
    // documentation and the overview for more details.
    absl::Span<const int64_t> output_dims() const { return output_dims_; }

   private:
    explicit ScalarIndexedArray(Array* source, Array* indices,
                                int64_t source_dim,
                                std::vector<int64_t> output_dims, Shape shape)
        : source_(source),
          indices_(indices),
          source_dim_(source_dim),
          output_dims_(std::move(output_dims)),
          shape_(std::move(shape)) {}

    Array* source_;
    Array* indices_;
    int64_t source_dim_;
    std::vector<int64_t> output_dims_;
    Shape shape_;

    friend class IndexedArrayAnalysis;
  };

  // A ScalarIndexedConstantArray is just a ScalarIndexedArray constrained to
  // have a ConstantArray instance as the source.  This is an ergonomic
  // concession -- in theory it is possible to just keep ScalarIndexedArray and
  // check source()->kind().
  class ScalarIndexedConstantArray : public ScalarIndexedArray {
   public:
    Kind kind() const override { return kScalarIndexedConstant; }

    const Literal& literal() const {
      return *source()->as<ConstantArray>()->literal();
    }

   private:
    explicit ScalarIndexedConstantArray(Array* source, Array* indices,
                                        int64_t source_dim,
                                        std::vector<int64_t> output_dims,
                                        Shape shape)
        : ScalarIndexedArray(source, indices, source_dim,
                             std::move(output_dims), std::move(shape)) {
      CHECK(dynamic_cast<ConstantArray*>(source));
    }

    friend class IndexedArrayAnalysis;
  };

  // Returns an Array instance for `instr`.  The IndexedArrayAnalysis instance
  // keeps ownership of the returned Array instance.
  //
  // Caching Behavior: IndexedArrayAnalysis has a cache mapping HLO
  // instructions to IndexedArrayAnalysis::Array instances.  This entire cache
  // becomes stale and may cause the analysis to return incorrect results if any
  // transitive operand (stopping at the containing computation) is modified for
  // any HLO instruction on which GetArrayFor has been invoked.
  //
  // NB!  By inspecting the implementation, you may be able to infer a stronger
  // caching guarantee than what is mentioned above.  Nevertheless, what is
  // stated above is the contract.
  absl::StatusOr<Array*> GetArrayFor(const HloInstruction* instr);

  // Pretty-prints the expression rooted at `root`.
  std::string ToString(Array* root, bool print_constants = false);

 private:
  // Helper function that ensures that every HLO instruction that is
  // transitively used by `root` has an entry in `cache_`.
  absl::Status TraverseAndPopulateCache(const HloInstruction* root);

  // Creates an Array instance for `instr` under the assumption that all
  // operations of `instr` are present in `cache_`.
  absl::StatusOr<Array*> ComputeArrayFor(const HloInstruction* instr);

  absl::StatusOr<Array*> ComputeArrayForConstant(const Literal& literal);

  absl::StatusOr<Array*> ComputeArrayForGather(
      const Shape& shape, const GatherDimensionNumbers& dim_numbers,
      absl::Span<const int64_t> slice_sizes, Array* source, Array* indices);

  absl::StatusOr<Array*> ComputeArrayForDotWithIndexedLhs(
      const Shape& shape, const DotDimensionNumbers& dim_numbers,
      const PrecisionConfig& precision_config, ScalarIndexedConstantArray* lhs,
      ConstantArray* rhs);

  absl::StatusOr<Array*> ComputeArrayForDotWithIndexedRhs(
      const Shape& shape, const DotDimensionNumbers& dim_numbers,
      const PrecisionConfig& precision_config, ConstantArray* lhs,
      ScalarIndexedConstantArray* rhs);

  absl::StatusOr<Array*> ComputeArrayForDot(
      const Shape& shape, const DotDimensionNumbers& dim_numbers,
      const PrecisionConfig& precision_config, Array* lhs, Array* rhs);

  // This tries to fold a ScalarIndexedArray which has another
  // ScalarIndexedArray as a source into a ScalarIndexedArray that instead has a
  // ScalarIndexedArray as indices.  If `source` happened to be a
  // ScalarIndexedConstantArray this can result in an expression that is more
  // canonical.
  //
  // As an example, consider a gather operation, G0, gathering 7 elements from
  // an array "Arr" of shape [100] resulting in an array of shape [7], and a
  // second gather operation, G1, which gathers 3 elements out of the result of
  // G0 resulting in an array of shape [3].  Let the indices uses by G0 be I0
  // (of shape [7]) and the indices used by G1 be I1 (of shape [3]).  We can
  // instead rewrite G1 to gather directly from "Arr" with the three indices
  // from I0 as per I1.  In other words, we can rewrite:
  //
  //    G0 = [Arr[i] for i in I0]
  //    G1 = [G0[i]  for i in I1]
  //
  // into
  //
  //    I2 = [I0[i]  for i in I1]
  //    G1 = [Arr[i] for i in I2]
  absl::StatusOr<ScalarIndexedArray*> FoldGatherOfGather(
      ScalarIndexedArray* source, Array* indices, int64_t source_dim,
      absl::Span<const int64_t> output_dims, Shape shape);

  // Reshapes a scalar-indexed node to remove the degenerate dimensions in its
  // output.  The result is always a scalar-indexed node.
  absl::StatusOr<ScalarIndexedArray*> ReshapeToRemoveDegenerateDims(
      ScalarIndexedArray* operand);

  // Reshapes a scalar-indexed node such that the result has the degenerate
  // dimensions `degenerate_dims`.  The result is always a scalar-indexed node.
  absl::StatusOr<ScalarIndexedArray*> ReshapeToAddDegenerateDims(
      ScalarIndexedArray* operand, absl::Span<const int64_t> degenerate_dims);

  absl::StatusOr<ScalarIndexedArray*> FoldReshapeOfGather(
      const Shape& shape, ScalarIndexedConstantArray* operand);
  absl::StatusOr<ScalarIndexedArray*> FoldReshapeOfGatherNoDegenerateDims(
      const Shape& shape, ScalarIndexedConstantArray* scalar_indexed);
  absl::StatusOr<Array*> ComputeArrayForReshape(const Shape& shape,
                                                Array* operand);

  absl::StatusOr<Array*> ComputeArrayForElementwiseBinaryOp(HloOpcode opcode,
                                                            Array* lhs,
                                                            Array* rhs);
  absl::StatusOr<Array*> ComputeArrayForElementwiseUnaryOp(HloOpcode opcode,
                                                           Array* operand);

  template <typename T, typename... Args>
  T* Construct(Args&&... args) {
    T* new_tensor = new T(std::forward<Args>(args)...);
    owned_tensors_.push_back(std::unique_ptr<T>(new_tensor));
    return new_tensor;
  }

  ScalarIndexedArray* ConstructScalarIndexedArray(
      Array* source, Array* indices, int64_t source_dim,
      std::vector<int64_t> output_dims, Shape shape) {
    if (source->kind() == Array::kConstant) {
      return Construct<ScalarIndexedConstantArray>(source, indices, source_dim,
                                                   std::move(output_dims),
                                                   std::move(shape));
    } else {
      return Construct<ScalarIndexedArray>(source, indices, source_dim,
                                           std::move(output_dims),
                                           std::move(shape));
    }
  }

  Literal* TakeOwnership(Literal literal) {
    owned_literals_.push_back(std::move(literal));
    return &owned_literals_.back();
  }

  absl::StatusOr<Literal*> TakeOwnership(
      absl::StatusOr<Literal> literal_or_error) {
    TF_ASSIGN_OR_RETURN(Literal literal, std::move(literal_or_error));
    owned_literals_.push_back(std::move(literal));
    return &owned_literals_.back();
  }

  std::vector<std::unique_ptr<Array>> owned_tensors_;
  std::vector<Literal> owned_literals_;
  absl::flat_hash_map<const HloInstruction*, Array*> cache_;
};

// A pass that prints all non-trivial results returned by IndexedArrayAnalysis.
// This pass is a no-op if !VLOG_IS_ON(2) so it should be fine to
// unconditionally add to the regular HLO pass pipeline.
class IndexedArrayAnalysisPrinterPass : public HloModulePass {
 public:
  absl::string_view name() const override {
    return "indexed-array-analysis-printer-pass";
  }
  using HloPassInterface::Run;
  absl::StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;
};

}  // namespace xla

#endif  // XLA_SERVICE_INDEXED_ARRAY_ANALYSIS_H_
