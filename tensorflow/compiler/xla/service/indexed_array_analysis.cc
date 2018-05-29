/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/indexed_array_analysis.h"
#include "tensorflow/compiler/xla/map_util.h"
#include "tensorflow/compiler/xla/service/hlo_evaluator.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/lib/gtl/flatset.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/lib/strings/strcat.h"

namespace xla {
namespace gtl = ::tensorflow::gtl;

namespace {
using Analysis = IndexedArrayAnalysis;
using UnknownArray = Analysis::UnknownArray;
using ConstantArray = Analysis::ConstantArray;
using ScalarIndexedArray = Analysis::ScalarIndexedArray;
using tensorflow::gtl::ArraySlice;
using tensorflow::str_util::Join;
}  // namespace

string IndexedArrayAnalysis::ToString(Array* root, bool print_constants) {
  switch (root->kind()) {
    case Array::kUnknown: {
      auto* unknown_tensor = root->as<UnknownArray>();
      return tensorflow::strings::StrCat("%",
                                         unknown_tensor->instruction().name());
    }

    case Array::kConstant: {
      if (print_constants) {
        string contents = root->as<ConstantArray>()->literal()->ToString();
        return tensorflow::strings::StrCat(
            "(constant ", ShapeUtil::HumanString(root->shape()), " ", contents,
            ")");
      }
      return tensorflow::strings::StrCat(
          "(constant ", ShapeUtil::HumanString(root->shape()), ")");
    }

    case Array::kScalarIndexedConstant:
    case Array::kScalarIndexed: {
      auto* indexed_array = root->as<ScalarIndexedArray>();
      string name = root->kind() == Array::kScalarIndexedConstant
                        ? "scalar-indexed-const"
                        : "scalar-indexed";
      return tensorflow::strings::StrCat(
          "(", name, " ", ToString(indexed_array->source(), print_constants),
          " ", ToString(indexed_array->indices(), print_constants), " ",
          indexed_array->source_dim(), "->[",
          Join(indexed_array->output_dims(), ","), "])");
    }
  }
}

StatusOr<Analysis::Array*> IndexedArrayAnalysis::GetArrayFor(
    const HloInstruction* instr) {
  auto it = cache_.find(instr);
  if (it != cache_.end()) {
    return it->second;
  }

  TF_RETURN_IF_ERROR(TraverseAndPopulateCache(instr));
  return FindOrDie(cache_, instr);
}

Status IndexedArrayAnalysis::TraverseAndPopulateCache(
    const HloInstruction* root) {
  // Depth first search over the DAG, invoking ComputeArrayFor in post order.
  // The HLO instructions already in the cache are considered leaves.

  gtl::InlinedVector<const HloInstruction*, 4> stack;

  enum DfsState { kDiscovered, kVisited };
  gtl::FlatMap<const HloInstruction*, DfsState> dfs_state_map;

  stack.push_back(root);
  InsertOrDie(&dfs_state_map, root, kDiscovered);

  do {
    const HloInstruction* instr = stack.back();
    if (cache_.count(instr)) {
      stack.pop_back();
      continue;
    }

    switch (FindOrDie(dfs_state_map, instr)) {
      case kDiscovered: {
        for (const HloInstruction* operand : instr->operands()) {
          if (!cache_.count(operand)) {
            stack.push_back(operand);
            CHECK(!dfs_state_map.count(operand) ||
                  dfs_state_map[operand] == kDiscovered);
            dfs_state_map[operand] = kDiscovered;
          }
        }
        dfs_state_map[instr] = kVisited;
        break;
      }

      case kVisited:
        stack.pop_back();
        TF_ASSIGN_OR_RETURN(Array * array, ComputeArrayFor(instr));
        InsertOrDie(&cache_, instr, array);
        break;
    }
  } while (!stack.empty());

  return Status::OK();
}

StatusOr<Analysis::Array*> IndexedArrayAnalysis::ComputeArrayFor(
    const HloInstruction* instr) {
  Array* computed_array;
  if (instr->IsElementwise() && instr->operand_count() == 1) {
    TF_ASSIGN_OR_RETURN(
        computed_array,
        ComputeArrayForElementwiseUnaryOp(
            instr->opcode(), FindOrDie(cache_, instr->operand(0))));
  } else if (instr->IsElementwise() && instr->operand_count() == 2) {
    TF_ASSIGN_OR_RETURN(
        computed_array,
        ComputeArrayForElementwiseBinaryOp(
            instr->opcode(), FindOrDie(cache_, instr->operand(0)),
            FindOrDie(cache_, instr->operand(1))));
  } else if (instr->opcode() == HloOpcode::kConstant) {
    TF_ASSIGN_OR_RETURN(computed_array,
                        ComputeArrayForConstant(instr->literal()));
  } else if (instr->opcode() == HloOpcode::kGather) {
    TF_ASSIGN_OR_RETURN(
        computed_array,
        ComputeArrayForGather(instr->shape(), instr->gather_dimension_numbers(),
                              instr->gather_window_bounds(),
                              FindOrDie(cache_, instr->operand(0)),
                              FindOrDie(cache_, instr->operand(1))));
  } else if (instr->opcode() == HloOpcode::kReshape) {
    TF_ASSIGN_OR_RETURN(
        computed_array,
        ComputeArrayForReshape(instr->shape(),
                               FindOrDie(cache_, instr->operand(0))));
  } else {
    computed_array = nullptr;
  }

  if (!computed_array) {
    computed_array = Construct<UnknownArray>(instr);
  }

  return computed_array;
}

StatusOr<Analysis::Array*> IndexedArrayAnalysis::ComputeArrayForConstant(
    const Literal& literal) {
  return Construct<ConstantArray>(&literal);
}

StatusOr<ScalarIndexedArray*> IndexedArrayAnalysis::FoldGatherOfGather(
    ScalarIndexedArray* source, Array* indices, int64 source_dim,
    tensorflow::gtl::ArraySlice<int64> output_dims, Shape shape) {
  // We want to transform Gather(Gather(A, X), Y) => Gather(A, Gather(X, Y)).
  // `source` is the inner Gather(A, X).

  Array* a = source->source();
  Array* x = source->indices();
  Array* y = indices;

  // This bit is slightly tricky, so we do a naive "simulation" of the two
  // consecutive gather operations to infer what the composed gather should look
  // like.

  enum class IndexComponent { Ungathered, GatheredFirst, GatheredSecond };

  std::vector<IndexComponent> simulated_index(a->shape().dimensions_size(),
                                              IndexComponent::Ungathered);

  // Simulate the first gather.
  EraseAt(&simulated_index, source->source_dim());
  for (int64 gather_dim : source->output_dims()) {
    simulated_index.insert(simulated_index.begin() + gather_dim,
                           IndexComponent::GatheredFirst);
  }

  // Simulate the second gather.
  EraseAt(&simulated_index, source_dim);
  for (int64 output_dim : output_dims) {
    simulated_index.insert(simulated_index.begin() + output_dim,
                           IndexComponent::GatheredSecond);
  }

  int64 source_dim_for_index_array =
      FindIndex(source->output_dims(), source_dim);
  CHECK_NE(source_dim_for_index_array, source->output_dims().size());

  std::vector<int64> output_dims_for_index_array;
  int64 gathered_index_components_seen = 0;
  for (IndexComponent simulation_dim : simulated_index) {
    if (simulation_dim == IndexComponent::GatheredSecond) {
      output_dims_for_index_array.push_back(gathered_index_components_seen);
    }
    if (simulation_dim != IndexComponent::Ungathered) {
      gathered_index_components_seen++;
    }
  }

  std::vector<int64> dim_sizes_for_composed_index;
  std::vector<int64> output_dims_for_new_gather;
  for (int64 i = 0, e = simulated_index.size(); i < e; i++) {
    if (simulated_index[i] != IndexComponent::Ungathered) {
      dim_sizes_for_composed_index.push_back(shape.dimensions(i));
      output_dims_for_new_gather.push_back(i);
    }
  }

  Array* inner_indices = ConstructScalarIndexedArray(
      x, y, source_dim_for_index_array, output_dims_for_index_array,
      ShapeUtil::MakeShape(x->shape().element_type(),
                           dim_sizes_for_composed_index));
  return ConstructScalarIndexedArray(a, inner_indices, source->source_dim(),
                                     output_dims_for_new_gather,
                                     std::move(shape));
}

StatusOr<Analysis::Array*> IndexedArrayAnalysis::ComputeArrayForGather(
    const Shape& shape, const GatherDimensionNumbers& dim_numbers,
    tensorflow::gtl::ArraySlice<int64> window_bounds, Array* source,
    Array* indices) {
  if (dim_numbers.index_vector_dim() != indices->shape().dimensions_size()) {
    return nullptr;
  }

  CHECK_EQ(dim_numbers.gather_dims_to_operand_dims_size(), 1);
  if (!c_binary_search(dim_numbers.elided_window_dims(),
                       dim_numbers.gather_dims_to_operand_dims(0))) {
    return nullptr;
  }

  int64 source_dim = dim_numbers.gather_dims_to_operand_dims(0);
  std::vector<int64> output_dims;
  for (int64 i = 0, e = shape.dimensions_size(); i < e; i++) {
    if (!c_binary_search(dim_numbers.output_window_dims(), i)) {
      output_dims.push_back(i);
    }
  }

  if (auto* indexed = dynamic_cast<ScalarIndexedArray*>(source)) {
    auto it = c_find(indexed->output_dims(), source_dim);
    if (it != indexed->output_dims().end()) {
      return FoldGatherOfGather(indexed, indices, source_dim, output_dims,
                                shape);
    }
  } else if (auto* constant = dynamic_cast<ConstantArray*>(source)) {
    return Construct<ScalarIndexedConstantArray>(constant, indices, source_dim,
                                                 output_dims, shape);
  }

  return Construct<ScalarIndexedArray>(source, indices, source_dim, output_dims,
                                       shape);
}

namespace {
// Returns an index into `values` such that the product of the range
// [values.begin()+index, values.end()) is equal to `product`.  If there is no
// such index, return -1.  All integers in `values` must be positive.
int64 FindSuffixWithProduct(ArraySlice<int64> values, int64 product) {
  DCHECK(c_all_of(values, [](int64 value) { return value > 0; }));

  int64 current_product = 1;
  int64 i;
  for (i = values.size() - 1; i >= 0 && product > current_product; --i) {
    current_product *= values[i];
  }

  if (product == current_product) {
    return i + 1;
  }

  return -1;
}

struct ReshapePassthroughDimPair {
  int64 result_dim;
  int64 operand_dim;
};

// Returns a set of dimension pairs such for all (result_dim, operand_dim) in
// the set:
//
// output_index[result_dim] = SourceIndexOfReshape(output_index)[operand_dim]
//
// The returned vector of pairs is sorted in both the result_dim and the
// operand_dim components.
std::vector<ReshapePassthroughDimPair> ComputeReshapePassthroughDimPairs(
    ArraySlice<int64> operand_shape, ArraySlice<int64> result_shape) {
  // A reshape can be seen as an index mapping from output index to input index:
  //
  // (i_0, ..., i_n) = f(o_0, ..., o_m)
  //
  // This function returns the pairs (j, k) for which the following invariant
  // holds for all indices in the shape:
  //
  //   o_j == i_k
  //
  // And this occurs when:
  //
  //    O_{j+1} * ... * O_n == I_{k+1} * ...  * I_m
  //
  // (where O_x are the sizes of the output shape and I_x are the sizes of the
  // input shape) and the size of the dimension j of the result is the same as
  // the size of dimension k in the operand.
  //
  // These conditions are sufficient because the Reshape HLO is spec'ed such
  // that the rightmost dimensions are always minor in the flattening and refine
  // operation.

  std::vector<ReshapePassthroughDimPair> result;
  int64 result_subarray_size = 1;
  for (int64 result_dim = result_shape.size() - 1; result_dim >= 0;
       --result_dim) {
    int64 candidate_operand_dim =
        FindSuffixWithProduct(operand_shape, result_subarray_size);

    // result_subarray_size does not include the elements in the current
    // `result_dim` dimension (we multiply in result_shape[result_dim] at the
    // end of loop body) so candidate_operand_dim can never be zero.
    CHECK_NE(candidate_operand_dim, 0);

    if (candidate_operand_dim != -1 &&
        result_shape[result_dim] == operand_shape[candidate_operand_dim - 1]) {
      result.push_back({/*result_dim=*/result_dim,
                        /*operand_dim=*/candidate_operand_dim - 1});
    }
    result_subarray_size *= result_shape[result_dim];
  }

  c_reverse(result);

  if (VLOG_IS_ON(3)) {
    std::vector<string> result_strings;
    c_transform(result, std::back_inserter(result_strings),
                [](ReshapePassthroughDimPair value) {
                  return tensorflow::strings::StrCat(value.result_dim, "->",
                                                     value.operand_dim);
                });
    VLOG(3) << "For a reshape from [" << Join(operand_shape, ",") << "] to ["
            << Join(result_shape, ",") << "] passthrough indices are ["
            << Join(result_strings, ",") << "]";
  }

  DCHECK(c_is_sorted(
      result, [](ReshapePassthroughDimPair lhs, ReshapePassthroughDimPair rhs) {
        return lhs.result_dim < rhs.result_dim;
      }));

  DCHECK(c_is_sorted(
      result, [](ReshapePassthroughDimPair lhs, ReshapePassthroughDimPair rhs) {
        return lhs.operand_dim < rhs.operand_dim;
      }));

  return result;
}

// Return true if `dim` is stated as an passthrough operand dim in
// `passthrough_dims`.
bool IsReshapePassthroughOperandDim(
    ArraySlice<ReshapePassthroughDimPair> passthrough_dims, int64 dim) {
  return c_any_of(passthrough_dims,
                  [&](ReshapePassthroughDimPair passthrough_dim_pair) {
                    return passthrough_dim_pair.operand_dim == dim;
                  });
}

// Maps `operand_dim` which must be an passthrough operand dimension to its
// corresponding passthrough result dimension based on `passthrough_dims`.
int64 MapPassthroughOperandDimToResultDim(
    ArraySlice<ReshapePassthroughDimPair> passthrough_dims, int64 operand_dim) {
  auto it = c_find_if(passthrough_dims,
                      [&](ReshapePassthroughDimPair passthrough_dim_pair) {
                        return passthrough_dim_pair.operand_dim == operand_dim;
                      });
  CHECK(it != passthrough_dims.end());
  return it->result_dim;
}

int64 FindSourcePositionForPassthroughResultDim(ArraySlice<int64> operand_shape,
                                                ArraySlice<int64> result_shape,
                                                int64 source_passthrough_dim) {
  int64 indexed_source_subarray_size =
      std::accumulate(operand_shape.begin() + source_passthrough_dim + 1,
                      operand_shape.end(), 1, std::multiplies<int64>());

  return FindSuffixWithProduct(result_shape, indexed_source_subarray_size);
}

};  // namespace

StatusOr<Analysis::Array*> IndexedArrayAnalysis::ComputeArrayForReshape(
    const Shape& shape, Array* operand) {
  auto* scalar_indexed = dynamic_cast<ScalarIndexedConstantArray*>(operand);
  if (!scalar_indexed) {
    return nullptr;
  }

  // Try to fold Reshape(ScalarIndexed(Const, Indices))
  //          => ScalarIndexed(Const', Indices)
  //
  // We can view the reshape and the scalar-indexed operations as functions that
  // map an output index (i.e. an index into the result) to an input index
  // (i.e. an index into the operand).  The key idea used here is that the
  // output-to-input mapping for some reshape operations may "pass through" some
  // output dimensions into the input space unchanged -- i.e. there may exist
  // output dimension "O" and input dimension "I" such that OutputIndex[O] is
  // always == InputIndexForReshape(OutputIndex)[I].  If these pass-through
  // dimensions in the input space of the reshape happen to be include all the
  // output dimensions for the scalar-indexed node then, roughly, the following
  // holds:
  //
  //    SourceIndexOfScalarIndexed(SourceIndexOfReshape(Idx))
  // == SourceIndexOfScalarIndexed(SourceIndexOfReshape(Ps ++ Qs))
  //
  //      Where Ps are the set of the pass-through components of Idx that are
  //      also the output dims of the scalar-indexed node, and Qs are the rest.
  //      For brevity, we're playing fast and loose with the notation here -- we
  //      don't literally require Idx to be a concatenation of Ps and Qs, as
  //      suggested by the "++".
  //
  // == SourceIndexOfScalarIndexed(Ps ++ SourceIndexOfReshape(Qs))
  //
  //      Again, we're playing fast and loose with the notation around "++".
  //      Generally this ++ will be a different function that the ++ in the
  //      previous step.
  //
  // If the scalar-indexed node has a constant as the source then the
  // SourceIndexOfReshape function can be "folded into" the constant itself by
  // reshaping it, leaving us with:
  //
  // == SourceIndexOfScalarIndexed(Ps ++ Qs)
  // == SourceIndexOfScalarIndexed(Idx)
  //
  // which is just a scalar-indexed node (with parameters different from the
  // scalar-indexed node we started with) with a reshaped constant as the
  // source.
  //
  // We can't fold SourceIndexOfReshape into the constant without introducing
  // another precondition: since the new scalar-indexed node will have a
  // reshaped (constant) array as its source it will, in general, have a
  // different source dimension than the original scalar-indexed node.  This
  // source dimension will have to be a passthrough dimension of the
  // SourceIndexOfReshape indexing function that is folded into the source. And
  // such a dimension need not exist so this is a non-trivial precondition.

  std::vector<ReshapePassthroughDimPair> reshape_passthrough_dims =
      ComputeReshapePassthroughDimPairs(
          /*operand_shape=*/AsInt64Slice(operand->shape().dimensions()),
          /*result_shape=*/AsInt64Slice(shape.dimensions()));

  auto is_reshape_passthrough_operand_dim = [&](int64 operand_dim) {
    return IsReshapePassthroughOperandDim(reshape_passthrough_dims,
                                          operand_dim);
  };

  if (!c_all_of(scalar_indexed->output_dims(),
                is_reshape_passthrough_operand_dim)) {
    return nullptr;
  }

  // To compute the shape of the source for the new scalar-indexed node we're
  // going to create, we first "undo" the scalar-indexed operation.
  std::vector<int64> new_scalar_indexed_source_shape(shape.dimensions().begin(),
                                                     shape.dimensions().end());
  for (int64 i = scalar_indexed->output_dims().size() - 1; i >= 0; i--) {
    int64 output_dim = scalar_indexed->output_dims()[i];
    int64 output_dim_after_reshape = MapPassthroughOperandDimToResultDim(
        reshape_passthrough_dims, output_dim);
    EraseAt(&new_scalar_indexed_source_shape, output_dim_after_reshape);
  }

  // After this, we need to add in the dimension that will be the source
  // dimension for the new scalar-indexed node.  A scalar-indexed node "removes"
  // the source dimensions and "adds" the output dimensions, so to get back to
  // the shape for the *source* of the scalar-indexed node we need to remove the
  // output dims (which we did above) and then add back the source dim (which we
  // are about to do below):

  const Shape& scalar_indexed_source_shape = scalar_indexed->source()->shape();

  int64 source_dim_for_new_scalar_indexed_node =
      FindSourcePositionForPassthroughResultDim(
          /*operand_shape=*/AsInt64Slice(
              scalar_indexed_source_shape.dimensions()),
          /*result_shape=*/new_scalar_indexed_source_shape,
          scalar_indexed->source_dim());

  // We may not be able to find a source dim for the new scalar-indexed node.
  // For instance consider:
  //
  //   operand = s32[3,5,2] constant({...})
  //   indices = s32[7] parameter(0)
  //   gather = s32[3,2,7] gather(operand, indices),
  //       output_window_dims={0,1},
  //       elided_window_dims={1},
  //       gather_dims_to_operand_dims={1},
  //       index_vector_dim=1,
  //       window_bounds={3,1,2}
  //   reshape = s32[6,7] reshape(gather)
  //
  // In this case the gather maps to:
  //    (scalar-indexed-const (constant s32[3,5,2]) %indices 1->[2])
  //
  // and the reshape passes through dimension 2 from its input into dimension 1
  // in its output.  However, we can't rewrite the reshape as a scalar-indexed
  // node because then we'd have to reshape the [3,5,2] `operand` array to
  // [6,5], but then dimension 1 of the reshaped [6,5] array indexes differently
  // (a.k.a. isn't pass-through) than the [3,5,2] array.

  if (source_dim_for_new_scalar_indexed_node == -1) {
    return nullptr;
  }

  InsertAt(
      &new_scalar_indexed_source_shape, source_dim_for_new_scalar_indexed_node,
      scalar_indexed_source_shape.dimensions(scalar_indexed->source_dim()));

  CHECK(IsReshapePassthroughOperandDim(
      ComputeReshapePassthroughDimPairs(
          /*operand_shape=*/AsInt64Slice(
              scalar_indexed_source_shape.dimensions()),
          /*result_shape=*/new_scalar_indexed_source_shape),
      scalar_indexed->source_dim()));

  auto map_passthrough_operand_dim_to_result_dim = [&](int64 result_dim) {
    return MapPassthroughOperandDimToResultDim(reshape_passthrough_dims,
                                               result_dim);
  };

  std::vector<int64> output_dims_for_new_scalar_indexed_node;
  c_transform(scalar_indexed->output_dims(),
              std::back_inserter(output_dims_for_new_scalar_indexed_node),
              map_passthrough_operand_dim_to_result_dim);

  TF_ASSIGN_OR_RETURN(const Literal* new_scalar_indexed_source_literal,
                      TakeOwnership(scalar_indexed->literal().Reshape(
                          new_scalar_indexed_source_shape)));
  TF_ASSIGN_OR_RETURN(
      Array * new_scalar_indexed_source,
      ComputeArrayForConstant(*new_scalar_indexed_source_literal));

  return ConstructScalarIndexedArray(
      new_scalar_indexed_source, scalar_indexed->indices(),
      source_dim_for_new_scalar_indexed_node,
      output_dims_for_new_scalar_indexed_node, shape);
}

StatusOr<Analysis::Array*>
IndexedArrayAnalysis::ComputeArrayForElementwiseBinaryOp(HloOpcode opcode,
                                                         Array* lhs,
                                                         Array* rhs) {
  // Try to fold BinaryOp(Broadcast(Const0), ScalarIndexed(Const1, Indices))
  //          => ScalarIndexed(BinaryOp(Broadcast'(Const0), Const1), Indices)
  //
  // We can do this if every output dimension from the scalar-indexed node is a
  // broadcasted dimension for the broadcast node.  Informally, the precondition
  // means Broadcast(Const0)[IDX] is solely a function of the components of IDX
  // that are not output-dims for the scalar-indexed node. In other words, for
  // every assignment to the non-output dims in IDX we have a "constant" LHS to
  // the BinaryOp.  This transform propagates this "constant" to the source for
  // the scalar-indexed node.

  ScalarIndexedConstantArray* lhs_scalar_indexed_const =
      dynamic_cast<ScalarIndexedConstantArray*>(lhs);
  ScalarIndexedConstantArray* rhs_scalar_indexed_const =
      dynamic_cast<ScalarIndexedConstantArray*>(rhs);

  bool lhs_is_indexed;

  // One of the operands must be scalar-indexed and the other must be a
  // broadcast of a constant.
  if (lhs_scalar_indexed_const && !rhs_scalar_indexed_const) {
    lhs_is_indexed = true;
  } else if (rhs_scalar_indexed_const && !lhs_scalar_indexed_const) {
    lhs_is_indexed = false;
  } else {
    return nullptr;
  }

  ScalarIndexedConstantArray* scalar_indexed_const =
      lhs_is_indexed ? lhs_scalar_indexed_const : rhs_scalar_indexed_const;
  UnknownArray* candidate_broadcast_array =
      dynamic_cast<UnknownArray*>(lhs_is_indexed ? rhs : lhs);
  if (!candidate_broadcast_array ||
      candidate_broadcast_array->instruction().opcode() !=
          HloOpcode::kBroadcast) {
    return nullptr;
  }

  const HloInstruction* broadcast_instr =
      &candidate_broadcast_array->instruction();
  const HloInstruction* broadcast_const_operand = broadcast_instr->operand(0);
  if (broadcast_const_operand->opcode() != HloOpcode::kConstant) {
    return nullptr;
  }

  ArraySlice<int64> broadcast_dims = broadcast_instr->dimensions();
  auto is_broadcasted_dim = [&](int64 output_dim) {
    return c_find(broadcast_dims, output_dim) == broadcast_dims.end();
  };

  // All of the output dims must be "broadcasted" dims for the other operand.
  if (!c_all_of(scalar_indexed_const->output_dims(), is_broadcasted_dim)) {
    return nullptr;
  }

  // To figure out the broadcast dimensions for the (constant) source for the
  // scalar-indexed node, we "simulate" the index transformation done by the
  // existing broadcsat:
  enum class IndexComponent { Broadcasted, NotBroadcasted };
  std::vector<IndexComponent> simulated_index(
      broadcast_instr->shape().dimensions_size(), IndexComponent::Broadcasted);
  for (int64 broadcast_dim : broadcast_dims) {
    simulated_index[broadcast_dim] = IndexComponent::NotBroadcasted;
  }

  // The scalar-indexed node "removes" the source dim and "inserts" the output
  // dims.  We do the opposite here to undo the scalar-indexed operation.
  ArraySlice<int64> output_dims = scalar_indexed_const->output_dims();
  for (int64 i = output_dims.size() - 1; i >= 0; --i) {
    CHECK(simulated_index[output_dims[i]] == IndexComponent::Broadcasted);
    EraseAt(&simulated_index, output_dims[i]);
  }

  InsertAt(&simulated_index, scalar_indexed_const->source_dim(),
           IndexComponent::Broadcasted);

  // new_inner_broadcast_dims holds the broadcast dimensions for the inner
  // BinaryOp(Broadcast'(Const0), Const1).  We now translate simulated_index to
  // new_inner_broadcast_dims.
  std::vector<int64> new_inner_broadcast_dims;
  for (int64 i = 0; i < simulated_index.size(); i++) {
    if (simulated_index[i] == IndexComponent::NotBroadcasted) {
      new_inner_broadcast_dims.push_back(i);
    }
  }

  // inner_broadcast_result is the Broadcast'(Const0) bit in
  // BinaryOp(Broadcast'(Const0), Const1)
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<Literal> inner_broadcast_result,
      broadcast_const_operand->literal().Broadcast(
          scalar_indexed_const->source()->shape(), new_inner_broadcast_dims));

  // literal_for_new_source is BinaryOp(Broadcast'(Const0), Const1)
  const Literal* literal_for_new_source;
  if (lhs_is_indexed) {
    TF_ASSIGN_OR_RETURN(
        literal_for_new_source,
        TakeOwnership(HloEvaluator{}.EvaluateElementwiseBinaryOp(
            opcode, scalar_indexed_const->literal(), *inner_broadcast_result)));
  } else {
    TF_ASSIGN_OR_RETURN(
        literal_for_new_source,
        TakeOwnership(HloEvaluator{}.EvaluateElementwiseBinaryOp(
            opcode, *inner_broadcast_result, scalar_indexed_const->literal())));
  }

  ConstantArray* new_source = Construct<ConstantArray>(literal_for_new_source);
  return Construct<ScalarIndexedConstantArray>(
      new_source, scalar_indexed_const->indices(),
      scalar_indexed_const->source_dim(),
      std::vector<int64>(scalar_indexed_const->output_dims().begin(),
                         scalar_indexed_const->output_dims().end()),
      scalar_indexed_const->shape());
}

StatusOr<Analysis::Array*>
IndexedArrayAnalysis::ComputeArrayForElementwiseUnaryOp(HloOpcode opcode,
                                                        Array* operand) {
  auto* scalar_indexed_const =
      dynamic_cast<ScalarIndexedConstantArray*>(operand);
  if (scalar_indexed_const == nullptr) {
    return nullptr;
  }

  // Fold UnaryOp(ScalarIndexed(Const, Indices))
  //   => ScalarIndexed(UnaryOp(Const), Indices)

  TF_ASSIGN_OR_RETURN(Literal * literal_for_new_source,
                      TakeOwnership(HloEvaluator{}.EvaluateElementwiseUnaryOp(
                          opcode, scalar_indexed_const->literal())));
  ConstantArray* new_source = Construct<ConstantArray>(literal_for_new_source);
  return Construct<ScalarIndexedConstantArray>(
      new_source, scalar_indexed_const->indices(),
      scalar_indexed_const->source_dim(),
      std::vector<int64>(scalar_indexed_const->output_dims().begin(),
                         scalar_indexed_const->output_dims().end()),
      scalar_indexed_const->shape());
}

tensorflow::StringPiece IndexedArrayAnalysisPrinterPass::name() const {
  return "indexed-array-analysis-printer-pass";
}

StatusOr<bool> IndexedArrayAnalysisPrinterPass::Run(HloModule* module) {
  if (!VLOG_IS_ON(2)) {
    return false;
  }

  IndexedArrayAnalysis analysis;
  for (auto* computation : module->MakeNonfusionComputations()) {
    for (auto* instr : computation->instructions()) {
      TF_ASSIGN_OR_RETURN(Analysis::Array * t, analysis.GetArrayFor(instr));
      if (!dynamic_cast<UnknownArray*>(t) && !dynamic_cast<ConstantArray*>(t)) {
        VLOG(2) << instr->ToString() << "   ->   " << analysis.ToString(t);
      }
    }
  }

  return false;
}

}  // namespace xla
