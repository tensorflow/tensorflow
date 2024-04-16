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

#include "xla/service/triangular_solve_expander.h"

#include <memory>
#include <vector>

#include "absl/types/span.h"
#include "xla/client/lib/constants.h"
#include "xla/client/lib/math.h"
#include "xla/client/lib/matrix.h"
#include "xla/client/lib/slicing.h"
#include "xla/client/xla_builder.h"
#include "xla/client/xla_computation.h"
#include "xla/literal.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/statusor.h"
#include "xla/util.h"

namespace xla {

namespace {

// Get the diagonal blocks of the coefficient matrix
XlaOp DiagonalBlocks(XlaOp a, int64_t block_size) {
  XlaBuilder* builder = a.builder();
  return builder->ReportErrorOrReturn([&]() -> absl::StatusOr<XlaOp> {
    TF_ASSIGN_OR_RETURN(Shape shape, builder->GetShape(a));
    int ndims = shape.rank();
    int64_t n = ShapeUtil::GetDimension(shape, -1);
    int64_t num_blocks = n / block_size;
    absl::Span<int64_t const> batch_dims = absl::MakeConstSpan(
        shape.dimensions().begin(), shape.dimensions().begin() + (ndims - 2));

    XlaOp diag_blocks;

    // If the coefficient matrix is exactly the block size, we just add a
    // singleton dimension i.e. [..., n, n] -> [..., 1, n, n]
    if (n == block_size) {
      std::vector<int64_t> permutation(ndims);
      std::iota(permutation.begin(), permutation.end(), 1);
      permutation.insert(permutation.end() - 2, 0);
      return Transpose(Broadcast(a, /*broadcast_sizes=*/{1}), permutation);
    }

    // We can grab entire blocks using gather
    if (n > block_size) {
      // Construct the starting indices of the diagonal blocks
      auto start_indices =
          Transpose(Broadcast(Mul(Iota(builder, S32, num_blocks),
                                  ConstantR0<int32_t>(builder, block_size)),
                              /*broadcast_sizes=*/{2}),
                    /*permutation=*/{1, 0});

      // Gather the diagonal blocks
      std::vector<int64_t> slice_sizes(ndims);
      GatherDimensionNumbers dim_numbers;
      for (int i = 0; i < ndims - 2; ++i) {
        dim_numbers.add_offset_dims(i);
        slice_sizes[i] = ShapeUtil::GetDimension(shape, i);
      }
      slice_sizes[ndims - 2] = slice_sizes[ndims - 1] = block_size;
      dim_numbers.add_offset_dims(ndims - 1);
      dim_numbers.add_offset_dims(ndims);
      dim_numbers.add_start_index_map(ndims - 2);
      dim_numbers.add_start_index_map(ndims - 1);
      dim_numbers.set_index_vector_dim(1);
      diag_blocks = Gather(a, start_indices, dim_numbers, slice_sizes);
    }

    // The last block might be smaller than the block size,
    // so we will need to pad it
    if (n % block_size != 0) {
      // Pad with identity matrix.
      auto last_blocks =
          SliceInMinorDims(a, {n - n % block_size, n - n % block_size}, {n, n});
      PaddingConfig config = MakeNoPaddingConfig(ndims);
      int64_t padding = block_size - n % block_size;
      config.mutable_dimensions(ndims - 2)->set_edge_padding_high(padding);
      last_blocks =
          Pad(last_blocks, Zero(builder, shape.element_type()), config);

      auto eye =
          IdentityMatrix(builder, shape.element_type(), padding, padding);
      config = MakeNoPaddingConfig(2);
      config.mutable_dimensions(0)->set_edge_padding_low(n % block_size);
      eye = Pad(eye, Zero(builder, shape.element_type()), config);
      eye = Broadcast(eye, batch_dims);
      last_blocks = ConcatInDim(builder, {last_blocks, eye}, ndims - 1);

      // Add a singleton dimension
      // i.e. [..., block_size, block_size] -> [..., 1, block_size, block_size]
      TF_ASSIGN_OR_RETURN(Shape blocks_shape, builder->GetShape(last_blocks));
      auto shape_dims = blocks_shape.dimensions();
      auto last_blocks_dims = std::vector<int64_t>(ndims);
      std::copy(shape_dims.begin(), shape_dims.end(), last_blocks_dims.begin());
      last_blocks_dims.insert(last_blocks_dims.end() - 2, 1);
      last_blocks = Reshape(last_blocks, last_blocks_dims);

      // Concatenate with the other blocks if necessary
      if (n > block_size) {
        diag_blocks =
            ConcatInDim(builder, {diag_blocks, last_blocks}, ndims - 2);
      } else {
        diag_blocks = last_blocks;
      }
    }

    return diag_blocks;
  });
}

XlaOp SolveWithInvertedDiagonalBlocks(XlaOp a, XlaOp b, XlaOp inv_diag_blocks,
                                      bool left_side, bool lower,
                                      bool transpose_a, bool conjugate_a,
                                      PrecisionConfig::Precision precision) {
  XlaBuilder* builder = a.builder();
  return builder->ReportErrorOrReturn([&]() -> absl::StatusOr<XlaOp> {
    TF_ASSIGN_OR_RETURN(Shape blocks_shape, builder->GetShape(inv_diag_blocks));
    TF_ASSIGN_OR_RETURN(Shape b_shape, builder->GetShape(b));
    int64_t block_size = ShapeUtil::GetDimension(blocks_shape, -1);

    TF_ASSIGN_OR_RETURN(Shape a_shape, builder->GetShape(a));
    int64_t ndims = a_shape.rank();
    int64_t n = ShapeUtil::GetDimension(a_shape, -1);
    int64_t num_blocks = n / block_size + (n % block_size != 0);
    int64_t m_dim = (left_side) ? -1 : -2;
    int64_t m = ShapeUtil::GetDimension(b_shape, m_dim);

    std::vector<XlaOp> update_ops;
    int bdims = b_shape.rank();
    int64_t block_dim = (left_side) ? bdims - 2 : bdims - 1;

    // Initialize the solution
    XlaOp x;

    // This loop is unrolled for performance reasons, but it could be expressed
    // rolled as well since the matrices are of the same size each iteration
    for (int i = 0; i < num_blocks; i++) {
      // High-level intuition: We have B[i] = L[i] @ X. Since L is upper
      // triangular this means B[i] = L[i, :i + 1] @ X[:i + 1]. We can split
      // this into two parts: B[i] = L[i, :i] @ X[:i] + L[i, i] @ X[i] which
      // can be solved for X[i] as X[i] = inv(L[i, i]) @ B[i] - L[i, :i] @ X[:i]

      // Decide whether we go from first block to last or vice versa
      bool backward = left_side ^ lower ^ transpose_a;
      auto j = backward ? num_blocks - 1 - i : i;

      // Get the size of the inverse blocks (the last one might be smaller)
      int64_t block = (n % block_size != 0 && j + 1 == num_blocks)
                          ? n % block_size
                          : block_size;
      auto inv_block =
          MaybeConjugate(Collapse(SliceInMinorDims(inv_diag_blocks, {j, 0, 0},
                                                   {j + 1, block, block}),
                                  /*dimensions=*/{ndims - 2, ndims - 1}),
                         conjugate_a);

      // Get the corresponding row of B
      int64_t k = std::min((j + 1) * block_size, n);
      std::vector<int64_t> start = {j * block_size, 0};
      std::vector<int64_t> end = {k, m};
      if (!left_side) {
        std::swap(start[0], start[1]);
        std::swap(end[0], end[1]);
      }
      auto b_row = SliceInMinorDims(b, start, end);

      XlaOp remainder;
      if (i == 0) {
        remainder = b_row;
      } else {
        // This matrix multiply get rid of a lot of multiplying with zero
        // (namely, X[i * block_size:] = 0), L[i, :i] @ X[:i]
        if (backward) {
          start = {j * block_size,
                   std::max(int64_t{0}, (num_blocks - i) * block_size)};
          end = {k, n};
        } else {
          start = {j * block_size, 0};
          end = {k, std::min(i * block_size, n)};
        }

        if (!left_side ^ transpose_a) {
          std::swap(start[0], start[1]);
          std::swap(end[0], end[1]);
        }
        auto a_row =
            MaybeConjugate(SliceInMinorDims(a, start, end), conjugate_a);
        if (left_side) {
          remainder = b_row - BatchDot(a_row, transpose_a, x, false, precision);
        } else {
          remainder = b_row - BatchDot(x, false, a_row, transpose_a, precision);
        }
      }

      XlaOp x_update;
      if (left_side) {
        x_update =
            BatchDot(inv_block, transpose_a, remainder, false, precision);
      } else {
        x_update =
            BatchDot(remainder, false, inv_block, transpose_a, precision);
      }

      if (i == 0) {
        x = x_update;
      } else {
        if (backward) {
          x = ConcatInDim(builder, {x_update, x}, block_dim);
        } else {
          x = ConcatInDim(builder, {x, x_update}, block_dim);
        }
      }
    }

    return x;
  });
}

}  // namespace

XlaOp TriangularSolveExpander::InvertDiagonalBlocks(
    XlaOp diag_blocks, bool lower_triangular,
    PrecisionConfig::Precision precision) {
  XlaBuilder* builder = diag_blocks.builder();
  return builder->ReportErrorOrReturn([&]() -> absl::StatusOr<XlaOp> {
    // Input is a batch of square lower triangular square matrices. Its shape is
    // (..., size, size). We resize this to (num_blocks, size, size).
    TF_ASSIGN_OR_RETURN(Shape shape, builder->GetShape(diag_blocks));
    int64_t block_size = ShapeUtil::GetDimension(shape, -1);
    int64_t num_blocks = ShapeUtil::ElementsIn(shape) / IPow(block_size, 2);
    diag_blocks = Reshape(diag_blocks, {num_blocks, block_size, block_size});

    // The input must be triangular because we rely on that when doing
    // multiplications later on
    diag_blocks = Triangle(diag_blocks, /*lower=*/lower_triangular);

    // Rescale blocks to be unit triangular, but avoid dividing by
    // zero (which can happen if the last block was padded) otherwise it will
    // introduce nans which will propagate
    auto diags = GetMatrixDiagonal(diag_blocks);
    auto ones = FullLike(diags, 1);
    diags = Select(Eq(diags, Zero(builder, shape.element_type())), ones, diags);
    auto scaled_diag_blocks = Div(diag_blocks, diags, {0, 2});

    // We can now use the fact that for an upper triangular matrix
    // [[L11, 0], [L21, L22]], given the inverses L11' and L22', we have
    // L22' = -L22' * L21 * L11'. In our case, L21 is a vector and our blocks
    // have been rescaled to be unit triangular, so L22 = L22' = 1.

    // Initialize the output matrix with -1s on the diagonal. We use -1 instead
    // of 1 because we cannot do matrix-vector multiplies with variable shapes
    // inside of a loop, or do irregularly shaped in-place updates. Hence,
    // L21 <- -L22 * L21 * L11 cannot be done naively. Instead, we update the
    // entire row i.e. we calculate
    // [L21 L22 0] <- -[L21 L22 0] @ diag_blocks([L11', -I, -I])
    // which means [L21 L22 0] <- [-L21 * L11', L22, 0].
    auto identity =
        IdentityMatrix(builder, shape.element_type(), block_size, block_size);
    auto neg_identity = -identity;

    // The first or last  diagonal element should be set to 1 instead of -1
    // though, since we never update it
    auto pos_one = Reshape(One(builder, shape.element_type()), {1, 1});
    auto start_index =
        ConstantR0<int>(builder, lower_triangular ? 0 : block_size - 1);
    auto output_block =
        DynamicUpdateSlice(neg_identity, pos_one,
                           /*start_indices=*/{start_index, start_index});

    // Broadcast diag([1, -1, -1, ...]) to every block
    XlaOp output = Broadcast(output_block,
                             /*broadcast_sizes=*/{num_blocks});

    // Now we construct a loop that performs matrix-vector multiplications
    // inverting the blocks one row at a time
    std::vector<Shape> tuple_shapes = {
        // The loop iteration counter is a scalar, incremented each iteration.
        ShapeUtil::MakeShape(S32, {}),
        // The output has the shape of A, with one row updated each iteration.
        ShapeUtil::MakeShape(shape.element_type(),
                             {num_blocks, block_size, block_size}),
        // The input is a loop invariant.
        ShapeUtil::MakeShape(shape.element_type(),
                             {num_blocks, block_size, block_size})};
    Shape tuple_shape = ShapeUtil::MakeTupleShape(tuple_shapes);

    auto init_i = One(builder, S32);
    auto init = Tuple(builder, {init_i, output, scaled_diag_blocks});

    // Construct the loop condition function.
    std::unique_ptr<XlaBuilder> condb =
        builder->CreateSubBuilder("InvertDiagCond");
    {
      auto i = GetTupleElement(
          Parameter(condb.get(), 0, tuple_shape, "InvertDiagCondTuple"), 0);
      Lt(i, ConstantR0<int32_t>(condb.get(), block_size));
    }
    TF_ASSIGN_OR_RETURN(auto cond, condb->Build());

    // Construct the loop body function.
    std::unique_ptr<XlaBuilder> bodyb =
        builder->CreateSubBuilder("InvertDiagBody");
    {
      auto input_tuple =
          Parameter(bodyb.get(), 0, tuple_shape, "InvertDiagBodyTuple");

      auto i = GetTupleElement(input_tuple, 0);
      auto body_out = GetTupleElement(input_tuple, 1);
      auto body_input = GetTupleElement(input_tuple, 2);

      auto zero = ConstantR0<int32_t>(bodyb.get(), 0);
      auto j = lower_triangular ? i : ScalarLike(i, block_size - 1) - i;
      auto input_row =
          DynamicSlice(body_input, {zero, j, zero},
                       /*slice_sizes=*/{num_blocks, 1, block_size});

      // We want -L21 L11^{-1}
      DotDimensionNumbers dnums;
      dnums.add_lhs_batch_dimensions(0);
      dnums.add_rhs_batch_dimensions(0);
      dnums.add_lhs_contracting_dimensions(2);
      dnums.add_rhs_contracting_dimensions(1);
      PrecisionConfig precision_proto;
      precision_proto.add_operand_precision(precision);
      precision_proto.add_operand_precision(precision);
      auto update = -DotGeneral(input_row, body_out, dnums, &precision_proto);

      body_out = DynamicUpdateSlice(body_out, update, {zero, j, zero});

      auto next_i = i + ScalarLike(i, 1);
      Tuple(bodyb.get(), {next_i, body_out, body_input});
    }
    TF_ASSIGN_OR_RETURN(auto body, bodyb->Build());

    // Construct the While loop and return the result,
    // return while_loop(cond_fun, body_fun, init)[1]
    auto invert_while = While(cond, body, init);
    auto inv_diag_blocks = GetTupleElement(invert_while, 1);
    // Undo the scaling
    inv_diag_blocks = Div(inv_diag_blocks, diags,
                          /*broadcast_dimensions=*/{0, 1});

    // Reshape back to original batch major dimensions
    return Reshape(inv_diag_blocks, shape.dimensions());
  });
}

XlaOp TriangularSolveExpander::SolveByInvertingDiagonalBlocks(
    XlaOp a, XlaOp b, bool left_side, bool lower, bool transpose_a,
    bool conjugate_a, bool unit_diagonal,
    PrecisionConfig::Precision precision) {
  XlaBuilder* builder = a.builder();
  return builder->ReportErrorOrReturn([&]() -> absl::StatusOr<XlaOp> {
    TF_ASSIGN_OR_RETURN(Shape a_shape, builder->GetShape(a));
    const int64_t ndims = a_shape.rank();
    int64_t k = ShapeUtil::GetDimension(a_shape, -1);

    // TODO(phawkins): consider pushing triangle masking into
    // InvertDiagonalBlocks.
    if (unit_diagonal) {
      // Mask everything but the subdiagonal/superdiagonal elements.
      a = lower ? Select(TriangleMask(a, -1), a, ZerosLike(a))
                : Select(TriangleMask(a, 0), ZerosLike(a), a);
      a = xla::Add(a, IdentityMatrix(builder, a_shape.element_type(), k, k),
                   /*broadcast_dimensions=*/{ndims - 2, ndims - 1});
    } else {
      // Mask off the ignored elements of the triangular matrix a.
      a = Triangle(a, lower);
    }

    // We find the diagonal blocks of the coefficient matrix
    int64_t block_size = std::min(block_size_, k);
    auto diag_blocks = DiagonalBlocks(a, block_size);

    // We invert these blocks in parallel using batched matrix-vector products
    auto inv_diag_blocks = InvertDiagonalBlocks(diag_blocks, lower, precision);

    // We now find the solution using GEMMs
    return SolveWithInvertedDiagonalBlocks(a, b, inv_diag_blocks, left_side,
                                           lower, transpose_a, conjugate_a,
                                           precision);
  });
}

// def trsm_left_lower_leftlooking(a, b):
//   n = a.shape[-1]
//   assert a.shape == (n, n)
//   b = b.copy()
//   for j in range(n):
//     b[j, :] = (b[j, :] - np.dot(a[j, :j], b[:j, :])) / a[j, j]
//   return b
XlaOp TriangularSolveExpander::SolveDirectly(
    XlaOp a, XlaOp b, bool left_side, bool lower, bool transpose_a,
    bool conjugate_a, bool unit_diagonal,
    PrecisionConfig::Precision precision) {
  XlaBuilder* builder = a.builder();
  return builder->ReportErrorOrReturn([&]() -> absl::StatusOr<XlaOp> {
    TF_ASSIGN_OR_RETURN(Shape a_shape, builder->GetShape(a));
    TF_ASSIGN_OR_RETURN(Shape b_shape, builder->GetShape(b));
    int64_t m = ShapeUtil::GetDimension(b_shape, -2);
    int64_t n = ShapeUtil::GetDimension(b_shape, -1);
    const int64_t a_size = ShapeUtil::GetDimension(a_shape, -1);
    a = MaybeConjugate(a, conjugate_a);
    bool backwards = transpose_a ^ lower ^ !left_side;
    for (int64_t i = 0; i < a_size; ++i) {
      int64_t j = backwards ? i : (a_size - i - 1);
      std::vector<int64_t> b_row_start, b_row_end;
      if (left_side) {
        b_row_start = {j, 0};
        b_row_end = {j + 1, n};
      } else {
        b_row_start = {0, j};
        b_row_end = {m, j + 1};
      }
      auto b_row = SliceInMinorDims(b, b_row_start, b_row_end);

      std::vector<int64_t> a_start = {j, backwards ? 0 : (j + 1)};
      std::vector<int64_t> a_end = {j + 1, backwards ? j : a_size};
      if (transpose_a ^ !left_side) {
        std::swap(a_start[0], a_start[1]);
        std::swap(a_end[0], a_end[1]);
      }
      auto a_chunk = SliceInMinorDims(a, a_start, a_end);
      if (left_side) {
        bool which = transpose_a ^ lower;
        auto b_chunk =
            SliceInMinorDims(b, {which ? 0 : (j + 1), 0}, {which ? j : m, n});
        b_row = b_row - BatchDot(a_chunk, /*transpose_x=*/transpose_a, b_chunk,
                                 /*transpose_y=*/false, precision);
      } else {
        bool which = transpose_a ^ !lower;
        auto b_chunk =
            SliceInMinorDims(b, {0, which ? 0 : (j + 1)}, {m, which ? j : n});
        b_row = b_row - BatchDot(b_chunk, /*transpose_x=*/false, a_chunk,
                                 /*transpose_y=*/transpose_a, precision);
      }
      if (!unit_diagonal) {
        auto a_diag = SliceInMinorDims(a, {j, j}, {j + 1, j + 1});
        b_row = b_row / a_diag;
      }

      b = UpdateSliceInMinorDims(b, b_row, b_row_start);
    }

    return b;
  });
}

XlaOp TriangularSolveExpander::BuildTriangularSolve(
    XlaOp a, XlaOp b, bool left_side, bool lower, bool transpose_a,
    bool conjugate_a, bool unit_diagonal, int64_t block_size,
    PrecisionConfig::Precision precision) {
  XlaBuilder* builder = a.builder();
  return builder->ReportErrorOrReturn([&]() -> absl::StatusOr<XlaOp> {
    TF_ASSIGN_OR_RETURN(Shape a_shape, builder->GetShape(a));
    TF_ASSIGN_OR_RETURN(Shape b_shape, builder->GetShape(b));
    if (a_shape.rank() != b_shape.rank()) {
      return InvalidArgument(
          "Arguments to TriangularSolve have shapes with different ranks: "
          "%s vs. %s",
          ShapeUtil::HumanString(a_shape), ShapeUtil::HumanString(b_shape));
    }
    const int64_t ndims = a_shape.rank();
    if (ndims < 2) {
      return InvalidArgument(
          "Arguments to TriangularSolve was rank %d but must have rank >= 2.",
          ndims);
    }
    // The batch dimensions must be equal.
    std::vector<int64_t> batch_dimensions;
    int64_t batch = 1;
    for (int i = 0; i < ndims - 2; ++i) {
      int64_t a_size = a_shape.dimensions(i);
      int64_t b_size = b_shape.dimensions(i);
      if (a_size != b_size) {
        return InvalidArgument(
            "Batch dimensions of arguments to TriangularSolve must be equal; "
            "shapes were %s and %s.",
            ShapeUtil::HumanString(a_shape), ShapeUtil::HumanString(b_shape));
      }
      batch_dimensions.push_back(a_size);
      batch *= a_size;
    }

    if (ShapeUtil::GetDimension(a_shape, -1) !=
        ShapeUtil::GetDimension(a_shape, -2)) {
      return InvalidArgument(
          "The 'a' argument to TriangularSolve must be a batched square matrix;"
          " shape was: %s",
          ShapeUtil::HumanString(a_shape));
    }
    const int64_t m = ShapeUtil::GetDimension(b_shape, -2);
    const int64_t n = ShapeUtil::GetDimension(b_shape, -1);
    if ((left_side ? m : n) != ShapeUtil::GetDimension(a_shape, -1)) {
      return InvalidArgument(
          "Arguments to TriangularSolve have incompatible matrix shapes %s and "
          "%s",
          ShapeUtil::HumanString(a_shape), ShapeUtil::HumanString(b_shape));
    }

    int64_t a_size = ShapeUtil::GetDimension(a_shape, -1);

    if (ShapeUtil::IsZeroElementArray(b_shape)) {
      // The output has the same shape as 'b', and since the output has zero
      // elements, any such array will do.
      return b;
    }

    // Degenerate case: 1x1 matrices.
    if (a_size == 1) {
      return unit_diagonal ? b : Div(b, MaybeConjugate(a, conjugate_a));
    }

    // Prefer the direct implementation whenever there is a nontrivial batch
    // dimension and the matrix is very small.
    if (UseDirectSolves() && batch > block_size_ / 16 &&
        a_size < block_size_ / 4) {
      return SolveDirectly(a, b, left_side, lower, transpose_a, conjugate_a,
                           unit_diagonal, precision);
    } else {
      return SolveByInvertingDiagonalBlocks(a, b, left_side, lower, transpose_a,
                                            conjugate_a, unit_diagonal,
                                            precision);
    }
  });
}

TriangularSolveExpander::TriangularSolveExpander(int64_t block_size)
    : block_size_(block_size) {
  CHECK_GE(block_size_, 1);
}

bool TriangularSolveExpander::InstructionMatchesPattern(
    HloInstruction* instruction) {
  return instruction->opcode() == HloOpcode::kTriangularSolve;
}

absl::StatusOr<HloInstruction*> TriangularSolveExpander::ExpandInstruction(
    HloInstruction* instruction) {
  const TriangularSolveOptions& options =
      instruction->triangular_solve_options();
  const std::string name = absl::StrFormat(
      "xla.triangular_solve_%s_%s_%s_%s_%s_%s",
      instruction->operand(0)->shape().ToString(),
      instruction->operand(1)->shape().ToString(),
      options.left_side() ? "left" : "right",
      options.lower() ? "lower" : "upper",
      TriangularSolveOptions_Transpose_Name(options.transpose_a()),
      options.unit_diagonal() ? "unit" : "nonunit");

  HloModule* module = instruction->GetModule();

  HloComputation*& computation =
      computation_cache_.emplace(name, nullptr).first->second;
  if (!computation) {
    // Builds a new expansion.
    //
    // We do something unusual here: we build the computation using the
    // XlaBuilder API, which is nominally an XLA client API. We do this because
    // the external APIs for building complicated computations (XlaBuilder)
    // are much more ergonomic than the internal ones. As it turns out,
    // XlaBuilder isn't really a client API—what it does is build a
    // HloModuleProto protocol buffer, that we can then deserialize and clone
    // into our HloModule. Ideally we would avoid the protocol buffer step;
    // that is left as an exercise for future work.
    XlaBuilder builder(name);
    XlaOp a = Parameter(&builder, 0, instruction->operand(0)->shape(), "a");
    XlaOp b = Parameter(&builder, 1, instruction->operand(1)->shape(), "b");
    bool transpose_a =
        options.transpose_a() != TriangularSolveOptions::NO_TRANSPOSE;
    bool conjugate_a = options.transpose_a() == TriangularSolveOptions::ADJOINT;

    BuildTriangularSolve(a, b, options.left_side(), options.lower(),
                         transpose_a, conjugate_a, options.unit_diagonal(),
                         /*block_size=*/block_size_,
                         /*precision=*/PrecisionConfig::HIGHEST);
    TF_ASSIGN_OR_RETURN(XlaComputation xla_computation, builder.Build());

    TF_ASSIGN_OR_RETURN(ProgramShape program_shape,
                        xla_computation.GetProgramShape());
    HloModuleConfig config(program_shape);
    TF_ASSIGN_OR_RETURN(auto new_module, HloModule::CreateFromProto(
                                             xla_computation.proto(), config));
    HloCloneContext context(module);
    computation =
        module->DeepCloneComputation(new_module->entry_computation(), &context);
  }

  return instruction->parent()->AddInstruction(HloInstruction::CreateCall(
      instruction->shape(), instruction->operands(), computation));
}

}  // namespace xla
