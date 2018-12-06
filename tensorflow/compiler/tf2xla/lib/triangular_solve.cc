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

#include "tensorflow/compiler/tf2xla/lib/triangular_solve.h"

#include <memory>
#include <vector>

#include "tensorflow/compiler/tf2xla/lib/util.h"
#include "tensorflow/compiler/xla/client/lib/constants.h"
#include "tensorflow/compiler/xla/client/lib/math.h"
#include "tensorflow/compiler/xla/client/lib/matrix.h"
#include "tensorflow/compiler/xla/client/lib/slicing.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/client/xla_computation.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/math/math_util.h"

namespace tensorflow {

// Get the diagonal blocks of the coefficient matrix
xla::XlaOp DiagonalBlocks(xla::XlaOp a, int64 block_size) {
  xla::XlaBuilder* builder = a.builder();
  return builder->ReportErrorOrReturn([&]() -> xla::StatusOr<xla::XlaOp> {
    TF_ASSIGN_OR_RETURN(xla::Shape shape, builder->GetShape(a));
    int ndims = xla::ShapeUtil::Rank(shape);
    int64 n = xla::ShapeUtil::GetDimension(shape, -1);
    int64 num_blocks = n / block_size;

    xla::XlaOp diag_blocks;

    // If the coefficient matrix is exactly the block size, we just add a
    // singleton dimension i.e. [..., n, n] -> [..., 1, n, n]
    if (n == block_size) {
      std::vector<int64> permutation(ndims);
      std::iota(permutation.begin(), permutation.end(), 1);
      permutation.insert(permutation.end() - 2, 0);
      return Transpose(Broadcast(a, /*broadcast_sizes=*/{1}), permutation);
    }

    // We can grab entire blocks using gather
    if (n > block_size) {
      // Construct the starting indices of the diagonal blocks
      auto start_indices =
          Transpose(Broadcast(Mul(Iota(builder, xla::S32, num_blocks),
                                  xla::ConstantR0<int32>(builder, block_size)),
                              /*broadcast_sizes=*/{2}),
                    /*permutation=*/{1, 0});

      // Gather the diagonal blocks
      xla::GatherDimensionNumbers dim_numbers;
      dim_numbers.add_offset_dims(ndims - 1);
      dim_numbers.add_offset_dims(ndims);
      dim_numbers.add_start_index_map(ndims - 2);
      dim_numbers.add_start_index_map(ndims - 1);
      dim_numbers.set_index_vector_dim(1);
      diag_blocks = Gather(a, start_indices, dim_numbers,
                           /*slice_sizes=*/{block_size, block_size});
    }

    // The last block might be smaller than the block size,
    // so we will need to pad it
    if (n % block_size != 0) {
      // Pad with zeros
      auto last_blocks =
          SliceInMinorDims(a, {n - n % block_size, n - n % block_size}, {n, n});
      xla::PaddingConfig config = xla::MakeNoPaddingConfig(ndims);
      int64 padding = block_size - n % block_size;
      config.mutable_dimensions(ndims - 1)->set_edge_padding_high(padding);
      config.mutable_dimensions(ndims - 2)->set_edge_padding_high(padding);
      last_blocks =
          Pad(last_blocks, Zero(builder, shape.element_type()), config);

      // Add a singleton dimension
      // i.e. [..., block_size, block_size] -> [..., 1, block_size, block_size]
      TF_ASSIGN_OR_RETURN(xla::Shape blocks_shape,
                          builder->GetShape(last_blocks));
      auto shape_dims = xla::AsInt64Slice(blocks_shape.dimensions());
      auto last_blocks_dims = std::vector<int64>(ndims);
      std::copy(shape_dims.begin(), shape_dims.end(), last_blocks_dims.begin());
      last_blocks_dims.insert(last_blocks_dims.end() - 2, 1);
      last_blocks = Reshape(last_blocks, last_blocks_dims);

      // Concatenate with the other blocks if necessary
      if (n > block_size) {
        diag_blocks =
            xla::ConcatInDim(builder, {diag_blocks, last_blocks}, ndims - 2);
      } else {
        diag_blocks = last_blocks;
      }
    }

    return diag_blocks;
  });
}

xla::XlaOp InvertDiagonalBlocks(xla::XlaOp diag_blocks, bool lower,
                                bool transpose_a, bool conjugate_a,
                                xla::PrecisionConfig::Precision precision) {
  xla::XlaBuilder* builder = diag_blocks.builder();
  return builder->ReportErrorOrReturn([&]() -> xla::StatusOr<xla::XlaOp> {
    // Input is a batch of square lower triangular square matrices. Its shape is
    // (..., size, size). We resize this to (num_blocks, size, size).
    TF_ASSIGN_OR_RETURN(xla::Shape shape, builder->GetShape(diag_blocks));
    int64 block_size = xla::ShapeUtil::GetDimension(shape, -1);
    int64 num_blocks = xla::ShapeUtil::ElementsIn(shape) /
                       tensorflow::MathUtil::IPow(block_size, 2);
    diag_blocks = Reshape(diag_blocks, {num_blocks, block_size, block_size});

    // The input must be triangular because we rely on that when doing
    // multiplications later on
    diag_blocks = Triangle(diag_blocks, /*lower=*/lower);

    // Rescale blocks to be unit triangular, but avoid dividing by
    // zero (which can happen if the last block was padded) otherwise it will
    // introduce nans which will propagate
    auto diags = GetMatrixDiagonal(diag_blocks);
    TF_ASSIGN_OR_RETURN(xla::Shape diags_shape, builder->GetShape(diags));
    auto one = ScalarLike(diags, 1);
    auto ones = Broadcast(one, xla::AsInt64Slice(diags_shape.dimensions()));
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
    auto start_index = (lower) ? 0 : block_size - 1;
    auto output_block = DynamicUpdateSlice(
        neg_identity, pos_one,
        /*start_indices=*/xla::ConstantR1<int>(builder, 2, start_index));

    // Broadcast diag([1, -1, -1, ...]) to every block
    xla::XlaOp output = Broadcast(output_block,
                                  /*broadcast_sizes=*/{num_blocks});

    // Now we construct a loop that performs matrix-vector multiplications
    // inverting the blocks one row at a time
    std::vector<xla::Shape> tuple_shapes = {
        // The loop iteration counter is a scalar, incremented each iteration.
        xla::ShapeUtil::MakeShape(xla::S32, {}),
        // The output has the shape of A, with one row updated each iteration.
        xla::ShapeUtil::MakeShape(shape.element_type(),
                                  {num_blocks, block_size, block_size}),
        // The input is a loop invariant.
        xla::ShapeUtil::MakeShape(shape.element_type(),
                                  {num_blocks, block_size, block_size})};
    xla::Shape tuple_shape = xla::ShapeUtil::MakeTupleShape(tuple_shapes);

    auto init_i = One(builder, xla::S32);
    auto init = xla::Tuple(builder, {init_i, output, scaled_diag_blocks});

    // Construct the loop condition function.
    std::unique_ptr<xla::XlaBuilder> condb =
        builder->CreateSubBuilder("InvertDiagCond");
    {
      auto i = GetTupleElement(
          Parameter(condb.get(), 0, tuple_shape, "InvertDiagCondTuple"), 0);
      Lt(i, xla::ConstantR0<int32>(condb.get(), block_size));
    }
    TF_ASSIGN_OR_RETURN(auto cond, condb->Build());

    // Construct the loop body function.
    std::unique_ptr<xla::XlaBuilder> bodyb =
        builder->CreateSubBuilder("InvertDiagBody");
    {
      auto input_tuple =
          Parameter(bodyb.get(), 0, tuple_shape, "InvertDiagBodyTuple");

      auto i = GetTupleElement(input_tuple, 0);
      auto body_out = GetTupleElement(input_tuple, 1);
      auto body_input = GetTupleElement(input_tuple, 2);

      auto zero = xla::ConstantR1<int32>(bodyb.get(), 1, 0);
      auto j = (lower) ? i : ScalarLike(i, block_size - 1) - i;
      auto start_indices =
          xla::ConcatInDim(bodyb.get(), {zero, Reshape(j, {1}), zero}, 0);
      auto input_row =
          DynamicSlice(body_input, start_indices,
                       /*slice_sizes=*/{num_blocks, 1, block_size});

      // We want -L21 L11^{-1}
      xla::DotDimensionNumbers dnums;
      dnums.add_lhs_batch_dimensions(0);
      dnums.add_rhs_batch_dimensions(0);
      dnums.add_lhs_contracting_dimensions(2);
      dnums.add_rhs_contracting_dimensions(1);
      xla::PrecisionConfig precision_proto;
      precision_proto.add_operand_precision(precision);
      precision_proto.add_operand_precision(precision);
      auto update = -DotGeneral(input_row, body_out, dnums, &precision_proto);

      body_out = DynamicUpdateSlice(body_out, update, start_indices);

      auto next_i = i + ScalarLike(i, 1);
      xla::Tuple(bodyb.get(), {next_i, body_out, body_input});
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
    return Reshape(inv_diag_blocks, xla::AsInt64Slice(shape.dimensions()));
  });
}

xla::XlaOp SolveWithInvertedDiagonalBlocks(
    xla::XlaOp a, xla::XlaOp b, xla::XlaOp inv_diag_blocks, bool left_side,
    bool lower, bool transpose_a, bool conjugate_a,
    xla::PrecisionConfig::Precision precision) {
  xla::XlaBuilder* builder = a.builder();
  return builder->ReportErrorOrReturn([&]() -> xla::StatusOr<xla::XlaOp> {
    TF_ASSIGN_OR_RETURN(xla::Shape blocks_shape,
                        builder->GetShape(inv_diag_blocks));
    TF_ASSIGN_OR_RETURN(xla::Shape b_shape, builder->GetShape(b));
    int64 block_size = xla::ShapeUtil::GetDimension(blocks_shape, -1);

    TF_ASSIGN_OR_RETURN(xla::Shape a_shape, builder->GetShape(a));
    int64 ndims = xla::ShapeUtil::Rank(a_shape);
    int64 n = xla::ShapeUtil::GetDimension(a_shape, -1);
    int64 num_blocks = n / block_size + (n % block_size != 0);
    int64 m_dim = (left_side) ? -1 : -2;
    int64 m = xla::ShapeUtil::GetDimension(b_shape, m_dim);

    // Initialize the solution
    auto x = ZerosLike(b);

    // This loop is unrolled for performance reasons, but it could be expressed
    // rolled as well since the matrices are of the same size each iteration
    for (int i = 0; i < num_blocks; i++) {
      // High-level intuition: We have B[i] = L[i] @ X. Since L is upper
      // triangular this means B[i] = L[i, :i + 1] @ X[:i + 1]. We can split
      // this into two parts: B[i] = L[i, :i] @ X[:i] + L[i, i] @ X[i] which
      // can be solved for X[i] as X[i] = inv(L[i, i]) @ B[i] - L[i, :i] @ X[:i]

      // Decide whether we go from first block to last or vice versa
      auto j = (left_side ^ lower ^ transpose_a) ? num_blocks - 1 - i : i;

      // Get the size of the inverse blocks (the last one might be smaller)
      int64 block = (n % block_size != 0 && j + 1 == num_blocks)
                        ? n % block_size
                        : block_size;
      auto inv_block =
          MaybeConjugate(Collapse(SliceInMinorDims(inv_diag_blocks, {j, 0, 0},
                                                   {j + 1, block, block}),
                                  /*dimensions=*/{ndims - 2, ndims - 1}),
                         conjugate_a);

      // Get the corresponding row of B
      int64 k = std::min((j + 1) * block_size, n);
      std::vector<int64> start = {j * block_size, 0};
      std::vector<int64> end = {k, m};
      if (!left_side) {
        std::swap(start[0], start[1]);
        std::swap(end[0], end[1]);
      }
      auto b_row = SliceInMinorDims(b, start, end);

      xla::XlaOp remainder;
      if (i == 0) {
        remainder = b_row;
      } else {
        // This matrix multiply involves a lot of multiplying with zero (namely,
        // X[i * block_size:] = 0), but this is faster than slicing...
        end = {k, n};
        if (!left_side) {
          std::swap(end[0], end[1]);
        }
        if (transpose_a) {
          std::swap(start[0], start[1]);
          std::swap(end[0], end[1]);
        }
        auto a_row =
            MaybeConjugate(SliceInMinorDims(a, start, end), conjugate_a);
        if (left_side) {
          remainder =
              b_row - BatchDot(MaybeTransposeInMinorDims(a_row, transpose_a), x,
                               precision);
        } else {
          remainder =
              b_row - BatchDot(x, MaybeTransposeInMinorDims(a_row, transpose_a),
                               precision);
        }
      }

      xla::XlaOp x_update;
      auto zero = Zero(builder, xla::S32);
      auto start_index =
          xla::ConstantR0WithType(builder, xla::S32, j * block_size);
      std::vector<xla::XlaOp> update_starts = {start_index, zero};
      if (left_side) {
        x_update = BatchDot(MaybeTransposeInMinorDims(inv_block, transpose_a),
                            remainder, precision);
      } else {
        x_update = BatchDot(remainder,
                            MaybeTransposeInMinorDims(inv_block, transpose_a),
                            precision);
        std::swap(update_starts[0], update_starts[1]);
      }
      x = DynamicUpdateSliceInMinorDims(x, x_update, /*starts=*/update_starts);
    }

    return x;
  });
}

xla::XlaOp TriangularSolve(xla::XlaOp a, xla::XlaOp b, bool left_side,
                           bool lower, bool transpose_a, bool conjugate_a,
                           int64 block_size,
                           xla::PrecisionConfig::Precision precision) {
  xla::XlaBuilder* builder = a.builder();
  return builder->ReportErrorOrReturn([&]() -> xla::StatusOr<xla::XlaOp> {
    TF_ASSIGN_OR_RETURN(xla::Shape a_shape, builder->GetShape(a));
    TF_ASSIGN_OR_RETURN(xla::Shape b_shape, builder->GetShape(b));
    if (xla::ShapeUtil::Rank(a_shape) != xla::ShapeUtil::Rank(b_shape)) {
      return errors::InvalidArgument(
          "Arguments to TriangularSolve have different ranks: ",
          xla::ShapeUtil::HumanString(a_shape), " vs. ",
          xla::ShapeUtil::HumanString(b_shape));
    }
    const int64 ndims = xla::ShapeUtil::Rank(a_shape);
    if (ndims < 2) {
      return errors::InvalidArgument(
          "Arguments to TriangularSolve must have rank >= 2: ", ndims);
    }
    // The batch dimensions must be equal.
    std::vector<int64> batch_dimensions;
    for (int i = 0; i < ndims - 2; ++i) {
      int64 a_size = a_shape.dimensions(i);
      int64 b_size = b_shape.dimensions(i);
      if (a_size != b_size) {
        return errors::InvalidArgument(
            "Batch dimensions of arguments to TriangularSolve must be equal: ",
            xla::ShapeUtil::HumanString(a_shape), " vs ",
            xla::ShapeUtil::HumanString(b_shape));
      }
      batch_dimensions.push_back(a_size);
    }

    if (xla::ShapeUtil::GetDimension(a_shape, -1) !=
        xla::ShapeUtil::GetDimension(a_shape, -2)) {
      return errors::InvalidArgument(
          "The 'a' arguments to TriangularSolve must be square matrices: ",
          xla::ShapeUtil::HumanString(a_shape));
    }
    const int64 m = xla::ShapeUtil::GetDimension(b_shape, -2);
    const int64 n = xla::ShapeUtil::GetDimension(b_shape, -1);
    if ((left_side ? m : n) != xla::ShapeUtil::GetDimension(a_shape, -1)) {
      return errors::InvalidArgument(
          "Arguments to TriangularSolve have incompatible matrix shapes: ",
          xla::ShapeUtil::HumanString(a_shape), " vs ",
          xla::ShapeUtil::HumanString(b_shape));
    }

    if (block_size < 1) {
      return errors::InvalidArgument(
          "block_size argument to TriangularSolve must be >= 1; got ",
          block_size);
    }

    // We find the diagonal blocks of the coefficient matrix
    auto diag_blocks = DiagonalBlocks(a, block_size);

    // We invert these blocks in parallel using batched matrix-vector products
    auto inv_diag_blocks = InvertDiagonalBlocks(diag_blocks, lower, transpose_a,
                                                conjugate_a, precision);

    // We now find the solution using GEMMs
    auto x =
        SolveWithInvertedDiagonalBlocks(a, b, inv_diag_blocks, left_side, lower,
                                        transpose_a, conjugate_a, precision);

    return x;
  });
}

}  // namespace tensorflow
