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

#include "tensorflow/compiler/tf2xla/lib/batch_dot.h"
#include "tensorflow/compiler/tf2xla/lib/util.h"
#include "tensorflow/compiler/xla/client/lib/constants.h"
#include "tensorflow/compiler/xla/client/lib/numeric.h"
#include "tensorflow/compiler/xla/client/xla_client/xla_builder.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/lib/core/errors.h"

namespace tensorflow {

xla::XlaOp TriangularSolve(xla::XlaOp a, xla::XlaOp b, bool left_side,
                           bool lower, bool transpose_a, bool conjugate_a,
                           int64 block_size) {
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
    const int ndims = xla::ShapeUtil::Rank(a_shape);
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

    std::map<int, xla::XlaComputation> base_computations;
    auto get_base_triangular_solve =
        [&](int k) -> xla::StatusOr<xla::XlaComputation*> {
      xla::XlaComputation& computation = base_computations[k];
      if (computation.IsNull()) {
        std::unique_ptr<xla::XlaBuilder> sub = builder->CreateSubBuilder(
            tensorflow::strings::StrCat("trsm_base_", k));

        auto a_param = xla::Parameter(
            sub.get(), 0,
            xla::ShapeUtil::MakeShape(b_shape.element_type(),
                                      ConcatVectors(batch_dimensions, {k, k})),
            "a");

        std::array<int64, 2> b_lastd;
        if (left_side) {
          b_lastd = {k, n};
        } else {
          b_lastd = {m, k};
        }
        auto b_param = xla::Parameter(
            sub.get(), 1,
            xla::ShapeUtil::MakeShape(b_shape.element_type(),
                                      ConcatVectors(batch_dimensions, b_lastd)),
            "b");

        // We use a left-looking or right-looking subroutine on the block
        // diagonal in the lower=true cases, while falling back to a recursive
        // call in others. The left-looking and right-looking subroutines are
        // written with a While loop and so yields much faster compile times.
        // Moreover, they can give higher performance on smaller (sub)problems.
        if (left_side && lower) {
          TriangularSolveLeftLooking(a_param, b_param, transpose_a,
                                     conjugate_a);
        } else if (!left_side && lower) {
          TriangularSolveRightLooking(a_param, b_param, transpose_a,
                                      conjugate_a);
        } else {
          TriangularSolve(a_param, b_param, left_side, lower, transpose_a,
                          conjugate_a,
                          /*block_size=*/1);
        }

        TF_ASSIGN_OR_RETURN(computation, sub->Build());
      }
      return &computation;
    };

    xla::XlaOp output = xla::ZerosLike(b);

    // Right-looking blocked triangular solve.
    // For an explanation of the algorithm, see the TRSM discussion in:
    // Goto, Kazushige, and Robert Van De Geijn. "High-performance
    // implementation of the level-3 BLAS." ACM Transactions on Mathematical
    // Software (TOMS) 35.1 (2008): 4.

    // In the code comments below, T = lambda x: np.swapaxes(x, -1, -2) if
    // conjugate_a is False, or T = lambda x: np.conj(np.swapaxes(x, -1, -2)) if
    // conjugate_a is True.

    if (!left_side && lower == transpose_a) {
      // for i in range(0, a.shape[-1], block_size):
      for (int64 i = 0; i < n; i += block_size) {
        int64 k = std::min(block_size, n - i);

        // output[..., :, i:i+k] = triangular_solve(
        //     a[..., i:i+k, i:i+k],
        //     b[..., :, i:i+k] - np.matmul(output[..., :, :i],
        //                                  a[..., :i, i:i+k]),
        //     ..., block_size=1)
        auto a_slice = SliceInMinorDims(a, {i, i}, {i + k, i + k});
        auto b_slice = SliceInMinorDims(b, {0, i}, {m, i + k});

        // Note that we multiply with the full output, since this is faster
        // than slicing, and output[..., :, i:] = 0
        xla::XlaOp a_prev;
        if (lower) {
          a_prev = SliceInMinorDims(a, {i, 0}, {i + k, n});
        } else {
          a_prev = SliceInMinorDims(a, {0, i}, {n, i + k});
        }
        auto prev_contribution = BatchDot(output, a_prev,
                                          /*transpose_x=*/false,
                                          /*transpose_y=*/transpose_a,
                                          /*conjugate_x=*/false,
                                          /*conjugate_y=*/conjugate_a);
        auto to_solve = b_slice - prev_contribution;

        xla::XlaOp update;
        if (k > 1) {
          TF_ASSIGN_OR_RETURN(xla::XlaComputation * solve,
                              get_base_triangular_solve(k));
          update = xla::Call(builder, *solve, {a_slice, to_solve});
        } else {
          auto a_slice_conj = MaybeConjugate(a_slice, conjugate_a);
          update = to_solve / a_slice_conj;
        }
        output = UpdateSliceInMinorDims(output, update, {0, i});
      }

    } else if (left_side && lower != transpose_a) {
      // for i in range(0, a.shape[-1], block_size):
      for (int64 i = 0; i < m; i += block_size) {
        int64 k = std::min(block_size, m - i);

        // output[..., i:i+k, :] = triangular_solve(
        //     a[..., i:i+k, i:i+k],
        //     b[..., i:i+k, :] - np.matmul(a[..., i:i+k, :i],
        //                                  output[..., :i, :]),
        //     ..., block_size=1)
        auto a_slice = SliceInMinorDims(a, {i, i}, {i + k, i + k});
        auto b_slice = SliceInMinorDims(b, {i, 0}, {i + k, n});

        xla::XlaOp a_prev;
        if (lower) {
          a_prev = SliceInMinorDims(a, {i, 0}, {i + k, m});
        } else {
          a_prev = SliceInMinorDims(a, {0, i}, {m, i + k});
        }
        auto prev_contribution = BatchDot(a_prev, output,
                                          /*transpose_x=*/transpose_a,
                                          /*transpose_y=*/false,
                                          /*conjugate_x=*/conjugate_a,
                                          /*conjugate_y=*/false);
        auto to_solve = b_slice - prev_contribution;

        xla::XlaOp update;
        if (k > 1) {
          TF_ASSIGN_OR_RETURN(xla::XlaComputation * solve,
                              get_base_triangular_solve(k));
          update = xla::Call(builder, *solve, {a_slice, to_solve});
        } else {
          auto a_slice_conj = MaybeConjugate(a_slice, conjugate_a);
          update = to_solve / a_slice_conj;
        }
        output = UpdateSliceInMinorDims(output, update, {i, 0});
      }
    } else if (!left_side && lower != transpose_a) {
      // for i in reversed(range(0, a.shape[-1], block_size)):
      const int64 last_blk_ix =
          xla::RoundUpToNearest(n, block_size) - block_size;
      for (int64 i = last_blk_ix; i >= 0; i -= block_size) {
        int64 k = std::min(block_size, n - i);

        // output[..., :, i:i+k] = triangular_solve(
        //     a[..., i:i+k, i:i+k],
        //     b[..., :, i:i+k] - np.matmul(output[..., :, :i],
        //                                  a[..., :i, i:i+k]),\
        //     ..., block_size=1)
        auto a_slice = SliceInMinorDims(a, {i, i}, {i + k, i + k});
        auto b_slice = SliceInMinorDims(b, {0, i}, {m, i + k});

        xla::XlaOp a_prev;
        if (lower) {
          a_prev = SliceInMinorDims(a, {0, i}, {n, i + k});
        } else {
          a_prev = SliceInMinorDims(a, {i, 0}, {i + k, n});
        }
        auto prev_contribution = BatchDot(output, a_prev,
                                          /*transpose_x=*/false,
                                          /*transpose_y=*/transpose_a,
                                          /*conjugate_x=*/false,
                                          /*conjugate_y=*/conjugate_a);
        auto to_solve = b_slice - prev_contribution;

        xla::XlaOp update;
        if (k > 1) {
          TF_ASSIGN_OR_RETURN(xla::XlaComputation * solve,
                              get_base_triangular_solve(k));
          update = xla::Call(builder, *solve, {a_slice, to_solve});
        } else {
          auto a_slice_conj = MaybeConjugate(a_slice, conjugate_a);
          update = to_solve / a_slice_conj;
        }
        output = UpdateSliceInMinorDims(output, update, {0, i});
      }
    } else {  // left_side && lower == transpose_a
      // for i in reversed(range(0, a.shape[-1], block_size)):
      const int64 last_blk_ix =
          xla::RoundUpToNearest(m, block_size) - block_size;
      for (int64 i = last_blk_ix; i >= 0; i -= block_size) {
        int64 k = std::min(block_size, m - i);

        // output[..., i:i+k, :] = triangular_solve(
        //     a[..., i:i+k, i:i+k],
        //     b[..., i:i+k, :] - np.matmul(a[..., i:i+k, :i],
        //                                  output[..., :i, :]),
        //     ..., block_size=1)
        auto a_slice = SliceInMinorDims(a, {i, i}, {i + k, i + k});
        auto b_slice = SliceInMinorDims(b, {i, 0}, {i + k, n});

        xla::XlaOp a_prev;
        if (lower) {
          a_prev = SliceInMinorDims(a, {0, i}, {m, i + k});
        } else {
          a_prev = SliceInMinorDims(a, {i, 0}, {i + k, m});
        }
        auto prev_contribution = BatchDot(a_prev, output,
                                          /*transpose_x=*/transpose_a,
                                          /*transpose_y=*/false,
                                          /*conjugate_x=*/conjugate_a,
                                          /*conjugate_y=*/false);
        auto to_solve = b_slice - prev_contribution;

        xla::XlaOp update;
        if (k > 1) {
          TF_ASSIGN_OR_RETURN(xla::XlaComputation * solve,
                              get_base_triangular_solve(k));
          update = xla::Call(builder, *solve, {a_slice, to_solve});
        } else {
          auto a_slice_conj = MaybeConjugate(a_slice, conjugate_a);
          update = to_solve / a_slice_conj;
        }
        output = UpdateSliceInMinorDims(output, update, {i, 0});
      }
    }

    return output;
  });
}

xla::XlaOp TriangularSolveLeftLooking(xla::XlaOp a, xla::XlaOp b,
                                      bool transpose_a, bool conjugate_a) {
  xla::XlaBuilder* builder = a.builder();
  return builder->ReportErrorOrReturn([&]() -> xla::StatusOr<xla::XlaOp> {
    TF_ASSIGN_OR_RETURN(xla::Shape a_shape, builder->GetShape(a));
    TF_ASSIGN_OR_RETURN(xla::Shape b_shape, builder->GetShape(b));
    const int64 m = xla::ShapeUtil::GetDimension(b_shape, -2);
    const int64 n = xla::ShapeUtil::GetDimension(b_shape, -1);
    const int64 ndims = xla::ShapeUtil::Rank(a_shape);

    std::vector<int64> batch_dimensions;
    int64 num_batches = 1;
    for (int i = 0; i < ndims - 2; ++i) {
      int64 a_size = a_shape.dimensions(i);
      batch_dimensions.push_back(a_size);
      num_batches = num_batches * a_size;
    }

    // Rescale the input to be unit triangular
    auto diag = Diagonal(a);
    xla::XlaOp scaled_a;
    std::vector<int64> broadcast_dimensions(ndims - 1);
    std::iota(broadcast_dimensions.begin(), broadcast_dimensions.end(), 0);
    if (transpose_a) {
      scaled_a = Div(a, diag, broadcast_dimensions);
    } else {
      // Broadcast over the rows
      broadcast_dimensions[ndims - 2] = ndims - 1;
      scaled_a = Div(a, diag, broadcast_dimensions);
    }

    // The main computation is performed in a While loop.

    // Allocate the output and set its first or last row,
    // output = np.zeros_like(b)
    // if transpose_a:
    //   output[..., m-1:, :] = b[..., m-1:, :] / a[..., m-1:, m-1:]
    // else:
    //   output[..., :1, :] = b[..., :1, :] / a[..., :1, :1]
    xla::XlaOp output = xla::ZerosLike(b);
    {
      auto i = transpose_a ? m - 1 : 0;
      auto a_slice = SliceInMinorDims(scaled_a, {i, i}, {i + 1, i + 1});
      auto b_slice = SliceInMinorDims(b, {i, 0}, {i + 1, n});
      auto a_slice_conj = MaybeConjugate(a_slice, conjugate_a);
      auto update = b_slice / a_slice_conj;
      output = UpdateSliceInMinorDims(output, update, {i, 0});
    }

    // Construct the initial loop carry tuple,
    // if transpose_a:
    //   init = (m-2, output, a, b)
    // else:
    //   init = (1, output, a, b)
    std::vector<xla::Shape> tuple_shapes = {
        // The loop iteration counter is a scalar, incremented each iteration.
        xla::ShapeUtil::MakeShape(xla::S32, {}),
        // The output has the shape of b, with one row updated each iteration.
        b_shape,
        // The coefficient matrix a is a loop invariant.
        a_shape,
        // The right-hand-side matrix b is a loop invariant.
        b_shape};
    xla::Shape tuple_shape = xla::ShapeUtil::MakeTupleShape(tuple_shapes);
    auto init_i = xla::ConstantR0<int32>(builder, transpose_a ? m - 2 : 1);
    auto init = xla::Tuple(builder, {init_i, output, scaled_a, b});

    // Construct the loop condition function,
    // def cond_fun(loop_carry):
    //   i, output, a, b = loop_carry
    //   return i >= 0 if transpose_a else i < m
    std::unique_ptr<xla::XlaBuilder> condb =
        builder->CreateSubBuilder("TriangularSolveLeftLookingWhileCond");
    {
      auto i = xla::GetTupleElement(
          xla::Parameter(condb.get(), 0, tuple_shape,
                         "TriangularSolveLeftLookingWhileTuple"),
          0);
      if (transpose_a) {
        xla::Ge(i, xla::ConstantR0<int32>(condb.get(), 0));
      } else {
        xla::Lt(i, xla::ConstantR0<int32>(condb.get(), m));
      }
    }
    TF_ASSIGN_OR_RETURN(auto cond, condb->Build());

    // Construct the loop body function,
    // def body_fun(loop_carry):
    //   i, output, a, b = loop_carry
    //   if transpose_a:
    //     a_row = np.swapaxes(a[..., i+1:, i:i+1], -1 -2)
    //   else:
    //     a_row = a[..., i:i+1, :i]
    //   result_row = b[..., i:i+1, :] - np.matmul(a_row, output[..., :, :])
    //   output[..., i:i+1, :] = result_row / a[..., i:i+1, i:i+1]
    //   if transpose_a:
    //     return (i - 1, output, a, b)
    //   else:
    //     return (i + 1, output, a, b)
    // We have to do some extra FLOPs propagating zeros in the matrix multiply
    // because we can't have the size of its arguments depend on the loop
    // counter.
    std::unique_ptr<xla::XlaBuilder> bodyb =
        builder->CreateSubBuilder("TriangularSolveLeftLookingWhileBody");
    {
      auto input_tuple = xla::Parameter(bodyb.get(), 0, tuple_shape,
                                        "TriangularSolveLeftLookingWhileTuple");

      // i, output, a, b = loop_carry
      auto i = xla::GetTupleElement(input_tuple, 0);
      auto body_out = xla::GetTupleElement(input_tuple, 1);
      auto body_a = xla::GetTupleElement(input_tuple, 2);
      auto body_b = xla::GetTupleElement(input_tuple, 3);
      auto zero = xla::ConstantR0<int32>(bodyb.get(), 0);

      // We'd like to implement this:
      //   if transpose_a:
      //     a_row = T(a[..., i+1:, i:i+1])
      //     result_row = (b[..., i:i+1, :]
      //                   - np.matmul(a_row, body_out[..., i+1:, :]))
      //   else:
      //     result_row = (b[..., i:i+1, :]
      //                   - np.matmul(a[..., i:i+1, :i], body_out[..., :i, :]))
      // But since we can't have intermediate array sizes depend on the loop
      // counter, we instead exploit the fact that we initialized the output to
      // all zeros and use that as zero-padding (doing unnecessary FLOPs).
      xla::XlaOp a_row;
      if (transpose_a) {
        a_row = DynamicSliceInMinorDims(body_a, {zero, i}, {m, 1});
      } else {
        a_row = DynamicSliceInMinorDims(body_a, {i, zero}, {1, m});
      }
      auto b_update = BatchDot(a_row, body_out,
                               /*transpose_x=*/transpose_a,
                               /*transpose_y=*/false,
                               /*conjugate_x=*/conjugate_a,
                               /*conjugate_y=*/false);
      auto result_row_slice =
          DynamicSliceInMinorDims(body_b, {i, zero}, {1, n});
      auto result_row = result_row_slice - b_update;

      // body_out[..., i:i+1, :] = result_row
      body_out = DynamicUpdateSliceInMinorDims(body_out, result_row, {i, zero});

      // if transpose_a:
      //   return (i - 1, body_out, a, b)
      // else:
      //   return (i + 1, body_out, a, b)
      auto next_i = xla::Add(
          i, xla::ConstantR0<int32>(bodyb.get(), transpose_a ? -1 : 1));
      xla::Tuple(bodyb.get(), {next_i, body_out, body_a, body_b});
    }
    TF_ASSIGN_OR_RETURN(auto body, bodyb->Build());

    // Construct the While loop and return the result,
    // return while_loop(cond_fun, body_fun, init)[1]
    auto triangular_solve_left_looking_while = xla::While(cond, body, init);
    output = xla::GetTupleElement(triangular_solve_left_looking_while, 1);
    auto scaling = MaybeConjugate(diag, conjugate_a);
    // Broadcast over the columns
    broadcast_dimensions[ndims - 2] = ndims - 2;
    return Div(output, scaling, broadcast_dimensions);
  });
}

xla::XlaOp TriangularSolveRightLooking(xla::XlaOp a, xla::XlaOp b,
                                       bool transpose_a, bool conjugate_a) {
  xla::XlaBuilder* builder = a.builder();
  return builder->ReportErrorOrReturn([&]() -> xla::StatusOr<xla::XlaOp> {
    TF_ASSIGN_OR_RETURN(xla::Shape a_shape, builder->GetShape(a));
    TF_ASSIGN_OR_RETURN(xla::Shape b_shape, builder->GetShape(b));
    const int64 m = xla::ShapeUtil::GetDimension(b_shape, -2);
    const int64 n = xla::ShapeUtil::GetDimension(b_shape, -1);
    const int64 ndims = xla::ShapeUtil::Rank(a_shape);

    std::vector<int64> batch_dimensions;
    int64 num_batches = 1;
    for (int i = 0; i < ndims - 2; ++i) {
      int64 a_size = a_shape.dimensions(i);
      batch_dimensions.push_back(a_size);
      num_batches = num_batches * a_size;
    }

    // Rescale the input to be unit triangular
    auto diag = Diagonal(a);
    xla::XlaOp scaled_a;
    std::vector<int64> broadcast_dimensions(ndims - 1);
    std::iota(broadcast_dimensions.begin(), broadcast_dimensions.end(), 0);
    if (transpose_a) {
      // Broadcast over the rows
      broadcast_dimensions[ndims - 2] = ndims - 1;
      scaled_a = Div(a, diag, broadcast_dimensions);
    } else {
      scaled_a = Div(a, diag, broadcast_dimensions);
    }

    // The main computation is performed in a While loop.
    xla::XlaOp output = xla::ZerosLike(b);

    // Construct the initial loop carry tuple,
    // if transpose_a:
    //   init = (0, output, a, b)
    // else:
    //   init = (n-1, output, a, b)
    std::vector<xla::Shape> tuple_shapes = {
        // The loop iteration counter is a scalar, incremented each iteration.
        xla::ShapeUtil::MakeShape(xla::S32, {}),
        // The output has the shape of b, with one row updated each iteration.
        b_shape,
        // The coefficient matrix a is a loop invariant.
        a_shape,
        // The right-hand-side matrix b is a loop invariant.
        b_shape};
    xla::Shape tuple_shape = xla::ShapeUtil::MakeTupleShape(tuple_shapes);
    auto init_i = xla::ConstantR0<int32>(builder, transpose_a ? 0 : n - 1);
    auto init = xla::Tuple(builder, {init_i, output, scaled_a, b});

    // Construct the loop condition function,
    // def cond_fun(loop_carry):
    //   i, output, a, b = loop_carry
    //   return i < n if transpose_a else i >= 0
    std::unique_ptr<xla::XlaBuilder> condb =
        builder->CreateSubBuilder("TriangularSolveRightLookingWhileCond");
    {
      auto i = xla::GetTupleElement(
          xla::Parameter(condb.get(), 0, tuple_shape,
                         "TriangularSolveRightLookingWhileTuple"),
          0);
      if (transpose_a) {
        xla::Lt(i, xla::ConstantR0<int32>(condb.get(), n));
      } else {
        xla::Ge(i, xla::ConstantR0<int32>(condb.get(), 0));
      }
    }
    TF_ASSIGN_OR_RETURN(auto cond, condb->Build());

    // Construct the loop body function,
    // def body_fun(loop_carry):
    //   i, output, a, b = loop_carry
    //   if transpose_a:
    //     a_row = np.swapaxes(a[..., :, i:i+1], -1, -2)
    //   else:
    //     a_row = a[..., :, i:i+1]
    //   result_row = b[..., :, i:i+1] - np.matmul(output, a_row)
    //   output[..., :, i:i+1] = result_row / a[..., i:i+1, i:i+1]
    //   if transpose_a:
    //     return (i - 1, output, a, b)
    //   else:
    //     return (i + 1, output, a, b)
    // We have to do some extra FLOPs propagating zeros in the matrix multiply
    // because we can't have the size of its arguments depend on the loop
    // counter.
    std::unique_ptr<xla::XlaBuilder> bodyb =
        builder->CreateSubBuilder("TriangularSolveRightLookingWhileBody");
    {
      auto input_tuple = xla::Parameter(
          bodyb.get(), 0, tuple_shape, "TriangularSolveRightLookingWhileTuple");

      // i, output, a, b = loop_carry
      auto i = xla::GetTupleElement(input_tuple, 0);
      auto body_out = xla::GetTupleElement(input_tuple, 1);
      auto body_a = xla::GetTupleElement(input_tuple, 2);
      auto body_b = xla::GetTupleElement(input_tuple, 3);
      auto zero = xla::ConstantR0<int32>(bodyb.get(), 0);

      // result = b - np.matmul(output, a)
      // result_row = result[..., :, i:i+1]
      auto body_b_slice = DynamicSliceInMinorDims(body_b, {zero, i}, {m, 1});
      xla::XlaOp a_slice;
      if (transpose_a) {
        a_slice = DynamicSliceInMinorDims(body_a, {i, zero}, {1, n});
      } else {
        a_slice = DynamicSliceInMinorDims(body_a, {zero, i}, {n, 1});
      }
      auto b_update = body_b_slice - BatchDot(body_out, a_slice,
                                              /*transpose_x=*/false,
                                              /*transpose_y=*/transpose_a,
                                              /*conjugate_x=*/false,
                                              /*conjugate_y=*/conjugate_a);

      // body_out[..., :, i:i+1] = b_update
      body_out = DynamicUpdateSliceInMinorDims(body_out, b_update, {zero, i});

      // if transpose_a:
      //   return (i + 1, body_out, a, b)
      // else:
      //   return (i - 1, body_out, a, b)
      auto next_i = xla::Add(
          i, xla::ConstantR0<int32>(bodyb.get(), transpose_a ? 1 : -1));
      xla::Tuple(bodyb.get(), {next_i, body_out, body_a, body_b});
    }
    TF_ASSIGN_OR_RETURN(auto body, bodyb->Build());

    // Construct the While loop and return the result,
    // return while_loop(cond_fun, body_fun, init)[1]
    auto triangular_solve_left_looking_while = xla::While(cond, body, init);
    output = xla::GetTupleElement(triangular_solve_left_looking_while, 1);
    auto scaling = MaybeConjugate(diag, conjugate_a);
    // Broadcast over the rows
    broadcast_dimensions[ndims - 2] = ndims - 1;
    return Div(output, scaling, broadcast_dimensions);
  });
}

}  // namespace tensorflow
