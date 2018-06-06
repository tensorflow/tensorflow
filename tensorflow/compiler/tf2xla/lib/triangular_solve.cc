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
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/lib/core/errors.h"

namespace tensorflow {

xla::StatusOr<xla::XlaOp> TriangularSolve(xla::XlaBuilder* builder,
                                          const xla::XlaOp& a, xla::XlaOp b,
                                          bool left_side, bool lower,
                                          bool transpose_a, bool conjugate_a,
                                          int64 block_size) {
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

      auto a_param = sub->Parameter(
          0,
          xla::ShapeUtil::MakeShape(
              b_shape.element_type(),
              PrependMajorDims(sub.get(), batch_dimensions, {k, k})),
          "a");

      std::array<int64, 2> b_lastd;
      if (left_side) {
        b_lastd = {k, n};
      } else {
        b_lastd = {m, k};
      }
      auto b_param = sub->Parameter(
          1,
          xla::ShapeUtil::MakeShape(
              b_shape.element_type(),
              PrependMajorDims(sub.get(), batch_dimensions, b_lastd)),
          "b");

      // We use a left-looking or right-looking subroutine on the block diagonal
      // in the lower=true cases, while falling back to a recursive call in
      // others. The left-looking and right-looking subroutines are written with
      // a While loop and so yields much faster compile times. Moreover, they
      // can give higher performance on smaller (sub)problems.
      if (left_side && lower) {
        TF_RETURN_IF_ERROR(TriangularSolveLeftLooking(sub.get(), a_param,
                                                      b_param, transpose_a,
                                                      conjugate_a)
                               .status());
      } else if (!left_side && lower) {
        TF_RETURN_IF_ERROR(TriangularSolveRightLooking(sub.get(), a_param,
                                                       b_param, transpose_a,
                                                       conjugate_a)
                               .status());
      } else {
        TF_RETURN_IF_ERROR(TriangularSolve(sub.get(), a_param, b_param,
                                           left_side, lower, transpose_a,
                                           conjugate_a,
                                           /*block_size=*/1)
                               .status());
      }

      TF_ASSIGN_OR_RETURN(computation, sub->Build());
    }
    return &computation;
  };

  xla::XlaOp output = Zeros(builder, b_shape);

  // Right-looking blocked triangular solve.
  // For an explanation of the algorithm, see the TRSM discussion in:
  // Goto, Kazushige, and Robert Van De Geijn. "High-performance implementation
  // of the level-3 BLAS." ACM Transactions on Mathematical Software (TOMS) 35.1
  // (2008): 4.

  // In the code comments below, T = lambda x: np.swapaxes(x, -1, -2) if
  // conjugate_a is False, or T = lambda x: np.conj(np.swapaxes(x, -1, -2)) if
  // conjugate_a is True.

  if (!left_side && lower == transpose_a) {
    // for i in range(0, a.shape[-1], block_size):
    for (int64 i = 0; i < n; i += block_size) {
      int64 k = std::min(block_size, n - i);

      // output[..., :, i:i+k] = triangular_solve(
      //     a[..., i:i+k, i:i+k], b[..., :, i:i+k], ..., block_size=1)
      TF_ASSIGN_OR_RETURN(auto a_slice,
                          SliceInMinorDims(builder, a, {i, i}, {i + k, i + k}));
      TF_ASSIGN_OR_RETURN(auto b_slice,
                          SliceInMinorDims(builder, b, {0, i}, {m, i + k}));
      xla::XlaOp update;
      if (k > 1) {
        TF_ASSIGN_OR_RETURN(xla::XlaComputation * solve,
                            get_base_triangular_solve(k));
        update = builder->Call(*solve, {a_slice, b_slice});
      } else {
        TF_ASSIGN_OR_RETURN(auto a_slice_conj,
                            MaybeConjugate(builder, a_slice, conjugate_a));
        update = builder->Div(b_slice, a_slice_conj);
      }
      TF_ASSIGN_OR_RETURN(
          output, UpdateSliceInMinorDims(builder, output, update, {0, i}));

      // if i + k < a.shape[-1]:
      //   a_slice_2 = a[..., i+k:, i:i+k] if lower else a[..., i:i+k, i+k:]
      //   a_slice_2 = T(a_slice_2) if transpose_a else a_slice_2
      //   b[..., :, i+k:] -= np.matmul(output[..., :, i:i+k], a_slice_2)
      if (i + k < n) {
        xla::XlaOp a_slice_2;
        if (lower) {
          TF_ASSIGN_OR_RETURN(
              a_slice_2, SliceInMinorDims(builder, a, {i + k, i}, {n, i + k}));
        } else {
          TF_ASSIGN_OR_RETURN(
              a_slice_2, SliceInMinorDims(builder, a, {i, i + k}, {i + k, n}));
        }

        TF_ASSIGN_OR_RETURN(auto b_update,
                            BatchDot(builder, update, a_slice_2,
                                     /*transpose_x=*/false,
                                     /*transpose_y=*/transpose_a,
                                     /*conjugate_x=*/false,
                                     /*conjugate_y=*/conjugate_a));
        TF_ASSIGN_OR_RETURN(auto b_slice_2,
                            SliceInMinorDims(builder, b, {0, i + k}, {m, n}));
        b_update = builder->Sub(b_slice_2, b_update);
        TF_ASSIGN_OR_RETURN(
            b, UpdateSliceInMinorDims(builder, b, b_update, {0, i + k}));
      }
    }

  } else if (left_side && lower != transpose_a) {
    // for i in range(0, a.shape[-1], block_size):
    for (int64 i = 0; i < m; i += block_size) {
      int64 k = std::min(block_size, m - i);

      // output[..., i:i+k, :] = triangular_solve(
      //     a[..., i:i+k, i:i+k], b[..., i:i+k, :], ..., block_size=1)
      TF_ASSIGN_OR_RETURN(auto a_slice,
                          SliceInMinorDims(builder, a, {i, i}, {i + k, i + k}));
      TF_ASSIGN_OR_RETURN(auto b_slice,
                          SliceInMinorDims(builder, b, {i, 0}, {i + k, n}));
      xla::XlaOp update;
      if (k > 1) {
        TF_ASSIGN_OR_RETURN(xla::XlaComputation * solve,
                            get_base_triangular_solve(k));
        update = builder->Call(*solve, {a_slice, b_slice});
      } else {
        TF_ASSIGN_OR_RETURN(auto a_slice_conj,
                            MaybeConjugate(builder, a_slice, conjugate_a));
        update = builder->Div(b_slice, a_slice_conj);
      }
      TF_ASSIGN_OR_RETURN(
          output, UpdateSliceInMinorDims(builder, output, update, {i, 0}));

      // if i + k < a.shape[-1]:
      //   a_slice_2 = a[..., i+k:, i:i+k] if lower else a[..., i:i+k, i+k:]
      //   a_slice_2 = T(a_slice_2) if transpose_a else a_slice_2
      //   b[..., i+k:, :] -= np.matmul(a_slice_2, output[..., i:i+k, :])
      if (i + k < m) {
        xla::XlaOp a_slice_2;
        if (lower) {
          TF_ASSIGN_OR_RETURN(
              a_slice_2, SliceInMinorDims(builder, a, {i + k, i}, {m, i + k}));
        } else {
          TF_ASSIGN_OR_RETURN(
              a_slice_2, SliceInMinorDims(builder, a, {i, i + k}, {i + k, m}));
        }

        TF_ASSIGN_OR_RETURN(auto b_update, BatchDot(builder, a_slice_2, update,
                                                    /*transpose_x=*/transpose_a,
                                                    /*transpose_y=*/false,
                                                    /*conjugate_x=*/conjugate_a,
                                                    /*conjugate_y=*/false));
        TF_ASSIGN_OR_RETURN(auto b_slice_2,
                            SliceInMinorDims(builder, b, {i + k, 0}, {m, n}));
        b_update = builder->Sub(b_slice_2, b_update);
        TF_ASSIGN_OR_RETURN(
            b, UpdateSliceInMinorDims(builder, b, b_update, {i + k, 0}));
      }
    }
  } else if (!left_side && lower != transpose_a) {
    // for i in reversed(range(0, a.shape[-1], block_size)):
    const int64 last_blk_ix = xla::RoundUpToNearest(n, block_size) - block_size;
    for (int64 i = last_blk_ix; i >= 0; i -= block_size) {
      int64 k = std::min(block_size, n - i);

      // output[..., :, i:i+k] triangular_solve(
      //     a[..., i:i+k, i:i+k], b[..., :, i:i+k], ..., block_size=1)
      TF_ASSIGN_OR_RETURN(auto a_slice,
                          SliceInMinorDims(builder, a, {i, i}, {i + k, i + k}));
      TF_ASSIGN_OR_RETURN(auto b_slice,
                          SliceInMinorDims(builder, b, {0, i}, {m, i + k}));
      xla::XlaOp update;
      if (k > 1) {
        TF_ASSIGN_OR_RETURN(xla::XlaComputation * solve,
                            get_base_triangular_solve(k));
        update = builder->Call(*solve, {a_slice, b_slice});
      } else {
        TF_ASSIGN_OR_RETURN(auto a_slice_conj,
                            MaybeConjugate(builder, a_slice, conjugate_a));
        update = builder->Div(b_slice, a_slice_conj);
      }
      TF_ASSIGN_OR_RETURN(
          output, UpdateSliceInMinorDims(builder, output, update, {0, i}));

      // if i - k >= 0:
      //   a_slice_2 = a[..., i:i+k, :i] if lower else a[..., :i, i:i+k]
      //   a_slice_2 = T(a_slice_2) if transpose_a else a_slice_2
      //   b[..., :, :i] -= np.matmul(out[..., :, i:i+k], a_slice_2)
      if (i - k >= 0) {
        xla::XlaOp a_slice_2;
        if (lower) {
          TF_ASSIGN_OR_RETURN(a_slice_2,
                              SliceInMinorDims(builder, a, {i, 0}, {i + k, i}));
        } else {
          TF_ASSIGN_OR_RETURN(a_slice_2,
                              SliceInMinorDims(builder, a, {0, i}, {i, i + k}));
        }

        TF_ASSIGN_OR_RETURN(auto b_update,
                            BatchDot(builder, update, a_slice_2,
                                     /*transpose_x=*/false,
                                     /*transpose_y=*/transpose_a,
                                     /*conjugate_x=*/false,
                                     /*conjugate_y=*/conjugate_a));
        TF_ASSIGN_OR_RETURN(auto b_slice_2,
                            SliceInMinorDims(builder, b, {0, 0}, {m, i}));
        b_update = builder->Sub(b_slice_2, b_update);
        TF_ASSIGN_OR_RETURN(
            b, UpdateSliceInMinorDims(builder, b, b_update, {0, 0}));
      }
    }
  } else {  // left_side && lower == transpose_a
    // for i in reversed(range(0, a.shape[-1], block_size)):
    const int64 last_blk_ix = xla::RoundUpToNearest(m, block_size) - block_size;
    for (int64 i = last_blk_ix; i >= 0; i -= block_size) {
      int64 k = std::min(block_size, m - i);

      // output[..., i:i+k, :] triangular_solve(
      //     a[..., i:i+k, i:i+k], b[..., i:i+k, :], ..., block_size=1)
      TF_ASSIGN_OR_RETURN(auto a_slice,
                          SliceInMinorDims(builder, a, {i, i}, {i + k, i + k}));
      TF_ASSIGN_OR_RETURN(auto b_slice,
                          SliceInMinorDims(builder, b, {i, 0}, {i + k, n}));
      xla::XlaOp update;
      if (k > 1) {
        TF_ASSIGN_OR_RETURN(xla::XlaComputation * solve,
                            get_base_triangular_solve(k));
        update = builder->Call(*solve, {a_slice, b_slice});
      } else {
        TF_ASSIGN_OR_RETURN(auto a_slice_conj,
                            MaybeConjugate(builder, a_slice, conjugate_a));
        update = builder->Div(b_slice, a_slice_conj);
      }
      TF_ASSIGN_OR_RETURN(
          output, UpdateSliceInMinorDims(builder, output, update, {i, 0}));

      // if i - k >= 0:
      //   a_slice_2 = a[..., i:i+k, :i] if lower else a[..., :i, i:i+k]
      //   a_slice_2 = T(a_slice_2) if transpose_a else a_slice_2
      //   b[..., :i, :] -= np.matmul(a_slice_2, out[..., i:i+k, :])
      if (i - k >= 0) {
        xla::XlaOp a_slice_2;
        if (lower) {
          TF_ASSIGN_OR_RETURN(a_slice_2,
                              SliceInMinorDims(builder, a, {i, 0}, {i + k, i}));
        } else {
          TF_ASSIGN_OR_RETURN(a_slice_2,
                              SliceInMinorDims(builder, a, {0, i}, {i, i + k}));
        }

        TF_ASSIGN_OR_RETURN(auto b_update, BatchDot(builder, a_slice_2, update,
                                                    /*transpose_x=*/transpose_a,
                                                    /*transpose_y=*/false,
                                                    /*conjugate_x=*/conjugate_a,
                                                    /*conjugate_y=*/false));
        TF_ASSIGN_OR_RETURN(auto b_slice_2,
                            SliceInMinorDims(builder, b, {0, 0}, {i, n}));
        b_update = builder->Sub(b_slice_2, b_update);
        TF_ASSIGN_OR_RETURN(
            b, UpdateSliceInMinorDims(builder, b, b_update, {0, 0}));
      }
    }
  }

  return output;
}

xla::StatusOr<xla::XlaOp> TriangularSolveLeftLooking(xla::XlaBuilder* builder,
                                                     const xla::XlaOp& a,
                                                     const xla::XlaOp& b,
                                                     bool transpose_a,
                                                     bool conjugate_a) {
  TF_ASSIGN_OR_RETURN(xla::Shape a_shape, builder->GetShape(a));
  TF_ASSIGN_OR_RETURN(xla::Shape b_shape, builder->GetShape(b));
  const int64 m = xla::ShapeUtil::GetDimension(b_shape, -2);
  const int64 n = xla::ShapeUtil::GetDimension(b_shape, -1);
  const int64 ndims = xla::ShapeUtil::Rank(a_shape);

  std::vector<int64> batch_dimensions;
  for (int i = 0; i < ndims - 2; ++i) {
    int64 a_size = a_shape.dimensions(i);
    batch_dimensions.push_back(a_size);
  }

  // The main computation is performed in a While loop.

  // Allocate the output and set its first or last row,
  // output = np.zeros_like(b)
  // if transpose_a:
  //   output[..., m-1:, :] = b[..., m-1:, :] / a[..., m-1:, m-1:]
  // else:
  //   output[..., :1, :] = b[..., :1, :] / a[..., :1, :1]
  xla::XlaOp output = Zeros(builder, b_shape);
  {
    auto i = transpose_a ? m - 1 : 0;
    TF_ASSIGN_OR_RETURN(auto a_slice,
                        SliceInMinorDims(builder, a, {i, i}, {i + 1, i + 1}));
    TF_ASSIGN_OR_RETURN(auto b_slice,
                        SliceInMinorDims(builder, b, {i, 0}, {i + 1, n}));
    TF_ASSIGN_OR_RETURN(auto a_slice_conj,
                        MaybeConjugate(builder, a_slice, conjugate_a));
    auto update = builder->Div(b_slice, a_slice_conj);
    TF_ASSIGN_OR_RETURN(
        output, UpdateSliceInMinorDims(builder, output, update, {i, 0}));
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
  auto init_i = builder->ConstantR0<int32>(transpose_a ? m - 2 : 1);
  auto init = builder->Tuple({init_i, output, a, b});

  // Construct the loop condition function,
  // def cond_fun(loop_carry):
  //   i, output, a, b = loop_carry
  //   return i >= 0 if transpose_a else i < m
  std::unique_ptr<xla::XlaBuilder> condb =
      builder->CreateSubBuilder("TriangularSolveLeftLookingWhileCond");
  {
    auto i = condb->GetTupleElement(
        condb->Parameter(0, tuple_shape,
                         "TriangularSolveLeftLookingWhileTuple"),
        0);
    if (transpose_a) {
      condb->Ge(i, condb->ConstantR0<int32>(0));
    } else {
      condb->Lt(i, condb->ConstantR0<int32>(m));
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
  // because we can't have the size of its arguments depend on the loop counter.
  std::unique_ptr<xla::XlaBuilder> bodyb =
      builder->CreateSubBuilder("TriangularSolveLeftLookingWhileBody");
  {
    auto input_tuple = bodyb->Parameter(0, tuple_shape,
                                        "TriangularSolveLeftLookingWhileTuple");

    // i, output, a, b = loop_carry
    auto i = bodyb->GetTupleElement(input_tuple, 0);
    auto body_out = bodyb->GetTupleElement(input_tuple, 1);
    auto body_a = bodyb->GetTupleElement(input_tuple, 2);
    auto body_b = bodyb->GetTupleElement(input_tuple, 3);
    auto zero = bodyb->ConstantR0<int32>(0);

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
      TF_ASSIGN_OR_RETURN(a_row, DynamicSliceInMinorDims(bodyb.get(), body_a,
                                                         {zero, i}, {m, 1}));
    } else {
      TF_ASSIGN_OR_RETURN(a_row, DynamicSliceInMinorDims(bodyb.get(), body_a,
                                                         {i, zero}, {1, m}));
    }
    TF_ASSIGN_OR_RETURN(auto b_update, BatchDot(bodyb.get(), a_row, body_out,
                                                /*transpose_x=*/transpose_a,
                                                /*transpose_y=*/false,
                                                /*conjugate_x=*/conjugate_a,
                                                /*conjugate_y=*/false));
    TF_ASSIGN_OR_RETURN(
        auto result_row_slice,
        DynamicSliceInMinorDims(bodyb.get(), body_b, {i, zero}, {1, n}));
    auto result_row = bodyb->Sub(result_row_slice, b_update);

    // body_out[..., i:i+1, :] = result_row / a[..., i:i+1, i:i+1]
    TF_ASSIGN_OR_RETURN(auto a_elt, DynamicSliceInMinorDims(bodyb.get(), body_a,
                                                            {i, i}, {1, 1}));
    TF_ASSIGN_OR_RETURN(auto a_elt_conj,
                        MaybeConjugate(bodyb.get(), a_elt, conjugate_a));
    auto div_result = bodyb->Div(result_row, a_elt_conj);
    TF_ASSIGN_OR_RETURN(body_out,
                        DynamicUpdateSliceInMinorDims(bodyb.get(), body_out,
                                                      div_result, {i, zero}));

    // if transpose_a:
    //   return (i - 1, body_out, a, b)
    // else:
    //   return (i + 1, body_out, a, b)
    auto next_i = bodyb->Add(i, bodyb->ConstantR0<int32>(transpose_a ? -1 : 1));
    bodyb->Tuple({next_i, body_out, body_a, body_b});
  }
  TF_ASSIGN_OR_RETURN(auto body, bodyb->Build());

  // Construct the While loop and return the result,
  // return while_loop(cond_fun, body_fun, init)[1]
  auto triangular_solve_left_looking_while = builder->While(cond, body, init);
  return builder->GetTupleElement(triangular_solve_left_looking_while, 1);
}

xla::StatusOr<xla::XlaOp> TriangularSolveRightLooking(xla::XlaBuilder* builder,
                                                      const xla::XlaOp& a,
                                                      const xla::XlaOp& b,
                                                      bool transpose_a,
                                                      bool conjugate_a) {
  TF_ASSIGN_OR_RETURN(xla::Shape a_shape, builder->GetShape(a));
  TF_ASSIGN_OR_RETURN(xla::Shape b_shape, builder->GetShape(b));
  const int64 m = xla::ShapeUtil::GetDimension(b_shape, -2);
  const int64 n = xla::ShapeUtil::GetDimension(b_shape, -1);
  const int64 ndims = xla::ShapeUtil::Rank(a_shape);

  std::vector<int64> batch_dimensions;
  for (int i = 0; i < ndims - 2; ++i) {
    int64 a_size = a_shape.dimensions(i);
    batch_dimensions.push_back(a_size);
  }

  // The main computation is performed in a While loop.
  xla::XlaOp output = Zeros(builder, b_shape);

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
  auto init_i = builder->ConstantR0<int32>(transpose_a ? 0 : n - 1);
  auto init = builder->Tuple({init_i, output, a, b});

  // Construct the loop condition function,
  // def cond_fun(loop_carry):
  //   i, output, a, b = loop_carry
  //   return i < n if transpose_a else i >= 0
  std::unique_ptr<xla::XlaBuilder> condb =
      builder->CreateSubBuilder("TriangularSolveRightLookingWhileCond");
  {
    auto i = condb->GetTupleElement(
        condb->Parameter(0, tuple_shape,
                         "TriangularSolveRightLookingWhileTuple"),
        0);
    if (transpose_a) {
      condb->Lt(i, condb->ConstantR0<int32>(n));
    } else {
      condb->Ge(i, condb->ConstantR0<int32>(0));
    }
  }
  TF_ASSIGN_OR_RETURN(auto cond, condb->Build());

  // Construct the loop body function,
  // def body_fun(loop_carry):
  //   i, output, a, b = loop_carry
  //   if transpose_a:
  //     a_row = np.swapaxes(a[..., :, i:i+1], -1 -2)
  //   else:
  //     a_row = a[..., :, i:i+1]
  //   result_row = b[..., :, i:i+1] - np.matmul(output, a_row)
  //   output[..., :, i:i+1] = result_row / a[..., i:i+1, i:i+1]
  //   if transpose_a:
  //     return (i - 1, output, a, b)
  //   else:
  //     return (i + 1, output, a, b)
  // We have to do some extra FLOPs propagating zeros in the matrix multiply
  // because we can't have the size of its arguments depend on the loop counter.
  std::unique_ptr<xla::XlaBuilder> bodyb =
      builder->CreateSubBuilder("TriangularSolveRightLookingWhileBody");
  {
    auto input_tuple = bodyb->Parameter(
        0, tuple_shape, "TriangularSolveRightLookingWhileTuple");

    // i, output, a, b = loop_carry
    auto i = bodyb->GetTupleElement(input_tuple, 0);
    auto body_out = bodyb->GetTupleElement(input_tuple, 1);
    auto body_a = bodyb->GetTupleElement(input_tuple, 2);
    auto body_b = bodyb->GetTupleElement(input_tuple, 3);
    auto zero = bodyb->ConstantR0<int32>(0);

    // We'd like to implement b[..., :, i:i+1] - np.matmul(output, a[..., :,
    // i:i+1]) But since we can't have intermediate array sizes depend on the
    // loop counter, we instead exploit the fact that we initialized the output
    // to all zeros and use that as zero-padding (doing unnecessary FLOPs).
    TF_ASSIGN_OR_RETURN(auto b_update, BatchDot(bodyb.get(), body_out, body_a,
                                                /*transpose_x=*/false,
                                                /*transpose_y=*/transpose_a,
                                                /*conjugate_x=*/false,
                                                /*conjugate_y=*/conjugate_a));
    // result = b - np.matmul(output, a)
    auto result = bodyb->Sub(body_b, b_update);
    // result_row = result[..., :, i:i+1]
    TF_ASSIGN_OR_RETURN(
        auto result_row,
        DynamicSliceInMinorDims(bodyb.get(), result, {zero, i}, {m, 1}));

    // body_out[..., :, i:i+1] = result_row / a[..., i:i+1, i:i+1]
    TF_ASSIGN_OR_RETURN(auto a_ii, DynamicSliceInMinorDims(bodyb.get(), body_a,
                                                           {i, i}, {1, 1}));
    TF_ASSIGN_OR_RETURN(auto a_ii_conj,
                        MaybeConjugate(bodyb.get(), a_ii, conjugate_a));
    auto div_result = bodyb->Div(result_row, a_ii_conj);
    TF_ASSIGN_OR_RETURN(body_out,
                        DynamicUpdateSliceInMinorDims(bodyb.get(), body_out,
                                                      div_result, {zero, i}));

    // if transpose_a:
    //   return (i + 1, body_out, a, b)
    // else:
    //   return (i - 1, body_out, a, b)
    auto next_i = bodyb->Add(i, bodyb->ConstantR0<int32>(transpose_a ? 1 : -1));
    bodyb->Tuple({next_i, body_out, body_a, body_b});
  }
  TF_ASSIGN_OR_RETURN(auto body, bodyb->Build());

  // Construct the While loop and return the result,
  // return while_loop(cond_fun, body_fun, init)[1]
  auto triangular_solve_left_looking_while = builder->While(cond, body, init);
  return builder->GetTupleElement(triangular_solve_left_looking_while, 1);
}

}  // namespace tensorflow
