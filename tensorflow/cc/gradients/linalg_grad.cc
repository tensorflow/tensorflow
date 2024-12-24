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

#include <algorithm>
#include <cmath>
#include <string>
#include <tuple>

#include "absl/container/btree_set.h"
#include "absl/container/flat_hash_set.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "tensorflow/cc/framework/grad_op_registry.h"
#include "tensorflow/cc/framework/gradients.h"
#include "tensorflow/cc/gradients/grad_helper.h"
#include "tensorflow/cc/ops/array_ops_internal.h"
#include "tensorflow/cc/ops/math_ops_internal.h"
#include "tensorflow/cc/ops/standard_ops.h"

namespace tensorflow {
namespace ops {
namespace {

constexpr absl::string_view kEllipsis = "...";

// Returns the axis (possibly negative) corresponding to a label.
//
// Returns the axis index of the axis label if it is before an ellipsis (or if
// the ellipsis is not present), and the negative index if it occurs after the
// ellipsis. E.g. index of `b` in `ab...cd`, is `1`, but that of `c` is `-2`.
//
// For multiple occurrences, returns the leftmost one. If not found, returns
// absl::nullopt.
//
// Parameters:
//   subscripts: A string denoting the einsum subscript (e.g. `ab...cd`)
//   label: The single character axis label.
absl::optional<int> EinsumGetAxisFromLabel(absl::string_view subscripts,
                                           char label) {
  std::vector<absl::string_view> splits = absl::StrSplit(subscripts, kEllipsis);
  auto index = splits[0].find(label);
  if (index != splits[0].npos) {
    return index;
  }
  if (splits.size() < 2) {
    return absl::nullopt;
  }
  index = splits[1].find(label);
  if (index != splits[1].npos) {
    return index - splits[1].length();
  }
  return absl::nullopt;
}

// Returns a tuple denoting the slice mapping to ellipsis.
//
// For a given subscript, returns a tuple (start, end) denoting the start
// axis index and the (negative) end axis index respectively. For any input
// Tensor `x` described by the subscript, `x[start:end]` would be the slice
// represented by the ellipsis. E.g. For `ab...cd` returns `[1, -2]`.
//
// If ellipsis is not present in `subscripts`, returns `(0, 0)`.
//
// Parameters:
//   subscripts: A string denoting the einsum subscript.
//   start: Output for the start index
//   end: Output for the end index (or nullopt to go to the end).
std::tuple<int, absl::optional<int>> EinsumGetBcastSubshape(
    absl::string_view subscripts) {
  int start = subscripts.find(kEllipsis);
  if (start == subscripts.npos) {
    return std::make_tuple(0, 0);
  }
  int remaining = subscripts.length() - (start + kEllipsis.length());
  absl::optional<int> end;
  if (remaining > 0) {
    end = -remaining;
  } else {
    end = absl::nullopt;
  }
  return std::make_tuple(start, end);
}

// Slices elements of a 1d tensor from [start,end].
// If end is nullopt, it goes to the end of the tensor.
// Supports negative values for end.
// This attempts to give the same result as tenspr[start:end] would give in
// Python.
Output Slice1dHelper(const Scope& scope, Output tensor, int start,
                     absl::optional<int> end) {
  if (end.has_value() && *end > 0) {
    return Slice(scope, tensor, Const(scope, start, TensorShape({1})),
                 Const(scope, *end - start, TensorShape({1})));
  } else {
    return Slice(scope, tensor, Const(scope, start, TensorShape({1})),
                 Add(scope, Shape(scope, tensor), end.value_or(0) - start));
  }
}

// Returns reduced subscripts and their corresponding dimensions and axes.
//
// Given a set of axis labels, returns their concatenated subscript, their
// corresponding dimensions from input_shape, and their corresponding axes.
// Note that the concatenated subscript `reduced_subs` may have axis labels
// from `reduced_label_set` in any order. For example, for the reduced label
// set `{b, d}`, subscripts `aabbcd` and input shape `[2,2,5,5,3,4]`, returns
// subscripts `bd`, dimensions `[5,4]` and axes `[2,5]`.
//
// Args:
//   reduced_label_set: Set of axis labels which appear in `subscripts`.
//   input_shape: A `Tensor` representing the shape of the einsum operand
//     corresponding to `subscripts`.
//   subscripts: A string denoting the einsum subscript.
//
// Returns:
//   reduced_subs: Subscripts formed by a concatenation of labels in
//     `reduced_label_set`.
//   reduced_dims: Dimensions from `input_shape` corresponding to each label
//     in `reduced_subs`.
//   reduced_axes: Axes described by `subscripts` corresponding to each label
//     in `reduced_subs`. If there are multiple occurrences in `subscripts`,
//     we consider only the leftmost one.
std::tuple<std::string, Output, Output> EinsumGetReducedSubscripts(
    const Scope& scope, const absl::btree_set<char>& reduced_label_set,
    Output input_shape, absl::string_view subscripts) {
  // Concatenate the sequence of reduced axis labels.
  const std::string reduced_subs =
      std::string(reduced_label_set.begin(), reduced_label_set.end());
  // Get the axis (may be positive, negative or zero) for each of the reduced
  // labels. If the same label appears multiple times, get the left-most axis.
  std::vector<int> reduced_axes;
  reduced_axes.reserve(reduced_subs.size());
  for (const char s : reduced_subs) {
    auto axis = EinsumGetAxisFromLabel(subscripts, s);
    if (!axis.has_value()) {
      // Should never happen.
      scope.UpdateStatus(errors::Internal(
          absl::StrCat("Missing axis", absl::string_view(&s, 1))));
    } else {
      reduced_axes.push_back(*axis);
    }
  }
  // Get the corresponding dimensions for each reduced axis.
  std::vector<Output> reduced_dims_inputs;
  reduced_dims_inputs.reserve(reduced_axes.size());
  for (const int i : reduced_axes) {
    if (i < 0) {
      reduced_dims_inputs.push_back(
          Gather(scope, input_shape, Add(scope, Size(scope, input_shape), i)));
    } else {
      reduced_dims_inputs.push_back(Gather(scope, input_shape, i));
    }
  }
  const Output reduced_dims = Stack(scope, reduced_dims_inputs);
  Tensor reduced_axes_tensor(
      DataType::DT_INT32, TensorShape({static_cast<int>(reduced_axes.size())}));
  std::copy_n(reduced_axes.begin(), reduced_axes.size(),
              reduced_axes_tensor.flat<int>().data());
  return std::make_tuple(reduced_subs, reduced_dims,
                         Const(scope, reduced_axes_tensor));
}

// Returns the gradient wrt input for a unary einsum with reductions.
//
//  scope: Scope for grad operations.
//  output_grad: The gradient wrt the output of a unary einsum operation.
//  output_subs: The output subscript. (E.g. `ac` for equation `abc->ac`).
//  input_subs: The input subscript. (E.g. `abc` for equation `abc->ac`).
//  input_shape: The shape of the input operand.
//  reduced_label_set: The set of axis labels appearing in `input_subs` but
//    not in `output_subs`.
Output EinsumGradReducedHelper(const Scope& scope, const Output& output_grad,
                               absl::string_view output_subs,
                               absl::string_view input_subs,
                               const Output& input_shape,
                               const absl::btree_set<char>& reduced_label_set) {
  // Let's say the einsum operation was "aabbcd->ca", where axis labels 'b' and
  // 'd' are reduced with input_shape [2,2,5,5,3,4]. Then obtain the reduced
  // subscripts "bd", corresponding dimensions [5,4] and axes [2,5].
  std::string reduced_subs;
  Output reduced_dims, reduced_axes;
  std::tie(reduced_subs, reduced_dims, reduced_axes) =
      EinsumGetReducedSubscripts(scope, reduced_label_set, input_shape,
                                 input_subs);
  // Whether either the input or the output subscripts have a repeated label.
  // This is true for "aabbcd->ca" or "abd->cca" but false for "abcd->ca".
  const int distinct_input_labels =
      absl::flat_hash_set<char>(input_subs.begin(), input_subs.end()).size();
  const int distinct_output_labels =
      absl::flat_hash_set<char>(output_subs.begin(), output_subs.end()).size();
  const bool has_repeated_labels =
      (distinct_input_labels + distinct_output_labels) <
      input_subs.length() + output_subs.length();
  // Compute the input subscripts without the reduced axis labels, e.g. "aac"
  // for the equation "aabbcd->ca".
  std::string input_subs_without_reduced_labels;
  for (const char s : input_subs) {
    if (!absl::c_linear_search(reduced_label_set, s)) {
      input_subs_without_reduced_labels.push_back(s);
    }
  }

  // The gradient wrt the input for the equation "abc->ac" (or, equivalently
  // reduce_sum(..., axis=1)) is just the gradient of the output tiled N times
  // along axis 1, where label 'b' represents a dimension of size N.
  //
  // If we're not dealing with repeated labels, and the non-reduced labels
  // doesn't need to be transposed, then just tiling is enough and there is no
  // need to call another einsum. For example, tiling is sufficient for
  // "abcd->ac". But for equations like "aabbcd->ac" (generalized traces) or
  // "abc->ca" (transpose), we'd need another einsum operation after tiling.
  if (!has_repeated_labels &&
      input_subs_without_reduced_labels == output_subs) {
    // Obtain the shape of the output, as if keepdims=True on reduce sum. E.g.
    // for the equation "abcd->ac" with input shape [2,5,3,4], we get the
    // reduced shape [2,1,3,1].
    auto reduced_shape = ReducedShapeHelper(scope, input_shape, reduced_axes);
    // Reshaping the gradient (wrt "ac") to [2,1,3,1] and broadcasting it to
    // the shape [2,5,3,4] results in the gradient wrt "abcd".
    return BroadcastTo(scope, Reshape(scope, output_grad, reduced_shape),
                       input_shape);
  }

  // If we *do* have traces or transpose operations, then prepend the extra
  // reduced dimensions to the front. E.g. Given the equation "aabbcd->ca" we'd
  // first obtain the VJP for "bdca->ca", and then the VJP for "aabbcd->bdca".
  //
  // Obtain the input shape with reduced dimensions prepended, viz. [5,4,3,2].
  // This is the shape of the intermediate "bdca".
  Output output_grad_shape = Shape(scope, output_grad);
  auto grad_shape_with_reduced_labels =
      Concat(scope, {reduced_dims, output_grad_shape}, /*axis=*/0);

  // Obtain the output shape of the reduction-only equation "bdca->ca" as if
  // keepdims=True; viz. [1,1,3,2]. Since we prepended the reduced labels,
  // we just have to prepend that many 1s to the output shape.

  auto reduced_shape = Concat(
      scope,
      {Const(scope, 1, TensorShape{static_cast<int>(reduced_label_set.size())}),
       output_grad_shape},
      /*axis=*/0);
  // Compute the VJP for the intermediate (viz. "bdca->ca") for which
  // broadcasting is sufficient.
  Output broadcasted_grad =
      BroadcastTo(scope, Reshape(scope, output_grad, reduced_shape),
                  grad_shape_with_reduced_labels);
  // Compute the VJP for the final step (viz. "aabbcd->bdca"). We can
  // use einsum with the input and output subscripts reversed (viz.
  // "bdca->aabbcd") since the output axis labels now appear in the
  // input subscripts.
  return Einsum(scope, {broadcasted_grad},
                absl::StrCat(reduced_subs, output_subs, "->", input_subs));
}

// Returns the gradient wrt an input operand for a binary einsum.
//
// This function does not handle (un)broadcasting. This must be done separately
// on the returned gradient.
//
// Args:
//   output_grad: The gradient wrt the output of a binary einsum operation.
//   other_operand: The complementary `Tensor` operand i.e. which is not the
//     input operand.
//   input_shape: A `Tensor` representing the shape of input operand.
//   input_subs: The subscripts of the input operand.
//   other_subs: The subscripts of the complementary operand.
//   output_subs: The output subscripts.
Output EinsumGradWrt(const Scope& scope, Output output_grad,
                     Output other_operand, Output input_shape,
                     absl::string_view input_subs, absl::string_view other_subs,
                     absl::string_view output_subs) {
  // Claim: For the einsum operation z = einsum("{eq_x},{eq_y}->{eq_z}", x, y),
  //   where the equation involves only Tensor contractions, generalized traces
  //   and transposes, the input gradients are given by the vector-jacobian
  //   products (VJPs):
  //
  //     grad_wrt_x = einsum("{eq_y},{eq_z}->{eq_x}", y, grad_wrt_z)
  //     grad_wrt_y = einsum("{eq_x},{eq_z}->{eq_y}", x, grad_wrt_z}
  //
  //   where grad_wrt_x and grad_wrt_y are the gradients with respect to inputs
  //   x and y and grad_wrt_z is the given gradient with respect to output z.
  //
  // Proof: For unary einsum equations involving only transpose ("ij->ji") and
  //   traces ("ii->i"), the linear mapping's Jacobian at input x is given
  //   by the function itself. We can verify that the linear map given by the
  //   VJP are einsums with the equations "ji->ij" and "i->ii" respectively,
  //   where the latter represents 'un-tracing', or filling the diagonal with
  //   the input axis and non-diagonal entries are zeros.
  //        Furthermore, recall that matrix multiplication, which is
  //   represented by the equation "ab,bc->ac", has its VJPs given by the
  //   einsum equations "ac,bc->ab" and "ab,ac->bc" (see, for example
  //   https://math.stackexchange.com/a/2755680). Combined with transposes and
  //   traces we can rewrite Tensor contractions as regular matrix
  //   multiplication. Since each of these operations have their VJPs described
  //   by einsums of the required pattern, the result follows.
  //
  // Accordingly, einsum operations except for those with reductions, e.g.
  // "abc,cd->ad" have their VJPs defined by:
  //   "{output_subs},{other_subs}->{input_subs}".
  //
  // But if there is a reduction, this would lead to the equation "ad,cd->abc"
  // which is invalid because the reduced axis label 'b' is present in the
  // output but not in any of the inputs. Therefore, we compute the VJP in two
  // steps: first we obtain VJP for "ac,cd->ad" and then we compute the VJP of
  // "abc->ac" or, equivalently, reduce_sum(..., axis=1).
  //
  // Compute the set of input axis labels which doesn't appear in either the
  // output subscripts or the other operand's subscript. E.g. the set {'b'} for
  // the equation "abc,cd->ad".
  absl::btree_set<char> reduced_label_set(input_subs.begin(), input_subs.end());
  for (const char x : output_subs) {
    reduced_label_set.erase(x);
  }
  for (const char x : other_subs) {
    reduced_label_set.erase(x);
  }
  reduced_label_set.erase('.');

  // Obtain the input subscripts with the reduced axis labels removed. E.g.
  // "ac" in the above example.
  std::string left_subs;
  for (const char s : input_subs) {
    if (!reduced_label_set.contains(s)) {
      left_subs.push_back(s);
    }
  }

  // Compute the gradient wrt the input, without accounting for the operation
  // "abc->ac". So, now we have the VJP of the operation "ac,cd->ad".
  Output grad_reduced =
      Einsum(scope, {output_grad, other_operand},
             absl::StrCat(output_subs, ",", other_subs, "->", left_subs));

  // If the reduced_label_set is empty, then we already have the gradient
  // wrt the input.
  if (reduced_label_set.empty()) {
    return grad_reduced;
  }
  // Otherwise, we currently have the gradient wrt the output of the reduction
  // operation "abc->ac". Invoke the subroutine for the gradient for unary
  // einsum with reductions.
  return EinsumGradReducedHelper(scope, grad_reduced, left_subs, input_subs,
                                 input_shape, reduced_label_set);
}

absl::Status EinsumGrad(const Scope& scope, const Operation& op,
                        const std::vector<Output>& grad_inputs,
                        std::vector<Output>* grad_outputs) {
  if (grad_inputs.size() != 1) {
    return errors::InvalidArgument("Expect 1 grad input.");
  }
  const Output& grad = grad_inputs[0];

  std::string equation;
  TF_RETURN_IF_ERROR(GetNodeAttr(op.node()->attrs(), "equation", &equation));
  std::vector<absl::string_view> equation_split =
      absl::StrSplit(equation, "->");
  if (equation_split.size() != 2) {
    return errors::InvalidArgument("Equation must contain a single ->");
  }

  const absl::string_view input_subs = equation_split[0];
  const absl::string_view output_subs = equation_split[1];
  if (op.num_inputs() == 1) {
    // For the unary einsum z = einsum("{eq_x}->{eq_z}", x), the gradient wrt
    // the input (VJP) is given by the reversed equation:
    //   grad_wrt_x = einsum("{eq_z}->{eq_x}", grad_wrt_z)
    // (See the justification in _GetGradWrt). This is valid unless there are
    // reduced axis labels; i.e. axis labels appearing in the input but not in
    // the output subscripts.
    auto input_shape = Shape(scope, op.input(0));
    // Find the axis labels which appear only in the input.
    absl::btree_set<char> reduced_label_set(input_subs.begin(),
                                            input_subs.end());
    for (const char x : output_subs) {
      reduced_label_set.erase(x);
    }
    reduced_label_set.erase('.');
    if (reduced_label_set.empty()) {
      grad_outputs->push_back(Einsum(
          scope, grad_inputs, absl::StrCat(output_subs, "->", input_subs)));
      return scope.status();
    }
    // We do have reduced axes, so we invoke the subroutine for reduced unary
    // einsums.
    grad_outputs->push_back(EinsumGradReducedHelper(
        scope, grad, output_subs, input_subs, input_shape, reduced_label_set));
    return scope.status();
  }

  std::vector<absl::string_view> subs = absl::StrSplit(input_subs, ',');
  if (subs.size() != 2) {
    return errors::InvalidArgument("Only 2 inputs are supported");
  }
  std::string x_subs(subs[0]);
  std::string y_subs(subs[1]);
  // Add ellipsis for broadcasted dimensions if any operand does not have it.
  // This is because the equation "...ij,jk->ik" may be valid if the 0th input's
  // batch shape is empty, but the VJP equation "jk,ik->...ij" is not valid
  // because only the output subscripts contain ellipsis.
  if (absl::StrContains(output_subs, kEllipsis)) {
    if (!absl::StrContains(x_subs, kEllipsis)) {
      absl::StrAppend(&x_subs, kEllipsis);
    }
    if (!absl::StrContains(y_subs, kEllipsis)) {
      absl::StrAppend(&y_subs, kEllipsis);
    }
  }

  // Obtain the gradients wrt the inputs x and y, without taking into account
  // the unbroadcasting.
  tensorflow::Output x = op.input(0);
  tensorflow::Output y = op.input(1);
  if (DataTypeIsComplex(grad.type())) {
    x = Conj(scope, x);
    y = Conj(scope, y);
  }

  const auto x_shape = Shape(scope, x);
  const auto y_shape = Shape(scope, y);
  Output grad_x =
      EinsumGradWrt(scope, grad, y, x_shape, x_subs, y_subs, output_subs);
  Output grad_y =
      EinsumGradWrt(scope, grad, x, y_shape, y_subs, x_subs, output_subs);

  if (!absl::StrContains(output_subs, kEllipsis)) {
    // If no ellipsis in the output; then no need to unbroadcast.
    grad_outputs->push_back(grad_x);
    grad_outputs->push_back(grad_y);
    return scope.status();
  }

  // Below we handle the case that broadcasting between x and y was necessary,
  // with x and y having possibly different batch shapes.

  // Obtain the range of axes which map to ellipsis. E.g. for subscripts
  // 'ab...c' and shape of rank 10; the range [3:-1] denotes the broadcasted
  // axes.
  int bx_start, by_start;
  absl::optional<int> bx_end, by_end;
  std::tie(bx_start, bx_end) = EinsumGetBcastSubshape(x_subs);
  std::tie(by_start, by_end) = EinsumGetBcastSubshape(y_subs);

  // Sum the gradient across the broadcasted axes.
  auto args = internal::BroadcastGradientArgs(
      scope, Slice1dHelper(scope, x_shape, bx_start, bx_end),
      Slice1dHelper(scope, y_shape, by_start, by_end));
  grad_x = Reshape(
      scope, ReduceSum(scope, grad_x, Add(scope, bx_start, args.r0)), x_shape);
  grad_y = Reshape(
      scope, ReduceSum(scope, grad_y, Add(scope, by_start, args.r1)), y_shape);
  grad_outputs->push_back(grad_x);
  grad_outputs->push_back(grad_y);
  return scope.status();
}

REGISTER_GRADIENT_OP("Einsum", EinsumGrad);

}  // namespace
}  // namespace ops
}  // namespace tensorflow
