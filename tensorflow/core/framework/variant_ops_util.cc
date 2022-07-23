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

#include "tensorflow/core/framework/variant_ops_util.h"

#include <functional>
#include <utility>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/variant.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {
// AddVariantTo efficiently performs:
//    temp[lhs_ix] <- array(lhs_ix) + array(rhs_ix)
// where array(ix) := (temp_filled[ix]
//                     ? temp[ix]
//                     : ctx->input(ix).scalar<Variant>()())
// This reduces (possibly expensive) copying of Variants from
// the inputs into temp at the lowest levels of the summation tree.
static inline Status AddVariantTo(
    OpKernelContext* ctx, const int lhs_ix, const int rhs_ix,
    gtl::InlinedVector<Variant, 4>* temp,
    gtl::InlinedVector<bool, 4>* temp_filled,
    std::function<Status(OpKernelContext*, const Variant&, const Variant&,
                         Variant*)>
        binary_add_variant) {
  Variant tmp;
  if (temp_filled->at(lhs_ix)) tmp = std::move(temp->at(lhs_ix));
  const Variant& a = temp_filled->at(lhs_ix)
                         ? tmp
                         : ctx->input(lhs_ix).template scalar<Variant>()();
  const Variant& b = temp_filled->at(rhs_ix)
                         ? temp->at(rhs_ix)
                         : ctx->input(rhs_ix).template scalar<Variant>()();
  Variant* c = &temp->at(lhs_ix);
  TF_RETURN_IF_ERROR(binary_add_variant(ctx, a, b, c));
  temp_filled->at(lhs_ix) = true;
  return OkStatus();
}

void AddNVariant(OpKernelContext* ctx,
                 std::function<Status(OpKernelContext*, const Variant&,
                                      const Variant&, Variant*)>
                     binary_add_variant) {
  const Tensor& input0 = ctx->input(0);
  const int num = ctx->num_inputs();

  if (num == 1) {
    ctx->set_output(0, input0);
    return;
  }

  for (int i = 0; i < num; ++i) {
    // Step 1: ensure unary variants.
    OP_REQUIRES(
        ctx, ctx->input(i).dims() == 0,
        errors::InvalidArgument(
            "AddN of non-scalar Tensor with dtype=DT_VARIANT is not "
            "supported; inputs[",
            i, " has shape: ", ctx->input(i).shape().DebugString(), "."));
  }

  // Step 2: Sum input variants in a tree-like structure using
  //   BinaryOpVariants(ADD_VARIANT_BINARY_OP, ...)
  //   For the output create a default-constructed variant object.
  //
  // Pairwise summation provides better numerical precision by
  // reducing round-off error:
  //
  //   https://en.wikipedia.org/wiki/Pairwise_summation
  //
  // These two vectors are used to store and mark intermediate sums.
  gtl::InlinedVector<bool, 4> temp_filled(num, false);
  gtl::InlinedVector<Variant, 4> temp(num);

  // Tree-based summation.
  int skip = 1;
  int n = num;
  while (skip < n) {
    int i = skip;
    while (i < n) {
      // TODO(ebrevdo, rmlarsen): Parallelize the pairwise summations in the
      // inner loop if the variants are "large".

      // x[i - skip] += x[i]
      OP_REQUIRES_OK(ctx, AddVariantTo(ctx, i - skip, i, &temp, &temp_filled,
                                       binary_add_variant));
      // We won't use this index again, recover its memory.
      temp[i].clear();
      i += 2 * skip;
    }
    if (i == n) {
      // x[0] += x[i - skip]
      OP_REQUIRES_OK(ctx, AddVariantTo(ctx, 0, i - skip, &temp, &temp_filled,
                                       binary_add_variant));
      // We won't use this index again, recover its memory.
      temp[i - skip].clear();
      n -= skip;
    }
    skip *= 2;
  }

  Tensor out(cpu_allocator(), DT_VARIANT, TensorShape({}));
  out.scalar<Variant>()() = std::move(temp[0]);
  ctx->set_output(0, out);
}
}  //  end namespace tensorflow
