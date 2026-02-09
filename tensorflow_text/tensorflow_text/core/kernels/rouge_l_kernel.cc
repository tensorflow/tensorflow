// Copyright 2025 TF.Text Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "tensorflow/core/framework/lookup_interface.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {
namespace text {

namespace {
}  // namespace


// ROUGE-L implementation based on
// https://www.microsoft.com/en-us/research/publication/
// rouge-a-package-for-automatic-evaluation-of-summaries/
template <typename SPLITS_TYPE, typename VALUES_TYPE>
class RougeLOp : public OpKernel {
 public:
  using ConstFlatSplits = typename TTypes<SPLITS_TYPE>::ConstFlat;
  using ConstFlatValues = typename TTypes<VALUES_TYPE>::ConstFlat;

  explicit RougeLOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& hyp_tensor = ctx->input(0);
    const auto hyp_tensor_flat = hyp_tensor.flat<VALUES_TYPE>();
    const Tensor& hyp_splits = ctx->input(1);
    const auto hyp_splits_flat = hyp_splits.flat<SPLITS_TYPE>();

    const Tensor& ref_tensor = ctx->input(2);
    const auto ref_tensor_flat = ref_tensor.flat<VALUES_TYPE>();
    const Tensor& ref_splits = ctx->input(3);
    const auto ref_splits_flat = ref_splits.flat<SPLITS_TYPE>();

    const Tensor& alpha_tensor = ctx->input(4);
    const auto alpha_scalar = alpha_tensor.scalar<float>();
    const float alpha = alpha_scalar();

    // Alpha must be <=1.
    OP_REQUIRES(ctx, alpha <= 1,
                errors::InvalidArgument("alpha must be <1 but was=", alpha));

    // Ref and Hyp must have the same number of rows.
    OP_REQUIRES(ctx, ref_splits_flat.size() == hyp_splits_flat.size(),
                errors::InvalidArgument(
                    "ref splits len=", ref_splits_flat.size(),
                    "must equal hyp splits len=", hyp_splits_flat.size()));

    // All inputs must be vectors.
    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(hyp_tensor.shape()),
                errors::InvalidArgument("hypotheses values must be a vector"));
    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(ref_tensor.shape()),
                errors::InvalidArgument("references values must be a vector"));
    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(hyp_splits.shape()),
                errors::InvalidArgument("hypotheses splits must be a vector"));
    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(ref_splits.shape()),
                errors::InvalidArgument("references splits must be a vector"));
    // Ref and Hyp must have at least one split.
    OP_REQUIRES(ctx, ref_splits_flat.size() > 0,
                errors::InvalidArgument(
                    "ref splits len=0; must have at least 1 split"));

    // Output is a dense Tensor containing one row per input row.
    TensorShape output_shape({ref_splits_flat.size() - 1});

    // Allocate the F-Measure output tensor.
    Tensor* f_measure_tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_output("f_measure", output_shape,
                                             &f_measure_tensor));
    auto f_measures_flat = f_measure_tensor->flat<float>();

    // Allocate the P-Measure output tensor.
    Tensor* p_measure_tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_output("p_measure", output_shape,
                                             &p_measure_tensor));
    auto p_measures_flat = p_measure_tensor->flat<float>();

    // Allocate the R-Measure output tensor.
    Tensor* r_measure_tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_output("r_measure", output_shape,
                                             &r_measure_tensor));
    auto r_measures_flat = r_measure_tensor->flat<float>();

    // Iterate over the splits, skipping the first split as it is always zero.
    for (int i = 1; i < hyp_splits_flat.size(); i++) {
      // Length of hyp and ref.
      SPLITS_TYPE lhyp = hyp_splits_flat(i) - hyp_splits_flat(i-1);
      SPLITS_TYPE lref = ref_splits_flat(i) - ref_splits_flat(i-1);
      // Length of longest common substring.
      int32 llcs = LongestCommonSubsequenceLength(hyp_splits_flat(i-1),
                                                  hyp_splits_flat(i),
                                                  hyp_tensor_flat,
                                                  ref_splits_flat(i-1),
                                                  ref_splits_flat(i),
                                                  ref_tensor_flat);
      auto measures = ComputeMeasures(lhyp, lref, llcs, alpha);
      f_measures_flat(i - 1) = std::get<0>(measures);
      p_measures_flat(i - 1) = std::get<1>(measures);
      r_measures_flat(i - 1) = std::get<2>(measures);
    }
  }

 private:
  // By using LCS, the ROUGE-L algorithm does not require consecutive matches
  // but rather credits the order of N-grams.
  int32 LongestCommonSubsequenceLength(
      const SPLITS_TYPE hyp_i,
      const SPLITS_TYPE hyp_j,
      const ConstFlatValues& hyp,
      const SPLITS_TYPE ref_i,
      const SPLITS_TYPE ref_j,
      const ConstFlatValues& ref) {
    SPLITS_TYPE lhyp = hyp_j - hyp_i;
    SPLITS_TYPE lref = ref_j - ref_i;
    // Create a scratch matrix to keep track of the LCS seen so far using DP.
    // http://www.algorithmist.com/index.php/Longest_Common_Subsequence
    Tensor scratch(DT_INT32, {lhyp + 2, lref + 2});
    auto scratch2d = scratch.matrix<int32>();
    for (SPLITS_TYPE x = hyp_i; x <= hyp_j + 1; x++) {
      for (SPLITS_TYPE y = ref_i; y <= ref_j + 1; y++) {
        SPLITS_TYPE a = x - hyp_i;
        SPLITS_TYPE b = y - ref_i;
        if (a == 0 || b == 0) {
          // If in first row or column, we write a zero to the table.
          scratch2d(a, b) = 0;
        } else if (x == hyp_j+1 || y == ref_j+1 || hyp(x-1) != ref(y-1)) {
          // If in the last row or column, or if the tokens are not equal,
          // carry the largest score seen in the cell above or to the left of
          // the current cell.
          scratch2d(a, b) =
              std::max({scratch2d(a - 1, b), scratch2d(a, b - 1)});
        } else {
          // If tokens are equal, we are part of a subsequence, so increment the
          // diagonal score.
          scratch2d(a, b) = scratch2d(a - 1, b - 1) + 1;
        }
      }
    }
    return scratch2d(lhyp, lref);
  }

  std::tuple<float, float, float> ComputeMeasures(const SPLITS_TYPE lhyp_int,
                                                  const SPLITS_TYPE lref_int,
                                                  const int32 llcs_int,
                                                  const float alpha) {
    const float lhyp = static_cast<float>(lhyp_int);
    const float lref = static_cast<float>(lref_int);
    const float llcs = static_cast<float>(llcs_int);
    const float p_lcs = llcs / (lhyp + 1e-12);
    const float r_lcs = llcs / (lref + 1e-12);
    // Use the tensor2tensor formulation if the alpha value is <0,
    // which does not make sense as a weighted average term.
    const float f_lcs = alpha < 0 ?
        ComputeTensor2TensorF(p_lcs, r_lcs) :
        ComputeOfficialF(p_lcs, r_lcs, alpha);
    return std::make_tuple(f_lcs, p_lcs, r_lcs);
  }

  float ComputeTensor2TensorF(const float p_lcs, const float r_lcs) {
    const float beta = p_lcs / (r_lcs + 1e-12);
    const float numerator = (1 + (beta * beta)) * r_lcs * p_lcs;
    const float denominator = r_lcs + ((beta * beta) * p_lcs);
    if (denominator > 0) {
      return numerator / denominator;
    }
    return 0;
  }

  float ComputeOfficialF(const float p_lcs, const float r_lcs,
                         const float alpha) {
    float denominator = (alpha * r_lcs + (1 - alpha) * p_lcs);
    if (denominator > 0) {
      return (p_lcs * r_lcs) / denominator;
    }
    return denominator;
  }

  TF_DISALLOW_COPY_AND_ASSIGN(RougeLOp);
};

#define REGISTER(VALUES_TYPE)                                           \
  REGISTER_KERNEL_BUILDER(Name("RougeL")                        \
                              .Device(DEVICE_CPU)                       \
                              .TypeConstraint<int32>("Tsplits")         \
                              .TypeConstraint<VALUES_TYPE>("Tvalues"),  \
                          RougeLOp<int32, VALUES_TYPE>);        \
  REGISTER_KERNEL_BUILDER(Name("RougeL")                        \
                              .Device(DEVICE_CPU)                       \
                              .TypeConstraint<int64>("Tsplits")         \
                              .TypeConstraint<VALUES_TYPE>("Tvalues"),  \
                          RougeLOp<int64, VALUES_TYPE>);

TF_CALL_int32(REGISTER);
TF_CALL_int64(REGISTER);
TF_CALL_string(REGISTER);
#undef REGISTER

}  // namespace text
}  // namespace tensorflow
