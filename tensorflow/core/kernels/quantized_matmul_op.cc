/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

// Implements a quantized eight-bit version of the matmul operation.

#define EIGEN_USE_THREADS

#define GEMMLOWP_ALLOW_SLOW_SCALAR_FALLBACK
#include "public/gemmlowp.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/kernels/meta_support.h"
#include "tensorflow/core/kernels/quantization_utils.h"
#include "tensorflow/core/kernels/reference_gemm.h"
#include "tensorflow/core/lib/core/errors.h"

namespace tensorflow {

// We have to break this out as a separate function because there are multiple
// combinations of transpose attributes we need to support, and they have to be
// compile-time constants to work with the templates used internally.
template <bool TransposeA, bool TransposeB, bool TransposeC>
void GemmlowpMultiply(OpKernelContext* op_context, const quint8* a_data,
                      const quint8* b_data, qint32* c_data, int m, int n, int k,
                      int offset_a, int offset_b, int lda, int ldb, int ldc) {
  const uint8* a_data_as_uint8 = &(a_data->value);
  const uint8* b_data_as_uint8 = &(b_data->value);
  int32* c_data_as_int32 = &(c_data->value);
  static const gemmlowp::MapOrder ResultOrder =
      !TransposeC ? gemmlowp::MapOrder::RowMajor : gemmlowp::MapOrder::ColMajor;
  static const gemmlowp::MapOrder LhsOrder =
      !TransposeA ? gemmlowp::MapOrder::RowMajor : gemmlowp::MapOrder::ColMajor;
  static const gemmlowp::MapOrder RhsOrder =
      !TransposeB ? gemmlowp::MapOrder::RowMajor : gemmlowp::MapOrder::ColMajor;
  gemmlowp::MatrixMap<const std::uint8_t, LhsOrder> lhs(a_data_as_uint8, m, k,
                                                        lda);
  gemmlowp::MatrixMap<const std::uint8_t, RhsOrder> rhs(b_data_as_uint8, k, n,
                                                        ldb);
  gemmlowp::MatrixMap<std::int32_t, ResultOrder> result(c_data_as_int32, m, n,
                                                        ldc);
  const std::tuple<> empty_pipeline = {};
  auto& worker_threads =
      *(op_context->device()->tensorflow_cpu_worker_threads());
  TensorflowGemmContext context(worker_threads.num_threads,
                                worker_threads.workers);
  gemmlowp::GemmWithOutputPipeline<std::uint8_t, std::int32_t,
                                   gemmlowp::DefaultL8R8BitDepthParams>(
      &context, lhs, rhs, &result, -offset_a, -offset_b, empty_pipeline);
  // Since gemmlowp uses assembly to write to the output, msan won't detect
  // the output buffer as written to, so we mark it manually.
  TF_ANNOTATE_MEMORY_IS_INITIALIZED(c_data_as_int32, m * n * sizeof(int32));
}

template <class T1, class T2, class Toutput>
class QuantizedMatMulOp : public OpKernel {
 public:
  explicit QuantizedMatMulOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("transpose_a", &transpose_a_));
    OP_REQUIRES_OK(context, context->GetAttr("transpose_b", &transpose_b_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& a = context->input(0);
    const Tensor& b = context->input(1);
    const float min_a = context->input(2).flat<float>()(0);
    const float max_a = context->input(3).flat<float>()(0);
    const float min_b = context->input(4).flat<float>()(0);
    const float max_b = context->input(5).flat<float>()(0);

    // Make sure that we have valid quantization ranges for the input buffers.
    // If the difference between the min and max is negative or zero, it makes
    // it hard to do meaningful intermediate operations on the values.
    OP_REQUIRES(context, (max_a > min_a),
                errors::InvalidArgument("max_a must be larger than min_a."));
    OP_REQUIRES(context, (max_b > min_b),
                errors::InvalidArgument("max_b must be larger than min_b."));
    const int32_t offset_a = FloatToQuantizedUnclamped<T1>(0.0f, min_a, max_a);
    const int32_t offset_b = FloatToQuantizedUnclamped<T2>(0.0f, min_b, max_b);
    const int32_t offset_c = 0;
    const int32_t mult_c = 1;
    const int32_t shift_c = 0;

    // Check that the dimensions of the two matrices are valid.
    OP_REQUIRES(context, TensorShapeUtils::IsMatrix(a.shape()),
                errors::InvalidArgument("In[0] is not a matrix"));
    OP_REQUIRES(context, TensorShapeUtils::IsMatrix(b.shape()),
                errors::InvalidArgument("In[1] is not a matrix"));
    Eigen::array<Eigen::IndexPair<Eigen::DenseIndex>, 1> dim_pair;
    dim_pair[0].first = transpose_a_ ? 0 : 1;
    dim_pair[0].second = transpose_b_ ? 1 : 0;

    OP_REQUIRES(context,
                a.dim_size(dim_pair[0].first) == b.dim_size(dim_pair[0].second),
                errors::InvalidArgument("Matrix size-incompatible: In[0]: ",
                                        a.shape().DebugString(),
                                        ", In[1]: ", b.shape().DebugString()));

    OP_REQUIRES(context, ((shift_c >= 0) && (shift_c <= 31)),
                errors::InvalidArgument("shift_c must be between 0 and 31, "
                                        "inclusive."));

    int a_dim_remaining = 1 - dim_pair[0].first;
    int b_dim_remaining = 1 - dim_pair[0].second;
    TensorShape out_shape(
        {a.dim_size(a_dim_remaining), b.dim_size(b_dim_remaining)});
    Tensor* c = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &c));
    CHECK(c);

    const T1* a_data = a.flat<T1>().data();
    const T2* b_data = b.flat<T2>().data();
    Toutput* c_data = c->flat<Toutput>().data();

    const bool transpose_c = false;
    const size_t m = a.dim_size(a_dim_remaining);
    const size_t n = b.dim_size(b_dim_remaining);
    const size_t k = a.dim_size(dim_pair[0].first);
    const size_t lda = a.dim_size(1);
    const size_t ldb = b.dim_size(1);
    const size_t ldc = n;

    if (meta::IsSupportedAndEnabled() && std::is_same<T1, quint8>() &&
        std::is_same<T2, quint8>() && std::is_same<Toutput, qint32>() &&
        (offset_c == 0) && (mult_c == 1) && (shift_c == 0) &&
        (transpose_c == false) && (k <= 2048)) {
      // Gemmlowp/meta code path works on 32 & 64 bit Arm with NEON Simd and
      // allows optimized quantized 8bit to 32bit gemm.
      meta::QuantizedGemm(context, transpose_a_, transpose_b_, a_data, b_data,
                          c_data, m, n, k, -offset_a, -offset_b, lda, ldb, ldc);
    } else if (std::is_same<T1, quint8>() && std::is_same<T2, quint8>() &&
               std::is_same<Toutput, qint32>() && (offset_c == 0) &&
               (mult_c == 1) && (shift_c == 0) && (transpose_c == false)) {
      // The gemmlowp optimized library only works for a particular set of data
      // types, so check if we meet those requirements and fall back to a slower
      // reference implementation if not.
      if (transpose_a_) {
        if (transpose_b_) {
          GemmlowpMultiply<true, true, false>(context, a_data, b_data, c_data,
                                              m, n, k, offset_a, offset_b, lda,
                                              ldb, ldc);
        } else {
          GemmlowpMultiply<true, false, false>(context, a_data, b_data, c_data,
                                               m, n, k, offset_a, offset_b, lda,
                                               ldb, ldc);
        }
      } else {
        if (transpose_b_) {
          GemmlowpMultiply<false, true, false>(context, a_data, b_data, c_data,
                                               m, n, k, offset_a, offset_b, lda,
                                               ldb, ldc);
        } else {
          GemmlowpMultiply<false, false, false>(context, a_data, b_data, c_data,
                                                m, n, k, offset_a, offset_b,
                                                lda, ldb, ldc);
        }
      }
    } else {
      ReferenceGemm<T1, T2, Toutput>(
          transpose_a_, transpose_b_, transpose_c, m, n, k, a_data, offset_a,
          lda, b_data, offset_b, ldb, c_data, shift_c, offset_c, mult_c, ldc);
    }

    float min_c_value;
    float max_c_value;
    QuantizationRangeForMultiplication<T1, T2, Toutput>(
        min_a, max_a, min_b, max_b, &min_c_value, &max_c_value);
    Tensor* c_min = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(1, {}, &c_min));
    c_min->flat<float>()(0) = min_c_value;

    Tensor* c_max = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(2, {}, &c_max));
    c_max->flat<float>()(0) = max_c_value;
  }

 private:
  bool transpose_a_;
  bool transpose_b_;
};

REGISTER_KERNEL_BUILDER(Name("QuantizedMatMul")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<quint8>("T1")
                            .TypeConstraint<quint8>("T2")
                            .TypeConstraint<qint32>("Toutput"),
                        QuantizedMatMulOp<quint8, quint8, qint32>);

}  // namespace tensorflow
