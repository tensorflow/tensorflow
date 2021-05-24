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

#define EIGEN_USE_THREADS

#include <vector>

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/concat_lib_cpu.h"
#include "tensorflow/core/kernels/quantization_utils.h"

namespace tensorflow {

namespace {
template <typename T>
struct RequantizeCopier {
  RequantizeCopier(
      const std::vector<std::pair<float, float>>* input_min_and_max,
      float output_min, float output_max)
      : output_min(output_min),
        output_max(output_max),
        input_min_and_max(input_min_and_max) {}

  inline void Copy(T* dst, const T* src, int input_index, size_t n) {
    const float input_min = (*input_min_and_max)[input_index].first;
    const float input_max = (*input_min_and_max)[input_index].second;
    if (input_min == output_min && input_max == output_max) {
      DCHECK(DataTypeCanUseMemcpy(DataTypeToEnum<T>::v()));
      memcpy(dst, src, n * sizeof(T));
    } else {
      Eigen::array<Eigen::DenseIndex, 1> dims;
      dims[0] = n;
      typename TTypes<T, 1>::UnalignedConstTensor input_array(src, dims);
      typename TTypes<T, 1>::UnalignedTensor output_array(dst, dims);

      QuantizedToFloatStruct<T> q2f(input_min, input_max);
      auto input_float = DEQUANTIZE_WITH_EIGEN(input_array, q2f);
      FloatToQuantizedStruct<T> f2q(output_min, output_max);
      auto input_requantized = QUANTIZE_WITH_EIGEN(input_float, f2q, T);

      // RequantizeCopier::Copy is called from within a shard of computation, so
      // don't use the threadpool device here, simply assign with default CPU
      // device.
      output_array = input_requantized;
    }
  }

  float output_min;
  float output_max;
  const std::vector<std::pair<float, float>>* input_min_and_max;
};
}  // namespace

template <typename T>
class QuantizedConcatOp : public OpKernel {
 public:
  typedef std::vector<std::unique_ptr<typename TTypes<T, 2>::ConstMatrix>>
      ConstMatrixVector;

  explicit QuantizedConcatOp(OpKernelConstruction* c) : OpKernel(c) {}

  void CalculateInputAndOutputRange(
      const OpInputList& input_mins, const OpInputList& input_maxes,
      const size_t N,
      std::vector<std::pair<float, float>>* input_mins_and_maxes,
      float* output_min, float* output_max) {
    input_mins_and_maxes->reserve(N);
    float overall_min = std::numeric_limits<float>::max();
    float overall_max = std::numeric_limits<float>::lowest();
    for (int i = 0; i < N; ++i) {
      const float input_min = input_mins[i].flat<float>()(0);
      const float input_max = input_maxes[i].flat<float>()(0);
      input_mins_and_maxes->emplace_back(input_min, input_max);
      overall_min = std::min(overall_min, input_min);
      overall_max = std::max(overall_max, input_max);
    }
    // Make sure min is no more than zero.
    overall_min = std::min(0.0f, overall_min);
    if (std::is_signed<T>::value) {
      // For signed, we want a symmetrical distribution including zero for the
      // output, so pick a range that meets that need.
      const float largest_value =
          std::max(std::abs(overall_min), std::abs(overall_max));
      *output_min = -largest_value;
      *output_max = largest_value;
    } else {
      *output_min = overall_min;
      *output_max = overall_max;
    }
  }

  int64 CalculateInputsDim(const TensorShape& input_shape,
                           const int32 concat_dim) {
    int64 inputs_flat_dim0 = 1;
    for (int d = 0; d < concat_dim; ++d) {
      inputs_flat_dim0 *= input_shape.dim_size(d);
    }
    return inputs_flat_dim0;
  }

  void CalculateConcatDims(const size_t N, const TensorShape& input_shape,
                           int input_dims, const OpInputList& values,
                           OpKernelContext* context, const int32 concat_dim,
                           const int64 inputs_flat_dim0,
                           ConstMatrixVector* inputs_flat,
                           int* output_concat_dim) {
    // Note that we reduce the concat of n-dimensional tensors into a two
    // dimensional concat. Assuming the dimensions of any input/output
    // tensor are {x0, x1,...,xn-1, y0, y1,...,ym-1}, where the concat is along
    // the dimension indicated with size y0, we flatten it to {x, y}, where y =
    // Prod_i(yi) and x = ((n > 0) ? Prod_i(xi) : 1).
    inputs_flat->reserve(N);
    *output_concat_dim = 0;
    const bool input_is_scalar = TensorShapeUtils::IsScalar(input_shape);
    for (int i = 0; i < N; ++i) {
      const auto in = values[i];
      const bool in_is_scalar = TensorShapeUtils::IsScalar(in.shape());
      OP_REQUIRES(
          context, in.dims() == input_dims || (input_is_scalar && in_is_scalar),
          errors::InvalidArgument(
              "ConcatOp : Ranks of all input tensors should match: shape[0] = ",
              input_shape.DebugString(), " vs. shape[", i,
              "] = ", in.shape().DebugString()));
      for (int j = 0; j < input_dims; ++j) {
        if (j == concat_dim) {
          continue;
        }
        OP_REQUIRES(
            context, in.dim_size(j) == input_shape.dim_size(j),
            errors::InvalidArgument(
                "ConcatOp : Dimensions of inputs should match: shape[0] = ",
                input_shape.DebugString(), " vs. shape[", i,
                "] = ", in.shape().DebugString()));
      }
      if (in.NumElements() > 0) {
        int64 inputs_flat_dim1 = in.NumElements() / inputs_flat_dim0;
        inputs_flat->emplace_back(new typename TTypes<T, 2>::ConstMatrix(
            in.shaped<T, 2>({inputs_flat_dim0, inputs_flat_dim1})));
      }
      *output_concat_dim += in.dims() > 0 ? in.dim_size(concat_dim) : 1;
    }
  }

  void Compute(OpKernelContext* context) override {
    const Tensor* concat_dim_tensor = nullptr;
    OP_REQUIRES_OK(context, context->input("concat_dim", &concat_dim_tensor));
    OP_REQUIRES(
        context, TensorShapeUtils::IsScalar(concat_dim_tensor->shape()),
        errors::InvalidArgument(
            "Concat dim tensor should be a scalar integer, but got shape ",
            concat_dim_tensor->shape().DebugString()));
    const int32 concat_dim = concat_dim_tensor->scalar<int32>()();
    OpInputList values;
    OP_REQUIRES_OK(context, context->input_list("values", &values));
    const size_t N = values.size();
    OpInputList input_mins;
    OP_REQUIRES_OK(context, context->input_list("input_mins", &input_mins));
    OP_REQUIRES(context, (input_mins.size() == N),
                errors::InvalidArgument(
                    "QuantizedConcatOp : Expected mins input list length ",
                    input_mins.size(), " to equal values length ", N));
    OpInputList input_maxes;
    OP_REQUIRES_OK(context, context->input_list("input_maxes", &input_maxes));
    OP_REQUIRES(context, (input_maxes.size() == N),
                errors::InvalidArgument(
                    "QuantizedConcatOp : Expected maxes input list length ",
                    input_maxes.size(), " to equal values length ", N));
    const int input_dims = values[0].dims();
    const TensorShape& input_shape = values[0].shape();
    OP_REQUIRES(
        context, (0 <= concat_dim && concat_dim < input_dims),
        errors::InvalidArgument(
            "ConcatOp : Expected concatenating dimensions in the range [", 0,
            ", ", input_dims, "), but got ", concat_dim));

    float output_min = std::numeric_limits<float>::max();
    float output_max = std::numeric_limits<float>::lowest();
    std::vector<std::pair<float, float>> input_mins_and_maxes;
    CalculateInputAndOutputRange(input_mins, input_maxes, N,
                                 &input_mins_and_maxes, &output_min,
                                 &output_max);
    const int64 inputs_flat_dim0 = CalculateInputsDim(input_shape, concat_dim);
    ConstMatrixVector inputs_flat;
    int output_concat_dim;
    CalculateConcatDims(N, input_shape, input_dims, values, context, concat_dim,
                        inputs_flat_dim0, &inputs_flat, &output_concat_dim);

    TensorShape output_shape(input_shape);
    // TODO(irving): Remove rank 0 case once !kAllowLegacyScalars
    if (output_shape.dims() == 0) {
      output_shape.AddDim(output_concat_dim);
    } else {
      output_shape.set_dim(concat_dim, output_concat_dim);
    }
    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));

    if (output->NumElements() > 0) {
      int64 output_dim1 = output->NumElements() / inputs_flat_dim0;
      auto output_flat = output->shaped<T, 2>({inputs_flat_dim0, output_dim1});
      ConcatCPUImpl<T>(
          context->device(), inputs_flat, sizeof(T) /* cost_per_unit */,
          RequantizeCopier<T>(&input_mins_and_maxes, output_min, output_max),
          &output_flat);
    }

    Tensor* output_min_tensor = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(1, {}, &output_min_tensor));
    output_min_tensor->flat<float>()(0) = output_min;

    Tensor* output_max_tensor = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(2, {}, &output_max_tensor));
    output_max_tensor->flat<float>()(0) = output_max;
  }
};

#define REGISTER_QUANTIZED_CONCAT(type)                  \
  REGISTER_KERNEL_BUILDER(Name("QuantizedConcat")        \
                              .Device(DEVICE_CPU)        \
                              .TypeConstraint<type>("T") \
                              .HostMemory("concat_dim"), \
                          QuantizedConcatOp<type>)

REGISTER_QUANTIZED_CONCAT(quint8);
REGISTER_QUANTIZED_CONCAT(qint32);

#undef REGISTER_QUANTIZED_CONCAT

}  // namespace tensorflow
