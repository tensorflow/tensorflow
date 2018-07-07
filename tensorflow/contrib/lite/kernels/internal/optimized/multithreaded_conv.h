/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CONTRIB_LITE_KERNELS_INTERNAL_MULTITHREAD_CONV
#define TENSORFLOW_CONTRIB_LITE_KERNELS_INTERNAL_MULTITHREAD_CONV

#include <assert.h>
#include <stdint.h>
#include <sys/types.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <tuple>
#include <type_traits>

#include "tensorflow/contrib/lite/builtin_op_data.h"
#include "tensorflow/contrib/lite/kernels/internal/common.h"
#include "tensorflow/contrib/lite/kernels/internal/optimized/eigen_spatial_convolutions.h"
#include "tensorflow/contrib/lite/kernels/internal/optimized/optimized_ops.h"
#include "tensorflow/contrib/lite/kernels/internal/types.h"

namespace tflite {
namespace multithreaded_ops {

class EigenThreadPoolWrapper : public Eigen::ThreadPoolInterface {
 public:
  explicit EigenThreadPoolWrapper(Eigen::ThreadPool* pool) : pool_(pool) {}
  ~EigenThreadPoolWrapper() override {}

  void Schedule(std::function<void()> fn) override {
    pool_->Schedule(std::move(fn));
  }
  int NumThreads() const override { return pool_->NumThreads(); }
  int CurrentThreadId() const override { return pool_->CurrentThreadId(); }

 private:
  Eigen::ThreadPool* pool_ = nullptr;
};

// We have a single global threadpool for all convolution operations. This means
// that inferences started from different threads may block each other, but
// since the underlying resource of CPU cores should be consumed by the
// operations anyway, it shouldn't affect overall performance.
const Eigen::ThreadPoolDevice& GetThreadPoolDevice() {
  const int thread_count = 4;
  std::vector<int> proc_set;
  static Eigen::ThreadPool* tp = new Eigen::ThreadPool(thread_count, proc_set);
  static EigenThreadPoolWrapper* thread_pool_wrapper =
      new EigenThreadPoolWrapper(tp);
  static Eigen::ThreadPoolDevice* device =
      new Eigen::ThreadPoolDevice(thread_pool_wrapper, thread_count);
  return *device;
}

// Shorthands for the types we need when interfacing with the EigenTensor
// library.
typedef Eigen::TensorMap<
    Eigen::Tensor<float, 2, Eigen::RowMajor, Eigen::DenseIndex>, Eigen::Aligned>
    EigenMatrix;
typedef Eigen::TensorMap<
    Eigen::Tensor<const float, 2, Eigen::RowMajor, Eigen::DenseIndex>,
    Eigen::Aligned>
    ConstEigenMatrix;

typedef Eigen::TensorMap<
    Eigen::Tensor<float, 4, Eigen::RowMajor, Eigen::DenseIndex>, Eigen::Aligned>
    EigenTensor;
typedef Eigen::TensorMap<
    Eigen::Tensor<const float, 4, Eigen::RowMajor, Eigen::DenseIndex>,
    Eigen::Aligned>
    ConstEigenTensor;

// Utility functions we need for the EigenTensor API.
template <typename Device, typename T>
struct MatMulConvFunctor {
  // Computes on device "d": out = in0 * in1, where * is matrix
  // multiplication.
  void operator()(
      const Device& d, EigenMatrix out, ConstEigenMatrix in0,
      ConstEigenMatrix in1,
      const Eigen::array<Eigen::IndexPair<Eigen::DenseIndex>, 1>& dim_pair) {
    out.device(d) = in0.contract(in1, dim_pair);
  }
};

template <class T>
class EigenTensorConvFunctor {
 private:
  Eigen::PaddingType TfLitePadding2EigenPadding(TfLitePadding padding) {
    switch (padding) {
      case kTfLitePaddingValid:
        return Eigen::PADDING_VALID;
      case kTfLitePaddingSame:
        return Eigen::PADDING_SAME;
      case kTfLitePaddingUnknown:
        assert(false);  // should never get here.
        return Eigen::PADDING_VALID;
    }
    return Eigen::PADDING_SAME;  // Prevent compiler warning about missing
                                 // return
  }

 public:
  void operator()(const T* input_data, T* im2col_buffer, int input_batches,
                  int input_height, int input_width, int input_depth,
                  const T* filter_data, int filter_height, int filter_width,
                  int filter_count, int stride_rows, int stride_cols,
                  int pad_width, int pad_height, TfLitePadding padding,
                  T* output_data, int output_height, int output_width) {
    const Eigen::ThreadPoolDevice& device = GetThreadPoolDevice();

    const bool is_1x1_kernel = (filter_height == 1 && filter_width == 1 &&
                                stride_rows == 1 && stride_cols == 1);
    if (is_1x1_kernel) {
      // For 1x1 kernel, the 2D convolution is reduced to matrix
      // multiplication.
      const int conv_width = output_height * output_width;
      Eigen::array<Eigen::IndexPair<Eigen::DenseIndex>, 1> dim_pair;
      dim_pair[0] = Eigen::IndexPair<Eigen::DenseIndex>(1, 0);
      EigenMatrix output(output_data, input_batches * conv_width, filter_count);
      ConstEigenMatrix input(input_data, input_batches * conv_width,
                             input_depth);
      ConstEigenMatrix filter(filter_data, input_depth, filter_count);
      MatMulConvFunctor<Eigen::ThreadPoolDevice, T>()(device, output, input,
                                                      filter, dim_pair);
    } else if (filter_height == input_height && filter_width == input_width &&
               pad_width == 0 && pad_height == 0) {
      // If the input data and filter have the same height/width,
      // the 2D convolution is reduced to matrix multiplication.
      const int k =  // Length of reduction dimension.
          filter_width * filter_height * input_depth;
      Eigen::array<Eigen::IndexPair<Eigen::DenseIndex>, 1> dim_pair;
      dim_pair[0] = Eigen::IndexPair<Eigen::DenseIndex>(1, 0);
      EigenMatrix output(output_data, 1, filter_count);
      ConstEigenMatrix input(input_data, 1, k);
      ConstEigenMatrix filter(filter_data, k, filter_count);
      MatMulConvFunctor<Eigen::ThreadPoolDevice, T>()(device, output, input,
                                                      filter, dim_pair);
    } else {
      EigenTensor output(output_data, input_batches, output_height,
                         output_width, filter_count);
      ConstEigenTensor input(input_data, input_batches, input_height,
                             input_width, input_depth);
      ConstEigenTensor filter(filter_data, filter_height, filter_width,
                              input_depth, filter_count);
      output.device(device) =
          Eigen::SpatialConvolution(input, filter, stride_cols, stride_rows,
                                    TfLitePadding2EigenPadding(padding));
    }
  }
};

inline void Conv(const float* input_data, const Dims<4>& input_dims,
                 const float* filter_data, const Dims<4>& filter_dims,
                 const float* bias_data, const Dims<4>& bias_dims,
                 int stride_width, int stride_height, int pad_width,
                 int pad_height, TfLitePadding padding,
                 float output_activation_min, float output_activation_max,
                 float* output_data, const Dims<4>& output_dims,
                 float* im2col_data, const Dims<4>& im2col_dims) {
  const int batches = MatchingArraySize(input_dims, 3, output_dims, 3);
  const int input_depth = MatchingArraySize(input_dims, 0, filter_dims, 0);
  const int output_depth = MatchingArraySize(filter_dims, 3, output_dims, 0);
  const int input_height = ArraySize(input_dims, 2);
  const int input_width = ArraySize(input_dims, 1);
  const int filter_height = ArraySize(filter_dims, 2);
  const int filter_width = ArraySize(filter_dims, 1);
  const int output_height = ArraySize(output_dims, 2);
  const int output_width = ArraySize(output_dims, 1);
  EigenTensorConvFunctor<float> conv_functor;
  conv_functor(input_data, im2col_data, batches, input_height, input_width,
               input_depth, filter_data, filter_height, filter_width,
               output_depth, stride_height, stride_width, pad_height, pad_width,
               padding, output_data, output_height, output_width);

  optimized_ops::AddBiasAndEvalActivationFunction(
      bias_data, bias_dims, output_data, output_dims, output_activation_min,
      output_activation_max);
}

}  // namespace multithreaded_ops
}  // namespace tflite

#endif  // TENSORFLOW_CONTRIB_LITE_KERNELS_INTERNAL_MULTITHREAD_CONV
