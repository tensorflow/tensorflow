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

#ifndef TENSORFLOW_CORE_KERNELS_EIGEN_BENCHMARK_H_
#define TENSORFLOW_CORE_KERNELS_EIGEN_BENCHMARK_H_

#include "unsupported/Eigen/CXX11/Tensor"  // from @eigen_archive
#include "xla/tsl/framework/convolution/eigen_spatial_convolutions.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/kernels/eigen_backward_cuboid_convolutions.h"
#include "tensorflow/core/kernels/eigen_backward_spatial_convolutions.h"
#include "tensorflow/core/kernels/eigen_cuboid_convolution.h"
#include "tensorflow/core/platform/test_benchmark.h"

using ::tensorflow::TTypes;

template <typename Scalar, typename Device>
class SpatialConvolutionBenchmarksSuite {
 public:
  using Input = TTypes<float, 4>::ConstTensor;
  using Filter = TTypes<float, 4>::ConstTensor;
  using Output = TTypes<float, 4>::Tensor;

  using Dimensions = Eigen::DSizes<Eigen::Index, 4>;

  SpatialConvolutionBenchmarksSuite(::testing::benchmark::State& state,
                                    Device& device)
      : state_(state), device_(device) {}

  Eigen::Index BufferSize(const Dimensions& dims) {
    return dims.TotalSize() * sizeof(Scalar);
  }

  void SpatialConvolution(Dimensions input_dims, Dimensions filter_dims) {
    Dimensions output_dims(input_dims[0],    // batch
                           input_dims[1],    // input_height
                           input_dims[2],    // input_width
                           filter_dims[3]);  // filter_count

    Scalar* input_data =
        static_cast<Scalar*>(device_.allocate(BufferSize(input_dims)));
    Scalar* filter_data =
        static_cast<Scalar*>(device_.allocate(BufferSize(filter_dims)));
    Scalar* output_data =
        static_cast<Scalar*>(device_.allocate(BufferSize(output_dims)));

    device_.memset(input_data, 123, BufferSize(input_dims));
    device_.memset(filter_data, 123, BufferSize(filter_dims));

    Input input(input_data, input_dims);
    Filter filter(filter_data, filter_dims);
    Output output(output_data, output_dims);

    for (auto s : state_) {
      output.device(device_) = Eigen::SpatialConvolution(input, filter);
      tensorflow::testing::DoNotOptimize(output);
    }

    device_.deallocate(input_data);
    device_.deallocate(filter_data);
    device_.deallocate(output_data);
  }

  void SpatialConvolutionBackwardInput(Dimensions input_dims,
                                       Dimensions filter_dims) {
    using OutputBackward = TTypes<float, 4>::ConstTensor;
    using InputBackward = TTypes<float, 4>::Tensor;

    Dimensions output_dims(input_dims[0],    // batch
                           input_dims[1],    // input_height
                           input_dims[2],    // input_width
                           filter_dims[3]);  // filter_count

    // Assuming that the convolution had SAME padding.
    Eigen::Index input_rows = input_dims[1];
    Eigen::Index input_cols = input_dims[2];

    Scalar* filter_data =
        static_cast<Scalar*>(device_.allocate(BufferSize(filter_dims)));
    Scalar* output_backward_data =
        static_cast<Scalar*>(device_.allocate(BufferSize(output_dims)));
    Scalar* input_backward_data =
        static_cast<Scalar*>(device_.allocate(BufferSize(input_dims)));

    device_.memset(filter_data, 123, BufferSize(filter_dims));
    device_.memset(output_backward_data, 123, BufferSize(output_dims));

    Filter filter(filter_data, filter_dims);
    OutputBackward output_backward(output_backward_data, output_dims);
    InputBackward input_backward(input_backward_data, input_dims);

    for (auto s : state_) {
      input_backward.device(device_) = Eigen::SpatialConvolutionBackwardInput(
          filter, output_backward, input_rows, input_cols);
      tensorflow::testing::DoNotOptimize(input_backward);
    }

    device_.deallocate(filter_data);
    device_.deallocate(output_backward_data);
    device_.deallocate(input_backward_data);
  }

  void SpatialConvolutionBackwardKernel(Dimensions input_dims,
                                        Dimensions filter_dims) {
    using OutputBackward = TTypes<float, 4>::ConstTensor;
    using FilterBackward = TTypes<float, 4>::Tensor;

    Dimensions output_dims(input_dims[0],    // batch
                           input_dims[1],    // input_height
                           input_dims[2],    // input_width
                           filter_dims[3]);  // filter_count

    // Assuming that the convolution had SAME padding.
    Eigen::Index filter_rows = filter_dims[0];
    Eigen::Index filter_cols = filter_dims[1];

    Scalar* input_data =
        static_cast<Scalar*>(device_.allocate(BufferSize(input_dims)));
    Scalar* output_backward_data =
        static_cast<Scalar*>(device_.allocate(BufferSize(output_dims)));
    Scalar* filter_backward_data =
        static_cast<Scalar*>(device_.allocate(BufferSize(filter_dims)));

    device_.memset(input_data, 123, BufferSize(input_dims));
    device_.memset(output_backward_data, 123, BufferSize(output_dims));

    Input input(input_data, input_dims);
    OutputBackward output_backward(output_backward_data, input_dims);
    FilterBackward filter_backward(filter_backward_data, filter_dims);

    for (auto s : state_) {
      filter_backward.device(device_) = Eigen::SpatialConvolutionBackwardKernel(
          input, output_backward, filter_rows, filter_cols);
      tensorflow::testing::DoNotOptimize(filter_backward);
    }

    device_.deallocate(input_data);
    device_.deallocate(output_backward_data);
    device_.deallocate(filter_backward_data);
  }

 private:
  ::testing::benchmark::State& state_;

  Device& device_;
};

template <typename Scalar, typename Device>
class CuboidConvolutionBenchmarksSuite {
 public:
  using Input = TTypes<float, 5>::ConstTensor;
  using Filter = TTypes<float, 5>::ConstTensor;
  using Output = TTypes<float, 5>::Tensor;

  using Dimensions = Eigen::DSizes<Eigen::Index, 5>;

  CuboidConvolutionBenchmarksSuite(::testing::benchmark::State& state,
                                   Device& device)
      : state_(state), device_(device) {}

  Eigen::Index BufferSize(const Dimensions& dims) {
    return dims.TotalSize() * sizeof(Scalar);
  }

  void CuboidConvolution(Dimensions input_dims, Dimensions filter_dims) {
    Dimensions output_dims(input_dims[0],    // batch
                           input_dims[1],    // input_height
                           input_dims[2],    // input_width
                           input_dims[3],    // input_planes
                           filter_dims[4]);  // filter_count

    Scalar* input_data =
        static_cast<Scalar*>(device_.allocate(BufferSize(input_dims)));
    Scalar* filter_data =
        static_cast<Scalar*>(device_.allocate(BufferSize(filter_dims)));
    Scalar* output_data =
        static_cast<Scalar*>(device_.allocate(BufferSize(output_dims)));

    device_.memset(input_data, 123, BufferSize(input_dims));
    device_.memset(filter_data, 123, BufferSize(filter_dims));

    Input input(input_data, input_dims);
    Filter filter(filter_data, filter_dims);
    Output output(output_data, output_dims);

    for (auto s : state_) {
      output.device(device_) = Eigen::CuboidConvolution(input, filter);
      tensorflow::testing::DoNotOptimize(output);
    }

    device_.deallocate(input_data);
    device_.deallocate(filter_data);
    device_.deallocate(output_data);
  }

  void CuboidConvolutionBackwardInput(Dimensions input_dims,
                                      Dimensions filter_dims) {
    Dimensions output_dims(input_dims[0],    // batch
                           input_dims[1],    // input_height
                           input_dims[2],    // input_width
                           input_dims[3],    // input_planes
                           filter_dims[4]);  // filter_count

    using OutputBackward = TTypes<float, 5>::ConstTensor;
    using InputBackward = TTypes<float, 5>::Tensor;

    // Assuming that the convolution had SAME padding.
    Eigen::Index input_rows = input_dims[1];
    Eigen::Index input_cols = input_dims[2];
    Eigen::Index input_planes = input_dims[3];

    Scalar* filter_data =
        static_cast<Scalar*>(device_.allocate(BufferSize(filter_dims)));
    Scalar* output_backward_data =
        static_cast<Scalar*>(device_.allocate(BufferSize(output_dims)));
    Scalar* input_backward_data =
        static_cast<Scalar*>(device_.allocate(BufferSize(input_dims)));

    device_.memset(filter_data, 123, BufferSize(filter_dims));
    device_.memset(output_backward_data, 123, BufferSize(output_dims));

    Filter filter(filter_data, filter_dims);
    OutputBackward output_backward(output_backward_data, output_dims);
    InputBackward input_backward(input_backward_data, input_dims);

    for (auto s : state_) {
      input_backward.device(device_) = Eigen::CuboidConvolutionBackwardInput(
          filter, output_backward, input_planes, input_rows, input_cols);
      tensorflow::testing::DoNotOptimize(input_backward);
    }

    device_.deallocate(filter_data);
    device_.deallocate(output_backward_data);
    device_.deallocate(input_backward_data);
  }

  void CuboidConvolutionBackwardKernel(Dimensions input_dims,
                                       Dimensions filter_dims) {
    using OutputBackward = TTypes<float, 5>::ConstTensor;
    using FilterBackward = TTypes<float, 5>::Tensor;

    Dimensions output_dims(input_dims[0],    // batch
                           input_dims[1],    // input_height
                           input_dims[2],    // input_width
                           input_dims[3],    // input_planes
                           filter_dims[4]);  // filter_count

    // Assuming that the convolution had SAME padding.
    Eigen::Index filter_rows = filter_dims[0];
    Eigen::Index filter_cols = filter_dims[1];
    Eigen::Index filter_planes = filter_dims[2];

    Scalar* input_data =
        static_cast<Scalar*>(device_.allocate(BufferSize(input_dims)));
    Scalar* output_backward_data =
        static_cast<Scalar*>(device_.allocate(BufferSize(output_dims)));
    Scalar* filter_backward_data =
        static_cast<Scalar*>(device_.allocate(BufferSize(filter_dims)));

    device_.memset(input_data, 123, BufferSize(input_dims));
    device_.memset(output_backward_data, 123, BufferSize(output_dims));

    Input input(input_data, input_dims);
    OutputBackward output_backward(output_backward_data, output_dims);
    FilterBackward filter_backward(filter_backward_data, filter_dims);

    for (auto s : state_) {
      filter_backward.device(device_) = Eigen::CuboidConvolutionBackwardKernel(
          input, output_backward, filter_planes, filter_rows, filter_cols);
      tensorflow::testing::DoNotOptimize(filter_backward);
    }

    device_.deallocate(input_data);
    device_.deallocate(output_backward_data);
    device_.deallocate(filter_backward_data);
  }

 private:
  ::testing::benchmark::State& state_;
  Device& device_;
};

#endif  // TENSORFLOW_CORE_KERNELS_EIGEN_BENCHMARK_H_
