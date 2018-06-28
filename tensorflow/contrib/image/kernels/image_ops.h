/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CONTRIB_IMAGE_KERNELS_IMAGE_OPS_H_
#define TENSORFLOW_CONTRIB_IMAGE_KERNELS_IMAGE_OPS_H_

// See docs in ../ops/image_ops.cc.

#define EIGEN_USE_THREADS

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

namespace generator {

enum Interpolation { INTERPOLATION_NEAREST, INTERPOLATION_BILINEAR };

using Eigen::array;
using Eigen::DenseIndex;

template <typename Device, typename T>
class ProjectiveGenerator {
 private:
  typename TTypes<T, 4>::ConstTensor input_;
  typename TTypes<float>::ConstMatrix transforms_;
  const Interpolation interpolation_;

 public:
  static const int kNumParameters = 8;

  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE
  ProjectiveGenerator(typename TTypes<T, 4>::ConstTensor input,
                      typename TTypes<float>::ConstMatrix transforms,
                      const Interpolation interpolation)
      : input_(input), transforms_(transforms), interpolation_(interpolation) {}

  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE T
  operator()(const array<DenseIndex, 4>& coords) const {
    const int64 output_y = coords[1];
    const int64 output_x = coords[2];
    const float* transform =
        transforms_.dimension(0) == 1
            ? transforms_.data()
            : &transforms_.data()[transforms_.dimension(1) * coords[0]];
    float projection = transform[6] * output_x + transform[7] * output_y + 1.f;
    if (projection == 0) {
      // Return the fill value (0) for infinite coordinates,
      // which are outside the input image
      return T(0);
    }
    const float input_x =
        (transform[0] * output_x + transform[1] * output_y + transform[2]) /
        projection;
    const float input_y =
        (transform[3] * output_x + transform[4] * output_y + transform[5]) /
        projection;

    // TODO(ringwalt): Add a fill value input.
#if (defined __CUDA_ARCH__) && (CUDART_VERSION < 8000)
    // On CUDA versions previous to 8.0, only __shared__ variables
    // could be declared as static in the device code.
    const T fill_value = T(0);
#else
    static const T fill_value = T(0);
#endif
    switch (interpolation_) {
      case INTERPOLATION_NEAREST:
        // Switch the order of x and y again for indexing into the image.
        return nearest_interpolation(coords[0], input_y, input_x, coords[3],
                                     fill_value);
      case INTERPOLATION_BILINEAR:
        return bilinear_interpolation(coords[0], input_y, input_x, coords[3],
                                      fill_value);
    }
    // Unreachable; ImageProjectiveTransform only uses INTERPOLATION_NEAREST
    // or INTERPOLATION_BILINEAR.
    return T(0);
  }

 private:
  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE T
  nearest_interpolation(const DenseIndex batch, const float y, const float x,
                        const DenseIndex channel, const T fill_value) const {
    return read_with_fill_value(batch, DenseIndex(std::round(y)),
                                DenseIndex(std::round(x)), channel, fill_value);
  }

  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE T
  bilinear_interpolation(const DenseIndex batch, const float y, const float x,
                         const DenseIndex channel, const T fill_value) const {
    const float y_floor = std::floor(y);
    const float x_floor = std::floor(x);
    const float y_ceil = y_floor + 1;
    const float x_ceil = x_floor + 1;
    // f(x, y_floor) = (x_ceil - x) / (x_ceil - x_floor) * f(x_floor, y_floor)
    //               + (x - x_floor) / (x_ceil - x_floor) * f(x_ceil, y_floor)
    const float value_yfloor =
        (x_ceil - x) * static_cast<float>(read_with_fill_value(
                           batch, DenseIndex(y_floor), DenseIndex(x_floor),
                           channel, fill_value)) +
        (x - x_floor) * static_cast<float>(read_with_fill_value(
                            batch, DenseIndex(y_floor), DenseIndex(x_ceil),
                            channel, fill_value));
    // f(x, y_ceil) = (x_ceil - x) / (x_ceil - x_floor) * f(x_floor, y_ceil)
    //              + (x - x_floor) / (x_ceil - x_floor) * f(x_ceil, y_ceil)
    const float value_yceil =
        (x_ceil - x) * static_cast<float>(read_with_fill_value(
                           batch, DenseIndex(y_ceil), DenseIndex(x_floor),
                           channel, fill_value)) +
        (x - x_floor) * static_cast<float>(read_with_fill_value(
                            batch, DenseIndex(y_ceil), DenseIndex(x_ceil),
                            channel, fill_value));
    // f(x, y) = (y_ceil - y) / (y_ceil - y_floor) * f(x, y_floor)
    //         + (y - y_floor) / (y_ceil - y_floor) * f(x, y_ceil)
    return T((y_ceil - y) * value_yfloor + (y - y_floor) * value_yceil);
  }

  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE T read_with_fill_value(
      const DenseIndex batch, const DenseIndex y, const DenseIndex x,
      const DenseIndex channel, const T fill_value) const {
    // batch and channel must be correct, because they are passed unchanged from
    // the input.
    return (0 <= y && y < input_.dimension(1) && 0 <= x &&
            x < input_.dimension(2))
               ? input_(array<DenseIndex, 4>{batch, y, x, channel})
               : fill_value;
  }
};

}  // end namespace generator

// NOTE(ringwalt): We MUST wrap the generate() call in a functor and explicitly
// instantiate the functor in image_ops_gpu.cu.cc. Otherwise, we will be missing
// some Eigen device code.
namespace functor {

using generator::Interpolation;
using generator::ProjectiveGenerator;

template <typename Device, typename T>
struct FillProjectiveTransform {
  typedef typename TTypes<T, 4>::Tensor OutputType;
  typedef typename TTypes<T, 4>::ConstTensor InputType;
  typedef typename TTypes<float, 2>::ConstTensor TransformsType;
  const Interpolation interpolation_;

  FillProjectiveTransform(Interpolation interpolation)
      : interpolation_(interpolation) {}

  EIGEN_ALWAYS_INLINE
  void operator()(const Device& device, OutputType* output,
                  const InputType& images,
                  const TransformsType& transform) const {
    output->device(device) = images.generate(
        ProjectiveGenerator<Device, T>(images, transform, interpolation_));
  }
};

}  // end namespace functor

}  // end namespace tensorflow

#endif  // TENSORFLOW_CONTRIB_IMAGE_KERNELS_IMAGE_OPS_H_
