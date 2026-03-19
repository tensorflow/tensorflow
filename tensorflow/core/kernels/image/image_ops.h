/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_KERNELS_IMAGE_IMAGE_OPS_H_
#define TENSORFLOW_CORE_KERNELS_IMAGE_IMAGE_OPS_H_

// See docs in ../ops/image_ops.cc.

#define EIGEN_USE_THREADS

#include "unsupported/Eigen/CXX11/Tensor"  // from @eigen_archive

#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

namespace generator {

enum Interpolation { NEAREST, BILINEAR };
enum Mode { FILL_REFLECT, FILL_WRAP, FILL_CONSTANT, FILL_NEAREST };

using Eigen::array;
using Eigen::DenseIndex;

// Follow scipy's implementation
// https://github.com/scipy/scipy/blob/master/scipy/ndimage/src/ni_interpolation.c
template <typename Device, Mode M>
struct MapCoordinate {
  float operator()(const float out_coord, const DenseIndex len);
};

template <typename Device>
struct MapCoordinate<Device, Mode::FILL_REFLECT> {
  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE float operator()(const float out_coord,
                                                         const DenseIndex len) {
    // Reflect [abcd] to [dcba|abcd|dcba].
    float in_coord = out_coord;
    if (in_coord < 0) {
      if (len <= 1) {
        in_coord = 0;
      } else {
        const DenseIndex sz2 = 2 * len;
        if (in_coord < sz2) {
          in_coord = sz2 * static_cast<DenseIndex>(-in_coord / sz2) + in_coord;
        }
        in_coord = (in_coord < -len) ? in_coord + sz2 : -in_coord - 1;
      }
    } else if (in_coord > len - 1) {
      if (len <= 1) {
        in_coord = 0;
      } else {
        const DenseIndex sz2 = 2 * len;
        in_coord -= sz2 * static_cast<DenseIndex>(in_coord / sz2);
        if (in_coord >= len) {
          in_coord = sz2 - in_coord - 1;
        }
      }
    }
    // clamp is necessary because when out_coord = 3.5 and len = 4,
    // in_coord = 3.5 and will be rounded to 4 in nearest interpolation.
    return Eigen::internal::scalar_clamp_op<float>(0.0f, len - 1)(in_coord);
  }
};

template <typename Device>
struct MapCoordinate<Device, Mode::FILL_WRAP> {
  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE float operator()(const float out_coord,
                                                         const DenseIndex len) {
    // Wrap [abcd] to [abcd|abcd|abcd].
    float in_coord = out_coord;
    if (in_coord < 0) {
      if (len <= 1) {
        in_coord = 0;
      } else {
        const DenseIndex sz = len - 1;
        in_coord += len * (static_cast<DenseIndex>(-in_coord / sz) + 1);
      }
    } else if (in_coord > len - 1) {
      if (len <= 1) {
        in_coord = 0;
      } else {
        const DenseIndex sz = len - 1;
        in_coord -= len * static_cast<DenseIndex>(in_coord / sz);
      }
    }
    // clamp is necessary because when out_coord = -0.5 and len = 4,
    // in_coord = 3.5 and will be rounded to 4 in nearest interpolation.
    return Eigen::internal::scalar_clamp_op<float>(0.0f, len - 1)(in_coord);
  }
};

template <typename Device>
struct MapCoordinate<Device, Mode::FILL_CONSTANT> {
  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE float operator()(const float out_coord,
                                                         const DenseIndex len) {
    return out_coord;
  }
};

template <typename Device>
struct MapCoordinate<Device, Mode::FILL_NEAREST> {
  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE float operator()(const float out_coord,
                                                         const DenseIndex len) {
    return Eigen::internal::scalar_clamp_op<float>(0.0f, len - 1)(out_coord);
  }
};

template <typename Device, typename T, Mode M>
class ProjectiveGenerator {
 private:
  typename TTypes<T, 4>::ConstTensor input_;
  typename TTypes<float>::ConstMatrix transforms_;
  const Interpolation interpolation_;
  const T fill_value_;

 public:
  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE
  ProjectiveGenerator(typename TTypes<T, 4>::ConstTensor input,
                      typename TTypes<float>::ConstMatrix transforms,
                      const Interpolation interpolation, const T fill_value)
      : input_(input),
        transforms_(transforms),
        interpolation_(interpolation),
        fill_value_(fill_value) {}

  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE T
  operator()(const array<DenseIndex, 4>& coords) const {
    const int64_t output_y = coords[1];
    const int64_t output_x = coords[2];
    const float* transform =
        transforms_.dimension(0) == 1
            ? transforms_.data()
            : &transforms_.data()[transforms_.dimension(1) * coords[0]];
    float projection = transform[6] * output_x + transform[7] * output_y + 1.f;
    if (projection == 0) {
      // Return the fill value for infinite coordinates,
      // which are outside the input image
      return fill_value_;
    }
    const float input_x =
        (transform[0] * output_x + transform[1] * output_y + transform[2]) /
        projection;
    const float input_y =
        (transform[3] * output_x + transform[4] * output_y + transform[5]) /
        projection;

    // Map out-of-boundary input coordinates to in-boundary based on fill_mode.
    auto map_functor = MapCoordinate<Device, M>();
    const float x = map_functor(input_x, input_.dimension(2));
    const float y = map_functor(input_y, input_.dimension(1));

    const DenseIndex batch = coords[0];
    const DenseIndex channels = coords[3];
    switch (interpolation_) {
      case NEAREST:
        return nearest_interpolation(batch, y, x, channels, fill_value_);
      case BILINEAR:
        return bilinear_interpolation(batch, y, x, channels, fill_value_);
    }
    // Unreachable; ImageProjectiveTransform only uses INTERPOLATION_NEAREST
    // or INTERPOLATION_BILINEAR.
    return fill_value_;
  }

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

namespace functor {

using generator::Interpolation;
using generator::Mode;
using generator::ProjectiveGenerator;

template <typename Device, typename T>
struct FillProjectiveTransform {
  typedef typename TTypes<T, 4>::Tensor OutputType;
  typedef typename TTypes<T, 4>::ConstTensor InputType;
  typedef typename TTypes<float, 2>::ConstTensor TransformsType;
  const Interpolation interpolation;

  explicit FillProjectiveTransform(Interpolation interpolation)
      : interpolation(interpolation) {}

  EIGEN_ALWAYS_INLINE
  void operator()(const Device& device, OutputType* output,
                  const InputType& images, const TransformsType& transform,
                  const Mode fill_mode, const T fill_value) const {
    switch (fill_mode) {
      case Mode::FILL_REFLECT:
        output->device(device) =
            output->generate(ProjectiveGenerator<Device, T, Mode::FILL_REFLECT>(
                images, transform, interpolation, fill_value));
        break;
      case Mode::FILL_WRAP:
        output->device(device) =
            output->generate(ProjectiveGenerator<Device, T, Mode::FILL_WRAP>(
                images, transform, interpolation, fill_value));
        break;
      case Mode::FILL_CONSTANT:
        output->device(device) = output->generate(
            ProjectiveGenerator<Device, T, Mode::FILL_CONSTANT>(
                images, transform, interpolation, fill_value));
        break;
      case Mode::FILL_NEAREST:
        output->device(device) =
            output->generate(ProjectiveGenerator<Device, T, Mode::FILL_NEAREST>(
                images, transform, interpolation, fill_value));
        break;
    }
  }
};

}  // end namespace functor

}  // end namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_IMAGE_IMAGE_OPS_H_
