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

using Eigen::array;
using Eigen::DenseIndex;

template <typename Device, typename T>
class ProjectiveGenerator {
 private:
  typename TTypes<T, 4>::ConstTensor input_;
  typename TTypes<float>::ConstMatrix transforms_;

 public:
  static const int kNumParameters = 8;

  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE
  ProjectiveGenerator(typename TTypes<T, 4>::ConstTensor input,
                      typename TTypes<float>::ConstMatrix transforms)
      : input_(input), transforms_(transforms) {}

  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE T
  operator()(const array<DenseIndex, 4>& coords) const {
    array<DenseIndex, 4> input_coords;
    input_coords[0] = coords[0];

    const int64 output_y = coords[1];
    const int64 output_x = coords[2];
    const float* transform =
        transforms_.dimension(0) == 1
            ? transforms_.data()
            : &transforms_.data()[transforms_.dimension(1) * coords[0]];
    float projection = transform[6] * output_x + transform[7] * output_y + 1.f;
    const int64 input_x = std::round(
        (transform[0] * output_x + transform[1] * output_y + transform[2]) /
        projection);
    const int64 input_y = std::round(
        (transform[3] * output_x + transform[4] * output_y + transform[5]) /
        projection);

    if (!(0 <= input_y && input_y < input_.dimension(1) && 0 <= input_x &&
          input_x < input_.dimension(2))) {
      // TODO(ringwalt): Add a fill value input.
      return T(0);
    }
    input_coords[1] = input_y;
    input_coords[2] = input_x;

    input_coords[3] = coords[3];

    return input_(input_coords);
  }
};

}  // end namespace generator

// NOTE(ringwalt): We MUST wrap the generate() call in a functor and explicitly
// instantiate the functor in image_ops_gpu.cu.cc. Otherwise, we will be missing
// some Eigen device code.
namespace functor {

using generator::ProjectiveGenerator;

template <typename Device, typename T>
struct FillProjectiveTransform {
  typedef typename TTypes<T, 4>::Tensor OutputType;
  typedef typename TTypes<T, 4>::ConstTensor InputType;
  typedef typename TTypes<float, 2>::ConstTensor TransformsType;

  FillProjectiveTransform() {}

  EIGEN_ALWAYS_INLINE
  void operator()(const Device& device, OutputType* output,
                  const InputType& images,
                  const TransformsType& transform) const {
    ProjectiveGenerator<Device, T> generator(images, transform);
    output->device(device) = images.generate(generator);
  }
};

}  // end namespace functor

}  // end namespace tensorflow

#endif  // TENSORFLOW_CONTRIB_IMAGE_KERNELS_IMAGE_OPS_H_
