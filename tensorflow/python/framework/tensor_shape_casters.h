/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_PYTHON_FRAMEWORK_TENSOR_SHAPE_CASTERS_H_
#define TENSORFLOW_PYTHON_FRAMEWORK_TENSOR_SHAPE_CASTERS_H_

#if true  // go/pybind11_include_order
#include "pybind11/pybind11.h"  // from @pybind11
#endif

#include <cstdint>
#include <vector>

#include "pybind11/stl.h"  // from @pybind11
#include "tensorflow/core/framework/tensor_shape.h"

namespace pybind11 {
namespace detail {

template <>
struct type_caster<tensorflow::PartialTensorShape> {
  using vec_t = std::vector<std::int64_t>;
  using vec_caster_t = make_caster<vec_t>;

  static handle cast(const tensorflow::PartialTensorShape& src,
                     return_value_policy policy, handle parent) {
    vec_t dim_sizes(src.dims());
    for (int i = 0; i < src.dims(); i++) {
      dim_sizes[i] = src.dim_size(i);
    }
    return vec_caster_t::cast(dim_sizes, policy, parent);
  }

  bool load(handle src, bool convert) {
    vec_caster_t vec_caster;
    if (!vec_caster.load(src, convert)) {
      return false;
    }
    value = tensorflow::PartialTensorShape(cast_op<const vec_t&>(vec_caster));
    return true;
  }

  PYBIND11_TYPE_CASTER(tensorflow::PartialTensorShape,
                       const_name("tensorflow::PartialTensorShape"));
};

}  // namespace detail
}  // namespace pybind11

#endif  // TENSORFLOW_PYTHON_FRAMEWORK_TENSOR_SHAPE_CASTERS_H_
