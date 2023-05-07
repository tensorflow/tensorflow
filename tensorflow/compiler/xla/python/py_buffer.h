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

#ifndef TENSORFLOW_COMPILER_XLA_PYTHON_PY_BUFFER_H_
#define TENSORFLOW_COMPILER_XLA_PYTHON_PY_BUFFER_H_

#include <memory>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

#include "absl/strings/string_view.h"
#include "absl/synchronization/notification.h"
#include "pybind11/numpy.h"  // from @pybind11
#include "pybind11/pybind11.h"  // from @pybind11
#include "tensorflow/compiler/xla/python/ifrt/array.h"
#include "tensorflow/compiler/xla/python/pjrt_ifrt/pjrt_array.h"
#include "tensorflow/compiler/xla/python/py_client.h"
#include "tensorflow/compiler/xla/python/traceback.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/types.h"

namespace xla {

// TODO(parkers): Move everything in this file to a better home.
struct PyHostValue {
  static Status CopyToHostAsync(std::shared_ptr<PyHostValue>& host_value,
                                std::optional<Shape>& dynamic_shape_holder,
                                ifrt::Array* ifrt_array);

  static StatusOr<pybind11::object> AsNumPyArray(
      std::shared_ptr<PyHostValue>& host_value,
      std::optional<Shape>& dynamic_shape_holder, ifrt::Array* ifrt_array,
      pybind11::handle this_obj);

  absl::Notification ready;
  Status status;
  std::shared_ptr<xla::Literal> value;
};

struct IfrtHelpers {
  static StatusOr<const Shape*> xla_dynamic_shape(
      ifrt::Array* ifrt_array, std::optional<Shape>& scratch);
  static StatusOr<tsl::RCReference<ifrt::Array>> CopyToDevice(
      ifrt::Array* ifrt_array, PjRtDevice* dst_device);
  static PjRtBuffer* pjrt_buffer(ifrt::Array* ifrt_array);
  static PjRtDevice* pjrt_device(ifrt::Array* ifrt_array);
  static pybind11::tuple python_shape(ifrt::Array* ifrt_array);
  static pybind11::dtype python_dtype(ifrt::Array* ifrt_array);
  static StatusOr<pybind11::dict> CudaArrayInterface(
      ifrt::Array* ifrt_array, std::optional<Shape>& scratch);
};

// TODO(hyeontaek): Move the following functions to a separate file.
StatusOr<ifrt::DType> ToIfRtDType(pybind11::dtype dtype);
StatusOr<pybind11::dtype> ToPybind11DType(ifrt::DType dtype);

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_PYTHON_PY_BUFFER_H_
