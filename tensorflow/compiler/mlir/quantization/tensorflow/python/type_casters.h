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
#ifndef TENSORFLOW_COMPILER_MLIR_QUANTIZATION_TENSORFLOW_PYTHON_TYPE_CASTERS_H_
#define TENSORFLOW_COMPILER_MLIR_QUANTIZATION_TENSORFLOW_PYTHON_TYPE_CASTERS_H_

#include <string>
#include <utility>

#include "absl/strings/string_view.h"
#include "pybind11/cast.h"  // from @pybind11
#include "pybind11/detail/common.h"  // from @pybind11
#include "pybind11/pytypes.h"  // from @pybind11
#include "pybind11_abseil/absl_casters.h"  // from @pybind11_abseil  // IWYU pragma: keep
#include "tensorflow/compiler/mlir/quantization/tensorflow/exported_model.pb.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/quantization_options.pb.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"
#include "tensorflow/python/lib/core/pybind11_lib.h"
#include "tsl/platform/protobuf.h"

namespace pybind11::detail {
namespace internal {

// Serializes a protobuf object. Raises python ValueError if serialization
// fails.
std::string Serialize(const tsl::protobuf::Message& protobuf_object) {
  const std::string serialized = protobuf_object.SerializeAsString();

  // Empty string means it failed to serialize the protobuf with an error. See
  // the docstring for SerializeAsString for details.
  if (serialized.empty()) {
    throw py::value_error("Failed to serialize protobuf object.");
  }

  return serialized;
}

}  // namespace internal

// Handles `ExportedModel` (c++) <-> `bytes` (python) conversion. The `bytes`
// object in the python layer is a serialization of `ExportedModel`.
//
// See https://pybind11.readthedocs.io/en/stable/advanced/cast/custom.html for
// further details on how custom type conversions work for pybind11.
template <>
struct type_caster<tensorflow::quantization::ExportedModel> {
 public:
  PYBIND11_TYPE_CASTER(tensorflow::quantization::ExportedModel,
                       const_name("ExportedModel"));

  // Loads an `ExportedModel` instance from a python `bytes` object (`src`).
  bool load(handle src, const bool convert) {
    auto caster = make_caster<absl::string_view>();
    // Make sure the user passed a valid python string.
    if (!caster.load(src, convert)) {
      return false;
    }

    const absl::string_view exported_model_serialized =
        cast_op<absl::string_view>(std::move(caster));

    // NOLINTNEXTLINE: Explicit std::string conversion required for OSS.
    return value.ParseFromString(std::string(exported_model_serialized));
  }

  // Constructs a `bytes` object after serializing `src`.
  static handle cast(tensorflow::quantization::ExportedModel&& src,
                     return_value_policy policy, handle parent) {
    // release() prevents the reference count from decreasing upon the
    // destruction of py::bytes and returns a raw python object handle.
    return py::bytes(internal::Serialize(src)).release();
  }

  // Constructs a `bytes` object after serializing `src`.
  static handle cast(const tensorflow::quantization::ExportedModel& src,
                     return_value_policy policy, handle parent) {
    // release() prevents the reference count from decreasing upon the
    // destruction of py::bytes and returns a raw python object handle.
    return py::bytes(internal::Serialize(src)).release();
  }
};

// Handles type conversion for `QuantizationOptions`.
template <>
struct type_caster<tensorflow::quantization::QuantizationOptions> {
 public:
  PYBIND11_TYPE_CASTER(tensorflow::quantization::QuantizationOptions,
                       const_name("QuantizationOptions"));

  // Python -> C++. Converts a serialized protobuf string and deserializes into
  // an instance of `QuantizationOptions`.
  bool load(handle src, const bool convert) {
    auto caster = make_caster<absl::string_view>();
    // The user should have passed a valid python string.
    if (!caster.load(src, convert)) {
      return false;
    }

    const absl::string_view quantization_opts_serialized =
        cast_op<absl::string_view>(std::move(caster));

    // NOLINTNEXTLINE: Explicit std::string conversion required for OSS.
    return value.ParseFromString(std::string(quantization_opts_serialized));
  }

  // C++ -> Python. Constructs a `bytes` object after serializing `src`.
  static handle cast(const tensorflow::quantization::QuantizationOptions& src,
                     return_value_policy policy, handle parent) {
    // release() prevents the reference count from decreasing upon the
    // destruction of py::bytes and returns a raw python object handle.
    return py::bytes(internal::Serialize(src)).release();
  }
};

template <>
struct type_caster<tensorflow::SignatureDef> {
 public:
  PYBIND11_TYPE_CASTER(tensorflow::SignatureDef, const_name("SignatureDef"));

  // Python->C++ conversion. Accepts a serialized `SignatureDef` string from the
  // python side.
  bool load(handle src, const bool convert) {
    auto caster = make_caster<absl::string_view>();
    if (!caster.load(src, convert)) return false;

    const absl::string_view signature_def_serialized =
        cast_op<absl::string_view>(std::move(caster));

    // NOLINTNEXTLINE: Explicit std::string conversion required for OSS.
    return value.ParseFromString(std::string(signature_def_serialized));
  }

  // C++->Python conversion. Returns a serialized `SignatureDef` string.
  static handle cast(const tensorflow::SignatureDef& src,
                     return_value_policy policy, handle parent) {
    return py::bytes(internal::Serialize(src)).release();
  }
};

}  // namespace pybind11::detail

#endif  // TENSORFLOW_COMPILER_MLIR_QUANTIZATION_TENSORFLOW_PYTHON_TYPE_CASTERS_H_
