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
#include <type_traits>
#include <utility>

#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "pybind11/cast.h"  // from @pybind11
#include "pybind11/detail/common.h"  // from @pybind11
#include "pybind11/pytypes.h"  // from @pybind11
#include "pybind11_abseil/absl_casters.h"  // from @pybind11_abseil  // IWYU pragma: keep
#include "tensorflow/compiler/mlir/quantization/stablehlo/quantization_config.pb.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/calibrator/calibration_statistics.pb.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/exported_model.pb.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/quantization_options.pb.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"
#include "tensorflow/python/lib/core/pybind11_lib.h"
#include "tsl/platform/protobuf.h"  // IWYU pragma: keep

namespace pybind11::detail {
namespace internal {

// Serializes a protobuf object. Raises python ValueError if serialization
// fails.
inline std::string Serialize(const tsl::protobuf::Message& protobuf_object) {
  const std::string serialized = protobuf_object.SerializeAsString();

  // Empty string means it failed to serialize the protobuf with an error. See
  // the docstring for SerializeAsString for details.
  if (serialized.empty()) {
    // Show the name of the protobuf message type to provide more information
    // and easier debugging.
    const std::string descriptor_name =
        protobuf_object.GetDescriptor() == nullptr
            ? "unknown"
            : protobuf_object.GetDescriptor()->full_name();
    throw py::value_error(absl::StrFormat(
        "Failed to serialize protobuf object: %s.", descriptor_name));
  }

  return serialized;
}

// Handles `ProtoT` (c++) <-> `bytes` (python) conversion. The `bytes`
// object in the python layer is a serialization of `ProtoT`.
//
// The caller of c++ interfaces should make sure to pass valid serialized
// `ProtoT` objects as arguments. Failing to do so results in raising a
// `ValueError`. Similarly, the python implementation of a c++ virtual member
// function that return an `ProtoT` should return a valid serialized `ProtoT`.
//
// See https://pybind11.readthedocs.io/en/stable/advanced/cast/custom.html
template <typename ProtoT, typename = std::enable_if_t<std::is_base_of_v<
                               tsl::protobuf::Message, ProtoT>>>
struct SerializedProtobufCaster {
 public:
  PYBIND11_TYPE_CASTER(ProtoT, const_name<ProtoT>());

  // Loads an `ProtoT` instance from a python `bytes` object (`src`).
  bool load(handle src, const bool convert) {
    auto caster = make_caster<absl::string_view>();
    // Make sure the user passed a valid python string.
    if (!caster.load(src, convert)) return false;

    const absl::string_view serialized_proto =
        cast_op<absl::string_view>(std::move(caster));

    // NOLINTNEXTLINE: Explicit std::string conversion required for OSS.
    return value.ParseFromString(std::string(serialized_proto));
  }

  // Constructs a `bytes` object by serializing `src`.
  static handle cast(ProtoT&& src, return_value_policy policy, handle parent) {
    // release() prevents the reference count from decreasing upon the
    // destruction of py::bytes and returns a raw python object handle.
    return py::bytes(Serialize(src)).release();
  }

  // Constructs a `bytes` object by serializing `src`.
  static handle cast(const ProtoT& src, return_value_policy policy,
                     handle parent) {
    // release() prevents the reference count from decreasing upon the
    // destruction of py::bytes and returns a raw python object handle.
    return py::bytes(Serialize(src)).release();
  }
};

}  // namespace internal

// The following explicit specializations of protobuf `type_caster`s for
// specific protobuf message types are there to have higher priority over those
// defined in `native_proto_caster.h` during the resolution process. This is
// because the type casters in `native_proto_caster.h`, which allow seamlessly
// exchanging protobuf messages across c++-python boundaries, potentially
// without serialization, fail in the open-source environment.
// Explicitly-specialized type casters for serialized protobufs are added on an
// on-demand basis for quantization library.
// TODO: b/308532051 - Make `native_proto_caster.h` work in the open-source
// environment.

template <>
struct type_caster<tensorflow::quantization::ExportedModel>
    : public internal::SerializedProtobufCaster<
          tensorflow::quantization::ExportedModel> {};

template <>
struct type_caster<tensorflow::quantization::QuantizationOptions>
    : public internal::SerializedProtobufCaster<
          tensorflow::quantization::QuantizationOptions> {};

template <>
struct type_caster<::stablehlo::quantization::CalibrationOptions>
    : public internal::SerializedProtobufCaster<
          ::stablehlo::quantization::CalibrationOptions> {};

template <>
struct type_caster<tensorflow::SignatureDef>
    : public internal::SerializedProtobufCaster<tensorflow::SignatureDef> {};

template <>
struct type_caster<tensorflow::GraphDef>
    : public internal::SerializedProtobufCaster<tensorflow::GraphDef> {};

template <>
struct type_caster<tensorflow::calibrator::CalibrationStatistics>
    : public internal::SerializedProtobufCaster<
          tensorflow::calibrator::CalibrationStatistics> {};

template <>
struct type_caster<stablehlo::quantization::QuantizationConfig>
    : public internal::SerializedProtobufCaster<
          stablehlo::quantization::QuantizationConfig> {};

template <>
struct type_caster<tensorflow::quantization::RepresentativeDatasetFile>
    : public internal::SerializedProtobufCaster<
          tensorflow::quantization::RepresentativeDatasetFile> {};

}  // namespace pybind11::detail

#endif  // TENSORFLOW_COMPILER_MLIR_QUANTIZATION_TENSORFLOW_PYTHON_TYPE_CASTERS_H_
