/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_PYTHON_TYPES_H_
#define TENSORFLOW_COMPILER_XLA_PYTHON_TYPES_H_

#include <memory>
#include <vector>

#include "absl/container/inlined_vector.h"
#include "absl/types/optional.h"
#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "tensorflow/compiler/xla/python/absl_casters.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/shape.h"
#include "tensorflow/compiler/xla/status.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/platform/protobuf.h"

namespace xla {

// Helper that converts a failing StatusOr to an exception.
// For use only inside pybind11 code.
template <typename T>
T ValueOrThrow(StatusOr<T> v) {
  if (!v.ok()) {
    throw std::runtime_error(v.status().ToString());
  }
  return v.ConsumeValueOrDie();
}

// Converts a NumPy dtype to a PrimitiveType.
StatusOr<PrimitiveType> DtypeToPrimitiveType(const pybind11::dtype& np_type);

// Converts a PrimitiveType to a Numpy dtype.
StatusOr<pybind11::dtype> PrimitiveTypeToDtype(PrimitiveType type);

// Returns a numpy-style format descriptor string for `type`.
StatusOr<std::string> FormatDescriptorForPrimitiveType(PrimitiveType type);

// Returns a numpy-style typestr for `type`, as returned by np.dtype(...).str
StatusOr<pybind11::str> TypeDescriptorForPrimitiveType(PrimitiveType type);

struct NumpyScalarTypes {
  pybind11::object np_bool;
  pybind11::object np_int8;
  pybind11::object np_int16;
  pybind11::object np_int32;
  pybind11::object np_int64;
  pybind11::object np_uint8;
  pybind11::object np_uint16;
  pybind11::object np_uint32;
  pybind11::object np_uint64;
  pybind11::object np_bfloat16;
  pybind11::object np_float16;
  pybind11::object np_float32;
  pybind11::object np_float64;
  pybind11::object np_complex64;
  pybind11::object np_complex128;
  pybind11::object np_longlong;
  pybind11::object np_intc;
};
const NumpyScalarTypes& GetNumpyScalarTypes();

// For S64/U64/F64/C128 types, returns the largest 32-bit equivalent.
PrimitiveType Squash64BitTypes(PrimitiveType type);

// Returns the strides for `shape`.
std::vector<ssize_t> ByteStridesForShape(const Shape& shape);

// Converts a literal to (possibly-nested tuples of) NumPy arrays.
// The literal's leaf arrays are not copied; instead the NumPy arrays share
// buffers with the literals. Takes ownership of `literal` and keeps the
// necessary pieces alive using Python reference counting.
// Requires the GIL.
StatusOr<pybind11::object> LiteralToPython(std::shared_ptr<Literal> literal);

// Converts a Python object into an XLA shape and a vector of leaf buffers.
// The leaf buffers correspond to a depth-first, left-to-right traversal of
// the Python value.
// Requires the GIL.
struct PythonBufferTree {
  // Holds a reference to the arrays pointed to by `leaves`, since we may
  // need to make a copy if the array is not in a C-style layout.
  absl::InlinedVector<pybind11::object, 1> arrays;
  absl::InlinedVector<BorrowingLiteral, 1> leaves;
  Shape shape;
};
StatusOr<PythonBufferTree> GetPythonBufferTree(
    const pybind11::object& argument);

// Converts a sequence of C++ ints to a Python tuple of ints.
// Pybind11 by default converts a std::vector<int64> to a Python list;
// we frequently want a tuple instead e.g. for shapes.
pybind11::tuple IntSpanToTuple(absl::Span<int64 const> xs);
pybind11::tuple IntSpanToTuple(absl::Span<int const> xs);

// Converts a Python sequence of integers to a std::vector<int64>
std::vector<int64> IntSequenceToVector(const pybind11::object& sequence);

// Private helper function used in the implementation of the type caster for
// xla::BorrowingLiteral. Converts a Python array-like object into a buffer
// pointer and shape.
struct CastToArrayResult {
  pybind11::object array;  // Holds a reference to the array to keep it alive.
  const char* buf_ptr;
  xla::Shape shape;
};
absl::optional<CastToArrayResult> CastToArray(pybind11::handle h);

}  // namespace xla

// This namespace is a documented pybind11 extension point.
// Caution: Unusually for Google code, this code uses C++ exceptions because
// they are the only mechanism for reporting cast failures to pybind11. However,
// the exceptions are local to the binding code.
namespace pybind11 {
namespace detail {

// Status, StatusOr. Failing statuses become Python exceptions; Status::OK()
// becomes None.
template <>
struct type_caster<xla::Status> {
 public:
  PYBIND11_TYPE_CASTER(xla::Status, _("Status"));

  static handle cast(xla::Status src, return_value_policy /* policy */,
                     handle /* parent */) {
    if (!src.ok()) {
      throw std::runtime_error(src.ToString());
    }
    return none().inc_ref();
  }
};

template <typename T>
struct type_caster<xla::StatusOr<T>> {
 public:
  using value_conv = make_caster<T>;

  PYBIND11_TYPE_CASTER(xla::StatusOr<T>,
                       _("StatusOr[") + value_conv::name + _("]"));

  static handle cast(xla::StatusOr<T> src, return_value_policy policy,
                     handle parent) {
    if (!src.ok()) {
      throw std::runtime_error(src.status().ToString());
    }
    return value_conv::cast(std::forward<xla::StatusOr<T>>(src).ValueOrDie(),
                            policy, parent);
  }
};

// Literals.
// Literal data can be passed to XLA as a NumPy array; its value can be
// cast to an xla::BorrowingLiteral or xla::LiteralSlice in a zero-copy way.
// We don't have any literal -> numpy conversions here, since all the methods
// that want to return arrays build Python objects directly.

template <>
struct type_caster<xla::BorrowingLiteral> {
 public:
  PYBIND11_TYPE_CASTER(xla::BorrowingLiteral, _("xla::BorrowingLiteral"));

  // Pybind appears to keep type_casters alive until the callee has run.
  absl::InlinedVector<pybind11::array, 1> arrays;

  bool load(handle input, bool) {
    // TODO(b/79707221): support nested tuples if/when XLA adds support for
    // nested BorrowingLiterals.
    if (pybind11::isinstance<pybind11::tuple>(input)) {
      pybind11::tuple tuple =
          pybind11::reinterpret_borrow<pybind11::tuple>(input);
      std::vector<xla::Shape> shapes;
      std::vector<const char*> buffers;
      arrays.reserve(tuple.size());
      shapes.reserve(tuple.size());
      buffers.reserve(tuple.size());
      for (pybind11::handle entry : tuple) {
        auto c = xla::CastToArray(entry);
        if (!c) {
          return false;
        }
        arrays.push_back(c->array);
        buffers.push_back(c->buf_ptr);
        shapes.push_back(c->shape);
      }
      value = xla::BorrowingLiteral(buffers,
                                    xla::ShapeUtil::MakeTupleShape(shapes));
    } else {
      auto c = xla::CastToArray(input);
      if (!c) {
        return false;
      }
      arrays.push_back(c->array);
      value = xla::BorrowingLiteral(c->buf_ptr, c->shape);
    }
    return true;
  }
};

template <>
struct type_caster<xla::LiteralSlice> {
 public:
  PYBIND11_TYPE_CASTER(xla::LiteralSlice, _("xla::LiteralSlice"));

  // Pybind appears to keep type_casters alive until the callee has run.
  type_caster<xla::BorrowingLiteral> literal_caster;

  bool load(handle handle, bool convert) {
    if (!literal_caster.load(handle, convert)) {
      return false;
    }
    value = static_cast<const xla::BorrowingLiteral&>(literal_caster);
    return true;
  }
};

// XLA protocol buffers
// We don't actually care that these are the protocol buffers, we merely want
// objects that duck type as protocol buffers. The client code currently avoids
// depending on Python protocol buffers to avoid conflicting definitions from
// different modules that both include XLA.

template <>
struct type_caster<xla::ConvolutionDimensionNumbers> {
 public:
  PYBIND11_TYPE_CASTER(xla::ConvolutionDimensionNumbers,
                       _("xla::ConvolutionDimensionNumbers"));

  // PyObject -> C++ conversion.
  bool load(handle handle, bool) {
    value.set_input_batch_dimension(
        getattr(handle, "input_batch_dimension").cast<xla::int64>());
    value.set_input_feature_dimension(
        getattr(handle, "input_feature_dimension").cast<xla::int64>());
    value.set_output_batch_dimension(
        getattr(handle, "output_batch_dimension").cast<xla::int64>());
    value.set_output_feature_dimension(
        getattr(handle, "output_feature_dimension").cast<xla::int64>());
    value.set_kernel_input_feature_dimension(
        getattr(handle, "kernel_input_feature_dimension").cast<xla::int64>());
    value.set_kernel_output_feature_dimension(
        getattr(handle, "kernel_output_feature_dimension").cast<xla::int64>());
    std::vector<xla::int64> dims;
    dims = getattr(handle, "input_spatial_dimensions")
               .cast<std::vector<xla::int64>>();
    std::copy(dims.begin(), dims.end(),
              tensorflow::protobuf::RepeatedFieldBackInserter(
                  value.mutable_input_spatial_dimensions()));
    dims = getattr(handle, "kernel_spatial_dimensions")
               .cast<std::vector<xla::int64>>();
    std::copy(dims.begin(), dims.end(),
              tensorflow::protobuf::RepeatedFieldBackInserter(
                  value.mutable_kernel_spatial_dimensions()));
    dims = getattr(handle, "output_spatial_dimensions")
               .cast<std::vector<xla::int64>>();
    std::copy(dims.begin(), dims.end(),
              tensorflow::protobuf::RepeatedFieldBackInserter(
                  value.mutable_output_spatial_dimensions()));
    return true;
  }
};

template <>
struct type_caster<xla::DotDimensionNumbers> {
 public:
  PYBIND11_TYPE_CASTER(xla::DotDimensionNumbers, _("xla::DotDimensionNumbers"));

  // PyObject -> C++ conversion.
  bool load(handle handle, bool) {
    std::vector<xla::int64> dims;
    dims = getattr(handle, "lhs_contracting_dimensions")
               .cast<std::vector<xla::int64>>();
    std::copy(dims.begin(), dims.end(),
              tensorflow::protobuf::RepeatedFieldBackInserter(
                  value.mutable_lhs_contracting_dimensions()));
    dims = getattr(handle, "rhs_contracting_dimensions")
               .cast<std::vector<xla::int64>>();
    std::copy(dims.begin(), dims.end(),
              tensorflow::protobuf::RepeatedFieldBackInserter(
                  value.mutable_rhs_contracting_dimensions()));
    dims =
        getattr(handle, "lhs_batch_dimensions").cast<std::vector<xla::int64>>();
    std::copy(dims.begin(), dims.end(),
              tensorflow::protobuf::RepeatedFieldBackInserter(
                  value.mutable_lhs_batch_dimensions()));
    dims =
        getattr(handle, "rhs_batch_dimensions").cast<std::vector<xla::int64>>();
    std::copy(dims.begin(), dims.end(),
              tensorflow::protobuf::RepeatedFieldBackInserter(
                  value.mutable_rhs_batch_dimensions()));
    return true;
  }
};

template <>
struct type_caster<xla::GatherDimensionNumbers> {
 public:
  PYBIND11_TYPE_CASTER(xla::GatherDimensionNumbers,
                       _("xla::GatherDimensionNumbers"));

  // PyObject -> C++ conversion.
  bool load(handle handle, bool) {
    std::vector<xla::int64> dims;
    dims = getattr(handle, "offset_dims").cast<std::vector<xla::int64>>();
    std::copy(dims.begin(), dims.end(),
              tensorflow::protobuf::RepeatedFieldBackInserter(
                  value.mutable_offset_dims()));
    dims =
        getattr(handle, "collapsed_slice_dims").cast<std::vector<xla::int64>>();
    std::copy(dims.begin(), dims.end(),
              tensorflow::protobuf::RepeatedFieldBackInserter(
                  value.mutable_collapsed_slice_dims()));
    dims = getattr(handle, "start_index_map").cast<std::vector<xla::int64>>();
    std::copy(dims.begin(), dims.end(),
              tensorflow::protobuf::RepeatedFieldBackInserter(
                  value.mutable_start_index_map()));
    value.set_index_vector_dim(
        getattr(handle, "index_vector_dim").cast<xla::int64>());
    return true;
  }
};

template <>
struct type_caster<xla::ScatterDimensionNumbers> {
 public:
  PYBIND11_TYPE_CASTER(xla::ScatterDimensionNumbers,
                       _("xla::ScatterDimensionNumbers"));

  // PyObject -> C++ conversion.
  bool load(handle handle, bool) {
    std::vector<xla::int64> dims;
    dims =
        getattr(handle, "update_window_dims").cast<std::vector<xla::int64>>();
    std::copy(dims.begin(), dims.end(),
              tensorflow::protobuf::RepeatedFieldBackInserter(
                  value.mutable_update_window_dims()));
    dims =
        getattr(handle, "inserted_window_dims").cast<std::vector<xla::int64>>();
    std::copy(dims.begin(), dims.end(),
              tensorflow::protobuf::RepeatedFieldBackInserter(
                  value.mutable_inserted_window_dims()));
    dims = getattr(handle, "scatter_dims_to_operand_dims")
               .cast<std::vector<xla::int64>>();
    std::copy(dims.begin(), dims.end(),
              tensorflow::protobuf::RepeatedFieldBackInserter(
                  value.mutable_scatter_dims_to_operand_dims()));
    value.set_index_vector_dim(
        getattr(handle, "index_vector_dim").cast<xla::int64>());
    return true;
  }
};

template <>
struct type_caster<xla::ReplicaGroup> {
 public:
  PYBIND11_TYPE_CASTER(xla::ReplicaGroup, _("xla::ReplicaGroup"));

  // PyObject -> C++ conversion.
  bool load(handle handle, bool) {
    std::vector<xla::int64> dims;
    dims = getattr(handle, "replica_ids").cast<std::vector<xla::int64>>();
    std::copy(dims.begin(), dims.end(),
              tensorflow::protobuf::RepeatedFieldBackInserter(
                  value.mutable_replica_ids()));
    return true;
  }
};

template <>
struct type_caster<xla::PaddingConfig> {
 public:
  PYBIND11_TYPE_CASTER(xla::PaddingConfig, _("xla::PaddingConfig"));

  // PyObject -> C++ conversion.
  bool load(handle handle, bool) {
    sequence dimensions =
        reinterpret_borrow<sequence>(getattr(handle, "dimensions"));

    for (const auto& dimension : dimensions) {
      xla::PaddingConfig::PaddingConfigDimension* config_dim =
          value.add_dimensions();
      config_dim->set_edge_padding_low(
          getattr(dimension, "edge_padding_low").cast<xla::int64>());
      config_dim->set_edge_padding_high(
          getattr(dimension, "edge_padding_high").cast<xla::int64>());
      config_dim->set_interior_padding(
          getattr(dimension, "interior_padding").cast<xla::int64>());
    }
    return true;
  }
};

template <>
struct type_caster<xla::OpMetadata> {
 public:
  PYBIND11_TYPE_CASTER(xla::OpMetadata, _("xla::OpMetadata"));

  // PyObject -> C++ conversion.
  bool load(handle handle, bool) {
    pybind11::handle op_type = getattr(handle, "op_type");
    if (!op_type.is_none()) {
      value.set_op_type(op_type.cast<std::string>());
    }
    pybind11::handle op_name = getattr(handle, "op_name");
    if (!op_name.is_none()) {
      value.set_op_name(op_name.cast<std::string>());
    }
    pybind11::handle source_file = getattr(handle, "source_file");
    if (!source_file.is_none()) {
      value.set_source_file(source_file.cast<std::string>());
    }
    pybind11::handle source_line = getattr(handle, "source_line");
    if (!source_line.is_none()) {
      value.set_source_line(source_line.cast<xla::int32>());
    }
    return true;
  }
};

template <>
struct type_caster<xla::PrecisionConfig> {
 public:
  PYBIND11_TYPE_CASTER(xla::PrecisionConfig, _("xla::PrecisionConfig"));

  // PyObject -> C++ conversion.
  bool load(handle handle, bool) {
    if (handle.is_none()) {
      return true;
    }

    sequence operand_precisions =
        reinterpret_borrow<sequence>(getattr(handle, "operand_precision"));

    for (const auto& operand_precision : operand_precisions) {
      value.add_operand_precision(
          operand_precision.cast<xla::PrecisionConfig::Precision>());
    }
    return true;
  }
};

template <>
struct type_caster<xla::OpSharding> {
 public:
  PYBIND11_TYPE_CASTER(xla::OpSharding, _("xla::OpSharding"));

  // PyObject -> C++ conversion.
  bool load(handle handle_obj, bool) {
    if (handle_obj.is_none()) {
      return true;
    }

    // Sets `type` field.
    handle sharding_type = getattr(handle_obj, "type");
    if (!sharding_type.is_none()) {
      value.set_type(sharding_type.cast<xla::OpSharding_Type>());
    }

    // Sets `tile_assignment_dimensions` field.
    std::vector<xla::int64> dims;
    dims = getattr(handle_obj, "tile_assignment_dimensions")
               .cast<std::vector<xla::int64>>();
    std::copy(dims.begin(), dims.end(),
              tensorflow::protobuf::RepeatedFieldBackInserter(
                  value.mutable_tile_assignment_dimensions()));

    // Sets `tile_assignment_devices` field.
    std::vector<xla::int64> devices;
    devices = getattr(handle_obj, "tile_assignment_devices")
                  .cast<std::vector<xla::int64>>();
    std::copy(devices.begin(), devices.end(),
              tensorflow::protobuf::RepeatedFieldBackInserter(
                  value.mutable_tile_assignment_devices()));

    // Sets `tuple_shardings` field.
    sequence tuple_shardings =
        reinterpret_borrow<sequence>(getattr(handle_obj, "tuple_shardings"));

    for (const auto& tuple_sharding : tuple_shardings) {
      xla::OpSharding* sharding = value.add_tuple_shardings();

      handle sharding_type = getattr(tuple_sharding, "type");
      if (!sharding_type.is_none()) {
        sharding->set_type(sharding_type.cast<xla::OpSharding_Type>());
      }
      std::vector<xla::int64> dims;
      dims = getattr(tuple_sharding, "tile_assignment_dimensions")
                 .cast<std::vector<xla::int64>>();
      std::copy(dims.begin(), dims.end(),
                tensorflow::protobuf::RepeatedFieldBackInserter(
                    sharding->mutable_tile_assignment_dimensions()));

      std::vector<xla::int64> devices;
      devices = getattr(tuple_sharding, "tile_assignment_devices")
                    .cast<std::vector<xla::int64>>();
      std::copy(devices.begin(), devices.end(),
                tensorflow::protobuf::RepeatedFieldBackInserter(
                    sharding->mutable_tile_assignment_devices()));

      sharding->set_replicate_on_last_tile_dim(
          getattr(tuple_sharding, "replicate_on_last_tile_dim").cast<bool>());
    }

    // Sets `replicate_on_last_tile_dim` field.
    value.set_replicate_on_last_tile_dim(
        getattr(handle_obj, "replicate_on_last_tile_dim").cast<bool>());

    return true;
  }
};

}  // namespace detail
}  // namespace pybind11

#endif  // TENSORFLOW_COMPILER_XLA_PYTHON_TYPES_H_
