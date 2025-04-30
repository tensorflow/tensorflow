/* Copyright 2019 The OpenXLA Authors.

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

#include <algorithm>
#include <cstdint>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/hash/hash.h"
#include "absl/types/span.h"
#include "nanobind/nanobind.h"
#include "nanobind/stl/optional.h"  // IWYU pragma: keep
#include "nanobind/stl/pair.h"  // IWYU pragma: keep
#include "nanobind/stl/shared_ptr.h"  // IWYU pragma: keep
#include "nanobind/stl/string.h"  // IWYU pragma: keep
#include "nanobind/stl/tuple.h"  // IWYU pragma: keep
#include "nanobind/stl/vector.h"  // IWYU pragma: keep
#include "xla/hlo/builder/lib/approx_topk.h"
#include "xla/hlo/builder/lib/approx_topk_shape.h"
#include "xla/hlo/builder/lib/comparators.h"
#include "xla/hlo/builder/lib/lu_decomposition.h"
#include "xla/hlo/builder/lib/math.h"
#include "xla/hlo/builder/lib/qr.h"
#include "xla/hlo/builder/lib/self_adjoint_eig.h"
#include "xla/hlo/builder/lib/sorting.h"
#include "xla/hlo/builder/lib/svd.h"
#include "xla/hlo/builder/xla_builder.h"
#include "xla/hlo/builder/xla_computation.h"
#include "xla/literal.h"
#include "xla/pjrt/status_casters.h"
#include "xla/python/nb_absl_span.h"  // IWYU pragma: keep
#include "xla/python/nb_helpers.h"
#include "xla/python/types.h"  // IWYU pragma: keep
#include "xla/service/hlo.pb.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/xla_data.pb.h"

namespace nb = nanobind;

namespace nanobind {
namespace detail {

// XLA protocol buffers
// We don't actually care that these are the protocol buffers, we merely want
// objects that duck type as protocol buffers. The client code currently avoids
// depending on Python protocol buffers to avoid conflicting definitions from
// different modules that both include XLA.

template <>
struct type_caster<xla::ConvolutionDimensionNumbers> {
 public:
  NB_TYPE_CASTER_FROM_PYTHON_ONLY(
      xla::ConvolutionDimensionNumbers,
      const_name("xla::ConvolutionDimensionNumbers"));

  // PyObject -> C++ conversion.
  bool from_python(handle handle, uint8_t, cleanup_list*) noexcept {
    try {
      value.set_input_batch_dimension(
          cast<int64_t>(getattr(handle, "input_batch_dimension")));
      value.set_input_feature_dimension(
          cast<int64_t>(getattr(handle, "input_feature_dimension")));
      value.set_output_batch_dimension(
          cast<int64_t>(getattr(handle, "output_batch_dimension")));
      value.set_output_feature_dimension(
          cast<int64_t>(getattr(handle, "output_feature_dimension")));
      value.set_kernel_input_feature_dimension(
          cast<int64_t>(getattr(handle, "kernel_input_feature_dimension")));
      value.set_kernel_output_feature_dimension(
          cast<int64_t>(getattr(handle, "kernel_output_feature_dimension")));
      std::vector<int64_t> dims;
      dims = cast<std::vector<int64_t>>(
          getattr(handle, "input_spatial_dimensions"));
      std::copy(dims.begin(), dims.end(),
                tsl::protobuf::RepeatedFieldBackInserter(
                    value.mutable_input_spatial_dimensions()));
      dims = cast<std::vector<int64_t>>(
          getattr(handle, "kernel_spatial_dimensions"));
      std::copy(dims.begin(), dims.end(),
                tsl::protobuf::RepeatedFieldBackInserter(
                    value.mutable_kernel_spatial_dimensions()));
      dims = cast<std::vector<int64_t>>(
          getattr(handle, "output_spatial_dimensions"));
      std::copy(dims.begin(), dims.end(),
                tsl::protobuf::RepeatedFieldBackInserter(
                    value.mutable_output_spatial_dimensions()));
      return true;
    } catch (...) {
      return false;
    }
  }
};

template <>
struct type_caster<xla::DotDimensionNumbers> {
 public:
  NB_TYPE_CASTER_FROM_PYTHON_ONLY(xla::DotDimensionNumbers,
                                  const_name("xla::DotDimensionNumbers"));

  // PyObject -> C++ conversion.
  bool from_python(handle handle, uint8_t flags, cleanup_list*) noexcept {
    try {
      std::vector<int64_t> dims = cast<std::vector<int64_t>>(
          getattr(handle, "lhs_contracting_dimensions"));
      std::copy(dims.begin(), dims.end(),
                tsl::protobuf::RepeatedFieldBackInserter(
                    value.mutable_lhs_contracting_dimensions()));
      dims = cast<std::vector<int64_t>>(
          getattr(handle, "rhs_contracting_dimensions"));
      std::copy(dims.begin(), dims.end(),
                tsl::protobuf::RepeatedFieldBackInserter(
                    value.mutable_rhs_contracting_dimensions()));
      dims =
          cast<std::vector<int64_t>>(getattr(handle, "lhs_batch_dimensions"));
      std::copy(dims.begin(), dims.end(),
                tsl::protobuf::RepeatedFieldBackInserter(
                    value.mutable_lhs_batch_dimensions()));
      dims =
          cast<std::vector<int64_t>>(getattr(handle, "rhs_batch_dimensions"));
      std::copy(dims.begin(), dims.end(),
                tsl::protobuf::RepeatedFieldBackInserter(
                    value.mutable_rhs_batch_dimensions()));
      return true;
    } catch (...) {
      return false;
    }
  }
};

template <>
struct type_caster<xla::GatherDimensionNumbers> {
 public:
  NB_TYPE_CASTER_FROM_PYTHON_ONLY(xla::GatherDimensionNumbers,
                                  const_name("xla::GatherDimensionNumbers"));

  // PyObject -> C++ conversion.
  bool from_python(handle handle, uint8_t, cleanup_list*) noexcept {
    try {
      std::vector<int64_t> dims;
      dims = cast<std::vector<int64_t>>(getattr(handle, "offset_dims"));
      std::copy(dims.begin(), dims.end(),
                tsl::protobuf::RepeatedFieldBackInserter(
                    value.mutable_offset_dims()));
      dims =
          cast<std::vector<int64_t>>(getattr(handle, "collapsed_slice_dims"));
      std::copy(dims.begin(), dims.end(),
                tsl::protobuf::RepeatedFieldBackInserter(
                    value.mutable_collapsed_slice_dims()));
      dims = cast<std::vector<int64_t>>(getattr(handle, "start_index_map"));
      std::copy(dims.begin(), dims.end(),
                tsl::protobuf::RepeatedFieldBackInserter(
                    value.mutable_start_index_map()));
      value.set_index_vector_dim(
          cast<int64_t>(getattr(handle, "index_vector_dim")));
      return true;
    } catch (...) {
      return false;
    }
  }
};

template <>
struct type_caster<xla::ScatterDimensionNumbers> {
 public:
  NB_TYPE_CASTER_FROM_PYTHON_ONLY(xla::ScatterDimensionNumbers,
                                  const_name("xla::ScatterDimensionNumbers"));

  // PyObject -> C++ conversion.
  bool from_python(handle handle, uint8_t, cleanup_list*) noexcept {
    try {
      std::vector<int64_t> dims;
      dims = cast<std::vector<int64_t>>(getattr(handle, "update_window_dims"));
      std::copy(dims.begin(), dims.end(),
                tsl::protobuf::RepeatedFieldBackInserter(
                    value.mutable_update_window_dims()));
      dims =
          cast<std::vector<int64_t>>(getattr(handle, "inserted_window_dims"));
      std::copy(dims.begin(), dims.end(),
                tsl::protobuf::RepeatedFieldBackInserter(
                    value.mutable_inserted_window_dims()));
      dims = cast<std::vector<int64_t>>(
          getattr(handle, "scatter_dims_to_operand_dims"));
      std::copy(dims.begin(), dims.end(),
                tsl::protobuf::RepeatedFieldBackInserter(
                    value.mutable_scatter_dims_to_operand_dims()));
      value.set_index_vector_dim(
          cast<int64_t>(getattr(handle, "index_vector_dim")));
      return true;
    } catch (...) {
      return false;
    }
  }
};

template <>
struct type_caster<xla::ReplicaGroup> {
 public:
  NB_TYPE_CASTER_FROM_PYTHON_ONLY(xla::ReplicaGroup,
                                  const_name("xla::ReplicaGroup"));

  // PyObject -> C++ conversion.
  bool from_python(handle handle, uint8_t, cleanup_list*) noexcept {
    try {
      auto dims = cast<std::vector<int64_t>>(getattr(handle, "replica_ids"));
      std::copy(dims.begin(), dims.end(),
                tsl::protobuf::RepeatedFieldBackInserter(
                    value.mutable_replica_ids()));
      return true;
    } catch (...) {
      return false;
    }
  }
};

template <>
struct type_caster<xla::PaddingConfig> {
 public:
  NB_TYPE_CASTER_FROM_PYTHON_ONLY(xla::PaddingConfig,
                                  const_name("xla::PaddingConfig"));

  // PyObject -> C++ conversion.
  bool from_python(handle handle, uint8_t, cleanup_list*) noexcept {
    try {
      sequence dimensions = borrow<sequence>(getattr(handle, "dimensions"));

      for (const auto& dimension : dimensions) {
        xla::PaddingConfig::PaddingConfigDimension* config_dim =
            value.add_dimensions();
        config_dim->set_edge_padding_low(
            cast<int64_t>(getattr(dimension, "edge_padding_low")));
        config_dim->set_edge_padding_high(
            cast<int64_t>(getattr(dimension, "edge_padding_high")));
        config_dim->set_interior_padding(
            cast<int64_t>(getattr(dimension, "interior_padding")));
      }
      return true;
    } catch (...) {
      return false;
    }
  }
};

template <>
struct type_caster<xla::PrecisionConfig> {
 public:
  NB_TYPE_CASTER_FROM_PYTHON_ONLY(xla::PrecisionConfig,
                                  const_name("xla::PrecisionConfig"));

  // PyObject -> C++ conversion.
  bool from_python(handle handle, uint8_t, cleanup_list*) noexcept {
    try {
      if (handle.is_none()) {
        return true;
      }

      sequence operand_precisions =
          borrow<sequence>(getattr(handle, "operand_precision"));

      for (const auto& operand_precision : operand_precisions) {
        value.add_operand_precision(
            cast<xla::PrecisionConfig::Precision>(operand_precision));
      }
      return true;
    } catch (...) {
      return false;
    }
  }
};

template <>
struct type_caster<xla::ResultAccuracy> {
 public:
  NB_TYPE_CASTER_FROM_PYTHON_ONLY(xla::ResultAccuracy,
                                  const_name("xla::ResultAccuracy"));
  // PyObject -> C++ conversion.
  bool from_python(handle handle, uint8_t, cleanup_list*) noexcept {
    try {
      if (handle.is_none()) {
        return true;
      }
      xla::ResultAccuracy::Mode mode =
          cast<xla::ResultAccuracy::Mode>(getattr(handle, "mode"));
      value.set_mode(mode);
      xla::ResultAccuracy::Tolerance* tolerance = value.mutable_tolerance();
      tolerance->set_atol(cast<float>(getattr(handle, "atol")));  // NOLINT
      tolerance->set_rtol(cast<float>(getattr(handle, "rtol")));
      tolerance->set_ulps(cast<int>(getattr(handle, "ulps")));
      return true;
    } catch (...) {
      return false;
    }
  }
};

}  // namespace detail
}  // namespace nanobind

namespace xla {

NB_MODULE(_ops, m) {
  nb::class_<ShapeIndex>(m, "ShapeIndex", R"(Represents an XLA ShapeIndex.

  An index for specifying a particular nested subshape within a shape. Used in
  ShapeUtil::GetSubshape and other interfaces. ShapeIndex defines a path through
  the Shape tree where each element of ShapeIndex indexes into a tuple (or
  nested tuple) within the shape. For a non-nested tuple, an index has a single
  element.)")
      .def("__init__",
           [](ShapeIndex* self, const std::vector<int64_t>& v) {
             new (self) ShapeIndex(v.begin(), v.end());
           })
      .def("__repr__", &ShapeIndex::ToString)
      .def("__eq__", [](const ShapeIndex& shape_ind,
                        const ShapeIndex& other) { return shape_ind == other; })
      .def("__ne__", [](const ShapeIndex& shape_ind,
                        const ShapeIndex& other) { return shape_ind != other; })
      .def("__hash__",
           [](const ShapeIndex& shape_ind) { return absl::HashOf(shape_ind); });

  nb::enum_<FftType>(m, "FftType")
      .value("FFT", FftType::FFT)
      .value("IFFT", FftType::IFFT)
      .value("RFFT", FftType::RFFT)
      .value("IRFFT", FftType::IRFFT);

  nb::enum_<PrecisionConfig::Precision>(m, "PrecisionConfig_Precision")
      .value("DEFAULT", PrecisionConfig::DEFAULT)
      .value("HIGH", PrecisionConfig::HIGH)
      .value("HIGHEST", PrecisionConfig::HIGHEST);

  nb::enum_<TriangularSolveOptions::Transpose>(
      m, "TriangularSolveOptions_Transpose")
      .value("TRANSPOSE_INVALID", TriangularSolveOptions::TRANSPOSE_INVALID)
      .value("NO_TRANSPOSE", TriangularSolveOptions::NO_TRANSPOSE)
      .value("TRANSPOSE", TriangularSolveOptions::TRANSPOSE)
      .value("ADJOINT", TriangularSolveOptions::ADJOINT);

  nb::enum_<RandomAlgorithm>(m, "RandomAlgorithm", nb::is_arithmetic())
      .value("RNG_DEFAULT", RandomAlgorithm::RNG_DEFAULT)
      .value("RNG_THREE_FRY", RandomAlgorithm::RNG_THREE_FRY)
      .value("RNG_PHILOX", RandomAlgorithm::RNG_PHILOX);

  nb::enum_<ResultAccuracy::Mode>(m, "ResultAccuracy_Mode")
      .value("DEFAULT", ResultAccuracy::DEFAULT)
      .value("HIGHEST", ResultAccuracy::HIGHEST);

  nb::enum_<CustomCallSchedule>(m, "CustomCallSchedule")
      .value("SCHEDULE_NONE", CustomCallSchedule::SCHEDULE_NONE)
      .value("SCHEDULE_LATEST", CustomCallSchedule::SCHEDULE_LATEST)
      .value("SCHEDULE_EARLIEST", CustomCallSchedule::SCHEDULE_EARLIEST);

  nb::enum_<CustomCallApiVersion>(m, "CustomCallApiVersion",
                                  nb::is_arithmetic())
      .value("API_VERSION_ORIGINAL", CustomCallApiVersion::API_VERSION_ORIGINAL)
      .value("API_VERSION_STATUS_RETURNING",
             CustomCallApiVersion::API_VERSION_STATUS_RETURNING)
      .value("API_VERSION_STATUS_RETURNING_UNIFIED",
             CustomCallApiVersion::API_VERSION_STATUS_RETURNING_UNIFIED)
      .value("API_VERSION_TYPED_FFI",
             CustomCallApiVersion::API_VERSION_TYPED_FFI);

  m.def("AfterAll", &AfterAll, nb::arg("builder"), nb::arg("tokens"));
  m.def("AllGather", &AllGather, nb::arg("operand"),
        nb::arg("all_gather_dimension"), nb::arg("shard_count"),
        nb::arg("replica_groups") = nb::list(),
        nb::arg("channel_id") = std::nullopt,
        nb::arg("shape_with_layout") = std::nullopt,
        nb::arg("use_global_device_ids") = std::nullopt);
  m.def("AllReduce",
        static_cast<XlaOp (*)(
            XlaOp, const XlaComputation&, absl::Span<const ReplicaGroup>,
            const std::optional<ChannelHandle>&, const std::optional<Shape>&,
            const std::optional<bool>)>(&AllReduce),
        nb::arg("operand"), nb::arg("computation"),
        nb::arg("replica_groups") = nb::list(),
        nb::arg("channel_id") = std::nullopt,
        nb::arg("shape_with_layout") = std::nullopt,
        nb::arg("use_global_device_ids") = std::nullopt);
  m.def("ReduceScatter", &ReduceScatter, nb::arg("operand"),
        nb::arg("computation"), nb::arg("scatter_dimension"),
        nb::arg("shard_count"), nb::arg("replica_groups") = nb::list(),
        nb::arg("channel_id") = std::nullopt, nb::arg("layout") = std::nullopt,
        nb::arg("use_global_device_ids") = std::nullopt);
  m.def("AllToAll", &AllToAll, nb::arg("operand"), nb::arg("split_dimension"),
        nb::arg("concat_dimension"), nb::arg("split_count"),
        nb::arg("replica_groups") = nb::list(),
        nb::arg("layout") = std::nullopt, nb::arg("channel_id") = std::nullopt);
  m.def("ApproxTopK", &ApproxTopK, nb::arg("builder"), nb::arg("operands"),
        nb::arg("init_values"), nb::arg("top_k"), nb::arg("reduction_dim"),
        nb::arg("comparator"), nb::arg("recall_target") = 0.9,
        nb::arg("aggregate_to_topk") = true,
        nb::arg("reduction_input_size_override") = -1);
  m.def("ApproxTopKFallback", &ApproxTopKFallback, nb::arg("builder"),
        nb::arg("operands"), nb::arg("init_values"), nb::arg("top_k"),
        nb::arg("reduction_dim"), nb::arg("comparator"),
        nb::arg("recall_target") = 0.9, nb::arg("aggregate_to_topk") = true,
        nb::arg("reduction_input_size_override") = -1);
  m.def("ApproxTopKReductionOutputSize",
        xla::ValueOrThrowWrapper(ApproxTopKReductionOutputSize),
        nb::arg("input_size"), nb::arg("rank"), nb::arg("top_k"),
        nb::arg("recall_target"), nb::arg("aggregate_to_topk") = true,
        nb::arg("input_size_override") = -1);
  m.def("BitcastConvertType", &BitcastConvertType, nb::arg("operand"),
        nb::arg("new_element_type"));
  m.def("Broadcast", &Broadcast, nb::arg("operand"), nb::arg("sizes"));
  m.def("BroadcastInDim", &BroadcastInDim, nb::arg("operand"), nb::arg("shape"),
        nb::arg("broadcast_dimensions"));
  m.def("Call", &Call, nb::arg("builder"), nb::arg("computation"),
        nb::arg("operands"));
  m.def("Cholesky", &Cholesky, nb::arg("a"), nb::arg("lower") = true);
  m.def("Clamp", &Clamp, nb::arg("min"), nb::arg("operand"), nb::arg("max"));
  m.def("Collapse", &Collapse, nb::arg("operand"), nb::arg("dimensions"));
  m.def("CollectivePermute", &CollectivePermute, nb::arg("operand"),
        nb::arg("source_target_pairs"), nb::arg("channel_id") = std::nullopt,
        nb::arg("inplace") = false);
  m.def("ConcatInDim", &ConcatInDim, nb::arg("builder"), nb::arg("operands"),
        nb::arg("dimension"));
  m.def("Conditional",
        static_cast<XlaOp (*)(XlaOp, absl::Span<const XlaComputation* const>,
                              absl::Span<const XlaOp>)>(&Conditional),
        nb::arg("branch_index"), nb::arg("branch_computations"),
        nb::arg("branch_operands"));
  m.def("Conditional",
        static_cast<XlaOp (*)(XlaOp, XlaOp, const XlaComputation&, XlaOp,
                              const XlaComputation&)>(&Conditional),
        nb::arg("predicate"), nb::arg("true_operand"),
        nb::arg("true_computation"), nb::arg("false_operand"),
        nb::arg("false_computation"));
  m.def("Constant", &ConstantLiteral, nb::arg("builder"), nb::arg("literal"));
  m.def("ConstantLiteral", &ConstantLiteral, nb::arg("builder"),
        nb::arg("literal"));
  m.def("ConvGeneralDilated", &ConvGeneralDilated, nb::arg("lhs"),
        nb::arg("rhs"), nb::arg("window_strides"), nb::arg("padding"),
        nb::arg("lhs_dilation"), nb::arg("rhs_dilation"),
        nb::arg("dimension_numbers"), nb::arg("feature_group_count") = 1,
        nb::arg("batch_group_count") = 1, nb::arg("precision_config") = nullptr,
        nb::arg("preferred_element_type") = std::nullopt,
        nb::arg("window_reversal") = std::nullopt);
  m.def("ConvertElementType", &ConvertElementType, nb::arg("operand"),
        nb::arg("new_element_type"));
  m.def("CreateToken", &CreateToken, nb::arg("builder"));
  m.def("CrossReplicaSum",
        static_cast<XlaOp (*)(XlaOp, absl::Span<const ReplicaGroup>)>(
            &CrossReplicaSum),
        nb::arg("operand"), nb::arg("replica_groups") = nb::list());
  m.def(
      "CustomCall",
      [](XlaBuilder* builder, const nb::bytes& call_target_name,
         absl::Span<const XlaOp> operands, const Shape& shape,
         const nb::bytes& opaque, bool has_side_effect,
         CustomCallSchedule schedule,
         CustomCallApiVersion api_version) -> XlaOp {
        std::string call_target_name_str(call_target_name.c_str(),
                                         call_target_name.size());
        std::string opaque_str(opaque.c_str(), opaque.size());
        return CustomCall(builder, call_target_name_str, operands, shape,
                          opaque_str, has_side_effect,
                          /*output_operand_aliasing=*/{},
                          /*literal=*/nullptr, schedule, api_version);
      },
      nb::arg("builder"), nb::arg("call_target_name"), nb::arg("operands"),
      nb::arg("shape"), nb::arg("opaque") = nb::bytes(""),
      nb::arg("has_side_effect") = false,
      nb::arg("schedule") = CustomCallSchedule::SCHEDULE_NONE,
      nb::arg("api_version") = CustomCallApiVersion::API_VERSION_ORIGINAL);
  m.def(
      "CustomCallWithLayout",
      [](XlaBuilder* builder, const nb::bytes& call_target_name,
         absl::Span<const XlaOp> operands, const Shape& shape_with_layout,
         absl::Span<const Shape> operand_shapes_with_layout,
         const nb::bytes& opaque, bool has_side_effect,
         CustomCallSchedule schedule,
         CustomCallApiVersion api_version) -> XlaOp {
        std::string call_target_name_str(call_target_name.c_str(),
                                         call_target_name.size());
        std::string opaque_str(opaque.c_str(), opaque.size());
        return CustomCallWithLayout(
            builder, call_target_name_str, operands, shape_with_layout,
            operand_shapes_with_layout, opaque_str, has_side_effect,
            /*output_operand_aliasing=*/{},
            /*literal=*/nullptr, schedule, api_version);
      },
      nb::arg("builder"), nb::arg("call_target_name"), nb::arg("operands"),
      nb::arg("shape_with_layout"), nb::arg("operand_shapes_with_layout"),
      nb::arg("opaque") = nb::bytes(""), nb::arg("has_side_effect") = false,
      nb::arg("schedule") = CustomCallSchedule::SCHEDULE_NONE,
      nb::arg("api_version") = CustomCallApiVersion::API_VERSION_ORIGINAL);
  m.def(
      "CustomCallWithAliasing",
      [](XlaBuilder* builder, const nb::bytes& call_target_name,
         absl::Span<const XlaOp> operands, const Shape& shape_with_layout,
         absl::Span<const Shape> operand_shapes_with_layout,
         const nb::bytes& opaque, bool has_side_effect,
         absl::Span<const std::pair<ShapeIndex, std::pair<int64_t, ShapeIndex>>>
             output_operand_aliasing,
         const Literal* literal, CustomCallSchedule schedule,
         CustomCallApiVersion api_version) -> XlaOp {
        std::string call_target_name_str(call_target_name.c_str(),
                                         call_target_name.size());
        std::string opaque_str(opaque.c_str(), opaque.size());
        return CustomCallWithLayout(
            builder, call_target_name_str, operands, shape_with_layout,
            operand_shapes_with_layout, opaque_str, has_side_effect,
            output_operand_aliasing, literal, schedule, api_version);
      },
      nb::arg("builder"), nb::arg("call_target_name"), nb::arg("operands"),
      nb::arg("shape_with_layout"), nb::arg("operand_shapes_with_layout"),
      nb::arg("opaque") = nb::bytes(""), nb::arg("has_side_effect") = false,
      nb::arg("output_operand_aliasing"), nb::arg("literal") = nullptr,
      nb::arg("schedule") = CustomCallSchedule::SCHEDULE_NONE,
      nb::arg("api_version") = CustomCallApiVersion::API_VERSION_ORIGINAL);
  m.def(
      "CustomCallWithComputation",
      [](XlaBuilder* builder, const nb::bytes& call_target_name,
         absl::Span<const XlaOp> operands, const XlaComputation& computation,
         const Shape& shape, const nb::bytes& opaque, bool has_side_effect,
         absl::Span<const std::pair<ShapeIndex, std::pair<int64_t, ShapeIndex>>>
             output_operand_aliasing,
         const Literal* literal, CustomCallSchedule schedule,
         CustomCallApiVersion api_version) -> XlaOp {
        std::string call_target_name_str(call_target_name.c_str(),
                                         call_target_name.size());
        std::string opaque_str(opaque.c_str(), opaque.size());
        return CustomCallWithComputation(
            builder, call_target_name_str, operands, computation, shape,
            opaque_str, has_side_effect, output_operand_aliasing, literal,
            schedule, api_version);
      },
      nb::arg("builder"), nb::arg("call_target_name"), nb::arg("operands"),
      nb::arg("computation"), nb::arg("shape"),
      nb::arg("opaque") = nb::bytes(""), nb::arg("has_side_effect") = false,
      nb::arg("output_operand_aliasing"), nb::arg("literal") = nullptr,
      nb::arg("schedule") = CustomCallSchedule::SCHEDULE_NONE,
      nb::arg("api_version") = CustomCallApiVersion::API_VERSION_ORIGINAL);
  m.def("Dot", &Dot, nb::arg("lhs"), nb::arg("rhs"),
        nb::arg("precision_config") = nullptr,
        nb::arg("preferred_element_type") = std::nullopt);
  m.def("DotGeneral", &DotGeneral, nb::arg("lhs"), nb::arg("rhs"),
        nb::arg("dimension_numbers"), nb::arg("precision_config") = nullptr,
        nb::arg("preferred_element_type") = std::nullopt);
  m.def("DynamicReshape",
        static_cast<XlaOp (*)(XlaOp, absl::Span<const XlaOp>,
                              absl::Span<const int64_t>,
                              const std::vector<bool>&)>(&DynamicReshape),
        nb::arg("operand"), nb::arg("dim_sizes"), nb::arg("new_size_bounds"),
        nb::arg("dims_are_dynamic"));
  m.def("DynamicSlice",
        static_cast<XlaOp (*)(XlaOp, absl::Span<const XlaOp>,
                              absl::Span<const int64_t>)>(&DynamicSlice),
        nb::arg("operand"), nb::arg("start_indices"), nb::arg("slice_sizes"));
  m.def("DynamicUpdateSlice",
        static_cast<XlaOp (*)(XlaOp, XlaOp, absl::Span<const XlaOp>)>(
            &DynamicUpdateSlice),
        nb::arg("operand"), nb::arg("update"), nb::arg("start_indices"));
  m.def(
      "Eigh",
      [](XlaOp a, bool lower, int64_t max_iter, float epsilon,
         bool sort_eigenvalues) -> std::pair<XlaOp, XlaOp> {
        auto eigh =
            SelfAdjointEig(a, lower, max_iter, epsilon, sort_eigenvalues);
        return std::make_pair(eigh.v, eigh.w);
      },
      nb::arg("a"), nb::arg("lower") = true, nb::arg("max_iter") = 15,
      nb::arg("epsilon") = 1e-5, nb::arg("sort_eigenvalues") = true);
  m.def("Fft", &Fft, nb::arg("operand"), nb::arg("fft_type"),
        nb::arg("fft_length"));
  m.def("Gather", &Gather, nb::arg("a"), nb::arg("start_indices"),
        nb::arg("dimension_numbers"), nb::arg("slice_sizes"),
        nb::arg("indices_are_sorted") = false);
  m.def("GetDimensionSize", &GetDimensionSize, nb::arg("operand"),
        nb::arg("dimension"));
  m.def("GetTupleElement", &GetTupleElement, nb::arg("tuple_data"),
        nb::arg("index"));
  m.def("InfeedWithToken", &InfeedWithToken, nb::arg("token"), nb::arg("shape"),
        nb::arg("config") = "");
  m.def("Iota",
        static_cast<XlaOp (*)(XlaBuilder*, const Shape&, int64_t)>(&Iota),
        nb::arg("builder"), nb::arg("shape"), nb::arg("iota_dimension"));
  m.def("Iota",
        static_cast<XlaOp (*)(XlaBuilder*, PrimitiveType, int64_t)>(&Iota),
        nb::arg("builder"), nb::arg("type"), nb::arg("size"));
  m.def(
      "LU",
      [](XlaOp a) -> std::tuple<XlaOp, XlaOp, XlaOp> {
        LuDecompositionResult lu = LuDecomposition(a);
        return std::make_tuple(lu.lu, lu.pivots, lu.permutation);
      },
      nb::arg("operand"));
  m.def("Map", &Map, nb::arg("builder"), nb::arg("operands"),
        nb::arg("computation"), nb::arg("dimensions"),
        nb::arg("static_operands") = nb::list());
  m.def("MultiCollectivePermute", &MultiCollectivePermute, nb::arg("operands"),
        nb::arg("source_target_pairs"), nb::arg("channel_id") = std::nullopt,
        nb::arg("inplace") = false);
  m.def("NextAfter", &NextAfter, nb::arg("from"), nb::arg("to"));
  m.def("OutfeedWithToken", &OutfeedWithToken, nb::arg("operand"),
        nb::arg("token"), nb::arg("shape_with_layout"),
        nb::arg("outfeed_config") = "");
  m.def("Pad", &Pad, nb::arg("operand"), nb::arg("padding_value"),
        nb::arg("padding_config"));
  m.def("Parameter",
        static_cast<XlaOp (*)(XlaBuilder*, int64_t, const Shape&,
                              const std::string&, const std::vector<bool>&)>(
            &Parameter),
        nb::arg("builder"), nb::arg("parameter_number"), nb::arg("shape"),
        nb::arg("name") = "",
        nb::arg("replicated_at_leaf_buffers") = std::vector<bool>());
  m.def("ProductOfElementaryHouseholderReflectors",
        &ProductOfElementaryHouseholderReflectors, nb::arg("a"),
        nb::arg("taus"));
  m.def(
      "QR",
      [](XlaOp a, bool full_matrices) -> std::pair<XlaOp, XlaOp> {
        XlaOp q, r;
        QrExplicit(a, full_matrices, q, r);
        return std::make_pair(q, r);
      },
      nb::arg("operand"), nb::arg("full_matrices"));
  m.def(
      "QrDecomposition",
      [](XlaOp a) -> std::pair<XlaOp, XlaOp> {
        QrDecomposition d = Qr(a);
        return std::make_pair(d.q_and_r, d.taus);
      },
      nb::arg("operand"));
  m.def("RecvFromHost", &RecvFromHost, nb::arg("token"), nb::arg("shape"),
        nb::arg("handle"));
  m.def("Reduce",
        static_cast<XlaOp (*)(XlaBuilder*, absl::Span<const XlaOp>,
                              absl::Span<const XlaOp>, const XlaComputation&,
                              absl::Span<const int64_t>)>(&Reduce),
        nb::arg("builder"), nb::arg("operands"), nb::arg("init_values"),
        nb::arg("computation"), nb::arg("dimensions_to_reduce"));
  m.def("ReducePrecision", &ReducePrecision, nb::arg("operand"),
        nb::arg("exponent_bits"), nb::arg("mantissa_bits"));
  m.def("ReduceWindowWithGeneralPadding",
        static_cast<XlaOp (*)(
            XlaOp, XlaOp, const XlaComputation&, absl::Span<const int64_t>,
            absl::Span<const int64_t>, absl::Span<const int64_t>,
            absl::Span<const int64_t>,
            absl::Span<const std::pair<int64_t, int64_t>>)>(
            &ReduceWindowWithGeneralPadding),
        nb::arg("operand"), nb::arg("init_value"), nb::arg("computation"),
        nb::arg("window_dimensions"), nb::arg("window_strides"),
        nb::arg("base_dilations"), nb::arg("window_dilations"),
        nb::arg("padding"));
  m.def("ReduceWindowWithGeneralPadding",
        static_cast<XlaOp (*)(absl::Span<const XlaOp>, absl::Span<const XlaOp>,
                              const XlaComputation&, absl::Span<const int64_t>,
                              absl::Span<const int64_t>,
                              absl::Span<const int64_t>,
                              absl::Span<const int64_t>,
                              absl::Span<const std::pair<int64_t, int64_t>>)>(
            &ReduceWindowWithGeneralPadding),
        nb::arg("operands"), nb::arg("init_values"), nb::arg("computation"),
        nb::arg("window_dimensions"), nb::arg("window_strides"),
        nb::arg("base_dilations"), nb::arg("window_dilations"),
        nb::arg("padding"));
  m.def("RemoveDynamicDimension", &RemoveDynamicDimension, nb::arg("operand"),
        nb::arg("dimension"));
  m.def("ReplicaId", &ReplicaId, nb::arg("builder"));
  m.def("Reshape",
        static_cast<XlaOp (*)(XlaOp, absl::Span<const int64_t>)>(&Reshape),
        nb::arg("operand"), nb::arg("new_sizes"));
  m.def("Rev", &Rev, nb::arg("operand"), nb::arg("dimensions"));
  m.def("RngBitGenerator", &RngBitGenerator, nb::arg("algorithm"),
        nb::arg("initial_state"), nb::arg("shape"));
  m.def("RngNormal", &RngNormal, nb::arg("mu"), nb::arg("sigma"),
        nb::arg("shape"));
  m.def("RngUniform", &RngUniform, nb::arg("a"), nb::arg("b"),
        nb::arg("shape"));
  m.def("Scatter",
        static_cast<XlaOp (*)(XlaOp, XlaOp, XlaOp, const XlaComputation&,
                              const ScatterDimensionNumbers&, bool, bool)>(
            &Scatter),
        nb::arg("input"), nb::arg("scatter_indices"), nb::arg("updates"),
        nb::arg("update_computation"), nb::arg("dimension_numbers"),
        nb::arg("indices_are_sorted") = false,
        nb::arg("unique_indices") = false);
  m.def("Scatter",
        static_cast<XlaOp (*)(absl::Span<const XlaOp>, XlaOp,
                              absl::Span<const XlaOp>, const XlaComputation&,
                              const ScatterDimensionNumbers&, bool, bool)>(
            &Scatter),
        nb::arg("inputs"), nb::arg("scatter_indices"), nb::arg("updates"),
        nb::arg("update_computation"), nb::arg("dimension_numbers"),
        nb::arg("indices_are_sorted") = false,
        nb::arg("unique_indices") = false);
  m.def("Select", &Select, nb::arg("pred"), nb::arg("on_true"),
        nb::arg("on_false"));
  m.def("SelectAndScatterWithGeneralPadding",
        &SelectAndScatterWithGeneralPadding, nb::arg("operand"),
        nb::arg("select"), nb::arg("window_dimensions"),
        nb::arg("window_strides"), nb::arg("padding"), nb::arg("source"),
        nb::arg("init_value"), nb::arg("scatter"));
  m.def("SendToHost", &SendToHost, nb::arg("operand"), nb::arg("token"),
        nb::arg("shape_with_layout"), nb::arg("handle"));
  m.def("SetDimensionSize", &SetDimensionSize, nb::arg("operand"),
        nb::arg("val"), nb::arg("dimension"));
  m.def("Slice", &Slice, nb::arg("operand"), nb::arg("start_indices"),
        nb::arg("limit_indices"), nb::arg("strides"));
  m.def("SliceInDim", &SliceInDim, nb::arg("operand"), nb::arg("start_index"),
        nb::arg("limit_index"), nb::arg("stride"), nb::arg("dimno"));
  m.def(
      "Sort",
      [](XlaBuilder* builder, absl::Span<const XlaOp> operands,
         std::optional<const XlaComputation*> comparator, int64_t dimension,
         bool is_stable) -> XlaOp {
        return builder->ReportErrorOrReturn([&]() -> XlaOp {
          std::vector<PrimitiveType> operand_types;
          operand_types.reserve(operands.size());
          for (const auto& operand : operands) {
            auto operand_shape = xla::ValueOrThrow(builder->GetShape(operand));
            operand_types.push_back(operand_shape.element_type());
          }

          if (comparator) {
            return Sort(operands, **comparator, dimension, is_stable);
          }
          return Sort(operands,
                      CreateScalarLtComputation(operand_types, builder),
                      dimension, is_stable);
        });
      },
      nb::arg("builder"), nb::arg("operands"),
      nb::arg("comparator") = std::nullopt, nb::arg("dimension") = -1,
      nb::arg("is_stable") = false);
  m.def(
      "SVD",
      [](XlaOp a, int64_t max_iter,
         float epsilon) -> std::tuple<XlaOp, XlaOp, XlaOp> {
        auto svd = SVD(a, max_iter, epsilon);
        return std::make_tuple(svd.u, svd.d, svd.v);
      },
      nb::arg("a"), nb::arg("max_iter") = 100, nb::arg("epsilon") = 1e-6);
  m.def(
      "TopK",
      [](XlaOp input, int64_t k) {
        return TopK(input, k, /*index_type=*/PrimitiveType::S32);
      },
      nb::arg("input"), nb::arg("k"));
  m.def("Transpose", &Transpose, nb::arg("operand"), nb::arg("permutation"));
  m.def("TriangularSolve", &TriangularSolve, nb::arg("a"), nb::arg("b"),
        nb::arg("left_side"), nb::arg("lower"), nb::arg("unit_diagonal"),
        nb::arg("transpose_a"));
  m.def("Tuple", &Tuple, nb::arg("builder"), nb::arg("elements"));
  m.def("While", &While, nb::arg("condition"), nb::arg("body"),
        nb::arg("init"));

  m.def("Igamma", &Igamma, nb::arg("a"), nb::arg("x"));
  m.def("Igammac", &Igammac, nb::arg("a"), nb::arg("x"));
  m.def("IgammaGradA", &IgammaGradA, nb::arg("a"), nb::arg("x"));
  m.def("RandomGammaGrad", &RandomGammaGrad, nb::arg("a"), nb::arg("x"));
  m.def("RegularizedIncompleteBeta", &RegularizedIncompleteBeta, nb::arg("a"),
        nb::arg("b"), nb::arg("x"));
  m.def("Zeta", &Zeta, nb::arg("x"), nb::arg("q"));

  m.def("Cbrt",
        static_cast<XlaOp (*)(XlaOp, const std::optional<ResultAccuracy>&)>(
            &Cbrt),
        nb::arg("operand"), nb::arg("result_accuracy") = std::nullopt);

  m.def(
      "Cos",
      static_cast<XlaOp (*)(XlaOp, const std::optional<ResultAccuracy>&)>(&Cos),
      nb::arg("operand"), nb::arg("result_accuracy") = std::nullopt);

  m.def(
      "Erf",
      static_cast<XlaOp (*)(XlaOp, const std::optional<ResultAccuracy>&)>(&Erf),
      nb::arg("operand"), nb::arg("result_accuracy") = std::nullopt);

  m.def(
      "Exp",
      static_cast<XlaOp (*)(XlaOp, const std::optional<ResultAccuracy>&)>(&Exp),
      nb::arg("operand"), nb::arg("result_accuracy") = std::nullopt);

  m.def("Expm1",
        static_cast<XlaOp (*)(XlaOp, const std::optional<ResultAccuracy>&)>(
            &Expm1),
        nb::arg("operand"), nb::arg("result_accuracy") = std::nullopt);

  m.def(
      "Log",
      static_cast<XlaOp (*)(XlaOp, const std::optional<ResultAccuracy>&)>(&Log),
      nb::arg("operand"), nb::arg("result_accuracy") = std::nullopt);

  m.def("Log1p",
        static_cast<XlaOp (*)(XlaOp, const std::optional<ResultAccuracy>&)>(
            &Log1p),
        nb::arg("operand"), nb::arg("result_accuracy") = std::nullopt);

  m.def("Logistic",
        static_cast<XlaOp (*)(XlaOp, const std::optional<ResultAccuracy>&)>(
            &Logistic),
        nb::arg("operand"), nb::arg("result_accuracy") = std::nullopt);

  m.def("Rsqrt",
        static_cast<XlaOp (*)(XlaOp, const std::optional<ResultAccuracy>&)>(
            &Rsqrt),
        nb::arg("operand"), nb::arg("result_accuracy") = std::nullopt);

  m.def(
      "Sin",
      static_cast<XlaOp (*)(XlaOp, const std::optional<ResultAccuracy>&)>(&Sin),
      nb::arg("operand"), nb::arg("result_accuracy") = std::nullopt);

  m.def("Sqrt",
        static_cast<XlaOp (*)(XlaOp, const std::optional<ResultAccuracy>&)>(
            &Sqrt),
        nb::arg("operand"), nb::arg("result_accuracy") = std::nullopt);

  m.def(
      "Tan",
      static_cast<XlaOp (*)(XlaOp, const std::optional<ResultAccuracy>&)>(&Tan),
      nb::arg("operand"), nb::arg("result_accuracy") = std::nullopt);

  m.def("Tanh",
        static_cast<XlaOp (*)(XlaOp, const std::optional<ResultAccuracy>&)>(
            &Tanh),
        nb::arg("operand"), nb::arg("result_accuracy") = std::nullopt);

#define BINARY_OP(op)                                                  \
  m.def(                                                               \
      #op,                                                             \
      [](XlaOp a, XlaOp b, std::optional<std::vector<int64_t>> dims) { \
        return dims ? op(a, b, *dims) : op(a, b);                      \
      },                                                               \
      nb::arg("lhs"), nb::arg("rhs"),                                  \
      nb::arg("broadcast_dimensions") = std::nullopt)
  BINARY_OP(Eq);
  BINARY_OP(Ne);
  BINARY_OP(Ge);
  BINARY_OP(Gt);
  BINARY_OP(Lt);
  BINARY_OP(Le);
  BINARY_OP(Add);
  BINARY_OP(Sub);
  BINARY_OP(Mul);
  BINARY_OP(Div);
  BINARY_OP(Rem);
  BINARY_OP(Max);
  BINARY_OP(Min);
  BINARY_OP(And);
  BINARY_OP(Or);
  BINARY_OP(Xor);
  BINARY_OP(ShiftLeft);
  BINARY_OP(ShiftRightArithmetic);
  BINARY_OP(ShiftRightLogical);
  BINARY_OP(Atan2);
  BINARY_OP(Pow);
  BINARY_OP(Complex);
#undef BINARY_OP

#define UNARY_OP(op) m.def(#op, &op)
  UNARY_OP(Not);
  UNARY_OP(PopulationCount);
  UNARY_OP(Clz);
  UNARY_OP(Abs);
  UNARY_OP(Floor);
  UNARY_OP(Ceil);
  UNARY_OP(Round);
  UNARY_OP(Sign);
  UNARY_OP(IsFinite);
  UNARY_OP(Neg);
  UNARY_OP(Square);
  UNARY_OP(Reciprocal);
  UNARY_OP(Erfc);
  UNARY_OP(ErfInv);
  UNARY_OP(Lgamma);
  UNARY_OP(Digamma);
  UNARY_OP(BesselI0e);
  UNARY_OP(BesselI1e);
  UNARY_OP(Acos);
  UNARY_OP(Asin);
  UNARY_OP(Atan);
  UNARY_OP(Acosh);
  UNARY_OP(Asinh);
  UNARY_OP(Atanh);
  UNARY_OP(Cosh);
  UNARY_OP(Sinh);
  UNARY_OP(Real);
  UNARY_OP(Imag);
  UNARY_OP(Conj);
  UNARY_OP(OptimizationBarrier);
#undef UNARY_OP
}  // NOLINT(readability/fn_size)

}  // namespace xla
