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

#include "xla/python/ops.h"

#include <algorithm>
#include <cstdint>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/types/span.h"
#include "third_party/nanobind/include/nanobind/nanobind.h"
#include "third_party/nanobind/include/nanobind/stl/optional.h"  // IWYU pragma: keep
#include "third_party/nanobind/include/nanobind/stl/pair.h"  // IWYU pragma: keep
#include "third_party/nanobind/include/nanobind/stl/shared_ptr.h"  // IWYU pragma: keep
#include "third_party/nanobind/include/nanobind/stl/string.h"  // IWYU pragma: keep
#include "third_party/nanobind/include/nanobind/stl/tuple.h"  // IWYU pragma: keep
#include "third_party/nanobind/include/nanobind/stl/vector.h"  // IWYU pragma: keep
#include "xla/client/lib/approx_topk.h"
#include "xla/client/lib/approx_topk_shape.h"
#include "xla/client/lib/comparators.h"
#include "xla/client/lib/lu_decomposition.h"
#include "xla/client/lib/math.h"
#include "xla/client/lib/qr.h"
#include "xla/client/lib/self_adjoint_eig.h"
#include "xla/client/lib/sorting.h"
#include "xla/client/lib/svd.h"
#include "xla/client/xla_builder.h"
#include "xla/client/xla_computation.h"
#include "xla/pjrt/status_casters.h"
#include "xla/python/nb_absl_span.h"  // IWYU pragma: keep
#include "xla/python/nb_helpers.h"
#include "xla/python/types.h"
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
  bool from_python(handle handle, uint8_t, cleanup_list*) {
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
  bool from_python(handle handle, uint8_t, cleanup_list*) {
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
  bool from_python(handle handle, uint8_t, cleanup_list*) {
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
  bool from_python(handle handle, uint8_t, cleanup_list*) {
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
  bool from_python(handle handle, uint8_t, cleanup_list*) {
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
  bool from_python(handle handle, uint8_t, cleanup_list*) {
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

}  // namespace detail
}  // namespace nanobind

namespace xla {

void BuildOpsSubmodule(nb::module_& m) {
  // ops submodule, containing free functions that add operators to an
  // XlaBuilder.
  nb::module_ ops = m.def_submodule("ops", "XLA operations");

  nb::enum_<TriangularSolveOptions::Transpose>(
      ops, "TriangularSolveOptions_Transpose")
      .value("TRANSPOSE_INVALID", TriangularSolveOptions::TRANSPOSE_INVALID)
      .value("NO_TRANSPOSE", TriangularSolveOptions::NO_TRANSPOSE)
      .value("TRANSPOSE", TriangularSolveOptions::TRANSPOSE)
      .value("ADJOINT", TriangularSolveOptions::ADJOINT);

  nb::enum_<RandomAlgorithm>(ops, "RandomAlgorithm")
      .value("RNG_DEFAULT", RandomAlgorithm::RNG_DEFAULT)
      .value("RNG_THREE_FRY", RandomAlgorithm::RNG_THREE_FRY)
      .value("RNG_PHILOX", RandomAlgorithm::RNG_PHILOX);

  nb::enum_<CustomCallSchedule>(ops, "CustomCallSchedule")
      .value("SCHEDULE_NONE", CustomCallSchedule::SCHEDULE_NONE)
      .value("SCHEDULE_LATEST", CustomCallSchedule::SCHEDULE_LATEST)
      .value("SCHEDULE_EARLIEST", CustomCallSchedule::SCHEDULE_EARLIEST);

  nb::enum_<CustomCallApiVersion>(ops, "CustomCallApiVersion")
      .value("API_VERSION_ORIGINAL", CustomCallApiVersion::API_VERSION_ORIGINAL)
      .value("API_VERSION_STATUS_RETURNING",
             CustomCallApiVersion::API_VERSION_STATUS_RETURNING)
      .value("API_VERSION_STATUS_RETURNING_UNIFIED",
             CustomCallApiVersion::API_VERSION_STATUS_RETURNING_UNIFIED)
      .value("API_VERSION_TYPED_FFI",
             CustomCallApiVersion::API_VERSION_TYPED_FFI);

  ops.def("AfterAll", &AfterAll, nb::arg("builder"), nb::arg("tokens"));
  ops.def("AllGather", &AllGather, nb::arg("operand"),
          nb::arg("all_gather_dimension"), nb::arg("shard_count"),
          nb::arg("replica_groups") = nb::list(),
          nb::arg("channel_id") = std::nullopt,
          nb::arg("shape_with_layout") = std::nullopt,
          nb::arg("use_global_device_ids") = std::nullopt);
  ops.def("AllReduce",
          static_cast<XlaOp (*)(
              XlaOp, const XlaComputation&, absl::Span<const ReplicaGroup>,
              const std::optional<ChannelHandle>&, const std::optional<Shape>&,
              const std::optional<bool>)>(&AllReduce),
          nb::arg("operand"), nb::arg("computation"),
          nb::arg("replica_groups") = nb::list(),
          nb::arg("channel_id") = std::nullopt,
          nb::arg("shape_with_layout") = std::nullopt,
          nb::arg("use_global_device_ids") = std::nullopt);
  ops.def("ReduceScatter", &ReduceScatter, nb::arg("operand"),
          nb::arg("computation"), nb::arg("scatter_dimension"),
          nb::arg("shard_count"), nb::arg("replica_groups") = nb::list(),
          nb::arg("channel_id") = std::nullopt,
          nb::arg("layout") = std::nullopt,
          nb::arg("use_global_device_ids") = std::nullopt);
  ops.def("AllToAll", &AllToAll, nb::arg("operand"), nb::arg("split_dimension"),
          nb::arg("concat_dimension"), nb::arg("split_count"),
          nb::arg("replica_groups") = nb::list(),
          nb::arg("layout") = std::nullopt,
          nb::arg("channel_id") = std::nullopt);
  ops.def("ApproxTopK", &ApproxTopK, nb::arg("builder"), nb::arg("operands"),
          nb::arg("init_values"), nb::arg("top_k"), nb::arg("reduction_dim"),
          nb::arg("comparator"), nb::arg("recall_target") = 0.9,
          nb::arg("aggregate_to_topk") = true,
          nb::arg("reduction_input_size_override") = -1);
  ops.def("ApproxTopKFallback", &ApproxTopKFallback, nb::arg("builder"),
          nb::arg("operands"), nb::arg("init_values"), nb::arg("top_k"),
          nb::arg("reduction_dim"), nb::arg("comparator"),
          nb::arg("recall_target") = 0.9, nb::arg("aggregate_to_topk") = true,
          nb::arg("reduction_input_size_override") = -1);
  ops.def("ApproxTopKReductionOutputSize",
          xla::ValueOrThrowWrapper(ApproxTopKReductionOutputSize),
          nb::arg("input_size"), nb::arg("rank"), nb::arg("top_k"),
          nb::arg("recall_target"), nb::arg("aggregate_to_topk") = true,
          nb::arg("input_size_override") = -1);
  ops.def("BitcastConvertType", &BitcastConvertType, nb::arg("operand"),
          nb::arg("new_element_type"));
  ops.def("Broadcast", &Broadcast, nb::arg("operand"), nb::arg("sizes"));
  ops.def("BroadcastInDim", &BroadcastInDim, nb::arg("operand"),
          nb::arg("shape"), nb::arg("broadcast_dimensions"));
  ops.def("Call", &Call, nb::arg("builder"), nb::arg("computation"),
          nb::arg("operands"));
  ops.def("Cholesky", &Cholesky, nb::arg("a"), nb::arg("lower") = true);
  ops.def("Clamp", &Clamp, nb::arg("min"), nb::arg("operand"), nb::arg("max"));
  ops.def("Collapse", &Collapse, nb::arg("operand"), nb::arg("dimensions"));
  ops.def("CollectivePermute", &CollectivePermute, nb::arg("operand"),
          nb::arg("source_target_pairs"), nb::arg("channel_id") = std::nullopt);
  ops.def("ConcatInDim", &ConcatInDim, nb::arg("builder"), nb::arg("operands"),
          nb::arg("dimension"));
  ops.def("Conditional",
          static_cast<XlaOp (*)(XlaOp, absl::Span<const XlaComputation* const>,
                                absl::Span<const XlaOp>)>(&Conditional),
          nb::arg("branch_index"), nb::arg("branch_computations"),
          nb::arg("branch_operands"));
  ops.def("Conditional",
          static_cast<XlaOp (*)(XlaOp, XlaOp, const XlaComputation&, XlaOp,
                                const XlaComputation&)>(&Conditional),
          nb::arg("predicate"), nb::arg("true_operand"),
          nb::arg("true_computation"), nb::arg("false_operand"),
          nb::arg("false_computation"));
  ops.def("Constant", &ConstantLiteral, nb::arg("builder"), nb::arg("literal"));
  ops.def("ConstantLiteral", &ConstantLiteral, nb::arg("builder"),
          nb::arg("literal"));
  ops.def("ConvGeneralDilated", &ConvGeneralDilated, nb::arg("lhs"),
          nb::arg("rhs"), nb::arg("window_strides"), nb::arg("padding"),
          nb::arg("lhs_dilation"), nb::arg("rhs_dilation"),
          nb::arg("dimension_numbers"), nb::arg("feature_group_count") = 1,
          nb::arg("batch_group_count") = 1,
          nb::arg("precision_config") = nullptr,
          nb::arg("preferred_element_type") = std::nullopt,
          nb::arg("window_reversal") = std::nullopt);
  ops.def("ConvertElementType", &ConvertElementType, nb::arg("operand"),
          nb::arg("new_element_type"));
  ops.def("CreateToken", &CreateToken, nb::arg("builder"));
  ops.def("CrossReplicaSum",
          static_cast<XlaOp (*)(XlaOp, absl::Span<const ReplicaGroup>)>(
              &CrossReplicaSum),
          nb::arg("operand"), nb::arg("replica_groups") = nb::list());
  ops.def(
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
  ops.def(
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
  ops.def(
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
  ops.def(
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
  ops.def("Dot", &Dot, nb::arg("lhs"), nb::arg("rhs"),
          nb::arg("precision_config") = nullptr,
          nb::arg("preferred_element_type") = std::nullopt);
  ops.def("DotGeneral", &DotGeneral, nb::arg("lhs"), nb::arg("rhs"),
          nb::arg("dimension_numbers"), nb::arg("precision_config") = nullptr,
          nb::arg("preferred_element_type") = std::nullopt);
  ops.def("DynamicReshape",
          static_cast<XlaOp (*)(XlaOp, absl::Span<const XlaOp>,
                                absl::Span<const int64_t>,
                                const std::vector<bool>&)>(&DynamicReshape),
          nb::arg("operand"), nb::arg("dim_sizes"), nb::arg("new_size_bounds"),
          nb::arg("dims_are_dynamic"));
  ops.def("DynamicSlice",
          static_cast<XlaOp (*)(XlaOp, absl::Span<const XlaOp>,
                                absl::Span<const int64_t>)>(&DynamicSlice),
          nb::arg("operand"), nb::arg("start_indices"), nb::arg("slice_sizes"));
  ops.def("DynamicUpdateSlice",
          static_cast<XlaOp (*)(XlaOp, XlaOp, absl::Span<const XlaOp>)>(
              &DynamicUpdateSlice),
          nb::arg("operand"), nb::arg("update"), nb::arg("start_indices"));
  ops.def(
      "Eigh",
      [](XlaOp a, bool lower, int64_t max_iter, float epsilon,
         bool sort_eigenvalues) -> std::pair<XlaOp, XlaOp> {
        auto eigh =
            SelfAdjointEig(a, lower, max_iter, epsilon, sort_eigenvalues);
        return std::make_pair(eigh.v, eigh.w);
      },
      nb::arg("a"), nb::arg("lower") = true, nb::arg("max_iter") = 15,
      nb::arg("epsilon") = 1e-5, nb::arg("sort_eigenvalues") = true);
  ops.def("Fft", &Fft, nb::arg("operand"), nb::arg("fft_type"),
          nb::arg("fft_length"));
  ops.def("Gather", &Gather, nb::arg("a"), nb::arg("start_indices"),
          nb::arg("dimension_numbers"), nb::arg("slice_sizes"),
          nb::arg("indices_are_sorted") = false);
  ops.def("GetDimensionSize", &GetDimensionSize, nb::arg("operand"),
          nb::arg("dimension"));
  ops.def("GetTupleElement", &GetTupleElement, nb::arg("tuple_data"),
          nb::arg("index"));
  ops.def("InfeedWithToken", &InfeedWithToken, nb::arg("token"),
          nb::arg("shape"), nb::arg("config") = "");
  ops.def("Iota",
          static_cast<XlaOp (*)(XlaBuilder*, const Shape&, int64_t)>(&Iota),
          nb::arg("builder"), nb::arg("shape"), nb::arg("iota_dimension"));
  ops.def("Iota",
          static_cast<XlaOp (*)(XlaBuilder*, PrimitiveType, int64_t)>(&Iota),
          nb::arg("builder"), nb::arg("type"), nb::arg("size"));
  ops.def(
      "LU",
      [](XlaOp a) -> std::tuple<XlaOp, XlaOp, XlaOp> {
        LuDecompositionResult lu = LuDecomposition(a);
        return std::make_tuple(lu.lu, lu.pivots, lu.permutation);
      },
      nb::arg("operand"));
  ops.def("Map", &Map, nb::arg("builder"), nb::arg("operands"),
          nb::arg("computation"), nb::arg("dimensions"),
          nb::arg("static_operands") = nb::list());
  ops.def("NextAfter", &NextAfter, nb::arg("from"), nb::arg("to"));
  ops.def("OutfeedWithToken", &OutfeedWithToken, nb::arg("operand"),
          nb::arg("token"), nb::arg("shape_with_layout"),
          nb::arg("outfeed_config") = "");
  ops.def("Pad", &Pad, nb::arg("operand"), nb::arg("padding_value"),
          nb::arg("padding_config"));
  ops.def("Parameter",
          static_cast<XlaOp (*)(XlaBuilder*, int64_t, const Shape&,
                                const std::string&, const std::vector<bool>&)>(
              &Parameter),
          nb::arg("builder"), nb::arg("parameter_number"), nb::arg("shape"),
          nb::arg("name") = "",
          nb::arg("replicated_at_leaf_buffers") = std::vector<bool>());
  ops.def("ProductOfElementaryHouseholderReflectors",
          &ProductOfElementaryHouseholderReflectors, nb::arg("a"),
          nb::arg("taus"));
  ops.def(
      "QR",
      [](XlaOp a, bool full_matrices) -> std::pair<XlaOp, XlaOp> {
        XlaOp q, r;
        QrExplicit(a, full_matrices, q, r);
        return std::make_pair(q, r);
      },
      nb::arg("operand"), nb::arg("full_matrices"));
  ops.def(
      "QrDecomposition",
      [](XlaOp a) -> std::pair<XlaOp, XlaOp> {
        QrDecomposition d = Qr(a);
        return std::make_pair(d.q_and_r, d.taus);
      },
      nb::arg("operand"));
  ops.def("RecvFromHost", &RecvFromHost, nb::arg("token"), nb::arg("shape"),
          nb::arg("handle"));
  ops.def("Reduce",
          static_cast<XlaOp (*)(XlaBuilder*, absl::Span<const XlaOp>,
                                absl::Span<const XlaOp>, const XlaComputation&,
                                absl::Span<const int64_t>)>(&Reduce),
          nb::arg("builder"), nb::arg("operands"), nb::arg("init_values"),
          nb::arg("computation"), nb::arg("dimensions_to_reduce"));
  ops.def("ReducePrecision", &ReducePrecision, nb::arg("operand"),
          nb::arg("exponent_bits"), nb::arg("mantissa_bits"));
  ops.def("ReduceWindowWithGeneralPadding",
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
  ops.def("ReduceWindowWithGeneralPadding",
          static_cast<XlaOp (*)(
              absl::Span<const XlaOp>, absl::Span<const XlaOp>,
              const XlaComputation&, absl::Span<const int64_t>,
              absl::Span<const int64_t>, absl::Span<const int64_t>,
              absl::Span<const int64_t>,
              absl::Span<const std::pair<int64_t, int64_t>>)>(
              &ReduceWindowWithGeneralPadding),
          nb::arg("operands"), nb::arg("init_values"), nb::arg("computation"),
          nb::arg("window_dimensions"), nb::arg("window_strides"),
          nb::arg("base_dilations"), nb::arg("window_dilations"),
          nb::arg("padding"));
  ops.def("RemoveDynamicDimension", &RemoveDynamicDimension, nb::arg("operand"),
          nb::arg("dimension"));
  ops.def("ReplicaId", &ReplicaId, nb::arg("builder"));
  ops.def("Reshape",
          static_cast<XlaOp (*)(XlaOp, absl::Span<const int64_t>,
                                absl::Span<const int64_t>)>(&Reshape),
          nb::arg("operand"), nb::arg("dimensions"), nb::arg("new_sizes"));
  ops.def("Reshape",
          static_cast<XlaOp (*)(XlaOp, absl::Span<const int64_t>)>(&Reshape),
          nb::arg("operand"), nb::arg("new_sizes"));
  ops.def("Rev", &Rev, nb::arg("operand"), nb::arg("dimensions"));
  ops.def("RngBitGenerator", &RngBitGenerator, nb::arg("algorithm"),
          nb::arg("initial_state"), nb::arg("shape"));
  ops.def("RngNormal", &RngNormal, nb::arg("mu"), nb::arg("sigma"),
          nb::arg("shape"));
  ops.def("RngUniform", &RngUniform, nb::arg("a"), nb::arg("b"),
          nb::arg("shape"));
  ops.def("Scatter",
          static_cast<XlaOp (*)(XlaOp, XlaOp, XlaOp, const XlaComputation&,
                                const ScatterDimensionNumbers&, bool, bool)>(
              &Scatter),
          nb::arg("input"), nb::arg("scatter_indices"), nb::arg("updates"),
          nb::arg("update_computation"), nb::arg("dimension_numbers"),
          nb::arg("indices_are_sorted") = false,
          nb::arg("unique_indices") = false);
  ops.def("Scatter",
          static_cast<XlaOp (*)(absl::Span<const XlaOp>, XlaOp,
                                absl::Span<const XlaOp>, const XlaComputation&,
                                const ScatterDimensionNumbers&, bool, bool)>(
              &Scatter),
          nb::arg("inputs"), nb::arg("scatter_indices"), nb::arg("updates"),
          nb::arg("update_computation"), nb::arg("dimension_numbers"),
          nb::arg("indices_are_sorted") = false,
          nb::arg("unique_indices") = false);
  ops.def("Select", &Select, nb::arg("pred"), nb::arg("on_true"),
          nb::arg("on_false"));
  ops.def("SelectAndScatterWithGeneralPadding",
          &SelectAndScatterWithGeneralPadding, nb::arg("operand"),
          nb::arg("select"), nb::arg("window_dimensions"),
          nb::arg("window_strides"), nb::arg("padding"), nb::arg("source"),
          nb::arg("init_value"), nb::arg("scatter"));
  ops.def("SendToHost", &SendToHost, nb::arg("operand"), nb::arg("token"),
          nb::arg("shape_with_layout"), nb::arg("handle"));
  ops.def("SetDimensionSize", &SetDimensionSize, nb::arg("operand"),
          nb::arg("val"), nb::arg("dimension"));
  ops.def("Slice", &Slice, nb::arg("operand"), nb::arg("start_indices"),
          nb::arg("limit_indices"), nb::arg("strides"));
  ops.def("SliceInDim", &SliceInDim, nb::arg("operand"), nb::arg("start_index"),
          nb::arg("limit_index"), nb::arg("stride"), nb::arg("dimno"));
  ops.def(
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
          } else {
            return Sort(operands,
                        CreateScalarLtComputation(operand_types, builder),
                        dimension, is_stable);
          }
        });
      },
      nb::arg("builder"), nb::arg("operands"),
      nb::arg("comparator") = std::nullopt, nb::arg("dimension") = -1,
      nb::arg("is_stable") = false);
  ops.def(
      "SVD",
      [](XlaOp a, int64_t max_iter,
         float epsilon) -> std::tuple<XlaOp, XlaOp, XlaOp> {
        auto svd = SVD(a, max_iter, epsilon);
        return std::make_tuple(svd.u, svd.d, svd.v);
      },
      nb::arg("a"), nb::arg("max_iter") = 100, nb::arg("epsilon") = 1e-6);
  ops.def(
      "TopK",
      [](XlaOp input, int64_t k) {
        return TopK(input, k, /*index_type=*/PrimitiveType::S32);
      },
      nb::arg("input"), nb::arg("k"));
  ops.def("Transpose", &Transpose, nb::arg("operand"), nb::arg("permutation"));
  ops.def("TriangularSolve", &TriangularSolve, nb::arg("a"), nb::arg("b"),
          nb::arg("left_side"), nb::arg("lower"), nb::arg("unit_diagonal"),
          nb::arg("transpose_a"));
  ops.def("Tuple", &Tuple, nb::arg("builder"), nb::arg("elements"));
  ops.def("While", &While, nb::arg("condition"), nb::arg("body"),
          nb::arg("init"));

  ops.def("Igamma", &Igamma, nb::arg("a"), nb::arg("x"));
  ops.def("Igammac", &Igammac, nb::arg("a"), nb::arg("x"));
  ops.def("IgammaGradA", &IgammaGradA, nb::arg("a"), nb::arg("x"));
  ops.def("RandomGammaGrad", &RandomGammaGrad, nb::arg("a"), nb::arg("x"));
  ops.def("RegularizedIncompleteBeta", &RegularizedIncompleteBeta, nb::arg("a"),
          nb::arg("b"), nb::arg("x"));
  ops.def("Zeta", &Zeta, nb::arg("x"), nb::arg("q"));

#define BINARY_OP(op)                                                  \
  ops.def(                                                             \
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

#define UNARY_OP(op) ops.def(#op, &op)
  UNARY_OP(Not);
  UNARY_OP(PopulationCount);
  UNARY_OP(Clz);
  UNARY_OP(Abs);
  UNARY_OP(Exp);
  UNARY_OP(Expm1);
  UNARY_OP(Floor);
  UNARY_OP(Ceil);
  UNARY_OP(Round);
  UNARY_OP(Log);
  UNARY_OP(Log1p);
  UNARY_OP(Sign);
  UNARY_OP(Cos);
  UNARY_OP(Sin);
  UNARY_OP(Tan);
  UNARY_OP(Tanh);
  UNARY_OP(IsFinite);
  UNARY_OP(Neg);
  UNARY_OP(Sqrt);
  UNARY_OP(Rsqrt);
  UNARY_OP(Cbrt);
  UNARY_OP(Square);
  UNARY_OP(Reciprocal);
  UNARY_OP(Erfc);
  UNARY_OP(Erf);
  UNARY_OP(ErfInv);
  UNARY_OP(Lgamma);
  UNARY_OP(Digamma);
  UNARY_OP(BesselI0e);
  UNARY_OP(BesselI1e);
  UNARY_OP(Acos);
  UNARY_OP(Asin);
  UNARY_OP(Atan);
  UNARY_OP(Tan);
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
}

}  // namespace xla
