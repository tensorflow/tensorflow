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

#include <cstdint>
#include <string>
#include <vector>

#include "absl/base/casts.h"
#include "absl/hash/hash.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/optional.h"
#include "absl/types/span.h"
#include "pybind11/cast.h"
#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"
#include "pybind11/pytypes.h"
#include "tensorflow/compiler/xla/client/client_library.h"
#include "tensorflow/compiler/xla/client/lib/comparators.h"
#include "tensorflow/compiler/xla/client/lib/math.h"
#include "tensorflow/compiler/xla/client/lib/qr.h"
#include "tensorflow/compiler/xla/client/lib/self_adjoint_eig.h"
#include "tensorflow/compiler/xla/client/lib/sorting.h"
#include "tensorflow/compiler/xla/client/lib/svd.h"
#include "tensorflow/compiler/xla/client/local_client.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/client/xla_computation.h"
#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/python/bfloat16.h"
#include "tensorflow/compiler/xla/python/cpu_device.h"
#include "tensorflow/compiler/xla/python/distributed/client.h"
#include "tensorflow/compiler/xla/python/distributed/distributed.h"
#include "tensorflow/compiler/xla/python/distributed/service.h"
#include "tensorflow/compiler/xla/python/dlpack.h"
#include "tensorflow/compiler/xla/python/local_client.h"
#include "tensorflow/compiler/xla/python/nvidia_gpu_device.h"
#include "tensorflow/compiler/xla/python/python_ref_manager.h"
#include "tensorflow/compiler/xla/python/types.h"
#include "tensorflow/compiler/xla/service/custom_call_target_registry.h"
#include "tensorflow/compiler/xla/service/hlo_graph_dumper.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_parser.h"
#include "tensorflow/compiler/xla/service/name_uniquer.h"
#include "tensorflow/compiler/xla/service/platform_util.h"
#include "tensorflow/compiler/xla/shape.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/profiler/lib/traceme.h"
#include "tensorflow/core/profiler/rpc/profiler_server.h"
#include "tensorflow/stream_executor/platform.h"

namespace xla {

namespace py = pybind11;

namespace {

struct Uniquer {
  absl::Mutex mu;
  NameUniquer name_uniquer TF_GUARDED_BY(mu);
};

Uniquer* GetUniquer() {
  static Uniquer* uniquer = new Uniquer;
  return uniquer;
}

static std::string UniquifyName(const std::string& name) {
  Uniquer* uniquer = GetUniquer();
  absl::MutexLock lock(&uniquer->mu);
  return uniquer->name_uniquer.GetUniqueName(name);
}

// Converts a computation to a serialized HloModuleProto.
StatusOr<py::bytes> GetComputationSerializedProto(
    const XlaComputation& computation) {
  std::string result;
  if (!computation.proto().SerializeToString(&result)) {
    return Unknown("Failed to serialize the HloModuleProto.");
  }
  return py::bytes(result);
}

StatusOr<std::shared_ptr<HloModule>> GetHloModule(
    const XlaComputation& computation) {
  TF_ASSIGN_OR_RETURN(const HloModuleConfig module_config,
                      HloModule::CreateModuleConfigFromProto(
                          computation.proto(), GetDebugOptionsFromFlags()));
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<HloModule> module,
      HloModule::CreateFromProto(computation.proto(), module_config));
  return std::shared_ptr<HloModule>(std::move(module));
}

// Converts a computation to textual HLO form.
StatusOr<std::string> GetComputationHloText(const XlaComputation& computation) {
  TF_ASSIGN_OR_RETURN(std::shared_ptr<HloModule> hlo_module,
                      GetHloModule(computation));
  HloPrintOptions options;
  options = HloPrintOptions::ShortParsable();
  options.set_print_large_constants(false);
  return hlo_module->ToString(options);
}

// Converts a computation to HLO dot graph form.
StatusOr<std::string> GetComputationHloDotGraph(
    const XlaComputation& computation) {
  TF_ASSIGN_OR_RETURN(std::shared_ptr<HloModule> hlo_module,
                      GetHloModule(computation));
  return RenderGraph(*hlo_module->entry_computation(), /*label=*/"",
                     hlo_module->config().debug_options(),
                     RenderedGraphFormat::kDot);
}

// Hashes the HLO module.
StatusOr<uint64> HashComputation(const XlaComputation& computation) {
  TF_ASSIGN_OR_RETURN(std::shared_ptr<HloModule> hlo_module,
                      GetHloModule(computation));
  return hlo_module->Hash();
}

// Registers a 'fn_capsule' as a CPU custom call target.
// 'fn_capsule' must be a void* pointer encapsulated in a PyCapsule object,
// with name "xla._CUSTOM_CALL_TARGET".
// 'platform' is an XLA platform name, e.g., "Host" or "CUDA".
Status PyRegisterCustomCallTarget(const std::string& fn_name,
                                  py::capsule capsule,
                                  const std::string& platform) {
  static const char* const kName = "xla._CUSTOM_CALL_TARGET";
  // TODO(phawkins): remove old name after fixing users.
  static const char* const kOldCpuName = "xla._CPU_CUSTOM_CALL_TARGET";
  if (absl::string_view(capsule.name()) != kName &&
      absl::string_view(capsule.name()) != kOldCpuName) {
    return InvalidArgument(
        "Argument to RegisterCustomCallTargetRegistry was not a "
        "xla._CUSTOM_CALL_TARGET capsule.");
  }
  CustomCallTargetRegistry::Global()->Register(
      fn_name, static_cast<void*>(capsule), platform);
  return Status::OK();
}

// PEP 3118 buffer protocol implementation.

// Extra data to be kept alive by the consumer of the buffer protocol.
struct ExtraBufferInfo {
  explicit ExtraBufferInfo(PyLocalBuffer::ScopedHold device_buffer)
      : device_buffer(std::move(device_buffer)) {}

  std::string format;
  std::vector<Py_ssize_t> strides;
  // We keep a reference to the TrackedDeviceBuffer that backs the
  // PyLocalBuffer. This prevents a use-after-free in the event that Delete() is
  // called on a buffer with an live buffer protocol view. It does however mean
  // that Delete() sometimes won't actually delete immediately.
  PyLocalBuffer::ScopedHold device_buffer;
};

int PyLocalBufferGetBuffer(PyObject* exporter, Py_buffer* view, int flags) {
  auto& buffer =
      py::reinterpret_borrow<py::object>(exporter).cast<PyLocalBuffer&>();
  Status status = [&]() {
    // Py_buffer objects are POD C structures, so we don't need to hold the GIL.
    // Additionally we call BlockHostUntilReady() below, which may block.
    py::gil_scoped_release gil_release;

    if (buffer.device()->platform_name() != "cpu") {
      return InvalidArgument(
          "Python buffer protocol is only defined for CPU buffers.");
    }
    if (!buffer.on_device_shape().IsArray()) {
      return InvalidArgument(
          "Python buffer protocol is only defined for array buffers.");
    }
    // If we allowed exports of formatted BF16 buffers, consumers would get
    // confused about the type because there is no way to describe BF16 to
    // Python.
    if (buffer.on_host_shape().element_type() == BF16 &&
        ((flags & PyBUF_FORMAT) == PyBUF_FORMAT)) {
      return InvalidArgument(
          "bfloat16 buffer format not supported by Python buffer protocol.");
    }
    if ((flags & PyBUF_WRITEABLE) == PyBUF_WRITEABLE) {
      return InvalidArgument("XLA buffers are read-only.");
    }
    PyLocalBuffer::ScopedHold device_buffer(
        buffer.GetBufferWithExternalReference());
    if (!device_buffer.status().ok()) {
      return InvalidArgument("Deleted buffer used in buffer protocol.");
    }
    const Shape& shape = buffer.on_host_shape();
    if (((flags & PyBUF_C_CONTIGUOUS) == PyBUF_C_CONTIGUOUS ||
         (flags & PyBUF_STRIDES) == PyBUF_ND) &&
        !LayoutUtil::IsMonotonicWithDim0Major(shape.layout())) {
      return InvalidArgument("Buffer is not in C-contiguous layout.");
    } else if ((flags & PyBUF_F_CONTIGUOUS) == PyBUF_F_CONTIGUOUS &&
               !LayoutUtil::IsMonotonicWithDim0Minor(shape.layout())) {
      return InvalidArgument("Buffer is not in F-contiguous layout.");
    } else if ((flags & PyBUF_ANY_CONTIGUOUS) == PyBUF_ANY_CONTIGUOUS &&
               !LayoutUtil::IsMonotonicWithDim0Major(shape.layout()) &&
               !LayoutUtil::IsMonotonicWithDim0Minor(shape.layout())) {
      return InvalidArgument("Buffer is not in contiguous layout.");
    }
    std::memset(view, 0, sizeof(Py_buffer));
    CHECK_EQ(device_buffer->device_memory().size(), 1);
    view->buf =
        const_cast<void*>(device_buffer->device_memory().front().opaque());
    auto extra = absl::make_unique<ExtraBufferInfo>(std::move(device_buffer));
    view->itemsize = ShapeUtil::ByteSizeOfPrimitiveType(shape.element_type());
    view->len = ShapeUtil::ByteSizeOf(shape);
    view->readonly = 1;
    if ((flags & PyBUF_FORMAT) == PyBUF_FORMAT) {
      TF_ASSIGN_OR_RETURN(extra->format, FormatDescriptorForPrimitiveType(
                                             shape.element_type()));
      view->format = const_cast<char*>(extra->format.c_str());
    }
    if ((flags & PyBUF_ND) == PyBUF_ND) {
      view->ndim = shape.dimensions_size();
      static_assert(sizeof(int64) == sizeof(Py_ssize_t),
                    "Py_ssize_t must be 64 bits");
      if (view->ndim != 0) {
        view->shape = reinterpret_cast<Py_ssize_t*>(
            const_cast<int64*>(shape.dimensions().data()));
        if ((flags & PyBUF_STRIDES) == PyBUF_STRIDES) {
          extra->strides = ByteStridesForShape(shape);
          view->strides = extra->strides.data();
        }
      }
    }
    TF_RETURN_IF_ERROR(buffer.BlockHostUntilReady());
    view->internal = extra.release();
    return Status::OK();
  }();
  if (!status.ok()) {
    PyErr_SetString(PyExc_BufferError, status.ToString().c_str());
    return -1;
  }
  view->obj = exporter;
  Py_INCREF(view->obj);
  return 0;
}

void PyLocalBufferReleaseBuffer(PyObject*, Py_buffer* buffer) {
  auto extra = static_cast<ExtraBufferInfo*>(buffer->internal);
  delete extra;
}

PyBufferProcs PyLocalBufferProcs = []() {
  PyBufferProcs procs;
  procs.bf_getbuffer = &PyLocalBufferGetBuffer;
  procs.bf_releasebuffer = &PyLocalBufferReleaseBuffer;
  return procs;
}();

// Implementation of the CUDA array interface for sharing GPU buffers with other
// Python libraries.
StatusOr<py::dict> PyLocalBufferCudaArrayInterface(
    const PyLocalBuffer& buffer) {
  if (buffer.device()->local_device_state()->executor()->platform_kind() !=
      se::PlatformKind::kCuda) {
    return InvalidArgument(
        "__cuda_array_interface__ is only defined for NVidia GPU buffers.");
  }
  if (!buffer.on_device_shape().IsArray()) {
    return InvalidArgument(
        "__cuda_array_interface__ is only defined for array buffers.");
  }
  if (buffer.on_host_shape().element_type() == BF16) {
    return InvalidArgument(
        "__cuda_array_interface__ is not supported for bfloat16 buffers.");
  }
  TF_RET_CHECK(
      LayoutUtil::IsMonotonicWithDim0Major(buffer.on_host_shape().layout()));
  TF_ASSIGN_OR_RETURN(ShapedBuffer shaped_buffer, buffer.AsShapedBuffer());

  py::dict result;
  result["shape"] = IntSpanToTuple(shaped_buffer.on_host_shape().dimensions());
  TF_ASSIGN_OR_RETURN(py::str typestr,
                      TypeDescriptorForPrimitiveType(
                          shaped_buffer.on_host_shape().element_type()));
  result["typestr"] = std::move(typestr);
  py::tuple data(2);
  data[0] = py::int_(
      absl::bit_cast<std::uintptr_t>(shaped_buffer.root_buffer().opaque()));
  data[1] = py::bool_(true);  // read-only
  result["data"] = std::move(data);
  result["version"] = py::int_(2);
  return result;
}

void BuildOpsSubmodule(py::module* m) {
  // ops submodule, containing free functions that add operators to an
  // XlaBuilder.
  py::module ops = m->def_submodule("ops", "XLA operations");

  py::enum_<TriangularSolveOptions::Transpose>(
      ops, "TriangularSolveOptions_Transpose")
      .value("TRANSPOSE_INVALID", TriangularSolveOptions::TRANSPOSE_INVALID)
      .value("NO_TRANSPOSE", TriangularSolveOptions::NO_TRANSPOSE)
      .value("TRANSPOSE", TriangularSolveOptions::TRANSPOSE)
      .value("ADJOINT", TriangularSolveOptions::ADJOINT);

  ops.def("AfterAll", &AfterAll, py::arg("builder"), py::arg("tokens"));
  ops.def(
      "AllReduce",
      static_cast<XlaOp (*)(
          XlaOp, const XlaComputation&, absl::Span<const ReplicaGroup>,
          const absl::optional<ChannelHandle>&, const absl::optional<Shape>&)>(
          &AllReduce),
      py::arg("operand"), py::arg("computation"),
      py::arg("replica_groups") = py::list(),
      py::arg("channel_id") = absl::nullopt,
      py::arg("shape_with_layout") = absl::nullopt);
  ops.def("AllToAll", &AllToAll, py::arg("operand"), py::arg("split_dimension"),
          py::arg("concat_dimension"), py::arg("split_count"),
          py::arg("replica_groups") = py::list());
  ops.def("CollectivePermute", &CollectivePermute, py::arg("operand"),
          py::arg("source_target_pairs"));
  ops.def("CreateToken", &CreateToken, py::arg("builder"));
  ops.def("CrossReplicaSum",
          static_cast<XlaOp (*)(XlaOp, absl::Span<const ReplicaGroup>)>(
              &CrossReplicaSum),
          py::arg("operand"), py::arg("replica_groups") = py::list());
  ops.def("BitcastConvertType", &BitcastConvertType, py::arg("operand"),
          py::arg("new_element_type"));
  ops.def("Broadcast", &Broadcast, py::arg("operand"), py::arg("sizes"));
  ops.def("BroadcastInDim", &BroadcastInDim, py::arg("operand"),
          py::arg("shape"), py::arg("broadcast_dimensions"));
  ops.def("Call", &Call, py::arg("builder"), py::arg("computation"),
          py::arg("operands"));
  ops.def("Cholesky", &Cholesky, py::arg("a"), py::arg("lower") = true);
  ops.def("Clamp", &Clamp, py::arg("min"), py::arg("operand"), py::arg("max"));
  ops.def("Collapse", &Collapse, py::arg("operand"), py::arg("dimensions"));
  ops.def("ConcatInDim", &ConcatInDim, py::arg("builder"), py::arg("operands"),
          py::arg("dimension"));
  ops.def("Conditional",
          static_cast<XlaOp (*)(XlaOp, absl::Span<const XlaComputation* const>,
                                absl::Span<const XlaOp>)>(&Conditional),
          py::arg("branch_index"), py::arg("branch_computations"),
          py::arg("branch_operands"));
  ops.def("Conditional",
          static_cast<XlaOp (*)(XlaOp, XlaOp, const XlaComputation&, XlaOp,
                                const XlaComputation&)>(&Conditional),
          py::arg("predicate"), py::arg("true_operand"),
          py::arg("true_computation"), py::arg("false_operand"),
          py::arg("false_computation"));
  ops.def("Constant", &ConstantLiteral, py::arg("builder"), py::arg("literal"));
  ops.def("ConstantLiteral", &ConstantLiteral, py::arg("builder"),
          py::arg("literal"));
  ops.def("ConvGeneralDilated", &ConvGeneralDilated, py::arg("lhs"),
          py::arg("rhs"), py::arg("window_strides"), py::arg("padding"),
          py::arg("lhs_dilation"), py::arg("rhs_dilation"),
          py::arg("dimension_numbers"), py::arg("feature_group_count") = 1,
          py::arg("batch_group_count") = 1,
          py::arg("precision_config") = nullptr);
  ops.def("ConvertElementType", &ConvertElementType, py::arg("operand"),
          py::arg("new_element_type"));
  ops.def(
      "CustomCall",
      [](XlaBuilder* builder, const py::bytes& call_target_name,
         absl::Span<const XlaOp> operands, const Shape& shape,
         const py::bytes& opaque) -> XlaOp {
        return CustomCall(builder, call_target_name, operands, shape, opaque);
      },
      py::arg("builder"), py::arg("call_target_name"), py::arg("operands"),
      py::arg("shape"), py::arg("opaque") = py::bytes(""));
  ops.def(
      "CustomCallWithLayout",
      [](XlaBuilder* builder, const py::bytes& call_target_name,
         absl::Span<const XlaOp> operands, const Shape& shape_with_layout,
         absl::Span<const Shape> operand_shapes_with_layout,
         const py::bytes& opaque) -> XlaOp {
        return CustomCallWithLayout(builder, call_target_name, operands,
                                    shape_with_layout,
                                    operand_shapes_with_layout, opaque);
      },
      py::arg("builder"), py::arg("call_target_name"), py::arg("operands"),
      py::arg("shape_with_layout"), py::arg("operand_shapes_with_layout"),
      py::arg("opaque") = py::bytes(""));
  ops.def("Dot", &Dot, py::arg("lhs"), py::arg("rhs"),
          py::arg("precision_config") = nullptr);
  ops.def("DotGeneral", &DotGeneral, py::arg("lhs"), py::arg("rhs"),
          py::arg("dimension_numbers"), py::arg("precision_config") = nullptr);
  ops.def("DynamicSlice",
          static_cast<XlaOp (*)(XlaOp, absl::Span<const XlaOp>,
                                absl::Span<const int64>)>(&DynamicSlice),
          py::arg("operand"), py::arg("start_indices"), py::arg("slice_sizes"));
  ops.def("DynamicUpdateSlice",
          static_cast<XlaOp (*)(XlaOp, XlaOp, absl::Span<const XlaOp>)>(
              &DynamicUpdateSlice),
          py::arg("operand"), py::arg("update"), py::arg("start_indices"));

  ops.def("Fft", &Fft, py::arg("operand"), py::arg("fft_type"),
          py::arg("fft_length"));

  ops.def("Gather", &Gather, py::arg("a"), py::arg("start_indices"),
          py::arg("dimension_numbers"), py::arg("slice_sizes"),
          py::arg("indices_are_sorted") = false);
  ops.def("GetTupleElement", &GetTupleElement, py::arg("tuple_data"),
          py::arg("index"));
  ops.def("InfeedWithToken", &InfeedWithToken, py::arg("token"),
          py::arg("shape"), py::arg("config") = "");
  ops.def("Iota",
          static_cast<XlaOp (*)(XlaBuilder*, const Shape&, int64)>(&Iota),
          py::arg("builder"), py::arg("shape"), py::arg("iota_dimension"));
  ops.def("Iota",
          static_cast<XlaOp (*)(XlaBuilder*, PrimitiveType, int64)>(&Iota),
          py::arg("builder"), py::arg("type"), py::arg("size"));
  ops.def("Map", &Map, py::arg("builder"), py::arg("operands"),
          py::arg("computation"), py::arg("dimensions"),
          py::arg("static_operands") = py::list());
  ops.def("NextAfter", &NextAfter, py::arg("from"), py::arg("to"));
  ops.def("OutfeedWithToken", &OutfeedWithToken, py::arg("operand"),
          py::arg("token"), py::arg("shape_with_layout"),
          py::arg("outfeed_config") = "");
  ops.def("Pad", &Pad, py::arg("operand"), py::arg("padding_value"),
          py::arg("padding_config"));
  ops.def("Parameter",
          static_cast<XlaOp (*)(XlaBuilder*, int64, const Shape&,
                                const std::string&, const std::vector<bool>&)>(
              &Parameter),
          py::arg("builder"), py::arg("parameter_number"), py::arg("shape"),
          py::arg("name") = "",
          py::arg("replicated_at_leaf_buffers") = std::vector<bool>());
  ops.def(
      "QR",
      [](XlaOp a, bool full_matrices) -> StatusOr<std::pair<XlaOp, XlaOp>> {
        TF_ASSIGN_OR_RETURN(auto qr, QRDecomposition(a, full_matrices));
        return std::make_pair(qr.q, qr.r);
      },
      py::arg("operand"), py::arg("full_matrices"));
  ops.def(
      "Eigh",
      [](XlaOp a, bool lower, int64 max_iter,
         float epsilon) -> std::pair<XlaOp, XlaOp> {
        auto eigh = SelfAdjointEig(a, lower, max_iter, epsilon);
        return std::make_pair(eigh.v, eigh.w);
      },
      py::arg("a"), py::arg("lower") = true, py::arg("max_iter") = 100,
      py::arg("epsilon") = 1e-6);
  ops.def(
      "SVD",
      [](XlaOp a, int64 max_iter,
         float epsilon) -> std::tuple<XlaOp, XlaOp, XlaOp> {
        auto svd = SVD(a, max_iter, epsilon);
        return std::make_tuple(svd.u, svd.d, svd.v);
      },
      py::arg("a"), py::arg("max_iter") = 100, py::arg("epsilon") = 1e-6);
  ops.def("Reduce",
          static_cast<XlaOp (*)(XlaBuilder*, absl::Span<const XlaOp>,
                                absl::Span<const XlaOp>, const XlaComputation&,
                                absl::Span<const int64>)>(&Reduce),
          py::arg("builder"), py::arg("operands"), py::arg("init_values"),
          py::arg("computation"), py::arg("dimensions_to_reduce"));
  ops.def("ReducePrecision", &ReducePrecision, py::arg("operand"),
          py::arg("exponent_bits"), py::arg("mantissa_bits"));
  ops.def("ReduceWindowWithGeneralPadding", &ReduceWindowWithGeneralPadding,
          py::arg("operand"), py::arg("init_value"), py::arg("computation"),
          py::arg("window_dimensions"), py::arg("window_strides"),
          py::arg("base_dilations"), py::arg("window_dilations"),
          py::arg("padding"));
  ops.def("ReplicaId", &ReplicaId, py::arg("builder"));
  ops.def("Reshape",
          static_cast<XlaOp (*)(XlaOp, absl::Span<const int64>,
                                absl::Span<const int64>)>(&Reshape),
          py::arg("operand"), py::arg("dimensions"), py::arg("new_sizes"));
  ops.def("Reshape",
          static_cast<XlaOp (*)(XlaOp, absl::Span<const int64>)>(&Reshape),
          py::arg("operand"), py::arg("new_sizes"));
  ops.def("Rev", &Rev, py::arg("operand"), py::arg("dimensions"));
  ops.def("RngNormal", &RngNormal, py::arg("mu"), py::arg("sigma"),
          py::arg("shape"));
  ops.def("RngUniform", &RngUniform, py::arg("a"), py::arg("b"),
          py::arg("shape"));
  ops.def("Scatter", &Scatter, py::arg("input"), py::arg("scatter_indices"),
          py::arg("updates"), py::arg("update_computation"),
          py::arg("dimension_numbers"), py::arg("indices_are_sorted") = false,
          py::arg("unique_indices") = false);
  ops.def("Select", &Select, py::arg("pred"), py::arg("on_true"),
          py::arg("on_false"));
  ops.def("SelectAndScatterWithGeneralPadding",
          &SelectAndScatterWithGeneralPadding, py::arg("operand"),
          py::arg("select"), py::arg("window_dimensions"),
          py::arg("window_strides"), py::arg("padding"), py::arg("source"),
          py::arg("init_value"), py::arg("scatter"));
  ops.def("Slice", &Slice, py::arg("operand"), py::arg("start_indices"),
          py::arg("limit_indices"), py::arg("strides"));
  ops.def("SliceInDim", &SliceInDim, py::arg("operand"), py::arg("start_index"),
          py::arg("limit_index"), py::arg("stride"), py::arg("dimno"));
  ops.def(
      "Sort",
      [](XlaBuilder* builder, absl::Span<const XlaOp> operands,
         absl::optional<const XlaComputation*> comparator, int64 dimension,
         bool is_stable) -> XlaOp {
        return builder->ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
          std::vector<PrimitiveType> operand_types;
          for (const auto& operand : operands) {
            TF_ASSIGN_OR_RETURN(auto operand_shape, builder->GetShape(operand));
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
      py::arg("builder"), py::arg("operands"),
      py::arg("comparator") = absl::nullopt, py::arg("dimension") = -1,
      py::arg("is_stable") = false);
  ops.def("TopK", &TopK, py::arg("input"), py::arg("k"));
  ops.def("Transpose", &Transpose, py::arg("operand"), py::arg("permutation"));
  ops.def("TriangularSolve", &TriangularSolve, py::arg("a"), py::arg("b"),
          py::arg("left_side"), py::arg("lower"), py::arg("unit_diagonal"),
          py::arg("transpose_a"));
  ops.def("Tuple", &Tuple, py::arg("builder"), py::arg("elements"));
  ops.def("While", &While, py::arg("condition"), py::arg("body"),
          py::arg("init"));

  ops.def("Igamma", &Igamma, py::arg("a"), py::arg("x"));
  ops.def("Igammac", &Igammac, py::arg("a"), py::arg("x"));
  ops.def("IgammaGradA", &IgammaGradA, py::arg("a"), py::arg("x"));
  ops.def("RandomGammaGrad", &RandomGammaGrad, py::arg("a"), py::arg("x"));
  ops.def("RegularizedIncompleteBeta", &RegularizedIncompleteBeta, py::arg("a"),
          py::arg("b"), py::arg("x"));

#define BINARY_OP(op)                                                 \
  ops.def(                                                            \
      #op,                                                            \
      [](XlaOp a, XlaOp b, absl::optional<std::vector<int64>> dims) { \
        return dims ? op(a, b, *dims) : op(a, b);                     \
      },                                                              \
      py::arg("lhs"), py::arg("rhs"),                                 \
      py::arg("broadcast_dimensions") = absl::nullopt)
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
  UNARY_OP(Tanh);
  UNARY_OP(IsFinite);
  UNARY_OP(Neg);
  UNARY_OP(Sqrt);
  UNARY_OP(Rsqrt);
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
#undef UNARY_OP
}

// Helper to implement TraceMe as a context manager in Python.
class TraceMeContextManager {
 public:
  explicit TraceMeContextManager(py::str name, py::kwargs kwargs)
      : name_(std::move(name)), kwargs_(std::move(kwargs)) {}

  void Enter() {
    if (IsEnabled()) {
      std::string name(name_);
      if (!kwargs_.empty()) {
        absl::StrAppend(&name, "#");
        bool first = true;
        for (const auto entry : kwargs_) {
          absl::StrAppend(&name, first ? "" : ",",
                          std::string(py::str(entry.first)), "=",
                          std::string(py::str(entry.second)));
          first = false;
        }
        absl::StrAppend(&name, "#");
      }
      traceme_.emplace(std::move(name));
    }
  }
  py::object Exit(const py::object& ex_type, const py::object& ex_value,
                  const py::object& traceback) {
    traceme_.reset();
    return py::none();
  }

  static bool IsEnabled() { return tensorflow::profiler::TraceMe::Active(); }

 private:
  py::str name_;
  py::kwargs kwargs_;
  absl::optional<tensorflow::profiler::TraceMe> traceme_;
};

void BuildProfilerSubmodule(py::module* m) {
  py::module profiler =
      m->def_submodule("profiler", "TensorFlow profiler integration");
  py::class_<tensorflow::ProfilerServer,
             std::unique_ptr<tensorflow::ProfilerServer>>
      profiler_server_class(profiler, "ProfilerServer");
  profiler.def(
      "start_server",
      [](int port) -> std::unique_ptr<tensorflow::ProfilerServer> {
        auto server = absl::make_unique<tensorflow::ProfilerServer>();
        server->StartProfilerServer(port);
        return server;
      },
      py::arg("port"));

  py::class_<TraceMeContextManager> traceme_class(profiler, "TraceMe");
  traceme_class.def(py::init<py::str, py::kwargs>())
      .def("__enter__", &TraceMeContextManager::Enter)
      .def("__exit__", &TraceMeContextManager::Exit)
      .def_static("IsEnabled", &TraceMeContextManager::IsEnabled);
}

}  // namespace

PYBIND11_MODULE(xla_extension, m) {
  // Caution: import_array1 works by initializing a static variable
  // (PyArray_API) which is *defined* in a NumPy header. import_array1() must
  // therefore be called from the *same translation unit* as any users of
  // NumPy C APIs.
  auto init_numpy = []() -> bool {
    // import_array1 might look like a function. It's not. It's a macro that
    // calls `return`, which is why we wrap it in this strange-looking lambda.
    import_array1(false);
    return true;
  };
  if (!init_numpy() || !InitializeNumpyAPIForTypes()) {
    throw std::runtime_error("Unable to initialize Numpy API");
  }

  // Types
  py::enum_<PrimitiveType>(m, "PrimitiveType")
      .value("PRIMITIVE_TYPE_INVALID", PRIMITIVE_TYPE_INVALID)
      .value("PRED", PRED)
      .value("S8", S8)
      .value("S16", S16)
      .value("S32", S32)
      .value("S64", S64)
      .value("U8", U8)
      .value("U16", U16)
      .value("U32", U32)
      .value("U64", U64)
      .value("F16", F16)
      .value("BF16", BF16)
      .value("F32", F32)
      .value("F64", F64)
      .value("C64", C64)
      .value("C128", C128)
      .value("TUPLE", TUPLE)
      .value("OPAQUE_TYPE", OPAQUE_TYPE)
      .value("TOKEN", TOKEN);

  m.def("bfloat16_dtype", Bfloat16Dtype);

  // Shapes
  py::class_<Shape> shape_class(m, "Shape");
  shape_class
      .def(py::init([](const string& s) {
        return absl::make_unique<Shape>(ValueOrThrow(ParseShape(s)));
      }))
      .def_static(
          "tuple_shape",
          [](std::vector<Shape> shapes) -> Shape {
            return ShapeUtil::MakeTupleShape(shapes);
          },
          "Constructs a tuple shape.")
      .def_static(
          "array_shape",
          [](PrimitiveType type, py::object dims_seq,
             absl::optional<py::object> layout_seq) -> Shape {
            std::vector<int64> dims = IntSequenceToVector(dims_seq);
            if (layout_seq) {
              std::vector<int64> layout = IntSequenceToVector(*layout_seq);
              return ShapeUtil::MakeShapeWithLayout(type, dims, layout);
            } else {
              Shape shape = ShapeUtil::MakeShape(type, dims);
              shape.clear_layout();
              return shape;
            }
          },
          "Constructs an array shape.", py::arg("type"), py::arg("dims"),
          py::arg("layout") = absl::nullopt)
      .def_static(
          "array_shape",
          [](py::dtype dtype, py::object dims_seq,
             absl::optional<py::object> layout_seq) -> Shape {
            PrimitiveType type = ValueOrThrow(DtypeToPrimitiveType(dtype));
            std::vector<int64> dims = IntSequenceToVector(dims_seq);
            if (layout_seq) {
              std::vector<int64> layout = IntSequenceToVector(*layout_seq);
              return ShapeUtil::MakeShapeWithLayout(type, dims, layout);
            } else {
              Shape shape = ShapeUtil::MakeShape(type, dims);
              shape.clear_layout();
              return shape;
            }
          },
          "Constructs an array shape.", py::arg("type"), py::arg("dims"),
          py::arg("layout") = absl::nullopt)
      .def_static("token_shape", []() { return ShapeUtil::MakeTokenShape(); })
      .def("dimensions",
           [](const Shape& shape) -> py::tuple {
             return IntSpanToTuple(shape.dimensions());
           })
      .def("xla_element_type", &Shape::element_type)
      .def("element_type",
           [](const Shape& shape) {
             return ValueOrThrow(PrimitiveTypeToDtype(shape.element_type()));
           })
      .def("numpy_dtype",
           [](const Shape& shape) {
             if (shape.IsTuple()) {
               return py::dtype("O");
             }
             return ValueOrThrow(PrimitiveTypeToDtype(shape.element_type()));
           })
      .def("is_tuple", &Shape::IsTuple)
      .def("is_array", &Shape::IsArray)
      .def("rank", &Shape::rank)
      .def("to_serialized_proto",
           [](const Shape& shape) {
             ShapeProto proto = shape.ToProto();
             return py::bytes(proto.SerializeAsString());
           })
      .def("tuple_shapes",
           [](const Shape& shape) {
             return std::vector<Shape>(shape.tuple_shapes());
           })
      .def("leaf_count",
           [](const Shape& shape) { return ShapeUtil::GetLeafCount(shape); })
      .def(
          "with_major_to_minor_layout_if_absent",
          [](const Shape& shape) {
            Shape out = shape;
            ShapeUtil::ForEachMutableSubshape(
                &out, [](Shape* subshape, const ShapeIndex&) {
                  if (!subshape->has_layout()) {
                    LayoutUtil::SetToDefaultLayout(subshape);
                  }
                });
            return out;
          },
          "Returns a copy of a shape with missing layouts set to "
          "major-to-minor.")
      .def("__eq__", [](const Shape& shape,
                        const Shape& other) { return shape == other; })
      .def("__ne__", [](const Shape& shape,
                        const Shape& other) { return shape != other; })
      .def("__hash__",
           [](const Shape& shape) { return absl::Hash<Shape>()(shape); })
      .def("__repr__", [](const Shape& shape) {
        return shape.ToString(/*print_layout=*/true);
      });

  py::class_<ProgramShape>(m, "ProgramShape")
      .def(py::init(
          [](absl::Span<const Shape> params, Shape result) -> ProgramShape {
            ProgramShape program_shape;
            for (const Shape& param : params) {
              *program_shape.add_parameters() = param;
            }
            *program_shape.mutable_result() = result;
            return program_shape;
          }))
      .def("parameter_shapes",
           static_cast<const std::vector<Shape>& (ProgramShape::*)() const>(
               &ProgramShape::parameters))
      .def("result_shape", &ProgramShape::result)
      .def("__repr__", &ProgramShape::ToString);

  // Literals
  py::class_<Literal, std::shared_ptr<Literal>>(m, "Literal")
      .def("__repr__", &Literal::ToString);
  py::class_<LiteralSlice> literal_slice(m, "LiteralSlice");
  py::implicitly_convertible<Literal, LiteralSlice>();
  py::implicitly_convertible<BorrowingLiteral, LiteralSlice>();

  // Device assignments
  py::class_<DeviceAssignment>(m, "DeviceAssignment")
      .def_static("create",
                  [](py::array_t<int> array) -> StatusOr<DeviceAssignment> {
                    if (array.ndim() != 2) {
                      return InvalidArgument(
                          "Argument to DeviceAssignment constructor must be a "
                          "2D array, received an %dD array.",
                          array.ndim());
                    }
                    DeviceAssignment result(array.shape(0), array.shape(1));
                    for (int i = 0; i < array.shape(0); ++i) {
                      for (int j = 0; j < array.shape(1); ++j) {
                        result(i, j) = array.at(i, j);
                      }
                    }
                    return result;
                  })
      .def("replica_count", &DeviceAssignment::replica_count)
      .def("computation_count", &DeviceAssignment::computation_count)
      .def("__repr__", &DeviceAssignment::ToString);

  py::class_<Device, ClientAndPtr<Device>>(
      m, "Device",
      "A descriptor of an available device.\n\nSubclasses are used to "
      "represent specific types of devices, e.g. CPUs, GPUs. Subclasses may "
      "have additional properties specific to that device type.")
      .def_property_readonly(
          "id", &Device::id,
          "Integer ID of this device.\n\nUnique across all available devices "
          "of this type, including remote devices on multi-host platforms.")
      .def_property_readonly("host_id", &Device::host_id,
                             "Integer ID of this device's host.\n\n"
                             "This is always 0 except on multi-host platforms.")
      .def_property_readonly("platform", &Device::platform_name)
      .def_property_readonly("device_kind", &Device::device_kind)
      .def("__str__", &Device::DebugString)
      .def("TransferToInfeed",
           [](const Device& device, const LiteralSlice& literal) {
             GlobalPyRefManager()->CollectGarbage();
             py::gil_scoped_release gil_release;
             TF_ASSIGN_OR_RETURN(LocalDeviceState * local_device,
                                 device.GetLocalDeviceState());
             return local_device->client()->TransferToInfeedLocal(
                 literal, local_device->device_ordinal());
           })
      .def(
          "TransferFromOutfeed",
          [](const Device& device, const Shape& shape) -> StatusOr<py::object> {
            GlobalPyRefManager()->CollectGarbage();
            std::shared_ptr<Literal> literal_shared;
            {
              py::gil_scoped_release gil_release;
              TF_ASSIGN_OR_RETURN(LocalDeviceState * local_device,
                                  device.GetLocalDeviceState());
              TF_ASSIGN_OR_RETURN(
                  Literal literal,
                  local_device->client()->TransferFromOutfeedLocal(
                      shape, local_device->device_ordinal()));

              literal_shared = std::make_shared<Literal>(std::move(literal));
            }
            return LiteralToPython(std::move(literal_shared));
          });

  py::class_<CpuDevice, Device, ClientAndPtr<CpuDevice>>(m, "CpuDevice")
      .def("__repr__", [](const CpuDevice& device) {
        return absl::StrFormat("CpuDevice(id=%i)", device.id());
      });

  py::class_<GpuDevice, Device, ClientAndPtr<GpuDevice>>(m, "GpuDevice")
      .def("__repr__", [](const GpuDevice& device) {
        return absl::StrFormat("GpuDevice(id=%i)", device.id());
      });

  // Local XLA client methods.

  // Custom-call targets.
  m.def("RegisterCustomCallTarget", &PyRegisterCustomCallTarget);

  py::class_<GpuAllocatorConfig> alloc_config(m, "GpuAllocatorConfig");
  alloc_config.def(py::init<>())
      .def_readwrite("kind", &GpuAllocatorConfig::kind)
      .def_readwrite("memory_fraction", &GpuAllocatorConfig::memory_fraction)
      .def_readwrite("preallocate", &GpuAllocatorConfig::preallocate);
  py::enum_<GpuAllocatorConfig::Kind>(alloc_config, "Kind")
      .value("DEFAULT", GpuAllocatorConfig::Kind::kDefault)
      .value("PLATFORM", GpuAllocatorConfig::Kind::kPlatform)
      .value("BFC", GpuAllocatorConfig::Kind::kBFC);

  py::class_<PyLocalClient, std::shared_ptr<PyLocalClient>>(m, "LocalClient")
      .def("device_count", &PyLocalClient::device_count)
      .def("local_device_count", &PyLocalClient::local_device_count)
      .def("devices",
           [](std::shared_ptr<PyLocalClient> client) {
             std::vector<ClientAndPtr<Device>> devices;
             devices.reserve(client->devices().size());
             for (const auto& device : client->devices()) {
               devices.push_back(WrapWithClient(client, device.get()));
             }
             return devices;
           })
      .def("local_devices",
           [](std::shared_ptr<PyLocalClient> client) {
             std::vector<ClientAndPtr<Device>> devices;
             devices.reserve(client->local_devices().size());
             for (Device* device : client->local_devices()) {
               devices.push_back(WrapWithClient(client, device));
             }
             return devices;
           })
      .def("host_id", &PyLocalClient::host_id)
      .def("GetDefaultDeviceAssignment",
           [](std::shared_ptr<PyLocalClient> client, int num_replicas,
              int num_partitions)
               -> StatusOr<std::vector<std::vector<ClientAndPtr<Device>>>> {
             TF_ASSIGN_OR_RETURN(DeviceAssignment device_assignment,
                                 client->GetDefaultDeviceAssignment(
                                     num_replicas, num_partitions));
             std::vector<std::vector<ClientAndPtr<Device>>> result;
             result.resize(num_replicas);
             for (int r = 0; r < num_replicas; ++r) {
               result[r].resize(num_partitions);
               for (int p = 0; p < num_partitions; ++p) {
                 int device_id = device_assignment(r, p);
                 auto iter = client->id_to_device().find(device_id);
                 CHECK(iter != client->id_to_device().end()) << device_id;
                 result[r][p] = WrapWithClient(client, iter->second);
               }
             }
             return result;
           })
      // TODO(skye): delete after all callers can handle 2D output
      .def("GetDefaultDeviceAssignment",
           [](std::shared_ptr<PyLocalClient> client,
              int num_replicas) -> StatusOr<std::vector<ClientAndPtr<Device>>> {
             TF_ASSIGN_OR_RETURN(DeviceAssignment device_assignment,
                                 client->GetDefaultDeviceAssignment(
                                     num_replicas, /*num_partitions=*/1));
             std::vector<ClientAndPtr<Device>> result;
             for (int i = 0; i < num_replicas; ++i) {
               int device_id = device_assignment(i, 0);
               auto iter = client->id_to_device().find(device_id);
               CHECK(iter != client->id_to_device().end()) << device_id;
               result.push_back(WrapWithClient(client, iter->second));
             }
             return result;
           })
      .def("CreateChannelHandle",
           [](PyLocalClient* client) {
             return client->client()->CreateChannelHandle();
           })
      .def("CreateDeviceToHostChannelHandle",
           [](PyLocalClient* client) {
             return client->client()->CreateDeviceToHostChannelHandle();
           })
      .def("CreateHostToDeviceChannelHandle", [](PyLocalClient* client) {
        return client->client()->CreateHostToDeviceChannelHandle();
      });

  m.def("get_cpu_client", &GetCpuClient, py::arg("asynchronous") = true);
  m.def("get_nvidia_gpu_client", &GetNvidiaGpuClient,
        py::arg("asynchronous") = true,
        py::arg("allocator_config") = GpuAllocatorConfig(),
        py::arg("distributed_client") = nullptr, py::arg("node_id") = 0);

  py::class_<PyLocalBuffer, ClientAndUniquePtr<PyLocalBuffer>> buffer(
      m, "PyLocalBuffer");
  buffer
      .def_static(
          "from_python",
          [](const pybind11::object& argument,
             std::shared_ptr<PyLocalClient> client, Device* device,
             bool force_copy) -> StatusOr<ClientAndUniquePtr<PyLocalBuffer>> {
            CHECK(device != nullptr);
            auto iter = client->id_to_device().find(device->id());
            if (iter->second != device) {
              return InvalidArgument(
                  "Cannot copy value to device '%s' with '%s' backend",
                  device->DebugString(), client->platform_name());
            }
            GlobalPyRefManager()->CollectGarbage();

            absl::optional<CastToArrayResult> c = CastToArray(argument);
            if (!c) {
              return InvalidArgument("from_python argument must be an array.");
            }

            TF_ASSIGN_OR_RETURN(PythonBufferTree tree,
                                GetPythonBufferTree(argument));
            std::shared_ptr<PythonRefManager::ManagedPyObjects> py_buffer_ref =
                GlobalPyRefManager()->ManageReference(std::move(c->array));

            py::gil_scoped_release gil_release;
            TF_ASSIGN_OR_RETURN(
                std::unique_ptr<PyLocalBuffer> buffer,
                PyLocalBuffer::FromHostBuffer(c->buf_ptr, c->shape, force_copy,
                                              std::move(py_buffer_ref),
                                              client.get(), device));
            return WrapWithClient(std::move(client), std::move(buffer));
          },
          py::arg("argument"), py::arg("client"), py::arg("device"),
          py::arg("force_copy") = false)
      .def("copy_to_device",
           [](PyLocalBuffer* buffer, const ClientAndPtr<Device>& dst_device)
               -> StatusOr<ClientAndUniquePtr<PyLocalBuffer>> {
             CHECK(dst_device.get() != nullptr);
             GlobalPyRefManager()->CollectGarbage();
             py::gil_scoped_release gil_release;
             TF_ASSIGN_OR_RETURN(std::unique_ptr<PyLocalBuffer> out,
                                 buffer->CopyToDevice(dst_device.get()));
             return WrapWithClient(dst_device.client, std::move(out));
           })
      .def("delete", &PyLocalBuffer::Delete)
      .def("block_host_until_ready",
           [](PyLocalBuffer* buffer) {
             GlobalPyRefManager()->CollectGarbage();
             py::gil_scoped_release gil_release;
             return buffer->BlockHostUntilReady();
           })
      .def("copy_to_host_async", &PyLocalBuffer::CopyToHostAsync,
           py::call_guard<py::gil_scoped_release>())
      .def(
          "to_py",
          [](py::object buffer_obj) -> StatusOr<py::object> {
            GlobalPyRefManager()->CollectGarbage();
            PyLocalBuffer* buffer = buffer_obj.cast<PyLocalBuffer*>();
            LocalDeviceState* state = buffer->device()->local_device_state();
            if (state->executor()->platform_kind() == se::PlatformKind::kHost &&
                buffer->on_device_shape().IsArray() &&
                buffer->on_device_shape().element_type() != BF16) {
              py::object out = py::reinterpret_steal<py::object>(
                  PyArray_FROM_O(buffer_obj.ptr()));
              CHECK(out.ptr() != nullptr)
                  << buffer->on_host_shape().ToString(/*print_layout=*/true);
              return out;
            }
            std::shared_ptr<Literal> literal;
            {
              py::gil_scoped_release gil_release;
              TF_ASSIGN_OR_RETURN(literal, buffer->ToLiteral());
            }
            return LiteralToPython(std::move(literal));
          })
      .def("shape", &PyLocalBuffer::on_host_shape)
      .def("device",
           [](const PyLocalBuffer& buffer) {
             return WrapWithClient(buffer.client()->shared_from_this(),
                                   buffer.device());
           })
      .def("platform", &PyLocalBuffer::platform_name)
      .def("is_deleted",
           [](PyLocalBuffer* buffer) { return buffer->IsDeleted(); })
      .def("unsafe_buffer_pointer",
           [](const PyLocalBuffer& buffer) -> StatusOr<std::uintptr_t> {
             TF_ASSIGN_OR_RETURN(ShapedBuffer shaped_buffer,
                                 buffer.AsShapedBuffer());
             if (shaped_buffer.on_device_shape().IsTuple()) {
               return Unimplemented(
                   "unsafe_buffer_pointer is not implemented for tuple "
                   "buffers.");
             }
             return absl::bit_cast<std::uintptr_t>(
                 shaped_buffer.root_buffer().opaque());
           })
      .def_property_readonly("__cuda_array_interface__",
                             &PyLocalBufferCudaArrayInterface);

  // pybind11's implementation of the buffer protocol doesn't allow for correct
  // error handling. We bypass it and implement the buffer protocol ourselves.
  PyTypeObject* buffer_type = reinterpret_cast<PyTypeObject*>(buffer.ptr());
  buffer_type->tp_as_buffer = &PyLocalBufferProcs;

  py::class_<PyLocalExecutable, ClientAndUniquePtr<PyLocalExecutable>>
      executable(m, "LocalExecutable");
  executable
      .def_static("Compile",
                  [](const XlaComputation& computation,
                     absl::optional<std::vector<Shape>> argument_layouts,
                     const ExecutableBuildOptions* build_options,
                     std::shared_ptr<PyLocalClient> client,
                     absl::optional<DeviceAssignment> device_assignment,
                     bool parameter_is_tupled_arguments)
                      -> StatusOr<ClientAndUniquePtr<PyLocalExecutable>> {
                    py::gil_scoped_release gil_release;
                    CompileOptions options;
                    options.argument_layouts = std::move(argument_layouts);
                    if (build_options) {
                      options.executable_build_options = *build_options;
                    }
                    options.parameter_is_tupled_arguments =
                        parameter_is_tupled_arguments;
                    if (device_assignment) {
                      options.executable_build_options.set_device_assignment(
                          *device_assignment);
                    }
                    TF_ASSIGN_OR_RETURN(
                        std::unique_ptr<PyLocalExecutable> executable,
                        PyLocalExecutable::Compile(computation, client.get(),
                                                   std::move(options)));
                    return WrapWithClient(std::move(client),
                                          std::move(executable));
                  })
      .def_static("Compile",
                  [](const XlaComputation& computation,
                     absl::optional<std::vector<Shape>> argument_layouts,
                     const ExecutableBuildOptions* build_options,
                     std::shared_ptr<PyLocalClient> client,
                     absl::optional<std::vector<std::vector<Device*>>>
                         device_assignment,
                     bool parameter_is_tupled_arguments)
                      -> StatusOr<ClientAndUniquePtr<PyLocalExecutable>> {
                    py::gil_scoped_release gil_release;
                    CompileOptions options;
                    options.argument_layouts = std::move(argument_layouts);
                    if (build_options) {
                      options.executable_build_options = *build_options;
                    }
                    options.parameter_is_tupled_arguments =
                        parameter_is_tupled_arguments;
                    if (device_assignment) {
                      TF_ASSIGN_OR_RETURN(
                          DeviceAssignment xla_assignment,
                          DevicesToDeviceAssignment(*device_assignment));
                      options.executable_build_options.set_device_assignment(
                          xla_assignment);
                    }
                    TF_ASSIGN_OR_RETURN(
                        std::unique_ptr<PyLocalExecutable> executable,
                        PyLocalExecutable::Compile(computation, client.get(),
                                                   std::move(options)));
                    return WrapWithClient(std::move(client),
                                          std::move(executable));
                  })
      .def("local_logical_device_ids",
           &PyLocalExecutable::local_logical_device_ids)
      .def("local_devices",
           [](const PyLocalExecutable& executable) {
             std::vector<ClientAndPtr<Device>> devices;
             devices.reserve(executable.local_devices().size());
             for (Device* device : executable.local_devices()) {
               devices.push_back(WrapWithClient(
                   executable.client()->shared_from_this(), device));
             }
             return devices;
           })
      .def("SizeOfGeneratedCodeInBytes",
           &PyLocalExecutable::SizeOfGeneratedCodeInBytes)
      .def("Delete", &PyLocalExecutable::Delete)
      .def(
          "Execute",
          [](const PyLocalExecutable& executable,
             absl::Span<PyLocalBuffer* const> args)
              -> StatusOr<std::vector<ClientAndUniquePtr<PyLocalBuffer>>> {
            py::gil_scoped_release gil_release;
            ExecuteOptions options;
            options.untuple_result = true;
            TF_ASSIGN_OR_RETURN(
                std::vector<std::unique_ptr<PyLocalBuffer>> output_buffers,
                executable.Execute(args, options));
            std::vector<ClientAndUniquePtr<PyLocalBuffer>> outputs;
            outputs.reserve(output_buffers.size());
            for (auto& buffer : output_buffers) {
              outputs.push_back(WrapWithClient(
                  executable.client()->shared_from_this(), std::move(buffer)));
            }
            return outputs;
          },
          py::arg("arguments"))
      .def(
          "ExecuteOnLocalDevices",
          [](const PyLocalExecutable& executable,
             absl::Span<const std::vector<PyLocalBuffer*>> args)
              -> StatusOr<
                  std::vector<std::vector<ClientAndUniquePtr<PyLocalBuffer>>>> {
            py::gil_scoped_release gil_release;
            ExecuteOptions options;
            options.untuple_result = true;
            TF_ASSIGN_OR_RETURN(
                std::vector<std::vector<std::unique_ptr<PyLocalBuffer>>>
                    output_buffers,
                executable.ExecuteOnLocalDevices(args, options));
            std::vector<std::vector<ClientAndUniquePtr<PyLocalBuffer>>> outputs;
            outputs.resize(output_buffers.size());
            for (int computation = 0; computation < output_buffers.size();
                 ++computation) {
              for (auto& buffer : output_buffers[computation]) {
                outputs[computation].push_back(
                    WrapWithClient(executable.client()->shared_from_this(),
                                   std::move(buffer)));
              }
            }
            return outputs;
          },
          py::arg("arguments"))
      .def(
          "get_hlo_modules",
          [](const PyLocalExecutable& executable)
              -> StatusOr<std::vector<std::shared_ptr<HloModule>>> {
            std::vector<std::shared_ptr<HloModule>> modules;
            modules.reserve(executable.executables().size());
            for (const auto& local_exec : executable.executables()) {
              if (!local_exec->executable()->has_module()) {
                return InvalidArgument("Executable does not have HLO modules.");
              }
              modules.push_back(local_exec->executable()->shared_module());
            }
            return std::move(modules);
          });

  py::class_<DebugOptions>(m, "DebugOptions")
      .def_property("xla_cpu_enable_fast_math",
                    &DebugOptions::xla_cpu_enable_fast_math,
                    &DebugOptions::set_xla_cpu_enable_fast_math)
      .def_property("xla_cpu_fast_math_honor_infs",
                    &DebugOptions::xla_cpu_fast_math_honor_infs,
                    &DebugOptions::set_xla_cpu_fast_math_honor_infs)
      .def_property("xla_cpu_fast_math_honor_nans",
                    &DebugOptions::xla_cpu_fast_math_honor_nans,
                    &DebugOptions::set_xla_cpu_fast_math_honor_nans)
      .def_property("xla_cpu_fast_math_honor_division",
                    &DebugOptions::xla_cpu_fast_math_honor_division,
                    &DebugOptions::set_xla_cpu_fast_math_honor_division)
      .def_property("xla_cpu_fast_math_honor_functions",
                    &DebugOptions::xla_cpu_fast_math_honor_functions,
                    &DebugOptions::set_xla_cpu_fast_math_honor_functions)
      .def_property("xla_gpu_enable_fast_min_max",
                    &DebugOptions::xla_gpu_enable_fast_min_max,
                    &DebugOptions::set_xla_gpu_enable_fast_min_max);

  py::class_<ExecutableBuildOptions>(m, "ExecutableBuildOptions")
      .def(py::init<>())
      .def_property(
          "result_layout",
          [](const ExecutableBuildOptions& options) -> absl::optional<Shape> {
            return options.result_layout()
                       ? absl::optional<Shape>(*options.result_layout())
                       : absl::nullopt;
          },
          &ExecutableBuildOptions::set_result_layout)
      .def_property("num_replicas", &ExecutableBuildOptions::num_replicas,
                    &ExecutableBuildOptions::set_num_replicas)
      .def_property("num_partitions", &ExecutableBuildOptions::num_partitions,
                    &ExecutableBuildOptions::set_num_partitions)
      .def_property_readonly(
          "debug_options", &ExecutableBuildOptions::mutable_debug_options,
          py::return_value_policy::reference, py::keep_alive<1, 0>());

  py::class_<XlaComputation>(m, "XlaComputation")
      .def(py::init([](const py::bytes& serialized_hlo_module_proto)
                        -> std::unique_ptr<XlaComputation> {
        HloModuleProto proto;
        proto.ParseFromString(serialized_hlo_module_proto);
        return absl::make_unique<XlaComputation>(proto);
      }))
      .def("GetProgramShape", &XlaComputation::GetProgramShape)
      .def("GetSerializedProto", &GetComputationSerializedProto)
      .def("GetHloText", &GetComputationHloText)
      .def("GetHloDotGraph", &GetComputationHloDotGraph)
      .def("Hash", &HashComputation)
      .def("get_hlo_module", &GetHloModule);

  py::class_<HloPrintOptions> hlo_print_options_class(m, "HloPrintOptions");
  hlo_print_options_class.def(py::init<>())
      .def_static("short_parsable", &HloPrintOptions::ShortParsable)
      .def_static("canonical", &HloPrintOptions::Canonical)
      .def_static("fingerprint", &HloPrintOptions::Fingerprint)
      .def_property("print_large_constants",
                    &HloPrintOptions::print_large_constants,
                    &HloPrintOptions::set_print_large_constants)
      .def_property("print_metadata", &HloPrintOptions::print_metadata,
                    &HloPrintOptions::set_print_metadata)
      .def_property("print_backend_config",
                    &HloPrintOptions::print_backend_config,
                    &HloPrintOptions::set_print_backend_config)
      .def_property("print_result_shape", &HloPrintOptions::print_result_shape,
                    &HloPrintOptions::set_print_result_shape)
      .def_property("print_operand_shape",
                    &HloPrintOptions::print_operand_shape,
                    &HloPrintOptions::set_print_operand_shape)
      .def_property("print_operand_names",
                    &HloPrintOptions::print_operand_names,
                    &HloPrintOptions::set_print_operand_names)
      .def_property("print_ids", &HloPrintOptions::print_ids,
                    &HloPrintOptions::set_print_ids)
      .def_property("print_extra_attributes",
                    &HloPrintOptions::print_extra_attributes,
                    &HloPrintOptions::set_print_extra_attributes)
      .def_property("print_program_shape",
                    &HloPrintOptions::print_program_shape,
                    &HloPrintOptions::set_print_program_shape)
      .def_property("print_percent", &HloPrintOptions::print_percent,
                    &HloPrintOptions::set_print_percent)
      .def_property("print_control_dependencies",
                    &HloPrintOptions::print_control_dependencies,
                    &HloPrintOptions::set_print_control_dependencies)
      .def_property("compact_operands", &HloPrintOptions::compact_operands,
                    &HloPrintOptions::set_compact_operands)
      .def_property("include_layout_in_shapes",
                    &HloPrintOptions::include_layout_in_shapes,
                    &HloPrintOptions::set_include_layout_in_shapes)
      .def_property("canonicalize_instruction_names",
                    &HloPrintOptions::canonicalize_instruction_names,
                    &HloPrintOptions::set_canonicalize_instruction_names)
      .def_property("canonicalize_computations",
                    &HloPrintOptions::canonicalize_computations,
                    &HloPrintOptions::set_canonicalize_computations)
      .def_property("indent_amount", &HloPrintOptions::indent_amount,
                    &HloPrintOptions::set_indent_amount)
      .def_property("is_in_nested_computation",
                    &HloPrintOptions::is_in_nested_computation,
                    &HloPrintOptions::set_is_in_nested_computation)
      .def_property(
          "leading_and_trailing_instructions_number",
          &HloPrintOptions::leading_and_trailing_instructions_number,
          &HloPrintOptions::set_leading_and_trailing_instructions_number);

  py::class_<HloModule, std::shared_ptr<HloModule>> hlo_module_class(
      m, "HloModule");
  hlo_module_class.def(
      "to_string",
      static_cast<std::string (HloModule::*)(const HloPrintOptions&) const>(
          &HloModule::ToString),
      py::arg("options") = HloPrintOptions());

  m.def("hlo_module_to_dot_graph",
        [](const HloModule& hlo_module) -> StatusOr<std::string> {
          return RenderGraph(*hlo_module.entry_computation(), /*label=*/"",
                             hlo_module.config().debug_options(),
                             RenderedGraphFormat::kDot);
        });

  py::class_<XlaOp> xla_op_class(m, "XlaOp");

  py::class_<XlaBuilder>(m, "XlaBuilder")
      .def(py::init([](const std::string& name) -> std::unique_ptr<XlaBuilder> {
        return absl::make_unique<XlaBuilder>(UniquifyName(name));
      }))
      .def(
          "Build",
          [](XlaBuilder& builder, absl::optional<XlaOp> root) {
            return root ? builder.Build(*root) : builder.Build();
          },
          "Builds a computation from the contents of the builder.",
          py::arg("root") = absl::nullopt)
      .def("ClearOpMetadata", &XlaBuilder::ClearOpMetadata)
      .def("GetShape", &XlaBuilder::GetShape)
      .def(
          "GetProgramShape",
          [](const XlaBuilder& builder,
             absl::optional<XlaOp> root) -> StatusOr<ProgramShape> {
            return root ? builder.GetProgramShape(*root)
                        : builder.GetProgramShape();
          },
          py::arg("root") = absl::nullopt)
      .def("IsConstant", &XlaBuilder::IsConstant)
      .def("SetOpMetadata", &XlaBuilder::SetOpMetadata)
      .def("SetSharding", &XlaBuilder::SetSharding)
      .def("ClearSharding", &XlaBuilder::ClearSharding)
      .def("SetUpAlias",
           [](XlaBuilder& builder, const std::vector<int64>& output_index,
              int64 param_number, const std::vector<int64>& param_index) {
             builder.SetUpAlias(
                 ShapeIndex(output_index.begin(), output_index.end()),
                 param_number,
                 ShapeIndex(param_index.begin(), param_index.end()));
           });

  m.def("BufferToDLPackManagedTensor", BufferToDLPackManagedTensor);
  m.def("DLPackManagedTensorToBuffer",
        [](const py::capsule& tensor, std::shared_ptr<PyLocalClient> client)
            -> StatusOr<ClientAndUniquePtr<PyLocalBuffer>> {
          TF_ASSIGN_OR_RETURN(
              std::unique_ptr<PyLocalBuffer> buffer,
              DLPackManagedTensorToBuffer(tensor, client.get()));
          return WrapWithClient(std::move(client), std::move(buffer));
        });

  py::enum_<PrecisionConfig::Precision>(m, "PrecisionConfig_Precision")
      .value("DEFAULT", PrecisionConfig::DEFAULT)
      .value("HIGH", PrecisionConfig::HIGH)
      .value("HIGHEST", PrecisionConfig::HIGHEST);

  py::enum_<OpSharding::Type>(m, "OpSharding_Type")
      .value("REPLICATED", OpSharding::REPLICATED)
      .value("MAXIMAL", OpSharding::MAXIMAL)
      .value("TUPLE", OpSharding::TUPLE)
      .value("OTHER", OpSharding::OTHER);

  py::enum_<ChannelHandle::ChannelType>(m, "ChannelHandle_ChannelType")
      .value("CHANNEL_TYPE_INVALID", ChannelHandle::CHANNEL_TYPE_INVALID)
      .value("DEVICE_TO_DEVICE", ChannelHandle::DEVICE_TO_DEVICE)
      .value("DEVICE_TO_HOST", ChannelHandle::DEVICE_TO_HOST)
      .value("HOST_TO_DEVICE", ChannelHandle::HOST_TO_DEVICE);

  py::class_<ChannelHandle>(m, "ChannelHandle")
      .def_property_readonly("type", &ChannelHandle::type)
      .def_property_readonly("handle", &ChannelHandle::handle)
      .def("__repr__", [](ChannelHandle* h) { return h->DebugString(); });

  py::enum_<FftType>(m, "FftType")
      .value("FFT", FftType::FFT)
      .value("IFFT", FftType::IFFT)
      .value("RFFT", FftType::RFFT)
      .value("IRFFT", FftType::IRFFT);

  BuildOpsSubmodule(&m);
  BuildProfilerSubmodule(&m);

  py::class_<DistributedRuntimeService,
             std::unique_ptr<DistributedRuntimeService>>
      distributed_runtime_service(m, "DistributedRuntimeService");
  py::class_<DistributedRuntimeClient,
             std::shared_ptr<DistributedRuntimeClient>>
      distributed_runtime_client(m, "DistributedRuntimeClient");

  m.def("get_distributed_runtime_service", &GetDistributedRuntimeService);
  m.def("get_distributed_runtime_client", &GetDistributedRuntimeClient);
}  // NOLINT(readability/fn_size)

}  // namespace xla
