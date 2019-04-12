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

#include "absl/strings/string_view.h"
#include "absl/types/optional.h"
#include "include/pybind11/pybind11.h"
#include "tensorflow/compiler/xla/client/client_library.h"
#include "tensorflow/compiler/xla/client/lib/math.h"
#include "tensorflow/compiler/xla/client/lib/qr.h"
#include "tensorflow/compiler/xla/client/lib/self_adjoint_eig.h"
#include "tensorflow/compiler/xla/client/lib/svd.h"
#include "tensorflow/compiler/xla/client/local_client.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/client/xla_computation.h"
#include "tensorflow/compiler/xla/python/local_client.h"
#include "tensorflow/compiler/xla/python/types.h"
#include "tensorflow/compiler/xla/python/xrt.h"
#include "tensorflow/compiler/xla/service/cpu/custom_call_target_registry.h"
#include "tensorflow/compiler/xla/service/platform_util.h"
#include "tensorflow/compiler/xla/shape.h"
#include "tensorflow/compiler/xla/shape_util.h"

namespace xla {
namespace xla_python {

namespace py = pybind11;

PYBIND11_MODULE(xla_extension, m) {
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
      .value("OPAQUE", OPAQUE)
      .value("TOKEN", TOKEN);

  // Shapes
  py::class_<Shape>(m, "Shape")
      .def_static(
          "Tuple",
          [](std::vector<Shape> shapes) -> Shape {
            return ShapeUtil::MakeTupleShape(shapes);
          },
          "Makes a tuple shape.")
      .def_static(
          "Array",
          [](PrimitiveType type, std::vector<int64> dims,
             absl::optional<std::vector<int64>> layout) -> Shape {
            if (layout) {
              return ShapeUtil::MakeShapeWithLayout(type, dims, *layout);
            } else {
              Shape shape = ShapeUtil::MakeShape(type, dims);
              shape.clear_layout();
              return shape;
            }
          },
          "Makes an array shape.", py::arg("type"), py::arg("dims"),
          py::arg("layout") = absl::nullopt)
      .def("dimensions",
           static_cast<const std::vector<int64>& (Shape::*)() const>(
               &Shape::dimensions))
      .def("element_type", &Shape::element_type)
      .def("tuple_shapes",
           static_cast<const std::vector<Shape>& (Shape::*)() const>(
               &Shape::tuple_shapes))
      .def("__repr__", [](const Shape& shape) {
        return shape.ToString(/*print_layouts=*/true);
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
      .def("Parameters",
           static_cast<const std::vector<Shape>& (ProgramShape::*)() const>(
               &ProgramShape::parameters))
      .def("Result", &ProgramShape::result)
      .def("__repr__", &ProgramShape::ToString);

  // Literals
  py::class_<Literal>(m, "Literal").def("__repr__", &Literal::ToString);
  py::class_<LiteralSlice>(m, "LiteralSlice");
  py::implicitly_convertible<Literal, LiteralSlice>();
  py::implicitly_convertible<BorrowingLiteral, LiteralSlice>();

  // Device assignments
  py::class_<DeviceAssignment>(m, "DeviceAssignment")
      .def_static("Create",
                  [](py::array_t<int> array) -> StatusOr<DeviceAssignment> {
                    if (array.ndim() != 2) {
                      return InvalidArgument(
                          "Argument to DeviceAssignment constructor must be a "
                          "2D array, "
                          "received an %dD array.",
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

  // Local XLA client methods.

  // CPU custom-call targets.
  m.def("RegisterCpuCustomCallTarget", &RegisterCpuCustomCallTarget);

  py::class_<PyLocalClient>(m, "LocalClient")
      .def_static("Get", &PyLocalClient::Get)
      .def("DeviceCount", &PyLocalClient::device_count)
      .def("TransferToInfeed", &PyLocalClient::TransferToInfeed)
      .def("TransferFromOutfeed", &PyLocalClient::TransferFromOutfeed);

  py::class_<LocalShapedBuffer>(m, "LocalShapedBuffer")
      .def_static("FromPython", &LocalShapedBuffer::FromPython)
      .def_static("FromPythonValues", &LocalShapedBuffer::FromPythonValues)
      .def("Delete", &LocalShapedBuffer::Delete)
      .def("DestructureTuple", &LocalShapedBuffer::DestructureTuple)
      .def("ToPython", &LocalShapedBuffer::ToPython)
      .def("shape", &LocalShapedBuffer::shape);

  py::class_<PyLocalExecutable>(m, "LocalExecutable")
      .def_static("Compile", &PyLocalExecutable::Compile,
                  py::call_guard<py::gil_scoped_release>())
      .def("DeviceOrdinals", &PyLocalExecutable::DeviceOrdinals)
      .def("Delete", &PyLocalExecutable::Delete)
      .def("Execute", &PyLocalExecutable::Execute,
           py::call_guard<py::gil_scoped_release>())
      .def("ExecutePerReplica", &PyLocalExecutable::ExecutePerReplica,
           py::call_guard<py::gil_scoped_release>());

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
                    &ExecutableBuildOptions::set_num_replicas);

  py::class_<XlaComputation>(m, "XlaComputation")
      .def("GetProgramShape", &XlaComputation::GetProgramShape)
      .def("GetSerializedProto", &GetComputationSerializedProto)
      .def("GetHloText", &GetComputationHloText)
      .def("GetHloDotGraph", &GetComputationHloDotGraph);

  py::class_<XlaOp>(m, "XlaOp");

  py::class_<XlaBuilder>(m, "XlaBuilder")
      .def(py::init<const std::string&>())
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
      .def("SetOpMetadata", &XlaBuilder::SetOpMetadata);

  // ops submodule, containing free functions that add operators to an
  // XlaBuilder.
  py::module ops = m.def_submodule("ops", "XLA operations");

  ops.def("AllToAll", &AllToAll);
  ops.def("CrossReplicaSum",
          static_cast<XlaOp (*)(const XlaOp&, absl::Span<const ReplicaGroup>)>(
              &CrossReplicaSum));
  ops.def("BitcastConvertType", &BitcastConvertType, py::arg("operand"),
          py::arg("new_element_type"));
  ops.def("Broadcast", &Broadcast, py::arg("operand"), py::arg("sizes"));
  ops.def("BroadcastInDim", &BroadcastInDim, py::arg("operand"),
          py::arg("shape"), py::arg("broadcast_dimensions"));
  ops.def("Call", &Call);
  ops.def("Cholesky", &Cholesky, py::arg("a"), py::arg("lower") = true);
  ops.def("Clamp", &Clamp);
  ops.def("Collapse", &Collapse, py::arg("operand"), py::arg("dimensions"));
  ops.def("ConcatInDim", &ConcatInDim);
  ops.def("Conditional",
          static_cast<XlaOp (*)(const XlaOp&,
                                absl::Span<const XlaComputation* const>,
                                absl::Span<const XlaOp>)>(&Conditional));
  ops.def(
      "Conditional",
      static_cast<XlaOp (*)(const XlaOp&, const XlaOp&, const XlaComputation&,
                            const XlaOp&, const XlaComputation&)>(
          &Conditional));
  ops.def("ConstantLiteral", &ConstantLiteral);
  ops.def("ConvGeneralDilated", &ConvGeneralDilated, py::arg("lhs"),
          py::arg("rhs"), py::arg("window_strides"), py::arg("padding"),
          py::arg("lhs_dilation"), py::arg("rhs_dilation"),
          py::arg("dimension_numbers"), py::arg("feature_group_count") = 1,
          py::arg("batch_group_count") = 1,
          py::arg("precision_config") = nullptr);
  ops.def("ConvertElementType", &ConvertElementType, py::arg("operand"),
          py::arg("new_element_type"));
  ops.def("CustomCall", &CustomCallWithLayout);
  ops.def("Dot", &Dot, py::arg("lhs"), py::arg("rhs"),
          py::arg("precision_config") = nullptr);
  ops.def("DotGeneral", &DotGeneral, py::arg("lhs"), py::arg("rhs"),
          py::arg("dimension_numbers"), py::arg("precision_config") = nullptr);
  ops.def("DynamicSlice",
          static_cast<XlaOp (*)(const XlaOp&, absl::Span<const XlaOp>,
                                absl::Span<const int64>)>(&DynamicSlice));
  ops.def("DynamicUpdateSlice",
          static_cast<XlaOp (*)(const XlaOp&, const XlaOp&,
                                absl::Span<const XlaOp>)>(&DynamicUpdateSlice));
  ops.def("Gather", &Gather, py::arg("a"), py::arg("start_indices"),
          py::arg("dimension_numbers"), py::arg("slice_sizes"));
  ops.def("GetTupleElement", &GetTupleElement);
  ops.def("Infeed", &Infeed, py::arg("builder"), py::arg("shape"),
          py::arg("config") = "");
  ops.def("Iota",
          static_cast<XlaOp (*)(XlaBuilder*, const Shape&, int64)>(&Iota));
  ops.def("Iota",
          static_cast<XlaOp (*)(XlaBuilder*, PrimitiveType, int64)>(&Iota));
  ops.def("Map", &Map);
  ops.def("Outfeed", &Outfeed, py::arg("operand"), py::arg("shape_with_layout"),
          py::arg("outfeed_config") = "");
  ops.def("Pad", &Pad);
  ops.def(
      "Parameter",
      static_cast<XlaOp (*)(XlaBuilder*, int64, const Shape&, const string&)>(
          &Parameter));
  ops.def("QR",
          [](XlaOp a, bool full_matrices) -> StatusOr<std::pair<XlaOp, XlaOp>> {
            TF_ASSIGN_OR_RETURN(auto qr, QRDecomposition(a, full_matrices));
            return std::make_pair(qr.q, qr.r);
          });
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
                                absl::Span<const int64>)>(&Reduce));
  ops.def("ReduceWindowWithGeneralPadding", &ReduceWindowWithGeneralPadding);
  ops.def("ReplicaId", &ReplicaId);
  ops.def("Reshape",
          static_cast<XlaOp (*)(const XlaOp&, absl::Span<const int64>,
                                absl::Span<const int64>)>(&Reshape));
  ops.def(
      "Reshape",
      static_cast<XlaOp (*)(const XlaOp&, absl::Span<const int64>)>(&Reshape));
  ops.def("Rev", &Rev, py::arg("operand"), py::arg("dimensions"));
  ops.def("RngNormal", &RngNormal);
  ops.def("RngUniform", &RngUniform);
  ops.def("Scatter", &Scatter);
  ops.def("Select", &Select);
  ops.def("SelectAndScatterWithGeneralPadding",
          &SelectAndScatterWithGeneralPadding);
  ops.def("Slice", &Slice);
  ops.def("SliceInDim", &SliceInDim, py::arg("operand"), py::arg("start_index"),
          py::arg("limit_index"), py::arg("stride"), py::arg("dimno"));
  ops.def("Sort",
          static_cast<XlaOp (*)(const XlaOp&, absl::Span<const XlaOp>, int64)>(
              &Sort),
          py::arg("keys"), py::arg("values"), py::arg("dimension") = -1);
  ops.def("Transpose", &Transpose);
  ops.def("TriangularSolve", &TriangularSolve);
  ops.def("Tuple", &Tuple);
  ops.def("While", &While);

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

  py::enum_<TriangularSolveOptions::Transpose>(
      m, "TriangularSolveOptions_Transpose")
      .value("TRANSPOSE_INVALID", TriangularSolveOptions::TRANSPOSE_INVALID)
      .value("NO_TRANSPOSE", TriangularSolveOptions::NO_TRANSPOSE)
      .value("TRANSPOSE", TriangularSolveOptions::TRANSPOSE)
      .value("ADJOINT", TriangularSolveOptions::ADJOINT);

  // TODO(phawkins): improve bindings for these types.
  py::class_<ChannelHandle>(m, "ChannelHandle");
  py::class_<PrecisionConfig>(m, "PrecisionConfig");

  tensorflow::AddXrtSubmodule(&m);
}

}  // namespace xla_python
}  // namespace xla
