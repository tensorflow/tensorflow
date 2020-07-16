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

#include "tensorflow/compiler/xla/python/ops.h"

#include <string>
#include <vector>

#include "absl/types/optional.h"
#include "absl/types/span.h"
#include "pybind11/attr.h"
#include "pybind11/pybind11.h"
#include "tensorflow/compiler/xla/client/lib/comparators.h"
#include "tensorflow/compiler/xla/client/lib/math.h"
#include "tensorflow/compiler/xla/client/lib/qr.h"
#include "tensorflow/compiler/xla/client/lib/self_adjoint_eig.h"
#include "tensorflow/compiler/xla/client/lib/sorting.h"
#include "tensorflow/compiler/xla/client/lib/svd.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/client/xla_computation.h"
#include "tensorflow/compiler/xla/python/types.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"

namespace xla {

namespace py = pybind11;

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
  ops.def("AllGather", &AllGather, py::arg("operand"),
          py::arg("all_gather_dimension"), py::arg("shard_count"),
          py::arg("replica_groups") = py::list(),
          py::arg("channel_id") = absl::nullopt,
          py::arg("shape_with_layout") = absl::nullopt);
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
          py::arg("replica_groups") = py::list(),
          py::arg("layout") = absl::nullopt);
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

}  // namespace xla
