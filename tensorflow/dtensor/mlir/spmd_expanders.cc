/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/dtensor/mlir/expansions/argmax_spmd_expander.h"
#include "tensorflow/dtensor/mlir/expansions/bias_add_spmd_expander.h"
#include "tensorflow/dtensor/mlir/expansions/broadcast_to_spmd_expander.h"
#include "tensorflow/dtensor/mlir/expansions/concat_spmd_expander.h"
#include "tensorflow/dtensor/mlir/expansions/control_flow_spmd_expander.h"
#include "tensorflow/dtensor/mlir/expansions/conv_spmd_expander.h"
#include "tensorflow/dtensor/mlir/expansions/cumsum_spmd_expander.h"
#include "tensorflow/dtensor/mlir/expansions/dataparallel_spmd_expander.h"
#include "tensorflow/dtensor/mlir/expansions/disable_copy_on_read_spmd_expander.h"
#include "tensorflow/dtensor/mlir/expansions/dtensor_op_spmd_expander.h"
#include "tensorflow/dtensor/mlir/expansions/einsum_spmd_expander.h"
#include "tensorflow/dtensor/mlir/expansions/elementwise_spmd_expander.h"
#include "tensorflow/dtensor/mlir/expansions/expanddims_spmd_expander.h"
#include "tensorflow/dtensor/mlir/expansions/fill_spmd_expander.h"
#include "tensorflow/dtensor/mlir/expansions/gather_spmd_expander.h"
#include "tensorflow/dtensor/mlir/expansions/identity_n_spmd_expander.h"
#include "tensorflow/dtensor/mlir/expansions/in_top_k_spmd_expander.h"
#include "tensorflow/dtensor/mlir/expansions/io_op_spmd_expander.h"
#include "tensorflow/dtensor/mlir/expansions/iterator_spmd_expander.h"
#include "tensorflow/dtensor/mlir/expansions/matmul_spmd_expander.h"
#include "tensorflow/dtensor/mlir/expansions/meta_spmd_expander.h"
#include "tensorflow/dtensor/mlir/expansions/nullary_spmd_expander.h"
#include "tensorflow/dtensor/mlir/expansions/optional_spmd_expander.h"
#include "tensorflow/dtensor/mlir/expansions/qr_spmd_expander.h"
#include "tensorflow/dtensor/mlir/expansions/random_op_spmd_expander.h"
#include "tensorflow/dtensor/mlir/expansions/range_spmd_expander.h"
#include "tensorflow/dtensor/mlir/expansions/reduce_spmd_expander.h"
#include "tensorflow/dtensor/mlir/expansions/replicated_spmd_expander.h"
#include "tensorflow/dtensor/mlir/expansions/resource_spmd_expander.h"
#include "tensorflow/dtensor/mlir/expansions/save_restore_spmd_expander.h"
#include "tensorflow/dtensor/mlir/expansions/scatter_spmd_expander.h"
#include "tensorflow/dtensor/mlir/expansions/segmentation_spmd_expander.h"
#include "tensorflow/dtensor/mlir/expansions/slice_spmd_expander.h"
#include "tensorflow/dtensor/mlir/expansions/softmax_spmd_expander.h"
#include "tensorflow/dtensor/mlir/expansions/sparse_to_dense_spmd_expander.h"
#include "tensorflow/dtensor/mlir/expansions/split_spmd_expander.h"
#include "tensorflow/dtensor/mlir/expansions/squeeze_spmd_expander.h"
#include "tensorflow/dtensor/mlir/expansions/tensorlist_getitem_spmd_expander.h"
#include "tensorflow/dtensor/mlir/expansions/tensorlist_reserve_spmd_expander.h"
#include "tensorflow/dtensor/mlir/expansions/tensorlist_setitem_spmd_expander.h"
#include "tensorflow/dtensor/mlir/expansions/top_k_spmd_expander.h"
#include "tensorflow/dtensor/mlir/expansions/trivial_spmd_expander.h"
#include "tensorflow/dtensor/mlir/expansions/unsupported_op_spmd_expander.h"
#include "tensorflow/dtensor/mlir/spmd_expander.h"

namespace tensorflow {
namespace dtensor {

// Nullary
REGISTER_SPMD(Const, TF::ConstOp, NullarySPMDExpander);

// Unary
REGISTER_SPMD(Abs, TF::AbsOp, ElementwiseSPMDExpander);
REGISTER_SPMD(Cast, TF::CastOp, ElementwiseSPMDExpander);
REGISTER_SPMD(Identity, TF::IdentityOp, ElementwiseSPMDExpander);
REGISTER_SPMD(Neg, TF::NegOp, ElementwiseSPMDExpander);
REGISTER_SPMD(ZerosLike, TF::ZerosLikeOp, ElementwiseSPMDExpander);
REGISTER_SPMD(OnesLike, TF::OnesLikeOp, ElementwiseSPMDExpander);
REGISTER_SPMD(Exp, TF::ExpOp, ElementwiseSPMDExpander);
REGISTER_SPMD(Sqrt, TF::SqrtOp, ElementwiseSPMDExpander);
REGISTER_SPMD(Rsqrt, TF::RsqrtOp, ElementwiseSPMDExpander);
REGISTER_SPMD(Log, TF::LogOp, ElementwiseSPMDExpander);
REGISTER_SPMD(StopGradient, TF::StopGradientOp, ElementwiseSPMDExpander);
REGISTER_SPMD(Reciprocal, TF::ReciprocalOp, ElementwiseSPMDExpander);
REGISTER_SPMD(Square, TF::SquareOp, ElementwiseSPMDExpander);
REGISTER_SPMD(Erf, TF::ErfOp, ElementwiseSPMDExpander);
REGISTER_SPMD(Tanh, TF::TanhOp, ElementwiseSPMDExpander);
REGISTER_SPMD(TanhGrad, TF::TanhGradOp, ElementwiseSPMDExpander);
REGISTER_SPMD(Relu, TF::ReluOp, ElementwiseSPMDExpander);
REGISTER_SPMD(ReluGrad, TF::ReluGradOp, ElementwiseSPMDExpander);
REGISTER_SPMD(Sigmoid, TF::SigmoidOp, ElementwiseSPMDExpander);
REGISTER_SPMD(SigmoidGrad, TF::SigmoidGradOp, ElementwiseSPMDExpander);
REGISTER_SPMD(IsFinite, TF::IsFiniteOp, ElementwiseSPMDExpander);

// Elementwise
REGISTER_SPMD(Add, TF::AddOp, ElementwiseSPMDExpander);
REGISTER_SPMD(AddV2, TF::AddV2Op, ElementwiseSPMDExpander);
REGISTER_SPMD(AddN, TF::AddNOp, ElementwiseSPMDExpander);
REGISTER_SPMD(RealDiv, TF::RealDivOp, ElementwiseSPMDExpander);
REGISTER_SPMD(Div, TF::DivOp, ElementwiseSPMDExpander);
REGISTER_SPMD(DivNoNan, TF::DivNoNanOp, ElementwiseSPMDExpander);
REGISTER_SPMD(Equal, TF::EqualOp, ElementwiseSPMDExpander);
REGISTER_SPMD(FloorDiv, TF::FloorDivOp, ElementwiseSPMDExpander);
REGISTER_SPMD(FloorMod, TF::FloorModOp, ElementwiseSPMDExpander);
REGISTER_SPMD(NotEqual, TF::NotEqualOp, ElementwiseSPMDExpander);
REGISTER_SPMD(Less, TF::LessOp, ElementwiseSPMDExpander);
REGISTER_SPMD(LessEqual, TF::LessEqualOp, ElementwiseSPMDExpander);
REGISTER_SPMD(LogicalAnd, TF::LogicalAndOp, ElementwiseSPMDExpander);
REGISTER_SPMD(LogicalNot, TF::LogicalNotOp, ElementwiseSPMDExpander);
REGISTER_SPMD(Maximum, TF::MaximumOp, ElementwiseSPMDExpander);
REGISTER_SPMD(Minimum, TF::MinimumOp, ElementwiseSPMDExpander);
REGISTER_SPMD(Mul, TF::MulOp, ElementwiseSPMDExpander);
REGISTER_SPMD(Select, TF::SelectOp, ElementwiseSPMDExpander);
REGISTER_SPMD(SelectV2, TF::SelectV2Op, ElementwiseSPMDExpander);
REGISTER_SPMD(Sub, TF::SubOp, ElementwiseSPMDExpander);
REGISTER_SPMD(SquaredDifference, TF::SquaredDifferenceOp,
              ElementwiseSPMDExpander);
REGISTER_SPMD(Greater, TF::GreaterOp, ElementwiseSPMDExpander);
REGISTER_SPMD(GreaterEqual, TF::GreaterEqualOp, ElementwiseSPMDExpander);
REGISTER_SPMD(RsqrtGrad, TF::RsqrtGradOp, ElementwiseSPMDExpander);
REGISTER_SPMD(SqrtGrad, TF::SqrtGradOp, ElementwiseSPMDExpander);
REGISTER_SPMD(Pow, TF::PowOp, ElementwiseSPMDExpander);
REGISTER_SPMD(BitwiseAnd, TF::BitwiseAndOp, ElementwiseSPMDExpander);
REGISTER_SPMD(BitwiseOr, TF::BitwiseOrOp, ElementwiseSPMDExpander);
REGISTER_SPMD(BitwiseXor, TF::BitwiseXorOp, ElementwiseSPMDExpander);
REGISTER_SPMD(LeftShift, TF::LeftShiftOp, ElementwiseSPMDExpander);
REGISTER_SPMD(RightShift, TF::RightShiftOp, ElementwiseSPMDExpander);
REGISTER_SPMD(LogicalOr, TF::LogicalOrOp, ElementwiseSPMDExpander);
REGISTER_SPMD(Cos, TF::CosOp, ElementwiseSPMDExpander);
REGISTER_SPMD(Acos, TF::AcosOp, ElementwiseSPMDExpander);
REGISTER_SPMD(Acosh, TF::AcoshOp, ElementwiseSPMDExpander);
REGISTER_SPMD(Angle, TF::AngleOp, ElementwiseSPMDExpander);
REGISTER_SPMD(Asin, TF::AsinOp, ElementwiseSPMDExpander);
REGISTER_SPMD(Asinh, TF::AsinhOp, ElementwiseSPMDExpander);
REGISTER_SPMD(Atan, TF::AtanOp, ElementwiseSPMDExpander);
REGISTER_SPMD(Atan2, TF::Atan2Op, ElementwiseSPMDExpander);
REGISTER_SPMD(Atanh, TF::AtanhOp, ElementwiseSPMDExpander);
REGISTER_SPMD(BesselI0e, TF::BesselI0eOp, ElementwiseSPMDExpander);
REGISTER_SPMD(BesselI1e, TF::BesselI1eOp, ElementwiseSPMDExpander);
REGISTER_SPMD(Betainc, TF::BetaincOp, ElementwiseSPMDExpander);
REGISTER_SPMD(Bitcast, TF::BitcastOp, ElementwiseSPMDExpander);
REGISTER_SPMD(Ceil, TF::CeilOp, ElementwiseSPMDExpander);
REGISTER_SPMD(CheckNumerics, TF::CheckNumericsOp, ElementwiseSPMDExpander);
REGISTER_SPMD(ClipByValue, TF::ClipByValueOp, ElementwiseSPMDExpander);
REGISTER_SPMD(Conj, TF::ConjOp, ElementwiseSPMDExpander);
REGISTER_SPMD(Cosh, TF::CoshOp, ElementwiseSPMDExpander);
REGISTER_SPMD(Complex, TF::ComplexOp, ElementwiseSPMDExpander);
REGISTER_SPMD(ComplexAbs, TF::ComplexAbsOp, ElementwiseSPMDExpander);
REGISTER_SPMD(Digamma, TF::DigammaOp, ElementwiseSPMDExpander);

// TODO(b/193924452): Add the following Ops once unit tests are there.
//
REGISTER_SPMD(Elu, TF::EluOp, ElementwiseSPMDExpander);
REGISTER_SPMD(EluGrad, TF::EluGradOp, ElementwiseSPMDExpander);
REGISTER_SPMD(Erfc, TF::ErfcOp, ElementwiseSPMDExpander);
REGISTER_SPMD(Erfinv, TF::ErfinvOp, ElementwiseSPMDExpander);
REGISTER_SPMD(Expm1, TF::Expm1Op, ElementwiseSPMDExpander);
REGISTER_SPMD(Floor, TF::FloorOp, ElementwiseSPMDExpander);
// REGISTER_SPMD(HSVToRGB, TF::HSVToRGBOp, ElementwiseSPMDExpander);
REGISTER_SPMD(Igamma, TF::IgammaOp, ElementwiseSPMDExpander);
REGISTER_SPMD(Igammac, TF::IgammacOp, ElementwiseSPMDExpander);
REGISTER_SPMD(IgammaGradA, TF::IgammaGradAOp, ElementwiseSPMDExpander);
REGISTER_SPMD(Imag, TF::ImagOp, ElementwiseSPMDExpander);
// REGISTER_SPMD(InplaceAdd, TF::InplaceAddOp, ElementwiseSPMDExpander);
// REGISTER_SPMD(InplaceUpdate, TF::InplaceUpdateOp, ElementwiseSPMDExpander);
REGISTER_SPMD(Inv, TF::InvOp, ElementwiseSPMDExpander);
REGISTER_SPMD(Invert, TF::InvertOp, ElementwiseSPMDExpander);
REGISTER_SPMD(IsInf, TF::IsInfOp, ElementwiseSPMDExpander);
REGISTER_SPMD(IsNan, TF::IsNanOp, ElementwiseSPMDExpander);
REGISTER_SPMD(LeakyRelu, TF::LeakyReluOp, ElementwiseSPMDExpander);
REGISTER_SPMD(LeakyReluGrad, TF::LeakyReluGradOp, ElementwiseSPMDExpander);
REGISTER_SPMD(Lgamma, TF::LgammaOp, ElementwiseSPMDExpander);
REGISTER_SPMD(Log1p, TF::Log1pOp, ElementwiseSPMDExpander);
REGISTER_SPMD(MulNoNan, TF::MulNoNanOp, ElementwiseSPMDExpander);
REGISTER_SPMD(Ndtri, TF::NdtriOp, ElementwiseSPMDExpander);
REGISTER_SPMD(NextAfter, TF::NextAfterOp, ElementwiseSPMDExpander);
REGISTER_SPMD(Polygamma, TF::PolygammaOp, ElementwiseSPMDExpander);
REGISTER_SPMD(PopulationCount, TF::PopulationCountOp, ElementwiseSPMDExpander);
REGISTER_SPMD(PreventGradient, TF::PreventGradientOp, ElementwiseSPMDExpander);
REGISTER_SPMD(Real, TF::RealOp, ElementwiseSPMDExpander);
REGISTER_SPMD(ReciprocalGrad, TF::ReciprocalGradOp, ElementwiseSPMDExpander);
REGISTER_SPMD(Relu6, TF::Relu6Op, ElementwiseSPMDExpander);
REGISTER_SPMD(Relu6Grad, TF::Relu6GradOp, ElementwiseSPMDExpander);
REGISTER_SPMD(Rint, TF::RintOp, ElementwiseSPMDExpander);
REGISTER_SPMD(Round, TF::RoundOp, ElementwiseSPMDExpander);
REGISTER_SPMD(Selu, TF::SeluOp, ElementwiseSPMDExpander);
REGISTER_SPMD(SeluGrad, TF::SeluGradOp, ElementwiseSPMDExpander);
REGISTER_SPMD(Sign, TF::SignOp, ElementwiseSPMDExpander);
REGISTER_SPMD(Sin, TF::SinOp, ElementwiseSPMDExpander);
REGISTER_SPMD(Sinh, TF::SinhOp, ElementwiseSPMDExpander);
// REGISTER_SPMD(Snapshot, TF::SnapshotOp, ElementwiseSPMDExpander);
REGISTER_SPMD(Softplus, TF::SoftplusOp, ElementwiseSPMDExpander);
// REGISTER_SPMD(SoftplusGrad, TF::SoftplusGradOp, ElementwiseSPMDExpander);
REGISTER_SPMD(Softsign, TF::SoftsignOp, ElementwiseSPMDExpander);
// REGISTER_SPMD(SoftsignGrad, TF::SoftsignGradOp, ElementwiseSPMDExpander);
REGISTER_SPMD(Tan, TF::TanOp, ElementwiseSPMDExpander);
// REGISTER_SPMD(TridiagonalSolve, TF::TridiagonalSolveOp,
// ElementwiseSPMDExpander);
REGISTER_SPMD(TruncateDiv, TF::TruncateDivOp, ElementwiseSPMDExpander);
REGISTER_SPMD(TruncateMod, TF::TruncateModOp, ElementwiseSPMDExpander);
REGISTER_SPMD(Xdivy, TF::XdivyOp, ElementwiseSPMDExpander);
REGISTER_SPMD(Xlog1py, TF::Xlog1pyOp, ElementwiseSPMDExpander);
REGISTER_SPMD(Xlogy, TF::XlogyOp, ElementwiseSPMDExpander);
REGISTER_SPMD(Zeta, TF::ZetaOp, ElementwiseSPMDExpander);

// IdentityN
// TODO(hongjunchoi): Make ElementwiseSPMDExpander support IdentityN.
REGISTER_SPMD(IdentityN, TF::IdentityNOp, IdentityNSPMDExpander);

// Range
REGISTER_SPMD(Range, TF::RangeOp, RangeSPMDExpander);

// Reductions
REGISTER_SPMD(All, TF::AllOp, ReduceSPMDExpander);
REGISTER_SPMD(Any, TF::AnyOp, ReduceSPMDExpander);
REGISTER_SPMD(Mean, TF::MeanOp, ReduceSPMDExpander);
REGISTER_SPMD(Max, TF::MaxOp, ReduceSPMDExpander);
REGISTER_SPMD(Min, TF::MinOp, ReduceSPMDExpander);
REGISTER_SPMD(Prod, TF::ProdOp, ReduceSPMDExpander);
REGISTER_SPMD(Sum, TF::SumOp, ReduceSPMDExpander);
REGISTER_SPMD(L2Loss, TF::L2LossOp, ReduceSPMDExpander);

// Convolution
REGISTER_SPMD(Conv2D, TF::Conv2DOp, ConvSPMDExpander);
REGISTER_SPMD(Conv2DBackpropFilter, TF::Conv2DBackpropFilterOp,
              ConvSPMDExpander);
REGISTER_SPMD(Conv2DBackpropFilterV2, TF::Conv2DBackpropFilterV2Op,
              ConvSPMDExpander);
REGISTER_SPMD(Conv2DBackpropInput, TF::Conv2DBackpropInputOp, ConvSPMDExpander);
REGISTER_SPMD(Conv2DBackpropInputV2, TF::Conv2DBackpropInputV2Op,
              ConvSPMDExpander);
REGISTER_SPMD(Conv3D, TF::Conv3DOp, ConvSPMDExpander);
REGISTER_SPMD(Conv3DBackpropFilter, TF::Conv3DBackpropFilterOp,
              ConvSPMDExpander);
REGISTER_SPMD(Conv3DBackpropFilterV2, TF::Conv3DBackpropFilterV2Op,
              ConvSPMDExpander);
REGISTER_SPMD(Conv3DBackpropInput, TF::Conv3DBackpropInputOp, ConvSPMDExpander);
REGISTER_SPMD(Conv3DBackpropInputV2, TF::Conv3DBackpropInputV2Op,
              ConvSPMDExpander);
REGISTER_SPMD(MaxPool, TF::MaxPoolOp, ConvSPMDExpander);
REGISTER_SPMD(MaxPoolGrad, TF::MaxPoolGradOp, ConvSPMDExpander);

// Metadata
REGISTER_SPMD(Rank, TF::RankOp, ShapeSPMDExpander);
REGISTER_SPMD(Shape, TF::ShapeOp, ShapeSPMDExpander);
REGISTER_SPMD(ShapeN, TF::ShapeNOp, ShapeSPMDExpander);

REGISTER_SPMD(BroadcastGradientArgs, TF::BroadcastGradientArgsOp,
              MetadataSPMDExpander);

// Resource ops
REGISTER_SPMD(AssignVariable, TF::AssignVariableOp, ResourceSPMDExpander);
REGISTER_SPMD(AssignAddVariable, TF::AssignAddVariableOp, ResourceSPMDExpander);
REGISTER_SPMD(AssignSubVariable, TF::AssignSubVariableOp, ResourceSPMDExpander);
REGISTER_SPMD(ReadVariable, TF::ReadVariableOp, ResourceSPMDExpander);
REGISTER_SPMD(VarHandle, TF::VarHandleOp, ResourceSPMDExpander);
REGISTER_SPMD(VarIsInitialized, TF::VarIsInitializedOp, ResourceSPMDExpander);
REGISTER_SPMD(DestroyResource, TF::DestroyResourceOp, ResourceSPMDExpander);

// Einsum
REGISTER_SPMD(Einsum, TF::EinsumOp, EinsumSPMDExpander);

// Matrix multiplication
REGISTER_SPMD(BatchMatMulV2, TF::BatchMatMulV2Op, MatMulSPMDExpander);
REGISTER_SPMD(MatMul, TF::MatMulOp, MatMulSPMDExpander);

// Stack/unstack (pack/unpack)
REGISTER_SPMD(Pack, TF::PackOp, PackSPMDExpander);
REGISTER_SPMD(Unpack, TF::UnpackOp, UnpackSPMDExpander);

// Reshape
REGISTER_SPMD(Reshape, TF::ReshapeOp, ReshapeSPMDExpander);
REGISTER_SPMD(Transpose, TF::TransposeOp, TransposeSPMDExpander);
REGISTER_SPMD(InvertPermutation, TF::InvertPermutationOp,
              ReplicatedOpSPMDExpander,
              /*relayout_when_sharded=*/true);

// Pad
REGISTER_SPMD(Pad, TF::PadOp, PadSPMDExpander);
REGISTER_SPMD(PadV2, TF::PadV2Op, PadSPMDExpander);

// Scatter/Gather
REGISTER_SPMD(GatherV2, TF::GatherV2Op, GatherV2SPMDExpander);
REGISTER_SPMD(GatherNd, TF::GatherNdOp, GatherNdSPMDExpander);
REGISTER_SPMD(ResourceGather, TF::ResourceGatherOp, ResourceGatherSPMDExpander);
REGISTER_SPMD(ScatterNd, TF::ScatterNdOp, ScatterNdOpSPMDExpander);
REGISTER_SPMD(TensorScatterUpdate, TF::TensorScatterUpdateOp,
              TensorScatterOpSPMDExpander);
REGISTER_SPMD(TensorScatterAdd, TF::TensorScatterAddOp,
              TensorScatterOpSPMDExpander);

// ArgMax/ArgMin
REGISTER_SPMD(ArgMax, TF::ArgMaxOp, ArgMaxSPMDExpander);

// Slice
REGISTER_SPMD(Slice, TF::SliceOp, SliceSPMDExpander);
REGISTER_SPMD(StridedSlice, TF::StridedSliceOp, StridedSliceSPMDExpander);
REGISTER_SPMD(TensorStridedSliceUpdate, TF::TensorStridedSliceUpdateOp,
              TensorStridedSliceUpdateSPMDExpander);
REGISTER_SPMD(StridedSliceGrad, TF::StridedSliceGradOp,
              StridedSliceGradSPMDExpander);

// Split
REGISTER_SPMD(Split, TF::SplitOp, SplitSPMDExpander);
REGISTER_SPMD(SplitV, TF::SplitVOp, SplitVSPMDExpander);

// Squeeze
REGISTER_SPMD(Squeeze, TF::SqueezeOp, SqueezeSPMDExpander);

// Concat
REGISTER_SPMD(Concat, TF::ConcatOp, ConcatSPMDExpander);
REGISTER_SPMD(ConcatV2, TF::ConcatV2Op, ConcatSPMDExpander);

// Softmax Loss ops
REGISTER_SPMD(SoftmaxCrossEntropyWithLogits,
              TF::SoftmaxCrossEntropyWithLogitsOp, SoftmaxLossOpSPMDExpander);
REGISTER_SPMD(SparseSoftmaxCrossEntropyWithLogits,
              TF::SparseSoftmaxCrossEntropyWithLogitsOp,
              SoftmaxLossOpSPMDExpander);

// Softmax ops
REGISTER_SPMD(Softmax, TF::SoftmaxOp, SoftmaxOpSPMDExpander);
REGISTER_SPMD(LogSoftmax, TF::LogSoftmaxOp, SoftmaxOpSPMDExpander);

// Random ops
// LINT.IfChange
REGISTER_SPMD(StatelessRandomUniform, TF::StatelessRandomUniformOp,
              RandomOpSPMDExpander);
REGISTER_SPMD(StatelessRandomUniformFullInt,
              TF::StatelessRandomUniformFullIntOp, RandomOpSPMDExpander);
REGISTER_SPMD(StatelessRandomNormal, TF::StatelessRandomNormalOp,
              RandomOpSPMDExpander);
REGISTER_SPMD(StatelessTruncatedNormal, TF::StatelessTruncatedNormalOp,
              RandomOpSPMDExpander);
// LINT.ThenChange(//tensorflow/dtensor/cc/small_constant_optimization.cc)
// Random V2 ops
REGISTER_SPMD(StatelessRandomGetKeyCounter, TF::StatelessRandomGetKeyCounterOp,
              ReplicatedOpSPMDExpander);
REGISTER_SPMD(RngReadAndSkip, TF::RngReadAndSkipOp, ReplicatedOpSPMDExpander);
REGISTER_SPMD(StatelessRandomNormalV2, TF::StatelessRandomNormalV2Op,
              RandomOpSPMDExpander);
REGISTER_SPMD(StatelessRandomUniformV2, TF::StatelessRandomUniformV2Op,
              RandomOpSPMDExpander);
REGISTER_SPMD(StatelessRandomUniformFullIntV2,
              TF::StatelessRandomUniformFullIntV2Op, RandomOpSPMDExpander);
REGISTER_SPMD(StatelessRandomUniformIntV2, TF::StatelessRandomUniformIntV2Op,
              RandomOpSPMDExpander);
REGISTER_SPMD(StatelessTruncatedNormalV2, TF::StatelessTruncatedNormalV2Op,
              RandomOpSPMDExpander);

// Input agnotics ops
REGISTER_SPMD(Fill, TF::FillOp, FillSPMDExpander);

// Tile
REGISTER_SPMD(Tile, TF::TileOp, TileSPMDExpander);

// Expansion of ResourceApply ops are no-ops as they are always element-wise.
// Also, ResourceApply ops do not have output values. As so, inferring layout
// from operand and consumers are trivial(no-op).
// Resource apply ops
REGISTER_SPMD(ResourceApplyAdagradV2, TF::ResourceApplyAdagradV2Op,
              NoOpSPMDExpander);
REGISTER_SPMD(ResourceApplyAdam, TF::ResourceApplyAdamOp, NoOpSPMDExpander);
REGISTER_SPMD(ResourceApplyGradientDescent, TF::ResourceApplyGradientDescentOp,
              NoOpSPMDExpander);
REGISTER_SPMD(ResourceApplyCenteredRMSProp, TF::ResourceApplyCenteredRMSPropOp,
              NoOpSPMDExpander);
REGISTER_SPMD(ResourceApplyKerasMomentum, TF::ResourceApplyKerasMomentumOp,
              NoOpSPMDExpander);
REGISTER_SPMD(ResourceApplyMomentum, TF::ResourceApplyMomentumOp,
              NoOpSPMDExpander);

// AssertOp
REGISTER_SPMD(Assert, TF::AssertOp, NoOpSPMDExpander);

// Terminator ops
REGISTER_SPMD(Return, tf_device::ReturnOp, TerminatorSPMDExpander);

// Onehot
REGISTER_SPMD(OneHot, TF::OneHotOp, OneHotSPMDExpander);
// ExpandDimsOp
REGISTER_SPMD(ExpandDims, TF::ExpandDimsOp, ExpandDimsExpander);
// UnsortedSegmentSumOp
REGISTER_SPMD(UnsortedSegmentSum, TF::UnsortedSegmentSumOp,
              UnsortedSegmentSumSPMDExpander);
// BroadcastToOp
REGISTER_SPMD(BroadcastTo, TF::BroadcastToOp, BroadcastToSPMDExpander);

// Save/Restore ops.
REGISTER_SPMD(SaveV2, TF::SaveV2Op, SaveRestoreSPMDExpander);
REGISTER_SPMD(MergeV2Checkpoints, TF::MergeV2CheckpointsOp,
              SaveRestoreSPMDExpander);
REGISTER_SPMD(RestoreV2, TF::RestoreV2Op, SaveRestoreSPMDExpander);
REGISTER_SPMD(DTensorRestoreV2, TF::DTensorRestoreV2Op,
              SaveRestoreSPMDExpander);
REGISTER_SPMD(DTensorShardedPrefix, TF::DTensorShardedPrefixOp,
              DTensorShardPrefixSPMDExpander);

// DTensor Virtual ops
REGISTER_SPMD(
    CopyToMesh, TF::CopyToMeshOp, UnsupportedOpSPMDExpander,
    "CopyToMesh should have been lowered to DTensorSend and DTensorRecv.");
REGISTER_SPMD(
    CopyToMeshGrad, TF::CopyToMeshGradOp, UnsupportedOpSPMDExpander,
    "CopyToMesh should have been lowered to DTensorSend and DTensorRecv.");
REGISTER_SPMD(Relayout, TF::RelayoutOp, RelayoutSPMDExpander);
REGISTER_SPMD(RelayoutLike, TF::RelayoutLikeOp, RelayoutLikeSPMDExpander);
REGISTER_SPMD(DTensorSend, TF::DTensorSend, DTensorSendSPMDExpander);
REGISTER_SPMD(DTensorRecv, TF::DTensorRecv, DTensorRecvSPMDExpander);

// TopKV2
REGISTER_SPMD(TopKV2, TF::TopKV2Op, TopKSPMDExpander);
// InTopKV2
REGISTER_SPMD(InTopKV2, TF::InTopKV2Op, InTopKSPMDExpander);

// Control flow
REGISTER_SPMD(WhileRegion, TF::WhileRegionOp, WhileRegionSPMDExpander);
REGISTER_SPMD(IfRegion, TF::IfRegionOp, IfRegionSPMDExpander);

// BiasAdd
REGISTER_SPMD(BiasAdd, TF::BiasAddOp, BiasAddExpander);
REGISTER_SPMD(BiasAddGrad, TF::BiasAddGradOp, ReduceSPMDExpander);

// QR
REGISTER_SPMD(Qr, TF::QrOp, QRSPMDExpander);

// Data Parallel
REGISTER_SPMD(AvgPool, TF::AvgPoolOp, DataparallelSPMDExpander,
              llvm::DenseMap<int, int>{{0, 3}},
              llvm::DenseMap<int, int>{{0, 3}});
REGISTER_SPMD(AvgPool3D, TF::AvgPool3DOp, DataparallelSPMDExpander,
              llvm::DenseMap<int, int>{{0, 4}},
              llvm::DenseMap<int, int>{{0, 4}});
REGISTER_SPMD(MaxPool3D, TF::MaxPool3DOp, DataparallelSPMDExpander,
              llvm::DenseMap<int, int>{{0, 4}},
              llvm::DenseMap<int, int>{{0, 4}});
REGISTER_SPMD(DepthwiseConv2dNative, TF::DepthwiseConv2dNativeOp,
              DataparallelSPMDExpander, llvm::DenseMap<int, int>{{0, 3}},
              llvm::DenseMap<int, int>{{0, 3}});
REGISTER_SPMD(ResizeBilinear, TF::ResizeBilinearOp, DataparallelSPMDExpander,
              llvm::DenseMap<int, int>{{0, 3}},
              llvm::DenseMap<int, int>{{0, 3}});
REGISTER_SPMD(ResizeNearestNeighbor, TF::ResizeNearestNeighborOp,
              DataparallelSPMDExpander, llvm::DenseMap<int, int>{{0, 3}},
              llvm::DenseMap<int, int>{{0, 3}});
REGISTER_SPMD(AdjustContrastv2, TF::AdjustContrastv2Op,
              DataparallelSPMDExpander, llvm::DenseMap<int, int>{{0, 3}},
              llvm::DenseMap<int, int>{{0, 3}});
REGISTER_SPMD(AdjustSaturation, TF::AdjustSaturationOp,
              DataparallelSPMDExpander, llvm::DenseMap<int, int>{{0, 3}},
              llvm::DenseMap<int, int>{{0, 3}});
REGISTER_SPMD(FFT, TF::FFTOp, DataparallelSPMDExpander,
              llvm::DenseMap<int, int>{{0, 1}},
              llvm::DenseMap<int, int>{{0, 1}});
REGISTER_SPMD(FFT2D, TF::FFT2DOp, DataparallelSPMDExpander,
              llvm::DenseMap<int, int>{{0, 1}},
              llvm::DenseMap<int, int>{{0, 1}});
REGISTER_SPMD(FFT3D, TF::FFT3DOp, DataparallelSPMDExpander,
              llvm::DenseMap<int, int>{{0, 1}},
              llvm::DenseMap<int, int>{{0, 1}});
REGISTER_SPMD(IFFT, TF::IFFTOp, DataparallelSPMDExpander,
              llvm::DenseMap<int, int>{{0, 1}},
              llvm::DenseMap<int, int>{{0, 1}});
REGISTER_SPMD(IFFT2D, TF::IFFT2DOp, DataparallelSPMDExpander,
              llvm::DenseMap<int, int>{{0, 1}},
              llvm::DenseMap<int, int>{{0, 1}});
REGISTER_SPMD(IFFT3D, TF::IFFT3DOp, DataparallelSPMDExpander,
              llvm::DenseMap<int, int>{{0, 1}},
              llvm::DenseMap<int, int>{{0, 1}});
REGISTER_SPMD(IRFFT, TF::IRFFTOp, DataparallelSPMDExpander,
              llvm::DenseMap<int, int>{{0, 1}},
              llvm::DenseMap<int, int>{{0, 1}});
REGISTER_SPMD(IRFFT2D, TF::IRFFT2DOp, DataparallelSPMDExpander,
              llvm::DenseMap<int, int>{{0, 1}},
              llvm::DenseMap<int, int>{{0, 1}});
REGISTER_SPMD(IRFFT3D, TF::IRFFT3DOp, DataparallelSPMDExpander,
              llvm::DenseMap<int, int>{{0, 1}},
              llvm::DenseMap<int, int>{{0, 1}});
REGISTER_SPMD(RFFT, TF::RFFTOp, DataparallelSPMDExpander,
              llvm::DenseMap<int, int>{{0, 1}},
              llvm::DenseMap<int, int>{{0, 1}});
REGISTER_SPMD(RFFT2D, TF::RFFT2DOp, DataparallelSPMDExpander,
              llvm::DenseMap<int, int>{{0, 1}},
              llvm::DenseMap<int, int>{{0, 1}});
REGISTER_SPMD(RFFT3D, TF::RFFT3DOp, DataparallelSPMDExpander,
              llvm::DenseMap<int, int>{{0, 1}},
              llvm::DenseMap<int, int>{{0, 1}});
REGISTER_SPMD(Cholesky, TF::CholeskyOp, DataparallelSPMDExpander,
              llvm::DenseMap<int, int>{{0, 2}},
              llvm::DenseMap<int, int>{{0, 2}});
// Data Parallel Grad Ops
REGISTER_SPMD(MaxPool3DGrad, TF::MaxPool3DGradOp, DataparallelSPMDExpander,
              llvm::DenseMap<int, int>{{0, 4}, {1, 4}, {2, 4}},
              llvm::DenseMap<int, int>{{0, 4}});
REGISTER_SPMD(MaxPool3DGradGrad, TF::MaxPool3DGradGradOp,
              DataparallelSPMDExpander,
              llvm::DenseMap<int, int>{{0, 4}, {1, 4}, {2, 4}},
              llvm::DenseMap<int, int>{{0, 4}});
REGISTER_SPMD(MaxPoolGradGrad, TF::MaxPoolGradGradOp, DataparallelSPMDExpander,
              llvm::DenseMap<int, int>{{0, 3}, {1, 3}, {2, 3}},
              llvm::DenseMap<int, int>{{0, 3}});
REGISTER_SPMD(ResizeBilinearGrad, TF::ResizeBilinearGradOp,
              DataparallelSPMDExpander,
              llvm::DenseMap<int, int>{{0, 3}, {1, 3}},
              llvm::DenseMap<int, int>{{0, 3}});
REGISTER_SPMD(ResizeNearestNeighborGrad, TF::ResizeNearestNeighborGradOp,
              DataparallelSPMDExpander, llvm::DenseMap<int, int>{{0, 3}},
              llvm::DenseMap<int, int>{{0, 3}});

// DiagPart
REGISTER_SPMD(DiagPart, TF::DiagPartOp, ReplicatedOpSPMDExpander,
              /*relayout_when_sharded=*/true);

// Cumsum
REGISTER_SPMD(Cumsum, TF::CumsumOp, CumsumSPMDExpander);

// SparseToDenseOp
REGISTER_SPMD(SparseToDense, TF::SparseToDenseOp, SparseToDenseSPMDExpander);

// StringFormat
REGISTER_SPMD(StringFormat, TF::StringFormatOp, ReplicatedOpSPMDExpander,
              /*relayout_when_sharded=*/true);
REGISTER_SPMD(StringToHashBucketFast, TF::StringToHashBucketFastOp,
              ElementwiseSPMDExpander);

// TensorList ops
REGISTER_SPMD(TensorListReserve, TF::TensorListReserveOp,
              TensorListReserveSPMDExpander);
REGISTER_SPMD(TensorListGetItem, TF::TensorListGetItemOp,
              TensorListGetItemSPMDExpander);
REGISTER_SPMD(TensorListSetItem, TF::TensorListSetItemOp,
              TensorListSetItemSPMDExpander);

// IO ops
REGISTER_SPMD(WriteSummary, TF::WriteSummaryOp, IOOpSPMDExpander);
REGISTER_SPMD(FlushSummaryWriter, TF::FlushSummaryWriterOp, IOOpSPMDExpander);
REGISTER_SPMD(DisableCopyOnRead, TF::DisableCopyOnReadOp,
              DisableCopyOnReadSPMDExpander);
REGISTER_SPMD(ShardedFilename, TF::ShardedFilenameOp, ReplicatedOpSPMDExpander);

// tf.data Optional ops
REGISTER_SPMD(OptionalHasValue, TF::OptionalHasValueOp,
              OptionalHasValueSPMDExpander);
REGISTER_SPMD(OptionalGetValue, TF::OptionalGetValueOp,
              OptionalGetValueSPMDExpander);

// tf.data Iterator ops
REGISTER_SPMD(IteratorGetNext, TF::IteratorGetNextOp,
              IteratorGetNextSPMDExpander);
REGISTER_SPMD(IteratorGetNextAsOptional, TF::IteratorGetNextAsOptionalOp,
              IteratorGetNextAsOptionalSPMDExpander);

// Unsupported ops.
REGISTER_SPMD(RandomNormal, TF::RandomUniformOp, UnsupportedOpSPMDExpander,
              /*error_message=*/
              "Stateful random operations are not supported in DTensor. Please "
              "use stateless random operations instead.");
REGISTER_SPMD(RandomNormalInt, TF::RandomUniformIntOp,
              UnsupportedOpSPMDExpander,
              /*error_message=*/
              "Stateful random operations are not supported in DTensor. Please "
              "use stateless random operations instead.");

// Unique
REGISTER_SPMD(Unique, TF::UniqueOp, ReplicatedOpSPMDExpander,
              /*relayout_when_sharded=*/true);

}  // namespace dtensor
}  // namespace tensorflow
