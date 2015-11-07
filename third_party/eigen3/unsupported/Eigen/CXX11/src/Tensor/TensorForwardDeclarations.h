// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2014 Benoit Steiner <benoit.steiner.goog@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_CXX11_TENSOR_TENSOR_FORWARD_DECLARATIONS_H
#define EIGEN_CXX11_TENSOR_TENSOR_FORWARD_DECLARATIONS_H

namespace Eigen {

template<typename Scalar_, std::size_t NumIndices_, int Options_ = 0, typename IndexType = DenseIndex> class Tensor;
template<typename Scalar_, typename Dimensions, int Options_ = 0, typename IndexType = DenseIndex> class TensorFixedSize;
template<typename Scalar_, int Options_ = 0, typename IndexType = DenseIndex> class TensorVarDim;
template<typename PlainObjectType, int Options_ = Unaligned> class TensorMap;
template<typename PlainObjectType> class TensorRef;
template<typename Derived, int AccessLevel = internal::accessors_level<Derived>::value> class TensorBase;

template<typename NullaryOp, typename PlainObjectType> class TensorCwiseNullaryOp;
template<typename UnaryOp, typename XprType> class TensorCwiseUnaryOp;
template<typename BinaryOp, typename LeftXprType, typename RightXprType> class TensorCwiseBinaryOp;
template<typename IfXprType, typename ThenXprType, typename ElseXprType> class TensorSelectOp;
template<typename Op, typename Dims, typename XprType> class TensorReductionOp;
template<typename XprType> class TensorIndexTupleOp;
template<typename ReduceOp, typename Dims, typename XprType> class TensorTupleReducerOp;
template<typename Axis, typename LeftXprType, typename RightXprType> class TensorConcatenationOp;
template<typename Dimensions, typename LeftXprType, typename RightXprType> class TensorContractionOp;
template<typename TargetType, typename XprType> class TensorConversionOp;
template<typename Dimensions, typename InputXprType, typename KernelXprType> class TensorConvolutionOp;
template<typename Dimensions, typename InputXprType, typename KernelXprType> class TensorConvolutionByFFTOp;
template<typename FFT, typename XprType, int FFTDataType, int FFTDirection> class TensorFFTOp;
template<typename IFFT, typename XprType, int ResultType> class TensorIFFTOp;
template<typename DFT, typename XprType, int ResultType> class TensorDFTOp;
template<typename IDFT, typename XprType, int ResultType> class TensorIDFTOp;
template<typename PatchDim, typename XprType> class TensorPatchOp;
template<DenseIndex Rows, DenseIndex Cols, typename XprType> class TensorImagePatchOp;
template<DenseIndex Planes, DenseIndex Rows, DenseIndex Cols, typename XprType> class TensorVolumePatchOp;
template<typename Broadcast, typename XprType> class TensorBroadcastingOp;
template<DenseIndex DimId, typename XprType> class TensorChippingOp;
template<typename NewDimensions, typename XprType> class TensorReshapingOp;
template<typename XprType> class TensorLayoutSwapOp;
template<typename StartIndices, typename Sizes, typename XprType> class TensorSlicingOp;
template<typename ReverseDimensions, typename XprType> class TensorReverseOp;
template<typename XprType> class TensorTrueIndicesOp;
template<typename PaddingDimensions, typename XprType> class TensorPaddingOp;
template<typename Shuffle, typename XprType> class TensorShufflingOp;
template<typename Strides, typename XprType> class TensorStridingOp;
template<typename Strides, typename XprType> class TensorInflationOp;
template<typename Generator, typename XprType> class TensorGeneratorOp;
template<typename LeftXprType, typename RightXprType> class TensorAssignOp;

template<typename CustomUnaryFunc, typename XprType> class TensorCustomUnaryOp;
template<typename CustomBinaryFunc, typename LhsXprType, typename RhsXprType> class TensorCustomBinaryOp;

template<typename XprType> class TensorEvalToOp;
template<typename XprType> class TensorForcedEvalOp;

template<typename ExpressionType, typename DeviceType> class TensorDevice;
template<typename Derived, typename Device> struct TensorEvaluator;

class DefaultDevice;
class ThreadPoolDevice;
class GpuDevice;

enum DFTResultType {
  RealPart = 0,
  ImagPart = 1,
  BothParts = 2
};

enum FFTDirection {
    FFT_FORWARD = 0,
    FFT_REVERSE = 1
};

namespace internal {
template <typename Device, typename Expression>
struct IsVectorizable {
  static const bool value = TensorEvaluator<Expression, Device>::PacketAccess;
};

template <typename Expression>
struct IsVectorizable<GpuDevice, Expression> {
  static const bool value = TensorEvaluator<Expression, GpuDevice>::PacketAccess &&
                            TensorEvaluator<Expression, GpuDevice>::IsAligned;
};

template <typename Device, typename Expression>
struct IsTileable {
  static const bool value = TensorEvaluator<Expression, Device>::BlockAccess;
};

template <typename Expression, typename Device,
          bool Vectorizable = IsVectorizable<Device, Expression>::value,
          bool Tileable = IsTileable<Device, Expression>::value>
class TensorExecutor;
}  // end namespace internal

}  // end namespace Eigen

#endif // EIGEN_CXX11_TENSOR_TENSOR_FORWARD_DECLARATIONS_H
