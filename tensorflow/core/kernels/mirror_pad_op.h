/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_KERNELS_MIRROR_PAD_OP_H_
#define TENSORFLOW_KERNELS_MIRROR_PAD_OP_H_

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/platform/types.h"

namespace Eigen {
template <typename PaddingDimensions, typename XprType>
class TensorMirrorPadOp;

namespace internal {
template <typename PaddingDimensions, typename XprType>
struct traits<TensorMirrorPadOp<PaddingDimensions, XprType>>
    : public traits<XprType> {
  typedef typename XprType::Scalar Scalar;
  typedef traits<XprType> XprTraits;
  typedef typename XprTraits::StorageKind StorageKind;
  typedef typename XprTraits::Index Index;
  typedef typename XprType::Nested Nested;
  typedef typename remove_reference<Nested>::type _Nested;
  static constexpr int NumDimensions = XprTraits::NumDimensions;
  static constexpr int Layout = XprTraits::Layout;
};

template <typename PaddingDimensions, typename XprType>
struct eval<TensorMirrorPadOp<PaddingDimensions, XprType>, Eigen::Dense> {
  typedef const TensorMirrorPadOp<PaddingDimensions, XprType>& type;
};

template <typename PaddingDimensions, typename XprType>
struct nested<
    TensorMirrorPadOp<PaddingDimensions, XprType>, 1,
    typename eval<TensorMirrorPadOp<PaddingDimensions, XprType>>::type> {
  typedef TensorMirrorPadOp<PaddingDimensions, XprType> type;
};
}  // namespace internal

template <typename PaddingDimensions, typename XprType>
class TensorMirrorPadOp
    : public TensorBase<TensorMirrorPadOp<PaddingDimensions, XprType>,
                        ReadOnlyAccessors> {
 public:
  typedef typename Eigen::internal::traits<TensorMirrorPadOp>::Scalar Scalar;
  typedef typename Eigen::NumTraits<Scalar>::Real RealScalar;
  typedef typename XprType::CoeffReturnType CoeffReturnType;
  typedef typename Eigen::internal::nested<TensorMirrorPadOp>::type Nested;
  typedef typename Eigen::internal::traits<TensorMirrorPadOp>::StorageKind
      StorageKind;
  typedef typename Eigen::internal::traits<TensorMirrorPadOp>::Index Index;

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE TensorMirrorPadOp(
      const XprType& expr, const PaddingDimensions& padding_dims, Index offset)
      : xpr_(expr), padding_dims_(padding_dims), offset_(offset) {}

  EIGEN_DEVICE_FUNC
  const PaddingDimensions& padding() const { return padding_dims_; }

  EIGEN_DEVICE_FUNC
  Index offset() const { return offset_; }

  EIGEN_DEVICE_FUNC
  const typename internal::remove_all<typename XprType::Nested>::type&
  expression() const {
    return xpr_;
  }

 protected:
  typename XprType::Nested xpr_;
  const PaddingDimensions padding_dims_;
  const Index offset_;
};

// Eval as rvalue
template <typename PaddingDimensions, typename ArgType, typename Device>
struct TensorEvaluator<const TensorMirrorPadOp<PaddingDimensions, ArgType>,
                       Device> {
  typedef TensorMirrorPadOp<PaddingDimensions, ArgType> XprType;
  typedef typename XprType::Index Index;
  static constexpr int Dims = internal::array_size<PaddingDimensions>::value;
  typedef DSizes<Index, Dims> Dimensions;
  typedef typename XprType::Scalar Scalar;
  typedef typename XprType::CoeffReturnType CoeffReturnType;
  // Copied from Eigen3 Github version 0e806c1.
  typedef typename PacketType<CoeffReturnType, Device>::type PacketReturnType;

  enum {
    IsAligned = false,
    PacketAccess = TensorEvaluator<ArgType, Device>::PacketAccess,
    BlockAccess = false,
    Layout = TensorEvaluator<ArgType, Device>::Layout,
    CoordAccess = true,
    RawAccess = false
  };

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE TensorEvaluator(const XprType& op,
                                                        const Device& device)
      : impl_(op.expression(), device), padding_(op.padding()) {
    EIGEN_STATIC_ASSERT(Dims > 0, YOU_MADE_A_PROGRAMMING_MISTAKE)

    // op.offset() == 0 if padding mode is symmetric.
    // op.offset() == 1 if padding mode is reflect.
    eigen_assert(op.offset() == 0 || op.offset() == 1);
    left_offset_ = -1 + op.offset();
    right_offset_ = -1 - op.offset();

    // This should trigger compilation error if padding dimensions and
    // expression dimensions do not match.
    dimensions_ = impl_.dimensions();
    for (int dim = 0; dim < Dims; ++dim) {
      eigen_assert(padding_[dim].first + op.offset() <= dimensions_[dim]);
      eigen_assert(padding_[dim].second + op.offset() <= dimensions_[dim]);
      dimensions_[dim] += padding_[dim].first + padding_[dim].second;
    }

    const auto& input_dims = impl_.dimensions();
    if (static_cast<int>(Layout) == static_cast<int>(ColMajor)) {
      input_strides_[0] = 1;
      output_strides_[0] = 1;
      for (int i = 0; i < Dims - 1; ++i) {
        input_strides_[i + 1] = input_strides_[i] * input_dims[i];
        output_strides_[i + 1] = output_strides_[i] * dimensions_[i];
      }
    } else {
      input_strides_[numext::maxi(0, Dims - 1)] = 1;
      output_strides_[numext::maxi(0, Dims - 1)] = 1;
      for (int i = Dims - 1; i > 0; --i) {
        input_strides_[i - 1] = input_strides_[i] * input_dims[i];
        output_strides_[i - 1] = output_strides_[i] * dimensions_[i];
      }
    }
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Dimensions& dimensions() const {
    return dimensions_;
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE bool evalSubExprsIfNeeded(Scalar*) {
    impl_.evalSubExprsIfNeeded(nullptr);
    return true;
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void cleanup() { impl_.cleanup(); }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE CoeffReturnType
  coeff(Index index) const {
    eigen_assert(index < dimensions().TotalSize());
    const Index input_index = ToInputIndex(index);
    return impl_.coeff(input_index);
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE CoeffReturnType
  coeff(array<Index, Dims> coords) const {
    for (int dim = 0; dim < Dims; ++dim) {
      coords[dim] = ToInputCoord(coords[dim], dim);
    }
    ReadInputHelper<TensorEvaluator<ArgType, Device>::CoordAccess> helper;
    return helper(coords, input_strides_, impl_);
  }

  template <int LoadMode>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE PacketReturnType
  packet(Index index) const {
    constexpr int kPacketSize =
        internal::unpacket_traits<PacketReturnType>::size;

    EIGEN_STATIC_ASSERT(kPacketSize > 1, YOU_MADE_A_PROGRAMMING_MISTAKE)
    eigen_assert(index + kPacketSize <= dimensions().TotalSize());

    // Find the effective inner-most dimension where padding actually happens.
    // NOTE: This is independent of index argument, and can be done in the
    // constructor to save computation. However, if packet access does not
    // happen, then moving to constructor will incur needless overhead.
    int dim = -1;
    if (static_cast<int>(Layout) == static_cast<int>(ColMajor)) {
      for (int k = 0; k < Dims; ++k) {
        if (padding_[k].first != 0 || padding_[k].second != 0) {
          dim = k;
          break;
        }
      }
    } else {
      for (int k = Dims - 1; k >= 0; --k) {
        if (padding_[k].first != 0 || padding_[k].second != 0) {
          dim = k;
          break;
        }
      }
    }

    const Index input_index = ToInputIndex(index);

    // If dim < 0, this means there is no padding at all.
    if (dim < 0) {
      return impl_.template packet<Unaligned>(input_index);
    }

    // Check if the way from the begin of the packet to the end of the packet
    // is paved with contiguous road. That is, the indices must be between the
    // padded region in the effective inner-most dimension.
    const Index left = padding_[dim].first * output_strides_[dim];
    const Index right =
        (dimensions_[dim] - padding_[dim].second) * output_strides_[dim];

    if (left <= index && (index + kPacketSize - 1) < right) {
      return impl_.template packet<Unaligned>(input_index);
    }

    // If the road is not contiguous, then fall back to coeff().
    EIGEN_ALIGN_MAX typename internal::remove_const<CoeffReturnType>::type
        values[kPacketSize];
    values[0] = impl_.coeff(input_index);
    for (int i = 1; i < kPacketSize; ++i) {
      values[i] = coeff(index + i);
    }
    PacketReturnType result = internal::pload<PacketReturnType>(values);
    return result;
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE TensorOpCost
  costPerCoeff(bool vectorized) const {
    constexpr int kPacketSize =
        internal::unpacket_traits<PacketReturnType>::size;

    const double compute_cost = Dims * (7 * TensorOpCost::AddCost<Index>() +
                                        2 * TensorOpCost::MulCost<Index>() +
                                        TensorOpCost::DivCost<Index>());
    return impl_.costPerCoeff(vectorized) +
           TensorOpCost(1, 0, compute_cost, vectorized, kPacketSize);
  }

  EIGEN_DEVICE_FUNC Scalar* data() const { return nullptr; }

 protected:
  using Coords = array<Index, Dims>;

  // Full template specialization is not allowed within non-fully specialized
  // template class. Adding a dummy parameter to make specializations partial.
  template <bool CoordAccess, bool dummy = true>
  struct ReadInputHelper;

  template <bool dummy>
  struct ReadInputHelper<false, dummy> {
    template <typename Eval>
    EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE Index
    operator()(const Coords& coord, const Coords& strides, const Eval& eval) {
      Index index = 0;
      for (int k = 0; k < Dims; ++k) {
        index += coord[k] * strides[k];
      }
      return eval.coeff(index);
    }
  };

  template <bool dummy>
  struct ReadInputHelper<true, dummy> {
    template <typename Eval>
    EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE Index
    operator()(const Coords& coord, const Coords& strides, const Eval& eval) {
      return eval.coeff(coord);
    }
  };

  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE Index ToInputCoord(Index k,
                                                           int dim) const {
    const Index m = impl_.dimensions()[dim];
    k -= padding_[dim].first;
    if (k < 0) {
      return -k + left_offset_;
    }
    if (k < m) {
      return k;
    }
    return m - (k - m) + right_offset_;
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Index
  ToInputIndex(const Coords& coords) const {
    Index input_index = 0;
    for (int dim = 0; dim < Dims; ++dim) {
      input_index += ToInputCoord(coords[dim], dim) * input_strides_[dim];
    }
    return input_index;
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Index ToInputIndex(Index index) const {
    Index input_index = 0;
    if (static_cast<int>(Layout) == static_cast<int>(ColMajor)) {
      for (int dim = Dims - 1; dim > 0; --dim) {
        const Index k = index / output_strides_[dim];
        index -= k * output_strides_[dim];
        input_index += ToInputCoord(k, dim) * input_strides_[dim];
      }
      input_index += ToInputCoord(index, 0);
    } else {
      for (int dim = 0; dim < Dims - 1; ++dim) {
        const Index k = index / output_strides_[dim];
        index -= k * output_strides_[dim];
        input_index += ToInputCoord(k, dim) * input_strides_[dim];
      }
      input_index += ToInputCoord(index, Dims - 1);
    }

    return input_index;
  }

  TensorEvaluator<ArgType, Device> impl_;
  PaddingDimensions padding_;
  Dimensions dimensions_;
  array<Index, Dims> input_strides_;
  array<Index, Dims> output_strides_;

  Index left_offset_;
  Index right_offset_;
};
}  // namespace Eigen

namespace tensorflow {
namespace functor {

// offset argument must be either 0 or 1. This controls whether the boundary
// values are replicated (offset == 0) or not replicated (offset == 1).
template <typename Device, typename T, typename Tpaddings, int Dims>
struct MirrorPad {
  void operator()(const Device& device,
                  typename TTypes<T, Dims, int32>::Tensor output,
                  typename TTypes<T, Dims, int32>::ConstTensor input,
                  typename TTypes<Tpaddings>::ConstMatrix padding, int offset) {
    Eigen::array<Eigen::IndexPair<int32>, Dims> padding_dims;

    for (int i = 0; i < Dims; ++i) {
      padding_dims[i] = Eigen::IndexPair<int32>(padding(i, 0), padding(i, 1));
    }

    output.device(device) = MirrorPadOp(input, padding_dims, offset);
  }

  template <typename PaddingDimensions, typename Derived>
  static const Eigen::TensorMirrorPadOp<PaddingDimensions, const Derived>
  MirrorPadOp(
      const Eigen::TensorBase<Derived, Eigen::ReadOnlyAccessors>& tensor,
      const PaddingDimensions& padding, int offset) {
    return Eigen::TensorMirrorPadOp<PaddingDimensions, const Derived>(
        static_cast<const Derived&>(tensor), padding, offset);
  }
};

// offset argument must be either 0 or 1. This controls whether the boundary
// values are replicated (offset == 0) or not replicated (offset == 1).
template <typename Device, typename T, typename Tpaddings, int Dims>
struct MirrorPadGrad {
  void operator()(const Device& device,
                  typename TTypes<T, Dims, int32>::Tensor output,
                  typename TTypes<T, Dims, int32>::ConstTensor input,
                  typename TTypes<Tpaddings>::ConstMatrix paddings, int offset,
                  typename TTypes<T, Dims, int32>::Tensor scratch) {
    // Copy the gradient input into the scratch buffer.
    scratch.device(device) = input;

    Eigen::array<int32, Dims> lhs_offsets;
    Eigen::array<int32, Dims> rhs_offsets;
    Eigen::array<int32, Dims> extents;
    Eigen::array<bool, Dims> reverses;

    for (int i = 0; i < Dims; ++i) {
      lhs_offsets[i] = 0;
      rhs_offsets[i] = 0;
      extents[i] = scratch.dimension(i);
      reverses[i] = false;
    }

    // At this point, the central part (non-padded area) does not include the
    // gradients back-propagated through padded areas. Those gradient components
    // need be added to the central part.
    //
    // Note that a gradient input element falls into a padded area iff in at
    // least one dimension i, the coordinate x(i) is in the range (python-style)
    // [:paddings(i,0)] or [-paddings(i,1):].

    for (int i = 0; i < Dims; ++i) {
      reverses[i] = true;

      // This handles the case when coordinate in dimension i is in the range
      // [:paddings(i,0)]. This portion is added to the range
      // [paddings(i,0) + offset:2 * paddings(i,0) + offset].
      if (paddings(i, 0) > 0) {
        rhs_offsets[i] = 0;
        lhs_offsets[i] = paddings(i, 0) + offset;
        extents[i] = paddings(i, 0);

        scratch.slice(lhs_offsets, extents).device(device) +=
            scratch.slice(rhs_offsets, extents).reverse(reverses);
      }

      // This handles the case when coordinate in dimension i is in the range
      // [-paddings(i,1):]. This portion is added to the range
      // [-2 * paddings(i,1) - offset:-paddings(i,1) - offset].
      if (paddings(i, 1) > 0) {
        rhs_offsets[i] = scratch.dimension(i) - paddings(i, 1);
        lhs_offsets[i] = rhs_offsets[i] - paddings(i, 1) - offset;
        extents[i] = paddings(i, 1);

        scratch.slice(lhs_offsets, extents).device(device) +=
            scratch.slice(rhs_offsets, extents).reverse(reverses);
      }

      reverses[i] = false;
      lhs_offsets[i] = paddings(i, 0);
      rhs_offsets[i] = paddings(i, 0);
      extents[i] = output.dimension(i);

      // At this point, scratch buffer contains gradient input as if paddings
      // for dimension k = 0,...,i are zeros. Therefore after the loop
      // termination, the central part of the scratch buffer contains the folded
      // gradients.
    }

    // Copy the central part of the scratch buffer to the output.
    output.device(device) = scratch.slice(rhs_offsets, extents);
  }
};
}  // namespace functor
}  // namespace tensorflow

#endif  // TENSORFLOW_KERNELS_MIRROR_PAD_OP_H_
