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

#ifndef TENSORFLOW_CORE_KERNELS_WRAP_PAD_OP_H_
#define TENSORFLOW_CORE_KERNELS_WRAP_PAD_OP_H_

#include "unsupported/Eigen/CXX11/Tensor"  // from @eigen_archive
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/platform/types.h"

namespace Eigen {
template <typename PaddingDimensions, typename XprType>
class TensorWrapPadOp;

namespace internal {
template <typename PaddingDimensions, typename XprType>
struct traits<TensorWrapPadOp<PaddingDimensions, XprType>>
    : public traits<XprType> {
  typedef typename XprType::Scalar Scalar;
  typedef traits<XprType> XprTraits;
  typedef typename XprTraits::StorageKind StorageKind;
  typedef typename XprTraits::Index Index;
  typedef typename XprType::Nested Nested;
  typedef std::remove_reference_t<Nested> _Nested;
  static constexpr int NumDimensions = XprTraits::NumDimensions;
  static constexpr int Layout = XprTraits::Layout;
};

template <typename PaddingDimensions, typename XprType>
struct eval<TensorWrapPadOp<PaddingDimensions, XprType>, Eigen::Dense> {
  typedef const TensorWrapPadOp<PaddingDimensions, XprType>& type;
};

template <typename PaddingDimensions, typename XprType>
struct nested<
    TensorWrapPadOp<PaddingDimensions, XprType>, 1,
    typename eval<TensorWrapPadOp<PaddingDimensions, XprType>>::type> {
  typedef TensorWrapPadOp<PaddingDimensions, XprType> type;
};
}  // namespace internal

template <typename PaddingDimensions, typename XprType>
class TensorWrapPadOp
    : public TensorBase<TensorWrapPadOp<PaddingDimensions, XprType>,
                        ReadOnlyAccessors> {
  public:
    typedef typename Eigen::internal::traits<TensorWrapPadOp>::Scalar Scalar;
    typedef typename Eigen::NumTraits<Scalar>::Real RealScalar;
    typedef typename XprType::CoeffReturnType CoeffReturnType;
    typedef typename Eigen::internal::nested<TensorWrapPadOp>::type Nested;
    typedef typename Eigen::internal::traits<TensorWrapPadOp>::StorageKind
        StorageKind;
    typedef typename Eigen::internal::traits<TensorWrapPadOp>::Index Index;

    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE TensorWrapPadOp(
        const XprType& expr, const PaddingDimensions& padding_dims)
        : xpr_(expr), padding_dims_(padding_dims) {}

    EIGEN_DEVICE_FUNC
    const PaddingDimensions& padding() const { return padding_dims_; }

    EIGEN_DEVICE_FUNC
    const typename internal::remove_all<typename XprType::Nested>::type&
    expression() const {
      return xpr_;
    }

  protected:
    typename XprType::Nested xpr_;
    const PaddingDimensions padding_dims_;
};

template <typename PaddingDimensions, typename ArgType, typename Device>
struct TensorEvaluator<const TensorWrapPadOp<PaddingDimensions, ArgType>,
                       Device> {
  typedef TensorWrapPadOp<PaddingDimensions, ArgType> XprType;
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
    BlockAccessV2 = false,
    PreferBlockAccess = false,
    Layout = TensorEvaluator<ArgType, Device>::Layout,
    CoordAccess = true,
    RawAccess = false
  };

  //===- Tensor block evaluation strategy (see TensorBlock.h) -------------===//
  typedef internal::TensorBlockNotImplemented TensorBlock;
  //===--------------------------------------------------------------------===//

  EIGEN_STRONG_INLINE TensorEvaluator(const XprType& op, const Device& device)
      : impl_(op.expression(), device), padding_(op.padding()) {
    EIGEN_STATIC_ASSERT(Dims > 0, YOU_MADE_A_PROGRAMMING_MISTAKE)

    // This should trigger compilation error if padding dimensions and
    // expression dimensions do not match.
    dimensions_ = impl_.dimensions();
    for (int dim = 0; dim < Dims; ++dim) {
      eigen_assert(padding_[dim].first + 1 <= dimensions_[dim]);
      eigen_assert(padding_[dim].second + 1 <= dimensions_[dim]);
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

  EIGEN_STRONG_INLINE bool evalSubExprsIfNeeded(Scalar*) {
    impl_.evalSubExprsIfNeeded(nullptr);
    return true;
  }

  EIGEN_STRONG_INLINE void cleanup() { impl_.cleanup(); }

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

    const Index index_mod = index % (dimensions_[dim] * output_strides_[dim]);
    if (left <= index_mod && (index_mod + kPacketSize - 1) < right) {
      return impl_.template packet<Unaligned>(input_index);
    }

    // If the road is not contiguous, then fall back to coeff().
    EIGEN_ALIGN_MAX std::remove_const_t<CoeffReturnType> values[kPacketSize];
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
        return m + k;
      }
      if (k < m) {
        return k;
      }
      return k - m; 
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
};
}  // namespace Eigen

namespace tensorflow {
namespace functor {

template <typename Device, typename T, typename Tpaddings, int Dims>
struct WrapPad {
  void operator()(const Device& device,
                  typename TTypes<T, Dims, int32>::Tensor output,
                  typename TTypes<T, Dims, int32>::ConstTensor input,
                  typename TTypes<Tpaddings>::ConstMatrix padding) {
    Eigen::array<Eigen::IndexPair<int32>, Dims> padding_dims;

    for (int i = 0; i < Dims; ++i) {
      padding_dims[i] = Eigen::IndexPair<int32>(padding(i, 0), padding(i, 1));
    }

    output.device(device) = WrapPadOp(input, padding_dims);
  }  

  template <typename PaddingDimensions, typename Derived>
  static const Eigen::TensorWrapPadOp<PaddingDimensions, const Derived>
  WrapPadOp(
      const Eigen::TensorBase<Derived, Eigen::ReadOnlyAccessors>& tensor,
      const PaddingDimensions& padding) {
    return Eigen::TensorWrapPadOp<PaddingDimensions, const Derived>(
        static_cast<const Derived&>(tensor), padding);
  }  
};

}  // namespace functor
}  // namespace tensorflow

#endif  // TESNORFLOW_CORE_KERNELS_WRAP_PAD_OP_H_
