// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2014 Benoit Steiner <benoit.steiner.goog@gmail.com>
// Copyright (C) 2015 Jianwei Cui <thucjw@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_CXX11_TENSOR_TENSOR_CONVOLUTIONBYFFT_H
#define EIGEN_CXX11_TENSOR_TENSOR_CONVOLUTIONBYFFT_H

namespace Eigen {

/** \class TensorConvolutionByFFT
  * \ingroup CXX11_Tensor_Module
  *
  * \brief Tensor convolution class.
  *
  *
  */
namespace internal {


template<typename Dimensions, typename InputXprType, typename KernelXprType>
struct traits<TensorConvolutionByFFTOp<Dimensions, InputXprType, KernelXprType> >
{
  // Type promotion to handle the case where the types of the lhs and the rhs are different.
  typedef typename promote_storage_type<typename InputXprType::Scalar,
                                        typename KernelXprType::Scalar>::ret Scalar;
  typedef typename packet_traits<Scalar>::type Packet;
  typedef typename promote_storage_type<typename traits<InputXprType>::StorageKind,
                                        typename traits<KernelXprType>::StorageKind>::ret StorageKind;
  typedef typename promote_index_type<typename traits<InputXprType>::Index,
                                      typename traits<KernelXprType>::Index>::type Index;
  typedef typename InputXprType::Nested LhsNested;
  typedef typename KernelXprType::Nested RhsNested;
  typedef typename remove_reference<LhsNested>::type _LhsNested;
  typedef typename remove_reference<RhsNested>::type _RhsNested;
  static const int NumDimensions = traits<InputXprType>::NumDimensions;
  static const int Layout = traits<InputXprType>::Layout;

  enum {
    Flags = 0,
  };
};

template<typename Dimensions, typename InputXprType, typename KernelXprType>
struct eval<TensorConvolutionByFFTOp<Dimensions, InputXprType, KernelXprType>, Eigen::Dense>
{
  typedef const TensorConvolutionByFFTOp<Dimensions, InputXprType, KernelXprType>& type;
};

template<typename Dimensions, typename InputXprType, typename KernelXprType>
struct nested<TensorConvolutionByFFTOp<Dimensions, InputXprType, KernelXprType>, 1, typename eval<TensorConvolutionByFFTOp<Dimensions, InputXprType, KernelXprType> >::type>
{
  typedef TensorConvolutionByFFTOp<Dimensions, InputXprType, KernelXprType> type;
};

}  // end namespace internal



template<typename Indices, typename InputXprType, typename KernelXprType>
class TensorConvolutionByFFTOp : public TensorBase<TensorConvolutionByFFTOp<Indices, InputXprType, KernelXprType> >
{
  public:
  typedef typename Eigen::internal::traits<TensorConvolutionByFFTOp>::Scalar Scalar;
  typedef typename Eigen::internal::traits<TensorConvolutionByFFTOp>::Packet Packet;
  typedef typename Eigen::NumTraits<Scalar>::Real RealScalar;
  typedef typename internal::promote_storage_type<typename InputXprType::CoeffReturnType,
                                                  typename KernelXprType::CoeffReturnType>::ret CoeffReturnType;
  typedef typename internal::promote_storage_type<typename InputXprType::PacketReturnType,
                                                  typename KernelXprType::PacketReturnType>::ret PacketReturnType;
  typedef typename Eigen::internal::nested<TensorConvolutionByFFTOp>::type Nested;
  typedef typename Eigen::internal::traits<TensorConvolutionByFFTOp>::StorageKind StorageKind;
  typedef typename Eigen::internal::traits<TensorConvolutionByFFTOp>::Index Index;

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE TensorConvolutionByFFTOp(const InputXprType& input, const KernelXprType& kernel, const Indices& dims)
      : m_input_xpr(input), m_kernel_xpr(kernel), m_indices(dims) {}

    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
    const Indices& indices() const { return m_indices; }

    /** \returns the nested expressions */
    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
    const typename internal::remove_all<typename InputXprType::Nested>::type&
    inputExpression() const { return m_input_xpr; }

    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
    const typename internal::remove_all<typename KernelXprType::Nested>::type&
    kernelExpression() const { return m_kernel_xpr; }

  protected:
    typename InputXprType::Nested m_input_xpr;
    typename KernelXprType::Nested m_kernel_xpr;
    const Indices m_indices;
};


template<typename Indices, typename InputArgType, typename KernelArgType, typename Device>
struct TensorEvaluator<const TensorConvolutionByFFTOp<Indices, InputArgType, KernelArgType>, Device>
{
  typedef TensorConvolutionByFFTOp<Indices, InputArgType, KernelArgType> XprType;

  typedef typename XprType::Scalar Scalar;
  typedef typename XprType::CoeffReturnType CoeffReturnType;
  typedef typename XprType::PacketReturnType PacketReturnType;

  typedef typename Eigen::NumTraits<Scalar>::Real RealScalar;

  static const int NumDims = internal::array_size<typename TensorEvaluator<InputArgType, Device>::Dimensions>::value;
  static const int NumKernelDims = internal::array_size<Indices>::value;
  typedef typename XprType::Index Index;
  typedef DSizes<Index, NumDims> Dimensions;

  enum {
    IsAligned = TensorEvaluator<InputArgType, Device>::IsAligned &
                TensorEvaluator<KernelArgType, Device>::IsAligned,
    PacketAccess = false,
    BlockAccess = false,
    Layout = TensorEvaluator<InputArgType, Device>::Layout,
    CoordAccess = false,  // to be implemented
  };

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE TensorEvaluator(const XprType& op, const Device& device)
      : m_inputImpl(op.inputExpression(), device), m_kernelImpl(op.kernelExpression(), device), m_kernelArg(op.kernelExpression()), m_kernel(NULL), m_local_kernel(false), m_device(device)
  {
    EIGEN_STATIC_ASSERT((static_cast<int>(TensorEvaluator<InputArgType, Device>::Layout) == static_cast<int>(TensorEvaluator<KernelArgType, Device>::Layout)), YOU_MADE_A_PROGRAMMING_MISTAKE);

    const typename TensorEvaluator<InputArgType, Device>::Dimensions& input_dims = m_inputImpl.dimensions();
    const typename TensorEvaluator<KernelArgType, Device>::Dimensions& kernel_dims = m_kernelImpl.dimensions();

    if (static_cast<int>(Layout) == static_cast<int>(ColMajor)) {
      m_inputStride[0] = 1;
      for (int i = 1; i < NumDims; ++i) {
        m_inputStride[i] = m_inputStride[i - 1] * input_dims[i - 1];
      }
    } else {
      m_inputStride[NumDims - 1] = 1;
      for (int i = NumDims - 2; i >= 0; --i) {
        m_inputStride[i] = m_inputStride[i + 1] * input_dims[i + 1];
      }
    }

    m_dimensions = m_inputImpl.dimensions();
    if (static_cast<int>(Layout) == static_cast<int>(ColMajor)) {
      for (int i = 0; i < NumKernelDims; ++i) {
        const Index index = op.indices()[i];
        const Index input_dim = input_dims[index];
        const Index kernel_dim = kernel_dims[i];
        const Index result_dim = input_dim - kernel_dim + 1;
        m_dimensions[index] = result_dim;
        if (i > 0) {
          m_kernelStride[i] = m_kernelStride[i - 1] * kernel_dims[i - 1];
        } else {
          m_kernelStride[0] = 1;
        }
        m_indexStride[i] = m_inputStride[index];
      }

      m_outputStride[0] = 1;
      for (int i = 1; i < NumDims; ++i) {
        m_outputStride[i] = m_outputStride[i - 1] * m_dimensions[i - 1];
      }
    } else {
      for (int i = NumKernelDims - 1; i >= 0; --i) {
        const Index index = op.indices()[i];
        const Index input_dim = input_dims[index];
        const Index kernel_dim = kernel_dims[i];
        const Index result_dim = input_dim - kernel_dim + 1;
        m_dimensions[index] = result_dim;
        if (i < NumKernelDims - 1) {
          m_kernelStride[i] = m_kernelStride[i + 1] * kernel_dims[i + 1];
        } else {
          m_kernelStride[NumKernelDims - 1] = 1;
        }
        m_indexStride[i] = m_inputStride[index];
      }

      m_outputStride[NumDims - 1] = 1;
      for (int i = NumDims - 2; i >= 0; --i) {
        m_outputStride[i] = m_outputStride[i + 1] * m_dimensions[i + 1];
      }
    }
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Dimensions& dimensions() const { return m_dimensions; }


  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE bool evalSubExprsIfNeeded(Scalar* data) {
    m_inputImpl.evalSubExprsIfNeeded(NULL);
    m_kernelImpl.evalSubExprsIfNeeded(NULL);

    typedef typename internal::traits<InputArgType>::Index TensorIndex;

    Tensor<Scalar, NumDims, Layout, TensorIndex> input(m_inputImpl.dimensions());
    for (int i = 0; i < m_inputImpl.dimensions().TotalSize(); ++i) {
      input.data()[i] = m_inputImpl.coeff(i);
    }

    Tensor<Scalar, NumDims, Layout, TensorIndex> kernel(m_kernelImpl.dimensions());
    for (int i = 0; i < m_kernelImpl.dimensions().TotalSize(); ++i) {
      kernel.data()[i] = m_kernelImpl.coeff(i);
    }

    array<std::pair<ptrdiff_t, ptrdiff_t>, NumDims> paddings;
    for (int i = 0; i < NumDims; ++i) {
      paddings[i] = std::make_pair(0, m_inputImpl.dimensions()[i] - m_kernelImpl.dimensions()[i]);
    }

    Eigen::array<bool, NumKernelDims> reverse;
    for (int i = 0; i < NumKernelDims; ++i) {
      reverse[i] = true;
    }

    Eigen::array<bool, NumDims> fft;
    for (int i = 0; i < NumDims; ++i) {
      fft[i] = i;
    }

    Eigen::DSizes<TensorIndex, NumDims> slice_offsets;
    for (int i = 0; i < NumDims; ++i) {
      slice_offsets[i] = m_kernelImpl.dimensions()[i] - 1;
    }

    Eigen::DSizes<TensorIndex, NumDims> slice_extents;
    for (int i = 0; i < NumDims; ++i) {
      slice_extents[i] = m_inputImpl.dimensions()[i] - m_kernelImpl.dimensions()[i] + 1;
    }

    Tensor<Scalar, NumDims, Layout, TensorIndex> kernel_variant =  kernel.reverse(reverse).pad(paddings);
    Tensor<std::complex<Scalar>, NumDims, Layout, TensorIndex> kernel_fft =  kernel_variant.template fft<Eigen::BothParts, FFT_FORWARD>(fft);
    //Tensor<std::complex<Scalar>, NumDims, Layout|IndexType> kernel_fft =  kernel.reverse(reverse).pad(paddings).template fft<2>(fft);
    Tensor<std::complex<Scalar>, NumDims, Layout, TensorIndex> input_fft = input.template fft<Eigen::BothParts, FFT_FORWARD>(fft);
    Tensor<std::complex<Scalar>, NumDims, Layout, TensorIndex> prod = (input_fft * kernel_fft).template fft<Eigen::BothParts, FFT_REVERSE>(fft);
    Tensor<std::complex<Scalar>, NumDims, Layout, TensorIndex> tensor_result = prod.slice(slice_offsets, slice_extents);

    for (int i = 0; i < tensor_result.size(); ++i) {
      data[i] = std::real(tensor_result.data()[i]);
    }
    return false;
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void cleanup() {
    m_inputImpl.cleanup();
    if (m_local_kernel) {
      m_device.deallocate((void*)m_kernel);
      m_local_kernel = false;
    }
    m_kernel = NULL;
  }

  void evalTo(typename XprType::Scalar* buffer) {
    evalSubExprsIfNeeded(NULL);
    for (int i = 0; i < dimensions().TotalSize(); ++i) {
      buffer[i] += coeff(i);
    }
    cleanup();
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE CoeffReturnType coeff(Index index) const
  {
    CoeffReturnType result = CoeffReturnType(0);
    return result;
  }

  EIGEN_DEVICE_FUNC Scalar* data() const { return NULL; }

 private:
  array<Index, NumDims> m_inputStride;
  array<Index, NumDims> m_outputStride;

  array<Index, NumKernelDims> m_indexStride;
  array<Index, NumKernelDims> m_kernelStride;
  TensorEvaluator<InputArgType, Device> m_inputImpl;
  TensorEvaluator<KernelArgType, Device> m_kernelImpl;
  Dimensions m_dimensions;

  KernelArgType m_kernelArg;
  const Scalar* m_kernel;
  bool m_local_kernel;
  const Device& m_device;
};

} // end namespace Eigen

#endif // EIGEN_CXX11_TENSOR_TENSOR_CONVOLUTIONBYFFT_H
