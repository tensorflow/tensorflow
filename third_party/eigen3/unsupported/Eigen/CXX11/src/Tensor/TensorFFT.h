// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2015 Jianwei Cui <thucjw@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_CXX11_TENSOR_TENSOR_FFT_H
#define EIGEN_CXX11_TENSOR_TENSOR_FFT_H
namespace Eigen {

/** \class TensorFFT
  * \ingroup CXX11_Tensor_Module
  *
  * \brief Tensor FFT class.
  *
  * TODO:
  * Vectorize the Cooley Tukey and the Bluestein algorithm
  * Add support for multithreaded evaluation
  * Improve the performance on GPU
  */

template <bool NeedUprade> struct MakeComplex {
  template <typename T>
  #if defined(EIGEN_USE_GPU) && defined(__CUDACC__) && !defined(__GCUDACC__)
  EIGEN_DEVICE_FUNC
  #endif
  T operator() (const T& val) const { return val; }
};

template <> struct MakeComplex<true> {
  template <typename T>
  #if defined(EIGEN_USE_GPU) && defined(__CUDACC__) && !defined(__GCUDACC__)
  EIGEN_DEVICE_FUNC
  #endif
  std::complex<T> operator() (const T& val) const { return std::complex<T>(val, 0); }
};

template <> struct MakeComplex<false> {
  template <typename T>
  #if defined(EIGEN_USE_GPU) && defined(__CUDACC__) && !defined(__GCUDACC__)
  EIGEN_DEVICE_FUNC
  #endif
  std::complex<T> operator() (const std::complex<T>& val) const { return val; }
};

template <int ResultType> struct PartOf {
  template <typename T> T operator() (const T& val) const { return val; }
};

template <> struct PartOf<RealPart> {
  template <typename T> T operator() (const std::complex<T>& val) const { return val.real(); }
};

template <> struct PartOf<ImagPart> {
  template <typename T> T operator() (const std::complex<T>& val) const { return val.imag(); }
};

namespace internal {
template <typename FFT, typename XprType, int FFTResultType, int FFTDir>
struct traits<TensorFFTOp<FFT, XprType, FFTResultType, FFTDir> > : public traits<XprType> {
  typedef traits<XprType> XprTraits;
  typedef typename NumTraits<typename XprTraits::Scalar>::Real RealScalar;
  typedef typename std::complex<RealScalar> ComplexScalar;
  typedef typename XprTraits::Scalar InputScalar;
  typedef typename conditional<FFTResultType == RealPart || FFTResultType == ImagPart, RealScalar, ComplexScalar>::type OutputScalar;
  typedef typename XprTraits::StorageKind StorageKind;
  typedef typename XprTraits::Index Index;
  typedef typename XprType::Nested Nested;
  typedef typename remove_reference<Nested>::type _Nested;
  static const int NumDimensions = XprTraits::NumDimensions;
  static const int Layout = XprTraits::Layout;
};

template <typename FFT, typename XprType, int FFTResultType, int FFTDirection>
struct eval<TensorFFTOp<FFT, XprType, FFTResultType, FFTDirection>, Eigen::Dense> {
  typedef const TensorFFTOp<FFT, XprType, FFTResultType, FFTDirection>& type;
};

template <typename FFT, typename XprType, int FFTResultType, int FFTDirection>
struct nested<TensorFFTOp<FFT, XprType, FFTResultType, FFTDirection>, 1, typename eval<TensorFFTOp<FFT, XprType, FFTResultType, FFTDirection> >::type> {
  typedef TensorFFTOp<FFT, XprType, FFTResultType, FFTDirection> type;
};

}  // end namespace internal

template <typename FFT, typename XprType, int FFTResultType, int FFTDir>
class TensorFFTOp : public TensorBase<TensorFFTOp<FFT, XprType, FFTResultType, FFTDir>, ReadOnlyAccessors> {
 public:
  typedef typename Eigen::internal::traits<TensorFFTOp>::Scalar Scalar;
  typedef typename Eigen::NumTraits<Scalar>::Real RealScalar;
  typedef typename std::complex<RealScalar> ComplexScalar;
  typedef typename internal::conditional<FFTResultType == RealPart || FFTResultType == ImagPart, RealScalar, ComplexScalar>::type OutputScalar;
  typedef OutputScalar CoeffReturnType;
  typedef typename Eigen::internal::nested<TensorFFTOp>::type Nested;
  typedef typename Eigen::internal::traits<TensorFFTOp>::StorageKind StorageKind;
  typedef typename Eigen::internal::traits<TensorFFTOp>::Index Index;

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE TensorFFTOp(const XprType& expr, const FFT& fft)
      : m_xpr(expr), m_fft(fft) {}

  EIGEN_DEVICE_FUNC
  const FFT& fft() const { return m_fft; }

  EIGEN_DEVICE_FUNC
  const typename internal::remove_all<typename XprType::Nested>::type& expression() const {
    return m_xpr;
  }

 protected:
  typename XprType::Nested m_xpr;
  const FFT m_fft;
};

// Eval as rvalue
template <typename FFT, typename ArgType, typename Device, int FFTResultType, int FFTDir>
struct TensorEvaluator<const TensorFFTOp<FFT, ArgType, FFTResultType, FFTDir>, Device> {
  typedef TensorFFTOp<FFT, ArgType, FFTResultType, FFTDir> XprType;
  typedef typename XprType::Index Index;
  static const int NumDims = internal::array_size<typename TensorEvaluator<ArgType, Device>::Dimensions>::value;
  typedef DSizes<Index, NumDims> Dimensions;
  typedef typename XprType::Scalar Scalar;
  typedef typename Eigen::NumTraits<Scalar>::Real RealScalar;
  typedef typename std::complex<RealScalar> ComplexScalar;
  typedef typename TensorEvaluator<ArgType, Device>::Dimensions InputDimensions;
  typedef internal::traits<XprType> XprTraits;
  typedef typename XprTraits::Scalar InputScalar;
  typedef typename internal::conditional<FFTResultType == RealPart || FFTResultType == ImagPart, RealScalar, ComplexScalar>::type OutputScalar;
  typedef OutputScalar CoeffReturnType;
  typedef typename PacketType<OutputScalar, Device>::type PacketReturnType;

  enum {
    IsAligned = false,
    PacketAccess = true,
    BlockAccess = false,
    Layout = TensorEvaluator<ArgType, Device>::Layout,
    CoordAccess = false,
  };

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE TensorEvaluator(const XprType& op, const Device& device) : m_data(NULL), m_impl(op.expression(), device), m_fft(op.fft()), m_device(device) {
    const typename TensorEvaluator<ArgType, Device>::Dimensions& input_dims = m_impl.dimensions();
    for (int i = 0; i < NumDims; ++i) {
      eigen_assert(input_dims[i] > 0);
      m_dimensions[i] = input_dims[i];
    }

    if (static_cast<int>(Layout) == static_cast<int>(ColMajor)) {
      m_strides[0] = 1;
      for (int i = 1; i < NumDims; ++i) {
        m_strides[i] = m_strides[i - 1] * m_dimensions[i - 1];
      }
    } else {
      m_strides[NumDims - 1] = 1;
      for (int i = NumDims - 2; i >= 0; --i) {
        m_strides[i] = m_strides[i + 1] * m_dimensions[i + 1];
      }
    }
    m_size = m_dimensions.TotalSize();
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Dimensions& dimensions() const {
    return m_dimensions;
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE bool evalSubExprsIfNeeded(OutputScalar* data) {
    m_impl.evalSubExprsIfNeeded(NULL);
    if (data) {
      evalToBuf(data);
      return false;
    } else {
      m_data = (CoeffReturnType*)m_device.allocate(sizeof(CoeffReturnType) * m_size);
      evalToBuf(m_data);
      return true;
    }
  }


  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void cleanup() {
    if (m_data) {
      m_device.deallocate(m_data);
      m_data = NULL;
    }
    m_impl.cleanup();
  }

  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE CoeffReturnType coeff(Index index) const {
    return m_data[index];
  }

  template<int LoadMode>
  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE PacketReturnType packet(Index index) const {
    return internal::ploadt<PacketReturnType, LoadMode>(m_data + index);
  }

  EIGEN_DEVICE_FUNC Scalar* data() const { return m_data; }


 private:
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void evalToBuf(OutputScalar* data) {
    const bool write_to_out = internal::is_same<OutputScalar, ComplexScalar>::value;
    ComplexScalar* buf = write_to_out ? (ComplexScalar*)data : (ComplexScalar*)m_device.allocate(sizeof(ComplexScalar) * m_size);

    for (int i = 0; i < m_size; ++i) {
      buf[i] = MakeComplex<internal::is_same<InputScalar, RealScalar>::value>()(m_impl.coeff(i));
    }

    for (int i = 0; i < m_fft.size(); ++i) {
      int dim = m_fft[i];
      eigen_assert(dim >= 0 && dim < NumDims);
      Index line_len = m_dimensions[dim];
      eigen_assert(line_len >= 1);
      ComplexScalar* line_buf = (ComplexScalar*)m_device.allocate(sizeof(ComplexScalar) * line_len);
      const bool is_power_of_two = isPowerOfTwo(line_len);
      const int good_composite = is_power_of_two ? 0 : findGoodComposite(line_len);
      const int log_len = is_power_of_two ? getLog2(line_len) : getLog2(good_composite);

      ComplexScalar* a = is_power_of_two ? NULL : (ComplexScalar*)m_device.allocate(sizeof(ComplexScalar) * good_composite);
      ComplexScalar* b = is_power_of_two ? NULL : (ComplexScalar*)m_device.allocate(sizeof(ComplexScalar) * good_composite);
      ComplexScalar* pos_j_base_powered = is_power_of_two ? NULL : (ComplexScalar*)m_device.allocate(sizeof(ComplexScalar) * (line_len + 1));
      if (!is_power_of_two) {
        ComplexScalar pos_j_base = ComplexScalar(std::cos(M_PI/line_len), std::sin(M_PI/line_len));
        for (int i = 0; i < line_len + 1; ++i) {
          pos_j_base_powered[i] = std::pow(pos_j_base, i * i);
        }
      }

      for (Index partial_index = 0; partial_index < m_size / line_len; ++partial_index) {
        Index base_offset = getBaseOffsetFromIndex(partial_index, dim);

        // get data into line_buf
        for (int j = 0; j < line_len; ++j) {
          Index offset = getIndexFromOffset(base_offset, dim, j);
          line_buf[j] = buf[offset];
        }

        // processs the line
        if (is_power_of_two) {
          processDataLineCooleyTukey(line_buf, line_len, log_len);
        }
        else {
          processDataLineBluestein(line_buf, line_len, good_composite, log_len, a, b, pos_j_base_powered);
        }

        // write back
        for (int j = 0; j < line_len; ++j) {
          const ComplexScalar div_factor = (FFTDir == FFT_FORWARD) ? ComplexScalar(1, 0) : ComplexScalar(line_len, 0);
          Index offset = getIndexFromOffset(base_offset, dim, j);
          buf[offset] =  line_buf[j] / div_factor;
        }
      }
      m_device.deallocate(line_buf);
      if (!pos_j_base_powered) {
        m_device.deallocate(a);
        m_device.deallocate(b);
        m_device.deallocate(pos_j_base_powered);
      }
    }

    if(!write_to_out) {
      for (int i = 0; i < m_size; ++i) {
        data[i] = PartOf<FFTResultType>()(buf[i]);
      }
      m_device.deallocate(buf);
    }
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE static bool isPowerOfTwo(int x) {
    eigen_assert(x > 0);
    return !(x & (x - 1));
  }

  //the composite number for padding, used in Bluestein's FFT algorithm
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE static int findGoodComposite(int n) {
    int i = 2;
    while (i < 2 * n - 1) i *= 2;
    return i;
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE static int getLog2(int m) {
    int log2m = 0;
    while (m >>= 1) log2m++;
    return log2m;
  }

  // Call Cooley Tukey algorithm directly, data length must be power of 2
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void processDataLineCooleyTukey(ComplexScalar* line_buf, int line_len, int log_len) {
    eigen_assert(isPowerOfTwo(line_len));
    scramble_FFT(line_buf, line_len);
    compute_1D_Butterfly<FFTDir>(line_buf, line_len, log_len);
  }

  // Call Bluestein's FFT algorithm, m is a good composite number greater than (2 * n - 1), used as the padding length
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void processDataLineBluestein(ComplexScalar* line_buf, int line_len, int good_composite, int log_len, ComplexScalar* a, ComplexScalar* b, const ComplexScalar* pos_j_base_powered) {
    int n = line_len;
    int m = good_composite;
    ComplexScalar* data = line_buf;

    for (int i = 0; i < n; ++i) {
      if(FFTDir == FFT_FORWARD) {
        a[i] = data[i] * std::conj(pos_j_base_powered[i]);
      }
      else {
        a[i] = data[i] * pos_j_base_powered[i];
      }
    }
    for (int i = n; i < m; ++i) {
      a[i] = ComplexScalar(0, 0);
    }

    for (int i = 0; i < n; ++i) {
      if(FFTDir == FFT_FORWARD) {
        b[i] = pos_j_base_powered[i];
      }
      else {
        b[i] = std::conj(pos_j_base_powered[i]);
      }
    }
    for (int i = n; i < m - n; ++i) {
      b[i] = ComplexScalar(0, 0);
    }
    for (int i = m - n; i < m; ++i) {
      if(FFTDir == FFT_FORWARD) {
        b[i] = pos_j_base_powered[m-i];
      }
      else {
        b[i] = std::conj(pos_j_base_powered[m-i]);
      }
    }

    scramble_FFT(a, m);
    compute_1D_Butterfly<FFT_FORWARD>(a, m, log_len);

    scramble_FFT(b, m);
    compute_1D_Butterfly<FFT_FORWARD>(b, m, log_len);

    for (int i = 0; i < m; ++i) {
      a[i] *= b[i];
    }

    scramble_FFT(a, m);
    compute_1D_Butterfly<FFT_REVERSE>(a, m, log_len);

    //Do the scaling after ifft
    for (int i = 0; i < m; ++i) {
      a[i] /= m;
    }

    for (int i = 0; i < n; ++i) {
      if(FFTDir == FFT_FORWARD) {
        data[i] = a[i] * std::conj(pos_j_base_powered[i]);
      }
      else {
        data[i] = a[i] * pos_j_base_powered[i];
      }
    }
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE static void scramble_FFT(ComplexScalar* data, int n) {
    eigen_assert(isPowerOfTwo(n));
    int j = 1;
    for (int i = 1; i < n; ++i){
      if (j > i) {
        std::swap(data[j-1], data[i-1]);
      }
      int m = n >> 1;
      while (m >= 2 && j > m) {
        j -= m;
        m >>= 1;
      }
      j += m;
    }
  }

  template<int Dir>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void compute_1D_Butterfly(ComplexScalar* data, int n, int n_power_of_2) {
    eigen_assert(isPowerOfTwo(n));
    if (n == 1) {
      return;
    }
    else if (n == 2) {
      ComplexScalar tmp = data[1];
      data[1] = data[0] - data[1];
      data[0] += tmp;
      return;
    }
    else if (n == 4) {
      ComplexScalar tmp[4];
      tmp[0] = data[0] + data[1];
      tmp[1] = data[0] - data[1];
      tmp[2] = data[2] + data[3];
      if(Dir == FFT_FORWARD) {
        tmp[3] = ComplexScalar(0.0, -1.0) * (data[2] - data[3]);
      }
      else {
        tmp[3] = ComplexScalar(0.0, 1.0) * (data[2] - data[3]);
      }
      data[0] = tmp[0] + tmp[2];
      data[1] = tmp[1] + tmp[3];
      data[2] = tmp[0] - tmp[2];
      data[3] = tmp[1] - tmp[3];
      return;
    }
    else if (n == 8) {
      ComplexScalar tmp_1[8];
      ComplexScalar tmp_2[8];

      tmp_1[0] = data[0] + data[1];
      tmp_1[1] = data[0] - data[1];
      tmp_1[2] = data[2] + data[3];
      if (Dir == FFT_FORWARD) {
        tmp_1[3] = (data[2] - data[3]) * ComplexScalar(0, -1);
      }
      else {
        tmp_1[3] = (data[2] - data[3]) * ComplexScalar(0, 1);
      }
      tmp_1[4] = data[4] + data[5];
      tmp_1[5] = data[4] - data[5];
      tmp_1[6] = data[6] + data[7];
      if (Dir == FFT_FORWARD) {
        tmp_1[7] = (data[6] - data[7]) * ComplexScalar(0, -1);
      }
      else {
        tmp_1[7] = (data[6] - data[7]) * ComplexScalar(0, 1);
      }
      tmp_2[0] = tmp_1[0] + tmp_1[2];
      tmp_2[1] = tmp_1[1] + tmp_1[3];
      tmp_2[2] = tmp_1[0] - tmp_1[2];
      tmp_2[3] = tmp_1[1] - tmp_1[3];
      tmp_2[4] = tmp_1[4] + tmp_1[6];
      // SQRT2DIV2 = sqrt(2)/2
      #define SQRT2DIV2 0.7071067811865476
      if (Dir == FFT_FORWARD) {
        tmp_2[5] = (tmp_1[5] + tmp_1[7]) * ComplexScalar(SQRT2DIV2, -SQRT2DIV2);
        tmp_2[6] = (tmp_1[4] - tmp_1[6]) * ComplexScalar(0, -1);
        tmp_2[7] = (tmp_1[5] - tmp_1[7]) * ComplexScalar(-SQRT2DIV2, -SQRT2DIV2);
      }
      else {
        tmp_2[5] = (tmp_1[5] + tmp_1[7]) * ComplexScalar(SQRT2DIV2, SQRT2DIV2);
        tmp_2[6] = (tmp_1[4] - tmp_1[6]) * ComplexScalar(0, 1);
        tmp_2[7] = (tmp_1[5] - tmp_1[7]) * ComplexScalar(-SQRT2DIV2, SQRT2DIV2);
      }
      data[0] = tmp_2[0] + tmp_2[4];
      data[1] = tmp_2[1] + tmp_2[5];
      data[2] = tmp_2[2] + tmp_2[6];
      data[3] = tmp_2[3] + tmp_2[7];
      data[4] = tmp_2[0] - tmp_2[4];
      data[5] = tmp_2[1] - tmp_2[5];
      data[6] = tmp_2[2] - tmp_2[6];
      data[7] = tmp_2[3] - tmp_2[7];

      return;
    }
    else {
      compute_1D_Butterfly<Dir>(data, n/2, n_power_of_2 - 1);
      compute_1D_Butterfly<Dir>(data + n/2, n/2, n_power_of_2 - 1);
      //Original code:
      //RealScalar wtemp = std::sin(M_PI/n);
      //RealScalar wpi =  -std::sin(2 * M_PI/n);
      RealScalar wtemp = m_sin_PI_div_n_LUT[n_power_of_2];
      RealScalar wpi;
      if (Dir == FFT_FORWARD) {
        wpi =  m_minus_sin_2_PI_div_n_LUT[n_power_of_2];
      }
      else {
        wpi = 0 - m_minus_sin_2_PI_div_n_LUT[n_power_of_2];
      }

      const ComplexScalar wp(wtemp, wpi);
      ComplexScalar w(1.0, 0.0);
      for(int i = 0; i < n/2; i++) {
        ComplexScalar temp(data[i + n/2] * w);
        data[i + n/2] = data[i] - temp;
        data[i] += temp;
        w += w * wp;
      }
      return;
    }
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Index getBaseOffsetFromIndex(Index index, Index omitted_dim) const {
    Index result = 0;

    if (static_cast<int>(Layout) == static_cast<int>(ColMajor)) {
      for (int i = NumDims - 1; i > omitted_dim; --i) {
        const Index partial_m_stride = m_strides[i] / m_dimensions[omitted_dim];
        const Index idx = index / partial_m_stride;
        index -= idx * partial_m_stride;
        result += idx * m_strides[i];
      }
      result += index;
    }
    else {
      for (int i = 0; i < omitted_dim; ++i) {
        const Index partial_m_stride = m_strides[i] / m_dimensions[omitted_dim];
        const Index idx = index / partial_m_stride;
        index -= idx * partial_m_stride;
        result += idx * m_strides[i];
      }
      result += index;
    }
    // Value of index_coords[omitted_dim] is not determined to this step
    return result;
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Index getIndexFromOffset(Index base, Index omitted_dim, Index offset) const {
    Index result = base + offset * m_strides[omitted_dim] ;
    return result;
  }

 protected:
  int m_size;
  const FFT& m_fft;
  Dimensions m_dimensions;
  array<Index, NumDims> m_strides;
  TensorEvaluator<ArgType, Device> m_impl;
  CoeffReturnType* m_data;
  const Device& m_device;

  // This will support a maximum FFT size of 2^32 for each dimension
  // m_sin_PI_div_n_LUT[i] = (-2) * std::sin(M_PI / std::pow(2,i)) ^ 2;
  RealScalar m_sin_PI_div_n_LUT[32] = {
  0.0,
  -2,
  -0.999999999999999,
  -0.292893218813453,
  -0.0761204674887130,
  -0.0192147195967696,
  -0.00481527332780311,
  -0.00120454379482761,
  -3.01181303795779e-04,
  -7.52981608554592e-05,
  -1.88247173988574e-05,
  -4.70619042382852e-06,
  -1.17654829809007e-06,
  -2.94137117780840e-07,
  -7.35342821488550e-08,
  -1.83835707061916e-08,
  -4.59589268710903e-09,
  -1.14897317243732e-09,
  -2.87243293150586e-10,
  -7.18108232902250e-11,
  -1.79527058227174e-11,
  -4.48817645568941e-12,
  -1.12204411392298e-12,
  -2.80511028480785e-13,
  -7.01277571201985e-14,
  -1.75319392800498e-14,
  -4.38298482001247e-15,
  -1.09574620500312e-15,
  -2.73936551250781e-16,
  -6.84841378126949e-17,
  -1.71210344531737e-17,
  -4.28025861329343e-18
  };

  // m_minus_sin_2_PI_div_n_LUT[i] = -std::sin(2 * M_PI / std::pow(2,i));
  RealScalar m_minus_sin_2_PI_div_n_LUT[32] = {
    0.0,
    0.0,
   -1.00000000000000e+00,
   -7.07106781186547e-01,
   -3.82683432365090e-01,
   -1.95090322016128e-01,
   -9.80171403295606e-02,
   -4.90676743274180e-02,
   -2.45412285229123e-02,
   -1.22715382857199e-02,
   -6.13588464915448e-03,
   -3.06795676296598e-03,
   -1.53398018628477e-03,
   -7.66990318742704e-04,
   -3.83495187571396e-04,
   -1.91747597310703e-04,
   -9.58737990959773e-05,
   -4.79368996030669e-05,
   -2.39684498084182e-05,
   -1.19842249050697e-05,
   -5.99211245264243e-06,
   -2.99605622633466e-06,
   -1.49802811316901e-06,
   -7.49014056584716e-07,
   -3.74507028292384e-07,
   -1.87253514146195e-07,
   -9.36267570730981e-08,
   -4.68133785365491e-08,
   -2.34066892682746e-08,
   -1.17033446341373e-08,
   -5.85167231706864e-09,
   -2.92583615853432e-09
  };
};

#if defined(EIGEN_USE_GPU) && defined(__CUDACC__) && !defined(__GCUDACC__)

template<typename OutputScalar, typename RealScalar, typename ComplexScalar, int ResultType>
struct writeToDeviceData {
  void operator()(OutputScalar* d_data, ComplexScalar* data_buf, size_t size) {
  }
};

template<typename OutputScalar, typename RealScalar, typename ComplexScalar>
struct writeToDeviceData<OutputScalar, RealScalar, ComplexScalar, Eigen::BothParts> {
  void operator()(OutputScalar* d_data, ComplexScalar* data_buf, size_t size) {
    cudaMemcpy(d_data, data_buf, size * sizeof(ComplexScalar), cudaMemcpyDeviceToDevice);
  }
};

template<typename OutputScalar, typename RealScalar, typename ComplexScalar>
struct writeToDeviceData<OutputScalar, RealScalar, ComplexScalar, Eigen::RealPart> {
  void operator()(OutputScalar* d_data, ComplexScalar* data_buf, size_t size) {
    cudaMemcpy2D(d_data, sizeof(RealScalar), (RealScalar*) data_buf, 2 * sizeof(RealScalar), sizeof(RealScalar), size, cudaMemcpyDeviceToDevice);
  }
};

template<typename OutputScalar, typename RealScalar, typename ComplexScalar>
struct writeToDeviceData<OutputScalar, RealScalar, ComplexScalar, Eigen::ImagPart> {
  void operator()(OutputScalar* d_data, ComplexScalar* data_buf, size_t size) {
    RealScalar* data_buf_offset = &(((RealScalar*) data_buf)[1]);
    cudaMemcpy2D(d_data, sizeof(RealScalar), data_buf_offset,        2 * sizeof(RealScalar), sizeof(RealScalar), size, cudaMemcpyDeviceToDevice);
  }
};

template <typename InputScalar, typename RealScalar, typename ComplexScalar, typename InputEvaluator>
__global__ void copyValues(ComplexScalar* d_data, InputEvaluator eval, int total_size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < total_size) {
    d_data[i] = MakeComplex<internal::is_same<InputScalar, RealScalar>::value>()(eval.coeff(i));
  }
}

template<typename Scalar, typename Index, int NumDims>
__global__ void fillLineBuf(Scalar* line_buf, Scalar* data_buf, int line_len,
                            array<Index, NumDims> coords, array<Index, NumDims> m_strides, int dim) {
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if(j < line_len) {
    coords[dim] = j;
    Index index = 0;
    for (int i = 0; i < NumDims; ++i) {
      index += coords[i] * m_strides[i];
    }
    line_buf[j] = data_buf[index];
  }
}

template<typename ComplexScalar, typename RealScalar, typename Index, int NumDims>
__global__ void writebackLineBuf(ComplexScalar* line_buf, ComplexScalar* data_buf, int line_len,
                                 array<Index, NumDims> coords, array<Index, NumDims> m_strides, int dim, RealScalar div_factor) {
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if(j < line_len) {
    coords[dim] = j;
    Index index = 0;
    for (int i = 0; i < NumDims; ++i) {
      index += coords[i] * m_strides[i];
    }

    data_buf[index] = line_buf[j];
    ((RealScalar*) data_buf)[2*index] /= div_factor;
    ((RealScalar*) data_buf)[2*index + 1] /= div_factor;
  }
}

template <typename FFT, typename ArgType, int FFTResultType, int FFTDir>
struct TensorEvaluator<const TensorFFTOp<FFT, ArgType, FFTResultType, FFTDir>, GpuDevice> {
  typedef TensorFFTOp<FFT, ArgType, FFTResultType, FFTDir> XprType;
  typedef typename XprType::Index Index;
  static const int NumDims = internal::array_size<typename TensorEvaluator<ArgType, GpuDevice>::Dimensions>::value;
  typedef DSizes<Index, NumDims> Dimensions;
  typedef typename XprType::Scalar Scalar;
  typedef typename XprType::Scalar InputScalar;
  typedef typename Eigen::NumTraits<Scalar>::Real RealScalar;
  typedef typename std::complex<RealScalar> ComplexScalar;
  typedef typename internal::conditional<FFTResultType == Eigen::BothParts, std::complex<RealScalar>, RealScalar>::type OutputScalar;
  typedef typename TensorEvaluator<ArgType, GpuDevice>::Dimensions InputDimensions;
  typedef OutputScalar CoeffReturnType;
  typedef typename PacketType<OutputScalar, GpuDevice>::type PacketReturnType;

  enum {
    IsAligned = false,
    PacketAccess = false,
    BlockAccess = false,
    Layout = TensorEvaluator<ArgType, GpuDevice>::Layout,
  };

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE TensorEvaluator(const XprType& op, const GpuDevice& device) : m_data_buf(NULL), m_impl(op.expression(), device), m_fft(op.fft()) {
    const typename TensorEvaluator<ArgType, GpuDevice>::Dimensions& input_dims = m_impl.dimensions();
    for (int i = 0; i < NumDims; ++i) {
      eigen_assert(input_dims[i] > 0);
      m_dimensions[i] = input_dims[i];
    }

    if (static_cast<int>(Layout) == static_cast<int>(ColMajor)) {
      m_strides[0] = 1;
      for (int i = 1; i < NumDims; ++i) {
        m_strides[i] = m_strides[i - 1] * m_dimensions[i - 1];
      }
    } else {
      m_strides[NumDims - 1] = 1;
      for (int i = NumDims - 2; i >= 0; --i) {
        m_strides[i] = m_strides[i + 1] * m_dimensions[i + 1];
      }
    }
    m_size = m_dimensions.TotalSize();
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Dimensions& dimensions() const {
    return m_dimensions;
  }

  EIGEN_STRONG_INLINE bool evalSubExprsIfNeeded(OutputScalar* d_data) {
    m_impl.evalSubExprsIfNeeded(NULL);
    if (d_data) {
      evalToDeviceData(d_data);
      return false;
    } else {
      evalToSelfDataBuf();
      return true;
    }
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Index getIndexFromCoords(const array<Index, NumDims> & coords) const {
    Index result = 0;
    for (int i = 0; i < NumDims; ++i) {
      result += coords[i] * m_strides[i];
    }
    return result;
  }

  EIGEN_STRONG_INLINE array<Index, NumDims> getPartialCoordsFromIndex(Index index, Index omitted_dim) const {
    array<Index, NumDims> partial_m_strides = m_strides;
    array<Index, NumDims> index_coords;

    if (static_cast<int>(Layout) == static_cast<int>(ColMajor)) {
      for (Index i = omitted_dim + 1; i < NumDims; ++i) {
        partial_m_strides[i] /= m_dimensions[omitted_dim];
      }
      for (int i = NumDims - 1; i > 0; --i) {
        if(omitted_dim == i) {
        }
        else {
          const Index idx = index / partial_m_strides[i];
          index -= idx * partial_m_strides[i];
          index_coords[i] = idx;
        }
      }
      index_coords[0] = index;
    }
    else {
      for (Index i = omitted_dim - 1; i >= 0; --i) {
        partial_m_strides[i] /= m_dimensions[omitted_dim];
      }
      for (int i = 0; i < NumDims - 1; ++i) {
        if(omitted_dim == i) {
        }
        else {
          const Index idx = index / partial_m_strides[i];
          index -= idx * partial_m_strides[i];
          index_coords[i] = idx;
        }
      }
      index_coords[NumDims - 1] = index;
    }
    // Value of index_coords[omitted_dim] is not determined to this step
    return index_coords;
  }

  void evalToSelfDataBuf() {
    cudaMalloc((void**) &m_data_buf, sizeof(OutputScalar) * m_size);
    evalToDeviceData(m_data_buf);
  }

  EIGEN_STRONG_INLINE void evalToDeviceData(OutputScalar* d_data) {
    ComplexScalar* data_buf;
    cudaMalloc((void**) &data_buf, sizeof(ComplexScalar) * m_size);

    int block_size = 128;
    int grid_size = m_size / block_size + 1;

    copyValues<InputScalar, RealScalar, ComplexScalar, TensorEvaluator<ArgType, GpuDevice> > <<<grid_size, block_size>>>(data_buf, m_impl, m_size);

    for (int i = 0; i < m_fft.size(); ++i) {
      int dim = m_fft[i];
      eigen_assert(dim >= 0 && dim < NumDims);
      int line_len = m_dimensions[dim];
      ComplexScalar* line_buf;
      cudaMalloc((void**) &line_buf, sizeof(ComplexScalar) * line_len);

      cufftHandle plan;
      cufftPlan1d(&plan, line_len, CUFFT_C2C, 1);

      for (Index partial_index = 0; partial_index < m_size/line_len; ++partial_index) {
        array<Index, NumDims> coords = getPartialCoordsFromIndex(partial_index, dim);
        // get data into line_buf
        int block_size = 128;
        int grid_size = line_len / block_size + 1;
        fillLineBuf<ComplexScalar, Index, NumDims> <<<grid_size, block_size>>>(line_buf, data_buf, line_len, coords, m_strides, dim);

        if(FFTDir == Eigen::FFT_FORWARD) {
          cufftExecC2C(plan, reinterpret_cast<cufftComplex *>(line_buf), reinterpret_cast<cufftComplex*>(line_buf), CUFFT_FORWARD);
        }
        else {
          cufftExecC2C(plan, reinterpret_cast<cufftComplex*>(line_buf), reinterpret_cast<cufftComplex*>(line_buf), CUFFT_INVERSE);
        }
        // write back
        RealScalar div_factor = (FFTDir == FFT_FORWARD) ? 1.0 : line_len;
        writebackLineBuf<ComplexScalar, RealScalar, Index, NumDims> <<<grid_size, block_size>>>(line_buf, data_buf, line_len, coords, m_strides, dim, div_factor);
        cudaDeviceSynchronize();

      }
      cufftDestroy(plan);
      cudaFree(line_buf);
    }
    writeToDeviceData<OutputScalar, RealScalar, ComplexScalar, FFTResultType>()(d_data, data_buf, m_size);
    cudaFree(data_buf);
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void cleanup() {
    if(m_data_buf != NULL) cudaFree(m_data_buf);
    m_impl.cleanup();
  }

  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE CoeffReturnType coeff(Index index) const {
    return m_data_buf[index];
  }

  template<int LoadMode>
  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE PacketReturnType packet(Index index) const {
    return internal::ploadt<PacketReturnType, LoadMode>(m_data_buf + index);
  }

  EIGEN_DEVICE_FUNC Scalar* data() const { return m_data_buf; }

 protected:
  int m_size;
  const FFT& m_fft;
  Dimensions m_dimensions;
  array<Index, NumDims> m_strides;
  TensorEvaluator<ArgType, GpuDevice> m_impl;
  OutputScalar* m_data_buf;

};
#endif

}  // end namespace Eigen
#endif //EIGEN_CXX11_TENSOR_TENSOR_FFT_H
