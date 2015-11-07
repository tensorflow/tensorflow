// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2014 Benoit Steiner <benoit.steiner.goog@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_CXX11_TENSOR_TENSOR_FUNCTORS_H
#define EIGEN_CXX11_TENSOR_TENSOR_FUNCTORS_H

namespace Eigen {
namespace internal {

namespace {
#if defined(EIGEN_USE_GPU) && defined(__CUDACC__) && defined(__CUDA_ARCH__)
__device__ int get_random_seed() {
    return clock();
}
#else
int get_random_seed() {
#ifdef _WIN32
    SYSTEMTIME st;
    GetSystemTime(&st);
    return st.wSecond + 1000 * st.wMilliseconds;
#elif __APPLE__
    return mach_absolute_time();
#else
    timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    return ts.tv_nsec;
#endif
}
#endif
}


// Standard reduction functors
template <typename T> struct SumReducer
{
  static const bool PacketAccess = true;
  static const bool IsStateful = false;

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void reduce(const T t, T* accum) const {
    (*accum) += t;
  }
  template <typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void reducePacket(const Packet& p, Packet* accum) const {
    (*accum) = padd<Packet>(*accum, p);
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE T initialize() const {
    return static_cast<T>(0);
  }
  template <typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet initializePacket() const {
    return pset1<Packet>(0);
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE T finalize(const T accum) const {
    return accum;
  }
  template <typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet finalizePacket(const Packet& vaccum) const {
    return vaccum;
  }
  template <typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE T finalizeBoth(const T saccum, const Packet& vaccum) const {
    return saccum + predux(vaccum);
  }
};

template <typename T> struct MeanReducer
{
  static const bool PacketAccess = true;
  static const bool IsStateful = true;

  MeanReducer() : scalarCount_(0), packetCount_(0) { }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void reduce(const T t, T* accum) {
    (*accum) += t;
    scalarCount_++;
  }
  template <typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void reducePacket(const Packet& p, Packet* accum) {
    (*accum) = padd<Packet>(*accum, p);
    packetCount_++;
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE T initialize() const {
    return static_cast<T>(0);
  }
  template <typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet initializePacket() const {
    return pset1<Packet>(0);
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE T finalize(const T accum) const {
    return accum / scalarCount_;
  }
  template <typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet finalizePacket(const Packet& vaccum) const {
    return pdiv(vaccum, pset1<Packet>(packetCount_));
  }
  template <typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE T finalizeBoth(const T saccum, const Packet& vaccum) const {
    return (saccum + predux(vaccum)) / (scalarCount_ + packetCount_ * unpacket_traits<Packet>::size);
  }

  protected:
    int scalarCount_;
    int packetCount_;
};

struct AndReducer
{
  static const bool PacketAccess = false;
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void reduce(bool t, bool* accum) const {
    *accum = *accum && t;
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE bool initialize() const {
    return true;
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE bool finalize(bool accum) const {
    return accum;
  }
};

struct OrReducer {
  static const bool PacketAccess = false;
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void reduce(bool t, bool* accum) const {
    *accum = *accum || t;
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE bool initialize() const {
    return false;
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE bool finalize(bool accum) const {
    return accum;
  }
};

template <typename T> struct MaxReducer
{
  static const bool PacketAccess = true;
  static const bool IsStateful = false;

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void reduce(const T t, T* accum) const {
    if (t > *accum) { *accum = t; }
  }
  template <typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void reducePacket(const Packet& p, Packet* accum) const {
    (*accum) = pmax<Packet>(*accum, p);
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE T initialize() const {
    return Eigen::NumTraits<T>::lowest();
  }
  template <typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet initializePacket() const {
    return pset1<Packet>(initialize());
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE T finalize(const T accum) const {
    return accum;
  }
  template <typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet finalizePacket(const Packet& vaccum) const {
    return vaccum;
  }
  template <typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE T finalizeBoth(const T saccum, const Packet& vaccum) const {
    return numext::maxi(saccum, predux_max(vaccum));
  }
};

template <typename T> struct MinReducer
{
  static const bool PacketAccess = true;
  static const bool IsStateful = false;

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void reduce(const T t, T* accum) const {
    if (t < *accum) { *accum = t; }
  }
  template <typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void reducePacket(const Packet& p, Packet* accum) const {
    (*accum) = pmin<Packet>(*accum, p);
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE T initialize() const {
    return Eigen::NumTraits<T>::highest();
  }
  template <typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet initializePacket() const {
    return pset1<Packet>(initialize());
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE T finalize(const T accum) const {
    return accum;
  }
  template <typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet finalizePacket(const Packet& vaccum) const {
    return vaccum;
  }
  template <typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE T finalizeBoth(const T saccum, const Packet& vaccum) const {
    return numext::mini(saccum, predux_min(vaccum));
  }
};


template <typename T> struct ProdReducer
{
  static const bool PacketAccess = true;
  static const bool IsStateful = false;

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void reduce(const T t, T* accum) const {
    (*accum) *= t;
  }
  template <typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void reducePacket(const Packet& p, Packet* accum) const {
    (*accum) = pmul<Packet>(*accum, p);
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE T initialize() const {
    return static_cast<T>(1);
  }
  template <typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet initializePacket() const {
    return pset1<Packet>(1);
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE T finalize(const T accum) const {
    return accum;
  }
  template <typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet finalizePacket(const Packet& vaccum) const {
    return vaccum;
  }
  template <typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE T finalizeBoth(const T saccum, const Packet& vaccum) const {
    return saccum * predux_mul(vaccum);
  }
};

#if !defined (EIGEN_USE_GPU) || !defined(__CUDACC__) || !defined(__CUDA_ARCH__)
// We're not compiling a cuda kernel
template <typename T> class UniformRandomGenerator {

 public:
  static const bool PacketAccess = true;

  // Uses the given "seed" if non-zero, otherwise uses a random seed.
  UniformRandomGenerator(unsigned int seed = 0) : m_seed(seed) {
    seed = seed ? seed : get_random_seed();
    srand(seed);
  }
  UniformRandomGenerator(const UniformRandomGenerator& other) {
    m_seed = other.m_seed;
  }

  template<typename Index>
  T operator()(Index, Index = 0) const {
    return random<T>();
  }
  template<typename Index>
  typename internal::packet_traits<T>::type packetOp(Index i, Index j = 0) const {
    const int packetSize = internal::packet_traits<T>::size;
    EIGEN_ALIGN_DEFAULT T values[packetSize];
    for (int i = 0; i < packetSize; ++i) {
      values[i] = random<T>();
    }
    return internal::pload<typename internal::packet_traits<T>::type>(values);
  }

 private:
  unsigned int m_seed;
};

#if __cplusplus > 199711
template <> class UniformRandomGenerator<float> {
 public:
  static const bool PacketAccess = true;

  // Uses the given "seed" if non-zero, otherwise uses a random seed.
  UniformRandomGenerator(unsigned int seed = 0) : m_seed(seed) {
    seed = seed ? seed : get_random_seed();
    m_generator.seed(seed);
  }
  UniformRandomGenerator(const UniformRandomGenerator<float>& other) {
    m_generator.seed(other(0, 0) * UINT_MAX);
    m_seed = other.m_seed;
  }

  template<typename Index>
  float operator()(Index, Index = 0) const {
    return m_distribution(m_generator);
  }
  template<typename Index>
  typename internal::packet_traits<float>::type packetOp(Index i, Index j = 0) const {
    const int packetSize = internal::packet_traits<float>::size;
    EIGEN_ALIGN_DEFAULT float values[packetSize];
    for (int i = 0; i < packetSize; ++i) {
      values[i] = this->operator()(i, j);
    }
    return internal::pload<typename internal::packet_traits<float>::type>(values);
  }

 private:
  UniformRandomGenerator& operator = (const UniformRandomGenerator&);
  // Make sure m_seed comes first to match the layout of the cpu
  // version of the code.
  unsigned int m_seed;
  mutable std::mt19937 m_generator;
  mutable std::uniform_real_distribution<float> m_distribution;
};

template <> class UniformRandomGenerator<double> {
 public:
  static const bool PacketAccess = true;

  // Uses the given "seed" if non-zero, otherwise uses a random seed.
  UniformRandomGenerator(unsigned int seed = 0) : m_seed(seed) {
    seed = seed ? seed : get_random_seed();
    m_generator.seed(seed);
  }
  UniformRandomGenerator(const UniformRandomGenerator<double>& other) {
    m_generator.seed(other(0, 0) * UINT_MAX);
    m_seed = other.m_seed;
  }

  template<typename Index>
  double operator()(Index, Index = 0) const {
    return m_distribution(m_generator);
  }
  template<typename Index>
  typename internal::packet_traits<double>::type packetOp(Index i, Index j = 0) const {
    const int packetSize = internal::packet_traits<double>::size;
    EIGEN_ALIGN_DEFAULT double values[packetSize];
    for (int i = 0; i < packetSize; ++i) {
      values[i] = this->operator()(i, j);
    }
    return internal::pload<typename internal::packet_traits<double>::type>(values);
  }

 private:
  UniformRandomGenerator& operator = (const UniformRandomGenerator&);
  // Make sure m_seed comes first to match the layout of the cpu
  // version of the code.
  unsigned int m_seed;
  mutable std::mt19937 m_generator;
  mutable std::uniform_real_distribution<double> m_distribution;
};
#endif

#else

// We're compiling a cuda kernel
template <typename T> class UniformRandomGenerator;

template <> class UniformRandomGenerator<float> {
 public:
  static const bool PacketAccess = true;

  // Uses the given "seed" if non-zero, otherwise uses a random seed.
  __device__ UniformRandomGenerator(unsigned int seed = 0) : m_seed(seed) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    seed = seed ? seed : get_random_seed();
    curand_init(seed, tid, 0, &m_state);
  }

  __device__ UniformRandomGenerator(const UniformRandomGenerator& other) {
    m_seed = other.m_seed;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int seed = m_seed ? m_seed : get_random_seed();
    curand_init(seed, tid, 0, &m_state);
  }

  template<typename Index>
  __device__ float operator()(Index, Index = 0) const {
    return curand_uniform(&m_state);
  }
  template<typename Index>
  __device__ float4 packetOp(Index, Index = 0) const {
    return curand_uniform4(&m_state);
  }

 private:
  unsigned int m_seed;
  mutable curandStatePhilox4_32_10_t m_state;
};

template <> class UniformRandomGenerator<double> {
 public:
  static const bool PacketAccess = true;

  // Uses the given "seed" if non-zero, otherwise uses a random seed.
  __device__ UniformRandomGenerator(unsigned int seed = 0) : m_seed(seed) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    seed = seed ? seed : get_random_seed();
    curand_init(seed, tid, 0, &m_state);
  }
  __device__ UniformRandomGenerator(const UniformRandomGenerator& other) {
    m_seed = other.m_seed;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int seed = m_seed ? m_seed : get_random_seed();
    curand_init(seed, tid, 0, &m_state);
  }
  template<typename Index>
  __device__ double operator()(Index, Index = 0) const {
    return curand_uniform_double(&m_state);
  }
  template<typename Index>
  __device__ double2 packetOp(Index, Index = 0) const {
    return curand_uniform2_double(&m_state);
  }

 private:
  unsigned int m_seed;
  mutable curandStatePhilox4_32_10_t m_state;
};

template <> class UniformRandomGenerator<std::complex<float> > {
 public:
  static const bool PacketAccess = false;

  // Uses the given "seed" if non-zero, otherwise uses a random seed.
  __device__ UniformRandomGenerator(unsigned int seed = 0) : m_seed(seed) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    seed = seed ? seed : get_random_seed();
    curand_init(seed, tid, 0, &m_state);
  }
  __device__ UniformRandomGenerator(const UniformRandomGenerator& other) {
    m_seed = other.m_seed;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int seed = m_seed ? m_seed : get_random_seed();
    curand_init(seed, tid, 0, &m_state);
  }
  template<typename Index>
  __device__ std::complex<float> operator()(Index, Index = 0) const {
    float4 vals = curand_uniform4(&m_state);
    return std::complex<float>(vals.x, vals.y);
  }

 private:
  unsigned int m_seed;
  mutable curandStatePhilox4_32_10_t m_state;
};

template <> class UniformRandomGenerator<std::complex<double> > {
 public:
  static const bool PacketAccess = false;

  // Uses the given "seed" if non-zero, otherwise uses a random seed.
  __device__ UniformRandomGenerator(unsigned int seed = 0) : m_seed(seed) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    seed = seed ? seed : get_random_seed();
    curand_init(seed, tid, 0, &m_state);
  }
  __device__ UniformRandomGenerator(const UniformRandomGenerator& other) {
    m_seed = other.m_seed;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int seed = m_seed ? m_seed : get_random_seed();
    curand_init(seed, tid, 0, &m_state);
  }
  template<typename Index>
  __device__ std::complex<double> operator()(Index, Index = 0) const {
    double2 vals = curand_uniform2_double(&m_state);
    return std::complex<double>(vals.x, vals.y);
  }

 private:
  unsigned int m_seed;
  mutable curandStatePhilox4_32_10_t m_state;
};

#endif


#if (!defined (EIGEN_USE_GPU) || !defined(__CUDACC__) || !defined(__CUDA_ARCH__)) && __cplusplus > 199711
// We're not compiling a cuda kernel
template <typename T> class NormalRandomGenerator {
 public:
  static const bool PacketAccess = true;

  // Uses the given "seed" if non-zero, otherwise uses a random seed.
  NormalRandomGenerator(unsigned int seed = 0) : m_distribution(0, 1), m_seed(seed) {
    seed = seed ? seed : get_random_seed();
    m_generator.seed(seed);
  }
  NormalRandomGenerator(const NormalRandomGenerator& other)
      : m_distribution(other.m_distribution), m_seed(other.m_seed) {
    m_generator.seed(other(0, 0) * UINT_MAX);
  }

  template<typename Index>
  T operator()(Index, Index = 0) const {
    return m_distribution(m_generator);
  }
  template<typename Index>
  typename internal::packet_traits<T>::type packetOp(Index, Index = 0) const {
    const int packetSize = internal::packet_traits<T>::size;
    EIGEN_ALIGN_DEFAULT T values[packetSize];
    for (int i = 0; i < packetSize; ++i) {
      values[i] = m_distribution(m_generator);
    }
    return internal::pload<typename internal::packet_traits<T>::type>(values);
  }

 private:
  unsigned int m_seed;
  mutable std::normal_distribution<T> m_distribution;
  mutable std::mt19937 m_generator;
};

#elif defined (EIGEN_USE_GPU) && defined(__CUDACC__) && defined(__CUDA_ARCH__)

// We're compiling a cuda kernel
template <typename T> class NormalRandomGenerator;

template <> class NormalRandomGenerator<float> {
 public:
  static const bool PacketAccess = true;

  // Uses the given "seed" if non-zero, otherwise uses a random seed.
  __device__ NormalRandomGenerator(unsigned int seed = 0) : m_seed(seed) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    seed = seed ? seed : get_random_seed();
    curand_init(seed, tid, 0, &m_state);
  }
  __device__ NormalRandomGenerator(const NormalRandomGenerator<float>& other) {
    m_seed = other.m_seed;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int seed = m_seed ? m_seed : get_random_seed();
    curand_init(seed, tid, 0, &m_state);
  }
  template<typename Index>
   __device__ float operator()(Index, Index = 0) const {
    return curand_normal(&m_state);
  }
  template<typename Index>
   __device__ float4 packetOp(Index, Index = 0) const {
    return curand_normal4(&m_state);
  }

 private:
  unsigned int m_seed;
  mutable curandStatePhilox4_32_10_t m_state;
};

template <> class NormalRandomGenerator<double> {
 public:
  static const bool PacketAccess = true;

  // Uses the given "seed" if non-zero, otherwise uses a random seed.
  __device__ NormalRandomGenerator(unsigned int seed = 0) : m_seed(seed) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    seed = seed ? seed : get_random_seed();
    curand_init(seed, tid, 0, &m_state);
  }
  __device__ NormalRandomGenerator(const NormalRandomGenerator<double>& other) {
    m_seed = other.m_seed;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int seed = m_seed ? m_seed : get_random_seed();
    curand_init(seed, tid, 0, &m_state);
  }
  template<typename Index>
  __device__ double operator()(Index, Index = 0) const {
    return curand_normal_double(&m_state);
  }
  template<typename Index>
  __device__ double2 packetOp(Index, Index = 0) const {
    return curand_normal2_double(&m_state);
  }

 private:
  unsigned int m_seed;
  mutable curandStatePhilox4_32_10_t m_state;
};


template <> class NormalRandomGenerator<std::complex<float> > {
 public:
  static const bool PacketAccess = false;

  // Uses the given "seed" if non-zero, otherwise uses a random seed.
  __device__ NormalRandomGenerator(unsigned int seed = 0) : m_seed(seed) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    seed = seed ? seed : get_random_seed();
    curand_init(seed, tid, 0, &m_state);
  }
  __device__ NormalRandomGenerator(const NormalRandomGenerator& other) {
    m_seed = other.m_seed;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int seed = m_seed ? m_seed : get_random_seed();
    curand_init(seed, tid, 0, &m_state);
  }
  template<typename Index>
  __device__ std::complex<float> operator()(Index, Index = 0) const {
    float4 vals = curand_normal4(&m_state);
    return std::complex<float>(vals.x, vals.y);
  }

 private:
  unsigned int m_seed;
  mutable curandStatePhilox4_32_10_t m_state;
};

template <> class NormalRandomGenerator<std::complex<double> > {
 public:
  static const bool PacketAccess = false;

  // Uses the given "seed" if non-zero, otherwise uses a random seed.
  __device__ NormalRandomGenerator(unsigned int seed = 0) : m_seed(seed) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    seed = seed ? seed : get_random_seed();
    curand_init(seed, tid, 0, &m_state);
  }
  __device__ NormalRandomGenerator(const NormalRandomGenerator& other) {
    m_seed = other.m_seed;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int seed = m_seed ? m_seed : get_random_seed();
    curand_init(seed, tid, 0, &m_state);
  }
  template<typename Index>
  __device__ std::complex<double> operator()(Index, Index = 0) const {
    double2 vals = curand_normal2_double(&m_state);
    return std::complex<double>(vals.x, vals.y);
  }

 private:
  unsigned int m_seed;
  mutable curandStatePhilox4_32_10_t m_state;
};
#else

template <typename T> class NormalRandomGenerator {
 public:
  // Uses the given "seed" if non-zero, otherwise uses a random seed.
  NormalRandomGenerator(unsigned int seed = 0) : m_seed(seed) {}

 private:
  unsigned int m_seed;
};

#endif


template <typename T, typename Index, size_t NumDims>
class GaussianGenerator {
 public:
  static const bool PacketAccess = false;

  EIGEN_DEVICE_FUNC GaussianGenerator(const array<T, NumDims>& means,
                                      const array<T, NumDims>& std_devs)
      : m_means(means) {
    for (int i = 0; i < NumDims; ++i) {
      m_two_sigmas[i] = std_devs[i] * std_devs[i] * 2;
    }
  }

  T operator()(const array<Index, NumDims>& coordinates) const {
    T tmp = T(0);
    for (int i = 0; i < NumDims; ++i) {
      T offset = coordinates[i] - m_means[i];
      tmp += offset * offset / m_two_sigmas[i];
    }
    return std::exp(-tmp);
  }

 private:
  array<T, NumDims> m_means;
  array<T, NumDims> m_two_sigmas;
};

template <typename T> struct ArgMaxTupleReducer
{
  static const bool PacketAccess = false;
  static const bool IsStateful = false;

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void reduce(const T t, T* accum) const {
    if (t.second > accum->second) { *accum = t; }
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE T initialize() const {
    return T(0, NumTraits<typename T::second_type>::lowest());
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE T finalize(const T& accum) const {
    return accum;
  }
};

template <typename T> struct ArgMinTupleReducer
{
  static const bool PacketAccess = false;
  static const bool IsStateful = false;

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void reduce(const T& t, T* accum) const {
    if (t.second < accum->second) { *accum = t; }
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE T initialize() const {
    return T(0, NumTraits<typename T::second_type>::highest());
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE T finalize(const T& accum) const {
    return accum;
  }
};

} // end namespace internal
} // end namespace Eigen

#endif // EIGEN_CXX11_TENSOR_TENSOR_FUNCTORS_H
