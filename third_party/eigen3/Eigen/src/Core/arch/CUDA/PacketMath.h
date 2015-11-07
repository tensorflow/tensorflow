// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2014 Benoit Steiner <benoit.steiner.goog@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_PACKET_MATH_CUDA_H
#define EIGEN_PACKET_MATH_CUDA_H

namespace Eigen {

namespace internal {
// Make sure this is only available when targeting a GPU: we don't want to
// introduce conflicts between these packet_traits definitions and the ones
// we'll use on the host side (SSE, AVX, ...)
#if defined(EIGEN_USE_GPU) && defined(__CUDACC__)
template<> struct is_arithmetic<float4>  { enum { value = true }; };
template<> struct is_arithmetic<double2> { enum { value = true }; };


template<> struct packet_traits<float> : default_packet_traits
{
  typedef float4 type;
  typedef float4 half;
  enum {
    Vectorizable = 1,
    AlignedOnScalar = 1,
    size=4,
    HasHalfPacket = 0,

    HasDiv  = 1,
    HasSin  = 0,
    HasCos  = 0,
    HasLog  = 1,
    HasExp  = 1,
    HasSqrt = 1,
    HasRsqrt = 1,

    HasBlend = 0,
    HasSelect = 1,
    HasEq = 1,
  };
};

template<> struct packet_traits<double> : default_packet_traits
{
  typedef double2 type;
  typedef double2 half;
  enum {
    Vectorizable = 1,
    AlignedOnScalar = 1,
    size=2,
    HasHalfPacket = 0,

    HasDiv  = 1,
    HasLog  = 1,
    HasExp  = 1,
    HasSqrt = 1,
    HasRsqrt = 1,

    HasBlend = 0,
    HasSelect = 1,
    HasEq = 1,
  };
};


template<> struct unpacket_traits<float4> { typedef float  type; enum {size=4}; typedef float4 half; };
template<> struct unpacket_traits<double2> { typedef double type; enum {size=2}; typedef double2 half; };

template<> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE float4 pset1<float4>(const float&  from) {
  return make_float4(from, from, from, from);
}
template<> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE double2 pset1<double2>(const double& from) {
  return make_double2(from, from);
}


template<> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE float4 plset<float>(const float& a) {
  return make_float4(a, a+1, a+2, a+3);
}
template<> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE double2 plset<double>(const double& a) {
  return make_double2(a, a+1);
}

template<> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE float4 padd<float4>(const float4& a, const float4& b) {
  return make_float4(a.x+b.x, a.y+b.y, a.z+b.z, a.w+b.w);
}
template<> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE double2 padd<double2>(const double2& a, const double2& b) {
  return make_double2(a.x+b.x, a.y+b.y);
}

template<> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE float4 psub<float4>(const float4& a, const float4& b) {
  return make_float4(a.x-b.x, a.y-b.y, a.z-b.z, a.w-b.w);
}
template<> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE double2 psub<double2>(const double2& a, const double2& b) {
  return make_double2(a.x-b.x, a.y-b.y);
}

template<> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE float4 peq<float4>(const float4& a, const float4& b) {
  return make_float4(a.x == b.x ? 1.f : 0, a.y == b.y ? 1.f : 0, a.z == b.z ? 1.f : 0, a.w == b.w ? 1.f : 0);
}
template<> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE double2 peq<double2>(const double2& a, const double2& b) {
  return make_double2(a.x == b.x ? 1. : 0, a.y == b.y ? 1. : 0);
}

template<> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE float4 ple<float4>(const float4& a, const float4& b) {
  return make_float4(a.x <= b.x ? 1.f : 0, a.y <= b.y ? 1.f : 0, a.z <= b.z ? 1.f : 0, a.w <= b.w ? 1.f : 0);
}
template<> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE double2 ple<double2>(const double2& a, const double2& b) {
  return make_double2(a.x <= b.x ? 1. : 0, a.y <= b.y ? 1. : 0);
}

template<> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE float4 plt<float4>(const float4& a, const float4& b) {
  return make_float4(a.x < b.x ? 1.f : 0, a.y < b.y ? 1.f : 0, a.z < b.z ? 1.f : 0, a.w < b.w ? 1.f : 0);
}
template<> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE double2 plt<double2>(const double2& a, const double2& b) {
  return make_double2(a.x < b.x ? 1. : 0, a.y < b.y ? 1. : 0);
}

template<> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE float4 pselect<float4>(const float4& a, const float4& b, const float4& c) {
  return make_float4(c.x ? b.x : a.x, c.y ? b.y : a.y, c.z ? b.z : a.z, c.w ? b.w : a.w);
}
template<> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE double2 pselect<double2>(const double2& a, const double2& b, const double2& c) {
  return make_double2(c.x ? b.x : a.x, c.y ? b.y : a.y);
}

template<> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE float4 pnegate(const float4& a) {
  return make_float4(-a.x, -a.y, -a.z, -a.w);
}
template<> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE double2 pnegate(const double2& a) {
  return make_double2(-a.x, -a.y);
}

template<> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE float4 pconj(const float4& a) { return a; }
template<> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE double2 pconj(const double2& a) { return a; }

template<> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE float4 pmul<float4>(const float4& a, const float4& b) {
  return make_float4(a.x*b.x, a.y*b.y, a.z*b.z, a.w*b.w);
}
template<> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE double2 pmul<double2>(const double2& a, const double2& b) {
  return make_double2(a.x*b.x, a.y*b.y);
}

template<> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE float4 pdiv<float4>(const float4& a, const float4& b) {
  return make_float4(a.x/b.x, a.y/b.y, a.z/b.z, a.w/b.w);
}
template<> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE double2 pdiv<double2>(const double2& a, const double2& b) {
  return make_double2(a.x/b.x, a.y/b.y);
}

template<> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE float4 pmin<float4>(const float4& a, const float4& b) {
  return make_float4(fminf(a.x, b.x), fminf(a.y, b.y), fminf(a.z, b.z), fminf(a.w, b.w));
}
template<> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE double2 pmin<double2>(const double2& a, const double2& b) {
  return make_double2(fmin(a.x, b.x), fmin(a.y, b.y));
}

template<> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE float4 pmax<float4>(const float4& a, const float4& b) {
  return make_float4(fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z), fmaxf(a.w, b.w));
}
template<> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE double2 pmax<double2>(const double2& a, const double2& b) {
  return make_double2(fmax(a.x, b.x), fmax(a.y, b.y));
}

template<> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE float4 pload<float4>(const float* from) {
  return *reinterpret_cast<const float4*>(from);
}

template<> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE double2 pload<double2>(const double* from) {
  return *reinterpret_cast<const double2*>(from);
}

template<> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE float4 ploadu<float4>(const float* from) {
  return make_float4(from[0], from[1], from[2], from[3]);
}
template<> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE double2 ploadu<double2>(const double* from) {
  return make_double2(from[0], from[1]);
}

template<> EIGEN_STRONG_INLINE float4 ploaddup<float4>(const float*   from) {
  return make_float4(from[0], from[0], from[1], from[1]);
}
template<> EIGEN_STRONG_INLINE double2 ploaddup<double2>(const double*  from) {
  return make_double2(from[0], from[0]);
}

template<> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void pstore<float>(float*   to, const float4& from) {
  *reinterpret_cast<float4*>(to) = from;
}

template<> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void pstore<double>(double* to, const double2& from) {
  *reinterpret_cast<double2*>(to) = from;
}

template<> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void pstoreu<float>(float*  to, const float4& from) {
  to[0] = from.x;
  to[1] = from.y;
  to[2] = from.z;
  to[3] = from.w;
}

template<> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void pstoreu<double>(double* to, const double2& from) {
  to[0] = from.x;
  to[1] = from.y;
}

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 350
template<>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE float4 ploadt_ro<float4, Aligned>(const float* from) {
  return __ldg((const float4*)from);
}
template<>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE double2 ploadt_ro<double2, Aligned>(const double* from) {
  return __ldg((const double2*)from);
}

template<>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE float4 ploadt_ro<float4, Unaligned>(const float* from) {
  return make_float4(__ldg(from+0), __ldg(from+1), __ldg(from+2), __ldg(from+3));
}
template<>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE double2 ploadt_ro<double2, Unaligned>(const double* from) {
  return make_double2(__ldg(from+0), __ldg(from+1));
}
#endif

template<> EIGEN_DEVICE_FUNC inline float4 pgather<float, float4>(const float* from, int stride) {
  return make_float4(from[0*stride], from[1*stride], from[2*stride], from[3*stride]);
}

template<> EIGEN_DEVICE_FUNC inline double2 pgather<double, double2>(const double* from, int stride) {
  return make_double2(from[0*stride], from[1*stride]);
}

template<> EIGEN_DEVICE_FUNC inline void pscatter<float, float4>(float* to, const float4& from, int stride) {
  to[stride*0] = from.x;
  to[stride*1] = from.y;
  to[stride*2] = from.z;
  to[stride*3] = from.w;
}
template<> EIGEN_DEVICE_FUNC inline void pscatter<double, double2>(double* to, const double2& from, int stride) {
  to[stride*0] = from.x;
  to[stride*1] = from.y;
}

template<> EIGEN_DEVICE_FUNC inline float  pfirst<float4>(const float4& a) {
  return a.x;
}
template<> EIGEN_DEVICE_FUNC inline double pfirst<double2>(const double2& a) {
  return a.x;
}

template<> EIGEN_DEVICE_FUNC inline float  predux<float4>(const float4& a) {
  return a.x + a.y + a.z + a.w;
}
template<> EIGEN_DEVICE_FUNC inline double predux<double2>(const double2& a) {
  return a.x + a.y;
}

template<> EIGEN_DEVICE_FUNC inline float  predux_max<float4>(const float4& a) {
  return fmaxf(fmaxf(a.x, a.y), fmaxf(a.z, a.w));
}
template<> EIGEN_DEVICE_FUNC inline double predux_max<double2>(const double2& a) {
  return fmax(a.x, a.y);
}

template<> EIGEN_DEVICE_FUNC inline float  predux_min<float4>(const float4& a) {
  return fminf(fminf(a.x, a.y), fminf(a.z, a.w));
}
template<> EIGEN_DEVICE_FUNC inline double predux_min<double2>(const double2& a) {
  return fmin(a.x, a.y);
}

template <>
EIGEN_DEVICE_FUNC inline float predux_mul<float4>(const float4& a) {
  return a.x * a.y * a.z * a.w;
}
template <>
EIGEN_DEVICE_FUNC inline double predux_mul<double2>(const double2& a) {
  return a.x * a.y;
}

template<> EIGEN_DEVICE_FUNC inline float4  pabs<float4>(const float4& a) {
  return make_float4(fabsf(a.x), fabsf(a.y), fabsf(a.z), fabsf(a.w));
}
template<> EIGEN_DEVICE_FUNC inline double2 pabs<double2>(const double2& a) {
  return make_double2(fabs(a.x), fabs(a.y));
}


template<> EIGEN_DEVICE_FUNC inline void
ptranspose(PacketBlock<float4,4>& kernel) {
  double tmp = kernel.packet[0].y;
  kernel.packet[0].y = kernel.packet[1].x;
  kernel.packet[1].x = tmp;

  tmp = kernel.packet[0].z;
  kernel.packet[0].z = kernel.packet[2].x;
  kernel.packet[2].x = tmp;

  tmp = kernel.packet[0].w;
  kernel.packet[0].w = kernel.packet[3].x;
  kernel.packet[3].x = tmp;

  tmp = kernel.packet[1].z;
  kernel.packet[1].z = kernel.packet[2].y;
  kernel.packet[2].y = tmp;

  tmp = kernel.packet[1].w;
  kernel.packet[1].w = kernel.packet[3].y;
  kernel.packet[3].y = tmp;

  tmp = kernel.packet[2].w;
  kernel.packet[2].w = kernel.packet[3].z;
  kernel.packet[3].z = tmp;
}

template<> EIGEN_DEVICE_FUNC inline void
ptranspose(PacketBlock<double2,2>& kernel) {
  double tmp = kernel.packet[0].y;
  kernel.packet[0].y = kernel.packet[1].x;
  kernel.packet[1].x = tmp;
}

#endif

} // end namespace internal

} // end namespace Eigen


#endif // EIGEN_PACKET_MATH_CUDA_H
