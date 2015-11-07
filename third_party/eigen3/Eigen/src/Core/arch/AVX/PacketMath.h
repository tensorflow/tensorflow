// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2014 Benoit Steiner (benoit.steiner.goog@gmail.com)
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_PACKET_MATH_AVX_H
#define EIGEN_PACKET_MATH_AVX_H

namespace Eigen {

namespace internal {

#ifndef EIGEN_CACHEFRIENDLY_PRODUCT_THRESHOLD
#define EIGEN_CACHEFRIENDLY_PRODUCT_THRESHOLD 8
#endif

#ifndef EIGEN_ARCH_DEFAULT_NUMBER_OF_REGISTERS
#define EIGEN_ARCH_DEFAULT_NUMBER_OF_REGISTERS (2*sizeof(void*))
#endif

#ifdef __FMA__
#ifndef EIGEN_HAS_SINGLE_INSTRUCTION_MADD
#define EIGEN_HAS_SINGLE_INSTRUCTION_MADD
#endif
#endif

typedef __m256  Packet8f;
typedef __m256i Packet8i;
typedef __m256d Packet4d;

template<> struct is_arithmetic<__m256>  { enum { value = true }; };
template<> struct is_arithmetic<__m256i> { enum { value = true }; };
template<> struct is_arithmetic<__m256d> { enum { value = true }; };

#define _EIGEN_DECLARE_CONST_Packet8f(NAME,X) \
  const Packet8f p8f_##NAME = pset1<Packet8f>(X)

#define _EIGEN_DECLARE_CONST_Packet8f_FROM_INT(NAME,X) \
  const Packet8f p8f_##NAME = (__m256)pset1<Packet8i>(X)

#define _EIGEN_DECLARE_CONST_Packet8i(NAME,X) \
  const Packet8i p8i_##NAME = pset1<Packet8i>(X)

#define _EIGEN_DECLARE_CONST_Packet4d(NAME,X) \
  const Packet4d p4d_##NAME = pset1<Packet4d>(X)


template<> struct packet_traits<float>  : default_packet_traits
{
  typedef Packet8f type;
  typedef Packet4f half;
  enum {
    Vectorizable = 1,
    AlignedOnScalar = 1,
    size=8,
    HasHalfPacket = 1,

    HasDiv  = 1,
    HasSin  = 1,
    HasCos  = 0,
    HasTanH = 1,
    HasBlend = 1,
    HasLog  = 1,
    HasExp  = 1,
    HasSqrt = 1,
    HasRsqrt = 1,
    HasSelect = 1,
    HasEq = 1,
  };
 };
template<> struct packet_traits<double> : default_packet_traits
{
  typedef Packet4d type;
  typedef Packet2d half;
  enum {
    Vectorizable = 1,
    AlignedOnScalar = 1,
    size = 4,
    HasHalfPacket = 1,

    HasDiv = 1,
    HasBlend = 1,
    HasExp = 1,
    HasSqrt = 1,
    HasRsqrt = 1,
    HasSelect = 1,
    HasEq = 1,
  };
};

/* Proper support for integers is only provided by AVX2. In the meantime, we'll
   use SSE instructions and packets to deal with integers.
template<> struct packet_traits<int>    : default_packet_traits
{
  typedef Packet8i type;
  enum {
    Vectorizable = 1,
    AlignedOnScalar = 1,
    size=8
  };
};
*/

template<> struct unpacket_traits<Packet8f> { typedef float  type; typedef Packet4f half; enum {size=8}; };
template<> struct unpacket_traits<Packet4d> { typedef double type; typedef Packet2d half; enum {size=4}; };
template<> struct unpacket_traits<Packet8i> { typedef int    type; typedef Packet4i half; enum {size=8}; };

template<> EIGEN_STRONG_INLINE Packet8f pset1<Packet8f>(const float&  from) { return _mm256_set1_ps(from); }
template<> EIGEN_STRONG_INLINE Packet4d pset1<Packet4d>(const double& from) { return _mm256_set1_pd(from); }
template<> EIGEN_STRONG_INLINE Packet8i pset1<Packet8i>(const int&    from) { return _mm256_set1_epi32(from); }

template<> EIGEN_STRONG_INLINE Packet8f pload1<Packet8f>(const float*  from) { return _mm256_broadcast_ss(from); }
template<> EIGEN_STRONG_INLINE Packet4d pload1<Packet4d>(const double* from) { return _mm256_broadcast_sd(from); }

template<> EIGEN_STRONG_INLINE Packet8f plset<float>(const float& a) { return _mm256_add_ps(_mm256_set1_ps(a), _mm256_set_ps(7.0,6.0,5.0,4.0,3.0,2.0,1.0,0.0)); }
template<> EIGEN_STRONG_INLINE Packet4d plset<double>(const double& a) { return _mm256_add_pd(_mm256_set1_pd(a), _mm256_set_pd(3.0,2.0,1.0,0.0)); }

template<> EIGEN_STRONG_INLINE Packet8f padd<Packet8f>(const Packet8f& a, const Packet8f& b) { return _mm256_add_ps(a,b); }
template<> EIGEN_STRONG_INLINE Packet4d padd<Packet4d>(const Packet4d& a, const Packet4d& b) { return _mm256_add_pd(a,b); }

template<> EIGEN_STRONG_INLINE Packet8f psub<Packet8f>(const Packet8f& a, const Packet8f& b) { return _mm256_sub_ps(a,b); }
template<> EIGEN_STRONG_INLINE Packet4d psub<Packet4d>(const Packet4d& a, const Packet4d& b) { return _mm256_sub_pd(a,b); }

template<> EIGEN_STRONG_INLINE Packet8f ple<Packet8f>(const Packet8f& a, const Packet8f& b) { return _mm256_cmp_ps(a,b,_CMP_NGT_UQ); }
template<> EIGEN_STRONG_INLINE Packet4d ple<Packet4d>(const Packet4d& a, const Packet4d& b) { return _mm256_cmp_pd(a,b,_CMP_NGT_UQ); }

template<> EIGEN_STRONG_INLINE Packet8f plt<Packet8f>(const Packet8f& a, const Packet8f& b) { return _mm256_cmp_ps(a,b,_CMP_NGE_UQ); }
template<> EIGEN_STRONG_INLINE Packet4d plt<Packet4d>(const Packet4d& a, const Packet4d& b) { return _mm256_cmp_pd(a,b,_CMP_NGE_UQ); }

template<> EIGEN_STRONG_INLINE Packet8f peq<Packet8f>(const Packet8f& a, const Packet8f& b) { return _mm256_cmp_ps(a,b,_CMP_EQ_UQ); }
template<> EIGEN_STRONG_INLINE Packet4d peq<Packet4d>(const Packet4d& a, const Packet4d& b) { return _mm256_cmp_pd(a,b,_CMP_EQ_UQ); }

template<> EIGEN_STRONG_INLINE Packet8f pselect<Packet8f>(const Packet8f& a, const Packet8f& b, const Packet8f& false_mask) { return _mm256_blendv_ps(a,b,false_mask); }
template<> EIGEN_STRONG_INLINE Packet4d pselect<Packet4d>(const Packet4d& a, const Packet4d& b, const Packet4d& false_mask) { return _mm256_blendv_pd(a,b,false_mask); }

template<> EIGEN_STRONG_INLINE Packet8f pnegate(const Packet8f& a)
{
  return _mm256_sub_ps(_mm256_set1_ps(0.0),a);
}
template<> EIGEN_STRONG_INLINE Packet4d pnegate(const Packet4d& a)
{
  return _mm256_sub_pd(_mm256_set1_pd(0.0),a);
}

template<> EIGEN_STRONG_INLINE Packet8f pconj(const Packet8f& a) { return a; }
template<> EIGEN_STRONG_INLINE Packet4d pconj(const Packet4d& a) { return a; }
template<> EIGEN_STRONG_INLINE Packet8i pconj(const Packet8i& a) { return a; }

template<> EIGEN_STRONG_INLINE Packet8f pmul<Packet8f>(const Packet8f& a, const Packet8f& b) { return _mm256_mul_ps(a,b); }
template<> EIGEN_STRONG_INLINE Packet4d pmul<Packet4d>(const Packet4d& a, const Packet4d& b) { return _mm256_mul_pd(a,b); }

#ifdef __FMA__
template<> EIGEN_STRONG_INLINE Packet8f pmadd(const Packet8f& a, const Packet8f& b, const Packet8f& c) {
#if EIGEN_COMP_GNUC || EIGEN_COMP_CLANG
  // clang stupidly generates a vfmadd213ps instruction plus some vmovaps on registers,
  // and gcc stupidly generates a vfmadd132ps instruction,
  // so let's enforce it to generate a vfmadd231ps instruction since the most common use case is to accumulate
  // the result of the product.
  Packet8f res = c;
  asm("vfmadd231ps %[a], %[b], %[c]" : [c] "+x" (res) : [a] "x" (a), [b] "x" (b));
  return res;
#else
  return _mm256_fmadd_ps(a,b,c);
#endif
}
template<> EIGEN_STRONG_INLINE Packet4d pmadd(const Packet4d& a, const Packet4d& b, const Packet4d& c) {
#if EIGEN_COMP_GNUC || EIGEN_COMP_CLANG
  // see above
  Packet4d res = c;
  asm("vfmadd231pd %[a], %[b], %[c]" : [c] "+x" (res) : [a] "x" (a), [b] "x" (b));
  return res;
#else
  return _mm256_fmadd_pd(a,b,c);
#endif
}
#endif

template<> EIGEN_STRONG_INLINE Packet8f pmin<Packet8f>(const Packet8f& a, const Packet8f& b) { return _mm256_min_ps(a,b); }
template<> EIGEN_STRONG_INLINE Packet4d pmin<Packet4d>(const Packet4d& a, const Packet4d& b) { return _mm256_min_pd(a,b); }

template<> EIGEN_STRONG_INLINE Packet8f pmax<Packet8f>(const Packet8f& a, const Packet8f& b) { return _mm256_max_ps(a,b); }
template<> EIGEN_STRONG_INLINE Packet4d pmax<Packet4d>(const Packet4d& a, const Packet4d& b) { return _mm256_max_pd(a,b); }

template<> EIGEN_STRONG_INLINE Packet8f pand<Packet8f>(const Packet8f& a, const Packet8f& b) { return _mm256_and_ps(a,b); }
template<> EIGEN_STRONG_INLINE Packet4d pand<Packet4d>(const Packet4d& a, const Packet4d& b) { return _mm256_and_pd(a,b); }

template<> EIGEN_STRONG_INLINE Packet8f por<Packet8f>(const Packet8f& a, const Packet8f& b) { return _mm256_or_ps(a,b); }
template<> EIGEN_STRONG_INLINE Packet4d por<Packet4d>(const Packet4d& a, const Packet4d& b) { return _mm256_or_pd(a,b); }

template<> EIGEN_STRONG_INLINE Packet8f pxor<Packet8f>(const Packet8f& a, const Packet8f& b) { return _mm256_xor_ps(a,b); }
template<> EIGEN_STRONG_INLINE Packet4d pxor<Packet4d>(const Packet4d& a, const Packet4d& b) { return _mm256_xor_pd(a,b); }

template<> EIGEN_STRONG_INLINE Packet8f pandnot<Packet8f>(const Packet8f& a, const Packet8f& b) { return _mm256_andnot_ps(a,b); }
template<> EIGEN_STRONG_INLINE Packet4d pandnot<Packet4d>(const Packet4d& a, const Packet4d& b) { return _mm256_andnot_pd(a,b); }

template<> EIGEN_STRONG_INLINE Packet8f pload<Packet8f>(const float*   from) { EIGEN_DEBUG_ALIGNED_LOAD return _mm256_load_ps(from); }
template<> EIGEN_STRONG_INLINE Packet4d pload<Packet4d>(const double*  from) { EIGEN_DEBUG_ALIGNED_LOAD return _mm256_load_pd(from); }
template<> EIGEN_STRONG_INLINE Packet8i pload<Packet8i>(const int*     from) { EIGEN_DEBUG_ALIGNED_LOAD return _mm256_load_si256(reinterpret_cast<const __m256i*>(from)); }

template<> EIGEN_STRONG_INLINE Packet8f ploadu<Packet8f>(const float* from) { EIGEN_DEBUG_UNALIGNED_LOAD return _mm256_loadu_ps(from); }
template<> EIGEN_STRONG_INLINE Packet4d ploadu<Packet4d>(const double* from) { EIGEN_DEBUG_UNALIGNED_LOAD return _mm256_loadu_pd(from); }
template<> EIGEN_STRONG_INLINE Packet8i ploadu<Packet8i>(const int* from) { EIGEN_DEBUG_UNALIGNED_LOAD return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(from)); }

// Loads 4 floats from memory a returns the packet {a0, a0  a1, a1, a2, a2, a3, a3}
template<> EIGEN_STRONG_INLINE Packet8f ploaddup<Packet8f>(const float* from)
{
  // TODO try to find a way to avoid the need of a temporary register
//   Packet8f tmp  = _mm256_castps128_ps256(_mm_loadu_ps(from));
//   tmp = _mm256_insertf128_ps(tmp, _mm_movehl_ps(_mm256_castps256_ps128(tmp),_mm256_castps256_ps128(tmp)), 1);
//   return _mm256_unpacklo_ps(tmp,tmp);

  // _mm256_insertf128_ps is very slow on Haswell, thus:
  Packet8f tmp = _mm256_broadcast_ps((const __m128*)(const void*)from);
  // mimic an "inplace" permutation of the lower 128bits using a blend
  tmp = _mm256_blend_ps(tmp,_mm256_castps128_ps256(_mm_permute_ps( _mm256_castps256_ps128(tmp), _MM_SHUFFLE(1,0,1,0))), 15);
  // then we can perform a consistent permutation on the global register to get everything in shape:
  return  _mm256_permute_ps(tmp, _MM_SHUFFLE(3,3,2,2));
}
// Loads 2 doubles from memory a returns the packet {a0, a0  a1, a1}
template<> EIGEN_STRONG_INLINE Packet4d ploaddup<Packet4d>(const double* from)
{
  Packet4d tmp = _mm256_broadcast_pd((const __m128d*)(const void*)from);
  return  _mm256_permute_pd(tmp, 3<<2);
}

// Loads 2 floats from memory a returns the packet {a0, a0  a0, a0, a1, a1, a1, a1}
template<> EIGEN_STRONG_INLINE Packet8f ploadquad<Packet8f>(const float* from)
{
  Packet8f tmp = _mm256_castps128_ps256(_mm_broadcast_ss(from));
  return _mm256_insertf128_ps(tmp, _mm_broadcast_ss(from+1), 1);
}

template<> EIGEN_STRONG_INLINE void pstore<float>(float*   to, const Packet8f& from) { EIGEN_DEBUG_ALIGNED_STORE _mm256_store_ps(to, from); }
template<> EIGEN_STRONG_INLINE void pstore<double>(double* to, const Packet4d& from) { EIGEN_DEBUG_ALIGNED_STORE _mm256_store_pd(to, from); }
template<> EIGEN_STRONG_INLINE void pstore<int>(int*       to, const Packet8i& from) { EIGEN_DEBUG_ALIGNED_STORE _mm256_storeu_si256(reinterpret_cast<__m256i*>(to), from); }

template<> EIGEN_STRONG_INLINE void pstoreu<float>(float*   to, const Packet8f& from) { EIGEN_DEBUG_UNALIGNED_STORE _mm256_storeu_ps(to, from); }
template<> EIGEN_STRONG_INLINE void pstoreu<double>(double* to, const Packet4d& from) { EIGEN_DEBUG_UNALIGNED_STORE _mm256_storeu_pd(to, from); }
template<> EIGEN_STRONG_INLINE void pstoreu<int>(int*       to, const Packet8i& from) { EIGEN_DEBUG_UNALIGNED_STORE _mm256_storeu_si256(reinterpret_cast<__m256i*>(to), from); }

// NOTE: leverage _mm256_i32gather_ps and _mm256_i32gather_pd if AVX2 instructions are available
template<> EIGEN_DEVICE_FUNC inline Packet8f pgather<float, Packet8f>(const float* from, int stride)
{
#ifdef EIGEN_VECTORIZE_AVX2
  return _mm256_i32gather_ps(from, _mm256_set1_epi32(stride), 4);
#else
  return _mm256_set_ps(from[7*stride], from[6*stride], from[5*stride], from[4*stride],
                       from[3*stride], from[2*stride], from[1*stride], from[0*stride]);
#endif
}
template<> EIGEN_DEVICE_FUNC inline Packet4d pgather<double, Packet4d>(const double* from, int stride)
{
#ifdef EIGEN_VECTORIZE_AVX2
  return _mm256_i32gather_pd(from, _mm_set1_epi32(stride), 8);
#else
  return _mm256_set_pd(from[3*stride], from[2*stride], from[1*stride], from[0*stride]);
#endif
}

template<> EIGEN_DEVICE_FUNC inline void pscatter<float, Packet8f>(float* to, const Packet8f& from, int stride)
{
  __m128 low = _mm256_extractf128_ps(from, 0);
  to[stride*0] = _mm_cvtss_f32(low);
  to[stride*1] = _mm_cvtss_f32(_mm_shuffle_ps(low, low, 1));
  to[stride*2] = _mm_cvtss_f32(_mm_shuffle_ps(low, low, 2));
  to[stride*3] = _mm_cvtss_f32(_mm_shuffle_ps(low, low, 3));

  __m128 high = _mm256_extractf128_ps(from, 1);
  to[stride*4] = _mm_cvtss_f32(high);
  to[stride*5] = _mm_cvtss_f32(_mm_shuffle_ps(high, high, 1));
  to[stride*6] = _mm_cvtss_f32(_mm_shuffle_ps(high, high, 2));
  to[stride*7] = _mm_cvtss_f32(_mm_shuffle_ps(high, high, 3));
}
template<> EIGEN_DEVICE_FUNC inline void pscatter<double, Packet4d>(double* to, const Packet4d& from, int stride)
{
  __m128d low = _mm256_extractf128_pd(from, 0);
  to[stride*0] = _mm_cvtsd_f64(low);
  to[stride*1] = _mm_cvtsd_f64(_mm_shuffle_pd(low, low, 1));
  __m128d high = _mm256_extractf128_pd(from, 1);
  to[stride*2] = _mm_cvtsd_f64(high);
  to[stride*3] = _mm_cvtsd_f64(_mm_shuffle_pd(high, high, 1));
}

template<> EIGEN_STRONG_INLINE void pstore1<Packet8f>(float* to, const float& a)
{
  Packet8f pa = pset1<Packet8f>(a);
  pstore(to, pa);
}
template<> EIGEN_STRONG_INLINE void pstore1<Packet4d>(double* to, const double& a)
{
  Packet4d pa = pset1<Packet4d>(a);
  pstore(to, pa);
}
template<> EIGEN_STRONG_INLINE void pstore1<Packet8i>(int* to, const int& a)
{
  Packet8i pa = pset1<Packet8i>(a);
  pstore(to, pa);
}

template<> EIGEN_STRONG_INLINE void prefetch<float>(const float*   addr) { _mm_prefetch((const char*)(addr), _MM_HINT_T0); }
template<> EIGEN_STRONG_INLINE void prefetch<double>(const double* addr) { _mm_prefetch((const char*)(addr), _MM_HINT_T0); }
template<> EIGEN_STRONG_INLINE void prefetch<int>(const int*       addr) { _mm_prefetch((const char*)(addr), _MM_HINT_T0); }

template<> EIGEN_STRONG_INLINE float  pfirst<Packet8f>(const Packet8f& a) {
  return _mm_cvtss_f32(_mm256_castps256_ps128(a));
}
template<> EIGEN_STRONG_INLINE double pfirst<Packet4d>(const Packet4d& a) {
  return _mm_cvtsd_f64(_mm256_castpd256_pd128(a));
}
template<> EIGEN_STRONG_INLINE int    pfirst<Packet8i>(const Packet8i& a) {
  return _mm_cvtsi128_si32(_mm256_castsi256_si128(a));
}


template<> EIGEN_STRONG_INLINE Packet8f preverse(const Packet8f& a)
{
  __m256 tmp = _mm256_shuffle_ps(a,a,0x1b);
  return _mm256_permute2f128_ps(tmp, tmp, 1);
}
template<> EIGEN_STRONG_INLINE Packet4d preverse(const Packet4d& a)
{
   __m256d tmp = _mm256_shuffle_pd(a,a,5);
  return _mm256_permute2f128_pd(tmp, tmp, 1);

  __m256d swap_halves = _mm256_permute2f128_pd(a,a,1);
    return _mm256_permute_pd(swap_halves,5);
}

// pabs should be ok
template<> EIGEN_STRONG_INLINE Packet8f pabs(const Packet8f& a)
{
  const Packet8f mask = _mm256_castsi256_ps(_mm256_setr_epi32(0x7FFFFFFF,0x7FFFFFFF,0x7FFFFFFF,0x7FFFFFFF,0x7FFFFFFF,0x7FFFFFFF,0x7FFFFFFF,0x7FFFFFFF));
  return _mm256_and_ps(a,mask);
}
template<> EIGEN_STRONG_INLINE Packet4d pabs(const Packet4d& a)
{
  const Packet4d mask = _mm256_castsi256_pd(_mm256_setr_epi32(0xFFFFFFFF,0x7FFFFFFF,0xFFFFFFFF,0x7FFFFFFF,0xFFFFFFFF,0x7FFFFFFF,0xFFFFFFFF,0x7FFFFFFF));
  return _mm256_and_pd(a,mask);
}

// preduxp should be ok
// FIXME: why is this ok? why isn't the simply implementation working as expected?
template<> EIGEN_STRONG_INLINE Packet8f preduxp<Packet8f>(const Packet8f* vecs)
{
    __m256 hsum1 = _mm256_hadd_ps(vecs[0], vecs[1]);
    __m256 hsum2 = _mm256_hadd_ps(vecs[2], vecs[3]);
    __m256 hsum3 = _mm256_hadd_ps(vecs[4], vecs[5]);
    __m256 hsum4 = _mm256_hadd_ps(vecs[6], vecs[7]);

    __m256 hsum5 = _mm256_hadd_ps(hsum1, hsum1);
    __m256 hsum6 = _mm256_hadd_ps(hsum2, hsum2);
    __m256 hsum7 = _mm256_hadd_ps(hsum3, hsum3);
    __m256 hsum8 = _mm256_hadd_ps(hsum4, hsum4);

    __m256 perm1 =  _mm256_permute2f128_ps(hsum5, hsum5, 0x23);
    __m256 perm2 =  _mm256_permute2f128_ps(hsum6, hsum6, 0x23);
    __m256 perm3 =  _mm256_permute2f128_ps(hsum7, hsum7, 0x23);
    __m256 perm4 =  _mm256_permute2f128_ps(hsum8, hsum8, 0x23);

    __m256 sum1 = _mm256_add_ps(perm1, hsum5);
    __m256 sum2 = _mm256_add_ps(perm2, hsum6);
    __m256 sum3 = _mm256_add_ps(perm3, hsum7);
    __m256 sum4 = _mm256_add_ps(perm4, hsum8);

    __m256 blend1 = _mm256_blend_ps(sum1, sum2, 0xcc);
    __m256 blend2 = _mm256_blend_ps(sum3, sum4, 0xcc);

    __m256 final = _mm256_blend_ps(blend1, blend2, 0xf0);
    return final;
}
template<> EIGEN_STRONG_INLINE Packet4d preduxp<Packet4d>(const Packet4d* vecs)
{
 Packet4d tmp0, tmp1;

  tmp0 = _mm256_hadd_pd(vecs[0], vecs[1]);
  tmp0 = _mm256_add_pd(tmp0, _mm256_permute2f128_pd(tmp0, tmp0, 1));

  tmp1 = _mm256_hadd_pd(vecs[2], vecs[3]);
  tmp1 = _mm256_add_pd(tmp1, _mm256_permute2f128_pd(tmp1, tmp1, 1));

  return _mm256_blend_pd(tmp0, tmp1, 0xC);
}

template<> EIGEN_STRONG_INLINE float predux<Packet8f>(const Packet8f& a)
{
  Packet8f tmp0 = _mm256_hadd_ps(a,_mm256_permute2f128_ps(a,a,1));
  tmp0 = _mm256_hadd_ps(tmp0,tmp0);
  return pfirst(_mm256_hadd_ps(tmp0, tmp0));
}
template<> EIGEN_STRONG_INLINE double predux<Packet4d>(const Packet4d& a)
{
  Packet4d tmp0 = _mm256_hadd_pd(a,_mm256_permute2f128_pd(a,a,1));
  return pfirst(_mm256_hadd_pd(tmp0,tmp0));
}

template<> EIGEN_STRONG_INLINE Packet4f predux4<Packet8f>(const Packet8f& a)
{
  return _mm_add_ps(_mm256_castps256_ps128(a),_mm256_extractf128_ps(a,1));
}

template<> EIGEN_STRONG_INLINE float predux_mul<Packet8f>(const Packet8f& a)
{
  Packet8f tmp;
  tmp = _mm256_mul_ps(a, _mm256_permute2f128_ps(a,a,1));
  tmp = _mm256_mul_ps(tmp, _mm256_shuffle_ps(tmp,tmp,_MM_SHUFFLE(1,0,3,2)));
  return pfirst(_mm256_mul_ps(tmp, _mm256_shuffle_ps(tmp,tmp,1)));
}
template<> EIGEN_STRONG_INLINE double predux_mul<Packet4d>(const Packet4d& a)
{
  Packet4d tmp;
  tmp = _mm256_mul_pd(a, _mm256_permute2f128_pd(a,a,1));
  return pfirst(_mm256_mul_pd(tmp, _mm256_shuffle_pd(tmp,tmp,1)));
}

template<> EIGEN_STRONG_INLINE float predux_min<Packet8f>(const Packet8f& a)
{
  Packet8f tmp = _mm256_min_ps(a, _mm256_permute2f128_ps(a,a,1));
  tmp = _mm256_min_ps(tmp, _mm256_shuffle_ps(tmp,tmp,_MM_SHUFFLE(1,0,3,2)));
  return pfirst(_mm256_min_ps(tmp, _mm256_shuffle_ps(tmp,tmp,1)));
}
template<> EIGEN_STRONG_INLINE double predux_min<Packet4d>(const Packet4d& a)
{
  Packet4d tmp = _mm256_min_pd(a, _mm256_permute2f128_pd(a,a,1));
  return pfirst(_mm256_min_pd(tmp, _mm256_shuffle_pd(tmp, tmp, 1)));
}

template<> EIGEN_STRONG_INLINE float predux_max<Packet8f>(const Packet8f& a)
{
  Packet8f tmp = _mm256_max_ps(a, _mm256_permute2f128_ps(a,a,1));
  tmp = _mm256_max_ps(tmp, _mm256_shuffle_ps(tmp,tmp,_MM_SHUFFLE(1,0,3,2)));
  return pfirst(_mm256_max_ps(tmp, _mm256_shuffle_ps(tmp,tmp,1)));
}

template<> EIGEN_STRONG_INLINE double predux_max<Packet4d>(const Packet4d& a)
{
  Packet4d tmp = _mm256_max_pd(a, _mm256_permute2f128_pd(a,a,1));
  return pfirst(_mm256_max_pd(tmp, _mm256_shuffle_pd(tmp, tmp, 1)));
}


template<int Offset>
struct palign_impl<Offset,Packet8f>
{
  static EIGEN_STRONG_INLINE void run(Packet8f& first, const Packet8f& second)
  {
    if (Offset==1)
    {
      first = _mm256_blend_ps(first, second, 1);
      Packet8f tmp = _mm256_permute_ps (first, _MM_SHUFFLE(0,3,2,1));
      first = _mm256_blend_ps(tmp, _mm256_permute2f128_ps (tmp, tmp, 1), 0x88);
    }
    else if (Offset==2)
    {
      first = _mm256_blend_ps(first, second, 3);
      Packet8f tmp = _mm256_permute_ps (first, _MM_SHUFFLE(1,0,3,2));
      first = _mm256_blend_ps(tmp, _mm256_permute2f128_ps (tmp, tmp, 1), 0xcc);
    }
    else if (Offset==3)
    {
      first = _mm256_blend_ps(first, second, 7);
      Packet8f tmp = _mm256_permute_ps (first, _MM_SHUFFLE(2,1,0,3));
      first = _mm256_blend_ps(tmp, _mm256_permute2f128_ps (tmp, tmp, 1), 0xee);
    }
    else if (Offset==4)
    {
      first = _mm256_blend_ps(first, second, 15);
      Packet8f tmp = _mm256_permute_ps (first, _MM_SHUFFLE(3,2,1,0));
      first = _mm256_permute_ps(_mm256_permute2f128_ps (tmp, tmp, 1), _MM_SHUFFLE(3,2,1,0));
    }
    else if (Offset==5)
    {
      first = _mm256_blend_ps(first, second, 31);
      first = _mm256_permute2f128_ps(first, first, 1);
      Packet8f tmp = _mm256_permute_ps (first, _MM_SHUFFLE(0,3,2,1));
      first = _mm256_permute2f128_ps(tmp, tmp, 1);
      first = _mm256_blend_ps(tmp, first, 0x88);
    }
    else if (Offset==6)
    {
      first = _mm256_blend_ps(first, second, 63);
      first = _mm256_permute2f128_ps(first, first, 1);
      Packet8f tmp = _mm256_permute_ps (first, _MM_SHUFFLE(1,0,3,2));
      first = _mm256_permute2f128_ps(tmp, tmp, 1);
      first = _mm256_blend_ps(tmp, first, 0xcc);
    }
    else if (Offset==7)
    {
      first = _mm256_blend_ps(first, second, 127);
      first = _mm256_permute2f128_ps(first, first, 1);
      Packet8f tmp = _mm256_permute_ps (first, _MM_SHUFFLE(2,1,0,3));
      first = _mm256_permute2f128_ps(tmp, tmp, 1);
      first = _mm256_blend_ps(tmp, first, 0xee);
    }
  }
};

template<int Offset>
struct palign_impl<Offset,Packet4d>
{
  static EIGEN_STRONG_INLINE void run(Packet4d& first, const Packet4d& second)
  {
    if (Offset==1)
    {
      first = _mm256_blend_pd(first, second, 1);
      __m256d tmp = _mm256_permute_pd(first, 5);
      first = _mm256_permute2f128_pd(tmp, tmp, 1);
      first = _mm256_blend_pd(tmp, first, 0xA);
    }
    else if (Offset==2)
    {
      first = _mm256_blend_pd(first, second, 3);
      first = _mm256_permute2f128_pd(first, first, 1);
    }
    else if (Offset==3)
    {
      first = _mm256_blend_pd(first, second, 7);
      __m256d tmp = _mm256_permute_pd(first, 5);
      first = _mm256_permute2f128_pd(tmp, tmp, 1);
      first = _mm256_blend_pd(tmp, first, 5);
    }
  }
};

template<> EIGEN_DEVICE_FUNC inline void
ptranspose(PacketBlock<Packet8f,8>& kernel) {
  __m256 T0 = _mm256_unpacklo_ps(kernel.packet[0], kernel.packet[1]);
  __m256 T1 = _mm256_unpackhi_ps(kernel.packet[0], kernel.packet[1]);
  __m256 T2 = _mm256_unpacklo_ps(kernel.packet[2], kernel.packet[3]);
  __m256 T3 = _mm256_unpackhi_ps(kernel.packet[2], kernel.packet[3]);
  __m256 T4 = _mm256_unpacklo_ps(kernel.packet[4], kernel.packet[5]);
  __m256 T5 = _mm256_unpackhi_ps(kernel.packet[4], kernel.packet[5]);
  __m256 T6 = _mm256_unpacklo_ps(kernel.packet[6], kernel.packet[7]);
  __m256 T7 = _mm256_unpackhi_ps(kernel.packet[6], kernel.packet[7]);
  __m256 S0 = _mm256_shuffle_ps(T0,T2,_MM_SHUFFLE(1,0,1,0));
  __m256 S1 = _mm256_shuffle_ps(T0,T2,_MM_SHUFFLE(3,2,3,2));
  __m256 S2 = _mm256_shuffle_ps(T1,T3,_MM_SHUFFLE(1,0,1,0));
  __m256 S3 = _mm256_shuffle_ps(T1,T3,_MM_SHUFFLE(3,2,3,2));
  __m256 S4 = _mm256_shuffle_ps(T4,T6,_MM_SHUFFLE(1,0,1,0));
  __m256 S5 = _mm256_shuffle_ps(T4,T6,_MM_SHUFFLE(3,2,3,2));
  __m256 S6 = _mm256_shuffle_ps(T5,T7,_MM_SHUFFLE(1,0,1,0));
  __m256 S7 = _mm256_shuffle_ps(T5,T7,_MM_SHUFFLE(3,2,3,2));
  kernel.packet[0] = _mm256_permute2f128_ps(S0, S4, 0x20);
  kernel.packet[1] = _mm256_permute2f128_ps(S1, S5, 0x20);
  kernel.packet[2] = _mm256_permute2f128_ps(S2, S6, 0x20);
  kernel.packet[3] = _mm256_permute2f128_ps(S3, S7, 0x20);
  kernel.packet[4] = _mm256_permute2f128_ps(S0, S4, 0x31);
  kernel.packet[5] = _mm256_permute2f128_ps(S1, S5, 0x31);
  kernel.packet[6] = _mm256_permute2f128_ps(S2, S6, 0x31);
  kernel.packet[7] = _mm256_permute2f128_ps(S3, S7, 0x31);
}

template<> EIGEN_DEVICE_FUNC inline void
ptranspose(PacketBlock<Packet8f,4>& kernel) {
  __m256 T0 = _mm256_unpacklo_ps(kernel.packet[0], kernel.packet[1]);
  __m256 T1 = _mm256_unpackhi_ps(kernel.packet[0], kernel.packet[1]);
  __m256 T2 = _mm256_unpacklo_ps(kernel.packet[2], kernel.packet[3]);
  __m256 T3 = _mm256_unpackhi_ps(kernel.packet[2], kernel.packet[3]);

  __m256 S0 = _mm256_shuffle_ps(T0,T2,_MM_SHUFFLE(1,0,1,0));
  __m256 S1 = _mm256_shuffle_ps(T0,T2,_MM_SHUFFLE(3,2,3,2));
  __m256 S2 = _mm256_shuffle_ps(T1,T3,_MM_SHUFFLE(1,0,1,0));
  __m256 S3 = _mm256_shuffle_ps(T1,T3,_MM_SHUFFLE(3,2,3,2));

  kernel.packet[0] = _mm256_permute2f128_ps(S0, S1, 0x20);
  kernel.packet[1] = _mm256_permute2f128_ps(S2, S3, 0x20);
  kernel.packet[2] = _mm256_permute2f128_ps(S0, S1, 0x31);
  kernel.packet[3] = _mm256_permute2f128_ps(S2, S3, 0x31);
}

template<> EIGEN_DEVICE_FUNC inline void
ptranspose(PacketBlock<Packet4d,4>& kernel) {
  __m256d T0 = _mm256_shuffle_pd(kernel.packet[0], kernel.packet[1], 15);
  __m256d T1 = _mm256_shuffle_pd(kernel.packet[0], kernel.packet[1], 0);
  __m256d T2 = _mm256_shuffle_pd(kernel.packet[2], kernel.packet[3], 15);
  __m256d T3 = _mm256_shuffle_pd(kernel.packet[2], kernel.packet[3], 0);

  kernel.packet[1] = _mm256_permute2f128_pd(T0, T2, 32);
  kernel.packet[3] = _mm256_permute2f128_pd(T0, T2, 49);
  kernel.packet[0] = _mm256_permute2f128_pd(T1, T3, 32);
  kernel.packet[2] = _mm256_permute2f128_pd(T1, T3, 49);
}

template<> EIGEN_STRONG_INLINE Packet8f pblend(const Selector<8>& ifPacket, const Packet8f& thenPacket, const Packet8f& elsePacket) {
  const __m256 zero = _mm256_setzero_ps();
  const __m256 select = _mm256_set_ps(ifPacket.select[7], ifPacket.select[6], ifPacket.select[5], ifPacket.select[4], ifPacket.select[3], ifPacket.select[2], ifPacket.select[1], ifPacket.select[0]);
  __m256 false_mask = _mm256_cmp_ps(select, zero, _CMP_EQ_UQ);
  return _mm256_blendv_ps(thenPacket, elsePacket, false_mask);
}
template<> EIGEN_STRONG_INLINE Packet4d pblend(const Selector<4>& ifPacket, const Packet4d& thenPacket, const Packet4d& elsePacket) {
  const __m256d zero = _mm256_setzero_pd();
  const __m256d select = _mm256_set_pd(ifPacket.select[3], ifPacket.select[2], ifPacket.select[1], ifPacket.select[0]);
  __m256d false_mask = _mm256_cmp_pd(select, zero, _CMP_EQ_UQ);
  return _mm256_blendv_pd(thenPacket, elsePacket, false_mask);
}

// Functions to print vectors of different types, makes debugging much easier.
namespace{
void print4f(char* name, __m128 val) {
  float temp[4] __attribute__((aligned(32)));
  _mm_store_ps(temp, val);
  printf("%s: ", name);
  for (int k = 0; k < 4; k++) printf("%.8e ", temp[k]);
  printf("\n");
}
void print8f(char* name, __m256 val) {
  float temp[8] __attribute__((aligned(32)));
  _mm256_store_ps(temp, val);
  printf("%s: ", name);
  for (int k = 0; k < 8; k++) printf("%.8e ", temp[k]);
  printf("\n");
}
void print4i(char* name, __m128i val) {
  int temp[4] __attribute__((aligned(32)));
  _mm_store_si128((__m128i*)temp, val);
  printf("%s: ", name);
  for (int k = 0; k < 4; k++) printf("%i ", temp[k]);
  printf("\n");
}
void print8i(char* name, __m256i val) {
  int temp[8] __attribute__((aligned(32)));
  _mm256_store_si256((__m256i*)temp, val);
  printf("%s: ", name);
  for (int k = 0; k < 8; k++) printf("%i ", temp[k]);
  printf("\n");
}
void print8b(char* name, __m256i val) {
  int temp[8] __attribute__((aligned(32)));
  _mm256_store_si256((__m256i*)temp, val);
  printf("%s: ", name);
  for (int k = 0; k < 8; k++) printf("0x%08x ", temp[k]);
  printf("\n");
}
void print4d(char* name, __m256d val) {
  double temp[4] __attribute__((aligned(32)));
  _mm256_store_pd(temp, val);
  printf("%s: ", name);
  for (int k = 0; k < 4; k++) printf("%.16e ", temp[k]);
  printf("\n");
}
};

} // end namespace internal

} // end namespace Eigen

#endif // EIGEN_PACKET_MATH_AVX_H
