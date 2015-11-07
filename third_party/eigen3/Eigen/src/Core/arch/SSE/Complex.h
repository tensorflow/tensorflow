// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2010 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_COMPLEX_SSE_H
#define EIGEN_COMPLEX_SSE_H

namespace Eigen {

namespace internal {

//---------- float ----------
struct Packet2cf
{
  EIGEN_STRONG_INLINE Packet2cf() {}
  EIGEN_STRONG_INLINE explicit Packet2cf(const __m128& a) : v(a) {}
  __m128  v;
};

// Use the packet_traits defined in AVX/PacketMath.h instead if we're going
// to leverage AVX instructions.
#ifndef EIGEN_VECTORIZE_AVX
template<> struct packet_traits<std::complex<float> >  : default_packet_traits
{
  typedef Packet2cf type;
  typedef Packet2cf half;
  enum {
    Vectorizable = 1,
    AlignedOnScalar = 1,
    size = 2,
    HasHalfPacket = 0,

    HasAdd    = 1,
    HasSub    = 1,
    HasMul    = 1,
    HasDiv    = 1,
    HasNegate = 1,
    HasAbs    = 0,
    HasAbs2   = 0,
    HasMin    = 0,
    HasMax    = 0,
    HasSetLinear = 0,
    HasBlend = 1,
  };
};
#endif

template<> struct unpacket_traits<Packet2cf> { typedef std::complex<float> type; enum {size=2}; typedef Packet2cf half; };

template<> EIGEN_STRONG_INLINE Packet2cf padd<Packet2cf>(const Packet2cf& a, const Packet2cf& b) { return Packet2cf(_mm_add_ps(a.v,b.v)); }
template<> EIGEN_STRONG_INLINE Packet2cf psub<Packet2cf>(const Packet2cf& a, const Packet2cf& b) { return Packet2cf(_mm_sub_ps(a.v,b.v)); }
template<> EIGEN_STRONG_INLINE Packet2cf pnegate(const Packet2cf& a)
{
  const __m128 mask = _mm_castsi128_ps(_mm_setr_epi32(0x80000000,0x80000000,0x80000000,0x80000000));
  return Packet2cf(_mm_xor_ps(a.v,mask));
}
template<> EIGEN_STRONG_INLINE Packet2cf pconj(const Packet2cf& a)
{
  const __m128 mask = _mm_castsi128_ps(_mm_setr_epi32(0x00000000,0x80000000,0x00000000,0x80000000));
  return Packet2cf(_mm_xor_ps(a.v,mask));
}

template<> EIGEN_STRONG_INLINE Packet2cf pmul<Packet2cf>(const Packet2cf& a, const Packet2cf& b)
{
  // TODO optimize it for SSE3 and 4
  #ifdef EIGEN_VECTORIZE_SSE3
  return Packet2cf(_mm_addsub_ps(_mm_mul_ps(_mm_moveldup_ps(a.v), b.v),
                                 _mm_mul_ps(_mm_movehdup_ps(a.v),
                                            vec4f_swizzle1(b.v, 1, 0, 3, 2))));
//   return Packet2cf(_mm_addsub_ps(_mm_mul_ps(vec4f_swizzle1(a.v, 0, 0, 2, 2), b.v),
//                                  _mm_mul_ps(vec4f_swizzle1(a.v, 1, 1, 3, 3),
//                                             vec4f_swizzle1(b.v, 1, 0, 3, 2))));
  #else
  const __m128 mask = _mm_castsi128_ps(_mm_setr_epi32(0x80000000,0x00000000,0x80000000,0x00000000));
  return Packet2cf(_mm_add_ps(_mm_mul_ps(vec4f_swizzle1(a.v, 0, 0, 2, 2), b.v),
                              _mm_xor_ps(_mm_mul_ps(vec4f_swizzle1(a.v, 1, 1, 3, 3),
                                                    vec4f_swizzle1(b.v, 1, 0, 3, 2)), mask)));
  #endif
}

template<> EIGEN_STRONG_INLINE Packet2cf pand   <Packet2cf>(const Packet2cf& a, const Packet2cf& b) { return Packet2cf(_mm_and_ps(a.v,b.v)); }
template<> EIGEN_STRONG_INLINE Packet2cf por    <Packet2cf>(const Packet2cf& a, const Packet2cf& b) { return Packet2cf(_mm_or_ps(a.v,b.v)); }
template<> EIGEN_STRONG_INLINE Packet2cf pxor   <Packet2cf>(const Packet2cf& a, const Packet2cf& b) { return Packet2cf(_mm_xor_ps(a.v,b.v)); }
template<> EIGEN_STRONG_INLINE Packet2cf pandnot<Packet2cf>(const Packet2cf& a, const Packet2cf& b) { return Packet2cf(_mm_andnot_ps(a.v,b.v)); }

template<> EIGEN_STRONG_INLINE Packet2cf pload <Packet2cf>(const std::complex<float>* from) { EIGEN_DEBUG_ALIGNED_LOAD return Packet2cf(pload<Packet4f>(&numext::real_ref(*from))); }
template<> EIGEN_STRONG_INLINE Packet2cf ploadu<Packet2cf>(const std::complex<float>* from) { EIGEN_DEBUG_UNALIGNED_LOAD return Packet2cf(ploadu<Packet4f>(&numext::real_ref(*from))); }

template<> EIGEN_STRONG_INLINE Packet2cf pset1<Packet2cf>(const std::complex<float>&  from)
{
  Packet2cf res;
#if EIGEN_GNUC_AT_MOST(4,2)
  // Workaround annoying "may be used uninitialized in this function" warning with gcc 4.2
  res.v = _mm_loadl_pi(_mm_set1_ps(0.0f), reinterpret_cast<const __m64*>(&from));
#elif EIGEN_GNUC_AT_LEAST(4,6)
  // Suppress annoying "may be used uninitialized in this function" warning with gcc >= 4.6
  #pragma GCC diagnostic push
  #pragma GCC diagnostic ignored "-Wuninitialized"
  res.v = _mm_loadl_pi(res.v, (const __m64*)&from);
  #pragma GCC diagnostic pop
#else
  res.v = _mm_loadl_pi(res.v, (const __m64*)&from);
#endif
  return Packet2cf(_mm_movelh_ps(res.v,res.v));
}

template<> EIGEN_STRONG_INLINE Packet2cf ploaddup<Packet2cf>(const std::complex<float>* from) { return pset1<Packet2cf>(*from); }

template<> EIGEN_STRONG_INLINE void pstore <std::complex<float> >(std::complex<float> *   to, const Packet2cf& from) { EIGEN_DEBUG_ALIGNED_STORE pstore(&numext::real_ref(*to), Packet4f(from.v)); }
template<> EIGEN_STRONG_INLINE void pstoreu<std::complex<float> >(std::complex<float> *   to, const Packet2cf& from) { EIGEN_DEBUG_UNALIGNED_STORE pstoreu(&numext::real_ref(*to), Packet4f(from.v)); }


template<> EIGEN_DEVICE_FUNC inline Packet2cf pgather<std::complex<float>, Packet2cf>(const std::complex<float>* from, int stride)
{
  return Packet2cf(_mm_set_ps(std::imag(from[1*stride]), std::real(from[1*stride]),
                              std::imag(from[0*stride]), std::real(from[0*stride])));
}

template<> EIGEN_DEVICE_FUNC inline void pscatter<std::complex<float>, Packet2cf>(std::complex<float>* to, const Packet2cf& from, int stride)
{
  to[stride*0] = std::complex<float>(_mm_cvtss_f32(_mm_shuffle_ps(from.v, from.v, 0)),
                                     _mm_cvtss_f32(_mm_shuffle_ps(from.v, from.v, 1)));
  to[stride*1] = std::complex<float>(_mm_cvtss_f32(_mm_shuffle_ps(from.v, from.v, 2)),
                                     _mm_cvtss_f32(_mm_shuffle_ps(from.v, from.v, 3)));
}

template<> EIGEN_STRONG_INLINE void prefetch<std::complex<float> >(const std::complex<float> *   addr) { _mm_prefetch((const char*)(addr), _MM_HINT_T0); }

template<> EIGEN_STRONG_INLINE std::complex<float>  pfirst<Packet2cf>(const Packet2cf& a)
{
  #if EIGEN_GNUC_AT_MOST(4,3)
  // Workaround gcc 4.2 ICE - this is not performance wise ideal, but who cares...
  // This workaround also fix invalid code generation with gcc 4.3
  EIGEN_ALIGN16 std::complex<float> res[2];
  _mm_store_ps((float*)res, a.v);
  return res[0];
  #else
  std::complex<float> res;
  _mm_storel_pi((__m64*)&res, a.v);
  return res;
  #endif
}

template<> EIGEN_STRONG_INLINE Packet2cf preverse(const Packet2cf& a) { return Packet2cf(_mm_castpd_ps(preverse(Packet2d(_mm_castps_pd(a.v))))); }

template<> EIGEN_STRONG_INLINE std::complex<float> predux<Packet2cf>(const Packet2cf& a)
{
  return pfirst(Packet2cf(_mm_add_ps(a.v, _mm_movehl_ps(a.v,a.v))));
}

template<> EIGEN_STRONG_INLINE Packet2cf preduxp<Packet2cf>(const Packet2cf* vecs)
{
  return Packet2cf(_mm_add_ps(_mm_movelh_ps(vecs[0].v,vecs[1].v), _mm_movehl_ps(vecs[1].v,vecs[0].v)));
}

template<> EIGEN_STRONG_INLINE std::complex<float> predux_mul<Packet2cf>(const Packet2cf& a)
{
  return pfirst(pmul(a, Packet2cf(_mm_movehl_ps(a.v,a.v))));
}

template<int Offset>
struct palign_impl<Offset,Packet2cf>
{
  static EIGEN_STRONG_INLINE void run(Packet2cf& first, const Packet2cf& second)
  {
    if (Offset==1)
    {
      first.v = _mm_movehl_ps(first.v, first.v);
      first.v = _mm_movelh_ps(first.v, second.v);
    }
  }
};

template<> struct conj_helper<Packet2cf, Packet2cf, false,true>
{
  EIGEN_STRONG_INLINE Packet2cf pmadd(const Packet2cf& x, const Packet2cf& y, const Packet2cf& c) const
  { return padd(pmul(x,y),c); }

  EIGEN_STRONG_INLINE Packet2cf pmul(const Packet2cf& a, const Packet2cf& b) const
  {
    #ifdef EIGEN_VECTORIZE_SSE3
    return internal::pmul(a, pconj(b));
    #else
    const __m128 mask = _mm_castsi128_ps(_mm_setr_epi32(0x00000000,0x80000000,0x00000000,0x80000000));
    return Packet2cf(_mm_add_ps(_mm_xor_ps(_mm_mul_ps(vec4f_swizzle1(a.v, 0, 0, 2, 2), b.v), mask),
                                _mm_mul_ps(vec4f_swizzle1(a.v, 1, 1, 3, 3),
                                           vec4f_swizzle1(b.v, 1, 0, 3, 2))));
    #endif
  }
};

template<> struct conj_helper<Packet2cf, Packet2cf, true,false>
{
  EIGEN_STRONG_INLINE Packet2cf pmadd(const Packet2cf& x, const Packet2cf& y, const Packet2cf& c) const
  { return padd(pmul(x,y),c); }

  EIGEN_STRONG_INLINE Packet2cf pmul(const Packet2cf& a, const Packet2cf& b) const
  {
    #ifdef EIGEN_VECTORIZE_SSE3
    return internal::pmul(pconj(a), b);
    #else
    const __m128 mask = _mm_castsi128_ps(_mm_setr_epi32(0x00000000,0x80000000,0x00000000,0x80000000));
    return Packet2cf(_mm_add_ps(_mm_mul_ps(vec4f_swizzle1(a.v, 0, 0, 2, 2), b.v),
                                _mm_xor_ps(_mm_mul_ps(vec4f_swizzle1(a.v, 1, 1, 3, 3),
                                                      vec4f_swizzle1(b.v, 1, 0, 3, 2)), mask)));
    #endif
  }
};

template<> struct conj_helper<Packet2cf, Packet2cf, true,true>
{
  EIGEN_STRONG_INLINE Packet2cf pmadd(const Packet2cf& x, const Packet2cf& y, const Packet2cf& c) const
  { return padd(pmul(x,y),c); }

  EIGEN_STRONG_INLINE Packet2cf pmul(const Packet2cf& a, const Packet2cf& b) const
  {
    #ifdef EIGEN_VECTORIZE_SSE3
    return pconj(internal::pmul(a, b));
    #else
    const __m128 mask = _mm_castsi128_ps(_mm_setr_epi32(0x00000000,0x80000000,0x00000000,0x80000000));
    return Packet2cf(_mm_sub_ps(_mm_xor_ps(_mm_mul_ps(vec4f_swizzle1(a.v, 0, 0, 2, 2), b.v), mask),
                                _mm_mul_ps(vec4f_swizzle1(a.v, 1, 1, 3, 3),
                                           vec4f_swizzle1(b.v, 1, 0, 3, 2))));
    #endif
  }
};

template<> struct conj_helper<Packet4f, Packet2cf, false,false>
{
  EIGEN_STRONG_INLINE Packet2cf pmadd(const Packet4f& x, const Packet2cf& y, const Packet2cf& c) const
  { return padd(c, pmul(x,y)); }

  EIGEN_STRONG_INLINE Packet2cf pmul(const Packet4f& x, const Packet2cf& y) const
  { return Packet2cf(Eigen::internal::pmul<Packet4f>(x, y.v)); }
};

template<> struct conj_helper<Packet2cf, Packet4f, false,false>
{
  EIGEN_STRONG_INLINE Packet2cf pmadd(const Packet2cf& x, const Packet4f& y, const Packet2cf& c) const
  { return padd(c, pmul(x,y)); }

  EIGEN_STRONG_INLINE Packet2cf pmul(const Packet2cf& x, const Packet4f& y) const
  { return Packet2cf(Eigen::internal::pmul<Packet4f>(x.v, y)); }
};

template<> EIGEN_STRONG_INLINE Packet2cf pdiv<Packet2cf>(const Packet2cf& a, const Packet2cf& b)
{
  // TODO optimize it for SSE3 and 4
  Packet2cf res = conj_helper<Packet2cf,Packet2cf,false,true>().pmul(a,b);
  __m128 s = _mm_mul_ps(b.v,b.v);
  return Packet2cf(_mm_div_ps(res.v,_mm_add_ps(s,_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(s), 0xb1)))));
}

EIGEN_STRONG_INLINE Packet2cf pcplxflip/*<Packet2cf>*/(const Packet2cf& x)
{
  return Packet2cf(vec4f_swizzle1(x.v, 1, 0, 3, 2));
}


//---------- double ----------
struct Packet1cd
{
  EIGEN_STRONG_INLINE Packet1cd() {}
  EIGEN_STRONG_INLINE explicit Packet1cd(const __m128d& a) : v(a) {}
  __m128d  v;
};

// Use the packet_traits defined in AVX/PacketMath.h instead if we're going
// to leverage AVX instructions.
#ifndef EIGEN_VECTORIZE_AVX
template<> struct packet_traits<std::complex<double> >  : default_packet_traits
{
  typedef Packet1cd type;
  typedef Packet1cd half;
  enum {
    Vectorizable = 1,
    AlignedOnScalar = 0,
    size = 1,
    HasHalfPacket = 0,

    HasAdd    = 1,
    HasSub    = 1,
    HasMul    = 1,
    HasDiv    = 1,
    HasNegate = 1,
    HasAbs    = 0,
    HasAbs2   = 0,
    HasMin    = 0,
    HasMax    = 0,
    HasSetLinear = 0
  };
};
#endif

template<> struct unpacket_traits<Packet1cd> { typedef std::complex<double> type; enum {size=1}; typedef Packet1cd half; };

template<> EIGEN_STRONG_INLINE Packet1cd padd<Packet1cd>(const Packet1cd& a, const Packet1cd& b) { return Packet1cd(_mm_add_pd(a.v,b.v)); }
template<> EIGEN_STRONG_INLINE Packet1cd psub<Packet1cd>(const Packet1cd& a, const Packet1cd& b) { return Packet1cd(_mm_sub_pd(a.v,b.v)); }
template<> EIGEN_STRONG_INLINE Packet1cd pnegate(const Packet1cd& a) { return Packet1cd(pnegate(Packet2d(a.v))); }
template<> EIGEN_STRONG_INLINE Packet1cd pconj(const Packet1cd& a)
{
  const __m128d mask = _mm_castsi128_pd(_mm_set_epi32(0x80000000,0x0,0x0,0x0));
  return Packet1cd(_mm_xor_pd(a.v,mask));
}

template<> EIGEN_STRONG_INLINE Packet1cd pmul<Packet1cd>(const Packet1cd& a, const Packet1cd& b)
{
  // TODO optimize it for SSE3 and 4
  #ifdef EIGEN_VECTORIZE_SSE3
  return Packet1cd(_mm_addsub_pd(_mm_mul_pd(vec2d_swizzle1(a.v, 0, 0), b.v),
                                 _mm_mul_pd(vec2d_swizzle1(a.v, 1, 1),
                                            vec2d_swizzle1(b.v, 1, 0))));
  #else
  const __m128d mask = _mm_castsi128_pd(_mm_set_epi32(0x0,0x0,0x80000000,0x0));
  return Packet1cd(_mm_add_pd(_mm_mul_pd(vec2d_swizzle1(a.v, 0, 0), b.v),
                              _mm_xor_pd(_mm_mul_pd(vec2d_swizzle1(a.v, 1, 1),
                                                    vec2d_swizzle1(b.v, 1, 0)), mask)));
  #endif
}

template<> EIGEN_STRONG_INLINE Packet1cd pand   <Packet1cd>(const Packet1cd& a, const Packet1cd& b) { return Packet1cd(_mm_and_pd(a.v,b.v)); }
template<> EIGEN_STRONG_INLINE Packet1cd por    <Packet1cd>(const Packet1cd& a, const Packet1cd& b) { return Packet1cd(_mm_or_pd(a.v,b.v)); }
template<> EIGEN_STRONG_INLINE Packet1cd pxor   <Packet1cd>(const Packet1cd& a, const Packet1cd& b) { return Packet1cd(_mm_xor_pd(a.v,b.v)); }
template<> EIGEN_STRONG_INLINE Packet1cd pandnot<Packet1cd>(const Packet1cd& a, const Packet1cd& b) { return Packet1cd(_mm_andnot_pd(a.v,b.v)); }

// FIXME force unaligned load, this is a temporary fix
template<> EIGEN_STRONG_INLINE Packet1cd pload <Packet1cd>(const std::complex<double>* from)
{ EIGEN_DEBUG_ALIGNED_LOAD return Packet1cd(pload<Packet2d>((const double*)from)); }
template<> EIGEN_STRONG_INLINE Packet1cd ploadu<Packet1cd>(const std::complex<double>* from)
{ EIGEN_DEBUG_UNALIGNED_LOAD return Packet1cd(ploadu<Packet2d>((const double*)from)); }
template<> EIGEN_STRONG_INLINE Packet1cd pset1<Packet1cd>(const std::complex<double>&  from)
{ /* here we really have to use unaligned loads :( */ return ploadu<Packet1cd>(&from); }

template<> EIGEN_STRONG_INLINE Packet1cd ploaddup<Packet1cd>(const std::complex<double>* from) { return pset1<Packet1cd>(*from); }

// FIXME force unaligned store, this is a temporary fix
template<> EIGEN_STRONG_INLINE void pstore <std::complex<double> >(std::complex<double> *   to, const Packet1cd& from) { EIGEN_DEBUG_ALIGNED_STORE pstore((double*)to, Packet2d(from.v)); }
template<> EIGEN_STRONG_INLINE void pstoreu<std::complex<double> >(std::complex<double> *   to, const Packet1cd& from) { EIGEN_DEBUG_UNALIGNED_STORE pstoreu((double*)to, Packet2d(from.v)); }

template<> EIGEN_STRONG_INLINE void prefetch<std::complex<double> >(const std::complex<double> *   addr) { _mm_prefetch((const char*)(addr), _MM_HINT_T0); }

template<> EIGEN_STRONG_INLINE std::complex<double>  pfirst<Packet1cd>(const Packet1cd& a)
{
  EIGEN_ALIGN16 double res[2];
  _mm_store_pd(res, a.v);
  return std::complex<double>(res[0],res[1]);
}

template<> EIGEN_STRONG_INLINE Packet1cd preverse(const Packet1cd& a) { return a; }

template<> EIGEN_STRONG_INLINE std::complex<double> predux<Packet1cd>(const Packet1cd& a)
{
  return pfirst(a);
}

template<> EIGEN_STRONG_INLINE Packet1cd preduxp<Packet1cd>(const Packet1cd* vecs)
{
  return vecs[0];
}

template<> EIGEN_STRONG_INLINE std::complex<double> predux_mul<Packet1cd>(const Packet1cd& a)
{
  return pfirst(a);
}

template<int Offset>
struct palign_impl<Offset,Packet1cd>
{
  static EIGEN_STRONG_INLINE void run(Packet1cd& /*first*/, const Packet1cd& /*second*/)
  {
    // FIXME is it sure we never have to align a Packet1cd?
    // Even though a std::complex<double> has 16 bytes, it is not necessarily aligned on a 16 bytes boundary...
  }
};

template<> struct conj_helper<Packet1cd, Packet1cd, false,true>
{
  EIGEN_STRONG_INLINE Packet1cd pmadd(const Packet1cd& x, const Packet1cd& y, const Packet1cd& c) const
  { return padd(pmul(x,y),c); }

  EIGEN_STRONG_INLINE Packet1cd pmul(const Packet1cd& a, const Packet1cd& b) const
  {
    #ifdef EIGEN_VECTORIZE_SSE3
    return internal::pmul(a, pconj(b));
    #else
    const __m128d mask = _mm_castsi128_pd(_mm_set_epi32(0x80000000,0x0,0x0,0x0));
    return Packet1cd(_mm_add_pd(_mm_xor_pd(_mm_mul_pd(vec2d_swizzle1(a.v, 0, 0), b.v), mask),
                                _mm_mul_pd(vec2d_swizzle1(a.v, 1, 1),
                                           vec2d_swizzle1(b.v, 1, 0))));
    #endif
  }
};

template<> struct conj_helper<Packet1cd, Packet1cd, true,false>
{
  EIGEN_STRONG_INLINE Packet1cd pmadd(const Packet1cd& x, const Packet1cd& y, const Packet1cd& c) const
  { return padd(pmul(x,y),c); }

  EIGEN_STRONG_INLINE Packet1cd pmul(const Packet1cd& a, const Packet1cd& b) const
  {
    #ifdef EIGEN_VECTORIZE_SSE3
    return internal::pmul(pconj(a), b);
    #else
    const __m128d mask = _mm_castsi128_pd(_mm_set_epi32(0x80000000,0x0,0x0,0x0));
    return Packet1cd(_mm_add_pd(_mm_mul_pd(vec2d_swizzle1(a.v, 0, 0), b.v),
                                _mm_xor_pd(_mm_mul_pd(vec2d_swizzle1(a.v, 1, 1),
                                                      vec2d_swizzle1(b.v, 1, 0)), mask)));
    #endif
  }
};

template<> struct conj_helper<Packet1cd, Packet1cd, true,true>
{
  EIGEN_STRONG_INLINE Packet1cd pmadd(const Packet1cd& x, const Packet1cd& y, const Packet1cd& c) const
  { return padd(pmul(x,y),c); }

  EIGEN_STRONG_INLINE Packet1cd pmul(const Packet1cd& a, const Packet1cd& b) const
  {
    #ifdef EIGEN_VECTORIZE_SSE3
    return pconj(internal::pmul(a, b));
    #else
    const __m128d mask = _mm_castsi128_pd(_mm_set_epi32(0x80000000,0x0,0x0,0x0));
    return Packet1cd(_mm_sub_pd(_mm_xor_pd(_mm_mul_pd(vec2d_swizzle1(a.v, 0, 0), b.v), mask),
                                _mm_mul_pd(vec2d_swizzle1(a.v, 1, 1),
                                           vec2d_swizzle1(b.v, 1, 0))));
    #endif
  }
};

template<> struct conj_helper<Packet2d, Packet1cd, false,false>
{
  EIGEN_STRONG_INLINE Packet1cd pmadd(const Packet2d& x, const Packet1cd& y, const Packet1cd& c) const
  { return padd(c, pmul(x,y)); }

  EIGEN_STRONG_INLINE Packet1cd pmul(const Packet2d& x, const Packet1cd& y) const
  { return Packet1cd(Eigen::internal::pmul<Packet2d>(x, y.v)); }
};

template<> struct conj_helper<Packet1cd, Packet2d, false,false>
{
  EIGEN_STRONG_INLINE Packet1cd pmadd(const Packet1cd& x, const Packet2d& y, const Packet1cd& c) const
  { return padd(c, pmul(x,y)); }

  EIGEN_STRONG_INLINE Packet1cd pmul(const Packet1cd& x, const Packet2d& y) const
  { return Packet1cd(Eigen::internal::pmul<Packet2d>(x.v, y)); }
};

template<> EIGEN_STRONG_INLINE Packet1cd pdiv<Packet1cd>(const Packet1cd& a, const Packet1cd& b)
{
  // TODO optimize it for SSE3 and 4
  Packet1cd res = conj_helper<Packet1cd,Packet1cd,false,true>().pmul(a,b);
  __m128d s = _mm_mul_pd(b.v,b.v);
  return Packet1cd(_mm_div_pd(res.v, _mm_add_pd(s,_mm_shuffle_pd(s, s, 0x1))));
}

EIGEN_STRONG_INLINE Packet1cd pcplxflip/*<Packet1cd>*/(const Packet1cd& x)
{
  return Packet1cd(preverse(Packet2d(x.v)));
}

template<> EIGEN_DEVICE_FUNC inline void
ptranspose(PacketBlock<Packet2cf,2>& kernel) {
  __m128d w1 = _mm_castps_pd(kernel.packet[0].v);
  __m128d w2 = _mm_castps_pd(kernel.packet[1].v);

  __m128 tmp = _mm_castpd_ps(_mm_unpackhi_pd(w1, w2));
  kernel.packet[0].v = _mm_castpd_ps(_mm_unpacklo_pd(w1, w2));
  kernel.packet[1].v = tmp;
}

template<>  EIGEN_STRONG_INLINE Packet2cf pblend(const Selector<2>& ifPacket, const Packet2cf& thenPacket, const Packet2cf& elsePacket) {
  __m128d result = pblend(ifPacket, _mm_castps_pd(thenPacket.v), _mm_castps_pd(elsePacket.v));
  return Packet2cf(_mm_castpd_ps(result));
}


} // end namespace internal

} // end namespace Eigen

#endif // EIGEN_COMPLEX_SSE_H
