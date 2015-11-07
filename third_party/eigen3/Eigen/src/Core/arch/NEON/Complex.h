// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2010 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_COMPLEX_NEON_H
#define EIGEN_COMPLEX_NEON_H

namespace Eigen {

namespace internal {

static uint32x4_t p4ui_CONJ_XOR = EIGEN_INIT_NEON_PACKET4(0x00000000, 0x80000000, 0x00000000, 0x80000000);
static uint32x2_t p2ui_CONJ_XOR = EIGEN_INIT_NEON_PACKET2(0x00000000, 0x80000000);

//---------- float ----------
struct Packet2cf
{
  EIGEN_STRONG_INLINE Packet2cf() {}
  EIGEN_STRONG_INLINE explicit Packet2cf(const Packet4f& a) : v(a) {}
  Packet4f  v;
};

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
    HasSetLinear = 0
  };
};

template<> struct unpacket_traits<Packet2cf> { typedef std::complex<float> type; enum {size=2}; typedef Packet2cf half; };

template<> EIGEN_STRONG_INLINE Packet2cf pset1<Packet2cf>(const std::complex<float>&  from)
{
  float32x2_t r64;
  r64 = vld1_f32((float *)&from);

  return Packet2cf(vcombine_f32(r64, r64));
}

template<> EIGEN_STRONG_INLINE Packet2cf padd<Packet2cf>(const Packet2cf& a, const Packet2cf& b) { return Packet2cf(padd<Packet4f>(a.v,b.v)); }
template<> EIGEN_STRONG_INLINE Packet2cf psub<Packet2cf>(const Packet2cf& a, const Packet2cf& b) { return Packet2cf(psub<Packet4f>(a.v,b.v)); }
template<> EIGEN_STRONG_INLINE Packet2cf pnegate(const Packet2cf& a) { return Packet2cf(pnegate<Packet4f>(a.v)); }
template<> EIGEN_STRONG_INLINE Packet2cf pconj(const Packet2cf& a)
{
  Packet4ui b = vreinterpretq_u32_f32(a.v);
  return Packet2cf(vreinterpretq_f32_u32(veorq_u32(b, p4ui_CONJ_XOR)));
}

template<> EIGEN_STRONG_INLINE Packet2cf pmul<Packet2cf>(const Packet2cf& a, const Packet2cf& b)
{
  Packet4f v1, v2;

  // Get the real values of a | a1_re | a1_re | a2_re | a2_re |
  v1 = vcombine_f32(vdup_lane_f32(vget_low_f32(a.v), 0), vdup_lane_f32(vget_high_f32(a.v), 0));
  // Get the real values of a | a1_im | a1_im | a2_im | a2_im |
  v2 = vcombine_f32(vdup_lane_f32(vget_low_f32(a.v), 1), vdup_lane_f32(vget_high_f32(a.v), 1));
  // Multiply the real a with b
  v1 = vmulq_f32(v1, b.v);
  // Multiply the imag a with b
  v2 = vmulq_f32(v2, b.v);
  // Conjugate v2 
  v2 = vreinterpretq_f32_u32(veorq_u32(vreinterpretq_u32_f32(v2), p4ui_CONJ_XOR));
  // Swap real/imag elements in v2.
  v2 = vrev64q_f32(v2);
  // Add and return the result
  return Packet2cf(vaddq_f32(v1, v2));
}

template<> EIGEN_STRONG_INLINE Packet2cf pand   <Packet2cf>(const Packet2cf& a, const Packet2cf& b)
{
  return Packet2cf(vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(a.v),vreinterpretq_u32_f32(b.v))));
}
template<> EIGEN_STRONG_INLINE Packet2cf por    <Packet2cf>(const Packet2cf& a, const Packet2cf& b)
{
  return Packet2cf(vreinterpretq_f32_u32(vorrq_u32(vreinterpretq_u32_f32(a.v),vreinterpretq_u32_f32(b.v))));
}
template<> EIGEN_STRONG_INLINE Packet2cf pxor   <Packet2cf>(const Packet2cf& a, const Packet2cf& b)
{
  return Packet2cf(vreinterpretq_f32_u32(veorq_u32(vreinterpretq_u32_f32(a.v),vreinterpretq_u32_f32(b.v))));
}
template<> EIGEN_STRONG_INLINE Packet2cf pandnot<Packet2cf>(const Packet2cf& a, const Packet2cf& b)
{
  return Packet2cf(vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(a.v),vreinterpretq_u32_f32(b.v))));
}

template<> EIGEN_STRONG_INLINE Packet2cf pload<Packet2cf>(const std::complex<float>* from) { EIGEN_DEBUG_ALIGNED_LOAD return Packet2cf(pload<Packet4f>((const float*)from)); }
template<> EIGEN_STRONG_INLINE Packet2cf ploadu<Packet2cf>(const std::complex<float>* from) { EIGEN_DEBUG_UNALIGNED_LOAD return Packet2cf(ploadu<Packet4f>((const float*)from)); }

template<> EIGEN_STRONG_INLINE Packet2cf ploaddup<Packet2cf>(const std::complex<float>* from) { return pset1<Packet2cf>(*from); }

template<> EIGEN_STRONG_INLINE void pstore <std::complex<float> >(std::complex<float> *   to, const Packet2cf& from) { EIGEN_DEBUG_ALIGNED_STORE pstore((float*)to, from.v); }
template<> EIGEN_STRONG_INLINE void pstoreu<std::complex<float> >(std::complex<float> *   to, const Packet2cf& from) { EIGEN_DEBUG_UNALIGNED_STORE pstoreu((float*)to, from.v); }

template<> EIGEN_DEVICE_FUNC inline Packet2cf pgather<std::complex<float>, Packet2cf>(const std::complex<float>* from, int stride)
{
  Packet4f res = pset1<Packet4f>(0.f);
  res = vsetq_lane_f32(std::real(from[0*stride]), res, 0);
  res = vsetq_lane_f32(std::imag(from[0*stride]), res, 1);
  res = vsetq_lane_f32(std::real(from[1*stride]), res, 2);
  res = vsetq_lane_f32(std::imag(from[1*stride]), res, 3);
  return Packet2cf(res);
}

template<> EIGEN_DEVICE_FUNC inline void pscatter<std::complex<float>, Packet2cf>(std::complex<float>* to, const Packet2cf& from, int stride)
{
  to[stride*0] = std::complex<float>(vgetq_lane_f32(from.v, 0), vgetq_lane_f32(from.v, 1));
  to[stride*1] = std::complex<float>(vgetq_lane_f32(from.v, 2), vgetq_lane_f32(from.v, 3));
}

template<> EIGEN_STRONG_INLINE void prefetch<std::complex<float> >(const std::complex<float> *   addr) { EIGEN_ARM_PREFETCH((float *)addr); }

template<> EIGEN_STRONG_INLINE std::complex<float>  pfirst<Packet2cf>(const Packet2cf& a)
{
  std::complex<float> EIGEN_ALIGN16 x[2];
  vst1q_f32((float *)x, a.v);
  return x[0];
}

template<> EIGEN_STRONG_INLINE Packet2cf preverse(const Packet2cf& a)
{
  float32x2_t a_lo, a_hi;
  Packet4f a_r128;

  a_lo = vget_low_f32(a.v);
  a_hi = vget_high_f32(a.v);
  a_r128 = vcombine_f32(a_hi, a_lo);

  return Packet2cf(a_r128);
}

template<> EIGEN_STRONG_INLINE Packet2cf pcplxflip<Packet2cf>(const Packet2cf& a)
{
  return Packet2cf(vrev64q_f32(a.v));
}

template<> EIGEN_STRONG_INLINE std::complex<float> predux<Packet2cf>(const Packet2cf& a)
{
  float32x2_t a1, a2;
  std::complex<float> s;

  a1 = vget_low_f32(a.v);
  a2 = vget_high_f32(a.v);
  a2 = vadd_f32(a1, a2);
  vst1_f32((float *)&s, a2);

  return s;
}

template<> EIGEN_STRONG_INLINE Packet2cf preduxp<Packet2cf>(const Packet2cf* vecs)
{
  Packet4f sum1, sum2, sum;

  // Add the first two 64-bit float32x2_t of vecs[0]
  sum1 = vcombine_f32(vget_low_f32(vecs[0].v), vget_low_f32(vecs[1].v));
  sum2 = vcombine_f32(vget_high_f32(vecs[0].v), vget_high_f32(vecs[1].v));
  sum = vaddq_f32(sum1, sum2);

  return Packet2cf(sum);
}

template<> EIGEN_STRONG_INLINE std::complex<float> predux_mul<Packet2cf>(const Packet2cf& a)
{
  float32x2_t a1, a2, v1, v2, prod;
  std::complex<float> s;

  a1 = vget_low_f32(a.v);
  a2 = vget_high_f32(a.v);
   // Get the real values of a | a1_re | a1_re | a2_re | a2_re |
  v1 = vdup_lane_f32(a1, 0);
  // Get the real values of a | a1_im | a1_im | a2_im | a2_im |
  v2 = vdup_lane_f32(a1, 1);
  // Multiply the real a with b
  v1 = vmul_f32(v1, a2);
  // Multiply the imag a with b
  v2 = vmul_f32(v2, a2);
  // Conjugate v2 
  v2 = vreinterpret_f32_u32(veor_u32(vreinterpret_u32_f32(v2), p2ui_CONJ_XOR));
  // Swap real/imag elements in v2.
  v2 = vrev64_f32(v2);
  // Add v1, v2
  prod = vadd_f32(v1, v2);

  vst1_f32((float *)&s, prod);

  return s;
}

template<int Offset>
struct palign_impl<Offset,Packet2cf>
{
  EIGEN_STRONG_INLINE static void run(Packet2cf& first, const Packet2cf& second)
  {
    if (Offset==1)
    {
      first.v = vextq_f32(first.v, second.v, 2);
    }
  }
};

template<> struct conj_helper<Packet2cf, Packet2cf, false,true>
{
  EIGEN_STRONG_INLINE Packet2cf pmadd(const Packet2cf& x, const Packet2cf& y, const Packet2cf& c) const
  { return padd(pmul(x,y),c); }

  EIGEN_STRONG_INLINE Packet2cf pmul(const Packet2cf& a, const Packet2cf& b) const
  {
    return internal::pmul(a, pconj(b));
  }
};

template<> struct conj_helper<Packet2cf, Packet2cf, true,false>
{
  EIGEN_STRONG_INLINE Packet2cf pmadd(const Packet2cf& x, const Packet2cf& y, const Packet2cf& c) const
  { return padd(pmul(x,y),c); }

  EIGEN_STRONG_INLINE Packet2cf pmul(const Packet2cf& a, const Packet2cf& b) const
  {
    return internal::pmul(pconj(a), b);
  }
};

template<> struct conj_helper<Packet2cf, Packet2cf, true,true>
{
  EIGEN_STRONG_INLINE Packet2cf pmadd(const Packet2cf& x, const Packet2cf& y, const Packet2cf& c) const
  { return padd(pmul(x,y),c); }

  EIGEN_STRONG_INLINE Packet2cf pmul(const Packet2cf& a, const Packet2cf& b) const
  {
    return pconj(internal::pmul(a, b));
  }
};

template<> EIGEN_STRONG_INLINE Packet2cf pdiv<Packet2cf>(const Packet2cf& a, const Packet2cf& b)
{
  // TODO optimize it for NEON
  Packet2cf res = conj_helper<Packet2cf,Packet2cf,false,true>().pmul(a,b);
  Packet4f s, rev_s;

  // this computes the norm
  s = vmulq_f32(b.v, b.v);
  rev_s = vrev64q_f32(s);

  return Packet2cf(pdiv(res.v, vaddq_f32(s,rev_s)));
}

template<> EIGEN_DEVICE_FUNC inline void
ptranspose(PacketBlock<Packet2cf,2>& kernel) {
  Packet4f tmp = vcombine_f32(vget_high_f32(kernel.packet[0].v), vget_high_f32(kernel.packet[1].v));
  kernel.packet[0].v = vcombine_f32(vget_low_f32(kernel.packet[0].v), vget_low_f32(kernel.packet[1].v));
  kernel.packet[1].v = tmp;
}

//---------- double ----------
#if EIGEN_ARCH_ARM64 && !EIGEN_APPLE_DOUBLE_NEON_BUG

static uint64x2_t p2ul_CONJ_XOR = EIGEN_INIT_NEON_PACKET2(0x0, 0x8000000000000000);

struct Packet1cd
{
  EIGEN_STRONG_INLINE Packet1cd() {}
  EIGEN_STRONG_INLINE explicit Packet1cd(const Packet2d& a) : v(a) {}
  Packet2d v;
};

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

template<> struct unpacket_traits<Packet1cd> { typedef std::complex<double> type; enum {size=1}; typedef Packet1cd half; };

template<> EIGEN_STRONG_INLINE Packet1cd pload<Packet1cd>(const std::complex<double>* from) { EIGEN_DEBUG_ALIGNED_LOAD return Packet1cd(pload<Packet2d>((const double*)from)); }
template<> EIGEN_STRONG_INLINE Packet1cd ploadu<Packet1cd>(const std::complex<double>* from) { EIGEN_DEBUG_UNALIGNED_LOAD return Packet1cd(ploadu<Packet2d>((const double*)from)); }

template<> EIGEN_STRONG_INLINE Packet1cd pset1<Packet1cd>(const std::complex<double>&  from)
{ /* here we really have to use unaligned loads :( */ return ploadu<Packet1cd>(&from); }

template<> EIGEN_STRONG_INLINE Packet1cd padd<Packet1cd>(const Packet1cd& a, const Packet1cd& b) { return Packet1cd(padd<Packet2d>(a.v,b.v)); }
template<> EIGEN_STRONG_INLINE Packet1cd psub<Packet1cd>(const Packet1cd& a, const Packet1cd& b) { return Packet1cd(psub<Packet2d>(a.v,b.v)); }
template<> EIGEN_STRONG_INLINE Packet1cd pnegate(const Packet1cd& a) { return Packet1cd(pnegate<Packet2d>(a.v)); }
template<> EIGEN_STRONG_INLINE Packet1cd pconj(const Packet1cd& a) { return Packet1cd(vreinterpretq_f64_u64(veorq_u64(vreinterpretq_u64_f64(a.v), p2ul_CONJ_XOR))); }

template<> EIGEN_STRONG_INLINE Packet1cd pmul<Packet1cd>(const Packet1cd& a, const Packet1cd& b)
{
  Packet2d v1, v2;

  // Get the real values of a 
  v1 = vdupq_lane_f64(vget_low_f64(a.v), 0);
  // Get the real values of a 
  v2 = vdupq_lane_f64(vget_high_f64(a.v), 1);
  // Multiply the real a with b
  v1 = vmulq_f64(v1, b.v);
  // Multiply the imag a with b
  v2 = vmulq_f64(v2, b.v);
  // Conjugate v2 
  v2 = vreinterpretq_f64_u64(veorq_u64(vreinterpretq_u64_f64(v2), p2ul_CONJ_XOR));
  // Swap real/imag elements in v2.
  v2 = preverse<Packet2d>(v2);
  // Add and return the result
  return Packet1cd(vaddq_f64(v1, v2));
}

template<> EIGEN_STRONG_INLINE Packet1cd pand   <Packet1cd>(const Packet1cd& a, const Packet1cd& b)
{
  return Packet1cd(vreinterpretq_f64_u64(vandq_u64(vreinterpretq_u64_f64(a.v),vreinterpretq_u64_f64(b.v))));
}
template<> EIGEN_STRONG_INLINE Packet1cd por    <Packet1cd>(const Packet1cd& a, const Packet1cd& b)
{
  return Packet1cd(vreinterpretq_f64_u64(vorrq_u64(vreinterpretq_u64_f64(a.v),vreinterpretq_u64_f64(b.v))));
}
template<> EIGEN_STRONG_INLINE Packet1cd pxor   <Packet1cd>(const Packet1cd& a, const Packet1cd& b)
{
  return Packet1cd(vreinterpretq_f64_u64(veorq_u64(vreinterpretq_u64_f64(a.v),vreinterpretq_u64_f64(b.v))));
}
template<> EIGEN_STRONG_INLINE Packet1cd pandnot<Packet1cd>(const Packet1cd& a, const Packet1cd& b)
{
  return Packet1cd(vreinterpretq_f64_u64(vbicq_u64(vreinterpretq_u64_f64(a.v),vreinterpretq_u64_f64(b.v))));
}

template<> EIGEN_STRONG_INLINE Packet1cd ploaddup<Packet1cd>(const std::complex<double>* from) { return pset1<Packet1cd>(*from); }

template<> EIGEN_STRONG_INLINE void pstore <std::complex<double> >(std::complex<double> *   to, const Packet1cd& from) { EIGEN_DEBUG_ALIGNED_STORE pstore((double*)to, from.v); }
template<> EIGEN_STRONG_INLINE void pstoreu<std::complex<double> >(std::complex<double> *   to, const Packet1cd& from) { EIGEN_DEBUG_UNALIGNED_STORE pstoreu((double*)to, from.v); }

template<> EIGEN_STRONG_INLINE void prefetch<std::complex<double> >(const std::complex<double> *   addr) { EIGEN_ARM_PREFETCH((double *)addr); }

template<> EIGEN_DEVICE_FUNC inline Packet1cd pgather<std::complex<double>, Packet1cd>(const std::complex<double>* from, int stride)
{
  Packet2d res = pset1<Packet2d>(0.0);
  res = vsetq_lane_f64(std::real(from[0*stride]), res, 0);
  res = vsetq_lane_f64(std::imag(from[0*stride]), res, 1);
  return Packet1cd(res);
}

template<> EIGEN_DEVICE_FUNC inline void pscatter<std::complex<double>, Packet1cd>(std::complex<double>* to, const Packet1cd& from, int stride)
{
  to[stride*0] = std::complex<double>(vgetq_lane_f64(from.v, 0), vgetq_lane_f64(from.v, 1));
}


template<> EIGEN_STRONG_INLINE std::complex<double>  pfirst<Packet1cd>(const Packet1cd& a)
{
  std::complex<double> EIGEN_ALIGN16 res;
  pstore<std::complex<double> >(&res, a);

  return res;
}

template<> EIGEN_STRONG_INLINE Packet1cd preverse(const Packet1cd& a) { return a; }

template<> EIGEN_STRONG_INLINE std::complex<double> predux<Packet1cd>(const Packet1cd& a) { return pfirst(a); }

template<> EIGEN_STRONG_INLINE Packet1cd preduxp<Packet1cd>(const Packet1cd* vecs) { return vecs[0]; }

template<> EIGEN_STRONG_INLINE std::complex<double> predux_mul<Packet1cd>(const Packet1cd& a) { return pfirst(a); }

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
    return internal::pmul(a, pconj(b));
  }
};

template<> struct conj_helper<Packet1cd, Packet1cd, true,false>
{
  EIGEN_STRONG_INLINE Packet1cd pmadd(const Packet1cd& x, const Packet1cd& y, const Packet1cd& c) const
  { return padd(pmul(x,y),c); }

  EIGEN_STRONG_INLINE Packet1cd pmul(const Packet1cd& a, const Packet1cd& b) const
  {
    return internal::pmul(pconj(a), b);
  }
};

template<> struct conj_helper<Packet1cd, Packet1cd, true,true>
{
  EIGEN_STRONG_INLINE Packet1cd pmadd(const Packet1cd& x, const Packet1cd& y, const Packet1cd& c) const
  { return padd(pmul(x,y),c); }

  EIGEN_STRONG_INLINE Packet1cd pmul(const Packet1cd& a, const Packet1cd& b) const
  {
    return pconj(internal::pmul(a, b));
  }
};

template<> EIGEN_STRONG_INLINE Packet1cd pdiv<Packet1cd>(const Packet1cd& a, const Packet1cd& b)
{
  // TODO optimize it for NEON
  Packet1cd res = conj_helper<Packet1cd,Packet1cd,false,true>().pmul(a,b);
  Packet2d s = pmul<Packet2d>(b.v, b.v);
  Packet2d rev_s = preverse<Packet2d>(s);

  return Packet1cd(pdiv(res.v, padd<Packet2d>(s,rev_s)));
}

EIGEN_STRONG_INLINE Packet1cd pcplxflip/*<Packet1cd>*/(const Packet1cd& x)
{
  return Packet1cd(preverse(Packet2d(x.v)));
}

EIGEN_STRONG_INLINE void ptranspose(PacketBlock<Packet1cd,2>& kernel)
{
  Packet2d tmp = vcombine_f64(vget_high_f64(kernel.packet[0].v), vget_high_f64(kernel.packet[1].v));
  kernel.packet[0].v = vcombine_f64(vget_low_f64(kernel.packet[0].v), vget_low_f64(kernel.packet[1].v));
  kernel.packet[1].v = tmp;
}
#endif // EIGEN_ARCH_ARM64


} // end namespace internal

} // end namespace Eigen

#endif // EIGEN_COMPLEX_NEON_H
