// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2010 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_COMPLEX32_ALTIVEC_H
#define EIGEN_COMPLEX32_ALTIVEC_H


namespace Eigen {

namespace internal {

static Packet4ui  p4ui_CONJ_XOR = vec_mergeh((Packet4ui)p4i_ZERO, (Packet4ui)p4f_ZERO_);//{ 0x00000000, 0x80000000, 0x00000000, 0x80000000 };
#ifdef EIGEN_VECTORIZE_VSX
#ifdef _BIG_ENDIAN
static Packet2ul  p2ul_CONJ_XOR1 = (Packet2ul) vec_sld((Packet4ui) p2d_ZERO_, (Packet4ui) p2l_ZERO, 8);//{ 0x8000000000000000, 0x0000000000000000 };
static Packet2ul  p2ul_CONJ_XOR2 = (Packet2ul) vec_sld((Packet4ui) p2l_ZERO,  (Packet4ui) p2d_ZERO_, 8);//{ 0x8000000000000000, 0x0000000000000000 };
#else
static Packet2ul  p2ul_CONJ_XOR1 = (Packet2ul) vec_sld((Packet4ui) p2l_ZERO,  (Packet4ui) p2d_ZERO_, 8);//{ 0x8000000000000000, 0x0000000000000000 };
static Packet2ul  p2ul_CONJ_XOR2 = (Packet2ul) vec_sld((Packet4ui) p2d_ZERO_, (Packet4ui) p2l_ZERO, 8);//{ 0x8000000000000000, 0x0000000000000000 };
#endif
#endif  // EIGEN_VECTORIZE_VSX

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
  Packet2cf res;
  /* On AltiVec we cannot load 64-bit registers, so wa have to take care of alignment */
  if((ptrdiff_t(&from) % 16) == 0)
    res.v = pload<Packet4f>((const float *)&from);
  else
    res.v = ploadu<Packet4f>((const float *)&from);
  res.v = vec_perm(res.v, res.v, p16uc_PSET64_HI);
  return res;
}

template<> EIGEN_DEVICE_FUNC inline Packet2cf pgather<std::complex<float>, Packet2cf>(const std::complex<float>* from, int stride)
{
  std::complex<float> EIGEN_ALIGN16 af[2];
  af[0] = from[0*stride];
  af[1] = from[1*stride];
  return Packet2cf(vec_ld(0, (const float*)af));
}
template<> EIGEN_DEVICE_FUNC inline void pscatter<std::complex<float>, Packet2cf>(std::complex<float>* to, const Packet2cf& from, int stride)
{
  std::complex<float> EIGEN_ALIGN16 af[2];
  vec_st(from.v, 0, (float*)af);
  to[0*stride] = af[0];
  to[1*stride] = af[1];
}


template<> EIGEN_STRONG_INLINE Packet2cf padd<Packet2cf>(const Packet2cf& a, const Packet2cf& b) { return Packet2cf(vec_add(a.v,b.v)); }
template<> EIGEN_STRONG_INLINE Packet2cf psub<Packet2cf>(const Packet2cf& a, const Packet2cf& b) { return Packet2cf(vec_sub(a.v,b.v)); }
template<> EIGEN_STRONG_INLINE Packet2cf pnegate(const Packet2cf& a) { return Packet2cf(pnegate(a.v)); }
template<> EIGEN_STRONG_INLINE Packet2cf pconj(const Packet2cf& a) { return Packet2cf((Packet4f)vec_xor((Packet4ui)a.v, p4ui_CONJ_XOR)); }

template<> EIGEN_STRONG_INLINE Packet2cf pmul<Packet2cf>(const Packet2cf& a, const Packet2cf& b)
{
  Packet4f v1, v2;

  // Permute and multiply the real parts of a and b
  v1 = vec_perm(a.v, a.v, p16uc_PSET32_WODD);
  // Get the imaginary parts of a
  v2 = vec_perm(a.v, a.v, p16uc_PSET32_WEVEN);
  // multiply a_re * b 
  v1 = vec_madd(v1, b.v, p4f_ZERO);
  // multiply a_im * b and get the conjugate result
  v2 = vec_madd(v2, b.v, p4f_ZERO);
  v2 = (Packet4f) vec_xor((Packet4ui)v2, p4ui_CONJ_XOR);
  // permute back to a proper order
  v2 = vec_perm(v2, v2, p16uc_COMPLEX32_REV);
  
  return Packet2cf(vec_add(v1, v2));
}

template<> EIGEN_STRONG_INLINE Packet2cf pand   <Packet2cf>(const Packet2cf& a, const Packet2cf& b) { return Packet2cf(vec_and(a.v,b.v)); }
template<> EIGEN_STRONG_INLINE Packet2cf por    <Packet2cf>(const Packet2cf& a, const Packet2cf& b) { return Packet2cf(vec_or(a.v,b.v)); }
template<> EIGEN_STRONG_INLINE Packet2cf pxor   <Packet2cf>(const Packet2cf& a, const Packet2cf& b) { return Packet2cf(vec_xor(a.v,b.v)); }
template<> EIGEN_STRONG_INLINE Packet2cf pandnot<Packet2cf>(const Packet2cf& a, const Packet2cf& b) { return Packet2cf(vec_and(a.v, vec_nor(b.v,b.v))); }

template<> EIGEN_STRONG_INLINE Packet2cf pload <Packet2cf>(const std::complex<float>* from) { EIGEN_DEBUG_ALIGNED_LOAD return Packet2cf(pload<Packet4f>((const float*)from)); }
template<> EIGEN_STRONG_INLINE Packet2cf ploadu<Packet2cf>(const std::complex<float>* from) { EIGEN_DEBUG_UNALIGNED_LOAD return Packet2cf(ploadu<Packet4f>((const float*)from)); }

template<> EIGEN_STRONG_INLINE Packet2cf ploaddup<Packet2cf>(const std::complex<float>*     from)
{
  return pset1<Packet2cf>(*from);
}

template<> EIGEN_STRONG_INLINE void pstore <std::complex<float> >(std::complex<float> *   to, const Packet2cf& from) { EIGEN_DEBUG_ALIGNED_STORE pstore((float*)to, from.v); }
template<> EIGEN_STRONG_INLINE void pstoreu<std::complex<float> >(std::complex<float> *   to, const Packet2cf& from) { EIGEN_DEBUG_UNALIGNED_STORE pstoreu((float*)to, from.v); }

#ifndef __VSX__
template<> EIGEN_STRONG_INLINE void prefetch<std::complex<float> >(const std::complex<float> *   addr) { vec_dstt((float *)addr, DST_CTRL(2,2,32), DST_CHAN); }
#endif

template<> EIGEN_STRONG_INLINE std::complex<float>  pfirst<Packet2cf>(const Packet2cf& a)
{
  std::complex<float> EIGEN_ALIGN16 res[2];
  pstore((float *)&res, a.v);

  return res[0];
}

template<> EIGEN_STRONG_INLINE Packet2cf preverse(const Packet2cf& a)
{
  Packet4f rev_a;
  rev_a = vec_perm(a.v, a.v, p16uc_COMPLEX32_REV2);
  return Packet2cf(rev_a);
}

template<> EIGEN_STRONG_INLINE std::complex<float> predux<Packet2cf>(const Packet2cf& a)
{
  Packet4f b;
  b = (Packet4f) vec_sld(a.v, a.v, 8);
  b = padd(a.v, b);
  return pfirst(Packet2cf(b));
}

template<> EIGEN_STRONG_INLINE Packet2cf preduxp<Packet2cf>(const Packet2cf* vecs)
{
  Packet4f b1, b2;
#ifdef _BIG_ENDIAN  
  b1 = (Packet4f) vec_sld(vecs[0].v, vecs[1].v, 8);
  b2 = (Packet4f) vec_sld(vecs[1].v, vecs[0].v, 8);
#else
  b1 = (Packet4f) vec_sld(vecs[1].v, vecs[0].v, 8);
  b2 = (Packet4f) vec_sld(vecs[0].v, vecs[1].v, 8);
#endif
  b2 = (Packet4f) vec_sld(b2, b2, 8);
  b2 = padd(b1, b2);

  return Packet2cf(b2);
}

template<> EIGEN_STRONG_INLINE std::complex<float> predux_mul<Packet2cf>(const Packet2cf& a)
{
  Packet4f b;
  Packet2cf prod;
  b = (Packet4f) vec_sld(a.v, a.v, 8);
  prod = pmul(a, Packet2cf(b));

  return pfirst(prod);
}

template<int Offset>
struct palign_impl<Offset,Packet2cf>
{
  static EIGEN_STRONG_INLINE void run(Packet2cf& first, const Packet2cf& second)
  {
    if (Offset==1)
    {
#ifdef _BIG_ENDIAN
      first.v = vec_sld(first.v, second.v, 8);
#else
      first.v = vec_sld(second.v, first.v, 8);
#endif
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
  // TODO optimize it for AltiVec
  Packet2cf res = conj_helper<Packet2cf,Packet2cf,false,true>().pmul(a,b);
  Packet4f s = vec_madd(b.v, b.v, p4f_ZERO);
  return Packet2cf(pdiv(res.v, vec_add(s,vec_perm(s, s, p16uc_COMPLEX32_REV))));
}

template<> EIGEN_STRONG_INLINE Packet2cf pcplxflip<Packet2cf>(const Packet2cf& x)
{
  return Packet2cf(vec_perm(x.v, x.v, p16uc_COMPLEX32_REV));
}

template<> EIGEN_STRONG_INLINE void ptranspose(PacketBlock<Packet2cf,2>& kernel)
{
  Packet4f tmp = vec_perm(kernel.packet[0].v, kernel.packet[1].v, p16uc_TRANSPOSE64_HI);
  kernel.packet[1].v = vec_perm(kernel.packet[0].v, kernel.packet[1].v, p16uc_TRANSPOSE64_LO);
  kernel.packet[0].v = tmp;
}

//---------- double ----------
#if defined(EIGEN_VECTORIZE_VSX)
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

template<> EIGEN_STRONG_INLINE Packet1cd pload <Packet1cd>(const std::complex<double>* from) { EIGEN_DEBUG_ALIGNED_LOAD return Packet1cd(pload<Packet2d>((const double*)from)); }
template<> EIGEN_STRONG_INLINE Packet1cd ploadu<Packet1cd>(const std::complex<double>* from) { EIGEN_DEBUG_UNALIGNED_LOAD return Packet1cd(ploadu<Packet2d>((const double*)from)); }
template<> EIGEN_STRONG_INLINE void pstore <std::complex<double> >(std::complex<double> *   to, const Packet1cd& from) { EIGEN_DEBUG_ALIGNED_STORE pstore((double*)to, from.v); }
template<> EIGEN_STRONG_INLINE void pstoreu<std::complex<double> >(std::complex<double> *   to, const Packet1cd& from) { EIGEN_DEBUG_UNALIGNED_STORE pstoreu((double*)to, from.v); }

template<> EIGEN_STRONG_INLINE Packet1cd pset1<Packet1cd>(const std::complex<double>&  from)
{ /* here we really have to use unaligned loads :( */ return ploadu<Packet1cd>(&from); }

// Google-local: Change type from DenseIndex to int in patch.
template<> EIGEN_DEVICE_FUNC inline Packet1cd pgather<std::complex<double>, Packet1cd>(const std::complex<double>* from, int/*DenseIndex*/ stride)
{
  std::complex<double> EIGEN_ALIGN16 af[2];
  af[0] = from[0*stride];
  af[1] = from[1*stride];
  return pload<Packet1cd>(af);
}
// Google-local: Change type from DenseIndex to int in patch.
template<> EIGEN_DEVICE_FUNC inline void pscatter<std::complex<double>, Packet1cd>(std::complex<double>* to, const Packet1cd& from, int/*DenseIndex*/ stride)
{
  std::complex<double> EIGEN_ALIGN16 af[2];
  pstore<std::complex<double> >(af, from);
  to[0*stride] = af[0];
  to[1*stride] = af[1];
}

template<> EIGEN_STRONG_INLINE Packet1cd padd<Packet1cd>(const Packet1cd& a, const Packet1cd& b) { return Packet1cd(vec_add(a.v,b.v)); }
template<> EIGEN_STRONG_INLINE Packet1cd psub<Packet1cd>(const Packet1cd& a, const Packet1cd& b) { return Packet1cd(vec_sub(a.v,b.v)); }
template<> EIGEN_STRONG_INLINE Packet1cd pnegate(const Packet1cd& a) { return Packet1cd(pnegate(Packet2d(a.v))); }
template<> EIGEN_STRONG_INLINE Packet1cd pconj(const Packet1cd& a) { return Packet1cd((Packet2d)vec_xor((Packet2d)a.v, (Packet2d)p2ul_CONJ_XOR2)); }

template<> EIGEN_STRONG_INLINE Packet1cd pmul<Packet1cd>(const Packet1cd& a, const Packet1cd& b)
{
  Packet2d a_re, a_im, v1, v2;

  // Permute and multiply the real parts of a and b
  a_re = vec_perm(a.v, a.v, p16uc_PSET64_HI);
  // Get the imaginary parts of a
  a_im = vec_perm(a.v, a.v, p16uc_PSET64_LO);
  // multiply a_re * b
  v1 = vec_madd(a_re, b.v, p2d_ZERO);
  // multiply a_im * b and get the conjugate result
  v2 = vec_madd(a_im, b.v, p2d_ZERO);
  v2 = (Packet2d) vec_sld((Packet4ui)v2, (Packet4ui)v2, 8);
  v2 = (Packet2d) vec_xor((Packet2d)v2, (Packet2d) p2ul_CONJ_XOR1);

  return Packet1cd(vec_add(v1, v2));
}

template<> EIGEN_STRONG_INLINE Packet1cd pand   <Packet1cd>(const Packet1cd& a, const Packet1cd& b) { return Packet1cd(vec_and(a.v,b.v)); }
template<> EIGEN_STRONG_INLINE Packet1cd por    <Packet1cd>(const Packet1cd& a, const Packet1cd& b) { return Packet1cd(vec_or(a.v,b.v)); }
template<> EIGEN_STRONG_INLINE Packet1cd pxor   <Packet1cd>(const Packet1cd& a, const Packet1cd& b) { return Packet1cd(vec_xor(a.v,b.v)); }
template<> EIGEN_STRONG_INLINE Packet1cd pandnot<Packet1cd>(const Packet1cd& a, const Packet1cd& b) { return Packet1cd(vec_and(a.v, vec_nor(b.v,b.v))); }

template<> EIGEN_STRONG_INLINE Packet1cd ploaddup<Packet1cd>(const std::complex<double>*     from)
{
  return pset1<Packet1cd>(*from);
}

#ifndef __VSX__
template<> EIGEN_STRONG_INLINE void prefetch<std::complex<double> >(const std::complex<double> *   addr) { vec_dstt((long *)addr, DST_CTRL(2,2,32), DST_CHAN); }
#endif

template<> EIGEN_STRONG_INLINE std::complex<double>  pfirst<Packet1cd>(const Packet1cd& a)
{
  std::complex<double> EIGEN_ALIGN16 res[2];
  pstore<std::complex<double> >(res, a);

  return res[0];
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
  // TODO optimize it for AltiVec
  Packet1cd res = conj_helper<Packet1cd,Packet1cd,false,true>().pmul(a,b);
  Packet2d s = vec_madd(b.v, b.v, p2d_ZERO_);
  return Packet1cd(pdiv(res.v, vec_add(s,vec_perm(s, s, p16uc_REVERSE64))));
}

EIGEN_STRONG_INLINE Packet1cd pcplxflip/*<Packet1cd>*/(const Packet1cd& x)
{
  return Packet1cd(preverse(Packet2d(x.v)));
}

EIGEN_STRONG_INLINE void ptranspose(PacketBlock<Packet1cd,2>& kernel)
{
  Packet2d tmp = vec_perm(kernel.packet[0].v, kernel.packet[1].v, p16uc_TRANSPOSE64_HI);
  kernel.packet[1].v = vec_perm(kernel.packet[0].v, kernel.packet[1].v, p16uc_TRANSPOSE64_LO);
  kernel.packet[0].v = tmp;
}
#endif // EIGEN_VECTORIZE_VSX
} // end namespace internal

} // end namespace Eigen

#endif // EIGEN_COMPLEX32_ALTIVEC_H
