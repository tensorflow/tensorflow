// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008 Gael Guennebaud <gael.guennebaud@inria.fr>
// Copyright (C) 2006-2008 Benoit Jacob <jacob.benoit.1@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_GENERIC_PACKET_MATH_H
#define EIGEN_GENERIC_PACKET_MATH_H

namespace Eigen {

namespace internal {

/** \internal
  * \file GenericPacketMath.h
  *
  * Default implementation for types not supported by the vectorization.
  * In practice these functions are provided to make easier the writing
  * of generic vectorized code.
  */

#ifndef EIGEN_DEBUG_ALIGNED_LOAD
#define EIGEN_DEBUG_ALIGNED_LOAD
#endif

#ifndef EIGEN_DEBUG_UNALIGNED_LOAD
#define EIGEN_DEBUG_UNALIGNED_LOAD
#endif

#ifndef EIGEN_DEBUG_ALIGNED_STORE
#define EIGEN_DEBUG_ALIGNED_STORE
#endif

#ifndef EIGEN_DEBUG_UNALIGNED_STORE
#define EIGEN_DEBUG_UNALIGNED_STORE
#endif

struct default_packet_traits
{
  enum {
    HasHalfPacket = 0,

    HasAdd    = 1,
    HasSub    = 1,
    HasMul    = 1,
    HasNegate = 1,
    HasAbs    = 1,
    HasAbs2   = 1,
    HasMin    = 1,
    HasMax    = 1,
    HasConj   = 1,
    HasSetLinear = 1,
    HasBlend  = 0,

    HasDiv    = 0,
    HasSqrt   = 0,
    HasRsqrt  = 0,
    HasExp    = 0,
    HasLog    = 0,
    HasPow    = 0,

    HasSin    = 0,
    HasCos    = 0,
    HasTan    = 0,
    HasASin   = 0,
    HasACos   = 0,
    HasATan   = 0,
    HasTanH   = 0
  };
};

template<typename T> struct packet_traits : default_packet_traits
{
  typedef T type;
  typedef T half;
  enum {
    Vectorizable = 0,
    size = 1,
    AlignedOnScalar = 0,
    HasHalfPacket = 0
  };
  enum {
    HasAdd    = 0,
    HasSub    = 0,
    HasMul    = 0,
    HasNegate = 0,
    HasAbs    = 0,
    HasAbs2   = 0,
    HasMin    = 0,
    HasMax    = 0,
    HasConj   = 0,
    HasSetLinear = 0
  };
};

template<typename T> struct packet_traits<const T> : packet_traits<T> { };


template <typename Src, typename Tgt> struct type_casting_traits {
  enum {
    VectorizedCast = 0,
    SrcCoeffRatio = 1,
    TgtCoeffRatio = 1
  };
};

template <typename T> struct type_casting_traits<T, T> {
  enum {
    VectorizedCast = 1,
    SrcCoeffRatio = 1,
    TgtCoeffRatio = 1
  };
};


/** \internal \returns static_cast<TgtType>(a) (coeff-wise) */
template <typename SrcPacket, typename TgtPacket>
EIGEN_DEVICE_FUNC inline TgtPacket
pcast(const SrcPacket& a) {
  return static_cast<TgtPacket>(a);
}
template <typename SrcPacket, typename TgtPacket>
EIGEN_DEVICE_FUNC inline TgtPacket
pcast(const SrcPacket& a, const SrcPacket& /*b*/) {
  return static_cast<TgtPacket>(a);
}

template <typename SrcPacket, typename TgtPacket>
EIGEN_DEVICE_FUNC inline TgtPacket
pcast(const SrcPacket& a, const SrcPacket& /*b*/, const SrcPacket& /*c*/, const SrcPacket& /*d*/) {
  return static_cast<TgtPacket>(a);
}

/** \internal \returns a + b (coeff-wise) */
template<typename Packet> EIGEN_DEVICE_FUNC inline Packet
padd(const Packet& a,
        const Packet& b) { return a+b; }

/** \internal \returns a - b (coeff-wise) */
template<typename Packet> EIGEN_DEVICE_FUNC inline Packet
psub(const Packet& a,
        const Packet& b) { return a-b; }

/** \internal \returns true for if a == b */
template<typename Packet> EIGEN_DEVICE_FUNC inline Packet
peq(const Packet& a, const Packet& b) { return a == b; }

/** \internal \returns true for if a < b */
template<typename Packet> EIGEN_DEVICE_FUNC inline Packet
plt(const Packet& a, const Packet& b) { return a < b; }

/** \internal \returns true for if a <= b */
template<typename Packet> EIGEN_DEVICE_FUNC inline Packet
ple(const Packet& a, const Packet& b) { return a <= b; }

/** \internal \returns b if false_mask is set, else a */
template<typename Packet> EIGEN_DEVICE_FUNC inline Packet
pselect(const Packet& a,
        const Packet& b,
        const Packet& false_mask) {
  return false_mask ? b : a;
}

/** \internal \returns -a (coeff-wise) */
template<typename Packet> EIGEN_DEVICE_FUNC inline Packet
pnegate(const Packet& a) { return -a; }

/** \internal \returns conj(a) (coeff-wise) */

template<typename Packet> EIGEN_DEVICE_FUNC inline Packet
pconj(const Packet& a) { return numext::conj(a); }

/** \internal \returns a * b (coeff-wise) */
template<typename Packet> EIGEN_DEVICE_FUNC inline Packet
pmul(const Packet& a,
        const Packet& b) { return a*b; }

/** \internal \returns a / b (coeff-wise) */
template<typename Packet> EIGEN_DEVICE_FUNC inline Packet
pdiv(const Packet& a,
        const Packet& b) { return a/b; }

/** \internal \returns the min of \a a and \a b  (coeff-wise) */
template<typename Packet> EIGEN_DEVICE_FUNC inline Packet
pmin(const Packet& a,
        const Packet& b) { return numext::mini(a, b); }

/** \internal \returns the max of \a a and \a b  (coeff-wise) */
template<typename Packet> EIGEN_DEVICE_FUNC inline Packet
pmax(const Packet& a,
        const Packet& b) { return numext::maxi(a, b); }

/** \internal \returns the absolute value of \a a */
template<typename Packet> EIGEN_DEVICE_FUNC inline Packet
pabs(const Packet& a) { using std::abs; return abs(a); }

/** \internal \returns the bitwise and of \a a and \a b */
template<typename Packet> EIGEN_DEVICE_FUNC inline Packet
pand(const Packet& a, const Packet& b) { return a & b; }

/** \internal \returns the bitwise or of \a a and \a b */
template<typename Packet> EIGEN_DEVICE_FUNC inline Packet
por(const Packet& a, const Packet& b) { return a | b; }

/** \internal \returns the bitwise xor of \a a and \a b */
template<typename Packet> EIGEN_DEVICE_FUNC inline Packet
pxor(const Packet& a, const Packet& b) { return a ^ b; }

/** \internal \returns the bitwise andnot of \a a and \a b */
template<typename Packet> EIGEN_DEVICE_FUNC inline Packet
pandnot(const Packet& a, const Packet& b) { return a & (!b); }

/** \internal \returns a packet version of \a *from, from must be 16 bytes aligned */
template<typename Packet> EIGEN_DEVICE_FUNC inline Packet
pload(const typename unpacket_traits<Packet>::type* from) { return *from; }

/** \internal \returns a packet version of \a *from, (un-aligned load) */
template<typename Packet> EIGEN_DEVICE_FUNC inline Packet
ploadu(const typename unpacket_traits<Packet>::type* from) { return *from; }

/** \internal \returns a packet with constant coefficients \a a, e.g.: (a,a,a,a) */
template<typename Packet> EIGEN_DEVICE_FUNC inline Packet
pset1(const typename unpacket_traits<Packet>::type& a) { return a; }

/** \internal \returns a packet with constant coefficients \a a[0], e.g.: (a[0],a[0],a[0],a[0]) */
template<typename Packet> EIGEN_DEVICE_FUNC inline Packet
pload1(const typename unpacket_traits<Packet>::type  *a) { return pset1<Packet>(*a); }

/** \internal \returns a packet with elements of \a *from duplicated.
  * For instance, for a packet of 8 elements, 4 scalars will be read from \a *from and
  * duplicated to form: {from[0],from[0],from[1],from[1],from[2],from[2],from[3],from[3]}
  * Currently, this function is only used for scalar * complex products.
  */
template<typename Packet> EIGEN_DEVICE_FUNC inline Packet
ploaddup(const typename unpacket_traits<Packet>::type* from) { return *from; }

/** \internal \returns a packet with elements of \a *from quadrupled.
  * For instance, for a packet of 8 elements, 2 scalars will be read from \a *from and
  * replicated to form: {from[0],from[0],from[0],from[0],from[1],from[1],from[1],from[1]}
  * Currently, this function is only used in matrix products.
  * For packet-size smaller or equal to 4, this function is equivalent to pload1
  */
template<typename Packet> EIGEN_DEVICE_FUNC inline Packet
ploadquad(const typename unpacket_traits<Packet>::type* from)
{ return pload1<Packet>(from); }

/** \internal equivalent to
  * \code
  * a0 = pload1(a+0);
  * a1 = pload1(a+1);
  * a2 = pload1(a+2);
  * a3 = pload1(a+3);
  * \endcode
  * \sa pset1, pload1, ploaddup, pbroadcast2
  */
template<typename Packet> EIGEN_DEVICE_FUNC
inline void pbroadcast4(const typename unpacket_traits<Packet>::type *a,
                        Packet& a0, Packet& a1, Packet& a2, Packet& a3)
{
  a0 = pload1<Packet>(a+0);
  a1 = pload1<Packet>(a+1);
  a2 = pload1<Packet>(a+2);
  a3 = pload1<Packet>(a+3);
}

/** \internal equivalent to
  * \code
  * a0 = pload1(a+0);
  * a1 = pload1(a+1);
  * \endcode
  * \sa pset1, pload1, ploaddup, pbroadcast4
  */
template<typename Packet> EIGEN_DEVICE_FUNC
inline void pbroadcast2(const typename unpacket_traits<Packet>::type *a,
                        Packet& a0, Packet& a1)
{
  a0 = pload1<Packet>(a+0);
  a1 = pload1<Packet>(a+1);
}

/** \internal \brief Returns a packet with coefficients (a,a+1,...,a+packet_size-1). */
template<typename Scalar> inline typename packet_traits<Scalar>::type
plset(const Scalar& a) { return a; }

/** \internal copy the packet \a from to \a *to, \a to must be 16 bytes aligned */
template<typename Scalar, typename Packet> EIGEN_DEVICE_FUNC inline void pstore(Scalar* to, const Packet& from)
{ (*to) = from; }

/** \internal copy the packet \a from to \a *to, (un-aligned store) */
template<typename Scalar, typename Packet> EIGEN_DEVICE_FUNC inline void pstoreu(Scalar* to, const Packet& from)
{  (*to) = from; }

 template<typename Scalar, typename Packet> EIGEN_DEVICE_FUNC inline Packet pgather(const Scalar* from, int /*stride*/)
 { return ploadu<Packet>(from); }

 template<typename Scalar, typename Packet> EIGEN_DEVICE_FUNC inline void pscatter(Scalar* to, const Packet& from, int /*stride*/)
 { pstore(to, from); }

/** \internal tries to do cache prefetching of \a addr */
template<typename Scalar> EIGEN_DEVICE_FUNC inline void prefetch(const Scalar* addr)
{
#ifdef __CUDA_ARCH__
#if defined(__LP64__)
  // 64-bit pointer operand constraint for inlined asm
  asm(" prefetch.L1 [ %1 ];" : "=l"(addr) : "l"(addr));
#else
  // 32-bit pointer operand constraint for inlined asm
  asm(" prefetch.L1 [ %1 ];" : "=r"(addr) : "r"(addr));
#endif
#elif !defined(_MSC_VER)
  __builtin_prefetch(addr);
#endif
}

/** \internal \returns the first element of a packet */
template<typename Packet> EIGEN_DEVICE_FUNC inline typename unpacket_traits<Packet>::type pfirst(const Packet& a)
{ return a; }

/** \internal \returns a packet where the element i contains the sum of the packet of \a vec[i] */
template<typename Packet> EIGEN_DEVICE_FUNC inline Packet
preduxp(const Packet* vecs) { return vecs[0]; }

/** \internal \returns the sum of the elements of \a a*/
template<typename Packet> EIGEN_DEVICE_FUNC inline typename unpacket_traits<Packet>::type predux(const Packet& a)
{ return a; }

/** \internal \returns the sum of the elements of \a a by block of 4 elements.
  * For a packet {a0, a1, a2, a3, a4, a5, a6, a7}, it returns a half packet {a0+a4, a1+a5, a2+a6, a3+a7}
  * For packet-size smaller or equal to 4, this boils down to a noop.
  */
template<typename Packet> EIGEN_DEVICE_FUNC inline
typename conditional<(unpacket_traits<Packet>::size%8)==0,typename unpacket_traits<Packet>::half,Packet>::type
predux4(const Packet& a)
{ return a; }

/** \internal \returns the product of the elements of \a a*/
template<typename Packet> EIGEN_DEVICE_FUNC inline typename unpacket_traits<Packet>::type predux_mul(const Packet& a)
{ return a; }

/** \internal \returns the min of the elements of \a a*/
template<typename Packet> EIGEN_DEVICE_FUNC inline typename unpacket_traits<Packet>::type predux_min(const Packet& a)
{ return a; }

/** \internal \returns the max of the elements of \a a*/
template<typename Packet> EIGEN_DEVICE_FUNC inline typename unpacket_traits<Packet>::type predux_max(const Packet& a)
{ return a; }

/** \internal \returns the reversed elements of \a a*/
template<typename Packet> EIGEN_DEVICE_FUNC inline Packet preverse(const Packet& a)
{ return a; }

template<size_t offset, typename Packet>
struct protate_impl
{
  // Empty so attempts to use this unimplemented path will fail to compile.
  // Only specializations of this template should be used.
};

/** \internal \returns a packet with the coefficients rotated to the right in little-endian convention,
  * by the given offset, e.g. for offset == 1:
  *     (packet[3], packet[2], packet[1], packet[0]) becomes (packet[0], packet[3], packet[2], packet[1])
  */
template<size_t offset, typename Packet> EIGEN_DEVICE_FUNC inline Packet protate(const Packet& a)
{
  return offset ? protate_impl<offset, Packet>::run(a) : a;
}

/** \internal \returns \a a with real and imaginary part flipped (for complex type only) */
template<typename Packet> EIGEN_DEVICE_FUNC inline Packet pcplxflip(const Packet& a)
{
  // FIXME: uncomment the following in case we drop the internal imag and real functions.
//   using std::imag;
//   using std::real;
  return Packet(imag(a),real(a));
}

/**************************
* Special math functions
***************************/

/** \internal \returns the sine of \a a (coeff-wise) */
template<typename Packet> EIGEN_DECLARE_FUNCTION_ALLOWING_MULTIPLE_DEFINITIONS
Packet psin(const Packet& a) { using std::sin; return sin(a); }

/** \internal \returns the cosine of \a a (coeff-wise) */
template<typename Packet> EIGEN_DECLARE_FUNCTION_ALLOWING_MULTIPLE_DEFINITIONS
Packet pcos(const Packet& a) { using std::cos; return cos(a); }

/** \internal \returns the tan of \a a (coeff-wise) */
template<typename Packet> EIGEN_DECLARE_FUNCTION_ALLOWING_MULTIPLE_DEFINITIONS
Packet ptan(const Packet& a) { using std::tan; return tan(a); }

/** \internal \returns the arc sine of \a a (coeff-wise) */
template<typename Packet> EIGEN_DECLARE_FUNCTION_ALLOWING_MULTIPLE_DEFINITIONS
Packet pasin(const Packet& a) { using std::asin; return asin(a); }

/** \internal \returns the arc cosine of \a a (coeff-wise) */
template<typename Packet> EIGEN_DECLARE_FUNCTION_ALLOWING_MULTIPLE_DEFINITIONS
Packet pacos(const Packet& a) { using std::acos; return acos(a); }

/** \internal \returns the atan of \a a (coeff-wise) */
template<typename Packet> EIGEN_DECLARE_FUNCTION_ALLOWING_MULTIPLE_DEFINITIONS
Packet patan(const Packet& a) { using std::atan; return atan(a); }

/** \internal \returns the exp of \a a (coeff-wise) */
template<typename Packet> EIGEN_DECLARE_FUNCTION_ALLOWING_MULTIPLE_DEFINITIONS
Packet pexp(const Packet& a) { using std::exp; return exp(a); }

/** \internal \returns the log of \a a (coeff-wise) */
template<typename Packet> EIGEN_DECLARE_FUNCTION_ALLOWING_MULTIPLE_DEFINITIONS
Packet plog(const Packet& a) { using std::log; return log(a); }

/** \internal \returns the square-root of \a a (coeff-wise) */
template<typename Packet> EIGEN_DECLARE_FUNCTION_ALLOWING_MULTIPLE_DEFINITIONS
Packet psqrt(const Packet& a) { using std::sqrt; return sqrt(a); }

/** \internal \returns the reciprocal square-root of \a a (coeff-wise) */
template<typename Packet> EIGEN_DECLARE_FUNCTION_ALLOWING_MULTIPLE_DEFINITIONS
Packet prsqrt(const Packet& a) {
  using std::sqrt;
  const Packet one(1);
  return one/sqrt(a);
}

// Default ptanh approximation threshold, assumes single precision
// floating point.
template<typename Packet> Packet ptanh_approx_threshold() {
  return pset1<Packet>(0.01);
}

/** \internal \returns the hyperbolic tan of \a a (coeff-wise) */
template<typename Packet> EIGEN_DECLARE_FUNCTION_ALLOWING_MULTIPLE_DEFINITIONS
Packet ptanh(const Packet& x)
{
  const Packet one = pset1<Packet>(1);
  const Packet two = pset1<Packet>(2);
  const Packet three = pset1<Packet>(3);
  const Packet thresh = ptanh_approx_threshold<Packet>();
  const Packet x2 = pmul(x, x);
  const Packet small_approx = pmul(x, psub(one, pdiv(x2, three)));
  const Packet med_approx = psub(one, pdiv(two, padd(pexp(pmul(two, x)), one)));

  // If |x| > thresh, tanh(x) = 1-2/(exp(2*x) + 1)
  // tanh(x) can be written: x(1 - x^2/3 + ...) for |x| < pi/2
  // Select a thresh s.t. |tanh(x) - x| = O(eps), where for floats,
  // If |x| < thresh, tanh(x) = x*(1-x^2/3)
  // Use theresh = 0.01 as this matches the float32 approximation
  // threshold on my system!
  return pselect(med_approx, small_approx, ple(pabs(x), thresh));
}

/***************************************************************************
* The following functions might not have to be overwritten for vectorized types
***************************************************************************/

/** \internal copy a packet with constant coeficient \a a (e.g., [a,a,a,a]) to \a *to. \a to must be 16 bytes aligned */
// NOTE: this function must really be templated on the packet type (think about different packet types for the same scalar type)
template<typename Packet>
inline void pstore1(typename unpacket_traits<Packet>::type* to, const typename unpacket_traits<Packet>::type& a)
{
  pstore(to, pset1<Packet>(a));
}

/** \internal \returns a * b + c (coeff-wise) */
template<typename Packet> EIGEN_DEVICE_FUNC inline Packet
pmadd(const Packet&  a,
         const Packet&  b,
         const Packet&  c)
{ return padd(pmul(a, b),c); }

/** \internal \returns a packet version of \a *from.
  * If LoadMode equals #Aligned, \a from must be 16 bytes aligned */
template<typename Packet, int LoadMode>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE Packet ploadt(const typename unpacket_traits<Packet>::type* from)
{
  if(LoadMode == Aligned)
    return pload<Packet>(from);
  else
    return ploadu<Packet>(from);
}

/** \internal copy the packet \a from to \a *to.
  * If StoreMode equals #Aligned, \a to must be 16 bytes aligned */
template<typename Scalar, typename Packet, int LoadMode>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE void pstoret(Scalar* to, const Packet& from)
{
  if(LoadMode == Aligned)
    pstore(to, from);
  else
    pstoreu(to, from);
}

/** \internal \returns a packet version of \a *from.
  * Unlike ploadt, ploadt_ro takes advantage of the read-only memory path on the
  * hardware if available to speedup the loading of data that won't be modified
  * by the current computation.
  */
template<typename Packet, int LoadMode>
inline Packet ploadt_ro(const typename unpacket_traits<Packet>::type* from)
{
  return ploadt<Packet, LoadMode>(from);
}

/** \internal default implementation of palign() allowing partial specialization */
template<int Offset,typename PacketType>
struct palign_impl
{
  // by default data are aligned, so there is nothing to be done :)
  static inline void run(PacketType&, const PacketType&) {}
};

/** \internal update \a first using the concatenation of the packet_size minus \a Offset last elements
  * of \a first and \a Offset first elements of \a second.
  *
  * This function is currently only used to optimize matrix-vector products on unligned matrices.
  * It takes 2 packets that represent a contiguous memory array, and returns a packet starting
  * at the position \a Offset. For instance, for packets of 4 elements, we have:
  *  Input:
  *  - first = {f0,f1,f2,f3}
  *  - second = {s0,s1,s2,s3}
  * Output:
  *   - if Offset==0 then {f0,f1,f2,f3}
  *   - if Offset==1 then {f1,f2,f3,s0}
  *   - if Offset==2 then {f2,f3,s0,s1}
  *   - if Offset==3 then {f3,s0,s1,s3}
  */
template<int Offset,typename PacketType>
inline void palign(PacketType& first, const PacketType& second)
{
  palign_impl<Offset,PacketType>::run(first,second);
}

/***************************************************************************
* Fast complex products (GCC generates a function call which is very slow)
***************************************************************************/

// Eigen+CUDA does not support complexes.
#ifndef __CUDACC__

template<> inline std::complex<float> pmul(const std::complex<float>& a, const std::complex<float>& b)
{ return std::complex<float>(real(a)*real(b) - imag(a)*imag(b), imag(a)*real(b) + real(a)*imag(b)); }

template<> inline std::complex<double> pmul(const std::complex<double>& a, const std::complex<double>& b)
{ return std::complex<double>(real(a)*real(b) - imag(a)*imag(b), imag(a)*real(b) + real(a)*imag(b)); }

#endif


/***************************************************************************
 * PacketBlock, that is a collection of N packets where the number of words
 * in the packet is a multiple of N.
***************************************************************************/
template <typename Packet,int N=unpacket_traits<Packet>::size> struct PacketBlock {
  Packet packet[N];
};

template<typename SquarePacketBlock> EIGEN_DEVICE_FUNC inline void
ptranspose(SquarePacketBlock& /*kernel*/) {
  // Nothing to do in the scalar case, i.e. a 1x1 matrix.
}


/***************************************************************************
 * Selector, i.e. vector of N boolean values used to select (i.e. blend)
 * words from 2 packets.
***************************************************************************/
template <size_t N> struct Selector {
  bool select[N];
};

template<typename Packet> EIGEN_DEVICE_FUNC inline Packet
pblend(const Selector<unpacket_traits<Packet>::size>& ifPacket, const Packet& thenPacket, const Packet& elsePacket) {
  return ifPacket.select[0] ? thenPacket : elsePacket;
}

} // end namespace internal

} // end namespace Eigen

#endif // EIGEN_GENERIC_PACKET_MATH_H
