// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008-2009 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_GENERAL_BLOCK_PANEL_H
#define EIGEN_GENERAL_BLOCK_PANEL_H


namespace Eigen {

namespace internal {

template<typename _LhsScalar, typename _RhsScalar, bool _ConjLhs=false, bool _ConjRhs=false>
class gebp_traits;


/** \internal \returns b if a<=0, and returns a otherwise. */
inline std::ptrdiff_t manage_caching_sizes_helper(std::ptrdiff_t a, std::ptrdiff_t b)
{
  return a<=0 ? b : a;
}

#if EIGEN_ARCH_i386_OR_x86_64
const std::ptrdiff_t defaultL1CacheSize = 32*1024;
const std::ptrdiff_t defaultL2CacheSize = 256*1024;
const std::ptrdiff_t defaultL3CacheSize = 2*1024*1024;
#else
const std::ptrdiff_t defaultL1CacheSize = 16*1024;
const std::ptrdiff_t defaultL2CacheSize = 512*1024;
const std::ptrdiff_t defaultL3CacheSize = 512*1024;
#endif

/** \internal */
inline void manage_caching_sizes(Action action, std::ptrdiff_t* l1, std::ptrdiff_t* l2, std::ptrdiff_t* l3)
{
  static bool m_cache_sizes_initialized = false;
  static std::ptrdiff_t m_l1CacheSize = 0;
  static std::ptrdiff_t m_l2CacheSize = 0;
  static std::ptrdiff_t m_l3CacheSize = 0;

  if(EIGEN_UNLIKELY(!m_cache_sizes_initialized))
  {
    int l1CacheSize, l2CacheSize, l3CacheSize;
    queryCacheSizes(l1CacheSize, l2CacheSize, l3CacheSize);
    m_l1CacheSize = manage_caching_sizes_helper(l1CacheSize, defaultL1CacheSize);
    m_l2CacheSize = manage_caching_sizes_helper(l2CacheSize, defaultL2CacheSize);
    m_l3CacheSize = manage_caching_sizes_helper(l3CacheSize, defaultL3CacheSize);
    m_cache_sizes_initialized = true;
  }

  if(EIGEN_UNLIKELY(action==SetAction))
  {
    // set the cpu cache size and cache all block sizes from a global cache size in byte
    eigen_internal_assert(l1!=0 && l2!=0);
    m_l1CacheSize = *l1;
    m_l2CacheSize = *l2;
    m_l3CacheSize = *l3;
  }
  else if(EIGEN_LIKELY(action==GetAction))
  {
    eigen_internal_assert(l1!=0 && l2!=0);
    *l1 = m_l1CacheSize;
    *l2 = m_l2CacheSize;
    *l3 = m_l3CacheSize;
  }
  else
  {
    eigen_internal_assert(false);
  }
}

#define CEIL(a, b) ((a)+(b)-1)/(b)

/* Helper for computeProductBlockingSizes.
 *
 * Given a m x k times k x n matrix product of scalar types \c LhsScalar and \c RhsScalar,
 * this function computes the blocking size parameters along the respective dimensions
 * for matrix products and related algorithms. The blocking sizes depends on various
 * parameters:
 * - the L1 and L2 cache sizes,
 * - the register level blocking sizes defined by gebp_traits,
 * - the number of scalars that fit into a packet (when vectorization is enabled).
 *
 * \sa setCpuCacheSizes */
template<typename LhsScalar, typename RhsScalar, int KcFactor, typename Index>
void evaluateProductBlockingSizesHeuristic(Index& k, Index& m, Index& n, Index num_threads = 1)
{
  // Explanations:
  // Let's recall the product algorithms form kc x nc horizontal panels B' on the rhs and
  // mc x kc blocks A' on the lhs. A' has to fit into L2 cache. Moreover, B' is processed
  // per kc x nr vertical small panels where nr is the blocking size along the n dimension
  // at the register level. For vectorization purpose, these small vertical panels are unpacked,
  // e.g., each coefficient is replicated to fit a packet. This small vertical panel has to
  // stay in L1 cache.
  typedef gebp_traits<LhsScalar,RhsScalar> Traits;
  typedef typename Traits::ResScalar ResScalar;
  enum {
    kdiv = KcFactor * (Traits::mr * sizeof(LhsScalar) + Traits::nr * sizeof(RhsScalar)),
    ksub = Traits::mr * Traits::nr * sizeof(ResScalar),
    k_mask = (0xffffffff/8)*8,

    mr = Traits::mr,
    mr_mask = (0xffffffff/mr)*mr,

    nr = Traits::nr,
    nr_mask = (0xffffffff/nr)*nr
  };

  std::ptrdiff_t l1, l2, l3;
  manage_caching_sizes(GetAction, &l1, &l2, &l3);

  // Increasing k gives us more time to prefetch the content of the "C"
  // registers. However once the latency is hidden there is no point in
  // increasing the value of k, so we'll cap it at 320 (value determined
  // experimentally).
  const Index k_cache = (std::min<Index>)((l1-ksub)/kdiv, 320);
  if (k_cache < k) {
    k = k_cache & k_mask;
    eigen_assert(k > 0);
  }

  const Index n_cache = (l2-l1) / (nr * sizeof(RhsScalar) * k);
  Index n_per_thread = CEIL(n, num_threads);
  if (n_cache <= n_per_thread) {
    // Don't exceed the capacity of the l2 cache.
    if (n_cache < nr) {
      n = nr;
    } else {
      n = n_cache & nr_mask;
      eigen_assert(n > 0);
    }
  } else {
    n = (std::min<Index>)(n, (n_per_thread + nr - 1) & nr_mask);
  }

  if (l3 > l2) {
    // l3 is shared between all cores, so we'll give each thread its own chunk of l3.
    const Index m_cache = (l3-l2) / (sizeof(LhsScalar) * k * num_threads);
    const Index m_per_thread = CEIL(m, num_threads);
    if(m_cache < m_per_thread && m_cache >= static_cast<Index>(mr)) {
      m = m_cache & mr_mask;
      eigen_assert(m > 0);
    } else {
      m = (std::min<Index>)(m, (m_per_thread + mr - 1) & mr_mask);
    }
  }
}

template <typename Index>
bool useSpecificBlockingSizes(Index& k, Index& m, Index& n)
{
#ifdef EIGEN_TEST_SPECIFIC_BLOCKING_SIZES
  if (EIGEN_TEST_SPECIFIC_BLOCKING_SIZES) {
    k = std::min<Index>(k, EIGEN_TEST_SPECIFIC_BLOCKING_SIZE_K);
    m = std::min<Index>(m, EIGEN_TEST_SPECIFIC_BLOCKING_SIZE_M);
    n = std::min<Index>(n, EIGEN_TEST_SPECIFIC_BLOCKING_SIZE_N);
    return true;
  }
#else
  EIGEN_UNUSED_VARIABLE(k)
  EIGEN_UNUSED_VARIABLE(m)
  EIGEN_UNUSED_VARIABLE(n)
#endif
  return false;
}

/** \brief Computes the blocking parameters for a m x k times k x n matrix product
  *
  * \param[in,out] k Input: the third dimension of the product. Output: the blocking size along the same dimension.
  * \param[in,out] m Input: the number of rows of the left hand side. Output: the blocking size along the same dimension.
  * \param[in,out] n Input: the number of columns of the right hand side. Output: the blocking size along the same dimension.
  *
  * Given a m x k times k x n matrix product of scalar types \c LhsScalar and \c RhsScalar,
  * this function computes the blocking size parameters along the respective dimensions
  * for matrix products and related algorithms.
  *
  * The blocking size parameters may be evaluated:
  *   - either by a heuristic based on cache sizes;
  *   - or using fixed prescribed values (for testing purposes).
  *
  * \sa setCpuCacheSizes */

template<typename LhsScalar, typename RhsScalar, int KcFactor, typename Index>
void computeProductBlockingSizes(Index& k, Index& m, Index& n, Index num_threads = 1)
{
  if (!k || !m || !n) {
    return;
  }

  if (!useSpecificBlockingSizes(k, m, n)) {
    evaluateProductBlockingSizesHeuristic<LhsScalar, RhsScalar, KcFactor>(k, m, n, num_threads);
  }

#if !EIGEN_ARCH_i386_OR_x86_64
  // The following code rounds k,m,n down to the nearest multiple of register-level blocking sizes.
  // We should always do that, and in upstream Eigen we always do that.
  // Unfortunately, we can't do that in Google3 on x86[-64] because this makes tiny differences in results and
  // we have some unfortunate tests require very specific relative errors which fail because of that,
  // at least //learning/laser/algorithms/wals:wals_batch_solver_test.
  // Note that this wouldn't make any difference if we had been using only correctly rounded values,
  // but we've not! See how in evaluateProductBlockingSizesHeuristic, we do the rounding down by
  // bit-masking, e.g. mr_mask = (0xffffffff/mr)*mr, implicitly assuming that mr is always a power of
  // two, which is not the case with the 3px4 kernel.
  typedef gebp_traits<LhsScalar,RhsScalar> Traits;
  enum {
    kr = 8,
    mr = Traits::mr,
    nr = Traits::nr
  };
  if (k > kr) k -= k % kr;
  if (m > mr) m -= m % mr;
  if (n > nr) n -= n % nr;
#endif
}

template<typename LhsScalar, typename RhsScalar, typename Index>
inline void computeProductBlockingSizes(Index& k, Index& m, Index& n, Index num_threads)
{
  computeProductBlockingSizes<LhsScalar,RhsScalar,1>(k, m, n, num_threads);
}

#ifdef EIGEN_HAS_SINGLE_INSTRUCTION_CJMADD
  #define CJMADD(CJ,A,B,C,T)  C = CJ.pmadd(A,B,C);
#else

  // FIXME (a bit overkill maybe ?)

  template<typename CJ, typename A, typename B, typename C, typename T> struct gebp_madd_selector {
    EIGEN_ALWAYS_INLINE static void run(const CJ& cj, A& a, B& b, C& c, T& /*t*/)
    {
      c = cj.pmadd(a,b,c);
    }
  };

  template<typename CJ, typename T> struct gebp_madd_selector<CJ,T,T,T,T> {
    EIGEN_ALWAYS_INLINE static void run(const CJ& cj, T& a, T& b, T& c, T& t)
    {
      t = b; t = cj.pmul(a,t); c = padd(c,t);
    }
  };

  template<typename CJ, typename A, typename B, typename C, typename T>
  EIGEN_STRONG_INLINE void gebp_madd(const CJ& cj, A& a, B& b, C& c, T& t)
  {
    gebp_madd_selector<CJ,A,B,C,T>::run(cj,a,b,c,t);
  }

  #define CJMADD(CJ,A,B,C,T)  gebp_madd(CJ,A,B,C,T);
//   #define CJMADD(CJ,A,B,C,T)  T = B; T = CJ.pmul(A,T); C = padd(C,T);
#endif

/* Vectorization logic
 *  real*real: unpack rhs to constant packets, ...
 *
 *  cd*cd : unpack rhs to (b_r,b_r), (b_i,b_i), mul to get (a_r b_r,a_i b_r) (a_r b_i,a_i b_i),
 *          storing each res packet into two packets (2x2),
 *          at the end combine them: swap the second and addsub them
 *  cf*cf : same but with 2x4 blocks
 *  cplx*real : unpack rhs to constant packets, ...
 *  real*cplx : load lhs as (a0,a0,a1,a1), and mul as usual
 */
template<typename _LhsScalar, typename _RhsScalar, bool _ConjLhs, bool _ConjRhs>
class gebp_traits
{
public:
  typedef _LhsScalar LhsScalar;
  typedef _RhsScalar RhsScalar;
  typedef typename scalar_product_traits<LhsScalar, RhsScalar>::ReturnType ResScalar;

  enum {
    ConjLhs = _ConjLhs,
    ConjRhs = _ConjRhs,
    Vectorizable = packet_traits<LhsScalar>::Vectorizable && packet_traits<RhsScalar>::Vectorizable,
    LhsPacketSize = Vectorizable ? packet_traits<LhsScalar>::size : 1,
    RhsPacketSize = Vectorizable ? packet_traits<RhsScalar>::size : 1,
    ResPacketSize = Vectorizable ? packet_traits<ResScalar>::size : 1,

    NumberOfRegisters = EIGEN_ARCH_DEFAULT_NUMBER_OF_REGISTERS,

    // register block size along the N direction must be 1 or 4
    nr = 4,

    // register block size along the M direction (currently, this one cannot be modified)
    default_mr = (EIGEN_PLAIN_ENUM_MIN(16,NumberOfRegisters)/2/nr)*LhsPacketSize,
#if defined(EIGEN_HAS_SINGLE_INSTRUCTION_MADD) && !defined(EIGEN_VECTORIZE_ALTIVEC) && !defined(EIGEN_VECTORIZE_VSX)
    // we assume 16 registers
    mr = Vectorizable ? 3*LhsPacketSize : default_mr,
#else
    mr = default_mr,
#endif

    LhsProgress = LhsPacketSize,
    RhsProgress = 1
  };

  typedef typename packet_traits<LhsScalar>::type  _LhsPacket;
  typedef typename packet_traits<RhsScalar>::type  _RhsPacket;
  typedef typename packet_traits<ResScalar>::type  _ResPacket;

  typedef typename conditional<Vectorizable,_LhsPacket,LhsScalar>::type LhsPacket;
  typedef typename conditional<Vectorizable,_RhsPacket,RhsScalar>::type RhsPacket;
  typedef typename conditional<Vectorizable,_ResPacket,ResScalar>::type ResPacket;

  typedef ResPacket AccPacket;

  EIGEN_STRONG_INLINE void initAcc(AccPacket& p)
  {
    p = pset1<ResPacket>(ResScalar(0));
  }

  EIGEN_STRONG_INLINE void broadcastRhs(const RhsScalar* b, RhsPacket& b0, RhsPacket& b1, RhsPacket& b2, RhsPacket& b3)
  {
    pbroadcast4(b, b0, b1, b2, b3);
  }

//   EIGEN_STRONG_INLINE void broadcastRhs(const RhsScalar* b, RhsPacket& b0, RhsPacket& b1)
//   {
//     pbroadcast2(b, b0, b1);
//   }

  template<typename RhsPacketType>
  EIGEN_STRONG_INLINE void loadRhs(const RhsScalar* b, RhsPacketType& dest) const
  {
    dest = pset1<RhsPacketType>(*b);
  }

  EIGEN_STRONG_INLINE void loadRhsQuad(const RhsScalar* b, RhsPacket& dest) const
  {
    dest = ploadquad<RhsPacket>(b);
  }

  template<typename LhsPacketType>
  EIGEN_STRONG_INLINE void loadLhs(const LhsScalar* a, LhsPacketType& dest) const
  {
    dest = pload<LhsPacketType>(a);
  }

  template<typename LhsPacketType>
  EIGEN_STRONG_INLINE void loadLhsUnaligned(const LhsScalar* a, LhsPacketType& dest) const
  {
    dest = ploadu<LhsPacketType>(a);
  }

  template<typename LhsPacketType, typename RhsPacketType, typename AccPacketType>
  EIGEN_STRONG_INLINE void madd(const LhsPacketType& a, const RhsPacketType& b, AccPacketType& c, AccPacketType& tmp) const
  {
    // It would be a lot cleaner to call pmadd all the time. Unfortunately if we
    // let gcc allocate the register in which to store the result of the pmul
    // (in the case where there is no FMA) gcc fails to figure out how to avoid
    // spilling register.
#ifdef EIGEN_HAS_SINGLE_INSTRUCTION_MADD
    EIGEN_UNUSED_VARIABLE(tmp);
    c = pmadd(a,b,c);
#else
    tmp = b; tmp = pmul(a,tmp); c = padd(c,tmp);
#endif
  }

  EIGEN_STRONG_INLINE void acc(const AccPacket& c, const ResPacket& alpha, ResPacket& r) const
  {
    r = pmadd(c,alpha,r);
  }

  template<typename ResPacketHalf>
  EIGEN_STRONG_INLINE void acc(const ResPacketHalf& c, const ResPacketHalf& alpha, ResPacketHalf& r) const
  {
    r = pmadd(c,alpha,r);
  }

protected:
//   conj_helper<LhsScalar,RhsScalar,ConjLhs,ConjRhs> cj;
//   conj_helper<LhsPacket,RhsPacket,ConjLhs,ConjRhs> pcj;
};

template<typename RealScalar, bool _ConjLhs>
class gebp_traits<std::complex<RealScalar>, RealScalar, _ConjLhs, false>
{
public:
  typedef std::complex<RealScalar> LhsScalar;
  typedef RealScalar RhsScalar;
  typedef typename scalar_product_traits<LhsScalar, RhsScalar>::ReturnType ResScalar;

  enum {
    ConjLhs = _ConjLhs,
    ConjRhs = false,
    Vectorizable = packet_traits<LhsScalar>::Vectorizable && packet_traits<RhsScalar>::Vectorizable,
    LhsPacketSize = Vectorizable ? packet_traits<LhsScalar>::size : 1,
    RhsPacketSize = Vectorizable ? packet_traits<RhsScalar>::size : 1,
    ResPacketSize = Vectorizable ? packet_traits<ResScalar>::size : 1,

    NumberOfRegisters = EIGEN_ARCH_DEFAULT_NUMBER_OF_REGISTERS,
    nr = 4,
#if defined(EIGEN_HAS_SINGLE_INSTRUCTION_MADD) && !defined(EIGEN_VECTORIZE_ALTIVEC) && !defined(EIGEN_VECTORIZE_VSX)
    // we assume 16 registers
    mr = 3*LhsPacketSize,
#else
    mr = (EIGEN_PLAIN_ENUM_MIN(16,NumberOfRegisters)/2/nr)*LhsPacketSize,
#endif

    LhsProgress = LhsPacketSize,
    RhsProgress = 1
  };

  typedef typename packet_traits<LhsScalar>::type  _LhsPacket;
  typedef typename packet_traits<RhsScalar>::type  _RhsPacket;
  typedef typename packet_traits<ResScalar>::type  _ResPacket;

  typedef typename conditional<Vectorizable,_LhsPacket,LhsScalar>::type LhsPacket;
  typedef typename conditional<Vectorizable,_RhsPacket,RhsScalar>::type RhsPacket;
  typedef typename conditional<Vectorizable,_ResPacket,ResScalar>::type ResPacket;

  typedef ResPacket AccPacket;

  EIGEN_STRONG_INLINE void initAcc(AccPacket& p)
  {
    p = pset1<ResPacket>(ResScalar(0));
  }

  EIGEN_STRONG_INLINE void loadRhs(const RhsScalar* b, RhsPacket& dest) const
  {
    dest = pset1<RhsPacket>(*b);
  }

  EIGEN_STRONG_INLINE void loadRhsQuad(const RhsScalar* b, RhsPacket& dest) const
  {
    dest = pset1<RhsPacket>(*b);
  }

  EIGEN_STRONG_INLINE void loadLhs(const LhsScalar* a, LhsPacket& dest) const
  {
    dest = pload<LhsPacket>(a);
  }

  EIGEN_STRONG_INLINE void loadLhsUnaligned(const LhsScalar* a, LhsPacket& dest) const
  {
    dest = ploadu<LhsPacket>(a);
  }

  EIGEN_STRONG_INLINE void broadcastRhs(const RhsScalar* b, RhsPacket& b0, RhsPacket& b1, RhsPacket& b2, RhsPacket& b3)
  {
    pbroadcast4(b, b0, b1, b2, b3);
  }

//   EIGEN_STRONG_INLINE void broadcastRhs(const RhsScalar* b, RhsPacket& b0, RhsPacket& b1)
//   {
//     pbroadcast2(b, b0, b1);
//   }

  EIGEN_STRONG_INLINE void madd(const LhsPacket& a, const RhsPacket& b, AccPacket& c, RhsPacket& tmp) const
  {
    madd_impl(a, b, c, tmp, typename conditional<Vectorizable,true_type,false_type>::type());
  }

  EIGEN_STRONG_INLINE void madd_impl(const LhsPacket& a, const RhsPacket& b, AccPacket& c, RhsPacket& tmp, const true_type&) const
  {
#ifdef EIGEN_HAS_SINGLE_INSTRUCTION_MADD
    EIGEN_UNUSED_VARIABLE(tmp);
    c.v = pmadd(a.v,b,c.v);
#else
    tmp = b; tmp = pmul(a.v,tmp); c.v = padd(c.v,tmp);
#endif
  }

  EIGEN_STRONG_INLINE void madd_impl(const LhsScalar& a, const RhsScalar& b, ResScalar& c, RhsScalar& /*tmp*/, const false_type&) const
  {
    c += a * b;
  }

  EIGEN_STRONG_INLINE void acc(const AccPacket& c, const ResPacket& alpha, ResPacket& r) const
  {
    r = cj.pmadd(c,alpha,r);
  }

protected:
  conj_helper<ResPacket,ResPacket,ConjLhs,false> cj;
};

template<typename Packet>
struct DoublePacket
{
  Packet first;
  Packet second;
};

template<typename Packet>
DoublePacket<Packet> padd(const DoublePacket<Packet> &a, const DoublePacket<Packet> &b)
{
  DoublePacket<Packet> res;
  res.first  = padd(a.first, b.first);
  res.second = padd(a.second,b.second);
  return res;
}

template<typename Packet>
const DoublePacket<Packet>& predux4(const DoublePacket<Packet> &a)
{
  return a;
}

template<typename Packet> struct unpacket_traits<DoublePacket<Packet> > { typedef DoublePacket<Packet> half; };
// template<typename Packet>
// DoublePacket<Packet> pmadd(const DoublePacket<Packet> &a, const DoublePacket<Packet> &b)
// {
//   DoublePacket<Packet> res;
//   res.first  = padd(a.first, b.first);
//   res.second = padd(a.second,b.second);
//   return res;
// }

template<typename RealScalar, bool _ConjLhs, bool _ConjRhs>
class gebp_traits<std::complex<RealScalar>, std::complex<RealScalar>, _ConjLhs, _ConjRhs >
{
public:
  typedef std::complex<RealScalar>  Scalar;
  typedef std::complex<RealScalar>  LhsScalar;
  typedef std::complex<RealScalar>  RhsScalar;
  typedef std::complex<RealScalar>  ResScalar;

  enum {
    ConjLhs = _ConjLhs,
    ConjRhs = _ConjRhs,
    Vectorizable = packet_traits<RealScalar>::Vectorizable
                && packet_traits<Scalar>::Vectorizable,
    RealPacketSize  = Vectorizable ? packet_traits<RealScalar>::size : 1,
    ResPacketSize   = Vectorizable ? packet_traits<ResScalar>::size : 1,
    LhsPacketSize = Vectorizable ? packet_traits<LhsScalar>::size : 1,
    RhsPacketSize = Vectorizable ? packet_traits<RhsScalar>::size : 1,

    // FIXME: should depend on NumberOfRegisters
    nr = 4,
    mr = ResPacketSize,

    LhsProgress = ResPacketSize,
    RhsProgress = 1
  };

  typedef typename packet_traits<RealScalar>::type RealPacket;
  typedef typename packet_traits<Scalar>::type     ScalarPacket;
  typedef DoublePacket<RealPacket> DoublePacketType;

  typedef typename conditional<Vectorizable,RealPacket,  Scalar>::type LhsPacket;
  typedef typename conditional<Vectorizable,DoublePacketType,Scalar>::type RhsPacket;
  typedef typename conditional<Vectorizable,ScalarPacket,Scalar>::type ResPacket;
  typedef typename conditional<Vectorizable,DoublePacketType,Scalar>::type AccPacket;

  EIGEN_STRONG_INLINE void initAcc(Scalar& p) { p = Scalar(0); }

  EIGEN_STRONG_INLINE void initAcc(DoublePacketType& p)
  {
    p.first   = pset1<RealPacket>(RealScalar(0));
    p.second  = pset1<RealPacket>(RealScalar(0));
  }

  // Scalar path
  EIGEN_STRONG_INLINE void loadRhs(const RhsScalar* b, ResPacket& dest) const
  {
    dest = pset1<ResPacket>(*b);
  }

  // Vectorized path
  EIGEN_STRONG_INLINE void loadRhs(const RhsScalar* b, DoublePacketType& dest) const
  {
    dest.first  = pset1<RealPacket>(real(*b));
    dest.second = pset1<RealPacket>(imag(*b));
  }

  EIGEN_STRONG_INLINE void loadRhsQuad(const RhsScalar* b, ResPacket& dest) const
  {
    loadRhs(b,dest);
  }
  EIGEN_STRONG_INLINE void loadRhsQuad(const RhsScalar* b, DoublePacketType& dest) const
  {
    eigen_internal_assert(unpacket_traits<ScalarPacket>::size<=4);
    loadRhs(b,dest);
  }

  EIGEN_STRONG_INLINE void broadcastRhs(const RhsScalar* b, RhsPacket& b0, RhsPacket& b1, RhsPacket& b2, RhsPacket& b3)
  {
    // FIXME not sure that's the best way to implement it!
    loadRhs(b+0, b0);
    loadRhs(b+1, b1);
    loadRhs(b+2, b2);
    loadRhs(b+3, b3);
  }

  // Vectorized path
  EIGEN_STRONG_INLINE void broadcastRhs(const RhsScalar* b, DoublePacketType& b0, DoublePacketType& b1)
  {
    // FIXME not sure that's the best way to implement it!
    loadRhs(b+0, b0);
    loadRhs(b+1, b1);
  }

  // Scalar path
  EIGEN_STRONG_INLINE void broadcastRhs(const RhsScalar* b, RhsScalar& b0, RhsScalar& b1)
  {
    // FIXME not sure that's the best way to implement it!
    loadRhs(b+0, b0);
    loadRhs(b+1, b1);
  }

  // nothing special here
  EIGEN_STRONG_INLINE void loadLhs(const LhsScalar* a, LhsPacket& dest) const
  {
    dest = pload<LhsPacket>((const typename unpacket_traits<LhsPacket>::type*)(a));
  }

  EIGEN_STRONG_INLINE void loadLhsUnaligned(const LhsScalar* a, LhsPacket& dest) const
  {
    dest = ploadu<LhsPacket>((const typename unpacket_traits<LhsPacket>::type*)(a));
  }

  EIGEN_STRONG_INLINE void madd(const LhsPacket& a, const RhsPacket& b, DoublePacketType& c, RhsPacket& /*tmp*/) const
  {
    c.first   = padd(pmul(a,b.first), c.first);
    c.second  = padd(pmul(a,b.second),c.second);
  }

  EIGEN_STRONG_INLINE void madd(const LhsPacket& a, const RhsPacket& b, ResPacket& c, RhsPacket& /*tmp*/) const
  {
    c = cj.pmadd(a,b,c);
  }

  EIGEN_STRONG_INLINE void acc(const Scalar& c, const Scalar& alpha, Scalar& r) const { r += alpha * c; }

  EIGEN_STRONG_INLINE void acc(const DoublePacketType& c, const ResPacket& alpha, ResPacket& r) const
  {
    // assemble c
    ResPacket tmp;
    if((!ConjLhs)&&(!ConjRhs))
    {
      tmp = pcplxflip(pconj(ResPacket(c.second)));
      tmp = padd(ResPacket(c.first),tmp);
    }
    else if((!ConjLhs)&&(ConjRhs))
    {
      tmp = pconj(pcplxflip(ResPacket(c.second)));
      tmp = padd(ResPacket(c.first),tmp);
    }
    else if((ConjLhs)&&(!ConjRhs))
    {
      tmp = pcplxflip(ResPacket(c.second));
      tmp = padd(pconj(ResPacket(c.first)),tmp);
    }
    else if((ConjLhs)&&(ConjRhs))
    {
      tmp = pcplxflip(ResPacket(c.second));
      tmp = psub(pconj(ResPacket(c.first)),tmp);
    }

    r = pmadd(tmp,alpha,r);
  }

protected:
  conj_helper<LhsScalar,RhsScalar,ConjLhs,ConjRhs> cj;
};

template<typename RealScalar, bool _ConjRhs>
class gebp_traits<RealScalar, std::complex<RealScalar>, false, _ConjRhs >
{
public:
  typedef std::complex<RealScalar>  Scalar;
  typedef RealScalar  LhsScalar;
  typedef Scalar      RhsScalar;
  typedef Scalar      ResScalar;

  enum {
    ConjLhs = false,
    ConjRhs = _ConjRhs,
    Vectorizable = packet_traits<RealScalar>::Vectorizable
                && packet_traits<Scalar>::Vectorizable,
    LhsPacketSize = Vectorizable ? packet_traits<LhsScalar>::size : 1,
    RhsPacketSize = Vectorizable ? packet_traits<RhsScalar>::size : 1,
    ResPacketSize = Vectorizable ? packet_traits<ResScalar>::size : 1,

    NumberOfRegisters = EIGEN_ARCH_DEFAULT_NUMBER_OF_REGISTERS,
    // FIXME: should depend on NumberOfRegisters
    nr = 4,
    mr = (EIGEN_PLAIN_ENUM_MIN(16,NumberOfRegisters)/2/nr)*ResPacketSize,

    LhsProgress = ResPacketSize,
    RhsProgress = 1
  };

  typedef typename packet_traits<LhsScalar>::type  _LhsPacket;
  typedef typename packet_traits<RhsScalar>::type  _RhsPacket;
  typedef typename packet_traits<ResScalar>::type  _ResPacket;

  typedef typename conditional<Vectorizable,_LhsPacket,LhsScalar>::type LhsPacket;
  typedef typename conditional<Vectorizable,_RhsPacket,RhsScalar>::type RhsPacket;
  typedef typename conditional<Vectorizable,_ResPacket,ResScalar>::type ResPacket;

  typedef ResPacket AccPacket;

  EIGEN_STRONG_INLINE void initAcc(AccPacket& p)
  {
    p = pset1<ResPacket>(ResScalar(0));
  }

  EIGEN_STRONG_INLINE void loadRhs(const RhsScalar* b, RhsPacket& dest) const
  {
    dest = pset1<RhsPacket>(*b);
  }

  void broadcastRhs(const RhsScalar* b, RhsPacket& b0, RhsPacket& b1, RhsPacket& b2, RhsPacket& b3)
  {
    pbroadcast4(b, b0, b1, b2, b3);
  }

//   EIGEN_STRONG_INLINE void broadcastRhs(const RhsScalar* b, RhsPacket& b0, RhsPacket& b1)
//   {
//     // FIXME not sure that's the best way to implement it!
//     b0 = pload1<RhsPacket>(b+0);
//     b1 = pload1<RhsPacket>(b+1);
//   }

  EIGEN_STRONG_INLINE void loadLhs(const LhsScalar* a, LhsPacket& dest) const
  {
    dest = ploaddup<LhsPacket>(a);
  }

  EIGEN_STRONG_INLINE void loadRhsQuad(const RhsScalar* b, RhsPacket& dest) const
  {
    eigen_internal_assert(unpacket_traits<RhsPacket>::size<=4);
    loadRhs(b,dest);
  }

  EIGEN_STRONG_INLINE void loadLhsUnaligned(const LhsScalar* a, LhsPacket& dest) const
  {
    dest = ploaddup<LhsPacket>(a);
  }

  EIGEN_STRONG_INLINE void madd(const LhsPacket& a, const RhsPacket& b, AccPacket& c, RhsPacket& tmp) const
  {
    madd_impl(a, b, c, tmp, typename conditional<Vectorizable,true_type,false_type>::type());
  }

  EIGEN_STRONG_INLINE void madd_impl(const LhsPacket& a, const RhsPacket& b, AccPacket& c, RhsPacket& tmp, const true_type&) const
  {
#ifdef EIGEN_HAS_SINGLE_INSTRUCTION_MADD
    EIGEN_UNUSED_VARIABLE(tmp);
    c.v = pmadd(a,b.v,c.v);
#else
    tmp = b; tmp.v = pmul(a,tmp.v); c = padd(c,tmp);
#endif

  }

  EIGEN_STRONG_INLINE void madd_impl(const LhsScalar& a, const RhsScalar& b, ResScalar& c, RhsScalar& /*tmp*/, const false_type&) const
  {
    c += a * b;
  }

  EIGEN_STRONG_INLINE void acc(const AccPacket& c, const ResPacket& alpha, ResPacket& r) const
  {
    r = cj.pmadd(alpha,c,r);
  }

protected:
  conj_helper<ResPacket,ResPacket,false,ConjRhs> cj;
};

// helper for the rotating kernel below
template <typename GebpKernel, bool UseRotatingKernel = GebpKernel::UseRotatingKernel>
struct PossiblyRotatingKernelHelper
{
  // default implementation, not rotating

  typedef typename GebpKernel::Traits Traits;
  typedef typename Traits::RhsScalar RhsScalar;
  typedef typename Traits::RhsPacket RhsPacket;
  typedef typename Traits::AccPacket AccPacket;

  const Traits& traits;
  EIGEN_ALWAYS_INLINE PossiblyRotatingKernelHelper(const Traits& t) : traits(t) {}


  template <size_t K, size_t Index> EIGEN_ALWAYS_INLINE
  void loadOrRotateRhs(RhsPacket& to, const RhsScalar* from) const
  {
    traits.loadRhs(from + (Index+4*K)*Traits::RhsProgress, to);
  }

  EIGEN_ALWAYS_INLINE void unrotateResult(AccPacket&,
                                          AccPacket&,
                                          AccPacket&,
                                          AccPacket&)
  {
  }
};

// rotating implementation
template <typename GebpKernel>
struct PossiblyRotatingKernelHelper<GebpKernel, true>
{
  typedef typename GebpKernel::Traits Traits;
  typedef typename Traits::RhsScalar RhsScalar;
  typedef typename Traits::RhsPacket RhsPacket;
  typedef typename Traits::AccPacket AccPacket;

  const Traits& traits;
  EIGEN_ALWAYS_INLINE PossiblyRotatingKernelHelper(const Traits& t) : traits(t) {}

  template <size_t K, size_t Index> EIGEN_ALWAYS_INLINE
  void loadOrRotateRhs(RhsPacket& to, const RhsScalar* from) const
  {
    if (Index == 0) {
      to = pload<RhsPacket>(from + 4*K*Traits::RhsProgress);
    } else {
      EIGEN_ASM_COMMENT("Do not reorder code, we're very tight on registers");
      to = protate<1>(to);
    }
  }

  EIGEN_ALWAYS_INLINE void unrotateResult(AccPacket& res0,
                                          AccPacket& res1,
                                          AccPacket& res2,
                                          AccPacket& res3)
  {
    PacketBlock<AccPacket> resblock;
    resblock.packet[0] = res0;
    resblock.packet[1] = res1;
    resblock.packet[2] = res2;
    resblock.packet[3] = res3;
    ptranspose(resblock);
    resblock.packet[3] = protate<1>(resblock.packet[3]);
    resblock.packet[2] = protate<2>(resblock.packet[2]);
    resblock.packet[1] = protate<3>(resblock.packet[1]);
    ptranspose(resblock);
    res0 = resblock.packet[0];
    res1 = resblock.packet[1];
    res2 = resblock.packet[2];
    res3 = resblock.packet[3];
  }
};

/* optimized GEneral packed Block * packed Panel product kernel
 *
 * Mixing type logic: C += A * B
 *  |  A  |  B  | comments
 *  |real |cplx | no vectorization yet, would require to pack A with duplication
 *  |cplx |real | easy vectorization
 */
template<typename LhsScalar, typename RhsScalar, typename Index, typename DataMapper, int mr, int nr, bool ConjugateLhs, bool ConjugateRhs>
struct gebp_kernel
{
  typedef gebp_traits<LhsScalar,RhsScalar,ConjugateLhs,ConjugateRhs> Traits;
  typedef typename Traits::ResScalar ResScalar;
  typedef typename Traits::LhsPacket LhsPacket;
  typedef typename Traits::RhsPacket RhsPacket;
  typedef typename Traits::ResPacket ResPacket;
  typedef typename Traits::AccPacket AccPacket;

  typedef gebp_traits<RhsScalar,LhsScalar,ConjugateRhs,ConjugateLhs> SwappedTraits;
  typedef typename SwappedTraits::ResScalar SResScalar;
  typedef typename SwappedTraits::LhsPacket SLhsPacket;
  typedef typename SwappedTraits::RhsPacket SRhsPacket;
  typedef typename SwappedTraits::ResPacket SResPacket;
  typedef typename SwappedTraits::AccPacket SAccPacket;

  typedef typename DataMapper::LinearMapper LinearMapper;

  enum {
    Vectorizable  = Traits::Vectorizable,
    LhsProgress   = Traits::LhsProgress,
    RhsProgress   = Traits::RhsProgress,
    ResPacketSize = Traits::ResPacketSize
  };

  EIGEN_DONT_INLINE
  void operator()(const DataMapper& res, const LhsScalar* blockA, const RhsScalar* blockB,
                  Index rows, Index depth, Index cols, ResScalar alpha,
                  Index strideA=-1, Index strideB=-1, Index offsetA=0, Index offsetB=0);

  static const bool UseRotatingKernel =
    EIGEN_ARCH_ARM &&
    internal::is_same<LhsScalar, float>::value &&
    internal::is_same<RhsScalar, float>::value &&
    internal::is_same<ResScalar, float>::value &&
    Traits::LhsPacketSize == 4 &&
    Traits::RhsPacketSize == 4 &&
    Traits::ResPacketSize == 4;
};

template<typename LhsScalar, typename RhsScalar, typename Index, typename DataMapper, int mr, int nr, bool ConjugateLhs, bool ConjugateRhs>
EIGEN_DONT_INLINE
void gebp_kernel<LhsScalar, RhsScalar, Index, DataMapper, mr, nr, ConjugateLhs, ConjugateRhs>
  ::operator()(const DataMapper& res, const LhsScalar* blockA, const RhsScalar* blockB,
               Index rows, Index depth, Index cols, ResScalar alpha,
               Index strideA, Index strideB, Index offsetA, Index offsetB)
  {
    Traits traits;
    SwappedTraits straits;

    if(strideA==-1) strideA = depth;
    if(strideB==-1) strideB = depth;
    conj_helper<LhsScalar,RhsScalar,ConjugateLhs,ConjugateRhs> cj;
    Index packet_cols4 = nr>=4 ? (cols/4) * 4 : 0;
    const Index peeled_mc3 = mr>=3*Traits::LhsProgress ? (rows/(3*LhsProgress))*(3*LhsProgress) : 0;
    const Index peeled_mc2 = mr>=2*Traits::LhsProgress ? peeled_mc3+((rows-peeled_mc3)/(2*LhsProgress))*(2*LhsProgress) : 0;
    const Index peeled_mc1 = mr>=1*Traits::LhsProgress ? (rows/(1*LhsProgress))*(1*LhsProgress) : 0;
    enum { pk = 8 }; // NOTE Such a large peeling factor is important for large matrices (~ +5% when >1000 on Haswell)
    const Index peeled_kc  = depth & ~(pk-1);
    const Index prefetch_res_offset = 0;
//     const Index depth2     = depth & ~1;

    //---------- Process 3 * LhsProgress rows at once ----------
    // This corresponds to 3*LhsProgress x nr register blocks.
    // Usually, make sense only with FMA
    if(mr>=3*Traits::LhsProgress)
    {
      PossiblyRotatingKernelHelper<gebp_kernel> possiblyRotatingKernelHelper(traits);

      // loops on each largest micro horizontal panel of lhs (3*Traits::LhsProgress x depth)
      for(Index i=0; i<peeled_mc3; i+=3*Traits::LhsProgress)
      {
        // loops on each largest micro vertical panel of rhs (depth * nr)
        for(Index j2=0; j2<packet_cols4; j2+=nr)
        {
          // We select a 3*Traits::LhsProgress x nr micro block of res which is entirely
          // stored into 3 x nr registers.

          const LhsScalar* blA = &blockA[i*strideA+offsetA*(3*Traits::LhsProgress)];
          prefetch(&blA[0]);
          const RhsScalar* blB = &blockB[j2*strideB+offsetB*nr];
          prefetch(&blB[0]);
          LhsPacket A0, A1;

          // gets res block as register
          AccPacket C0, C1, C2,  C3,
                    C4, C5, C6,  C7,
                    C8, C9, C10, C11;
          traits.initAcc(C0);  traits.initAcc(C1);  traits.initAcc(C2);  traits.initAcc(C3);
          traits.initAcc(C4);  traits.initAcc(C5);  traits.initAcc(C6);  traits.initAcc(C7);
          traits.initAcc(C8);  traits.initAcc(C9);  traits.initAcc(C10); traits.initAcc(C11);

          LinearMapper r0 = res.getLinearMapper(i, j2 + 0);
          LinearMapper r1 = res.getLinearMapper(i, j2 + 1);
          LinearMapper r2 = res.getLinearMapper(i, j2 + 2);
          LinearMapper r3 = res.getLinearMapper(i, j2 + 3);

          r0.prefetch(0);
          r1.prefetch(0);
          r2.prefetch(0);
          r3.prefetch(0);

          // performs "inner" products
          for(Index k=0; k<peeled_kc; k+=pk)
          {
            EIGEN_ASM_COMMENT("begin gebp micro kernel 3pX4");
            RhsPacket B_0, T0;
            LhsPacket A2;

#define EIGEN_GEBP_ONESTEP(K) \
            do { \
              EIGEN_ASM_COMMENT("begin step of gebp micro kernel 3pX4"); \
              EIGEN_ASM_COMMENT("Note: these asm comments work around bug 935!"); \
              internal::prefetch(blA+(3*K+16)*LhsProgress); \
              if (EIGEN_ARCH_ARM) internal::prefetch(blB+(4*K+16)*RhsProgress); /* Bug 953 */ \
              traits.loadLhs(&blA[(0+3*K)*LhsProgress], A0);  \
              traits.loadLhs(&blA[(1+3*K)*LhsProgress], A1);  \
              traits.loadLhs(&blA[(2+3*K)*LhsProgress], A2);  \
              possiblyRotatingKernelHelper.template loadOrRotateRhs<K, 0>(B_0, blB); \
              traits.madd(A0, B_0, C0, T0); \
              traits.madd(A1, B_0, C4, T0); \
              traits.madd(A2, B_0, C8, B_0); \
              possiblyRotatingKernelHelper.template loadOrRotateRhs<K, 1>(B_0, blB); \
              traits.madd(A0, B_0, C1, T0); \
              traits.madd(A1, B_0, C5, T0); \
              traits.madd(A2, B_0, C9, B_0); \
              possiblyRotatingKernelHelper.template loadOrRotateRhs<K, 2>(B_0, blB); \
              traits.madd(A0, B_0, C2,  T0); \
              traits.madd(A1, B_0, C6,  T0); \
              traits.madd(A2, B_0, C10, B_0); \
              possiblyRotatingKernelHelper.template loadOrRotateRhs<K, 3>(B_0, blB); \
              traits.madd(A0, B_0, C3 , T0); \
              traits.madd(A1, B_0, C7,  T0); \
              traits.madd(A2, B_0, C11, B_0); \
              EIGEN_ASM_COMMENT("end step of gebp micro kernel 3pX4"); \
            } while(false)

            internal::prefetch(blB);
            EIGEN_GEBP_ONESTEP(0);
            EIGEN_GEBP_ONESTEP(1);
            EIGEN_GEBP_ONESTEP(2);
            EIGEN_GEBP_ONESTEP(3);
            EIGEN_GEBP_ONESTEP(4);
            EIGEN_GEBP_ONESTEP(5);
            EIGEN_GEBP_ONESTEP(6);
            EIGEN_GEBP_ONESTEP(7);

            blB += pk*4*RhsProgress;
            blA += pk*3*Traits::LhsProgress;

            EIGEN_ASM_COMMENT("end gebp micro kernel 3pX4");
          }
          // process remaining peeled loop
          for(Index k=peeled_kc; k<depth; k++)
          {
            RhsPacket B_0, T0;
            LhsPacket A2;
            EIGEN_GEBP_ONESTEP(0);
            blB += 4*RhsProgress;
            blA += 3*Traits::LhsProgress;
          }
#undef EIGEN_GEBP_ONESTEP

          possiblyRotatingKernelHelper.unrotateResult(C0, C1, C2, C3);
          possiblyRotatingKernelHelper.unrotateResult(C4, C5, C6, C7);
          possiblyRotatingKernelHelper.unrotateResult(C8, C9, C10, C11);

          ResPacket R0, R1, R2;
          ResPacket alphav = pset1<ResPacket>(alpha);

          R0 = r0.loadPacket(0 * Traits::ResPacketSize);
          R1 = r0.loadPacket(1 * Traits::ResPacketSize);
          R2 = r0.loadPacket(2 * Traits::ResPacketSize);
          traits.acc(C0, alphav, R0);
          traits.acc(C4, alphav, R1);
          traits.acc(C8, alphav, R2);
          r0.storePacket(0 * Traits::ResPacketSize, R0);
          r0.storePacket(1 * Traits::ResPacketSize, R1);
          r0.storePacket(2 * Traits::ResPacketSize, R2);

          R0 = r1.loadPacket(0 * Traits::ResPacketSize);
          R1 = r1.loadPacket(1 * Traits::ResPacketSize);
          R2 = r1.loadPacket(2 * Traits::ResPacketSize);
          traits.acc(C1, alphav, R0);
          traits.acc(C5, alphav, R1);
          traits.acc(C9, alphav, R2);
          r1.storePacket(0 * Traits::ResPacketSize, R0);
          r1.storePacket(1 * Traits::ResPacketSize, R1);
          r1.storePacket(2 * Traits::ResPacketSize, R2);

          R0 = r2.loadPacket(0 * Traits::ResPacketSize);
          R1 = r2.loadPacket(1 * Traits::ResPacketSize);
          R2 = r2.loadPacket(2 * Traits::ResPacketSize);
          traits.acc(C2, alphav, R0);
          traits.acc(C6, alphav, R1);
          traits.acc(C10, alphav, R2);
          r2.storePacket(0 * Traits::ResPacketSize, R0);
          r2.storePacket(1 * Traits::ResPacketSize, R1);
          r2.storePacket(2 * Traits::ResPacketSize, R2);

          R0 = r3.loadPacket(0 * Traits::ResPacketSize);
          R1 = r3.loadPacket(1 * Traits::ResPacketSize);
          R2 = r3.loadPacket(2 * Traits::ResPacketSize);
          traits.acc(C3, alphav, R0);
          traits.acc(C7, alphav, R1);
          traits.acc(C11, alphav, R2);
          r3.storePacket(0 * Traits::ResPacketSize, R0);
          r3.storePacket(1 * Traits::ResPacketSize, R1);
          r3.storePacket(2 * Traits::ResPacketSize, R2);
        }

        // Deal with remaining columns of the rhs
        for(Index j2=packet_cols4; j2<cols; j2++)
        {
          // One column at a time
          const LhsScalar* blA = &blockA[i*strideA+offsetA*(3*Traits::LhsProgress)];
          prefetch(&blA[0]);
          const RhsScalar* blB = &blockB[j2*strideB+offsetB];
          prefetch(&blB[0]);
          // gets res block as register
          AccPacket C0, C4, C8;
          traits.initAcc(C0);
          traits.initAcc(C4);
          traits.initAcc(C8);

          LinearMapper r0 = res.getLinearMapper(i, j2);
          r0.prefetch(0);
          LhsPacket A0, A1, A2;

          // performs "inner" products
          for(Index k=0; k<peeled_kc; k+=pk)
          {
            EIGEN_ASM_COMMENT("begin gebp micro kernel 3pX1");
            RhsPacket B_0;
#define EIGEN_GEBGP_ONESTEP(K) \
            do { \
              EIGEN_ASM_COMMENT("begin step of gebp micro kernel 3pX1"); \
              EIGEN_ASM_COMMENT("Note: these asm comments work around bug 935!"); \
              traits.loadLhs(&blA[(0+3*K)*LhsProgress], A0);  \
              traits.loadLhs(&blA[(1+3*K)*LhsProgress], A1);  \
              traits.loadLhs(&blA[(2+3*K)*LhsProgress], A2);  \
              traits.loadRhs(&blB[(0+K)*RhsProgress], B_0);   \
              traits.madd(A0, B_0, C0, B_0); \
              traits.madd(A1, B_0, C4, B_0); \
              traits.madd(A2, B_0, C8, B_0); \
              EIGEN_ASM_COMMENT("end step of gebp micro kernel 3pX1"); \
            } while(false)

            EIGEN_GEBGP_ONESTEP(0);
            EIGEN_GEBGP_ONESTEP(1);
            EIGEN_GEBGP_ONESTEP(2);
            EIGEN_GEBGP_ONESTEP(3);
            EIGEN_GEBGP_ONESTEP(4);
            EIGEN_GEBGP_ONESTEP(5);
            EIGEN_GEBGP_ONESTEP(6);
            EIGEN_GEBGP_ONESTEP(7);

            blB += pk*RhsProgress;
            blA += pk*3*Traits::LhsProgress;

            EIGEN_ASM_COMMENT("end gebp micro kernel 3pX1");
          }

          // process remaining peeled loop
          for(Index k=peeled_kc; k<depth; k++)
          {
            RhsPacket B_0;
            EIGEN_GEBGP_ONESTEP(0);
            blB += RhsProgress;
            blA += 3*Traits::LhsProgress;
          }
#undef EIGEN_GEBGP_ONESTEP
          ResPacket R0, R1, R2;
          ResPacket alphav = pset1<ResPacket>(alpha);

          R0 = r0.loadPacket(0 * Traits::ResPacketSize);
          R1 = r0.loadPacket(1 * Traits::ResPacketSize);
          R2 = r0.loadPacket(2 * Traits::ResPacketSize);
          traits.acc(C0, alphav, R0);
          traits.acc(C4, alphav, R1);
          traits.acc(C8, alphav, R2);
          r0.storePacket(0 * Traits::ResPacketSize, R0);
          r0.storePacket(1 * Traits::ResPacketSize, R1);
          r0.storePacket(2 * Traits::ResPacketSize, R2);
        }
      }
    }

    //---------- Process 2 * LhsProgress rows at once ----------
    if(mr>=2*Traits::LhsProgress)
    {
      // loops on each largest micro horizontal panel of lhs (2*LhsProgress x depth)
      for(Index i=peeled_mc3; i<peeled_mc2; i+=2*LhsProgress)
      {
        // loops on each largest micro vertical panel of rhs (depth * nr)
        for(Index j2=0; j2<packet_cols4; j2+=nr)
        {
          // We select a 2*Traits::LhsProgress x nr micro block of res which is entirely
          // stored into 2 x nr registers.

          const LhsScalar* blA = &blockA[i*strideA+offsetA*(2*Traits::LhsProgress)];
          prefetch(&blA[0]);
          const RhsScalar* blB = &blockB[j2*strideB+offsetB*nr];
          prefetch(&blB[0]);

          // gets res block as register
          AccPacket C0, C1, C2, C3,
                    C4, C5, C6, C7;
          traits.initAcc(C0); traits.initAcc(C1); traits.initAcc(C2); traits.initAcc(C3);
          traits.initAcc(C4); traits.initAcc(C5); traits.initAcc(C6); traits.initAcc(C7);

          LinearMapper r0 = res.getLinearMapper(i, j2 + 0);
          LinearMapper r1 = res.getLinearMapper(i, j2 + 1);
          LinearMapper r2 = res.getLinearMapper(i, j2 + 2);
          LinearMapper r3 = res.getLinearMapper(i, j2 + 3);

          r0.prefetch(prefetch_res_offset);
          r1.prefetch(prefetch_res_offset);
          r2.prefetch(prefetch_res_offset);
          r3.prefetch(prefetch_res_offset);

          LhsPacket A0, A1;

          // performs "inner" products
          for(Index k=0; k<peeled_kc; k+=pk)
          {
            EIGEN_ASM_COMMENT("begin gebp micro kernel 2pX4");
            RhsPacket B_0, B1, B2, B3, T0;

            // The 2 ASM comments in the #define are intended to prevent gcc
            // from optimizing the code accross steps since it ends up spilling
            // registers in this case.
   #define EIGEN_GEBGP_ONESTEP(K) \
            do {                                                                \
              EIGEN_ASM_COMMENT("begin step of gebp micro kernel 2pX4");        \
              EIGEN_ASM_COMMENT("Note: these asm comments work around bug 935!"); \
              traits.loadLhs(&blA[(0+2*K)*LhsProgress], A0);                    \
              traits.loadLhs(&blA[(1+2*K)*LhsProgress], A1);                    \
              traits.broadcastRhs(&blB[(0+4*K)*RhsProgress], B_0, B1, B2, B3);  \
              traits.madd(A0, B_0, C0, T0);                                     \
              traits.madd(A1, B_0, C4, B_0);                                    \
              traits.madd(A0, B1,  C1, T0);                                     \
              traits.madd(A1, B1,  C5, B1);                                     \
              traits.madd(A0, B2,  C2, T0);                                     \
              traits.madd(A1, B2,  C6, B2);                                     \
              traits.madd(A0, B3,  C3, T0);                                     \
              traits.madd(A1, B3,  C7, B3);                                     \
              EIGEN_ASM_COMMENT("end step of gebp micro kernel 2pX4");          \
            } while(false)

            prefetch(&blB[pk*4*RhsProgress]);
            EIGEN_GEBGP_ONESTEP(0);
            EIGEN_GEBGP_ONESTEP(1);
            EIGEN_GEBGP_ONESTEP(2);
            EIGEN_GEBGP_ONESTEP(3);
            EIGEN_GEBGP_ONESTEP(4);
            EIGEN_GEBGP_ONESTEP(5);
            EIGEN_GEBGP_ONESTEP(6);
            EIGEN_GEBGP_ONESTEP(7);

            blB += pk*4*RhsProgress;
            blA += pk*(2*Traits::LhsProgress);

            EIGEN_ASM_COMMENT("end gebp micro kernel 2pX4");
          }
          // process remaining peeled loop
          for(Index k=peeled_kc; k<depth; k++)
          {
            RhsPacket B_0, B1, B2, B3, T0;
            EIGEN_GEBGP_ONESTEP(0);
            blB += 4*RhsProgress;
            blA += 2*Traits::LhsProgress;
          }
#undef EIGEN_GEBGP_ONESTEP

          ResPacket R0, R1, R2, R3;
          ResPacket alphav = pset1<ResPacket>(alpha);

          R0 = r0.loadPacket(0 * Traits::ResPacketSize);
          R1 = r0.loadPacket(1 * Traits::ResPacketSize);
          R2 = r1.loadPacket(0 * Traits::ResPacketSize);
          R3 = r1.loadPacket(1 * Traits::ResPacketSize);
          traits.acc(C0, alphav, R0);
          traits.acc(C4, alphav, R1);
          traits.acc(C1, alphav, R2);
          traits.acc(C5, alphav, R3);
          r0.storePacket(0 * Traits::ResPacketSize, R0);
          r0.storePacket(1 * Traits::ResPacketSize, R1);
          r1.storePacket(0 * Traits::ResPacketSize, R2);
          r1.storePacket(1 * Traits::ResPacketSize, R3);

          R0 = r2.loadPacket(0 * Traits::ResPacketSize);
          R1 = r2.loadPacket(1 * Traits::ResPacketSize);
          R2 = r3.loadPacket(0 * Traits::ResPacketSize);
          R3 = r3.loadPacket(1 * Traits::ResPacketSize);
          traits.acc(C2,  alphav, R0);
          traits.acc(C6,  alphav, R1);
          traits.acc(C3,  alphav, R2);
          traits.acc(C7,  alphav, R3);
          r2.storePacket(0 * Traits::ResPacketSize, R0);
          r2.storePacket(1 * Traits::ResPacketSize, R1);
          r3.storePacket(0 * Traits::ResPacketSize, R2);
          r3.storePacket(1 * Traits::ResPacketSize, R3);
        }

        // Deal with remaining columns of the rhs
        for(Index j2=packet_cols4; j2<cols; j2++)
        {
          // One column at a time
          const LhsScalar* blA = &blockA[i*strideA+offsetA*(2*Traits::LhsProgress)];
          prefetch(&blA[0]);
          const RhsScalar* blB = &blockB[j2*strideB+offsetB];
          prefetch(&blB[0]);

          // gets res block as register
          AccPacket C0, C4;
          traits.initAcc(C0);
          traits.initAcc(C4);

          LinearMapper r0 = res.getLinearMapper(i, j2);
          r0.prefetch(prefetch_res_offset);
          LhsPacket A0, A1;

          // performs "inner" products
          for(Index k=0; k<peeled_kc; k+=pk)
          {
            EIGEN_ASM_COMMENT("begin gebp micro kernel 2pX1");
            RhsPacket B_0, B1;

#define EIGEN_GEBGP_ONESTEP(K) \
            do {                                                                  \
              EIGEN_ASM_COMMENT("begin step of gebp micro kernel 2pX1");          \
              EIGEN_ASM_COMMENT("Note: these asm comments work around bug 935!"); \
              traits.loadLhs(&blA[(0+2*K)*LhsProgress], A0);                      \
              traits.loadLhs(&blA[(1+2*K)*LhsProgress], A1);                      \
              traits.loadRhs(&blB[(0+K)*RhsProgress], B_0);                       \
              traits.madd(A0, B_0, C0, B1);                                       \
              traits.madd(A1, B_0, C4, B_0);                                      \
              EIGEN_ASM_COMMENT("end step of gebp micro kernel 2pX1");            \
            } while(false)

            EIGEN_GEBGP_ONESTEP(0);
            EIGEN_GEBGP_ONESTEP(1);
            EIGEN_GEBGP_ONESTEP(2);
            EIGEN_GEBGP_ONESTEP(3);
            EIGEN_GEBGP_ONESTEP(4);
            EIGEN_GEBGP_ONESTEP(5);
            EIGEN_GEBGP_ONESTEP(6);
            EIGEN_GEBGP_ONESTEP(7);

            blB += pk*RhsProgress;
            blA += pk*2*Traits::LhsProgress;

            EIGEN_ASM_COMMENT("end gebp micro kernel 2pX1");
          }

          // process remaining peeled loop
          for(Index k=peeled_kc; k<depth; k++)
          {
            RhsPacket B_0, B1;
            EIGEN_GEBGP_ONESTEP(0);
            blB += RhsProgress;
            blA += 2*Traits::LhsProgress;
          }
#undef EIGEN_GEBGP_ONESTEP
          ResPacket R0, R1;
          ResPacket alphav = pset1<ResPacket>(alpha);

          R0 = r0.loadPacket(0 * Traits::ResPacketSize);
          R1 = r0.loadPacket(1 * Traits::ResPacketSize);
          traits.acc(C0, alphav, R0);
          traits.acc(C4, alphav, R1);
          r0.storePacket(0 * Traits::ResPacketSize, R0);
          r0.storePacket(1 * Traits::ResPacketSize, R1);
        }
      }
    }
    //---------- Process 1 * LhsProgress rows at once ----------
    if(mr>=1*Traits::LhsProgress)
    {
      // loops on each largest micro horizontal panel of lhs (1*LhsProgress x depth)
      for(Index i=peeled_mc2; i<peeled_mc1; i+=1*LhsProgress)
      {
        // loops on each largest micro vertical panel of rhs (depth * nr)
        for(Index j2=0; j2<packet_cols4; j2+=nr)
        {
          // We select a 1*Traits::LhsProgress x nr micro block of res which is entirely
          // stored into 1 x nr registers.

          const LhsScalar* blA = &blockA[i*strideA+offsetA*(1*Traits::LhsProgress)];
          prefetch(&blA[0]);
          const RhsScalar* blB = &blockB[j2*strideB+offsetB*nr];
          prefetch(&blB[0]);

          // gets res block as register
          AccPacket C0, C1, C2, C3;
          traits.initAcc(C0);
          traits.initAcc(C1);
          traits.initAcc(C2);
          traits.initAcc(C3);

          LinearMapper r0 = res.getLinearMapper(i, j2 + 0);
          LinearMapper r1 = res.getLinearMapper(i, j2 + 1);
          LinearMapper r2 = res.getLinearMapper(i, j2 + 2);
          LinearMapper r3 = res.getLinearMapper(i, j2 + 3);

          r0.prefetch(prefetch_res_offset);
          r1.prefetch(prefetch_res_offset);
          r2.prefetch(prefetch_res_offset);
          r3.prefetch(prefetch_res_offset);
          LhsPacket A0;

          // performs "inner" products
          for(Index k=0; k<peeled_kc; k+=pk)
          {
            EIGEN_ASM_COMMENT("begin gebp micro kernel 1pX4");
            RhsPacket B_0, B1, B2, B3;

#define EIGEN_GEBGP_ONESTEP(K) \
            do {                                                                \
              EIGEN_ASM_COMMENT("begin step of gebp micro kernel 1pX4");        \
              EIGEN_ASM_COMMENT("Note: these asm comments work around bug 935!"); \
              traits.loadLhs(&blA[(0+1*K)*LhsProgress], A0);                    \
              traits.broadcastRhs(&blB[(0+4*K)*RhsProgress], B_0, B1, B2, B3);  \
              traits.madd(A0, B_0, C0, B_0);                                    \
              traits.madd(A0, B1,  C1, B1);                                     \
              traits.madd(A0, B2,  C2, B2);                                     \
              traits.madd(A0, B3,  C3, B3);                                     \
              EIGEN_ASM_COMMENT("end step of gebp micro kernel 1pX4");          \
            } while(false)

            EIGEN_GEBGP_ONESTEP(0);
            EIGEN_GEBGP_ONESTEP(1);
            EIGEN_GEBGP_ONESTEP(2);
            EIGEN_GEBGP_ONESTEP(3);
            EIGEN_GEBGP_ONESTEP(4);
            EIGEN_GEBGP_ONESTEP(5);
            EIGEN_GEBGP_ONESTEP(6);
            EIGEN_GEBGP_ONESTEP(7);

            blB += pk*4*RhsProgress;
            blA += pk*1*LhsProgress;

            EIGEN_ASM_COMMENT("end gebp micro kernel 1pX4");
          }
          // process remaining peeled loop
          for(Index k=peeled_kc; k<depth; k++)
          {
            RhsPacket B_0, B1, B2, B3;
            EIGEN_GEBGP_ONESTEP(0);
            blB += 4*RhsProgress;
            blA += 1*LhsProgress;
          }
#undef EIGEN_GEBGP_ONESTEP

          ResPacket R0, R1;
          ResPacket alphav = pset1<ResPacket>(alpha);

          R0 = r0.loadPacket(0 * Traits::ResPacketSize);
          R1 = r1.loadPacket(0 * Traits::ResPacketSize);
          traits.acc(C0, alphav, R0);
          traits.acc(C1,  alphav, R1);
          r0.storePacket(0 * Traits::ResPacketSize, R0);
          r1.storePacket(0 * Traits::ResPacketSize, R1);

          R0 = r2.loadPacket(0 * Traits::ResPacketSize);
          R1 = r3.loadPacket(0 * Traits::ResPacketSize);
          traits.acc(C2,  alphav, R0);
          traits.acc(C3,  alphav, R1);
          r2.storePacket(0 * Traits::ResPacketSize, R0);
          r3.storePacket(0 * Traits::ResPacketSize, R1);
        }

        // Deal with remaining columns of the rhs
        for(Index j2=packet_cols4; j2<cols; j2++)
        {
          // One column at a time
          const LhsScalar* blA = &blockA[i*strideA+offsetA*(1*Traits::LhsProgress)];
          prefetch(&blA[0]);
          const RhsScalar* blB = &blockB[j2*strideB+offsetB];
          prefetch(&blB[0]);

          // gets res block as register
          AccPacket C0;
          traits.initAcc(C0);

          LinearMapper r0 = res.getLinearMapper(i, j2);
          LhsPacket A0;

          // performs "inner" products
          for(Index k=0; k<peeled_kc; k+=pk)
          {
            EIGEN_ASM_COMMENT("begin gebp micro kernel 2pX1");
            RhsPacket B_0;

#define EIGEN_GEBGP_ONESTEP(K) \
            do {                                                                \
              EIGEN_ASM_COMMENT("begin step of gebp micro kernel 2pX1");        \
              EIGEN_ASM_COMMENT("Note: these asm comments work around bug 935!"); \
              traits.loadLhs(&blA[(0+1*K)*LhsProgress], A0);                    \
              traits.loadRhs(&blB[(0+K)*RhsProgress], B_0);                     \
              traits.madd(A0, B_0, C0, B_0);                                    \
              EIGEN_ASM_COMMENT("end step of gebp micro kernel 2pX1");          \
            } while(false)

            EIGEN_GEBGP_ONESTEP(0);
            EIGEN_GEBGP_ONESTEP(1);
            EIGEN_GEBGP_ONESTEP(2);
            EIGEN_GEBGP_ONESTEP(3);
            EIGEN_GEBGP_ONESTEP(4);
            EIGEN_GEBGP_ONESTEP(5);
            EIGEN_GEBGP_ONESTEP(6);
            EIGEN_GEBGP_ONESTEP(7);

            blB += pk*RhsProgress;
            blA += pk*1*Traits::LhsProgress;

            EIGEN_ASM_COMMENT("end gebp micro kernel 2pX1");
          }

          // process remaining peeled loop
          for(Index k=peeled_kc; k<depth; k++)
          {
            RhsPacket B_0;
            EIGEN_GEBGP_ONESTEP(0);
            blB += RhsProgress;
            blA += 1*Traits::LhsProgress;
          }
#undef EIGEN_GEBGP_ONESTEP
          ResPacket R0;
          ResPacket alphav = pset1<ResPacket>(alpha);
          R0 = r0.loadPacket(0 * Traits::ResPacketSize);
          traits.acc(C0, alphav, R0);
          r0.storePacket(0 * Traits::ResPacketSize, R0);
        }
      }
    }
    //---------- Process remaining rows, 1 by 1 ----------
    for(Index i=peeled_mc1; i<rows; i+=1)
    {
      // loop on each panel of the rhs
      for(Index j2=0; j2<packet_cols4; j2+=nr)
      {
        const LhsScalar* blA = &blockA[i*strideA+offsetA];
        prefetch(&blA[0]);
        const RhsScalar* blB = &blockB[j2*strideB+offsetB*nr];
        prefetch(&blB[0]);

        if( (SwappedTraits::LhsProgress % 4)==0 )
        {
          // NOTE The following piece of code wont work for 512 bit registers
          SAccPacket C0, C1, C2, C3;
          straits.initAcc(C0);
          straits.initAcc(C1);
          straits.initAcc(C2);
          straits.initAcc(C3);

          const Index spk   = (std::max)(1,SwappedTraits::LhsProgress/4);
          const Index endk  = (depth/spk)*spk;
          const Index endk4 = (depth/(spk*4))*(spk*4);

          Index k=0;
          for(; k<endk4; k+=4*spk)
          {
            prefetch(&blB[4*SwappedTraits::LhsProgress]);

            SLhsPacket A0,A1,A2,A3;
            SRhsPacket B_0,B_1,B_2,B_3;

            straits.loadLhsUnaligned(blB+0*SwappedTraits::LhsProgress, A0);
            straits.loadLhsUnaligned(blB+1*SwappedTraits::LhsProgress, A1);
            straits.loadRhsQuad(blA+0*spk, B_0);
            straits.loadRhsQuad(blA+1*spk, B_1);
            straits.madd(A0,B_0,C0,B_0);
            straits.madd(A1,B_1,C1,B_1);

            straits.loadLhsUnaligned(blB+2*SwappedTraits::LhsProgress, A2);
            straits.loadLhsUnaligned(blB+3*SwappedTraits::LhsProgress, A3);
            straits.loadRhsQuad(blA+2*spk, B_2);
            straits.loadRhsQuad(blA+3*spk, B_3);
            straits.madd(A2,B_2,C2,B_2);
            straits.madd(A3,B_3,C3,B_3);

            blB += 4*SwappedTraits::LhsProgress;
            blA += 4*spk;
          }
          C0 = padd(padd(C0,C1),padd(C2,C3));
          for(; k<endk; k+=spk)
          {
            SLhsPacket A0;
            SRhsPacket B_0;

            straits.loadLhsUnaligned(blB, A0);
            straits.loadRhsQuad(blA, B_0);
            straits.madd(A0,B_0,C0,B_0);

            blB += SwappedTraits::LhsProgress;
            blA += spk;
          }
          if(SwappedTraits::LhsProgress==8)
          {
            // Special case where we have to first reduce the accumulation register C0
            typedef typename conditional<SwappedTraits::LhsProgress==8,typename unpacket_traits<SResPacket>::half,SResPacket>::type SResPacketHalf;
            typedef typename conditional<SwappedTraits::LhsProgress==8,typename unpacket_traits<SLhsPacket>::half,SLhsPacket>::type SLhsPacketHalf;
            typedef typename conditional<SwappedTraits::LhsProgress==8,typename unpacket_traits<SLhsPacket>::half,SRhsPacket>::type SRhsPacketHalf;
            typedef typename conditional<SwappedTraits::LhsProgress==8,typename unpacket_traits<SAccPacket>::half,SAccPacket>::type SAccPacketHalf;

            SResPacketHalf R = res.template gatherPacket<SResPacketHalf>(i, j2);
            SResPacketHalf alphav = pset1<SResPacketHalf>(alpha);

            if(depth-endk>0)
            {
              // We have to handle the last row of the rhs which corresponds to a half-packet
              SLhsPacketHalf a0;
              SRhsPacketHalf b0;
              straits.loadLhsUnaligned(blB, a0);
              straits.loadRhs(blA, b0);
              SAccPacketHalf c0 = predux4(C0);
              straits.madd(a0,b0,c0,b0);
              straits.acc(c0, alphav, R);
            }
            else
            {
                straits.acc(predux4(C0), alphav, R);
            }
            res.scatterPacket(i, j2, R);
          }
          else
          {
            SResPacket R = res.template gatherPacket<SResPacket>(i, j2);
            SResPacket alphav = pset1<SResPacket>(alpha);
            straits.acc(C0, alphav, R);
            res.scatterPacket(i, j2, R);
          }
        }
        else // scalar path
        {
          // get a 1 x 4 res block as registers
          ResScalar C0(0), C1(0), C2(0), C3(0);

          for(Index k=0; k<depth; k++)
          {
            LhsScalar A0 = blA[k];
            RhsScalar B_0 = blB[0];
            RhsScalar B_1 = blB[1];
            CJMADD(cj,A0,B_0,C0, B_0);
            CJMADD(cj,A0,B_1,C1, B_1);
            RhsScalar B_2 = blB[2];
            RhsScalar B_3 = blB[3];
            CJMADD(cj,A0,B_2,C2, B_2);
            CJMADD(cj,A0,B_3,C3, B_3);

            blB += 4;
          }
          res(i, j2 + 0) += alpha * C0;
          res(i, j2 + 1) += alpha * C1;
          res(i, j2 + 2) += alpha * C2;
          res(i, j2 + 3) += alpha * C3;
        }
      }

      // remaining columns
      for(Index j2=packet_cols4; j2<cols; j2++)
      {
        const LhsScalar* blA = &blockA[i*strideA+offsetA];
        //          prefetch(blA);
        // gets a 1 x 1 res block as registers
        ResScalar C0(0);
        const RhsScalar* blB = &blockB[j2*strideB+offsetB];
        for(Index k=0; k<depth; k++)
        {
          LhsScalar A0 = blA[k];
          RhsScalar B_0 = blB[k];
          CJMADD(cj, A0, B_0, C0, B_0);
        }
        res(i, j2) += alpha * C0;
      }
    }
  }


#undef CJMADD

// pack a block of the lhs
// The traversal is as follow (mr==4):
//   0  4  8 12 ...
//   1  5  9 13 ...
//   2  6 10 14 ...
//   3  7 11 15 ...
//
//  16 20 24 28 ...
//  17 21 25 29 ...
//  18 22 26 30 ...
//  19 23 27 31 ...
//
//  32 33 34 35 ...
//  36 36 38 39 ...
template<typename Scalar, typename Index, typename DataMapper, int Pack1, int Pack2, bool Conjugate, bool PanelMode>
struct gemm_pack_lhs<Scalar, Index, DataMapper, Pack1, Pack2, ColMajor, Conjugate, PanelMode>
{
  typedef typename DataMapper::LinearMapper LinearMapper;
  EIGEN_DONT_INLINE void operator()(Scalar* blockA, const DataMapper& lhs, Index depth, Index rows, Index stride=0, Index offset=0);
};

template<typename Scalar, typename Index, typename DataMapper, int Pack1, int Pack2, bool Conjugate, bool PanelMode>
EIGEN_DONT_INLINE void gemm_pack_lhs<Scalar, Index, DataMapper, Pack1, Pack2, ColMajor, Conjugate, PanelMode>
  ::operator()(Scalar* blockA, const DataMapper& lhs, Index depth, Index rows, Index stride, Index offset)
{
  typedef typename packet_traits<Scalar>::type Packet;
  enum { PacketSize = packet_traits<Scalar>::size };

  EIGEN_ASM_COMMENT("EIGEN PRODUCT PACK LHS");
  EIGEN_UNUSED_VARIABLE(stride);
  EIGEN_UNUSED_VARIABLE(offset);
  eigen_assert(((!PanelMode) && stride==0 && offset==0) || (PanelMode && stride>=depth && offset<=stride));
  eigen_assert( ((Pack1%PacketSize)==0 && Pack1<=4*PacketSize) || (Pack1<=4) );
  conj_if<NumTraits<Scalar>::IsComplex && Conjugate> cj;

  const Index peeled_mc3 = Pack1>=3*PacketSize ? (rows/(3*PacketSize))*(3*PacketSize) : 0;
  const Index peeled_mc2 = Pack1>=2*PacketSize ? peeled_mc3+((rows-peeled_mc3)/(2*PacketSize))*(2*PacketSize) : 0;
  const Index peeled_mc1 = Pack1>=1*PacketSize ? (rows/(1*PacketSize))*(1*PacketSize) : 0;
  const Index peeled_mc0 = Pack2>=1*PacketSize ? peeled_mc1
                         : Pack2>1             ? (rows/Pack2)*Pack2 : 0;

  Index i=0;

  // Pack 3 packets
  if(Pack1>=3*PacketSize)
  {
    if(PanelMode)
    {
      for(; i<peeled_mc3; i+=3*PacketSize)
      {
        blockA += (3*PacketSize) * offset;

        for(Index k=0; k<depth; k++)
        {
          Packet A, B, C;
          A = lhs.loadPacket(i+0*PacketSize, k);
          B = lhs.loadPacket(i+1*PacketSize, k);
          C = lhs.loadPacket(i+2*PacketSize, k);
          pstore(blockA+0*PacketSize, cj.pconj(A));
          pstore(blockA+1*PacketSize, cj.pconj(B));
          pstore(blockA+2*PacketSize, cj.pconj(C));
          blockA += 3*PacketSize;
        }
        blockA += (3*PacketSize) * (stride-offset-depth);
      }
    }
    else
    {
      // Read the data from DRAM as sequentially as possible. We're writing to
      // SRAM so the order of the writes shouldn't impact performance.
      for(Index k=0; k<depth; k++)
      {
        Scalar* localBlockA = blockA + 3*PacketSize*k;
        for(Index local_i = i; local_i<peeled_mc3; local_i+=3*PacketSize)
        {
          Packet A, B, C;
          A = lhs.loadPacket(local_i+0*PacketSize, k);
          B = lhs.loadPacket(local_i+1*PacketSize, k);
          C = lhs.loadPacket(local_i+2*PacketSize, k);
          pstore(localBlockA+0*PacketSize, cj.pconj(A));
          pstore(localBlockA+1*PacketSize, cj.pconj(B));
          pstore(localBlockA+2*PacketSize, cj.pconj(C));
          localBlockA += 3*PacketSize*depth;
        }
      }
      blockA += depth*peeled_mc3;
      i = peeled_mc3;
    }
  }
  // Pack 2 packets
  if(Pack1>=2*PacketSize)
  {
    if(PanelMode)
    {
      for(; i<peeled_mc2; i+=2*PacketSize)
      {
        blockA += (2*PacketSize) * offset;

        for(Index k=0; k<depth; k++)
        {
          Packet A, B;
          A = lhs.loadPacket(i+0*PacketSize, k);
          B = lhs.loadPacket(i+1*PacketSize, k);
          pstore(blockA+0*PacketSize, cj.pconj(A));
          pstore(blockA+1*PacketSize, cj.pconj(B));
          blockA += 2*PacketSize;
        }
        blockA += (2*PacketSize) * (stride-offset-depth);
      }
    }
    else
    {
      // Read the data from RAM as sequentially as possible.
      for(Index k=0; k<depth; k++)
      {
        Scalar* localBlockA = blockA + 2*PacketSize*k;
        for(Index local_i = i; local_i<peeled_mc2; local_i+=2*PacketSize)
        {
          Packet A, B;
          A = lhs.loadPacket(local_i+0*PacketSize, k);
          B = lhs.loadPacket(local_i+1*PacketSize, k);
          pstore(localBlockA+0*PacketSize, cj.pconj(A));
          pstore(localBlockA+1*PacketSize, cj.pconj(B));
          localBlockA += 2*PacketSize*depth;
        }
      }
      blockA += depth*(peeled_mc2-i);
      i = peeled_mc2;
    }
  }
  // Pack 1 packets
  if(Pack1>=1*PacketSize)
  {
    if(PanelMode)
    {
      for(; i<peeled_mc1; i+=1*PacketSize)
      {
        blockA += (1*PacketSize) * offset;

        for(Index k=0; k<depth; k++)
        {
          Packet A;
          A = lhs.loadPacket(i+0*PacketSize, k);
          pstore(blockA, cj.pconj(A));
          blockA+=PacketSize;
        }
        blockA += (1*PacketSize) * (stride-offset-depth);
      }
    }
    else
    {
      // Read the data from RAM as sequentially as possible.
      for(Index k=0; k<depth; k++)
      {
        Scalar* localBlockA = blockA + PacketSize*k;
        for(Index local_i = i; local_i<peeled_mc1; local_i+=1*PacketSize)
        {
          Packet A;
          A = lhs.loadPacket(local_i+0*PacketSize, k);
          pstore(localBlockA, cj.pconj(A));
          localBlockA += PacketSize*depth;
        }
      }
      blockA += depth*(peeled_mc1-i);
      i = peeled_mc1;
    }
  }
  // Pack scalars
  if(Pack2<PacketSize && Pack2>1)
  {
    for(; i<peeled_mc0; i+=Pack2)
    {
      if (PanelMode) {
        blockA += Pack2 * offset;
      }

      for(Index k=0; k<depth; k++) {
        const LinearMapper dm0 = lhs.getLinearMapper(i, k);
        for(Index w=0; w<Pack2; w++) {
          *blockA = cj(dm0(w));
          blockA += 1;
        }
      }

      if(PanelMode) blockA += Pack2 * (stride-offset-depth);
    }
  }
  for(; i<rows; i++)
  {
    if(PanelMode) blockA += offset;
    for(Index k=0; k<depth; k++) {
      *blockA = cj(lhs(i, k));
      blockA += 1;
    }
    if(PanelMode) blockA += (stride-offset-depth);
  }
}

template<typename Scalar, typename Index, typename DataMapper, int Pack1, int Pack2, bool Conjugate, bool PanelMode>
struct gemm_pack_lhs<Scalar, Index, DataMapper, Pack1, Pack2, RowMajor, Conjugate, PanelMode>
{
  typedef typename DataMapper::LinearMapper LinearMapper;
  EIGEN_DONT_INLINE void operator()(Scalar* blockA, const DataMapper& lhs, Index depth, Index rows, Index stride=0, Index offset=0);
};

template<typename Scalar, typename Index, typename DataMapper, int Pack1, int Pack2, bool Conjugate, bool PanelMode>
EIGEN_DONT_INLINE void gemm_pack_lhs<Scalar, Index, DataMapper, Pack1, Pack2, RowMajor, Conjugate, PanelMode>
  ::operator()(Scalar* blockA, const DataMapper& lhs, Index depth, Index rows, Index stride, Index offset)
{
  typedef typename packet_traits<Scalar>::type Packet;
  enum { PacketSize = packet_traits<Scalar>::size };

  EIGEN_ASM_COMMENT("EIGEN PRODUCT PACK LHS");
  EIGEN_UNUSED_VARIABLE(stride);
  EIGEN_UNUSED_VARIABLE(offset);
  eigen_assert(((!PanelMode) && stride==0 && offset==0) || (PanelMode && stride>=depth && offset<=stride));
  conj_if<NumTraits<Scalar>::IsComplex && Conjugate> cj;

//   const Index peeled_mc3 = Pack1>=3*PacketSize ? (rows/(3*PacketSize))*(3*PacketSize) : 0;
//   const Index peeled_mc2 = Pack1>=2*PacketSize ? peeled_mc3+((rows-peeled_mc3)/(2*PacketSize))*(2*PacketSize) : 0;
//   const Index peeled_mc1 = Pack1>=1*PacketSize ? (rows/(1*PacketSize))*(1*PacketSize) : 0;

  int pack = Pack1;
  Index i = 0;
  while(pack>0)
  {
    Index remaining_rows = rows-i;
    Index peeled_mc = i+(remaining_rows/pack)*pack;
    for(; i<peeled_mc; i+=pack)
    {
      if(PanelMode) blockA += pack * offset;

      const Index peeled_k = (depth/PacketSize)*PacketSize;
      Index k=0;
      if(pack>=PacketSize)
      {
        for(; k<peeled_k; k+=PacketSize)
        {
          for (Index m = 0; m < pack; m += PacketSize)
          {
            PacketBlock<Packet> kernel;
            for (int p = 0; p < PacketSize; ++p) kernel.packet[p] = lhs.loadPacket(i+p+m, k);
            ptranspose(kernel);
            for (int p = 0; p < PacketSize; ++p) pstore(blockA+m+(pack)*p, cj.pconj(kernel.packet[p]));
          }
          blockA += PacketSize*pack;
        }
      }
      for(; k<depth; k++)
      {
        Index w=0;
        for(; w<pack-3; w+=4)
        {
          Scalar a(cj(lhs(i+w+0, k))),
                 b(cj(lhs(i+w+1, k))),
                 c(cj(lhs(i+w+2, k))),
                 d(cj(lhs(i+w+3, k)));
          blockA[0] = a;
          blockA[1] = b;
          blockA[2] = c;
          blockA[3] = d;
          blockA += 4;
        }
        if(pack%4)
          for(;w<pack;++w) {
            *blockA = cj(lhs(i+w, k));
            blockA += 1;
          }
      }

      if(PanelMode) blockA += pack * (stride-offset-depth);
    }

    pack -= PacketSize;
    if(pack<Pack2 && (pack+PacketSize)!=Pack2)
      pack = Pack2;
  }

  for(; i<rows; i++)
  {
    if(PanelMode) blockA += offset;
    for(Index k=0; k<depth; k++) {
      *blockA = cj(lhs(i, k));
      blockA += 1;
    }
    if(PanelMode) blockA += (stride-offset-depth);
  }
}

// copy a complete panel of the rhs
// this version is optimized for column major matrices
// The traversal order is as follow: (nr==4):
//  0  1  2  3   12 13 14 15   24 27
//  4  5  6  7   16 17 18 19   25 28
//  8  9 10 11   20 21 22 23   26 29
//  .  .  .  .    .  .  .  .    .  .
template<typename Scalar, typename Index, typename DataMapper, int nr, bool Conjugate, bool PanelMode>
struct gemm_pack_rhs<Scalar, Index, DataMapper, nr, ColMajor, Conjugate, PanelMode>
{
  typedef typename packet_traits<Scalar>::type Packet;
  typedef typename DataMapper::LinearMapper LinearMapper;
  enum { PacketSize = packet_traits<Scalar>::size };
  EIGEN_DONT_INLINE void operator()(Scalar* blockB, const DataMapper& rhs, Index depth, Index cols, Index stride=0, Index offset=0);
};

template<typename Scalar, typename Index, typename DataMapper, int nr, bool Conjugate, bool PanelMode>
EIGEN_DONT_INLINE void gemm_pack_rhs<Scalar, Index, DataMapper, nr, ColMajor, Conjugate, PanelMode>
::operator()(Scalar* blockB, const DataMapper& rhs, Index depth, Index cols, Index stride, Index offset)
{
  EIGEN_ASM_COMMENT("EIGEN PRODUCT PACK RHS COLMAJOR");
  EIGEN_UNUSED_VARIABLE(stride);
  EIGEN_UNUSED_VARIABLE(offset);
  eigen_assert(((!PanelMode) && stride==0 && offset==0) || (PanelMode && stride>=depth && offset<=stride));
  conj_if<NumTraits<Scalar>::IsComplex && Conjugate> cj;
  Index packet_cols8 = nr>=8 ? (cols/8) * 8 : 0;
  Index packet_cols4 = nr>=4 ? (cols/4) * 4 : 0;
  const Index peeled_k = (depth/PacketSize)*PacketSize;
//   if(nr>=8)
//   {
//     for(Index j2=0; j2<packet_cols8; j2+=8)
//     {
//       // skip what we have before
//       if(PanelMode) count += 8 * offset;
//       const Scalar* b0 = &rhs[(j2+0)*rhsStride];
//       const Scalar* b1 = &rhs[(j2+1)*rhsStride];
//       const Scalar* b2 = &rhs[(j2+2)*rhsStride];
//       const Scalar* b3 = &rhs[(j2+3)*rhsStride];
//       const Scalar* b4 = &rhs[(j2+4)*rhsStride];
//       const Scalar* b5 = &rhs[(j2+5)*rhsStride];
//       const Scalar* b6 = &rhs[(j2+6)*rhsStride];
//       const Scalar* b7 = &rhs[(j2+7)*rhsStride];
//       Index k=0;
//       if(PacketSize==8) // TODO enbale vectorized transposition for PacketSize==4
//       {
//         for(; k<peeled_k; k+=PacketSize) {
//           PacketBlock<Packet> kernel;
//           for (int p = 0; p < PacketSize; ++p) {
//             kernel.packet[p] = ploadu<Packet>(&rhs[(j2+p)*rhsStride+k]);
//           }
//           ptranspose(kernel);
//           for (int p = 0; p < PacketSize; ++p) {
//             pstoreu(blockB+count, cj.pconj(kernel.packet[p]));
//             count+=PacketSize;
//           }
//         }
//       }
//       for(; k<depth; k++)
//       {
//         blockB[count+0] = cj(b0[k]);
//         blockB[count+1] = cj(b1[k]);
//         blockB[count+2] = cj(b2[k]);
//         blockB[count+3] = cj(b3[k]);
//         blockB[count+4] = cj(b4[k]);
//         blockB[count+5] = cj(b5[k]);
//         blockB[count+6] = cj(b6[k]);
//         blockB[count+7] = cj(b7[k]);
//         count += 8;
//       }
//       // skip what we have after
//       if(PanelMode) count += 8 * (stride-offset-depth);
//     }
//   }

  if(nr>=4)
  {
    for(Index j2=packet_cols8; j2<packet_cols4; j2+=4)
    {
      // skip what we have before
      if(PanelMode) blockB += 4 * offset;

      // TODO: each of these makes a copy of the stride :(
      const LinearMapper dm0 = rhs.getLinearMapper(0, j2 + 0);
      const LinearMapper dm1 = rhs.getLinearMapper(0, j2 + 1);
      const LinearMapper dm2 = rhs.getLinearMapper(0, j2 + 2);
      const LinearMapper dm3 = rhs.getLinearMapper(0, j2 + 3);

      Index k=0;
      if((PacketSize%4)==0) // TODO enable vectorized transposition for PacketSize==2 ??
      {
        for(; k<peeled_k; k+=PacketSize) {
          PacketBlock<Packet, 4> kernel;
          kernel.packet[0] = dm0.loadPacket(k);
          kernel.packet[1] = dm1.loadPacket(k);
          kernel.packet[2] = dm2.loadPacket(k);
          kernel.packet[3] = dm3.loadPacket(k);
          ptranspose(kernel);
          pstoreu(blockB+0*PacketSize, cj.pconj(kernel.packet[0]));
          pstoreu(blockB+1*PacketSize, cj.pconj(kernel.packet[1]));
          pstoreu(blockB+2*PacketSize, cj.pconj(kernel.packet[2]));
          pstoreu(blockB+3*PacketSize, cj.pconj(kernel.packet[3]));
          blockB+=4*PacketSize;
        }
      }
      for(; k<depth; k++)
      {
        blockB[0] = cj(dm0(k));
        blockB[1] = cj(dm1(k));
        blockB[2] = cj(dm2(k));
        blockB[3] = cj(dm3(k));
        blockB += 4;
      }
      // skip what we have after
      if(PanelMode) blockB += 4 * (stride-offset-depth);
    }
  }

  // copy the remaining columns one at a time (nr==1)
  for(Index j2=packet_cols4; j2<cols; ++j2)
  {
    const LinearMapper dm0 = rhs.getLinearMapper(0, j2);
    if(PanelMode) blockB += offset;
    for(Index k=0; k<depth; k++)
    {
      *blockB = cj(dm0(k));
      blockB += 1;
    }
    if(PanelMode) blockB += (stride-offset-depth);
  }
}

// this version is optimized for row major matrices
template<typename Scalar, typename Index, typename DataMapper, int nr, bool Conjugate, bool PanelMode>
struct gemm_pack_rhs<Scalar, Index, DataMapper, nr, RowMajor, Conjugate, PanelMode>
{
  typedef typename packet_traits<Scalar>::type Packet;
  typedef typename packet_traits<Scalar>::half HalfPacket;
  typedef typename DataMapper::LinearMapper LinearMapper;
  enum {
    PacketSize = packet_traits<Scalar>::size,
    HalfPacketSize = packet_traits<Scalar>::HasHalfPacket ? unpacket_traits<typename packet_traits<Scalar>::half>::size : 0
  };
  EIGEN_DONT_INLINE void operator()(Scalar* blockB, const DataMapper& rhs, Index depth, Index cols, Index stride=0, Index offset=0);
};

template<typename Scalar, typename Index, typename DataMapper, int nr, bool Conjugate, bool PanelMode>
EIGEN_DONT_INLINE void gemm_pack_rhs<Scalar, Index, DataMapper, nr, RowMajor, Conjugate, PanelMode>
  ::operator()(Scalar* blockB, const DataMapper& rhs, Index depth, Index cols, Index stride, Index offset)
{
  EIGEN_ASM_COMMENT("EIGEN PRODUCT PACK RHS ROWMAJOR");
  EIGEN_UNUSED_VARIABLE(stride);
  EIGEN_UNUSED_VARIABLE(offset);
  eigen_assert(((!PanelMode) && stride==0 && offset==0) || (PanelMode && stride>=depth && offset<=stride));
  conj_if<NumTraits<Scalar>::IsComplex && Conjugate> cj;
  Index packet_cols8 = nr>=8 ? (cols/8) * 8 : 0;
  Index packet_cols4 = nr>=4 ? (cols/4) * 4 : 0;

//   if(nr>=8)
//   {
//     for(Index j2=0; j2<packet_cols8; j2+=8)
//     {
//       // skip what we have before
//       if(PanelMode) count += 8 * offset;
//       for(Index k=0; k<depth; k++)
//       {
//         if (PacketSize==8) {
//           Packet A = ploadu<Packet>(&rhs[k*rhsStride + j2]);
//           pstoreu(blockB+count, cj.pconj(A));
//         } else if (PacketSize==4) {
//           Packet A = ploadu<Packet>(&rhs[k*rhsStride + j2]);
//           Packet B = ploadu<Packet>(&rhs[k*rhsStride + j2 + PacketSize]);
//           pstoreu(blockB+count, cj.pconj(A));
//           pstoreu(blockB+count+PacketSize, cj.pconj(B));
//         } else {
//           const Scalar* b0 = &rhs[k*rhsStride + j2];
//           blockB[count+0] = cj(b0[0]);
//           blockB[count+1] = cj(b0[1]);
//           blockB[count+2] = cj(b0[2]);
//           blockB[count+3] = cj(b0[3]);
//           blockB[count+4] = cj(b0[4]);
//           blockB[count+5] = cj(b0[5]);
//           blockB[count+6] = cj(b0[6]);
//           blockB[count+7] = cj(b0[7]);
//         }
//         count += 8;
//       }
//       // skip what we have after
//       if(PanelMode) count += 8 * (stride-offset-depth);
//     }
//   }
  if(nr>=4)
  {
    for(Index j2=packet_cols8; j2<packet_cols4; j2+=4)
    {
      // skip what we have before
      if(PanelMode) blockB += 4 * offset;
      for(Index k=0; k<depth; k++)
      {
        if (PacketSize==4) {
          Packet A = rhs.loadPacket(k, j2);
          pstore(blockB, cj.pconj(A));
          blockB += PacketSize;
        }
        else if (HalfPacketSize==4) {
          HalfPacket A = rhs.loadHalfPacket(k, j2);
          pstore<Scalar, HalfPacket>(blockB, cj.pconj(A));
          blockB += HalfPacketSize;
        }
        else {
          const LinearMapper dm0 = rhs.getLinearMapper(k, j2);
          blockB[0] = cj(dm0(0));
          blockB[1] = cj(dm0(1));
          blockB[2] = cj(dm0(2));
          blockB[3] = cj(dm0(3));
          blockB += 4;
        }
      }
      // skip what we have after
      if(PanelMode) blockB += 4 * (stride-offset-depth);
    }
  }
  // copy the remaining columns one at a time (nr==1)
  for(Index j2=packet_cols4; j2<cols; ++j2)
  {
    if(PanelMode) blockB += offset;
    for(Index k=0; k<depth; k++)
    {
      *blockB = cj(rhs(k, j2));
      blockB += 1;
    }
    if(PanelMode) blockB += stride-offset-depth;
  }
}

} // end namespace internal

/** \returns the currently set level 1 cpu cache size (in bytes) used to estimate the ideal blocking size parameters.
  * \sa setCpuCacheSize */
inline std::ptrdiff_t l1CacheSize()
{
  std::ptrdiff_t l1, l2, l3;
  internal::manage_caching_sizes(GetAction, &l1, &l2, &l3);
  return l1;
}

/** \returns the currently set level 2 cpu cache size (in bytes) used to estimate the ideal blocking size parameters.
  * \sa setCpuCacheSize */
inline std::ptrdiff_t l2CacheSize()
{
  std::ptrdiff_t l1, l2, l3;
  internal::manage_caching_sizes(GetAction, &l1, &l2, &l3);
  return l2;
}

/** \returns the currently set level 3 cpu cache size (in bytes) used to estimate the ideal blocking size parameters.
  * \sa setCpuCacheSize */
inline std::ptrdiff_t l3CacheSize()
{
  std::ptrdiff_t l1, l2, l3;
  internal::manage_caching_sizes(GetAction, &l1, &l2, &l3);
  return l3;
}

/** Set the cpu L1 and L2 cache sizes (in bytes).
  * These values are use to adjust the size of the blocks
  * for the algorithms working per blocks.
  *
  * \sa computeProductBlockingSizes */
inline void setCpuCacheSizes(std::ptrdiff_t l1, std::ptrdiff_t l2, std::ptrdiff_t l3)
{
  internal::manage_caching_sizes(SetAction, &l1, &l2, &l3);
}

} // end namespace Eigen

#endif // EIGEN_GENERAL_BLOCK_PANEL_H
