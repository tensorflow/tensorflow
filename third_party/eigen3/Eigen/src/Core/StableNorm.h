// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2009 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_STABLENORM_H
#define EIGEN_STABLENORM_H

namespace Eigen { 

namespace internal {

template<typename ExpressionType, typename Scalar>
inline void stable_norm_kernel(const ExpressionType& bl, Scalar& ssq, Scalar& scale, Scalar& invScale)
{
  using std::max;
  Scalar maxCoeff = bl.cwiseAbs().maxCoeff();
  
  if (maxCoeff>scale)
  {
    ssq = ssq * numext::abs2(scale/maxCoeff);
    Scalar tmp = Scalar(1)/maxCoeff;
    if(tmp > NumTraits<Scalar>::highest())
    {
      invScale = NumTraits<Scalar>::highest();
      scale = Scalar(1)/invScale;
    }
    else
    {
      scale = maxCoeff;
      invScale = tmp;
    }
  }
  
  // TODO if the maxCoeff is much much smaller than the current scale,
  // then we can neglect this sub vector
  if(scale>Scalar(0)) // if scale==0, then bl is 0 
    ssq += (bl*invScale).squaredNorm();
}

template<typename Derived>
inline typename NumTraits<typename traits<Derived>::Scalar>::Real
blueNorm_impl(const EigenBase<Derived>& _vec)
{
  typedef typename Derived::RealScalar RealScalar;  
  typedef typename Derived::Index Index;
  using std::pow;
  using std::sqrt;
  using std::abs;
  const Derived& vec(_vec.derived());
  static bool initialized = false;
  static RealScalar b1, b2, s1m, s2m, overfl, rbig, relerr;
  if(!initialized)
  {
    int ibeta, it, iemin, iemax, iexp;
    RealScalar eps;
    // This program calculates the machine-dependent constants
    // bl, b2, slm, s2m, relerr overfl
    // from the "basic" machine-dependent numbers
    // nbig, ibeta, it, iemin, iemax, rbig.
    // The following define the basic machine-dependent constants.
    // For portability, the PORT subprograms "ilmaeh" and "rlmach"
    // are used. For any specific computer, each of the assignment
    // statements can be replaced
    ibeta = std::numeric_limits<RealScalar>::radix;                 // base for floating-point numbers
    it    = std::numeric_limits<RealScalar>::digits;                // number of base-beta digits in mantissa
    iemin = std::numeric_limits<RealScalar>::min_exponent;          // minimum exponent
    iemax = std::numeric_limits<RealScalar>::max_exponent;          // maximum exponent
    rbig  = (std::numeric_limits<RealScalar>::max)();               // largest floating-point number

    iexp  = -((1-iemin)/2);
    b1    = RealScalar(pow(RealScalar(ibeta),RealScalar(iexp)));    // lower boundary of midrange
    iexp  = (iemax + 1 - it)/2;
    b2    = RealScalar(pow(RealScalar(ibeta),RealScalar(iexp)));    // upper boundary of midrange

    iexp  = (2-iemin)/2;
    s1m   = RealScalar(pow(RealScalar(ibeta),RealScalar(iexp)));    // scaling factor for lower range
    iexp  = - ((iemax+it)/2);
    s2m   = RealScalar(pow(RealScalar(ibeta),RealScalar(iexp)));    // scaling factor for upper range

    overfl  = rbig*s2m;                                             // overflow boundary for abig
    eps     = RealScalar(pow(double(ibeta), 1-it));
    relerr  = sqrt(eps);                                            // tolerance for neglecting asml
    initialized = true;
  }
  Index n = vec.size();
  RealScalar ab2 = b2 / RealScalar(n);
  RealScalar asml = RealScalar(0);
  RealScalar amed = RealScalar(0);
  RealScalar abig = RealScalar(0);
  for(typename Derived::InnerIterator it(vec, 0); it; ++it)
  {
    RealScalar ax = abs(it.value());
    if(ax > ab2)     abig += numext::abs2(ax*s2m);
    else if(ax < b1) asml += numext::abs2(ax*s1m);
    else             amed += numext::abs2(ax);
  }
  if(abig > RealScalar(0))
  {
    abig = sqrt(abig);
    if(abig > overfl)
    {
      return rbig;
    }
    if(amed > RealScalar(0))
    {
      abig = abig/s2m;
      amed = sqrt(amed);
    }
    else
      return abig/s2m;
  }
  else if(asml > RealScalar(0))
  {
    if (amed > RealScalar(0))
    {
      abig = sqrt(amed);
      amed = sqrt(asml) / s1m;
    }
    else
      return sqrt(asml)/s1m;
  }
  else
    return sqrt(amed);
  asml = numext::mini(abig, amed);
  abig = numext::maxi(abig, amed);
  if(asml <= abig*relerr)
    return abig;
  else
    return abig * sqrt(RealScalar(1) + numext::abs2(asml/abig));
}

} // end namespace internal

/** \returns the \em l2 norm of \c *this avoiding underflow and overflow.
  * This version use a blockwise two passes algorithm:
  *  1 - find the absolute largest coefficient \c s
  *  2 - compute \f$ s \Vert \frac{*this}{s} \Vert \f$ in a standard way
  *
  * For architecture/scalar types supporting vectorization, this version
  * is faster than blueNorm(). Otherwise the blueNorm() is much faster.
  *
  * \sa norm(), blueNorm(), hypotNorm()
  */
template<typename Derived>
inline typename NumTraits<typename internal::traits<Derived>::Scalar>::Real
MatrixBase<Derived>::stableNorm() const
{
  using std::sqrt;
  const Index blockSize = 4096;
  RealScalar scale(0);
  RealScalar invScale(1);
  RealScalar ssq(0); // sum of square
  enum {
    Alignment = (int(Flags)&DirectAccessBit) || (int(Flags)&AlignedBit) ? 1 : 0
  };
  Index n = size();
  Index bi = internal::first_aligned(derived());
  if (bi>0)
    internal::stable_norm_kernel(this->head(bi), ssq, scale, invScale);
  for (; bi<n; bi+=blockSize)
    internal::stable_norm_kernel(this->segment(bi,numext::mini(blockSize, n - bi)).template forceAlignedAccessIf<Alignment>(), ssq, scale, invScale);
  return scale * sqrt(ssq);
}

/** \returns the \em l2 norm of \c *this using the Blue's algorithm.
  * A Portable Fortran Program to Find the Euclidean Norm of a Vector,
  * ACM TOMS, Vol 4, Issue 1, 1978.
  *
  * For architecture/scalar types without vectorization, this version
  * is much faster than stableNorm(). Otherwise the stableNorm() is faster.
  *
  * \sa norm(), stableNorm(), hypotNorm()
  */
template<typename Derived>
inline typename NumTraits<typename internal::traits<Derived>::Scalar>::Real
MatrixBase<Derived>::blueNorm() const
{
  return internal::blueNorm_impl(*this);
}

/** \returns the \em l2 norm of \c *this avoiding undeflow and overflow.
  * This version use a concatenation of hypot() calls, and it is very slow.
  *
  * \sa norm(), stableNorm()
  */
template<typename Derived>
inline typename NumTraits<typename internal::traits<Derived>::Scalar>::Real
MatrixBase<Derived>::hypotNorm() const
{
  return this->cwiseAbs().redux(internal::scalar_hypot_op<RealScalar>());
}

} // end namespace Eigen

#endif // EIGEN_STABLENORM_H
