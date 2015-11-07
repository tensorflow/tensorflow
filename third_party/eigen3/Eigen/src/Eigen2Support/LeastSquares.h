// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2006-2009 Benoit Jacob <jacob.benoit.1@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN2_LEASTSQUARES_H
#define EIGEN2_LEASTSQUARES_H

namespace Eigen { 

/** \ingroup LeastSquares_Module
  *
  * \leastsquares_module
  *
  * For a set of points, this function tries to express
  * one of the coords as a linear (affine) function of the other coords.
  *
  * This is best explained by an example. This function works in full
  * generality, for points in a space of arbitrary dimension, and also over
  * the complex numbers, but for this example we will work in dimension 3
  * over the real numbers (doubles).
  *
  * So let us work with the following set of 5 points given by their
  * \f$(x,y,z)\f$ coordinates:
  * @code
    Vector3d points[5];
    points[0] = Vector3d( 3.02, 6.89, -4.32 );
    points[1] = Vector3d( 2.01, 5.39, -3.79 );
    points[2] = Vector3d( 2.41, 6.01, -4.01 );
    points[3] = Vector3d( 2.09, 5.55, -3.86 );
    points[4] = Vector3d( 2.58, 6.32, -4.10 );
  * @endcode
  * Suppose that we want to express the second coordinate (\f$y\f$) as a linear
  * expression in \f$x\f$ and \f$z\f$, that is,
  * \f[ y=ax+bz+c \f]
  * for some constants \f$a,b,c\f$. Thus, we want to find the best possible
  * constants \f$a,b,c\f$ so that the plane of equation \f$y=ax+bz+c\f$ fits
  * best the five above points. To do that, call this function as follows:
  * @code
    Vector3d coeffs; // will store the coefficients a, b, c
    linearRegression(
      5,
      &points,
      &coeffs,
      1 // the coord to express as a function of
        // the other ones. 0 means x, 1 means y, 2 means z.
    );
  * @endcode
  * Now the vector \a coeffs is approximately
  * \f$( 0.495 ,  -1.927 ,  -2.906 )\f$.
  * Thus, we get \f$a=0.495, b = -1.927, c = -2.906\f$. Let us check for
  * instance how near points[0] is from the plane of equation \f$y=ax+bz+c\f$.
  * Looking at the coords of points[0], we see that:
  * \f[ax+bz+c = 0.495 * 3.02 + (-1.927) * (-4.32) + (-2.906) = 6.91.\f]
  * On the other hand, we have \f$y=6.89\f$. We see that the values
  * \f$6.91\f$ and \f$6.89\f$
  * are near, so points[0] is very near the plane of equation \f$y=ax+bz+c\f$.
  *
  * Let's now describe precisely the parameters:
  * @param numPoints the number of points
  * @param points the array of pointers to the points on which to perform the linear regression
  * @param result pointer to the vector in which to store the result.
                  This vector must be of the same type and size as the
                  data points. The meaning of its coords is as follows.
                  For brevity, let \f$n=Size\f$,
                  \f$r_i=result[i]\f$,
                  and \f$f=funcOfOthers\f$. Denote by
                  \f$x_0,\ldots,x_{n-1}\f$
                  the n coordinates in the n-dimensional space.
                  Then the resulting equation is:
                  \f[ x_f = r_0 x_0 + \cdots + r_{f-1}x_{f-1}
                   + r_{f+1}x_{f+1} + \cdots + r_{n-1}x_{n-1} + r_n. \f]
  * @param funcOfOthers Determines which coord to express as a function of the
                        others. Coords are numbered starting from 0, so that a
                        value of 0 means \f$x\f$, 1 means \f$y\f$,
                        2 means \f$z\f$, ...
  *
  * \sa fitHyperplane()
  */
template<typename VectorType>
void linearRegression(int numPoints,
                      VectorType **points,
                      VectorType *result,
                      int funcOfOthers )
{
  typedef typename VectorType::Scalar Scalar;
  typedef Hyperplane<Scalar, VectorType::SizeAtCompileTime> HyperplaneType;
  const int size = points[0]->size();
  result->resize(size);
  HyperplaneType h(size);
  fitHyperplane(numPoints, points, &h);
  for(int i = 0; i < funcOfOthers; i++)
    result->coeffRef(i) = - h.coeffs()[i] / h.coeffs()[funcOfOthers];
  for(int i = funcOfOthers; i < size; i++)
    result->coeffRef(i) = - h.coeffs()[i+1] / h.coeffs()[funcOfOthers];
}

/** \ingroup LeastSquares_Module
  *
  * \leastsquares_module
  *
  * This function is quite similar to linearRegression(), so we refer to the
  * documentation of this function and only list here the differences.
  *
  * The main difference from linearRegression() is that this function doesn't
  * take a \a funcOfOthers argument. Instead, it finds a general equation
  * of the form
  * \f[ r_0 x_0 + \cdots + r_{n-1}x_{n-1} + r_n = 0, \f]
  * where \f$n=Size\f$, \f$r_i=retCoefficients[i]\f$, and we denote by
  * \f$x_0,\ldots,x_{n-1}\f$ the n coordinates in the n-dimensional space.
  *
  * Thus, the vector \a retCoefficients has size \f$n+1\f$, which is another
  * difference from linearRegression().
  *
  * In practice, this function performs an hyper-plane fit in a total least square sense
  * via the following steps:
  *  1 - center the data to the mean
  *  2 - compute the covariance matrix
  *  3 - pick the eigenvector corresponding to the smallest eigenvalue of the covariance matrix
  * The ratio of the smallest eigenvalue and the second one gives us a hint about the relevance
  * of the solution. This value is optionally returned in \a soundness.
  *
  * \sa linearRegression()
  */
template<typename VectorType, typename HyperplaneType>
void fitHyperplane(int numPoints,
                   VectorType **points,
                   HyperplaneType *result,
                   typename NumTraits<typename VectorType::Scalar>::Real* soundness = 0)
{
  typedef typename VectorType::Scalar Scalar;
  typedef Matrix<Scalar,VectorType::SizeAtCompileTime,VectorType::SizeAtCompileTime> CovMatrixType;
  EIGEN_STATIC_ASSERT_VECTOR_ONLY(VectorType)
  ei_assert(numPoints >= 1);
  int size = points[0]->size();
  ei_assert(size+1 == result->coeffs().size());

  // compute the mean of the data
  VectorType mean = VectorType::Zero(size);
  for(int i = 0; i < numPoints; ++i)
    mean += *(points[i]);
  mean /= numPoints;

  // compute the covariance matrix
  CovMatrixType covMat = CovMatrixType::Zero(size, size);
  VectorType remean = VectorType::Zero(size);
  for(int i = 0; i < numPoints; ++i)
  {
    VectorType diff = (*(points[i]) - mean).conjugate();
    covMat += diff * diff.adjoint();
  }

  // now we just have to pick the eigen vector with smallest eigen value
  SelfAdjointEigenSolver<CovMatrixType> eig(covMat);
  result->normal() = eig.eigenvectors().col(0);
  if (soundness)
    *soundness = eig.eigenvalues().coeff(0)/eig.eigenvalues().coeff(1);

  // let's compute the constant coefficient such that the
  // plane pass trough the mean point:
  result->offset() = - (result->normal().cwise()* mean).sum();
}

} // end namespace Eigen

#endif // EIGEN2_LEASTSQUARES_H
