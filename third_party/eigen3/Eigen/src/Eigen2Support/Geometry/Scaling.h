// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008 Gael Guennebaud <g.gael@free.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

// no include guard, we'll include this twice from All.h from Eigen2Support, and it's internal anyway

namespace Eigen { 

/** \geometry_module \ingroup Geometry_Module
  *
  * \class Scaling
  *
  * \brief Represents a possibly non uniform scaling transformation
  *
  * \param _Scalar the scalar type, i.e., the type of the coefficients.
  * \param _Dim the  dimension of the space, can be a compile time value or Dynamic
  *
  * \note This class is not aimed to be used to store a scaling transformation,
  * but rather to make easier the constructions and updates of Transform objects.
  *
  * \sa class Translation, class Transform
  */
template<typename _Scalar, int _Dim>
class Scaling
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW_IF_VECTORIZABLE_FIXED_SIZE(_Scalar,_Dim)
  /** dimension of the space */
  enum { Dim = _Dim };
  /** the scalar type of the coefficients */
  typedef _Scalar Scalar;
  /** corresponding vector type */
  typedef Matrix<Scalar,Dim,1> VectorType;
  /** corresponding linear transformation matrix type */
  typedef Matrix<Scalar,Dim,Dim> LinearMatrixType;
  /** corresponding translation type */
  typedef Translation<Scalar,Dim> TranslationType;
  /** corresponding affine transformation type */
  typedef Transform<Scalar,Dim> TransformType;

protected:

  VectorType m_coeffs;

public:

  /** Default constructor without initialization. */
  Scaling() {}
  /** Constructs and initialize a uniform scaling transformation */
  explicit inline Scaling(const Scalar& s) { m_coeffs.setConstant(s); }
  /** 2D only */
  inline Scaling(const Scalar& sx, const Scalar& sy)
  {
    ei_assert(Dim==2);
    m_coeffs.x() = sx;
    m_coeffs.y() = sy;
  }
  /** 3D only */
  inline Scaling(const Scalar& sx, const Scalar& sy, const Scalar& sz)
  {
    ei_assert(Dim==3);
    m_coeffs.x() = sx;
    m_coeffs.y() = sy;
    m_coeffs.z() = sz;
  }
  /** Constructs and initialize the scaling transformation from a vector of scaling coefficients */
  explicit inline Scaling(const VectorType& coeffs) : m_coeffs(coeffs) {}

  const VectorType& coeffs() const { return m_coeffs; }
  VectorType& coeffs() { return m_coeffs; }

  /** Concatenates two scaling */
  inline Scaling operator* (const Scaling& other) const
  { return Scaling(coeffs().cwise() * other.coeffs()); }

  /** Concatenates a scaling and a translation */
  inline TransformType operator* (const TranslationType& t) const;

  /** Concatenates a scaling and an affine transformation */
  inline TransformType operator* (const TransformType& t) const;

  /** Concatenates a scaling and a linear transformation matrix */
  // TODO returns an expression
  inline LinearMatrixType operator* (const LinearMatrixType& other) const
  { return coeffs().asDiagonal() * other; }

  /** Concatenates a linear transformation matrix and a scaling */
  // TODO returns an expression
  friend inline LinearMatrixType operator* (const LinearMatrixType& other, const Scaling& s)
  { return other * s.coeffs().asDiagonal(); }

  template<typename Derived>
  inline LinearMatrixType operator*(const RotationBase<Derived,Dim>& r) const
  { return *this * r.toRotationMatrix(); }

  /** Applies scaling to vector */
  inline VectorType operator* (const VectorType& other) const
  { return coeffs().asDiagonal() * other; }

  /** \returns the inverse scaling */
  inline Scaling inverse() const
  { return Scaling(coeffs().cwise().inverse()); }

  inline Scaling& operator=(const Scaling& other)
  {
    m_coeffs = other.m_coeffs;
    return *this;
  }

  /** \returns \c *this with scalar type casted to \a NewScalarType
    *
    * Note that if \a NewScalarType is equal to the current scalar type of \c *this
    * then this function smartly returns a const reference to \c *this.
    */
  template<typename NewScalarType>
  inline typename internal::cast_return_type<Scaling,Scaling<NewScalarType,Dim> >::type cast() const
  { return typename internal::cast_return_type<Scaling,Scaling<NewScalarType,Dim> >::type(*this); }

  /** Copy constructor with scalar type conversion */
  template<typename OtherScalarType>
  inline explicit Scaling(const Scaling<OtherScalarType,Dim>& other)
  { m_coeffs = other.coeffs().template cast<Scalar>(); }

  /** \returns \c true if \c *this is approximately equal to \a other, within the precision
    * determined by \a prec.
    *
    * \sa MatrixBase::isApprox() */
  bool isApprox(const Scaling& other, typename NumTraits<Scalar>::Real prec = precision<Scalar>()) const
  { return m_coeffs.isApprox(other.m_coeffs, prec); }

};

/** \addtogroup Geometry_Module */
//@{
typedef Scaling<float, 2> Scaling2f;
typedef Scaling<double,2> Scaling2d;
typedef Scaling<float, 3> Scaling3f;
typedef Scaling<double,3> Scaling3d;
//@}

template<typename Scalar, int Dim>
inline typename Scaling<Scalar,Dim>::TransformType
Scaling<Scalar,Dim>::operator* (const TranslationType& t) const
{
  TransformType res;
  res.matrix().setZero();
  res.linear().diagonal() = coeffs();
  res.translation() = m_coeffs.cwise() * t.vector();
  res(Dim,Dim) = Scalar(1);
  return res;
}

template<typename Scalar, int Dim>
inline typename Scaling<Scalar,Dim>::TransformType
Scaling<Scalar,Dim>::operator* (const TransformType& t) const
{
  TransformType res = t;
  res.prescale(m_coeffs);
  return res;
}

} // end namespace Eigen
