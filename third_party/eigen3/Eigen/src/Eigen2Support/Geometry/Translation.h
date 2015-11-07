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
  * \class Translation
  *
  * \brief Represents a translation transformation
  *
  * \param _Scalar the scalar type, i.e., the type of the coefficients.
  * \param _Dim the  dimension of the space, can be a compile time value or Dynamic
  *
  * \note This class is not aimed to be used to store a translation transformation,
  * but rather to make easier the constructions and updates of Transform objects.
  *
  * \sa class Scaling, class Transform
  */
template<typename _Scalar, int _Dim>
class Translation
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
  /** corresponding scaling transformation type */
  typedef Scaling<Scalar,Dim> ScalingType;
  /** corresponding affine transformation type */
  typedef Transform<Scalar,Dim> TransformType;

protected:

  VectorType m_coeffs;

public:

  /** Default constructor without initialization. */
  Translation() {}
  /**  */
  inline Translation(const Scalar& sx, const Scalar& sy)
  {
    ei_assert(Dim==2);
    m_coeffs.x() = sx;
    m_coeffs.y() = sy;
  }
  /**  */
  inline Translation(const Scalar& sx, const Scalar& sy, const Scalar& sz)
  {
    ei_assert(Dim==3);
    m_coeffs.x() = sx;
    m_coeffs.y() = sy;
    m_coeffs.z() = sz;
  }
  /** Constructs and initialize the scaling transformation from a vector of scaling coefficients */
  explicit inline Translation(const VectorType& vector) : m_coeffs(vector) {}

  const VectorType& vector() const { return m_coeffs; }
  VectorType& vector() { return m_coeffs; }

  /** Concatenates two translation */
  inline Translation operator* (const Translation& other) const
  { return Translation(m_coeffs + other.m_coeffs); }

  /** Concatenates a translation and a scaling */
  inline TransformType operator* (const ScalingType& other) const;

  /** Concatenates a translation and a linear transformation */
  inline TransformType operator* (const LinearMatrixType& linear) const;

  template<typename Derived>
  inline TransformType operator*(const RotationBase<Derived,Dim>& r) const
  { return *this * r.toRotationMatrix(); }

  /** Concatenates a linear transformation and a translation */
  // its a nightmare to define a templated friend function outside its declaration
  friend inline TransformType operator* (const LinearMatrixType& linear, const Translation& t)
  {
    TransformType res;
    res.matrix().setZero();
    res.linear() = linear;
    res.translation() = linear * t.m_coeffs;
    res.matrix().row(Dim).setZero();
    res(Dim,Dim) = Scalar(1);
    return res;
  }

  /** Concatenates a translation and an affine transformation */
  inline TransformType operator* (const TransformType& t) const;

  /** Applies translation to vector */
  inline VectorType operator* (const VectorType& other) const
  { return m_coeffs + other; }

  /** \returns the inverse translation (opposite) */
  Translation inverse() const { return Translation(-m_coeffs); }

  Translation& operator=(const Translation& other)
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
  inline typename internal::cast_return_type<Translation,Translation<NewScalarType,Dim> >::type cast() const
  { return typename internal::cast_return_type<Translation,Translation<NewScalarType,Dim> >::type(*this); }

  /** Copy constructor with scalar type conversion */
  template<typename OtherScalarType>
  inline explicit Translation(const Translation<OtherScalarType,Dim>& other)
  { m_coeffs = other.vector().template cast<Scalar>(); }

  /** \returns \c true if \c *this is approximately equal to \a other, within the precision
    * determined by \a prec.
    *
    * \sa MatrixBase::isApprox() */
  bool isApprox(const Translation& other, typename NumTraits<Scalar>::Real prec = precision<Scalar>()) const
  { return m_coeffs.isApprox(other.m_coeffs, prec); }

};

/** \addtogroup Geometry_Module */
//@{
typedef Translation<float, 2> Translation2f;
typedef Translation<double,2> Translation2d;
typedef Translation<float, 3> Translation3f;
typedef Translation<double,3> Translation3d;
//@}


template<typename Scalar, int Dim>
inline typename Translation<Scalar,Dim>::TransformType
Translation<Scalar,Dim>::operator* (const ScalingType& other) const
{
  TransformType res;
  res.matrix().setZero();
  res.linear().diagonal() = other.coeffs();
  res.translation() = m_coeffs;
  res(Dim,Dim) = Scalar(1);
  return res;
}

template<typename Scalar, int Dim>
inline typename Translation<Scalar,Dim>::TransformType
Translation<Scalar,Dim>::operator* (const LinearMatrixType& linear) const
{
  TransformType res;
  res.matrix().setZero();
  res.linear() = linear;
  res.translation() = m_coeffs;
  res.matrix().row(Dim).setZero();
  res(Dim,Dim) = Scalar(1);
  return res;
}

template<typename Scalar, int Dim>
inline typename Translation<Scalar,Dim>::TransformType
Translation<Scalar,Dim>::operator* (const TransformType& t) const
{
  TransformType res = t;
  res.pretranslate(m_coeffs);
  return res;
}

} // end namespace Eigen
