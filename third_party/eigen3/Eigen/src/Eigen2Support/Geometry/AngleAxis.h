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
  * \class AngleAxis
  *
  * \brief Represents a 3D rotation as a rotation angle around an arbitrary 3D axis
  *
  * \param _Scalar the scalar type, i.e., the type of the coefficients.
  *
  * The following two typedefs are provided for convenience:
  * \li \c AngleAxisf for \c float
  * \li \c AngleAxisd for \c double
  *
  * \addexample AngleAxisForEuler \label How to define a rotation from Euler-angles
  *
  * Combined with MatrixBase::Unit{X,Y,Z}, AngleAxis can be used to easily
  * mimic Euler-angles. Here is an example:
  * \include AngleAxis_mimic_euler.cpp
  * Output: \verbinclude AngleAxis_mimic_euler.out
  *
  * \note This class is not aimed to be used to store a rotation transformation,
  * but rather to make easier the creation of other rotation (Quaternion, rotation Matrix)
  * and transformation objects.
  *
  * \sa class Quaternion, class Transform, MatrixBase::UnitX()
  */

template<typename _Scalar> struct ei_traits<AngleAxis<_Scalar> >
{
  typedef _Scalar Scalar;
};

template<typename _Scalar>
class AngleAxis : public RotationBase<AngleAxis<_Scalar>,3>
{
  typedef RotationBase<AngleAxis<_Scalar>,3> Base;

public:

  using Base::operator*;

  enum { Dim = 3 };
  /** the scalar type of the coefficients */
  typedef _Scalar Scalar;
  typedef Matrix<Scalar,3,3> Matrix3;
  typedef Matrix<Scalar,3,1> Vector3;
  typedef Quaternion<Scalar> QuaternionType;

protected:

  Vector3 m_axis;
  Scalar m_angle;

public:

  /** Default constructor without initialization. */
  AngleAxis() {}

  /** Constructs and initialize the angle-axis rotation from an \a angle in radian
    * and an \a axis which must be normalized. */
  template<typename Derived>
  inline AngleAxis(Scalar angle, const MatrixBase<Derived>& axis) : m_axis(axis), m_angle(angle)
  {
    using std::sqrt;
    using std::abs;
    // since we compare against 1, this is equal to computing the relative error
    eigen_assert( abs(m_axis.derived().squaredNorm() - 1) < sqrt( NumTraits<Scalar>::dummy_precision() ) );
  }

  /** Constructs and initialize the angle-axis rotation from a quaternion \a q. */
  inline AngleAxis(const QuaternionType& q) { *this = q; }

  /** Constructs and initialize the angle-axis rotation from a 3x3 rotation matrix. */
  template<typename Derived>
  inline explicit AngleAxis(const MatrixBase<Derived>& m) { *this = m; }

  Scalar angle() const { return m_angle; }
  Scalar& angle() { return m_angle; }

  const Vector3& axis() const { return m_axis; }
  Vector3& axis() { return m_axis; }

  /** Concatenates two rotations */
  inline QuaternionType operator* (const AngleAxis& other) const
  { return QuaternionType(*this) * QuaternionType(other); }

  /** Concatenates two rotations */
  inline QuaternionType operator* (const QuaternionType& other) const
  { return QuaternionType(*this) * other; }

  /** Concatenates two rotations */
  friend inline QuaternionType operator* (const QuaternionType& a, const AngleAxis& b)
  { return a * QuaternionType(b); }

  /** Concatenates two rotations */
  inline Matrix3 operator* (const Matrix3& other) const
  { return toRotationMatrix() * other; }

  /** Concatenates two rotations */
  inline friend Matrix3 operator* (const Matrix3& a, const AngleAxis& b)
  { return a * b.toRotationMatrix(); }

  /** Applies rotation to vector */
  inline Vector3 operator* (const Vector3& other) const
  { return toRotationMatrix() * other; }

  /** \returns the inverse rotation, i.e., an angle-axis with opposite rotation angle */
  AngleAxis inverse() const
  { return AngleAxis(-m_angle, m_axis); }

  AngleAxis& operator=(const QuaternionType& q);
  template<typename Derived>
  AngleAxis& operator=(const MatrixBase<Derived>& m);

  template<typename Derived>
  AngleAxis& fromRotationMatrix(const MatrixBase<Derived>& m);
  Matrix3 toRotationMatrix(void) const;

  /** \returns \c *this with scalar type casted to \a NewScalarType
    *
    * Note that if \a NewScalarType is equal to the current scalar type of \c *this
    * then this function smartly returns a const reference to \c *this.
    */
  template<typename NewScalarType>
  inline typename internal::cast_return_type<AngleAxis,AngleAxis<NewScalarType> >::type cast() const
  { return typename internal::cast_return_type<AngleAxis,AngleAxis<NewScalarType> >::type(*this); }

  /** Copy constructor with scalar type conversion */
  template<typename OtherScalarType>
  inline explicit AngleAxis(const AngleAxis<OtherScalarType>& other)
  {
    m_axis = other.axis().template cast<Scalar>();
    m_angle = Scalar(other.angle());
  }

  /** \returns \c true if \c *this is approximately equal to \a other, within the precision
    * determined by \a prec.
    *
    * \sa MatrixBase::isApprox() */
  bool isApprox(const AngleAxis& other, typename NumTraits<Scalar>::Real prec = precision<Scalar>()) const
  { return m_axis.isApprox(other.m_axis, prec) && ei_isApprox(m_angle,other.m_angle, prec); }
};

/** \ingroup Geometry_Module
  * single precision angle-axis type */
typedef AngleAxis<float> AngleAxisf;
/** \ingroup Geometry_Module
  * double precision angle-axis type */
typedef AngleAxis<double> AngleAxisd;

/** Set \c *this from a quaternion.
  * The axis is normalized.
  */
template<typename Scalar>
AngleAxis<Scalar>& AngleAxis<Scalar>::operator=(const QuaternionType& q)
{
  Scalar n2 = q.vec().squaredNorm();
  if (n2 < precision<Scalar>()*precision<Scalar>())
  {
    m_angle = 0;
    m_axis << 1, 0, 0;
  }
  else
  {
    m_angle = 2*std::acos(q.w());
    m_axis = q.vec() / ei_sqrt(n2);

    using std::sqrt;
    using std::abs;
    // since we compare against 1, this is equal to computing the relative error
    eigen_assert( abs(m_axis.derived().squaredNorm() - 1) < sqrt( NumTraits<Scalar>::dummy_precision() ) );
  }
  return *this;
}

/** Set \c *this from a 3x3 rotation matrix \a mat.
  */
template<typename Scalar>
template<typename Derived>
AngleAxis<Scalar>& AngleAxis<Scalar>::operator=(const MatrixBase<Derived>& mat)
{
  // Since a direct conversion would not be really faster,
  // let's use the robust Quaternion implementation:
  return *this = QuaternionType(mat);
}

/** Constructs and \returns an equivalent 3x3 rotation matrix.
  */
template<typename Scalar>
typename AngleAxis<Scalar>::Matrix3
AngleAxis<Scalar>::toRotationMatrix(void) const
{
  Matrix3 res;
  Vector3 sin_axis  = ei_sin(m_angle) * m_axis;
  Scalar c = ei_cos(m_angle);
  Vector3 cos1_axis = (Scalar(1)-c) * m_axis;

  Scalar tmp;
  tmp = cos1_axis.x() * m_axis.y();
  res.coeffRef(0,1) = tmp - sin_axis.z();
  res.coeffRef(1,0) = tmp + sin_axis.z();

  tmp = cos1_axis.x() * m_axis.z();
  res.coeffRef(0,2) = tmp + sin_axis.y();
  res.coeffRef(2,0) = tmp - sin_axis.y();

  tmp = cos1_axis.y() * m_axis.z();
  res.coeffRef(1,2) = tmp - sin_axis.x();
  res.coeffRef(2,1) = tmp + sin_axis.x();

  res.diagonal() = (cos1_axis.cwise() * m_axis).cwise() + c;

  return res;
}

} // end namespace Eigen
