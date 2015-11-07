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

// this file aims to contains the various representations of rotation/orientation
// in 2D and 3D space excepted Matrix and Quaternion.

/** \class RotationBase
  *
  * \brief Common base class for compact rotation representations
  *
  * \param Derived is the derived type, i.e., a rotation type
  * \param _Dim the dimension of the space
  */
template<typename Derived, int _Dim>
class RotationBase
{
  public:
    enum { Dim = _Dim };
    /** the scalar type of the coefficients */
    typedef typename ei_traits<Derived>::Scalar Scalar;
    
    /** corresponding linear transformation matrix type */
    typedef Matrix<Scalar,Dim,Dim> RotationMatrixType;

    inline const Derived& derived() const { return *static_cast<const Derived*>(this); }
    inline Derived& derived() { return *static_cast<Derived*>(this); }

    /** \returns an equivalent rotation matrix */
    inline RotationMatrixType toRotationMatrix() const { return derived().toRotationMatrix(); }

    /** \returns the inverse rotation */
    inline Derived inverse() const { return derived().inverse(); }

    /** \returns the concatenation of the rotation \c *this with a translation \a t */
    inline Transform<Scalar,Dim> operator*(const Translation<Scalar,Dim>& t) const
    { return toRotationMatrix() * t; }

    /** \returns the concatenation of the rotation \c *this with a scaling \a s */
    inline RotationMatrixType operator*(const Scaling<Scalar,Dim>& s) const
    { return toRotationMatrix() * s; }

    /** \returns the concatenation of the rotation \c *this with an affine transformation \a t */
    inline Transform<Scalar,Dim> operator*(const Transform<Scalar,Dim>& t) const
    { return toRotationMatrix() * t; }
};

/** \geometry_module
  *
  * Constructs a Dim x Dim rotation matrix from the rotation \a r
  */
template<typename _Scalar, int _Rows, int _Cols, int _Storage, int _MaxRows, int _MaxCols>
template<typename OtherDerived>
Matrix<_Scalar, _Rows, _Cols, _Storage, _MaxRows, _MaxCols>
::Matrix(const RotationBase<OtherDerived,ColsAtCompileTime>& r)
{
  EIGEN_STATIC_ASSERT_MATRIX_SPECIFIC_SIZE(Matrix,int(OtherDerived::Dim),int(OtherDerived::Dim))
  *this = r.toRotationMatrix();
}

/** \geometry_module
  *
  * Set a Dim x Dim rotation matrix from the rotation \a r
  */
template<typename _Scalar, int _Rows, int _Cols, int _Storage, int _MaxRows, int _MaxCols>
template<typename OtherDerived>
Matrix<_Scalar, _Rows, _Cols, _Storage, _MaxRows, _MaxCols>&
Matrix<_Scalar, _Rows, _Cols, _Storage, _MaxRows, _MaxCols>
::operator=(const RotationBase<OtherDerived,ColsAtCompileTime>& r)
{
  EIGEN_STATIC_ASSERT_MATRIX_SPECIFIC_SIZE(Matrix,int(OtherDerived::Dim),int(OtherDerived::Dim))
  return *this = r.toRotationMatrix();
}

/** \internal
  *
  * Helper function to return an arbitrary rotation object to a rotation matrix.
  *
  * \param Scalar the numeric type of the matrix coefficients
  * \param Dim the dimension of the current space
  *
  * It returns a Dim x Dim fixed size matrix.
  *
  * Default specializations are provided for:
  *   - any scalar type (2D),
  *   - any matrix expression,
  *   - any type based on RotationBase (e.g., Quaternion, AngleAxis, Rotation2D)
  *
  * Currently ei_toRotationMatrix is only used by Transform.
  *
  * \sa class Transform, class Rotation2D, class Quaternion, class AngleAxis
  */
template<typename Scalar, int Dim>
static inline Matrix<Scalar,2,2> ei_toRotationMatrix(const Scalar& s)
{
  EIGEN_STATIC_ASSERT(Dim==2,YOU_MADE_A_PROGRAMMING_MISTAKE)
  return Rotation2D<Scalar>(s).toRotationMatrix();
}

template<typename Scalar, int Dim, typename OtherDerived>
static inline Matrix<Scalar,Dim,Dim> ei_toRotationMatrix(const RotationBase<OtherDerived,Dim>& r)
{
  return r.toRotationMatrix();
}

template<typename Scalar, int Dim, typename OtherDerived>
static inline const MatrixBase<OtherDerived>& ei_toRotationMatrix(const MatrixBase<OtherDerived>& mat)
{
  EIGEN_STATIC_ASSERT(OtherDerived::RowsAtCompileTime==Dim && OtherDerived::ColsAtCompileTime==Dim,
    YOU_MADE_A_PROGRAMMING_MISTAKE)
  return mat;
}

} // end namespace Eigen
