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

template<typename Other,
         int OtherRows=Other::RowsAtCompileTime,
         int OtherCols=Other::ColsAtCompileTime>
struct ei_quaternion_assign_impl;

/** \geometry_module \ingroup Geometry_Module
  *
  * \class Quaternion
  *
  * \brief The quaternion class used to represent 3D orientations and rotations
  *
  * \param _Scalar the scalar type, i.e., the type of the coefficients
  *
  * This class represents a quaternion \f$ w+xi+yj+zk \f$ that is a convenient representation of
  * orientations and rotations of objects in three dimensions. Compared to other representations
  * like Euler angles or 3x3 matrices, quatertions offer the following advantages:
  * \li \b compact storage (4 scalars)
  * \li \b efficient to compose (28 flops),
  * \li \b stable spherical interpolation
  *
  * The following two typedefs are provided for convenience:
  * \li \c Quaternionf for \c float
  * \li \c Quaterniond for \c double
  *
  * \sa  class AngleAxis, class Transform
  */

template<typename _Scalar> struct ei_traits<Quaternion<_Scalar> >
{
  typedef _Scalar Scalar;
};

template<typename _Scalar>
class Quaternion : public RotationBase<Quaternion<_Scalar>,3>
{
  typedef RotationBase<Quaternion<_Scalar>,3> Base;

public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW_IF_VECTORIZABLE_FIXED_SIZE(_Scalar,4)

  using Base::operator*;

  /** the scalar type of the coefficients */
  typedef _Scalar Scalar;

  /** the type of the Coefficients 4-vector */
  typedef Matrix<Scalar, 4, 1> Coefficients;
  /** the type of a 3D vector */
  typedef Matrix<Scalar,3,1> Vector3;
  /** the equivalent rotation matrix type */
  typedef Matrix<Scalar,3,3> Matrix3;
  /** the equivalent angle-axis type */
  typedef AngleAxis<Scalar> AngleAxisType;

  /** \returns the \c x coefficient */
  inline Scalar x() const { return m_coeffs.coeff(0); }
  /** \returns the \c y coefficient */
  inline Scalar y() const { return m_coeffs.coeff(1); }
  /** \returns the \c z coefficient */
  inline Scalar z() const { return m_coeffs.coeff(2); }
  /** \returns the \c w coefficient */
  inline Scalar w() const { return m_coeffs.coeff(3); }

  /** \returns a reference to the \c x coefficient */
  inline Scalar& x() { return m_coeffs.coeffRef(0); }
  /** \returns a reference to the \c y coefficient */
  inline Scalar& y() { return m_coeffs.coeffRef(1); }
  /** \returns a reference to the \c z coefficient */
  inline Scalar& z() { return m_coeffs.coeffRef(2); }
  /** \returns a reference to the \c w coefficient */
  inline Scalar& w() { return m_coeffs.coeffRef(3); }

  /** \returns a read-only vector expression of the imaginary part (x,y,z) */
  inline const Block<const Coefficients,3,1> vec() const { return m_coeffs.template start<3>(); }

  /** \returns a vector expression of the imaginary part (x,y,z) */
  inline Block<Coefficients,3,1> vec() { return m_coeffs.template start<3>(); }

  /** \returns a read-only vector expression of the coefficients (x,y,z,w) */
  inline const Coefficients& coeffs() const { return m_coeffs; }

  /** \returns a vector expression of the coefficients (x,y,z,w) */
  inline Coefficients& coeffs() { return m_coeffs; }

  /** Default constructor leaving the quaternion uninitialized. */
  inline Quaternion() {}

  /** Constructs and initializes the quaternion \f$ w+xi+yj+zk \f$ from
    * its four coefficients \a w, \a x, \a y and \a z.
    *
    * \warning Note the order of the arguments: the real \a w coefficient first,
    * while internally the coefficients are stored in the following order:
    * [\c x, \c y, \c z, \c w]
    */
  inline Quaternion(Scalar w, Scalar x, Scalar y, Scalar z)
  { m_coeffs << x, y, z, w; }

  /** Copy constructor */
  inline Quaternion(const Quaternion& other) { m_coeffs = other.m_coeffs; }

  /** Constructs and initializes a quaternion from the angle-axis \a aa */
  explicit inline Quaternion(const AngleAxisType& aa) { *this = aa; }

  /** Constructs and initializes a quaternion from either:
    *  - a rotation matrix expression,
    *  - a 4D vector expression representing quaternion coefficients.
    * \sa operator=(MatrixBase<Derived>)
    */
  template<typename Derived>
  explicit inline Quaternion(const MatrixBase<Derived>& other) { *this = other; }

  Quaternion& operator=(const Quaternion& other);
  Quaternion& operator=(const AngleAxisType& aa);
  template<typename Derived>
  Quaternion& operator=(const MatrixBase<Derived>& m);

  /** \returns a quaternion representing an identity rotation
    * \sa MatrixBase::Identity()
    */
  static inline Quaternion Identity() { return Quaternion(1, 0, 0, 0); }

  /** \sa Quaternion::Identity(), MatrixBase::setIdentity()
    */
  inline Quaternion& setIdentity() { m_coeffs << 0, 0, 0, 1; return *this; }

  /** \returns the squared norm of the quaternion's coefficients
    * \sa Quaternion::norm(), MatrixBase::squaredNorm()
    */
  inline Scalar squaredNorm() const { return m_coeffs.squaredNorm(); }

  /** \returns the norm of the quaternion's coefficients
    * \sa Quaternion::squaredNorm(), MatrixBase::norm()
    */
  inline Scalar norm() const { return m_coeffs.norm(); }

  /** Normalizes the quaternion \c *this
    * \sa normalized(), MatrixBase::normalize() */
  inline void normalize() { m_coeffs.normalize(); }
  /** \returns a normalized version of \c *this
    * \sa normalize(), MatrixBase::normalized() */
  inline Quaternion normalized() const { return Quaternion(m_coeffs.normalized()); }

  /** \returns the dot product of \c *this and \a other
    * Geometrically speaking, the dot product of two unit quaternions
    * corresponds to the cosine of half the angle between the two rotations.
    * \sa angularDistance()
    */
  inline Scalar eigen2_dot(const Quaternion& other) const { return m_coeffs.eigen2_dot(other.m_coeffs); }

  inline Scalar angularDistance(const Quaternion& other) const;

  Matrix3 toRotationMatrix(void) const;

  template<typename Derived1, typename Derived2>
  Quaternion& setFromTwoVectors(const MatrixBase<Derived1>& a, const MatrixBase<Derived2>& b);

  inline Quaternion operator* (const Quaternion& q) const;
  inline Quaternion& operator*= (const Quaternion& q);

  Quaternion inverse(void) const;
  Quaternion conjugate(void) const;

  Quaternion slerp(Scalar t, const Quaternion& other) const;

  template<typename Derived>
  Vector3 operator* (const MatrixBase<Derived>& vec) const;

  /** \returns \c *this with scalar type casted to \a NewScalarType
    *
    * Note that if \a NewScalarType is equal to the current scalar type of \c *this
    * then this function smartly returns a const reference to \c *this.
    */
  template<typename NewScalarType>
  inline typename internal::cast_return_type<Quaternion,Quaternion<NewScalarType> >::type cast() const
  { return typename internal::cast_return_type<Quaternion,Quaternion<NewScalarType> >::type(*this); }

  /** Copy constructor with scalar type conversion */
  template<typename OtherScalarType>
  inline explicit Quaternion(const Quaternion<OtherScalarType>& other)
  { m_coeffs = other.coeffs().template cast<Scalar>(); }

  /** \returns \c true if \c *this is approximately equal to \a other, within the precision
    * determined by \a prec.
    *
    * \sa MatrixBase::isApprox() */
  bool isApprox(const Quaternion& other, typename NumTraits<Scalar>::Real prec = precision<Scalar>()) const
  { return m_coeffs.isApprox(other.m_coeffs, prec); }

protected:
  Coefficients m_coeffs;
};

/** \ingroup Geometry_Module
  * single precision quaternion type */
typedef Quaternion<float> Quaternionf;
/** \ingroup Geometry_Module
  * double precision quaternion type */
typedef Quaternion<double> Quaterniond;

// Generic Quaternion * Quaternion product
template<typename Scalar> inline Quaternion<Scalar>
ei_quaternion_product(const Quaternion<Scalar>& a, const Quaternion<Scalar>& b)
{
  return Quaternion<Scalar>
  (
    a.w() * b.w() - a.x() * b.x() - a.y() * b.y() - a.z() * b.z(),
    a.w() * b.x() + a.x() * b.w() + a.y() * b.z() - a.z() * b.y(),
    a.w() * b.y() + a.y() * b.w() + a.z() * b.x() - a.x() * b.z(),
    a.w() * b.z() + a.z() * b.w() + a.x() * b.y() - a.y() * b.x()
  );
}

/** \returns the concatenation of two rotations as a quaternion-quaternion product */
template <typename Scalar>
inline Quaternion<Scalar> Quaternion<Scalar>::operator* (const Quaternion& other) const
{
  return ei_quaternion_product(*this,other);
}

/** \sa operator*(Quaternion) */
template <typename Scalar>
inline Quaternion<Scalar>& Quaternion<Scalar>::operator*= (const Quaternion& other)
{
  return (*this = *this * other);
}

/** Rotation of a vector by a quaternion.
  * \remarks If the quaternion is used to rotate several points (>1)
  * then it is much more efficient to first convert it to a 3x3 Matrix.
  * Comparison of the operation cost for n transformations:
  *   - Quaternion:    30n
  *   - Via a Matrix3: 24 + 15n
  */
template <typename Scalar>
template<typename Derived>
inline typename Quaternion<Scalar>::Vector3
Quaternion<Scalar>::operator* (const MatrixBase<Derived>& v) const
{
    // Note that this algorithm comes from the optimization by hand
    // of the conversion to a Matrix followed by a Matrix/Vector product.
    // It appears to be much faster than the common algorithm found
    // in the litterature (30 versus 39 flops). It also requires two
    // Vector3 as temporaries.
    Vector3 uv;
    uv = 2 * this->vec().cross(v);
    return v + this->w() * uv + this->vec().cross(uv);
}

template<typename Scalar>
inline Quaternion<Scalar>& Quaternion<Scalar>::operator=(const Quaternion& other)
{
  m_coeffs = other.m_coeffs;
  return *this;
}

/** Set \c *this from an angle-axis \a aa and returns a reference to \c *this
  */
template<typename Scalar>
inline Quaternion<Scalar>& Quaternion<Scalar>::operator=(const AngleAxisType& aa)
{
  Scalar ha = Scalar(0.5)*aa.angle(); // Scalar(0.5) to suppress precision loss warnings
  this->w() = ei_cos(ha);
  this->vec() = ei_sin(ha) * aa.axis();
  return *this;
}

/** Set \c *this from the expression \a xpr:
  *   - if \a xpr is a 4x1 vector, then \a xpr is assumed to be a quaternion
  *   - if \a xpr is a 3x3 matrix, then \a xpr is assumed to be rotation matrix
  *     and \a xpr is converted to a quaternion
  */
template<typename Scalar>
template<typename Derived>
inline Quaternion<Scalar>& Quaternion<Scalar>::operator=(const MatrixBase<Derived>& xpr)
{
  ei_quaternion_assign_impl<Derived>::run(*this, xpr.derived());
  return *this;
}

/** Convert the quaternion to a 3x3 rotation matrix */
template<typename Scalar>
inline typename Quaternion<Scalar>::Matrix3
Quaternion<Scalar>::toRotationMatrix(void) const
{
  // NOTE if inlined, then gcc 4.2 and 4.4 get rid of the temporary (not gcc 4.3 !!)
  // if not inlined then the cost of the return by value is huge ~ +35%,
  // however, not inlining this function is an order of magnitude slower, so
  // it has to be inlined, and so the return by value is not an issue
  Matrix3 res;

  const Scalar tx  = Scalar(2)*this->x();
  const Scalar ty  = Scalar(2)*this->y();
  const Scalar tz  = Scalar(2)*this->z();
  const Scalar twx = tx*this->w();
  const Scalar twy = ty*this->w();
  const Scalar twz = tz*this->w();
  const Scalar txx = tx*this->x();
  const Scalar txy = ty*this->x();
  const Scalar txz = tz*this->x();
  const Scalar tyy = ty*this->y();
  const Scalar tyz = tz*this->y();
  const Scalar tzz = tz*this->z();

  res.coeffRef(0,0) = Scalar(1)-(tyy+tzz);
  res.coeffRef(0,1) = txy-twz;
  res.coeffRef(0,2) = txz+twy;
  res.coeffRef(1,0) = txy+twz;
  res.coeffRef(1,1) = Scalar(1)-(txx+tzz);
  res.coeffRef(1,2) = tyz-twx;
  res.coeffRef(2,0) = txz-twy;
  res.coeffRef(2,1) = tyz+twx;
  res.coeffRef(2,2) = Scalar(1)-(txx+tyy);

  return res;
}

/** Sets *this to be a quaternion representing a rotation sending the vector \a a to the vector \a b.
  *
  * \returns a reference to *this.
  *
  * Note that the two input vectors do \b not have to be normalized.
  */
template<typename Scalar>
template<typename Derived1, typename Derived2>
inline Quaternion<Scalar>& Quaternion<Scalar>::setFromTwoVectors(const MatrixBase<Derived1>& a, const MatrixBase<Derived2>& b)
{
  Vector3 v0 = a.normalized();
  Vector3 v1 = b.normalized();
  Scalar c = v0.eigen2_dot(v1);

  // if dot == 1, vectors are the same
  if (ei_isApprox(c,Scalar(1)))
  {
    // set to identity
    this->w() = 1; this->vec().setZero();
    return *this;
  }
  // if dot == -1, vectors are opposites
  if (ei_isApprox(c,Scalar(-1)))
  {
    this->vec() = v0.unitOrthogonal();
    this->w() = 0;
    return *this;
  }

  Vector3 axis = v0.cross(v1);
  Scalar s = ei_sqrt((Scalar(1)+c)*Scalar(2));
  Scalar invs = Scalar(1)/s;
  this->vec() = axis * invs;
  this->w() = s * Scalar(0.5);

  return *this;
}

/** \returns the multiplicative inverse of \c *this
  * Note that in most cases, i.e., if you simply want the opposite rotation,
  * and/or the quaternion is normalized, then it is enough to use the conjugate.
  *
  * \sa Quaternion::conjugate()
  */
template <typename Scalar>
inline Quaternion<Scalar> Quaternion<Scalar>::inverse() const
{
  // FIXME should this function be called multiplicativeInverse and conjugate() be called inverse() or opposite()  ??
  Scalar n2 = this->squaredNorm();
  if (n2 > 0)
    return Quaternion(conjugate().coeffs() / n2);
  else
  {
    // return an invalid result to flag the error
    return Quaternion(Coefficients::Zero());
  }
}

/** \returns the conjugate of the \c *this which is equal to the multiplicative inverse
  * if the quaternion is normalized.
  * The conjugate of a quaternion represents the opposite rotation.
  *
  * \sa Quaternion::inverse()
  */
template <typename Scalar>
inline Quaternion<Scalar> Quaternion<Scalar>::conjugate() const
{
  return Quaternion(this->w(),-this->x(),-this->y(),-this->z());
}

/** \returns the angle (in radian) between two rotations
  * \sa eigen2_dot()
  */
template <typename Scalar>
inline Scalar Quaternion<Scalar>::angularDistance(const Quaternion& other) const
{
  double d = ei_abs(this->eigen2_dot(other));
  if (d>=1.0)
    return 0;
  return Scalar(2) * std::acos(d);
}

/** \returns the spherical linear interpolation between the two quaternions
  * \c *this and \a other at the parameter \a t
  */
template <typename Scalar>
Quaternion<Scalar> Quaternion<Scalar>::slerp(Scalar t, const Quaternion& other) const
{
  static const Scalar one = Scalar(1) - machine_epsilon<Scalar>();
  Scalar d = this->eigen2_dot(other);
  Scalar absD = ei_abs(d);

  Scalar scale0;
  Scalar scale1;

  if (absD>=one)
  {
    scale0 = Scalar(1) - t;
    scale1 = t;
  }
  else
  {
    // theta is the angle between the 2 quaternions
    Scalar theta = std::acos(absD);
    Scalar sinTheta = ei_sin(theta);

    scale0 = ei_sin( ( Scalar(1) - t ) * theta) / sinTheta;
    scale1 = ei_sin( ( t * theta) ) / sinTheta;
    if (d<0)
      scale1 = -scale1;
  }

  return Quaternion<Scalar>(scale0 * coeffs() + scale1 * other.coeffs());
}

// set from a rotation matrix
template<typename Other>
struct ei_quaternion_assign_impl<Other,3,3>
{
  typedef typename Other::Scalar Scalar;
  static inline void run(Quaternion<Scalar>& q, const Other& mat)
  {
    // This algorithm comes from  "Quaternion Calculus and Fast Animation",
    // Ken Shoemake, 1987 SIGGRAPH course notes
    Scalar t = mat.trace();
    if (t > 0)
    {
      t = ei_sqrt(t + Scalar(1.0));
      q.w() = Scalar(0.5)*t;
      t = Scalar(0.5)/t;
      q.x() = (mat.coeff(2,1) - mat.coeff(1,2)) * t;
      q.y() = (mat.coeff(0,2) - mat.coeff(2,0)) * t;
      q.z() = (mat.coeff(1,0) - mat.coeff(0,1)) * t;
    }
    else
    {
      int i = 0;
      if (mat.coeff(1,1) > mat.coeff(0,0))
        i = 1;
      if (mat.coeff(2,2) > mat.coeff(i,i))
        i = 2;
      int j = (i+1)%3;
      int k = (j+1)%3;

      t = ei_sqrt(mat.coeff(i,i)-mat.coeff(j,j)-mat.coeff(k,k) + Scalar(1.0));
      q.coeffs().coeffRef(i) = Scalar(0.5) * t;
      t = Scalar(0.5)/t;
      q.w() = (mat.coeff(k,j)-mat.coeff(j,k))*t;
      q.coeffs().coeffRef(j) = (mat.coeff(j,i)+mat.coeff(i,j))*t;
      q.coeffs().coeffRef(k) = (mat.coeff(k,i)+mat.coeff(i,k))*t;
    }
  }
};

// set from a vector of coefficients assumed to be a quaternion
template<typename Other>
struct ei_quaternion_assign_impl<Other,4,1>
{
  typedef typename Other::Scalar Scalar;
  static inline void run(Quaternion<Scalar>& q, const Other& vec)
  {
    q.coeffs() = vec;
  }
};

} // end namespace Eigen
