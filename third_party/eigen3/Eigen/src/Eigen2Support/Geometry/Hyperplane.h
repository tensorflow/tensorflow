// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008 Gael Guennebaud <g.gael@free.fr>
// Copyright (C) 2008 Benoit Jacob <jacob.benoit.1@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

// no include guard, we'll include this twice from All.h from Eigen2Support, and it's internal anyway

namespace Eigen { 

/** \geometry_module \ingroup Geometry_Module
  *
  * \class Hyperplane
  *
  * \brief A hyperplane
  *
  * A hyperplane is an affine subspace of dimension n-1 in a space of dimension n.
  * For example, a hyperplane in a plane is a line; a hyperplane in 3-space is a plane.
  *
  * \param _Scalar the scalar type, i.e., the type of the coefficients
  * \param _AmbientDim the dimension of the ambient space, can be a compile time value or Dynamic.
  *             Notice that the dimension of the hyperplane is _AmbientDim-1.
  *
  * This class represents an hyperplane as the zero set of the implicit equation
  * \f$ n \cdot x + d = 0 \f$ where \f$ n \f$ is a unit normal vector of the plane (linear part)
  * and \f$ d \f$ is the distance (offset) to the origin.
  */
template <typename _Scalar, int _AmbientDim>
class Hyperplane
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW_IF_VECTORIZABLE_FIXED_SIZE(_Scalar,_AmbientDim==Dynamic ? Dynamic : _AmbientDim+1)
  enum { AmbientDimAtCompileTime = _AmbientDim };
  typedef _Scalar Scalar;
  typedef typename NumTraits<Scalar>::Real RealScalar;
  typedef Matrix<Scalar,AmbientDimAtCompileTime,1> VectorType;
  typedef Matrix<Scalar,int(AmbientDimAtCompileTime)==Dynamic
                        ? Dynamic
                        : int(AmbientDimAtCompileTime)+1,1> Coefficients;
  typedef Block<Coefficients,AmbientDimAtCompileTime,1> NormalReturnType;

  /** Default constructor without initialization */
  inline Hyperplane() {}

  /** Constructs a dynamic-size hyperplane with \a _dim the dimension
    * of the ambient space */
  inline explicit Hyperplane(int _dim) : m_coeffs(_dim+1) {}

  /** Construct a plane from its normal \a n and a point \a e onto the plane.
    * \warning the vector normal is assumed to be normalized.
    */
  inline Hyperplane(const VectorType& n, const VectorType& e)
    : m_coeffs(n.size()+1)
  {
    normal() = n;
    offset() = -e.eigen2_dot(n);
  }

  /** Constructs a plane from its normal \a n and distance to the origin \a d
    * such that the algebraic equation of the plane is \f$ n \cdot x + d = 0 \f$.
    * \warning the vector normal is assumed to be normalized.
    */
  inline Hyperplane(const VectorType& n, Scalar d)
    : m_coeffs(n.size()+1)
  {
    normal() = n;
    offset() = d;
  }

  /** Constructs a hyperplane passing through the two points. If the dimension of the ambient space
    * is greater than 2, then there isn't uniqueness, so an arbitrary choice is made.
    */
  static inline Hyperplane Through(const VectorType& p0, const VectorType& p1)
  {
    Hyperplane result(p0.size());
    result.normal() = (p1 - p0).unitOrthogonal();
    result.offset() = -result.normal().eigen2_dot(p0);
    return result;
  }

  /** Constructs a hyperplane passing through the three points. The dimension of the ambient space
    * is required to be exactly 3.
    */
  static inline Hyperplane Through(const VectorType& p0, const VectorType& p1, const VectorType& p2)
  {
    EIGEN_STATIC_ASSERT_VECTOR_SPECIFIC_SIZE(VectorType, 3)
    Hyperplane result(p0.size());
    result.normal() = (p2 - p0).cross(p1 - p0).normalized();
    result.offset() = -result.normal().eigen2_dot(p0);
    return result;
  }

  /** Constructs a hyperplane passing through the parametrized line \a parametrized.
    * If the dimension of the ambient space is greater than 2, then there isn't uniqueness,
    * so an arbitrary choice is made.
    */
  // FIXME to be consitent with the rest this could be implemented as a static Through function ??
  explicit Hyperplane(const ParametrizedLine<Scalar, AmbientDimAtCompileTime>& parametrized)
  {
    normal() = parametrized.direction().unitOrthogonal();
    offset() = -normal().eigen2_dot(parametrized.origin());
  }

  ~Hyperplane() {}

  /** \returns the dimension in which the plane holds */
  inline int dim() const { return int(AmbientDimAtCompileTime)==Dynamic ? m_coeffs.size()-1 : int(AmbientDimAtCompileTime); }

  /** normalizes \c *this */
  void normalize(void)
  {
    m_coeffs /= normal().norm();
  }

  /** \returns the signed distance between the plane \c *this and a point \a p.
    * \sa absDistance()
    */
  inline Scalar signedDistance(const VectorType& p) const { return p.eigen2_dot(normal()) + offset(); }

  /** \returns the absolute distance between the plane \c *this and a point \a p.
    * \sa signedDistance()
    */
  inline Scalar absDistance(const VectorType& p) const { return ei_abs(signedDistance(p)); }

  /** \returns the projection of a point \a p onto the plane \c *this.
    */
  inline VectorType projection(const VectorType& p) const { return p - signedDistance(p) * normal(); }

  /** \returns a constant reference to the unit normal vector of the plane, which corresponds
    * to the linear part of the implicit equation.
    */
  inline const NormalReturnType normal() const { return NormalReturnType(*const_cast<Coefficients*>(&m_coeffs),0,0,dim(),1); }

  /** \returns a non-constant reference to the unit normal vector of the plane, which corresponds
    * to the linear part of the implicit equation.
    */
  inline NormalReturnType normal() { return NormalReturnType(m_coeffs,0,0,dim(),1); }

  /** \returns the distance to the origin, which is also the "constant term" of the implicit equation
    * \warning the vector normal is assumed to be normalized.
    */
  inline const Scalar& offset() const { return m_coeffs.coeff(dim()); }

  /** \returns a non-constant reference to the distance to the origin, which is also the constant part
    * of the implicit equation */
  inline Scalar& offset() { return m_coeffs(dim()); }

  /** \returns a constant reference to the coefficients c_i of the plane equation:
    * \f$ c_0*x_0 + ... + c_{d-1}*x_{d-1} + c_d = 0 \f$
    */
  inline const Coefficients& coeffs() const { return m_coeffs; }

  /** \returns a non-constant reference to the coefficients c_i of the plane equation:
    * \f$ c_0*x_0 + ... + c_{d-1}*x_{d-1} + c_d = 0 \f$
    */
  inline Coefficients& coeffs() { return m_coeffs; }

  /** \returns the intersection of *this with \a other.
    *
    * \warning The ambient space must be a plane, i.e. have dimension 2, so that \c *this and \a other are lines.
    *
    * \note If \a other is approximately parallel to *this, this method will return any point on *this.
    */
  VectorType intersection(const Hyperplane& other)
  {
    EIGEN_STATIC_ASSERT_VECTOR_SPECIFIC_SIZE(VectorType, 2)
    Scalar det = coeffs().coeff(0) * other.coeffs().coeff(1) - coeffs().coeff(1) * other.coeffs().coeff(0);
    // since the line equations ax+by=c are normalized with a^2+b^2=1, the following tests
    // whether the two lines are approximately parallel.
    if(ei_isMuchSmallerThan(det, Scalar(1)))
    {   // special case where the two lines are approximately parallel. Pick any point on the first line.
        if(ei_abs(coeffs().coeff(1))>ei_abs(coeffs().coeff(0)))
            return VectorType(coeffs().coeff(1), -coeffs().coeff(2)/coeffs().coeff(1)-coeffs().coeff(0));
        else
            return VectorType(-coeffs().coeff(2)/coeffs().coeff(0)-coeffs().coeff(1), coeffs().coeff(0));
    }
    else
    {   // general case
        Scalar invdet = Scalar(1) / det;
        return VectorType(invdet*(coeffs().coeff(1)*other.coeffs().coeff(2)-other.coeffs().coeff(1)*coeffs().coeff(2)),
                          invdet*(other.coeffs().coeff(0)*coeffs().coeff(2)-coeffs().coeff(0)*other.coeffs().coeff(2)));
    }
  }

  /** Applies the transformation matrix \a mat to \c *this and returns a reference to \c *this.
    *
    * \param mat the Dim x Dim transformation matrix
    * \param traits specifies whether the matrix \a mat represents an Isometry
    *               or a more generic Affine transformation. The default is Affine.
    */
  template<typename XprType>
  inline Hyperplane& transform(const MatrixBase<XprType>& mat, TransformTraits traits = Affine)
  {
    if (traits==Affine)
      normal() = mat.inverse().transpose() * normal();
    else if (traits==Isometry)
      normal() = mat * normal();
    else
    {
      ei_assert("invalid traits value in Hyperplane::transform()");
    }
    return *this;
  }

  /** Applies the transformation \a t to \c *this and returns a reference to \c *this.
    *
    * \param t the transformation of dimension Dim
    * \param traits specifies whether the transformation \a t represents an Isometry
    *               or a more generic Affine transformation. The default is Affine.
    *               Other kind of transformations are not supported.
    */
  inline Hyperplane& transform(const Transform<Scalar,AmbientDimAtCompileTime>& t,
                                TransformTraits traits = Affine)
  {
    transform(t.linear(), traits);
    offset() -= t.translation().eigen2_dot(normal());
    return *this;
  }

  /** \returns \c *this with scalar type casted to \a NewScalarType
    *
    * Note that if \a NewScalarType is equal to the current scalar type of \c *this
    * then this function smartly returns a const reference to \c *this.
    */
  template<typename NewScalarType>
  inline typename internal::cast_return_type<Hyperplane,
           Hyperplane<NewScalarType,AmbientDimAtCompileTime> >::type cast() const
  {
    return typename internal::cast_return_type<Hyperplane,
                    Hyperplane<NewScalarType,AmbientDimAtCompileTime> >::type(*this);
  }

  /** Copy constructor with scalar type conversion */
  template<typename OtherScalarType>
  inline explicit Hyperplane(const Hyperplane<OtherScalarType,AmbientDimAtCompileTime>& other)
  { m_coeffs = other.coeffs().template cast<Scalar>(); }

  /** \returns \c true if \c *this is approximately equal to \a other, within the precision
    * determined by \a prec.
    *
    * \sa MatrixBase::isApprox() */
  bool isApprox(const Hyperplane& other, typename NumTraits<Scalar>::Real prec = precision<Scalar>()) const
  { return m_coeffs.isApprox(other.m_coeffs, prec); }

protected:

  Coefficients m_coeffs;
};

} // end namespace Eigen
