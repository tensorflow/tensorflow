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
  * \class ParametrizedLine
  *
  * \brief A parametrized line
  *
  * A parametrized line is defined by an origin point \f$ \mathbf{o} \f$ and a unit
  * direction vector \f$ \mathbf{d} \f$ such that the line corresponds to
  * the set \f$ l(t) = \mathbf{o} + t \mathbf{d} \f$, \f$ l \in \mathbf{R} \f$.
  *
  * \param _Scalar the scalar type, i.e., the type of the coefficients
  * \param _AmbientDim the dimension of the ambient space, can be a compile time value or Dynamic.
  */
template <typename _Scalar, int _AmbientDim>
class ParametrizedLine
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW_IF_VECTORIZABLE_FIXED_SIZE(_Scalar,_AmbientDim)
  enum { AmbientDimAtCompileTime = _AmbientDim };
  typedef _Scalar Scalar;
  typedef typename NumTraits<Scalar>::Real RealScalar;
  typedef Matrix<Scalar,AmbientDimAtCompileTime,1> VectorType;

  /** Default constructor without initialization */
  inline ParametrizedLine() {}

  /** Constructs a dynamic-size line with \a _dim the dimension
    * of the ambient space */
  inline explicit ParametrizedLine(int _dim) : m_origin(_dim), m_direction(_dim) {}

  /** Initializes a parametrized line of direction \a direction and origin \a origin.
    * \warning the vector direction is assumed to be normalized.
    */
  ParametrizedLine(const VectorType& origin, const VectorType& direction)
    : m_origin(origin), m_direction(direction) {}

  explicit ParametrizedLine(const Hyperplane<_Scalar, _AmbientDim>& hyperplane);

  /** Constructs a parametrized line going from \a p0 to \a p1. */
  static inline ParametrizedLine Through(const VectorType& p0, const VectorType& p1)
  { return ParametrizedLine(p0, (p1-p0).normalized()); }

  ~ParametrizedLine() {}

  /** \returns the dimension in which the line holds */
  inline int dim() const { return m_direction.size(); }

  const VectorType& origin() const { return m_origin; }
  VectorType& origin() { return m_origin; }

  const VectorType& direction() const { return m_direction; }
  VectorType& direction() { return m_direction; }

  /** \returns the squared distance of a point \a p to its projection onto the line \c *this.
    * \sa distance()
    */
  RealScalar squaredDistance(const VectorType& p) const
  {
    VectorType diff = p-origin();
    return (diff - diff.eigen2_dot(direction())* direction()).squaredNorm();
  }
  /** \returns the distance of a point \a p to its projection onto the line \c *this.
    * \sa squaredDistance()
    */
  RealScalar distance(const VectorType& p) const { return ei_sqrt(squaredDistance(p)); }

  /** \returns the projection of a point \a p onto the line \c *this. */
  VectorType projection(const VectorType& p) const
  { return origin() + (p-origin()).eigen2_dot(direction()) * direction(); }

  Scalar intersection(const Hyperplane<_Scalar, _AmbientDim>& hyperplane);

  /** \returns \c *this with scalar type casted to \a NewScalarType
    *
    * Note that if \a NewScalarType is equal to the current scalar type of \c *this
    * then this function smartly returns a const reference to \c *this.
    */
  template<typename NewScalarType>
  inline typename internal::cast_return_type<ParametrizedLine,
           ParametrizedLine<NewScalarType,AmbientDimAtCompileTime> >::type cast() const
  {
    return typename internal::cast_return_type<ParametrizedLine,
                    ParametrizedLine<NewScalarType,AmbientDimAtCompileTime> >::type(*this);
  }

  /** Copy constructor with scalar type conversion */
  template<typename OtherScalarType>
  inline explicit ParametrizedLine(const ParametrizedLine<OtherScalarType,AmbientDimAtCompileTime>& other)
  {
    m_origin = other.origin().template cast<Scalar>();
    m_direction = other.direction().template cast<Scalar>();
  }

  /** \returns \c true if \c *this is approximately equal to \a other, within the precision
    * determined by \a prec.
    *
    * \sa MatrixBase::isApprox() */
  bool isApprox(const ParametrizedLine& other, typename NumTraits<Scalar>::Real prec = precision<Scalar>()) const
  { return m_origin.isApprox(other.m_origin, prec) && m_direction.isApprox(other.m_direction, prec); }

protected:

  VectorType m_origin, m_direction;
};

/** Constructs a parametrized line from a 2D hyperplane
  *
  * \warning the ambient space must have dimension 2 such that the hyperplane actually describes a line
  */
template <typename _Scalar, int _AmbientDim>
inline ParametrizedLine<_Scalar, _AmbientDim>::ParametrizedLine(const Hyperplane<_Scalar, _AmbientDim>& hyperplane)
{
  EIGEN_STATIC_ASSERT_VECTOR_SPECIFIC_SIZE(VectorType, 2)
  direction() = hyperplane.normal().unitOrthogonal();
  origin() = -hyperplane.normal()*hyperplane.offset();
}

/** \returns the parameter value of the intersection between \c *this and the given hyperplane
  */
template <typename _Scalar, int _AmbientDim>
inline _Scalar ParametrizedLine<_Scalar, _AmbientDim>::intersection(const Hyperplane<_Scalar, _AmbientDim>& hyperplane)
{
  return -(hyperplane.offset()+origin().eigen2_dot(hyperplane.normal()))
          /(direction().eigen2_dot(hyperplane.normal()));
}

} // end namespace Eigen
