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
  * \nonstableyet
  *
  * \class AlignedBox
  *
  * \brief An axis aligned box
  *
  * \param _Scalar the type of the scalar coefficients
  * \param _AmbientDim the dimension of the ambient space, can be a compile time value or Dynamic.
  *
  * This class represents an axis aligned box as a pair of the minimal and maximal corners.
  */
template <typename _Scalar, int _AmbientDim>
class AlignedBox
{
public:
EIGEN_MAKE_ALIGNED_OPERATOR_NEW_IF_VECTORIZABLE_FIXED_SIZE(_Scalar,_AmbientDim==Dynamic ? Dynamic : _AmbientDim+1)
  enum { AmbientDimAtCompileTime = _AmbientDim };
  typedef _Scalar Scalar;
  typedef typename NumTraits<Scalar>::Real RealScalar;
  typedef Matrix<Scalar,AmbientDimAtCompileTime,1> VectorType;

  /** Default constructor initializing a null box. */
  inline AlignedBox()
  { if (AmbientDimAtCompileTime!=Dynamic) setNull(); }

  /** Constructs a null box with \a _dim the dimension of the ambient space. */
  inline explicit AlignedBox(int _dim) : m_min(_dim), m_max(_dim)
  { setNull(); }

  /** Constructs a box with extremities \a _min and \a _max. */
  inline AlignedBox(const VectorType& _min, const VectorType& _max) : m_min(_min), m_max(_max) {}

  /** Constructs a box containing a single point \a p. */
  inline explicit AlignedBox(const VectorType& p) : m_min(p), m_max(p) {}

  ~AlignedBox() {}

  /** \returns the dimension in which the box holds */
  inline int dim() const { return AmbientDimAtCompileTime==Dynamic ? m_min.size()-1 : AmbientDimAtCompileTime; }

  /** \returns true if the box is null, i.e, empty. */
  inline bool isNull() const { return (m_min.cwise() > m_max).any(); }

  /** Makes \c *this a null/empty box. */
  inline void setNull()
  {
    m_min.setConstant( (std::numeric_limits<Scalar>::max)());
    m_max.setConstant(-(std::numeric_limits<Scalar>::max)());
  }

  /** \returns the minimal corner */
  inline const VectorType& (min)() const { return m_min; }
  /** \returns a non const reference to the minimal corner */
  inline VectorType& (min)() { return m_min; }
  /** \returns the maximal corner */
  inline const VectorType& (max)() const { return m_max; }
  /** \returns a non const reference to the maximal corner */
  inline VectorType& (max)() { return m_max; }

  /** \returns true if the point \a p is inside the box \c *this. */
  inline bool contains(const VectorType& p) const
  { return (m_min.cwise()<=p).all() && (p.cwise()<=m_max).all(); }

  /** \returns true if the box \a b is entirely inside the box \c *this. */
  inline bool contains(const AlignedBox& b) const
  { return (m_min.cwise()<=(b.min)()).all() && ((b.max)().cwise()<=m_max).all(); }

  /** Extends \c *this such that it contains the point \a p and returns a reference to \c *this. */
  inline AlignedBox& extend(const VectorType& p)
  { m_min = (m_min.cwise().min)(p); m_max = (m_max.cwise().max)(p); return *this; }

  /** Extends \c *this such that it contains the box \a b and returns a reference to \c *this. */
  inline AlignedBox& extend(const AlignedBox& b)
  { m_min = (m_min.cwise().min)(b.m_min); m_max = (m_max.cwise().max)(b.m_max); return *this; }

  /** Clamps \c *this by the box \a b and returns a reference to \c *this. */
  inline AlignedBox& clamp(const AlignedBox& b)
  { m_min = (m_min.cwise().max)(b.m_min); m_max = (m_max.cwise().min)(b.m_max); return *this; }

  /** Translate \c *this by the vector \a t and returns a reference to \c *this. */
  inline AlignedBox& translate(const VectorType& t)
  { m_min += t; m_max += t; return *this; }

  /** \returns the squared distance between the point \a p and the box \c *this,
    * and zero if \a p is inside the box.
    * \sa exteriorDistance()
    */
  inline Scalar squaredExteriorDistance(const VectorType& p) const;

  /** \returns the distance between the point \a p and the box \c *this,
    * and zero if \a p is inside the box.
    * \sa squaredExteriorDistance()
    */
  inline Scalar exteriorDistance(const VectorType& p) const
  { return ei_sqrt(squaredExteriorDistance(p)); }

  /** \returns \c *this with scalar type casted to \a NewScalarType
    *
    * Note that if \a NewScalarType is equal to the current scalar type of \c *this
    * then this function smartly returns a const reference to \c *this.
    */
  template<typename NewScalarType>
  inline typename internal::cast_return_type<AlignedBox,
           AlignedBox<NewScalarType,AmbientDimAtCompileTime> >::type cast() const
  {
    return typename internal::cast_return_type<AlignedBox,
                    AlignedBox<NewScalarType,AmbientDimAtCompileTime> >::type(*this);
  }

  /** Copy constructor with scalar type conversion */
  template<typename OtherScalarType>
  inline explicit AlignedBox(const AlignedBox<OtherScalarType,AmbientDimAtCompileTime>& other)
  {
    m_min = (other.min)().template cast<Scalar>();
    m_max = (other.max)().template cast<Scalar>();
  }

  /** \returns \c true if \c *this is approximately equal to \a other, within the precision
    * determined by \a prec.
    *
    * \sa MatrixBase::isApprox() */
  bool isApprox(const AlignedBox& other, typename NumTraits<Scalar>::Real prec = precision<Scalar>()) const
  { return m_min.isApprox(other.m_min, prec) && m_max.isApprox(other.m_max, prec); }

protected:

  VectorType m_min, m_max;
};

template<typename Scalar,int AmbiantDim>
inline Scalar AlignedBox<Scalar,AmbiantDim>::squaredExteriorDistance(const VectorType& p) const
{
  Scalar dist2(0);
  Scalar aux;
  for (int k=0; k<dim(); ++k)
  {
    if ((aux = (p[k]-m_min[k]))<Scalar(0))
      dist2 += aux*aux;
    else if ( (aux = (m_max[k]-p[k]))<Scalar(0))
      dist2 += aux*aux;
  }
  return dist2;
}

} // end namespace Eigen
