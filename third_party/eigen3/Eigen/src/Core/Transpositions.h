// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2010-2011 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_TRANSPOSITIONS_H
#define EIGEN_TRANSPOSITIONS_H

namespace Eigen {

/** \class Transpositions
  * \ingroup Core_Module
  *
  * \brief Represents a sequence of transpositions (row/column interchange)
  *
  * \param SizeAtCompileTime the number of transpositions, or Dynamic
  * \param MaxSizeAtCompileTime the maximum number of transpositions, or Dynamic. This optional parameter defaults to SizeAtCompileTime. Most of the time, you should not have to specify it.
  *
  * This class represents a permutation transformation as a sequence of \em n transpositions
  * \f$[T_{n-1} \ldots T_{i} \ldots T_{0}]\f$. It is internally stored as a vector of integers \c indices.
  * Each transposition \f$ T_{i} \f$ applied on the left of a matrix (\f$ T_{i} M\f$) interchanges
  * the rows \c i and \c indices[i] of the matrix \c M.
  * A transposition applied on the right (e.g., \f$ M T_{i}\f$) yields a column interchange.
  *
  * Compared to the class PermutationMatrix, such a sequence of transpositions is what is
  * computed during a decomposition with pivoting, and it is faster when applying the permutation in-place.
  *
  * To apply a sequence of transpositions to a matrix, simply use the operator * as in the following example:
  * \code
  * Transpositions tr;
  * MatrixXf mat;
  * mat = tr * mat;
  * \endcode
  * In this example, we detect that the matrix appears on both side, and so the transpositions
  * are applied in-place without any temporary or extra copy.
  *
  * \sa class PermutationMatrix
  */

namespace internal {
template<typename TranspositionType, typename MatrixType, int Side, bool Transposed=false> struct transposition_matrix_product_retval;
}

template<typename Derived>
class TranspositionsBase
{
    typedef internal::traits<Derived> Traits;

  public:

    typedef typename Traits::IndicesType IndicesType;
    typedef typename IndicesType::Scalar Index;

    Derived& derived() { return *static_cast<Derived*>(this); }
    const Derived& derived() const { return *static_cast<const Derived*>(this); }

    /** Copies the \a other transpositions into \c *this */
    template<typename OtherDerived>
    Derived& operator=(const TranspositionsBase<OtherDerived>& other)
    {
      indices() = other.indices();
      return derived();
    }

    #ifndef EIGEN_PARSED_BY_DOXYGEN
    /** This is a special case of the templated operator=. Its purpose is to
      * prevent a default operator= from hiding the templated operator=.
      */
    Derived& operator=(const TranspositionsBase& other)
    {
      indices() = other.indices();
      return derived();
    }
    #endif

    /** \returns the number of transpositions */
    inline Index size() const { return indices().size(); }

    /** Direct access to the underlying index vector */
    inline const Index& coeff(Index i) const { return indices().coeff(i); }
    /** Direct access to the underlying index vector */
    inline Index& coeffRef(Index i) { return indices().coeffRef(i); }
    /** Direct access to the underlying index vector */
    inline const Index& operator()(Index i) const { return indices()(i); }
    /** Direct access to the underlying index vector */
    inline Index& operator()(Index i) { return indices()(i); }
    /** Direct access to the underlying index vector */
    inline const Index& operator[](Index i) const { return indices()(i); }
    /** Direct access to the underlying index vector */
    inline Index& operator[](Index i) { return indices()(i); }

    /** const version of indices(). */
    const IndicesType& indices() const { return derived().indices(); }
    /** \returns a reference to the stored array representing the transpositions. */
    IndicesType& indices() { return derived().indices(); }

    /** Resizes to given size. */
    inline void resize(Index newSize)
    {
      indices().resize(newSize);
    }

    /** Sets \c *this to represents an identity transformation */
    void setIdentity()
    {
      for(int i = 0; i < indices().size(); ++i)
        coeffRef(i) = i;
    }

    // FIXME: do we want such methods ?
    // might be usefull when the target matrix expression is complex, e.g.:
    // object.matrix().block(..,..,..,..) = trans * object.matrix().block(..,..,..,..);
    /*
    template<typename MatrixType>
    void applyForwardToRows(MatrixType& mat) const
    {
      for(Index k=0 ; k<size() ; ++k)
        if(m_indices(k)!=k)
          mat.row(k).swap(mat.row(m_indices(k)));
    }

    template<typename MatrixType>
    void applyBackwardToRows(MatrixType& mat) const
    {
      for(Index k=size()-1 ; k>=0 ; --k)
        if(m_indices(k)!=k)
          mat.row(k).swap(mat.row(m_indices(k)));
    }
    */

    /** \returns the inverse transformation */
    inline Transpose<TranspositionsBase> inverse() const
    { return Transpose<TranspositionsBase>(derived()); }

    /** \returns the tranpose transformation */
    inline Transpose<TranspositionsBase> transpose() const
    { return Transpose<TranspositionsBase>(derived()); }

  protected:
};

namespace internal {
template<int SizeAtCompileTime, int MaxSizeAtCompileTime, typename IndexType>
struct traits<Transpositions<SizeAtCompileTime,MaxSizeAtCompileTime,IndexType> >
{
  typedef IndexType Index;
  typedef Matrix<Index, SizeAtCompileTime, 1, 0, MaxSizeAtCompileTime, 1> IndicesType;
};
}

template<int SizeAtCompileTime, int MaxSizeAtCompileTime, typename IndexType>
class Transpositions : public TranspositionsBase<Transpositions<SizeAtCompileTime,MaxSizeAtCompileTime,IndexType> >
{
    typedef internal::traits<Transpositions> Traits;
  public:

    typedef TranspositionsBase<Transpositions> Base;
    typedef typename Traits::IndicesType IndicesType;
    typedef typename IndicesType::Scalar Index;

    inline Transpositions() {}

    /** Copy constructor. */
    template<typename OtherDerived>
    inline Transpositions(const TranspositionsBase<OtherDerived>& other)
      : m_indices(other.indices()) {}

    #ifndef EIGEN_PARSED_BY_DOXYGEN
    /** Standard copy constructor. Defined only to prevent a default copy constructor
      * from hiding the other templated constructor */
    inline Transpositions(const Transpositions& other) : m_indices(other.indices()) {}
    #endif

    /** Generic constructor from expression of the transposition indices. */
    template<typename Other>
    explicit inline Transpositions(const MatrixBase<Other>& a_indices) : m_indices(a_indices)
    {}

    /** Copies the \a other transpositions into \c *this */
    template<typename OtherDerived>
    Transpositions& operator=(const TranspositionsBase<OtherDerived>& other)
    {
      return Base::operator=(other);
    }

    #ifndef EIGEN_PARSED_BY_DOXYGEN
    /** This is a special case of the templated operator=. Its purpose is to
      * prevent a default operator= from hiding the templated operator=.
      */
    Transpositions& operator=(const Transpositions& other)
    {
      m_indices = other.m_indices;
      return *this;
    }
    #endif

    /** Constructs an uninitialized permutation matrix of given size.
      */
    inline Transpositions(Index size) : m_indices(size)
    {}

    /** const version of indices(). */
    const IndicesType& indices() const { return m_indices; }
    /** \returns a reference to the stored array representing the transpositions. */
    IndicesType& indices() { return m_indices; }

  protected:

    IndicesType m_indices;
};


namespace internal {
template<int SizeAtCompileTime, int MaxSizeAtCompileTime, typename IndexType, int _PacketAccess>
struct traits<Map<Transpositions<SizeAtCompileTime,MaxSizeAtCompileTime,IndexType>,_PacketAccess> >
{
  typedef IndexType Index;
  typedef Map<const Matrix<Index,SizeAtCompileTime,1,0,MaxSizeAtCompileTime,1>, _PacketAccess> IndicesType;
};
}

template<int SizeAtCompileTime, int MaxSizeAtCompileTime, typename IndexType, int PacketAccess>
class Map<Transpositions<SizeAtCompileTime,MaxSizeAtCompileTime,IndexType>,PacketAccess>
 : public TranspositionsBase<Map<Transpositions<SizeAtCompileTime,MaxSizeAtCompileTime,IndexType>,PacketAccess> >
{
    typedef internal::traits<Map> Traits;
  public:

    typedef TranspositionsBase<Map> Base;
    typedef typename Traits::IndicesType IndicesType;
    typedef typename IndicesType::Scalar Index;

    inline Map(const Index* indicesPtr)
      : m_indices(indicesPtr)
    {}

    inline Map(const Index* indicesPtr, Index size)
      : m_indices(indicesPtr,size)
    {}

    /** Copies the \a other transpositions into \c *this */
    template<typename OtherDerived>
    Map& operator=(const TranspositionsBase<OtherDerived>& other)
    {
      return Base::operator=(other);
    }

    #ifndef EIGEN_PARSED_BY_DOXYGEN
    /** This is a special case of the templated operator=. Its purpose is to
      * prevent a default operator= from hiding the templated operator=.
      */
    Map& operator=(const Map& other)
    {
      m_indices = other.m_indices;
      return *this;
    }
    #endif

    /** const version of indices(). */
    const IndicesType& indices() const { return m_indices; }

    /** \returns a reference to the stored array representing the transpositions. */
    IndicesType& indices() { return m_indices; }

  protected:

    IndicesType m_indices;
};

namespace internal {
template<typename _IndicesType>
struct traits<TranspositionsWrapper<_IndicesType> >
{
  typedef typename _IndicesType::Scalar Index;
  typedef _IndicesType IndicesType;
};
}

template<typename _IndicesType>
class TranspositionsWrapper
 : public TranspositionsBase<TranspositionsWrapper<_IndicesType> >
{
    typedef internal::traits<TranspositionsWrapper> Traits;
  public:

    typedef TranspositionsBase<TranspositionsWrapper> Base;
    typedef typename Traits::IndicesType IndicesType;
    typedef typename IndicesType::Scalar Index;

    inline TranspositionsWrapper(IndicesType& a_indices)
      : m_indices(a_indices)
    {}

    /** Copies the \a other transpositions into \c *this */
    template<typename OtherDerived>
    TranspositionsWrapper& operator=(const TranspositionsBase<OtherDerived>& other)
    {
      return Base::operator=(other);
    }

    #ifndef EIGEN_PARSED_BY_DOXYGEN
    /** This is a special case of the templated operator=. Its purpose is to
      * prevent a default operator= from hiding the templated operator=.
      */
    TranspositionsWrapper& operator=(const TranspositionsWrapper& other)
    {
      m_indices = other.m_indices;
      return *this;
    }
    #endif

    /** const version of indices(). */
    const IndicesType& indices() const { return m_indices; }

    /** \returns a reference to the stored array representing the transpositions. */
    IndicesType& indices() { return m_indices; }

  protected:

    const typename IndicesType::Nested m_indices;
};

/** \returns the \a matrix with the \a transpositions applied to the columns.
  */
template<typename Derived, typename TranspositionsDerived>
inline const internal::transposition_matrix_product_retval<TranspositionsDerived, Derived, OnTheRight>
operator*(const MatrixBase<Derived>& matrix,
          const TranspositionsBase<TranspositionsDerived> &transpositions)
{
  return internal::transposition_matrix_product_retval
           <TranspositionsDerived, Derived, OnTheRight>
           (transpositions.derived(), matrix.derived());
}

/** \returns the \a matrix with the \a transpositions applied to the rows.
  */
template<typename Derived, typename TranspositionDerived>
inline const internal::transposition_matrix_product_retval
               <TranspositionDerived, Derived, OnTheLeft>
operator*(const TranspositionsBase<TranspositionDerived> &transpositions,
          const MatrixBase<Derived>& matrix)
{
  return internal::transposition_matrix_product_retval
           <TranspositionDerived, Derived, OnTheLeft>
           (transpositions.derived(), matrix.derived());
}

namespace internal {

template<typename TranspositionType, typename MatrixType, int Side, bool Transposed>
struct traits<transposition_matrix_product_retval<TranspositionType, MatrixType, Side, Transposed> >
{
  typedef typename MatrixType::PlainObject ReturnType;
};

template<typename TranspositionType, typename MatrixType, int Side, bool Transposed>
struct transposition_matrix_product_retval
 : public ReturnByValue<transposition_matrix_product_retval<TranspositionType, MatrixType, Side, Transposed> >
{
    typedef typename remove_all<typename MatrixType::Nested>::type MatrixTypeNestedCleaned;
    typedef typename TranspositionType::Index Index;

    transposition_matrix_product_retval(const TranspositionType& tr, const MatrixType& matrix)
      : m_transpositions(tr), m_matrix(matrix)
    {}

    inline Index rows() const { return m_matrix.rows(); }
    inline Index cols() const { return m_matrix.cols(); }

    template<typename Dest> inline void evalTo(Dest& dst) const
    {
      const Index size = m_transpositions.size();
      Index j = 0;

      if(!(is_same<MatrixTypeNestedCleaned,Dest>::value && extract_data(dst) == extract_data(m_matrix)))
        dst = m_matrix;

      for(Index k=(Transposed?size-1:0) ; Transposed?k>=0:k<size ; Transposed?--k:++k)
        if((j=m_transpositions.coeff(k))!=k)
        {
          if(Side==OnTheLeft)
            dst.row(k).swap(dst.row(j));
          else if(Side==OnTheRight)
            dst.col(k).swap(dst.col(j));
        }
    }

  protected:
    const TranspositionType& m_transpositions;
    typename MatrixType::Nested m_matrix;
};

} // end namespace internal

/* Template partial specialization for transposed/inverse transpositions */

template<typename TranspositionsDerived>
class Transpose<TranspositionsBase<TranspositionsDerived> >
{
    typedef TranspositionsDerived TranspositionType;
    typedef typename TranspositionType::IndicesType IndicesType;
  public:

    Transpose(const TranspositionType& t) : m_transpositions(t) {}

    inline int size() const { return m_transpositions.size(); }

    /** \returns the \a matrix with the inverse transpositions applied to the columns.
      */
    template<typename Derived> friend
    inline const internal::transposition_matrix_product_retval<TranspositionType, Derived, OnTheRight, true>
    operator*(const MatrixBase<Derived>& matrix, const Transpose& trt)
    {
      return internal::transposition_matrix_product_retval<TranspositionType, Derived, OnTheRight, true>(trt.m_transpositions, matrix.derived());
    }

    /** \returns the \a matrix with the inverse transpositions applied to the rows.
      */
    template<typename Derived>
    inline const internal::transposition_matrix_product_retval<TranspositionType, Derived, OnTheLeft, true>
    operator*(const MatrixBase<Derived>& matrix) const
    {
      return internal::transposition_matrix_product_retval<TranspositionType, Derived, OnTheLeft, true>(m_transpositions, matrix.derived());
    }

  protected:
    const TranspositionType& m_transpositions;
};

} // end namespace Eigen

#endif // EIGEN_TRANSPOSITIONS_H
