// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2009 Benoit Jacob <jacob.benoit.1@gmail.com>
// Copyright (C) 2009-2011 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_PERMUTATIONMATRIX_H
#define EIGEN_PERMUTATIONMATRIX_H

namespace Eigen { 

template<int RowCol,typename IndicesType,typename MatrixType, typename StorageKind> class PermutedImpl;

/** \class PermutationBase
  * \ingroup Core_Module
  *
  * \brief Base class for permutations
  *
  * \param Derived the derived class
  *
  * This class is the base class for all expressions representing a permutation matrix,
  * internally stored as a vector of integers.
  * The convention followed here is that if \f$ \sigma \f$ is a permutation, the corresponding permutation matrix
  * \f$ P_\sigma \f$ is such that if \f$ (e_1,\ldots,e_p) \f$ is the canonical basis, we have:
  *  \f[ P_\sigma(e_i) = e_{\sigma(i)}. \f]
  * This convention ensures that for any two permutations \f$ \sigma, \tau \f$, we have:
  *  \f[ P_{\sigma\circ\tau} = P_\sigma P_\tau. \f]
  *
  * Permutation matrices are square and invertible.
  *
  * Notice that in addition to the member functions and operators listed here, there also are non-member
  * operator* to multiply any kind of permutation object with any kind of matrix expression (MatrixBase)
  * on either side.
  *
  * \sa class PermutationMatrix, class PermutationWrapper
  */

namespace internal {

template<typename PermutationType, typename MatrixType, int Side, bool Transposed=false>
struct permut_matrix_product_retval;
template<typename PermutationType, typename MatrixType, int Side, bool Transposed=false>
struct permut_sparsematrix_product_retval;
enum PermPermProduct_t {PermPermProduct};

} // end namespace internal

template<typename Derived>
class PermutationBase : public EigenBase<Derived>
{
    typedef internal::traits<Derived> Traits;
    typedef EigenBase<Derived> Base;
  public:

    #ifndef EIGEN_PARSED_BY_DOXYGEN
    typedef typename Traits::IndicesType IndicesType;
    enum {
      Flags = Traits::Flags,
      CoeffReadCost = Traits::CoeffReadCost,
      RowsAtCompileTime = Traits::RowsAtCompileTime,
      ColsAtCompileTime = Traits::ColsAtCompileTime,
      MaxRowsAtCompileTime = Traits::MaxRowsAtCompileTime,
      MaxColsAtCompileTime = Traits::MaxColsAtCompileTime
    };
    typedef typename Traits::Scalar Scalar;
    typedef typename Traits::Index Index;
    typedef Matrix<Scalar,RowsAtCompileTime,ColsAtCompileTime,0,MaxRowsAtCompileTime,MaxColsAtCompileTime>
            DenseMatrixType;
    typedef PermutationMatrix<IndicesType::SizeAtCompileTime,IndicesType::MaxSizeAtCompileTime,Index>
            PlainPermutationType;
    using Base::derived;
    #endif

    /** Copies the other permutation into *this */
    template<typename OtherDerived>
    Derived& operator=(const PermutationBase<OtherDerived>& other)
    {
      indices() = other.indices();
      return derived();
    }

    /** Assignment from the Transpositions \a tr */
    template<typename OtherDerived>
    Derived& operator=(const TranspositionsBase<OtherDerived>& tr)
    {
      setIdentity(tr.size());
      for(Index k=size()-1; k>=0; --k)
        applyTranspositionOnTheRight(k,tr.coeff(k));
      return derived();
    }

    #ifndef EIGEN_PARSED_BY_DOXYGEN
    /** This is a special case of the templated operator=. Its purpose is to
      * prevent a default operator= from hiding the templated operator=.
      */
    Derived& operator=(const PermutationBase& other)
    {
      indices() = other.indices();
      return derived();
    }
    #endif

    /** \returns the number of rows */
    inline Index rows() const { return Index(indices().size()); }

    /** \returns the number of columns */
    inline Index cols() const { return Index(indices().size()); }

    /** \returns the size of a side of the respective square matrix, i.e., the number of indices */
    inline Index size() const { return Index(indices().size()); }

    #ifndef EIGEN_PARSED_BY_DOXYGEN
    template<typename DenseDerived>
    void evalTo(MatrixBase<DenseDerived>& other) const
    {
      other.setZero();
      for (int i=0; i<rows();++i)
        other.coeffRef(indices().coeff(i),i) = typename DenseDerived::Scalar(1);
    }
    #endif

    /** \returns a Matrix object initialized from this permutation matrix. Notice that it
      * is inefficient to return this Matrix object by value. For efficiency, favor using
      * the Matrix constructor taking EigenBase objects.
      */
    DenseMatrixType toDenseMatrix() const
    {
      return derived();
    }

    /** const version of indices(). */
    const IndicesType& indices() const { return derived().indices(); }
    /** \returns a reference to the stored array representing the permutation. */
    IndicesType& indices() { return derived().indices(); }

    /** Resizes to given size.
      */
    inline void resize(Index newSize)
    {
      indices().resize(newSize);
    }

    /** Sets *this to be the identity permutation matrix */
    void setIdentity()
    {
      for(Index i = 0; i < size(); ++i)
        indices().coeffRef(i) = i;
    }

    /** Sets *this to be the identity permutation matrix of given size.
      */
    void setIdentity(Index newSize)
    {
      resize(newSize);
      setIdentity();
    }

    /** Multiplies *this by the transposition \f$(ij)\f$ on the left.
      *
      * \returns a reference to *this.
      *
      * \warning This is much slower than applyTranspositionOnTheRight(int,int):
      * this has linear complexity and requires a lot of branching.
      *
      * \sa applyTranspositionOnTheRight(int,int)
      */
    Derived& applyTranspositionOnTheLeft(Index i, Index j)
    {
      eigen_assert(i>=0 && j>=0 && i<size() && j<size());
      for(Index k = 0; k < size(); ++k)
      {
        if(indices().coeff(k) == i) indices().coeffRef(k) = j;
        else if(indices().coeff(k) == j) indices().coeffRef(k) = i;
      }
      return derived();
    }

    /** Multiplies *this by the transposition \f$(ij)\f$ on the right.
      *
      * \returns a reference to *this.
      *
      * This is a fast operation, it only consists in swapping two indices.
      *
      * \sa applyTranspositionOnTheLeft(int,int)
      */
    Derived& applyTranspositionOnTheRight(Index i, Index j)
    {
      eigen_assert(i>=0 && j>=0 && i<size() && j<size());
      std::swap(indices().coeffRef(i), indices().coeffRef(j));
      return derived();
    }

    /** \returns the inverse permutation matrix.
      *
      * \note \note_try_to_help_rvo
      */
    inline Transpose<PermutationBase> inverse() const
    { return derived(); }
    /** \returns the tranpose permutation matrix.
      *
      * \note \note_try_to_help_rvo
      */
    inline Transpose<PermutationBase> transpose() const
    { return derived(); }

    /**** multiplication helpers to hopefully get RVO ****/

  
#ifndef EIGEN_PARSED_BY_DOXYGEN
  protected:
    template<typename OtherDerived>
    void assignTranspose(const PermutationBase<OtherDerived>& other)
    {
      for (int i=0; i<rows();++i) indices().coeffRef(other.indices().coeff(i)) = i;
    }
    template<typename Lhs,typename Rhs>
    void assignProduct(const Lhs& lhs, const Rhs& rhs)
    {
      eigen_assert(lhs.cols() == rhs.rows());
      for (int i=0; i<rows();++i) indices().coeffRef(i) = lhs.indices().coeff(rhs.indices().coeff(i));
    }
#endif

  public:

    /** \returns the product permutation matrix.
      *
      * \note \note_try_to_help_rvo
      */
    template<typename Other>
    inline PlainPermutationType operator*(const PermutationBase<Other>& other) const
    { return PlainPermutationType(internal::PermPermProduct, derived(), other.derived()); }

    /** \returns the product of a permutation with another inverse permutation.
      *
      * \note \note_try_to_help_rvo
      */
    template<typename Other>
    inline PlainPermutationType operator*(const Transpose<PermutationBase<Other> >& other) const
    { return PlainPermutationType(internal::PermPermProduct, *this, other.eval()); }

    /** \returns the product of an inverse permutation with another permutation.
      *
      * \note \note_try_to_help_rvo
      */
    template<typename Other> friend
    inline PlainPermutationType operator*(const Transpose<PermutationBase<Other> >& other, const PermutationBase& perm)
    { return PlainPermutationType(internal::PermPermProduct, other.eval(), perm); }

  protected:

};

/** \class PermutationMatrix
  * \ingroup Core_Module
  *
  * \brief Permutation matrix
  *
  * \param SizeAtCompileTime the number of rows/cols, or Dynamic
  * \param MaxSizeAtCompileTime the maximum number of rows/cols, or Dynamic. This optional parameter defaults to SizeAtCompileTime. Most of the time, you should not have to specify it.
  * \param IndexType the interger type of the indices
  *
  * This class represents a permutation matrix, internally stored as a vector of integers.
  *
  * \sa class PermutationBase, class PermutationWrapper, class DiagonalMatrix
  */

namespace internal {
template<int SizeAtCompileTime, int MaxSizeAtCompileTime, typename IndexType>
struct traits<PermutationMatrix<SizeAtCompileTime, MaxSizeAtCompileTime, IndexType> >
 : traits<Matrix<IndexType,SizeAtCompileTime,SizeAtCompileTime,0,MaxSizeAtCompileTime,MaxSizeAtCompileTime> >
{
  typedef IndexType Index;
  typedef Matrix<IndexType, SizeAtCompileTime, 1, 0, MaxSizeAtCompileTime, 1> IndicesType;
};
}

template<int SizeAtCompileTime, int MaxSizeAtCompileTime, typename IndexType>
class PermutationMatrix : public PermutationBase<PermutationMatrix<SizeAtCompileTime, MaxSizeAtCompileTime, IndexType> >
{
    typedef PermutationBase<PermutationMatrix> Base;
    typedef internal::traits<PermutationMatrix> Traits;
  public:

    #ifndef EIGEN_PARSED_BY_DOXYGEN
    typedef typename Traits::IndicesType IndicesType;
    #endif

    inline PermutationMatrix()
    {}

    /** Constructs an uninitialized permutation matrix of given size.
      */
    inline PermutationMatrix(int size) : m_indices(size)
    {}

    /** Copy constructor. */
    template<typename OtherDerived>
    inline PermutationMatrix(const PermutationBase<OtherDerived>& other)
      : m_indices(other.indices()) {}

    #ifndef EIGEN_PARSED_BY_DOXYGEN
    /** Standard copy constructor. Defined only to prevent a default copy constructor
      * from hiding the other templated constructor */
    inline PermutationMatrix(const PermutationMatrix& other) : m_indices(other.indices()) {}
    #endif

    /** Generic constructor from expression of the indices. The indices
      * array has the meaning that the permutations sends each integer i to indices[i].
      *
      * \warning It is your responsibility to check that the indices array that you passes actually
      * describes a permutation, i.e., each value between 0 and n-1 occurs exactly once, where n is the
      * array's size.
      */
    template<typename Other>
    explicit inline PermutationMatrix(const MatrixBase<Other>& a_indices) : m_indices(a_indices)
    {}

    /** Convert the Transpositions \a tr to a permutation matrix */
    template<typename Other>
    explicit PermutationMatrix(const TranspositionsBase<Other>& tr)
      : m_indices(tr.size())
    {
      *this = tr;
    }

    /** Copies the other permutation into *this */
    template<typename Other>
    PermutationMatrix& operator=(const PermutationBase<Other>& other)
    {
      m_indices = other.indices();
      return *this;
    }

    /** Assignment from the Transpositions \a tr */
    template<typename Other>
    PermutationMatrix& operator=(const TranspositionsBase<Other>& tr)
    {
      return Base::operator=(tr.derived());
    }

    #ifndef EIGEN_PARSED_BY_DOXYGEN
    /** This is a special case of the templated operator=. Its purpose is to
      * prevent a default operator= from hiding the templated operator=.
      */
    PermutationMatrix& operator=(const PermutationMatrix& other)
    {
      m_indices = other.m_indices;
      return *this;
    }
    #endif

    /** const version of indices(). */
    const IndicesType& indices() const { return m_indices; }
    /** \returns a reference to the stored array representing the permutation. */
    IndicesType& indices() { return m_indices; }


    /**** multiplication helpers to hopefully get RVO ****/

#ifndef EIGEN_PARSED_BY_DOXYGEN
    template<typename Other>
    PermutationMatrix(const Transpose<PermutationBase<Other> >& other)
      : m_indices(other.nestedPermutation().size())
    {
      for (int i=0; i<m_indices.size();++i) m_indices.coeffRef(other.nestedPermutation().indices().coeff(i)) = i;
    }
    template<typename Lhs,typename Rhs>
    PermutationMatrix(internal::PermPermProduct_t, const Lhs& lhs, const Rhs& rhs)
      : m_indices(lhs.indices().size())
    {
      Base::assignProduct(lhs,rhs);
    }
#endif

  protected:

    IndicesType m_indices;
};


namespace internal {
template<int SizeAtCompileTime, int MaxSizeAtCompileTime, typename IndexType, int _PacketAccess>
struct traits<Map<PermutationMatrix<SizeAtCompileTime, MaxSizeAtCompileTime, IndexType>,_PacketAccess> >
 : traits<Matrix<IndexType,SizeAtCompileTime,SizeAtCompileTime,0,MaxSizeAtCompileTime,MaxSizeAtCompileTime> >
{
  typedef IndexType Index;
  typedef Map<const Matrix<IndexType, SizeAtCompileTime, 1, 0, MaxSizeAtCompileTime, 1>, _PacketAccess> IndicesType;
};
}

template<int SizeAtCompileTime, int MaxSizeAtCompileTime, typename IndexType, int _PacketAccess>
class Map<PermutationMatrix<SizeAtCompileTime, MaxSizeAtCompileTime, IndexType>,_PacketAccess>
  : public PermutationBase<Map<PermutationMatrix<SizeAtCompileTime, MaxSizeAtCompileTime, IndexType>,_PacketAccess> >
{
    typedef PermutationBase<Map> Base;
    typedef internal::traits<Map> Traits;
  public:

    #ifndef EIGEN_PARSED_BY_DOXYGEN
    typedef typename Traits::IndicesType IndicesType;
    typedef typename IndicesType::Scalar Index;
    #endif

    inline Map(const Index* indicesPtr)
      : m_indices(indicesPtr)
    {}

    inline Map(const Index* indicesPtr, Index size)
      : m_indices(indicesPtr,size)
    {}

    /** Copies the other permutation into *this */
    template<typename Other>
    Map& operator=(const PermutationBase<Other>& other)
    { return Base::operator=(other.derived()); }

    /** Assignment from the Transpositions \a tr */
    template<typename Other>
    Map& operator=(const TranspositionsBase<Other>& tr)
    { return Base::operator=(tr.derived()); }

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
    /** \returns a reference to the stored array representing the permutation. */
    IndicesType& indices() { return m_indices; }

  protected:

    IndicesType m_indices;
};

/** \class PermutationWrapper
  * \ingroup Core_Module
  *
  * \brief Class to view a vector of integers as a permutation matrix
  *
  * \param _IndicesType the type of the vector of integer (can be any compatible expression)
  *
  * This class allows to view any vector expression of integers as a permutation matrix.
  *
  * \sa class PermutationBase, class PermutationMatrix
  */

struct PermutationStorage {};

template<typename _IndicesType> class TranspositionsWrapper;
namespace internal {
template<typename _IndicesType>
struct traits<PermutationWrapper<_IndicesType> >
{
  typedef PermutationStorage StorageKind;
  typedef typename _IndicesType::Scalar Scalar;
  typedef typename _IndicesType::Scalar Index;
  typedef _IndicesType IndicesType;
  enum {
    RowsAtCompileTime = _IndicesType::SizeAtCompileTime,
    ColsAtCompileTime = _IndicesType::SizeAtCompileTime,
    MaxRowsAtCompileTime = IndicesType::MaxRowsAtCompileTime,
    MaxColsAtCompileTime = IndicesType::MaxColsAtCompileTime,
    Flags = 0,
    CoeffReadCost = _IndicesType::CoeffReadCost
  };
};
}

template<typename _IndicesType>
class PermutationWrapper : public PermutationBase<PermutationWrapper<_IndicesType> >
{
    typedef PermutationBase<PermutationWrapper> Base;
    typedef internal::traits<PermutationWrapper> Traits;
  public:

    #ifndef EIGEN_PARSED_BY_DOXYGEN
    typedef typename Traits::IndicesType IndicesType;
    #endif

    inline PermutationWrapper(const IndicesType& a_indices)
      : m_indices(a_indices)
    {}

    /** const version of indices(). */
    const typename internal::remove_all<typename IndicesType::Nested>::type&
    indices() const { return m_indices; }

  protected:

    typename IndicesType::Nested m_indices;
};

/** \returns the matrix with the permutation applied to the columns.
  */
template<typename Derived, typename PermutationDerived>
inline const internal::permut_matrix_product_retval<PermutationDerived, Derived, OnTheRight>
operator*(const MatrixBase<Derived>& matrix,
          const PermutationBase<PermutationDerived> &permutation)
{
  return internal::permut_matrix_product_retval
           <PermutationDerived, Derived, OnTheRight>
           (permutation.derived(), matrix.derived());
}

/** \returns the matrix with the permutation applied to the rows.
  */
template<typename Derived, typename PermutationDerived>
inline const internal::permut_matrix_product_retval
               <PermutationDerived, Derived, OnTheLeft>
operator*(const PermutationBase<PermutationDerived> &permutation,
          const MatrixBase<Derived>& matrix)
{
  return internal::permut_matrix_product_retval
           <PermutationDerived, Derived, OnTheLeft>
           (permutation.derived(), matrix.derived());
}

namespace internal {

template<typename PermutationType, typename MatrixType, int Side, bool Transposed>
struct traits<permut_matrix_product_retval<PermutationType, MatrixType, Side, Transposed> >
{
  typedef typename MatrixType::PlainObject ReturnType;
};

template<typename PermutationType, typename MatrixType, int Side, bool Transposed>
struct permut_matrix_product_retval
 : public ReturnByValue<permut_matrix_product_retval<PermutationType, MatrixType, Side, Transposed> >
{
    typedef typename remove_all<typename MatrixType::Nested>::type MatrixTypeNestedCleaned;
    typedef typename MatrixType::Index Index;

    permut_matrix_product_retval(const PermutationType& perm, const MatrixType& matrix)
      : m_permutation(perm), m_matrix(matrix)
    {}

    inline Index rows() const { return m_matrix.rows(); }
    inline Index cols() const { return m_matrix.cols(); }

    template<typename Dest> inline void evalTo(Dest& dst) const
    {
      const Index n = Side==OnTheLeft ? rows() : cols();
      // FIXME we need an is_same for expression that is not sensitive to constness. For instance
      // is_same_xpr<Block<const Matrix>, Block<Matrix> >::value should be true.
      if(is_same<MatrixTypeNestedCleaned,Dest>::value && extract_data(dst) == extract_data(m_matrix))
      {
        // apply the permutation inplace
        Matrix<bool,PermutationType::RowsAtCompileTime,1,0,PermutationType::MaxRowsAtCompileTime> mask(m_permutation.size());
        mask.fill(false);
        Index r = 0;
        while(r < m_permutation.size())
        {
          // search for the next seed
          while(r<m_permutation.size() && mask[r]) r++;
          if(r>=m_permutation.size())
            break;
          // we got one, let's follow it until we are back to the seed
          Index k0 = r++;
          Index kPrev = k0;
          mask.coeffRef(k0) = true;
          for(Index k=m_permutation.indices().coeff(k0); k!=k0; k=m_permutation.indices().coeff(k))
          {
                  Block<Dest, Side==OnTheLeft ? 1 : Dest::RowsAtCompileTime, Side==OnTheRight ? 1 : Dest::ColsAtCompileTime>(dst, k)
            .swap(Block<Dest, Side==OnTheLeft ? 1 : Dest::RowsAtCompileTime, Side==OnTheRight ? 1 : Dest::ColsAtCompileTime>
                       (dst,((Side==OnTheLeft) ^ Transposed) ? k0 : kPrev));

            mask.coeffRef(k) = true;
            kPrev = k;
          }
        }
      }
      else
      {
        for(int i = 0; i < n; ++i)
        {
          Block<Dest, Side==OnTheLeft ? 1 : Dest::RowsAtCompileTime, Side==OnTheRight ? 1 : Dest::ColsAtCompileTime>
               (dst, ((Side==OnTheLeft) ^ Transposed) ? m_permutation.indices().coeff(i) : i)

          =

          Block<const MatrixTypeNestedCleaned,Side==OnTheLeft ? 1 : MatrixType::RowsAtCompileTime,Side==OnTheRight ? 1 : MatrixType::ColsAtCompileTime>
               (m_matrix, ((Side==OnTheRight) ^ Transposed) ? m_permutation.indices().coeff(i) : i);
        }
      }
    }

  protected:
    const PermutationType& m_permutation;
    typename MatrixType::Nested m_matrix;
};

/* Template partial specialization for transposed/inverse permutations */

template<typename Derived>
struct traits<Transpose<PermutationBase<Derived> > >
 : traits<Derived>
{};

} // end namespace internal

template<typename Derived>
class Transpose<PermutationBase<Derived> >
  : public EigenBase<Transpose<PermutationBase<Derived> > >
{
    typedef Derived PermutationType;
    typedef typename PermutationType::IndicesType IndicesType;
    typedef typename PermutationType::PlainPermutationType PlainPermutationType;
  public:

    #ifndef EIGEN_PARSED_BY_DOXYGEN
    typedef internal::traits<PermutationType> Traits;
    typedef typename Derived::DenseMatrixType DenseMatrixType;
    enum {
      Flags = Traits::Flags,
      CoeffReadCost = Traits::CoeffReadCost,
      RowsAtCompileTime = Traits::RowsAtCompileTime,
      ColsAtCompileTime = Traits::ColsAtCompileTime,
      MaxRowsAtCompileTime = Traits::MaxRowsAtCompileTime,
      MaxColsAtCompileTime = Traits::MaxColsAtCompileTime
    };
    typedef typename Traits::Scalar Scalar;
    #endif

    Transpose(const PermutationType& p) : m_permutation(p) {}

    inline int rows() const { return m_permutation.rows(); }
    inline int cols() const { return m_permutation.cols(); }

    #ifndef EIGEN_PARSED_BY_DOXYGEN
    template<typename DenseDerived>
    void evalTo(MatrixBase<DenseDerived>& other) const
    {
      other.setZero();
      for (int i=0; i<rows();++i)
        other.coeffRef(i, m_permutation.indices().coeff(i)) = typename DenseDerived::Scalar(1);
    }
    #endif

    /** \return the equivalent permutation matrix */
    PlainPermutationType eval() const { return *this; }

    DenseMatrixType toDenseMatrix() const { return *this; }

    /** \returns the matrix with the inverse permutation applied to the columns.
      */
    template<typename OtherDerived> friend
    inline const internal::permut_matrix_product_retval<PermutationType, OtherDerived, OnTheRight, true>
    operator*(const MatrixBase<OtherDerived>& matrix, const Transpose& trPerm)
    {
      return internal::permut_matrix_product_retval<PermutationType, OtherDerived, OnTheRight, true>(trPerm.m_permutation, matrix.derived());
    }

    /** \returns the matrix with the inverse permutation applied to the rows.
      */
    template<typename OtherDerived>
    inline const internal::permut_matrix_product_retval<PermutationType, OtherDerived, OnTheLeft, true>
    operator*(const MatrixBase<OtherDerived>& matrix) const
    {
      return internal::permut_matrix_product_retval<PermutationType, OtherDerived, OnTheLeft, true>(m_permutation, matrix.derived());
    }

    const PermutationType& nestedPermutation() const { return m_permutation; }

  protected:
    const PermutationType& m_permutation;
};

template<typename Derived>
const PermutationWrapper<const Derived> MatrixBase<Derived>::asPermutation() const
{
  return derived();
}

} // end namespace Eigen

#endif // EIGEN_PERMUTATIONMATRIX_H
