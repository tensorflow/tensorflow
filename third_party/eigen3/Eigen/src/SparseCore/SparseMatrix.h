// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008-2010 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_SPARSEMATRIX_H
#define EIGEN_SPARSEMATRIX_H

namespace Eigen { 

/** \ingroup SparseCore_Module
  *
  * \class SparseMatrix
  *
  * \brief A versatible sparse matrix representation
  *
  * This class implements a more versatile variants of the common \em compressed row/column storage format.
  * Each colmun's (resp. row) non zeros are stored as a pair of value with associated row (resp. colmiun) index.
  * All the non zeros are stored in a single large buffer. Unlike the \em compressed format, there might be extra
  * space inbetween the nonzeros of two successive colmuns (resp. rows) such that insertion of new non-zero
  * can be done with limited memory reallocation and copies.
  *
  * A call to the function makeCompressed() turns the matrix into the standard \em compressed format
  * compatible with many library.
  *
  * More details on this storage sceheme are given in the \ref TutorialSparse "manual pages".
  *
  * \tparam _Scalar the scalar type, i.e. the type of the coefficients
  * \tparam _Options Union of bit flags controlling the storage scheme. Currently the only possibility
  *                 is ColMajor or RowMajor. The default is 0 which means column-major.
  * \tparam _Index the type of the indices. It has to be a \b signed type (e.g., short, int, std::ptrdiff_t). Default is \c int.
  *
  * This class can be extended with the help of the plugin mechanism described on the page
  * \ref TopicCustomizingEigen by defining the preprocessor symbol \c EIGEN_SPARSEMATRIX_PLUGIN.
  */

namespace internal {
template<typename _Scalar, int _Options, typename _Index>
struct traits<SparseMatrix<_Scalar, _Options, _Index> >
{
  typedef _Scalar Scalar;
  typedef _Index Index;
  typedef Sparse StorageKind;
  typedef MatrixXpr XprKind;
  enum {
    RowsAtCompileTime = Dynamic,
    ColsAtCompileTime = Dynamic,
    MaxRowsAtCompileTime = Dynamic,
    MaxColsAtCompileTime = Dynamic,
    Flags = _Options | NestByRefBit | LvalueBit,
    CoeffReadCost = NumTraits<Scalar>::ReadCost,
    SupportedAccessPatterns = InnerRandomAccessPattern
  };
};

template<typename _Scalar, int _Options, typename _Index, int DiagIndex>
struct traits<Diagonal<const SparseMatrix<_Scalar, _Options, _Index>, DiagIndex> >
{
  typedef SparseMatrix<_Scalar, _Options, _Index> MatrixType;
  typedef typename nested<MatrixType>::type MatrixTypeNested;
  typedef typename remove_reference<MatrixTypeNested>::type _MatrixTypeNested;

  typedef _Scalar Scalar;
  typedef Dense StorageKind;
  typedef _Index Index;
  typedef MatrixXpr XprKind;

  enum {
    RowsAtCompileTime = Dynamic,
    ColsAtCompileTime = 1,
    MaxRowsAtCompileTime = Dynamic,
    MaxColsAtCompileTime = 1,
    Flags = 0,
    CoeffReadCost = _MatrixTypeNested::CoeffReadCost*10
  };
};

} // end namespace internal

template<typename _Scalar, int _Options, typename _Index>
class SparseMatrix
  : public SparseMatrixBase<SparseMatrix<_Scalar, _Options, _Index> >
{
  public:
    EIGEN_SPARSE_PUBLIC_INTERFACE(SparseMatrix)
    EIGEN_SPARSE_INHERIT_ASSIGNMENT_OPERATOR(SparseMatrix, +=)
    EIGEN_SPARSE_INHERIT_ASSIGNMENT_OPERATOR(SparseMatrix, -=)

    typedef MappedSparseMatrix<Scalar,Flags> Map;
    using Base::IsRowMajor;
    typedef internal::CompressedStorage<Scalar,Index> Storage;
    enum {
      Options = _Options
    };

  protected:

    typedef SparseMatrix<Scalar,(Flags&~RowMajorBit)|(IsRowMajor?RowMajorBit:0)> TransposedSparseMatrix;

    Index m_outerSize;
    Index m_innerSize;
    Index* m_outerIndex;
    Index* m_innerNonZeros;     // optional, if null then the data is compressed
    Storage m_data;
    
    Eigen::Map<Matrix<Index,Dynamic,1> > innerNonZeros() { return Eigen::Map<Matrix<Index,Dynamic,1> >(m_innerNonZeros, m_innerNonZeros?m_outerSize:0); }
    const  Eigen::Map<const Matrix<Index,Dynamic,1> > innerNonZeros() const { return Eigen::Map<const Matrix<Index,Dynamic,1> >(m_innerNonZeros, m_innerNonZeros?m_outerSize:0); }

  public:
    
    /** \returns whether \c *this is in compressed form. */
    inline bool isCompressed() const { return m_innerNonZeros==0; }

    /** \returns the number of rows of the matrix */
    inline Index rows() const { return IsRowMajor ? m_outerSize : m_innerSize; }
    /** \returns the number of columns of the matrix */
    inline Index cols() const { return IsRowMajor ? m_innerSize : m_outerSize; }

    /** \returns the number of rows (resp. columns) of the matrix if the storage order column major (resp. row major) */
    inline Index innerSize() const { return m_innerSize; }
    /** \returns the number of columns (resp. rows) of the matrix if the storage order column major (resp. row major) */
    inline Index outerSize() const { return m_outerSize; }
    
    /** \returns a const pointer to the array of values.
      * This function is aimed at interoperability with other libraries.
      * \sa innerIndexPtr(), outerIndexPtr() */
    inline const Scalar* valuePtr() const { return &m_data.value(0); }
    /** \returns a non-const pointer to the array of values.
      * This function is aimed at interoperability with other libraries.
      * \sa innerIndexPtr(), outerIndexPtr() */
    inline Scalar* valuePtr() { return &m_data.value(0); }

    /** \returns a const pointer to the array of inner indices.
      * This function is aimed at interoperability with other libraries.
      * \sa valuePtr(), outerIndexPtr() */
    inline const Index* innerIndexPtr() const { return &m_data.index(0); }
    /** \returns a non-const pointer to the array of inner indices.
      * This function is aimed at interoperability with other libraries.
      * \sa valuePtr(), outerIndexPtr() */
    inline Index* innerIndexPtr() { return &m_data.index(0); }

    /** \returns a const pointer to the array of the starting positions of the inner vectors.
      * This function is aimed at interoperability with other libraries.
      * \sa valuePtr(), innerIndexPtr() */
    inline const Index* outerIndexPtr() const { return m_outerIndex; }
    /** \returns a non-const pointer to the array of the starting positions of the inner vectors.
      * This function is aimed at interoperability with other libraries.
      * \sa valuePtr(), innerIndexPtr() */
    inline Index* outerIndexPtr() { return m_outerIndex; }

    /** \returns a const pointer to the array of the number of non zeros of the inner vectors.
      * This function is aimed at interoperability with other libraries.
      * \warning it returns the null pointer 0 in compressed mode */
    inline const Index* innerNonZeroPtr() const { return m_innerNonZeros; }
    /** \returns a non-const pointer to the array of the number of non zeros of the inner vectors.
      * This function is aimed at interoperability with other libraries.
      * \warning it returns the null pointer 0 in compressed mode */
    inline Index* innerNonZeroPtr() { return m_innerNonZeros; }

    /** \internal */
    inline Storage& data() { return m_data; }
    /** \internal */
    inline const Storage& data() const { return m_data; }

    /** \returns the value of the matrix at position \a i, \a j
      * This function returns Scalar(0) if the element is an explicit \em zero */
    inline Scalar coeff(Index row, Index col) const
    {
      eigen_assert(row>=0 && row<rows() && col>=0 && col<cols());
      
      const Index outer = IsRowMajor ? row : col;
      const Index inner = IsRowMajor ? col : row;
      Index end = m_innerNonZeros ? m_outerIndex[outer] + m_innerNonZeros[outer] : m_outerIndex[outer+1];
      return m_data.atInRange(m_outerIndex[outer], end, inner);
    }

    /** \returns a non-const reference to the value of the matrix at position \a i, \a j
      *
      * If the element does not exist then it is inserted via the insert(Index,Index) function
      * which itself turns the matrix into a non compressed form if that was not the case.
      *
      * This is a O(log(nnz_j)) operation (binary search) plus the cost of insert(Index,Index)
      * function if the element does not already exist.
      */
    inline Scalar& coeffRef(Index row, Index col)
    {
      eigen_assert(row>=0 && row<rows() && col>=0 && col<cols());
      
      const Index outer = IsRowMajor ? row : col;
      const Index inner = IsRowMajor ? col : row;

      Index start = m_outerIndex[outer];
      Index end = m_innerNonZeros ? m_outerIndex[outer] + m_innerNonZeros[outer] : m_outerIndex[outer+1];
      eigen_assert(end>=start && "you probably called coeffRef on a non finalized matrix");
      if(end<=start)
        return insert(row,col);
      const Index p = m_data.searchLowerIndex(start,end-1,inner);
      if((p<end) && (m_data.index(p)==inner))
        return m_data.value(p);
      else
        return insert(row,col);
    }

    /** \returns a reference to a novel non zero coefficient with coordinates \a row x \a col.
      * The non zero coefficient must \b not already exist.
      *
      * If the matrix \c *this is in compressed mode, then \c *this is turned into uncompressed
      * mode while reserving room for 2 non zeros per inner vector. It is strongly recommended to first
      * call reserve(const SizesType &) to reserve a more appropriate number of elements per
      * inner vector that better match your scenario.
      *
      * This function performs a sorted insertion in O(1) if the elements of each inner vector are
      * inserted in increasing inner index order, and in O(nnz_j) for a random insertion.
      *
      */
    Scalar& insert(Index row, Index col)
    {
      eigen_assert(row>=0 && row<rows() && col>=0 && col<cols());
      
      if(isCompressed())
      {
        reserve(Matrix<Index,Dynamic,1>::Constant(outerSize(), 2));
      }
      return insertUncompressed(row,col);
    }

  public:

    class InnerIterator;
    class ReverseInnerIterator;

    /** Removes all non zeros but keep allocated memory */
    inline void setZero()
    {
      m_data.clear();
      memset(m_outerIndex, 0, (m_outerSize+1)*sizeof(Index));
      if(m_innerNonZeros)
        memset(m_innerNonZeros, 0, (m_outerSize)*sizeof(Index));
    }

    /** \returns the number of non zero coefficients */
    inline Index nonZeros() const
    {
      if(m_innerNonZeros)
        return innerNonZeros().sum();
      return static_cast<Index>(m_data.size());
    }

    /** Preallocates \a reserveSize non zeros.
      *
      * Precondition: the matrix must be in compressed mode. */
    inline void reserve(Index reserveSize)
    {
      eigen_assert(isCompressed() && "This function does not make sense in non compressed mode.");
      m_data.reserve(reserveSize);
    }
    
    #ifdef EIGEN_PARSED_BY_DOXYGEN
    /** Preallocates \a reserveSize[\c j] non zeros for each column (resp. row) \c j.
      *
      * This function turns the matrix in non-compressed mode */
    template<class SizesType>
    inline void reserve(const SizesType& reserveSizes);
    #else
    template<class SizesType>
    inline void reserve(const SizesType& reserveSizes, const typename SizesType::value_type& enableif = typename SizesType::value_type())
    {
      EIGEN_UNUSED_VARIABLE(enableif);
      reserveInnerVectors(reserveSizes);
    }
    template<class SizesType>
    inline void reserve(const SizesType& reserveSizes, const typename SizesType::Scalar& enableif =
    #if (!EIGEN_COMP_MSVC) || (EIGEN_COMP_MSVC>=1500) // MSVC 2005 fails to compile with this typename
        typename
    #endif
        SizesType::Scalar())
    {
      EIGEN_UNUSED_VARIABLE(enableif);
      reserveInnerVectors(reserveSizes);
    }
    #endif // EIGEN_PARSED_BY_DOXYGEN
  protected:
    template<class SizesType>
    inline void reserveInnerVectors(const SizesType& reserveSizes)
    {
      if(isCompressed())
      {
        std::size_t totalReserveSize = 0;
        // turn the matrix into non-compressed mode
        m_innerNonZeros = static_cast<Index*>(std::malloc(m_outerSize * sizeof(Index)));
        if (!m_innerNonZeros) internal::throw_std_bad_alloc();
        
        // temporarily use m_innerSizes to hold the new starting points.
        Index* newOuterIndex = m_innerNonZeros;
        
        Index count = 0;
        for(Index j=0; j<m_outerSize; ++j)
        {
          newOuterIndex[j] = count;
          count += reserveSizes[j] + (m_outerIndex[j+1]-m_outerIndex[j]);
          totalReserveSize += reserveSizes[j];
        }
        m_data.reserve(totalReserveSize);
        Index previousOuterIndex = m_outerIndex[m_outerSize];
        for(Index j=m_outerSize-1; j>=0; --j)
        {
          Index innerNNZ = previousOuterIndex - m_outerIndex[j];
          for(Index i=innerNNZ-1; i>=0; --i)
          {
            m_data.index(newOuterIndex[j]+i) = m_data.index(m_outerIndex[j]+i);
            m_data.value(newOuterIndex[j]+i) = m_data.value(m_outerIndex[j]+i);
          }
          previousOuterIndex = m_outerIndex[j];
          m_outerIndex[j] = newOuterIndex[j];
          m_innerNonZeros[j] = innerNNZ;
        }
        m_outerIndex[m_outerSize] = m_outerIndex[m_outerSize-1] + m_innerNonZeros[m_outerSize-1] + reserveSizes[m_outerSize-1];
        
        m_data.resize(m_outerIndex[m_outerSize]);
      }
      else
      {
        Index* newOuterIndex = static_cast<Index*>(std::malloc((m_outerSize+1)*sizeof(Index)));
        if (!newOuterIndex) internal::throw_std_bad_alloc();
        
        Index count = 0;
        for(Index j=0; j<m_outerSize; ++j)
        {
          newOuterIndex[j] = count;
          Index alreadyReserved = (m_outerIndex[j+1]-m_outerIndex[j]) - m_innerNonZeros[j];
          Index toReserve = std::max<Index>(reserveSizes[j], alreadyReserved);
          count += toReserve + m_innerNonZeros[j];
        }
        newOuterIndex[m_outerSize] = count;
        
        m_data.resize(count);
        for(Index j=m_outerSize-1; j>=0; --j)
        {
          Index offset = newOuterIndex[j] - m_outerIndex[j];
          if(offset>0)
          {
            Index innerNNZ = m_innerNonZeros[j];
            for(Index i=innerNNZ-1; i>=0; --i)
            {
              m_data.index(newOuterIndex[j]+i) = m_data.index(m_outerIndex[j]+i);
              m_data.value(newOuterIndex[j]+i) = m_data.value(m_outerIndex[j]+i);
            }
          }
        }
        
        std::swap(m_outerIndex, newOuterIndex);
        std::free(newOuterIndex);
      }
      
    }
  public:

    //--- low level purely coherent filling ---

    /** \internal
      * \returns a reference to the non zero coefficient at position \a row, \a col assuming that:
      * - the nonzero does not already exist
      * - the new coefficient is the last one according to the storage order
      *
      * Before filling a given inner vector you must call the statVec(Index) function.
      *
      * After an insertion session, you should call the finalize() function.
      *
      * \sa insert, insertBackByOuterInner, startVec */
    inline Scalar& insertBack(Index row, Index col)
    {
      return insertBackByOuterInner(IsRowMajor?row:col, IsRowMajor?col:row);
    }

    /** \internal
      * \sa insertBack, startVec */
    inline Scalar& insertBackByOuterInner(Index outer, Index inner)
    {
      eigen_assert(size_t(m_outerIndex[outer+1]) == m_data.size() && "Invalid ordered insertion (invalid outer index)");
      eigen_assert( (m_outerIndex[outer+1]-m_outerIndex[outer]==0 || m_data.index(m_data.size()-1)<inner) && "Invalid ordered insertion (invalid inner index)");
      Index p = m_outerIndex[outer+1];
      ++m_outerIndex[outer+1];
      m_data.append(Scalar(0), inner);
      return m_data.value(p);
    }

    /** \internal
      * \warning use it only if you know what you are doing */
    inline Scalar& insertBackByOuterInnerUnordered(Index outer, Index inner)
    {
      Index p = m_outerIndex[outer+1];
      ++m_outerIndex[outer+1];
      m_data.append(Scalar(0), inner);
      return m_data.value(p);
    }

    /** \internal
      * \sa insertBack, insertBackByOuterInner */
    inline void startVec(Index outer)
    {
      eigen_assert(m_outerIndex[outer]==Index(m_data.size()) && "You must call startVec for each inner vector sequentially");
      eigen_assert(m_outerIndex[outer+1]==0 && "You must call startVec for each inner vector sequentially");
      m_outerIndex[outer+1] = m_outerIndex[outer];
    }

    /** \internal
      * Must be called after inserting a set of non zero entries using the low level compressed API.
      */
    inline void finalize()
    {
      if(isCompressed())
      {
        Index size = static_cast<Index>(m_data.size());
        Index i = m_outerSize;
        // find the last filled column
        while (i>=0 && m_outerIndex[i]==0)
          --i;
        ++i;
        while (i<=m_outerSize)
        {
          m_outerIndex[i] = size;
          ++i;
        }
      }
    }

    //---

    template<typename InputIterators>
    void setFromTriplets(const InputIterators& begin, const InputIterators& end);

    void sumupDuplicates();

    //---
    
    /** \internal
      * same as insert(Index,Index) except that the indices are given relative to the storage order */
    Scalar& insertByOuterInner(Index j, Index i)
    {
      return insert(IsRowMajor ? j : i, IsRowMajor ? i : j);
    }

    /** Turns the matrix into the \em compressed format.
      */
    void makeCompressed()
    {
      if(isCompressed())
        return;
      
      Index oldStart = m_outerIndex[1];
      m_outerIndex[1] = m_innerNonZeros[0];
      for(Index j=1; j<m_outerSize; ++j)
      {
        Index nextOldStart = m_outerIndex[j+1];
        Index offset = oldStart - m_outerIndex[j];
        if(offset>0)
        {
          for(Index k=0; k<m_innerNonZeros[j]; ++k)
          {
            m_data.index(m_outerIndex[j]+k) = m_data.index(oldStart+k);
            m_data.value(m_outerIndex[j]+k) = m_data.value(oldStart+k);
          }
        }
        m_outerIndex[j+1] = m_outerIndex[j] + m_innerNonZeros[j];
        oldStart = nextOldStart;
      }
      std::free(m_innerNonZeros);
      m_innerNonZeros = 0;
      m_data.resize(m_outerIndex[m_outerSize]);
      m_data.squeeze();
    }

    /** Turns the matrix into the uncompressed mode */
    void uncompress()
    {
      if(m_innerNonZeros != 0)
        return; 
      m_innerNonZeros = static_cast<Index*>(std::malloc(m_outerSize * sizeof(Index)));
      for (Index i = 0; i < m_outerSize; i++)
      {
        m_innerNonZeros[i] = m_outerIndex[i+1] - m_outerIndex[i]; 
      }
    }
    
    /** Suppresses all nonzeros which are \b much \b smaller \b than \a reference under the tolerence \a epsilon */
    void prune(const Scalar& reference, const RealScalar& epsilon = NumTraits<RealScalar>::dummy_precision())
    {
      prune(default_prunning_func(reference,epsilon));
    }
    
    /** Turns the matrix into compressed format, and suppresses all nonzeros which do not satisfy the predicate \a keep.
      * The functor type \a KeepFunc must implement the following function:
      * \code
      * bool operator() (const Index& row, const Index& col, const Scalar& value) const;
      * \endcode
      * \sa prune(Scalar,RealScalar)
      */
    template<typename KeepFunc>
    void prune(const KeepFunc& keep = KeepFunc())
    {
      // TODO optimize the uncompressed mode to avoid moving and allocating the data twice
      // TODO also implement a unit test
      makeCompressed();

      Index k = 0;
      for(Index j=0; j<m_outerSize; ++j)
      {
        Index previousStart = m_outerIndex[j];
        m_outerIndex[j] = k;
        Index end = m_outerIndex[j+1];
        for(Index i=previousStart; i<end; ++i)
        {
          if(keep(IsRowMajor?j:m_data.index(i), IsRowMajor?m_data.index(i):j, m_data.value(i)))
          {
            m_data.value(k) = m_data.value(i);
            m_data.index(k) = m_data.index(i);
            ++k;
          }
        }
      }
      m_outerIndex[m_outerSize] = k;
      m_data.resize(k,0);
    }

    /** Resizes the matrix to a \a rows x \a cols matrix leaving old values untouched.
      * \sa resizeNonZeros(Index), reserve(), setZero()
      */
    void conservativeResize(Index rows, Index cols) 
    {
      // No change
      if (this->rows() == rows && this->cols() == cols) return;
      
      // If one dimension is null, then there is nothing to be preserved
      if(rows==0 || cols==0) return resize(rows,cols);

      Index innerChange = IsRowMajor ? cols - this->cols() : rows - this->rows();
      Index outerChange = IsRowMajor ? rows - this->rows() : cols - this->cols();
      Index newInnerSize = IsRowMajor ? cols : rows;

      // Deals with inner non zeros
      if (m_innerNonZeros)
      {
        // Resize m_innerNonZeros
        Index *newInnerNonZeros = static_cast<Index*>(std::realloc(m_innerNonZeros, (m_outerSize + outerChange) * sizeof(Index)));
        if (!newInnerNonZeros) internal::throw_std_bad_alloc();
        m_innerNonZeros = newInnerNonZeros;
        
        for(Index i=m_outerSize; i<m_outerSize+outerChange; i++)          
          m_innerNonZeros[i] = 0;
      } 
      else if (innerChange < 0) 
      {
        // Inner size decreased: allocate a new m_innerNonZeros
        m_innerNonZeros = static_cast<Index*>(std::malloc((m_outerSize+outerChange+1) * sizeof(Index)));
        if (!m_innerNonZeros) internal::throw_std_bad_alloc();
        for(Index i = 0; i < m_outerSize; i++)
          m_innerNonZeros[i] = m_outerIndex[i+1] - m_outerIndex[i];
      }
      
      // Change the m_innerNonZeros in case of a decrease of inner size
      if (m_innerNonZeros && innerChange < 0)
      {
        for(Index i = 0; i < m_outerSize + (std::min)(outerChange, Index(0)); i++)
        {
          Index &n = m_innerNonZeros[i];
          Index start = m_outerIndex[i];
          while (n > 0 && m_data.index(start+n-1) >= newInnerSize) --n; 
        }
      }
      
      m_innerSize = newInnerSize;

      // Re-allocate outer index structure if necessary
      if (outerChange == 0)
        return;
          
      Index *newOuterIndex = static_cast<Index*>(std::realloc(m_outerIndex, (m_outerSize + outerChange + 1) * sizeof(Index)));
      if (!newOuterIndex) internal::throw_std_bad_alloc();
      m_outerIndex = newOuterIndex;
      if (outerChange > 0)
      {
        Index last = m_outerSize == 0 ? 0 : m_outerIndex[m_outerSize];
        for(Index i=m_outerSize; i<m_outerSize+outerChange+1; i++)          
          m_outerIndex[i] = last; 
      }
      m_outerSize += outerChange;
    }
    
    /** Resizes the matrix to a \a rows x \a cols matrix and initializes it to zero.
      * \sa resizeNonZeros(Index), reserve(), setZero()
      */
    void resize(Index rows, Index cols)
    {
      const Index outerSize = IsRowMajor ? rows : cols;
      m_innerSize = IsRowMajor ? cols : rows;
      m_data.clear();
      if (m_outerSize != outerSize || m_outerSize==0)
      {
        std::free(m_outerIndex);
        m_outerIndex = static_cast<Index*>(std::malloc((outerSize + 1) * sizeof(Index)));
        if (!m_outerIndex) internal::throw_std_bad_alloc();
        
        m_outerSize = outerSize;
      }
      if(m_innerNonZeros)
      {
        std::free(m_innerNonZeros);
        m_innerNonZeros = 0;
      }
      memset(m_outerIndex, 0, (m_outerSize+1)*sizeof(Index));
    }

    /** \internal
      * Resize the nonzero vector to \a size */
    void resizeNonZeros(Index size)
    {
      // TODO remove this function
      m_data.resize(size);
    }

    /** \returns a const expression of the diagonal coefficients */
    const Diagonal<const SparseMatrix> diagonal() const { return *this; }

    /** Default constructor yielding an empty \c 0 \c x \c 0 matrix */
    inline SparseMatrix()
      : m_outerSize(-1), m_innerSize(0), m_outerIndex(0), m_innerNonZeros(0)
    {
      check_template_parameters();
      resize(0, 0);
    }

    /** Constructs a \a rows \c x \a cols empty matrix */
    inline SparseMatrix(Index rows, Index cols)
      : m_outerSize(0), m_innerSize(0), m_outerIndex(0), m_innerNonZeros(0)
    {
      check_template_parameters();
      resize(rows, cols);
    }

    /** Constructs a sparse matrix from the sparse expression \a other */
    template<typename OtherDerived>
    inline SparseMatrix(const SparseMatrixBase<OtherDerived>& other)
      : m_outerSize(0), m_innerSize(0), m_outerIndex(0), m_innerNonZeros(0)
    {
      EIGEN_STATIC_ASSERT((internal::is_same<Scalar, typename OtherDerived::Scalar>::value),
        YOU_MIXED_DIFFERENT_NUMERIC_TYPES__YOU_NEED_TO_USE_THE_CAST_METHOD_OF_MATRIXBASE_TO_CAST_NUMERIC_TYPES_EXPLICITLY)
      check_template_parameters();
      *this = other.derived();
    }
    
    /** Constructs a sparse matrix from the sparse selfadjoint view \a other */
    template<typename OtherDerived, unsigned int UpLo>
    inline SparseMatrix(const SparseSelfAdjointView<OtherDerived, UpLo>& other)
      : m_outerSize(0), m_innerSize(0), m_outerIndex(0), m_innerNonZeros(0)
    {
      check_template_parameters();
      *this = other;
    }

    /** Copy constructor (it performs a deep copy) */
    inline SparseMatrix(const SparseMatrix& other)
      : Base(), m_outerSize(0), m_innerSize(0), m_outerIndex(0), m_innerNonZeros(0)
    {
      check_template_parameters();
      *this = other.derived();
    }

    /** \brief Copy constructor with in-place evaluation */
    template<typename OtherDerived>
    SparseMatrix(const ReturnByValue<OtherDerived>& other)
      : Base(), m_outerSize(0), m_innerSize(0), m_outerIndex(0), m_innerNonZeros(0)
    {
      check_template_parameters();
      initAssignment(other);
      other.evalTo(*this);
    }

    /** Swaps the content of two sparse matrices of the same type.
      * This is a fast operation that simply swaps the underlying pointers and parameters. */
    inline void swap(SparseMatrix& other)
    {
      //EIGEN_DBG_SPARSE(std::cout << "SparseMatrix:: swap\n");
      std::swap(m_outerIndex, other.m_outerIndex);
      std::swap(m_innerSize, other.m_innerSize);
      std::swap(m_outerSize, other.m_outerSize);
      std::swap(m_innerNonZeros, other.m_innerNonZeros);
      m_data.swap(other.m_data);
    }

    /** Sets *this to the identity matrix */
    inline void setIdentity()
    {
      eigen_assert(rows() == cols() && "ONLY FOR SQUARED MATRICES");
      this->m_data.resize(rows());
      Eigen::Map<Matrix<Index, Dynamic, 1> >(&this->m_data.index(0), rows()).setLinSpaced(0, rows()-1);
      Eigen::Map<Matrix<Scalar, Dynamic, 1> >(&this->m_data.value(0), rows()).setOnes();
      Eigen::Map<Matrix<Index, Dynamic, 1> >(this->m_outerIndex, rows()+1).setLinSpaced(0, rows());
    }
    inline SparseMatrix& operator=(const SparseMatrix& other)
    {
      if (other.isRValue())
      {
        swap(other.const_cast_derived());
      }
      else if(this!=&other)
      {
        initAssignment(other);
        if(other.isCompressed())
        {
          internal::smart_copy(other.m_outerIndex, other.m_outerIndex + m_outerSize + 1, m_outerIndex);
          m_data = other.m_data;
        }
        else
        {
          Base::operator=(other);
        }
      }
      return *this;
    }

    #ifndef EIGEN_PARSED_BY_DOXYGEN
    template<typename Lhs, typename Rhs>
    inline SparseMatrix& operator=(const SparseSparseProduct<Lhs,Rhs>& product)
    { return Base::operator=(product); }
    
    template<typename OtherDerived>
    inline SparseMatrix& operator=(const ReturnByValue<OtherDerived>& other)
    {
      initAssignment(other);
      return Base::operator=(other.derived());
    }
    
    template<typename OtherDerived>
    inline SparseMatrix& operator=(const EigenBase<OtherDerived>& other)
    { return Base::operator=(other.derived()); }
    #endif

    template<typename OtherDerived>
    EIGEN_DONT_INLINE SparseMatrix& operator=(const SparseMatrixBase<OtherDerived>& other);

    friend std::ostream & operator << (std::ostream & s, const SparseMatrix& m)
    {
      EIGEN_DBG_SPARSE(
        s << "Nonzero entries:\n";
        if(m.isCompressed())
          for (Index i=0; i<m.nonZeros(); ++i)
            s << "(" << m.m_data.value(i) << "," << m.m_data.index(i) << ") ";
        else
          for (Index i=0; i<m.outerSize(); ++i)
          {
            Index p = m.m_outerIndex[i];
            Index pe = m.m_outerIndex[i]+m.m_innerNonZeros[i];
            Index k=p;
            for (; k<pe; ++k)
              s << "(" << m.m_data.value(k) << "," << m.m_data.index(k) << ") ";
            for (; k<m.m_outerIndex[i+1]; ++k)
              s << "(_,_) ";
          }
        s << std::endl;
        s << std::endl;
        s << "Outer pointers:\n";
        for (Index i=0; i<m.outerSize(); ++i)
          s << m.m_outerIndex[i] << " ";
        s << " $" << std::endl;
        if(!m.isCompressed())
        {
          s << "Inner non zeros:\n";
          for (Index i=0; i<m.outerSize(); ++i)
            s << m.m_innerNonZeros[i] << " ";
          s << " $" << std::endl;
        }
        s << std::endl;
      );
      s << static_cast<const SparseMatrixBase<SparseMatrix>&>(m);
      return s;
    }

    /** Destructor */
    inline ~SparseMatrix()
    {
      std::free(m_outerIndex);
      std::free(m_innerNonZeros);
    }

#ifndef EIGEN_PARSED_BY_DOXYGEN
    /** Overloaded for performance */
    Scalar sum() const;
#endif
    
#   ifdef EIGEN_SPARSEMATRIX_PLUGIN
#     include EIGEN_SPARSEMATRIX_PLUGIN
#   endif

protected:

    template<typename Other>
    void initAssignment(const Other& other)
    {
      resize(other.rows(), other.cols());
      if(m_innerNonZeros)
      {
        std::free(m_innerNonZeros);
        m_innerNonZeros = 0;
      }
    }

    /** \internal
      * \sa insert(Index,Index) */
    EIGEN_DONT_INLINE Scalar& insertCompressed(Index row, Index col);

    /** \internal
      * A vector object that is equal to 0 everywhere but v at the position i */
    class SingletonVector
    {
        Index m_index;
        Index m_value;
      public:
        typedef Index value_type;
        SingletonVector(Index i, Index v)
          : m_index(i), m_value(v)
        {}

        Index operator[](Index i) const { return i==m_index ? m_value : 0; }
    };

    /** \internal
      * \sa insert(Index,Index) */
    EIGEN_DONT_INLINE Scalar& insertUncompressed(Index row, Index col);

public:
    /** \internal
      * \sa insert(Index,Index) */
    EIGEN_STRONG_INLINE Scalar& insertBackUncompressed(Index row, Index col)
    {
      const Index outer = IsRowMajor ? row : col;
      const Index inner = IsRowMajor ? col : row;

      eigen_assert(!isCompressed());
      eigen_assert(m_innerNonZeros[outer]<=(m_outerIndex[outer+1] - m_outerIndex[outer]));

      Index p = m_outerIndex[outer] + m_innerNonZeros[outer]++;
      m_data.index(p) = inner;
      return (m_data.value(p) = 0);
    }

private:
  static void check_template_parameters()
  {
    EIGEN_STATIC_ASSERT(NumTraits<Index>::IsSigned,THE_INDEX_TYPE_MUST_BE_A_SIGNED_TYPE);
    EIGEN_STATIC_ASSERT((Options&(ColMajor|RowMajor))==Options,INVALID_MATRIX_TEMPLATE_PARAMETERS);
  }

  struct default_prunning_func {
    default_prunning_func(const Scalar& ref, const RealScalar& eps) : reference(ref), epsilon(eps) {}
    inline bool operator() (const Index&, const Index&, const Scalar& value) const
    {
      return !internal::isMuchSmallerThan(value, reference, epsilon);
    }
    Scalar reference;
    RealScalar epsilon;
  };
};

template<typename Scalar, int _Options, typename _Index>
class SparseMatrix<Scalar,_Options,_Index>::InnerIterator
{
  public:
    InnerIterator(const SparseMatrix& mat, Index outer)
      : m_values(mat.valuePtr()), m_indices(mat.innerIndexPtr()), m_outer(outer), m_id(mat.m_outerIndex[outer])
    {
      if(mat.isCompressed())
        m_end = mat.m_outerIndex[outer+1];
      else
        m_end = m_id + mat.m_innerNonZeros[outer];
    }

    inline InnerIterator& operator++() { m_id++; return *this; }

    inline const Scalar& value() const { return m_values[m_id]; }
    inline Scalar& valueRef() { return const_cast<Scalar&>(m_values[m_id]); }

    inline Index index() const { return m_indices[m_id]; }
    inline Index outer() const { return m_outer; }
    inline Index row() const { return IsRowMajor ? m_outer : index(); }
    inline Index col() const { return IsRowMajor ? index() : m_outer; }

    inline operator bool() const { return (m_id < m_end); }

  protected:
    const Scalar* m_values;
    const Index* m_indices;
    const Index m_outer;
    Index m_id;
    Index m_end;
};

template<typename Scalar, int _Options, typename _Index>
class SparseMatrix<Scalar,_Options,_Index>::ReverseInnerIterator
{
  public:
    ReverseInnerIterator(const SparseMatrix& mat, Index outer)
      : m_values(mat.valuePtr()), m_indices(mat.innerIndexPtr()), m_outer(outer), m_start(mat.m_outerIndex[outer])
    {
      if(mat.isCompressed())
        m_id = mat.m_outerIndex[outer+1];
      else
        m_id = m_start + mat.m_innerNonZeros[outer];
    }

    inline ReverseInnerIterator& operator--() { --m_id; return *this; }

    inline const Scalar& value() const { return m_values[m_id-1]; }
    inline Scalar& valueRef() { return const_cast<Scalar&>(m_values[m_id-1]); }

    inline Index index() const { return m_indices[m_id-1]; }
    inline Index outer() const { return m_outer; }
    inline Index row() const { return IsRowMajor ? m_outer : index(); }
    inline Index col() const { return IsRowMajor ? index() : m_outer; }

    inline operator bool() const { return (m_id > m_start); }

  protected:
    const Scalar* m_values;
    const Index* m_indices;
    const Index m_outer;
    Index m_id;
    const Index m_start;
};

namespace internal {

template<typename InputIterator, typename SparseMatrixType>
void set_from_triplets(const InputIterator& begin, const InputIterator& end, SparseMatrixType& mat, int Options = 0)
{
  EIGEN_UNUSED_VARIABLE(Options);
  enum { IsRowMajor = SparseMatrixType::IsRowMajor };
  typedef typename SparseMatrixType::Scalar Scalar;
  typedef typename SparseMatrixType::Index Index;
  SparseMatrix<Scalar,IsRowMajor?ColMajor:RowMajor> trMat(mat.rows(),mat.cols());

  if(begin!=end)
  {
    // pass 1: count the nnz per inner-vector
    Matrix<Index,Dynamic,1> wi(trMat.outerSize());
    wi.setZero();
    for(InputIterator it(begin); it!=end; ++it)
    {
      eigen_assert(it->row()>=0 && it->row()<mat.rows() && it->col()>=0 && it->col()<mat.cols());
      wi(IsRowMajor ? it->col() : it->row())++;
    }

    // pass 2: insert all the elements into trMat
    trMat.reserve(wi);
    for(InputIterator it(begin); it!=end; ++it)
      trMat.insertBackUncompressed(it->row(),it->col()) = it->value();

    // pass 3:
    trMat.sumupDuplicates();
  }

  // pass 4: transposed copy -> implicit sorting
  mat = trMat;
}

}


/** Fill the matrix \c *this with the list of \em triplets defined by the iterator range \a begin - \a end.
  *
  * A \em triplet is a tuple (i,j,value) defining a non-zero element.
  * The input list of triplets does not have to be sorted, and can contains duplicated elements.
  * In any case, the result is a \b sorted and \b compressed sparse matrix where the duplicates have been summed up.
  * This is a \em O(n) operation, with \em n the number of triplet elements.
  * The initial contents of \c *this is destroyed.
  * The matrix \c *this must be properly resized beforehand using the SparseMatrix(Index,Index) constructor,
  * or the resize(Index,Index) method. The sizes are not extracted from the triplet list.
  *
  * The \a InputIterators value_type must provide the following interface:
  * \code
  * Scalar value() const; // the value
  * Scalar row() const;   // the row index i
  * Scalar col() const;   // the column index j
  * \endcode
  * See for instance the Eigen::Triplet template class.
  *
  * Here is a typical usage example:
  * \code
    typedef Triplet<double> T;
    std::vector<T> tripletList;
    triplets.reserve(estimation_of_entries);
    for(...)
    {
      // ...
      tripletList.push_back(T(i,j,v_ij));
    }
    SparseMatrixType m(rows,cols);
    m.setFromTriplets(tripletList.begin(), tripletList.end());
    // m is ready to go!
  * \endcode
  *
  * \warning The list of triplets is read multiple times (at least twice). Therefore, it is not recommended to define
  * an abstract iterator over a complex data-structure that would be expensive to evaluate. The triplets should rather
  * be explicitely stored into a std::vector for instance.
  */
template<typename Scalar, int _Options, typename _Index>
template<typename InputIterators>
void SparseMatrix<Scalar,_Options,_Index>::setFromTriplets(const InputIterators& begin, const InputIterators& end)
{
  internal::set_from_triplets(begin, end, *this);
}

/** \internal */
template<typename Scalar, int _Options, typename _Index>
void SparseMatrix<Scalar,_Options,_Index>::sumupDuplicates()
{
  eigen_assert(!isCompressed());
  // TODO, in practice we should be able to use m_innerNonZeros for that task
  Matrix<Index,Dynamic,1> wi(innerSize());
  wi.fill(-1);
  Index count = 0;
  // for each inner-vector, wi[inner_index] will hold the position of first element into the index/value buffers
  for(Index j=0; j<outerSize(); ++j)
  {
    Index start   = count;
    Index oldEnd  = m_outerIndex[j]+m_innerNonZeros[j];
    for(Index k=m_outerIndex[j]; k<oldEnd; ++k)
    {
      Index i = m_data.index(k);
      if(wi(i)>=start)
      {
        // we already meet this entry => accumulate it
        m_data.value(wi(i)) += m_data.value(k);
      }
      else
      {
        m_data.value(count) = m_data.value(k);
        m_data.index(count) = m_data.index(k);
        wi(i) = count;
        ++count;
      }
    }
    m_outerIndex[j] = start;
  }
  m_outerIndex[m_outerSize] = count;

  // turn the matrix into compressed form
  std::free(m_innerNonZeros);
  m_innerNonZeros = 0;
  m_data.resize(m_outerIndex[m_outerSize]);
}

template<typename Scalar, int _Options, typename _Index>
template<typename OtherDerived>
EIGEN_DONT_INLINE SparseMatrix<Scalar,_Options,_Index>& SparseMatrix<Scalar,_Options,_Index>::operator=(const SparseMatrixBase<OtherDerived>& other)
{
  EIGEN_STATIC_ASSERT((internal::is_same<Scalar, typename OtherDerived::Scalar>::value),
        YOU_MIXED_DIFFERENT_NUMERIC_TYPES__YOU_NEED_TO_USE_THE_CAST_METHOD_OF_MATRIXBASE_TO_CAST_NUMERIC_TYPES_EXPLICITLY)
  
  const bool needToTranspose = (Flags & RowMajorBit) != (OtherDerived::Flags & RowMajorBit);
  if (needToTranspose)
  {
    // two passes algorithm:
    //  1 - compute the number of coeffs per dest inner vector
    //  2 - do the actual copy/eval
    // Since each coeff of the rhs has to be evaluated twice, let's evaluate it if needed
    typedef typename internal::nested<OtherDerived,2>::type OtherCopy;
    typedef typename internal::remove_all<OtherCopy>::type _OtherCopy;
    OtherCopy otherCopy(other.derived());

    SparseMatrix dest(other.rows(),other.cols());
    Eigen::Map<Matrix<Index, Dynamic, 1> > (dest.m_outerIndex,dest.outerSize()).setZero();

    // pass 1
    // FIXME the above copy could be merged with that pass
    for (Index j=0; j<otherCopy.outerSize(); ++j)
      for (typename _OtherCopy::InnerIterator it(otherCopy, j); it; ++it)
        ++dest.m_outerIndex[it.index()];

    // prefix sum
    Index count = 0;
    Matrix<Index,Dynamic,1> positions(dest.outerSize());
    for (Index j=0; j<dest.outerSize(); ++j)
    {
      Index tmp = dest.m_outerIndex[j];
      dest.m_outerIndex[j] = count;
      positions[j] = count;
      count += tmp;
    }
    dest.m_outerIndex[dest.outerSize()] = count;
    // alloc
    dest.m_data.resize(count);
    // pass 2
    for (Index j=0; j<otherCopy.outerSize(); ++j)
    {
      for (typename _OtherCopy::InnerIterator it(otherCopy, j); it; ++it)
      {
        Index pos = positions[it.index()]++;
        dest.m_data.index(pos) = j;
        dest.m_data.value(pos) = it.value();
      }
    }
    this->swap(dest);
    return *this;
  }
  else
  {
    if(other.isRValue())
      initAssignment(other.derived());
    // there is no special optimization
    return Base::operator=(other.derived());
  }
}

template<typename _Scalar, int _Options, typename _Index>
EIGEN_DONT_INLINE typename SparseMatrix<_Scalar,_Options,_Index>::Scalar& SparseMatrix<_Scalar,_Options,_Index>::insertUncompressed(Index row, Index col)
{
  eigen_assert(!isCompressed());

  const Index outer = IsRowMajor ? row : col;
  const Index inner = IsRowMajor ? col : row;

  Index room = m_outerIndex[outer+1] - m_outerIndex[outer];
  Index innerNNZ = m_innerNonZeros[outer];
  if(innerNNZ>=room)
  {
    // this inner vector is full, we need to reallocate the whole buffer :(
    reserve(SingletonVector(outer,std::max<Index>(2,innerNNZ)));
  }

  Index startId = m_outerIndex[outer];
  Index p = startId + m_innerNonZeros[outer];
  while ( (p > startId) && (m_data.index(p-1) > inner) )
  {
    m_data.index(p) = m_data.index(p-1);
    m_data.value(p) = m_data.value(p-1);
    --p;
  }
  eigen_assert((p<=startId || m_data.index(p-1)!=inner) && "you cannot insert an element that already exists, you must call coeffRef to this end");

  m_innerNonZeros[outer]++;

  m_data.index(p) = inner;
  return (m_data.value(p) = 0);
}

template<typename _Scalar, int _Options, typename _Index>
EIGEN_DONT_INLINE typename SparseMatrix<_Scalar,_Options,_Index>::Scalar& SparseMatrix<_Scalar,_Options,_Index>::insertCompressed(Index row, Index col)
{
  eigen_assert(isCompressed());

  const Index outer = IsRowMajor ? row : col;
  const Index inner = IsRowMajor ? col : row;

  Index previousOuter = outer;
  if (m_outerIndex[outer+1]==0)
  {
    // we start a new inner vector
    while (previousOuter>=0 && m_outerIndex[previousOuter]==0)
    {
      m_outerIndex[previousOuter] = static_cast<Index>(m_data.size());
      --previousOuter;
    }
    m_outerIndex[outer+1] = m_outerIndex[outer];
  }

  // here we have to handle the tricky case where the outerIndex array
  // starts with: [ 0 0 0 0 0 1 ...] and we are inserted in, e.g.,
  // the 2nd inner vector...
  bool isLastVec = (!(previousOuter==-1 && m_data.size()!=0))
                && (size_t(m_outerIndex[outer+1]) == m_data.size());

  size_t startId = m_outerIndex[outer];
  // FIXME let's make sure sizeof(long int) == sizeof(size_t)
  size_t p = m_outerIndex[outer+1];
  ++m_outerIndex[outer+1];

  float reallocRatio = 1;
  if (m_data.allocatedSize()<=m_data.size())
  {
    // if there is no preallocated memory, let's reserve a minimum of 32 elements
    if (m_data.size()==0)
    {
      m_data.reserve(32);
    }
    else
    {
      // we need to reallocate the data, to reduce multiple reallocations
      // we use a smart resize algorithm based on the current filling ratio
      // in addition, we use float to avoid integers overflows
      float nnzEstimate = float(m_outerIndex[outer])*float(m_outerSize)/float(outer+1);
      reallocRatio = (nnzEstimate-float(m_data.size()))/float(m_data.size());
      // furthermore we bound the realloc ratio to:
      //   1) reduce multiple minor realloc when the matrix is almost filled
      //   2) avoid to allocate too much memory when the matrix is almost empty
      reallocRatio = (std::min)((std::max)(reallocRatio,1.5f),8.f);
    }
  }
  m_data.resize(m_data.size()+1,reallocRatio);

  if (!isLastVec)
  {
    if (previousOuter==-1)
    {
      // oops wrong guess.
      // let's correct the outer offsets
      for (Index k=0; k<=(outer+1); ++k)
        m_outerIndex[k] = 0;
      Index k=outer+1;
      while(m_outerIndex[k]==0)
        m_outerIndex[k++] = 1;
      while (k<=m_outerSize && m_outerIndex[k]!=0)
        m_outerIndex[k++]++;
      p = 0;
      --k;
      k = m_outerIndex[k]-1;
      while (k>0)
      {
        m_data.index(k) = m_data.index(k-1);
        m_data.value(k) = m_data.value(k-1);
        k--;
      }
    }
    else
    {
      // we are not inserting into the last inner vec
      // update outer indices:
      Index j = outer+2;
      while (j<=m_outerSize && m_outerIndex[j]!=0)
        m_outerIndex[j++]++;
      --j;
      // shift data of last vecs:
      Index k = m_outerIndex[j]-1;
      while (k>=Index(p))
      {
        m_data.index(k) = m_data.index(k-1);
        m_data.value(k) = m_data.value(k-1);
        k--;
      }
    }
  }

  while ( (p > startId) && (m_data.index(p-1) > inner) )
  {
    m_data.index(p) = m_data.index(p-1);
    m_data.value(p) = m_data.value(p-1);
    --p;
  }

  m_data.index(p) = inner;
  return (m_data.value(p) = 0);
}

} // end namespace Eigen

#endif // EIGEN_SPARSEMATRIX_H
