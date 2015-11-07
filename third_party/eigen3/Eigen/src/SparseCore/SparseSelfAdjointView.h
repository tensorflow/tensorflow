// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2009 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_SPARSE_SELFADJOINTVIEW_H
#define EIGEN_SPARSE_SELFADJOINTVIEW_H

namespace Eigen { 

/** \ingroup SparseCore_Module
  * \class SparseSelfAdjointView
  *
  * \brief Pseudo expression to manipulate a triangular sparse matrix as a selfadjoint matrix.
  *
  * \param MatrixType the type of the dense matrix storing the coefficients
  * \param UpLo can be either \c #Lower or \c #Upper
  *
  * This class is an expression of a sefladjoint matrix from a triangular part of a matrix
  * with given dense storage of the coefficients. It is the return type of MatrixBase::selfadjointView()
  * and most of the time this is the only way that it is used.
  *
  * \sa SparseMatrixBase::selfadjointView()
  */
template<typename Lhs, typename Rhs, int UpLo>
class SparseSelfAdjointTimeDenseProduct;

template<typename Lhs, typename Rhs, int UpLo>
class DenseTimeSparseSelfAdjointProduct;

namespace internal {
  
template<typename MatrixType, unsigned int UpLo>
struct traits<SparseSelfAdjointView<MatrixType,UpLo> > : traits<MatrixType> {
};

template<int SrcUpLo,int DstUpLo,typename MatrixType,int DestOrder>
void permute_symm_to_symm(const MatrixType& mat, SparseMatrix<typename MatrixType::Scalar,DestOrder,typename MatrixType::Index>& _dest, const typename MatrixType::Index* perm = 0);

template<int UpLo,typename MatrixType,int DestOrder>
void permute_symm_to_fullsymm(const MatrixType& mat, SparseMatrix<typename MatrixType::Scalar,DestOrder,typename MatrixType::Index>& _dest, const typename MatrixType::Index* perm = 0);

}

template<typename MatrixType, unsigned int UpLo> class SparseSelfAdjointView
  : public EigenBase<SparseSelfAdjointView<MatrixType,UpLo> >
{
  public:

    typedef typename MatrixType::Scalar Scalar;
    typedef typename MatrixType::Index Index;
    typedef Matrix<Index,Dynamic,1> VectorI;
    typedef typename MatrixType::Nested MatrixTypeNested;
    typedef typename internal::remove_all<MatrixTypeNested>::type _MatrixTypeNested;

    inline SparseSelfAdjointView(const MatrixType& matrix) : m_matrix(matrix)
    {
      eigen_assert(rows()==cols() && "SelfAdjointView is only for squared matrices");
    }

    inline Index rows() const { return m_matrix.rows(); }
    inline Index cols() const { return m_matrix.cols(); }

    /** \internal \returns a reference to the nested matrix */
    const _MatrixTypeNested& matrix() const { return m_matrix; }
    _MatrixTypeNested& matrix() { return m_matrix.const_cast_derived(); }

    /** \returns an expression of the matrix product between a sparse self-adjoint matrix \c *this and a sparse matrix \a rhs.
      *
      * Note that there is no algorithmic advantage of performing such a product compared to a general sparse-sparse matrix product.
      * Indeed, the SparseSelfadjointView operand is first copied into a temporary SparseMatrix before computing the product.
      */
    template<typename OtherDerived>
    SparseSparseProduct<typename OtherDerived::PlainObject, OtherDerived>
    operator*(const SparseMatrixBase<OtherDerived>& rhs) const
    {
      return SparseSparseProduct<typename OtherDerived::PlainObject, OtherDerived>(*this, rhs.derived());
    }

    /** \returns an expression of the matrix product between a sparse matrix \a lhs and a sparse self-adjoint matrix \a rhs.
      *
      * Note that there is no algorithmic advantage of performing such a product compared to a general sparse-sparse matrix product.
      * Indeed, the SparseSelfadjointView operand is first copied into a temporary SparseMatrix before computing the product.
      */
    template<typename OtherDerived> friend
    SparseSparseProduct<OtherDerived, typename OtherDerived::PlainObject >
    operator*(const SparseMatrixBase<OtherDerived>& lhs, const SparseSelfAdjointView& rhs)
    {
      return SparseSparseProduct<OtherDerived, typename OtherDerived::PlainObject>(lhs.derived(), rhs);
    }
    
    /** Efficient sparse self-adjoint matrix times dense vector/matrix product */
    template<typename OtherDerived>
    SparseSelfAdjointTimeDenseProduct<MatrixType,OtherDerived,UpLo>
    operator*(const MatrixBase<OtherDerived>& rhs) const
    {
      return SparseSelfAdjointTimeDenseProduct<MatrixType,OtherDerived,UpLo>(m_matrix, rhs.derived());
    }

    /** Efficient dense vector/matrix times sparse self-adjoint matrix product */
    template<typename OtherDerived> friend
    DenseTimeSparseSelfAdjointProduct<OtherDerived,MatrixType,UpLo>
    operator*(const MatrixBase<OtherDerived>& lhs, const SparseSelfAdjointView& rhs)
    {
      return DenseTimeSparseSelfAdjointProduct<OtherDerived,_MatrixTypeNested,UpLo>(lhs.derived(), rhs.m_matrix);
    }

    /** Perform a symmetric rank K update of the selfadjoint matrix \c *this:
      * \f$ this = this + \alpha ( u u^* ) \f$ where \a u is a vector or matrix.
      *
      * \returns a reference to \c *this
      *
      * To perform \f$ this = this + \alpha ( u^* u ) \f$ you can simply
      * call this function with u.adjoint().
      */
    template<typename DerivedU>
    SparseSelfAdjointView& rankUpdate(const SparseMatrixBase<DerivedU>& u, const Scalar& alpha = Scalar(1));
    
    /** \internal triggered by sparse_matrix = SparseSelfadjointView; */
    template<typename DestScalar,int StorageOrder> void evalTo(SparseMatrix<DestScalar,StorageOrder,Index>& _dest) const
    {
      internal::permute_symm_to_fullsymm<UpLo>(m_matrix, _dest);
    }
    
    template<typename DestScalar> void evalTo(DynamicSparseMatrix<DestScalar,ColMajor,Index>& _dest) const
    {
      // TODO directly evaluate into _dest;
      SparseMatrix<DestScalar,ColMajor,Index> tmp(_dest.rows(),_dest.cols());
      internal::permute_symm_to_fullsymm<UpLo>(m_matrix, tmp);
      _dest = tmp;
    }
    
    /** \returns an expression of P H P^-1 */
    SparseSymmetricPermutationProduct<_MatrixTypeNested,UpLo> twistedBy(const PermutationMatrix<Dynamic,Dynamic,Index>& perm) const
    {
      return SparseSymmetricPermutationProduct<_MatrixTypeNested,UpLo>(m_matrix, perm);
    }
    
    template<typename SrcMatrixType,int SrcUpLo>
    SparseSelfAdjointView& operator=(const SparseSymmetricPermutationProduct<SrcMatrixType,SrcUpLo>& permutedMatrix)
    {
      permutedMatrix.evalTo(*this);
      return *this;
    }


    SparseSelfAdjointView& operator=(const SparseSelfAdjointView& src)
    {
      PermutationMatrix<Dynamic> pnull;
      return *this = src.twistedBy(pnull);
    }

    template<typename SrcMatrixType,unsigned int SrcUpLo>
    SparseSelfAdjointView& operator=(const SparseSelfAdjointView<SrcMatrixType,SrcUpLo>& src)
    {
      PermutationMatrix<Dynamic> pnull;
      return *this = src.twistedBy(pnull);
    }
    

    // const SparseLLT<PlainObject, UpLo> llt() const;
    // const SparseLDLT<PlainObject, UpLo> ldlt() const;

  protected:

    typename MatrixType::Nested m_matrix;
    mutable VectorI m_countPerRow;
    mutable VectorI m_countPerCol;
};

/***************************************************************************
* Implementation of SparseMatrixBase methods
***************************************************************************/

template<typename Derived>
template<unsigned int UpLo>
const SparseSelfAdjointView<Derived, UpLo> SparseMatrixBase<Derived>::selfadjointView() const
{
  return derived();
}

template<typename Derived>
template<unsigned int UpLo>
SparseSelfAdjointView<Derived, UpLo> SparseMatrixBase<Derived>::selfadjointView()
{
  return derived();
}

/***************************************************************************
* Implementation of SparseSelfAdjointView methods
***************************************************************************/

template<typename MatrixType, unsigned int UpLo>
template<typename DerivedU>
SparseSelfAdjointView<MatrixType,UpLo>&
SparseSelfAdjointView<MatrixType,UpLo>::rankUpdate(const SparseMatrixBase<DerivedU>& u, const Scalar& alpha)
{
  SparseMatrix<Scalar,MatrixType::Flags&RowMajorBit?RowMajor:ColMajor> tmp = u * u.adjoint();
  if(alpha==Scalar(0))
    m_matrix.const_cast_derived() = tmp.template triangularView<UpLo>();
  else
    m_matrix.const_cast_derived() += alpha * tmp.template triangularView<UpLo>();

  return *this;
}

/***************************************************************************
* Implementation of sparse self-adjoint time dense matrix
***************************************************************************/

namespace internal {
template<typename Lhs, typename Rhs, int UpLo>
struct traits<SparseSelfAdjointTimeDenseProduct<Lhs,Rhs,UpLo> >
 : traits<ProductBase<SparseSelfAdjointTimeDenseProduct<Lhs,Rhs,UpLo>, Lhs, Rhs> >
{
  typedef Dense StorageKind;
};
}

template<typename Lhs, typename Rhs, int UpLo>
class SparseSelfAdjointTimeDenseProduct
  : public ProductBase<SparseSelfAdjointTimeDenseProduct<Lhs,Rhs,UpLo>, Lhs, Rhs>
{
  public:
    EIGEN_PRODUCT_PUBLIC_INTERFACE(SparseSelfAdjointTimeDenseProduct)

    SparseSelfAdjointTimeDenseProduct(const Lhs& lhs, const Rhs& rhs) : Base(lhs,rhs)
    {}

    template<typename Dest> void scaleAndAddTo(Dest& dest, const Scalar& alpha) const
    {
      EIGEN_ONLY_USED_FOR_DEBUG(alpha);
      // TODO use alpha
      eigen_assert(alpha==Scalar(1) && "alpha != 1 is not implemented yet, sorry");
      typedef typename internal::remove_all<Lhs>::type _Lhs;
      typedef typename _Lhs::InnerIterator LhsInnerIterator;
      enum {
        LhsIsRowMajor = (_Lhs::Flags&RowMajorBit)==RowMajorBit,
        ProcessFirstHalf =
                 ((UpLo&(Upper|Lower))==(Upper|Lower))
              || ( (UpLo&Upper) && !LhsIsRowMajor)
              || ( (UpLo&Lower) && LhsIsRowMajor),
        ProcessSecondHalf = !ProcessFirstHalf
      };
      for (Index j=0; j<m_lhs.outerSize(); ++j)
      {
        LhsInnerIterator i(m_lhs,j);
        if (ProcessSecondHalf)
        {
          while (i && i.index()<j) ++i;
          if(i && i.index()==j)
          {
            dest.row(j) += i.value() * m_rhs.row(j);
            ++i;
          }
        }
        for(; (ProcessFirstHalf ? i && i.index() < j : i) ; ++i)
        {
          Index a = LhsIsRowMajor ? j : i.index();
          Index b = LhsIsRowMajor ? i.index() : j;
          typename Lhs::Scalar v = i.value();
          dest.row(a) += (v) * m_rhs.row(b);
          dest.row(b) += numext::conj(v) * m_rhs.row(a);
        }
        if (ProcessFirstHalf && i && (i.index()==j))
          dest.row(j) += i.value() * m_rhs.row(j);
      }
    }

  private:
    SparseSelfAdjointTimeDenseProduct& operator=(const SparseSelfAdjointTimeDenseProduct&);
};

namespace internal {
template<typename Lhs, typename Rhs, int UpLo>
struct traits<DenseTimeSparseSelfAdjointProduct<Lhs,Rhs,UpLo> >
 : traits<ProductBase<DenseTimeSparseSelfAdjointProduct<Lhs,Rhs,UpLo>, Lhs, Rhs> >
{};
}

template<typename Lhs, typename Rhs, int UpLo>
class DenseTimeSparseSelfAdjointProduct
  : public ProductBase<DenseTimeSparseSelfAdjointProduct<Lhs,Rhs,UpLo>, Lhs, Rhs>
{
  public:
    EIGEN_PRODUCT_PUBLIC_INTERFACE(DenseTimeSparseSelfAdjointProduct)

    DenseTimeSparseSelfAdjointProduct(const Lhs& lhs, const Rhs& rhs) : Base(lhs,rhs)
    {}

    template<typename Dest> void scaleAndAddTo(Dest& /*dest*/, const Scalar& /*alpha*/) const
    {
      // TODO
    }

  private:
    DenseTimeSparseSelfAdjointProduct& operator=(const DenseTimeSparseSelfAdjointProduct&);
};

/***************************************************************************
* Implementation of symmetric copies and permutations
***************************************************************************/
namespace internal {
  
template<typename MatrixType, int UpLo>
struct traits<SparseSymmetricPermutationProduct<MatrixType,UpLo> > : traits<MatrixType> {
};

template<int UpLo,typename MatrixType,int DestOrder>
void permute_symm_to_fullsymm(const MatrixType& mat, SparseMatrix<typename MatrixType::Scalar,DestOrder,typename MatrixType::Index>& _dest, const typename MatrixType::Index* perm)
{
  typedef typename MatrixType::Index Index;
  typedef typename MatrixType::Scalar Scalar;
  typedef SparseMatrix<Scalar,DestOrder,Index> Dest;
  typedef Matrix<Index,Dynamic,1> VectorI;
  
  Dest& dest(_dest.derived());
  enum {
    StorageOrderMatch = int(Dest::IsRowMajor) == int(MatrixType::IsRowMajor)
  };
  
  Index size = mat.rows();
  VectorI count;
  count.resize(size);
  count.setZero();
  dest.resize(size,size);
  for(Index j = 0; j<size; ++j)
  {
    Index jp = perm ? perm[j] : j;
    for(typename MatrixType::InnerIterator it(mat,j); it; ++it)
    {
      Index i = it.index();
      Index r = it.row();
      Index c = it.col();
      Index ip = perm ? perm[i] : i;
      if(UpLo==(Upper|Lower))
        count[StorageOrderMatch ? jp : ip]++;
      else if(r==c)
        count[ip]++;
      else if(( UpLo==Lower && r>c) || ( UpLo==Upper && r<c))
      {
        count[ip]++;
        count[jp]++;
      }
    }
  }
  Index nnz = count.sum();
  
  // reserve space
  dest.resizeNonZeros(nnz);
  dest.outerIndexPtr()[0] = 0;
  for(Index j=0; j<size; ++j)
    dest.outerIndexPtr()[j+1] = dest.outerIndexPtr()[j] + count[j];
  for(Index j=0; j<size; ++j)
    count[j] = dest.outerIndexPtr()[j];
  
  // copy data
  for(Index j = 0; j<size; ++j)
  {
    for(typename MatrixType::InnerIterator it(mat,j); it; ++it)
    {
      Index i = it.index();
      Index r = it.row();
      Index c = it.col();
      
      Index jp = perm ? perm[j] : j;
      Index ip = perm ? perm[i] : i;
      
      if(UpLo==(Upper|Lower))
      {
        Index k = count[StorageOrderMatch ? jp : ip]++;
        dest.innerIndexPtr()[k] = StorageOrderMatch ? ip : jp;
        dest.valuePtr()[k] = it.value();
      }
      else if(r==c)
      {
        Index k = count[ip]++;
        dest.innerIndexPtr()[k] = ip;
        dest.valuePtr()[k] = it.value();
      }
      else if(( (UpLo&Lower)==Lower && r>c) || ( (UpLo&Upper)==Upper && r<c))
      {
        if(!StorageOrderMatch)
          std::swap(ip,jp);
        Index k = count[jp]++;
        dest.innerIndexPtr()[k] = ip;
        dest.valuePtr()[k] = it.value();
        k = count[ip]++;
        dest.innerIndexPtr()[k] = jp;
        dest.valuePtr()[k] = numext::conj(it.value());
      }
    }
  }
}

template<int _SrcUpLo,int _DstUpLo,typename MatrixType,int DstOrder>
void permute_symm_to_symm(const MatrixType& mat, SparseMatrix<typename MatrixType::Scalar,DstOrder,typename MatrixType::Index>& _dest, const typename MatrixType::Index* perm)
{
  typedef typename MatrixType::Index Index;
  typedef typename MatrixType::Scalar Scalar;
  SparseMatrix<Scalar,DstOrder,Index>& dest(_dest.derived());
  typedef Matrix<Index,Dynamic,1> VectorI;
  enum {
    SrcOrder = MatrixType::IsRowMajor ? RowMajor : ColMajor,
    StorageOrderMatch = int(SrcOrder) == int(DstOrder),
    DstUpLo = DstOrder==RowMajor ? (_DstUpLo==Upper ? Lower : Upper) : _DstUpLo,
    SrcUpLo = SrcOrder==RowMajor ? (_SrcUpLo==Upper ? Lower : Upper) : _SrcUpLo
  };
  
  Index size = mat.rows();
  VectorI count(size);
  count.setZero();
  dest.resize(size,size);
  for(Index j = 0; j<size; ++j)
  {
    Index jp = perm ? perm[j] : j;
    for(typename MatrixType::InnerIterator it(mat,j); it; ++it)
    {
      Index i = it.index();
      if((int(SrcUpLo)==int(Lower) && i<j) || (int(SrcUpLo)==int(Upper) && i>j))
        continue;
                  
      Index ip = perm ? perm[i] : i;
      count[int(DstUpLo)==int(Lower) ? (std::min)(ip,jp) : (std::max)(ip,jp)]++;
    }
  }
  dest.outerIndexPtr()[0] = 0;
  for(Index j=0; j<size; ++j)
    dest.outerIndexPtr()[j+1] = dest.outerIndexPtr()[j] + count[j];
  dest.resizeNonZeros(dest.outerIndexPtr()[size]);
  for(Index j=0; j<size; ++j)
    count[j] = dest.outerIndexPtr()[j];
  
  for(Index j = 0; j<size; ++j)
  {
    
    for(typename MatrixType::InnerIterator it(mat,j); it; ++it)
    {
      Index i = it.index();
      if((int(SrcUpLo)==int(Lower) && i<j) || (int(SrcUpLo)==int(Upper) && i>j))
        continue;
                  
      Index jp = perm ? perm[j] : j;
      Index ip = perm? perm[i] : i;
      
      Index k = count[int(DstUpLo)==int(Lower) ? (std::min)(ip,jp) : (std::max)(ip,jp)]++;
      dest.innerIndexPtr()[k] = int(DstUpLo)==int(Lower) ? (std::max)(ip,jp) : (std::min)(ip,jp);
      
      if(!StorageOrderMatch) std::swap(ip,jp);
      if( ((int(DstUpLo)==int(Lower) && ip<jp) || (int(DstUpLo)==int(Upper) && ip>jp)))
        dest.valuePtr()[k] = numext::conj(it.value());
      else
        dest.valuePtr()[k] = it.value();
    }
  }
}

}

template<typename MatrixType,int UpLo>
class SparseSymmetricPermutationProduct
  : public EigenBase<SparseSymmetricPermutationProduct<MatrixType,UpLo> >
{
  public:
    typedef typename MatrixType::Scalar Scalar;
    typedef typename MatrixType::Index Index;
  protected:
    typedef PermutationMatrix<Dynamic,Dynamic,Index> Perm;
  public:
    typedef Matrix<Index,Dynamic,1> VectorI;
    typedef typename MatrixType::Nested MatrixTypeNested;
    typedef typename internal::remove_all<MatrixTypeNested>::type _MatrixTypeNested;
    
    SparseSymmetricPermutationProduct(const MatrixType& mat, const Perm& perm)
      : m_matrix(mat), m_perm(perm)
    {}
    
    inline Index rows() const { return m_matrix.rows(); }
    inline Index cols() const { return m_matrix.cols(); }
    
    template<typename DestScalar, int Options, typename DstIndex>
    void evalTo(SparseMatrix<DestScalar,Options,DstIndex>& _dest) const
    {
//       internal::permute_symm_to_fullsymm<UpLo>(m_matrix,_dest,m_perm.indices().data());
      SparseMatrix<DestScalar,(Options&RowMajor)==RowMajor ? ColMajor : RowMajor, DstIndex> tmp;
      internal::permute_symm_to_fullsymm<UpLo>(m_matrix,tmp,m_perm.indices().data());
      _dest = tmp;
    }
    
    template<typename DestType,unsigned int DestUpLo> void evalTo(SparseSelfAdjointView<DestType,DestUpLo>& dest) const
    {
      internal::permute_symm_to_symm<UpLo,DestUpLo>(m_matrix,dest.matrix(),m_perm.indices().data());
    }
    
  protected:
    MatrixTypeNested m_matrix;
    const Perm& m_perm;

};

} // end namespace Eigen

#endif // EIGEN_SPARSE_SELFADJOINTVIEW_H
