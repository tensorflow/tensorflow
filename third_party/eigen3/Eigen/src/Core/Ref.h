// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2012 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_REF_H
#define EIGEN_REF_H

namespace Eigen { 

template<typename Derived> class RefBase;
template<typename PlainObjectType, int Options = 0,
         typename StrideType = typename internal::conditional<PlainObjectType::IsVectorAtCompileTime,InnerStride<1>,OuterStride<> >::type > class Ref;

/** \class Ref
  * \ingroup Core_Module
  *
  * \brief A matrix or vector expression mapping an existing expressions
  *
  * \tparam PlainObjectType the equivalent matrix type of the mapped data
  * \tparam Options specifies whether the pointer is \c #Aligned, or \c #Unaligned.
  *                The default is \c #Unaligned.
  * \tparam StrideType optionally specifies strides. By default, Ref implies a contiguous storage along the inner dimension (inner stride==1),
  *                   but accept a variable outer stride (leading dimension).
  *                   This can be overridden by specifying strides.
  *                   The type passed here must be a specialization of the Stride template, see examples below.
  *
  * This class permits to write non template functions taking Eigen's object as parameters while limiting the number of copies.
  * A Ref<> object can represent either a const expression or a l-value:
  * \code
  * // in-out argument:
  * void foo1(Ref<VectorXf> x);
  *
  * // read-only const argument:
  * void foo2(const Ref<const VectorXf>& x);
  * \endcode
  *
  * In the in-out case, the input argument must satisfies the constraints of the actual Ref<> type, otherwise a compilation issue will be triggered.
  * By default, a Ref<VectorXf> can reference any dense vector expression of float having a contiguous memory layout.
  * Likewise, a Ref<MatrixXf> can reference any column major dense matrix expression of float whose column's elements are contiguously stored with
  * the possibility to have a constant space inbetween each column, i.e.: the inner stride mmust be equal to 1, but the outer-stride (or leading dimension),
  * can be greater than the number of rows.
  *
  * In the const case, if the input expression does not match the above requirement, then it is evaluated into a temporary before being passed to the function.
  * Here are some examples:
  * \code
  * MatrixXf A;
  * VectorXf a;
  * foo1(a.head());             // OK
  * foo1(A.col());              // OK
  * foo1(A.row());              // compilation error because here innerstride!=1
  * foo2(A.row());              // The row is copied into a contiguous temporary
  * foo2(2*a);                  // The expression is evaluated into a temporary
  * foo2(A.col().segment(2,4)); // No temporary
  * \endcode
  *
  * The range of inputs that can be referenced without temporary can be enlarged using the last two template parameter.
  * Here is an example accepting an innerstride!=1:
  * \code
  * // in-out argument:
  * void foo3(Ref<VectorXf,0,InnerStride<> > x);
  * foo3(A.row());              // OK
  * \endcode
  * The downside here is that the function foo3 might be significantly slower than foo1 because it won't be able to exploit vectorization, and will involved more
  * expensive address computations even if the input is contiguously stored in memory. To overcome this issue, one might propose to overloads internally calling a
  * template function, e.g.:
  * \code
  * // in the .h:
  * void foo(const Ref<MatrixXf>& A);
  * void foo(const Ref<MatrixXf,0,Stride<> >& A);
  *
  * // in the .cpp:
  * template<typename TypeOfA> void foo_impl(const TypeOfA& A) {
  *     ... // crazy code goes here
  * }
  * void foo(const Ref<MatrixXf>& A) { foo_impl(A); }
  * void foo(const Ref<MatrixXf,0,Stride<> >& A) { foo_impl(A); }
  * \endcode
  *
  *
  * \sa PlainObjectBase::Map(), \ref TopicStorageOrders
  */

namespace internal {

template<typename _PlainObjectType, int _Options, typename _StrideType>
struct traits<Ref<_PlainObjectType, _Options, _StrideType> >
  : public traits<Map<_PlainObjectType, _Options, _StrideType> >
{
  typedef _PlainObjectType PlainObjectType;
  typedef _StrideType StrideType;
  enum {
    Options = _Options,
    Flags = traits<Map<_PlainObjectType, _Options, _StrideType> >::Flags | NestByRefBit
  };

  template<typename Derived> struct match {
    enum {
      HasDirectAccess = internal::has_direct_access<Derived>::ret,
      StorageOrderMatch = PlainObjectType::IsVectorAtCompileTime || Derived::IsVectorAtCompileTime || ((PlainObjectType::Flags&RowMajorBit)==(Derived::Flags&RowMajorBit)),
      InnerStrideMatch = int(StrideType::InnerStrideAtCompileTime)==int(Dynamic)
                      || int(StrideType::InnerStrideAtCompileTime)==int(Derived::InnerStrideAtCompileTime)
                      || (int(StrideType::InnerStrideAtCompileTime)==0 && int(Derived::InnerStrideAtCompileTime)==1),
      OuterStrideMatch = Derived::IsVectorAtCompileTime
                      || int(StrideType::OuterStrideAtCompileTime)==int(Dynamic) || int(StrideType::OuterStrideAtCompileTime)==int(Derived::OuterStrideAtCompileTime),
      AlignmentMatch = (_Options!=Aligned) || ((PlainObjectType::Flags&AlignedBit)==0) || ((traits<Derived>::Flags&AlignedBit)==AlignedBit),
      MatchAtCompileTime = HasDirectAccess && StorageOrderMatch && InnerStrideMatch && OuterStrideMatch && AlignmentMatch
    };
    typedef typename internal::conditional<MatchAtCompileTime,internal::true_type,internal::false_type>::type type;
  };
  
};

template<typename Derived>
struct traits<RefBase<Derived> > : public traits<Derived> {};

}

template<typename Derived> class RefBase
 : public MapBase<Derived>
{
  typedef typename internal::traits<Derived>::PlainObjectType PlainObjectType;
  typedef typename internal::traits<Derived>::StrideType StrideType;

public:

  typedef MapBase<Derived> Base;
  EIGEN_DENSE_PUBLIC_INTERFACE(RefBase)

  inline Index innerStride() const
  {
    return StrideType::InnerStrideAtCompileTime != 0 ? m_stride.inner() : 1;
  }

  inline Index outerStride() const
  {
    return StrideType::OuterStrideAtCompileTime != 0 ? m_stride.outer()
         : IsVectorAtCompileTime ? this->size()
         : int(Flags)&RowMajorBit ? this->cols()
         : this->rows();
  }

  RefBase()
    : Base(0,RowsAtCompileTime==Dynamic?0:RowsAtCompileTime,ColsAtCompileTime==Dynamic?0:ColsAtCompileTime),
      // Stride<> does not allow default ctor for Dynamic strides, so let' initialize it with dummy values:
      m_stride(StrideType::OuterStrideAtCompileTime==Dynamic?0:StrideType::OuterStrideAtCompileTime,
               StrideType::InnerStrideAtCompileTime==Dynamic?0:StrideType::InnerStrideAtCompileTime)
  {}
  
  EIGEN_INHERIT_ASSIGNMENT_OPERATORS(RefBase)

protected:

  typedef Stride<StrideType::OuterStrideAtCompileTime,StrideType::InnerStrideAtCompileTime> StrideBase;

  template<typename Expression>
  void construct(Expression& expr)
  {
    if(PlainObjectType::RowsAtCompileTime==1)
    {
      eigen_assert(expr.rows()==1 || expr.cols()==1);
      ::new (static_cast<Base*>(this)) Base(expr.data(), 1, expr.size());
    }
    else if(PlainObjectType::ColsAtCompileTime==1)
    {
      eigen_assert(expr.rows()==1 || expr.cols()==1);
      ::new (static_cast<Base*>(this)) Base(expr.data(), expr.size(), 1);
    }
    else
      ::new (static_cast<Base*>(this)) Base(expr.data(), expr.rows(), expr.cols());
    
    if(Expression::IsVectorAtCompileTime && (!PlainObjectType::IsVectorAtCompileTime) && ((Expression::Flags&RowMajorBit)!=(PlainObjectType::Flags&RowMajorBit)))
      ::new (&m_stride) StrideBase(expr.innerStride(), StrideType::InnerStrideAtCompileTime==0?0:1);
    else
      ::new (&m_stride) StrideBase(StrideType::OuterStrideAtCompileTime==0?0:expr.outerStride(),
                                   StrideType::InnerStrideAtCompileTime==0?0:expr.innerStride());    
  }

  StrideBase m_stride;
};


template<typename PlainObjectType, int Options, typename StrideType> class Ref
  : public RefBase<Ref<PlainObjectType, Options, StrideType> >
{
    typedef internal::traits<Ref> Traits;
  public:

    typedef RefBase<Ref> Base;
    EIGEN_DENSE_PUBLIC_INTERFACE(Ref)


    #ifndef EIGEN_PARSED_BY_DOXYGEN
    template<typename Derived>
    inline Ref(PlainObjectBase<Derived>& expr,
               typename internal::enable_if<bool(Traits::template match<Derived>::MatchAtCompileTime),Derived>::type* = 0)
    {
      Base::construct(expr);
    }
    template<typename Derived>
    inline Ref(const DenseBase<Derived>& expr,
               typename internal::enable_if<bool(internal::is_lvalue<Derived>::value&&bool(Traits::template match<Derived>::MatchAtCompileTime)),Derived>::type* = 0,
               int = Derived::ThisConstantIsPrivateInPlainObjectBase)
    #else
    template<typename Derived>
    inline Ref(DenseBase<Derived>& expr)
    #endif
    {
      Base::construct(expr.const_cast_derived());
    }

    EIGEN_INHERIT_ASSIGNMENT_OPERATORS(Ref)

};

// this is the const ref version
template<typename TPlainObjectType, int Options, typename StrideType> class Ref<const TPlainObjectType, Options, StrideType>
  : public RefBase<Ref<const TPlainObjectType, Options, StrideType> >
{
    typedef internal::traits<Ref> Traits;
  public:

    typedef RefBase<Ref> Base;
    EIGEN_DENSE_PUBLIC_INTERFACE(Ref)

    template<typename Derived>
    inline Ref(const DenseBase<Derived>& expr)
    {
//      std::cout << match_helper<Derived>::HasDirectAccess << "," << match_helper<Derived>::OuterStrideMatch << "," << match_helper<Derived>::InnerStrideMatch << "\n";
//      std::cout << int(StrideType::OuterStrideAtCompileTime) << " - " << int(Derived::OuterStrideAtCompileTime) << "\n";
//      std::cout << int(StrideType::InnerStrideAtCompileTime) << " - " << int(Derived::InnerStrideAtCompileTime) << "\n";
      construct(expr.derived(), typename Traits::template match<Derived>::type());
    }

  protected:

    template<typename Expression>
    void construct(const Expression& expr,internal::true_type)
    {
      Base::construct(expr);
    }

    template<typename Expression>
    void construct(const Expression& expr, internal::false_type)
    {
      m_object.lazyAssign(expr);
      Base::construct(m_object);
    }

  protected:
    TPlainObjectType m_object;
};

} // end namespace Eigen

#endif // EIGEN_REF_H
