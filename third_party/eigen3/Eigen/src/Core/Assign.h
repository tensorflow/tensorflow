// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2007 Michael Olbrich <michael.olbrich@gmx.net>
// Copyright (C) 2006-2010 Benoit Jacob <jacob.benoit.1@gmail.com>
// Copyright (C) 2008 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_ASSIGN_H
#define EIGEN_ASSIGN_H

namespace Eigen {

namespace internal {

/***************************************************************************
* Part 1 : the logic deciding a strategy for traversal and unrolling       *
***************************************************************************/

template <typename Derived, typename OtherDerived>
struct assign_traits
{
public:
  enum {
    DstIsAligned = Derived::Flags & AlignedBit,
    DstHasDirectAccess = Derived::Flags & DirectAccessBit,
    SrcIsAligned = OtherDerived::Flags & AlignedBit,
    JointAlignment = bool(DstIsAligned) && bool(SrcIsAligned) ? Aligned : Unaligned
  };

private:
  enum {
    InnerSize = int(Derived::IsVectorAtCompileTime) ? int(Derived::SizeAtCompileTime)
              : int(Derived::Flags)&RowMajorBit ? int(Derived::ColsAtCompileTime)
              : int(Derived::RowsAtCompileTime),
    InnerMaxSize = int(Derived::IsVectorAtCompileTime) ? int(Derived::MaxSizeAtCompileTime)
              : int(Derived::Flags)&RowMajorBit ? int(Derived::MaxColsAtCompileTime)
              : int(Derived::MaxRowsAtCompileTime),
    MaxSizeAtCompileTime = Derived::SizeAtCompileTime,
    PacketSize = packet_traits<typename Derived::Scalar>::size
  };

  enum {
    StorageOrdersAgree = (int(Derived::IsRowMajor) == int(OtherDerived::IsRowMajor)),
    MightVectorize = StorageOrdersAgree
                  && (int(Derived::Flags) & int(OtherDerived::Flags) & ActualPacketAccessBit),
    MayInnerVectorize  = MightVectorize && int(InnerSize)!=Dynamic && int(InnerSize)%int(PacketSize)==0
                       && int(DstIsAligned) && int(SrcIsAligned),
    MayLinearize = StorageOrdersAgree && (int(Derived::Flags) & int(OtherDerived::Flags) & LinearAccessBit),
    MayLinearVectorize = MightVectorize && MayLinearize && DstHasDirectAccess
                       && (DstIsAligned || MaxSizeAtCompileTime == Dynamic),
      /* If the destination isn't aligned, we have to do runtime checks and we don't unroll,
         so it's only good for large enough sizes. */
    MaySliceVectorize  = MightVectorize && DstHasDirectAccess
                       && (int(InnerMaxSize)==Dynamic || int(InnerMaxSize)>=3*PacketSize)
      /* slice vectorization can be slow, so we only want it if the slices are big, which is
         indicated by InnerMaxSize rather than InnerSize, think of the case of a dynamic block
         in a fixed-size matrix */
  };

public:
  enum {
    Traversal = int(MayInnerVectorize)  ? int(InnerVectorizedTraversal)
              : int(MayLinearVectorize) ? int(LinearVectorizedTraversal)
              : int(MaySliceVectorize)  ? int(SliceVectorizedTraversal)
              : int(MayLinearize)       ? int(LinearTraversal)
                                        : int(DefaultTraversal),
    Vectorized = int(Traversal) == InnerVectorizedTraversal
              || int(Traversal) == LinearVectorizedTraversal
              || int(Traversal) == SliceVectorizedTraversal
  };

private:
  enum {
    UnrollingLimit      = EIGEN_UNROLLING_LIMIT * (Vectorized ? int(PacketSize) : 1),
    MayUnrollCompletely = int(Derived::SizeAtCompileTime) != Dynamic
                       && int(OtherDerived::CoeffReadCost) != Dynamic
                       && int(Derived::SizeAtCompileTime) * int(OtherDerived::CoeffReadCost) <= int(UnrollingLimit),
    MayUnrollInner      = int(InnerSize) != Dynamic
                       && int(OtherDerived::CoeffReadCost) != Dynamic
                       && int(InnerSize) * int(OtherDerived::CoeffReadCost) <= int(UnrollingLimit)
  };

public:
  enum {
    Unrolling = (int(Traversal) == int(InnerVectorizedTraversal) || int(Traversal) == int(DefaultTraversal))
                ? (
                    int(MayUnrollCompletely) ? int(CompleteUnrolling)
                  : int(MayUnrollInner)      ? int(InnerUnrolling)
                                             : int(NoUnrolling)
                  )
              : int(Traversal) == int(LinearVectorizedTraversal)
                ? ( bool(MayUnrollCompletely) && bool(DstIsAligned) ? int(CompleteUnrolling) : int(NoUnrolling) )
              : int(Traversal) == int(LinearTraversal)
                ? ( bool(MayUnrollCompletely) ? int(CompleteUnrolling) : int(NoUnrolling) )
              : int(NoUnrolling)
  };

#ifdef EIGEN_DEBUG_ASSIGN
  static void debug()
  {
    EIGEN_DEBUG_VAR(DstIsAligned)
    EIGEN_DEBUG_VAR(SrcIsAligned)
    EIGEN_DEBUG_VAR(JointAlignment)
    EIGEN_DEBUG_VAR(Derived::SizeAtCompileTime)
    EIGEN_DEBUG_VAR(OtherDerived::CoeffReadCost)
    EIGEN_DEBUG_VAR(InnerSize)
    EIGEN_DEBUG_VAR(InnerMaxSize)
    EIGEN_DEBUG_VAR(PacketSize)
    EIGEN_DEBUG_VAR(StorageOrdersAgree)
    EIGEN_DEBUG_VAR(MightVectorize)
    EIGEN_DEBUG_VAR(MayLinearize)
    EIGEN_DEBUG_VAR(MayInnerVectorize)
    EIGEN_DEBUG_VAR(MayLinearVectorize)
    EIGEN_DEBUG_VAR(MaySliceVectorize)
    EIGEN_DEBUG_VAR(Traversal)
    EIGEN_DEBUG_VAR(UnrollingLimit)
    EIGEN_DEBUG_VAR(MayUnrollCompletely)
    EIGEN_DEBUG_VAR(MayUnrollInner)
    EIGEN_DEBUG_VAR(Unrolling)
  }
#endif
};

/***************************************************************************
* Part 2 : meta-unrollers
***************************************************************************/

/************************
*** Default traversal ***
************************/

template<typename Derived1, typename Derived2, int Index, int Stop>
struct assign_DefaultTraversal_CompleteUnrolling
{
  enum {
    outer = Index / Derived1::InnerSizeAtCompileTime,
    inner = Index % Derived1::InnerSizeAtCompileTime
  };

  EIGEN_DEVICE_FUNC
  static EIGEN_STRONG_INLINE void run(Derived1 &dst, const Derived2 &src)
  {
    dst.copyCoeffByOuterInner(outer, inner, src);
    assign_DefaultTraversal_CompleteUnrolling<Derived1, Derived2, Index+1, Stop>::run(dst, src);
  }
};

template<typename Derived1, typename Derived2, int Stop>
struct assign_DefaultTraversal_CompleteUnrolling<Derived1, Derived2, Stop, Stop>
{
  EIGEN_DEVICE_FUNC 
  static EIGEN_STRONG_INLINE void run(Derived1 &, const Derived2 &) {}
};

template<typename Derived1, typename Derived2, int Index, int Stop>
struct assign_DefaultTraversal_InnerUnrolling
{
  EIGEN_DEVICE_FUNC 
  static EIGEN_STRONG_INLINE void run(Derived1 &dst, const Derived2 &src, typename Derived1::Index outer)
  {
    dst.copyCoeffByOuterInner(outer, Index, src);
    assign_DefaultTraversal_InnerUnrolling<Derived1, Derived2, Index+1, Stop>::run(dst, src, outer);
  }
};

template<typename Derived1, typename Derived2, int Stop>
struct assign_DefaultTraversal_InnerUnrolling<Derived1, Derived2, Stop, Stop>
{
  EIGEN_DEVICE_FUNC 
  static EIGEN_STRONG_INLINE void run(Derived1 &, const Derived2 &, typename Derived1::Index) {}
};

/***********************
*** Linear traversal ***
***********************/

template<typename Derived1, typename Derived2, int Index, int Stop>
struct assign_LinearTraversal_CompleteUnrolling
{
  EIGEN_DEVICE_FUNC 
  static EIGEN_STRONG_INLINE void run(Derived1 &dst, const Derived2 &src)
  {
    dst.copyCoeff(Index, src);
    assign_LinearTraversal_CompleteUnrolling<Derived1, Derived2, Index+1, Stop>::run(dst, src);
  }
};

template<typename Derived1, typename Derived2, int Stop>
struct assign_LinearTraversal_CompleteUnrolling<Derived1, Derived2, Stop, Stop>
{
  EIGEN_DEVICE_FUNC 
  static EIGEN_STRONG_INLINE void run(Derived1 &, const Derived2 &) {}
};

/**************************
*** Inner vectorization ***
**************************/

template<typename Derived1, typename Derived2, int Index, int Stop>
struct assign_innervec_CompleteUnrolling
{
  enum {
    outer = Index / Derived1::InnerSizeAtCompileTime,
    inner = Index % Derived1::InnerSizeAtCompileTime,
    JointAlignment = assign_traits<Derived1,Derived2>::JointAlignment
  };

  static EIGEN_STRONG_INLINE void run(Derived1 &dst, const Derived2 &src)
  {
    dst.template copyPacketByOuterInner<Derived2, Aligned, JointAlignment>(outer, inner, src);
    assign_innervec_CompleteUnrolling<Derived1, Derived2,
      Index+packet_traits<typename Derived1::Scalar>::size, Stop>::run(dst, src);
  }
};

template<typename Derived1, typename Derived2, int Stop>
struct assign_innervec_CompleteUnrolling<Derived1, Derived2, Stop, Stop>
{
  static EIGEN_STRONG_INLINE void run(Derived1 &, const Derived2 &) {}
};

template<typename Derived1, typename Derived2, int Index, int Stop>
struct assign_innervec_InnerUnrolling
{
  static EIGEN_STRONG_INLINE void run(Derived1 &dst, const Derived2 &src, typename Derived1::Index outer)
  {
    dst.template copyPacketByOuterInner<Derived2, Aligned, Aligned>(outer, Index, src);
    assign_innervec_InnerUnrolling<Derived1, Derived2,
      Index+packet_traits<typename Derived1::Scalar>::size, Stop>::run(dst, src, outer);
  }
};

template<typename Derived1, typename Derived2, int Stop>
struct assign_innervec_InnerUnrolling<Derived1, Derived2, Stop, Stop>
{
  static EIGEN_STRONG_INLINE void run(Derived1 &, const Derived2 &, typename Derived1::Index) {}
};

/***************************************************************************
* Part 3 : implementation of all cases
***************************************************************************/

template<typename Derived1, typename Derived2,
         int Traversal = assign_traits<Derived1, Derived2>::Traversal,
         int Unrolling = assign_traits<Derived1, Derived2>::Unrolling,
         int Version = Specialized>
struct assign_impl;

/************************
*** Default traversal ***
************************/

template<typename Derived1, typename Derived2, int Unrolling, int Version>
struct assign_impl<Derived1, Derived2, InvalidTraversal, Unrolling, Version>
{
  EIGEN_DEVICE_FUNC
  static inline void run(Derived1 &, const Derived2 &) { }
};

template<typename Derived1, typename Derived2, int Version>
struct assign_impl<Derived1, Derived2, DefaultTraversal, NoUnrolling, Version>
{
  typedef typename Derived1::Index Index;
  EIGEN_DEVICE_FUNC 
  static inline void run(Derived1 &dst, const Derived2 &src)
  {
    const Index innerSize = dst.innerSize();
    const Index outerSize = dst.outerSize();
    for(Index outer = 0; outer < outerSize; ++outer)
      for(Index inner = 0; inner < innerSize; ++inner)
        dst.copyCoeffByOuterInner(outer, inner, src);
  }
};

template<typename Derived1, typename Derived2, int Version>
struct assign_impl<Derived1, Derived2, DefaultTraversal, CompleteUnrolling, Version>
{
  EIGEN_DEVICE_FUNC
  static EIGEN_STRONG_INLINE void run(Derived1 &dst, const Derived2 &src)
  {
    assign_DefaultTraversal_CompleteUnrolling<Derived1, Derived2, 0, Derived1::SizeAtCompileTime>
      ::run(dst, src);
  }
};

template<typename Derived1, typename Derived2, int Version>
struct assign_impl<Derived1, Derived2, DefaultTraversal, InnerUnrolling, Version>
{
  typedef typename Derived1::Index Index;
  EIGEN_DEVICE_FUNC 
  static EIGEN_STRONG_INLINE void run(Derived1 &dst, const Derived2 &src)
  {
    const Index outerSize = dst.outerSize();
    for(Index outer = 0; outer < outerSize; ++outer)
      assign_DefaultTraversal_InnerUnrolling<Derived1, Derived2, 0, Derived1::InnerSizeAtCompileTime>
        ::run(dst, src, outer);
  }
};

/***********************
*** Linear traversal ***
***********************/

template<typename Derived1, typename Derived2, int Version>
struct assign_impl<Derived1, Derived2, LinearTraversal, NoUnrolling, Version>
{
  typedef typename Derived1::Index Index;
  EIGEN_DEVICE_FUNC
  static inline void run(Derived1 &dst, const Derived2 &src)
  {
    const Index size = dst.size();
    for(Index i = 0; i < size; ++i)
      dst.copyCoeff(i, src);
  }
};

template<typename Derived1, typename Derived2, int Version>
struct assign_impl<Derived1, Derived2, LinearTraversal, CompleteUnrolling, Version>
{
  EIGEN_DEVICE_FUNC
  static EIGEN_STRONG_INLINE void run(Derived1 &dst, const Derived2 &src)
  {
    assign_LinearTraversal_CompleteUnrolling<Derived1, Derived2, 0, Derived1::SizeAtCompileTime>
      ::run(dst, src);
  }
};

/**************************
*** Inner vectorization ***
**************************/

template<typename Derived1, typename Derived2, int Version>
struct assign_impl<Derived1, Derived2, InnerVectorizedTraversal, NoUnrolling, Version>
{
  typedef typename Derived1::Index Index;
  static inline void run(Derived1 &dst, const Derived2 &src)
  {
    const Index innerSize = dst.innerSize();
    const Index outerSize = dst.outerSize();
    const Index packetSize = packet_traits<typename Derived1::Scalar>::size;
    for(Index outer = 0; outer < outerSize; ++outer)
      for(Index inner = 0; inner < innerSize; inner+=packetSize)
        dst.template copyPacketByOuterInner<Derived2, Aligned, Aligned>(outer, inner, src);
  }
};

template<typename Derived1, typename Derived2, int Version>
struct assign_impl<Derived1, Derived2, InnerVectorizedTraversal, CompleteUnrolling, Version>
{
  static EIGEN_STRONG_INLINE void run(Derived1 &dst, const Derived2 &src)
  {
    assign_innervec_CompleteUnrolling<Derived1, Derived2, 0, Derived1::SizeAtCompileTime>
      ::run(dst, src);
  }
};

template<typename Derived1, typename Derived2, int Version>
struct assign_impl<Derived1, Derived2, InnerVectorizedTraversal, InnerUnrolling, Version>
{
  typedef typename Derived1::Index Index;
  static EIGEN_STRONG_INLINE void run(Derived1 &dst, const Derived2 &src)
  {
    const Index outerSize = dst.outerSize();
    for(Index outer = 0; outer < outerSize; ++outer)
      assign_innervec_InnerUnrolling<Derived1, Derived2, 0, Derived1::InnerSizeAtCompileTime>
        ::run(dst, src, outer);
  }
};

/***************************
*** Linear vectorization ***
***************************/

template <bool IsAligned = false>
struct unaligned_assign_impl
{
  template <typename Derived, typename OtherDerived>
  static EIGEN_STRONG_INLINE void run(const Derived&, OtherDerived&, typename Derived::Index, typename Derived::Index) {}
};

template <>
struct unaligned_assign_impl<false>
{
  // MSVC must not inline this functions. If it does, it fails to optimize the
  // packet access path.
#ifdef _MSC_VER
  template <typename Derived, typename OtherDerived>
  static EIGEN_DONT_INLINE void run(const Derived& src, OtherDerived& dst, typename Derived::Index start, typename Derived::Index end)
#else
  template <typename Derived, typename OtherDerived>
  static EIGEN_STRONG_INLINE void run(const Derived& src, OtherDerived& dst, typename Derived::Index start, typename Derived::Index end)
#endif
  {
    for (typename Derived::Index index = start; index < end; ++index)
      dst.copyCoeff(index, src);
  }
};

template<typename Derived1, typename Derived2, int Version>
struct assign_impl<Derived1, Derived2, LinearVectorizedTraversal, NoUnrolling, Version>
{
  typedef typename Derived1::Index Index;
  static EIGEN_STRONG_INLINE void run(Derived1 &dst, const Derived2 &src)
  {
    const Index size = dst.size();
    typedef packet_traits<typename Derived1::Scalar> PacketTraits;
    enum {
      packetSize = PacketTraits::size,
      dstAlignment = PacketTraits::AlignedOnScalar ? Aligned : int(assign_traits<Derived1,Derived2>::DstIsAligned) ,
      srcAlignment = assign_traits<Derived1,Derived2>::JointAlignment
    };
    const Index alignedStart = assign_traits<Derived1,Derived2>::DstIsAligned ? 0
                             : internal::first_aligned(&dst.coeffRef(0), size);
    const Index alignedEnd = alignedStart + ((size-alignedStart)/packetSize)*packetSize;

    unaligned_assign_impl<assign_traits<Derived1,Derived2>::DstIsAligned!=0>::run(src,dst,0,alignedStart);

    for(Index index = alignedStart; index < alignedEnd; index += packetSize)
    {
      dst.template copyPacket<Derived2, dstAlignment, srcAlignment>(index, src);
    }

    unaligned_assign_impl<>::run(src,dst,alignedEnd,size);
  }
};

template<typename Derived1, typename Derived2, int Version>
struct assign_impl<Derived1, Derived2, LinearVectorizedTraversal, CompleteUnrolling, Version>
{
  typedef typename Derived1::Index Index;
  static EIGEN_STRONG_INLINE void run(Derived1 &dst, const Derived2 &src)
  {
    enum { size = Derived1::SizeAtCompileTime,
           packetSize = packet_traits<typename Derived1::Scalar>::size,
           alignedSize = (size/packetSize)*packetSize };

    assign_innervec_CompleteUnrolling<Derived1, Derived2, 0, alignedSize>::run(dst, src);
    assign_DefaultTraversal_CompleteUnrolling<Derived1, Derived2, alignedSize, size>::run(dst, src);
  }
};

/**************************
*** Slice vectorization ***
***************************/

template<typename Derived1, typename Derived2, int Version>
struct assign_impl<Derived1, Derived2, SliceVectorizedTraversal, NoUnrolling, Version>
{
  typedef typename Derived1::Index Index;
  static inline void run(Derived1 &dst, const Derived2 &src)
  {
    typedef packet_traits<typename Derived1::Scalar> PacketTraits;
    enum {
      packetSize = PacketTraits::size,
      alignable = PacketTraits::AlignedOnScalar,
      dstAlignment = alignable ? Aligned : int(assign_traits<Derived1,Derived2>::DstIsAligned) ,
      srcAlignment = assign_traits<Derived1,Derived2>::JointAlignment
    };
    const Index packetAlignedMask = packetSize - 1;
    const Index innerSize = dst.innerSize();
    const Index outerSize = dst.outerSize();
    const Index alignedStep = alignable ? (packetSize - dst.outerStride() % packetSize) & packetAlignedMask : 0;
    Index alignedStart = ((!alignable) || assign_traits<Derived1,Derived2>::DstIsAligned) ? 0
                       : internal::first_aligned(&dst.coeffRef(0,0), innerSize);

    for(Index outer = 0; outer < outerSize; ++outer)
    {
      const Index alignedEnd = alignedStart + ((innerSize-alignedStart) & ~packetAlignedMask);
      // do the non-vectorizable part of the assignment
      for(Index inner = 0; inner<alignedStart ; ++inner)
        dst.copyCoeffByOuterInner(outer, inner, src);

      // do the vectorizable part of the assignment
      for(Index inner = alignedStart; inner<alignedEnd; inner+=packetSize)
        dst.template copyPacketByOuterInner<Derived2, dstAlignment, Unaligned>(outer, inner, src);

      // do the non-vectorizable part of the assignment
      for(Index inner = alignedEnd; inner<innerSize ; ++inner)
        dst.copyCoeffByOuterInner(outer, inner, src);

      alignedStart = std::min<Index>((alignedStart+alignedStep)%packetSize, innerSize);
    }
  }
};

} // end namespace internal

/***************************************************************************
* Part 4 : implementation of DenseBase methods
***************************************************************************/

template<typename Derived>
template<typename OtherDerived>
EIGEN_STRONG_INLINE Derived& DenseBase<Derived>
  ::lazyAssign(const DenseBase<OtherDerived>& other)
{
  enum{
    SameType = internal::is_same<typename Derived::Scalar,typename OtherDerived::Scalar>::value
  };

  EIGEN_STATIC_ASSERT_LVALUE(Derived)
  EIGEN_STATIC_ASSERT_SAME_MATRIX_SIZE(Derived,OtherDerived)
  EIGEN_STATIC_ASSERT(SameType,YOU_MIXED_DIFFERENT_NUMERIC_TYPES__YOU_NEED_TO_USE_THE_CAST_METHOD_OF_MATRIXBASE_TO_CAST_NUMERIC_TYPES_EXPLICITLY)

#ifdef EIGEN_TEST_EVALUATORS
  
#ifdef EIGEN_DEBUG_ASSIGN
  internal::copy_using_evaluator_traits<Derived, OtherDerived>::debug();
#endif
  eigen_assert(rows() == other.rows() && cols() == other.cols());
  internal::call_dense_assignment_loop(derived(),other.derived());
  
#else // EIGEN_TEST_EVALUATORS

#ifdef EIGEN_DEBUG_ASSIGN
  internal::assign_traits<Derived, OtherDerived>::debug();
#endif
  eigen_assert(rows() == other.rows() && cols() == other.cols());
  internal::assign_impl<Derived, OtherDerived, int(SameType) ? int(internal::assign_traits<Derived, OtherDerived>::Traversal)
                                                             : int(InvalidTraversal)>::run(derived(),other.derived());
  
#endif // EIGEN_TEST_EVALUATORS
  
#ifndef EIGEN_NO_DEBUG
  checkTransposeAliasing(other.derived());
#endif
  return derived();
}

namespace internal {

template<typename Derived, typename OtherDerived,
         bool EvalBeforeAssigning = (int(internal::traits<OtherDerived>::Flags) & EvalBeforeAssigningBit) != 0,
         bool NeedToTranspose = ((int(Derived::RowsAtCompileTime) == 1 && int(OtherDerived::ColsAtCompileTime) == 1)
                              |   // FIXME | instead of || to please GCC 4.4.0 stupid warning "suggest parentheses around &&".
                                  // revert to || as soon as not needed anymore.
                                  (int(Derived::ColsAtCompileTime) == 1 && int(OtherDerived::RowsAtCompileTime) == 1))
                              && int(Derived::SizeAtCompileTime) != 1>
struct assign_selector;

template<typename Derived, typename OtherDerived>
struct assign_selector<Derived,OtherDerived,false,false> {
  EIGEN_DEVICE_FUNC
  static EIGEN_STRONG_INLINE Derived& run(Derived& dst, const OtherDerived& other) { return dst.lazyAssign(other.derived()); }
  template<typename ActualDerived, typename ActualOtherDerived>
  EIGEN_DEVICE_FUNC
  static EIGEN_STRONG_INLINE Derived& evalTo(ActualDerived& dst, const ActualOtherDerived& other) { other.evalTo(dst); return dst; }
};
template<typename Derived, typename OtherDerived>
struct assign_selector<Derived,OtherDerived,true,false> {
  EIGEN_DEVICE_FUNC
  static EIGEN_STRONG_INLINE Derived& run(Derived& dst, const OtherDerived& other) { return dst.lazyAssign(other.eval()); }
};
template<typename Derived, typename OtherDerived>
struct assign_selector<Derived,OtherDerived,false,true> {
  EIGEN_DEVICE_FUNC
  static EIGEN_STRONG_INLINE Derived& run(Derived& dst, const OtherDerived& other) { return dst.lazyAssign(other.transpose()); }
  template<typename ActualDerived, typename ActualOtherDerived>
  EIGEN_DEVICE_FUNC
  static EIGEN_STRONG_INLINE Derived& evalTo(ActualDerived& dst, const ActualOtherDerived& other) { Transpose<ActualDerived> dstTrans(dst); other.evalTo(dstTrans); return dst; }
};
template<typename Derived, typename OtherDerived>
struct assign_selector<Derived,OtherDerived,true,true> {
  EIGEN_DEVICE_FUNC
  static EIGEN_STRONG_INLINE Derived& run(Derived& dst, const OtherDerived& other) { return dst.lazyAssign(other.transpose().eval()); }
};

} // end namespace internal

template<typename Derived>
template<typename OtherDerived>
EIGEN_DEVICE_FUNC
EIGEN_STRONG_INLINE Derived& DenseBase<Derived>::operator=(const DenseBase<OtherDerived>& other)
{
  return internal::assign_selector<Derived,OtherDerived>::run(derived(), other.derived());
}

template<typename Derived>
EIGEN_DEVICE_FUNC
EIGEN_STRONG_INLINE Derived& DenseBase<Derived>::operator=(const DenseBase& other)
{
  return internal::assign_selector<Derived,Derived>::run(derived(), other.derived());
}

template<typename Derived>
EIGEN_DEVICE_FUNC
EIGEN_STRONG_INLINE Derived& MatrixBase<Derived>::operator=(const MatrixBase& other)
{
  return internal::assign_selector<Derived,Derived>::run(derived(), other.derived());
}

template<typename Derived>
template <typename OtherDerived>
EIGEN_DEVICE_FUNC
EIGEN_STRONG_INLINE Derived& MatrixBase<Derived>::operator=(const DenseBase<OtherDerived>& other)
{
  return internal::assign_selector<Derived,OtherDerived>::run(derived(), other.derived());
}

template<typename Derived>
template <typename OtherDerived>
EIGEN_DEVICE_FUNC
EIGEN_STRONG_INLINE Derived& MatrixBase<Derived>::operator=(const EigenBase<OtherDerived>& other)
{
  return internal::assign_selector<Derived,OtherDerived,false>::evalTo(derived(), other.derived());
}

template<typename Derived>
template<typename OtherDerived>
EIGEN_DEVICE_FUNC
EIGEN_STRONG_INLINE Derived& MatrixBase<Derived>::operator=(const ReturnByValue<OtherDerived>& other)
{
  return internal::assign_selector<Derived,OtherDerived,false>::evalTo(derived(), other.derived());
}

} // end namespace Eigen

#endif // EIGEN_ASSIGN_H
