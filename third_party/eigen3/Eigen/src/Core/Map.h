// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2007-2010 Benoit Jacob <jacob.benoit.1@gmail.com>
// Copyright (C) 2008 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_MAP_H
#define EIGEN_MAP_H

namespace Eigen { 

/** \class Map
  * \ingroup Core_Module
  *
  * \brief A matrix or vector expression mapping an existing array of data.
  *
  * \tparam PlainObjectType the equivalent matrix type of the mapped data
  * \tparam MapOptions specifies whether the pointer is \c #Aligned, or \c #Unaligned.
  *                The default is \c #Unaligned.
  * \tparam StrideType optionally specifies strides. By default, Map assumes the memory layout
  *                   of an ordinary, contiguous array. This can be overridden by specifying strides.
  *                   The type passed here must be a specialization of the Stride template, see examples below.
  *
  * This class represents a matrix or vector expression mapping an existing array of data.
  * It can be used to let Eigen interface without any overhead with non-Eigen data structures,
  * such as plain C arrays or structures from other libraries. By default, it assumes that the
  * data is laid out contiguously in memory. You can however override this by explicitly specifying
  * inner and outer strides.
  *
  * Here's an example of simply mapping a contiguous array as a \ref TopicStorageOrders "column-major" matrix:
  * \include Map_simple.cpp
  * Output: \verbinclude Map_simple.out
  *
  * If you need to map non-contiguous arrays, you can do so by specifying strides:
  *
  * Here's an example of mapping an array as a vector, specifying an inner stride, that is, the pointer
  * increment between two consecutive coefficients. Here, we're specifying the inner stride as a compile-time
  * fixed value.
  * \include Map_inner_stride.cpp
  * Output: \verbinclude Map_inner_stride.out
  *
  * Here's an example of mapping an array while specifying an outer stride. Here, since we're mapping
  * as a column-major matrix, 'outer stride' means the pointer increment between two consecutive columns.
  * Here, we're specifying the outer stride as a runtime parameter. Note that here \c OuterStride<> is
  * a short version of \c OuterStride<Dynamic> because the default template parameter of OuterStride
  * is  \c Dynamic
  * \include Map_outer_stride.cpp
  * Output: \verbinclude Map_outer_stride.out
  *
  * For more details and for an example of specifying both an inner and an outer stride, see class Stride.
  *
  * \b Tip: to change the array of data mapped by a Map object, you can use the C++
  * placement new syntax:
  *
  * Example: \include Map_placement_new.cpp
  * Output: \verbinclude Map_placement_new.out
  *
  * This class is the return type of PlainObjectBase::Map() but can also be used directly.
  *
  * \sa PlainObjectBase::Map(), \ref TopicStorageOrders
  */

namespace internal {
template<typename PlainObjectType, int MapOptions, typename StrideType>
struct traits<Map<PlainObjectType, MapOptions, StrideType> >
  : public traits<PlainObjectType>
{
  typedef traits<PlainObjectType> TraitsBase;
  typedef typename PlainObjectType::Index Index;
  typedef typename PlainObjectType::Scalar Scalar;
  enum {
    InnerStrideAtCompileTime = StrideType::InnerStrideAtCompileTime == 0
                             ? int(PlainObjectType::InnerStrideAtCompileTime)
                             : int(StrideType::InnerStrideAtCompileTime),
    OuterStrideAtCompileTime = StrideType::OuterStrideAtCompileTime == 0
                             ? int(PlainObjectType::OuterStrideAtCompileTime)
                             : int(StrideType::OuterStrideAtCompileTime),
    HasNoInnerStride = InnerStrideAtCompileTime == 1,
    HasNoOuterStride = StrideType::OuterStrideAtCompileTime == 0,
    HasNoStride = HasNoInnerStride && HasNoOuterStride,
    IsAligned = bool(EIGEN_ALIGN) && ((int(MapOptions)&Aligned)==Aligned),
    IsDynamicSize = PlainObjectType::SizeAtCompileTime==Dynamic,
    KeepsPacketAccess = bool(HasNoInnerStride)
                        && ( bool(IsDynamicSize)
                           || HasNoOuterStride
                           || ( OuterStrideAtCompileTime!=Dynamic
                           && ((static_cast<int>(sizeof(Scalar))*OuterStrideAtCompileTime)%EIGEN_ALIGN_BYTES)==0 ) ),
    Flags0 = TraitsBase::Flags & (~NestByRefBit),
    Flags1 = IsAligned ? (int(Flags0) | AlignedBit) : (int(Flags0) & ~AlignedBit),
    Flags2 = (bool(HasNoStride) || bool(PlainObjectType::IsVectorAtCompileTime))
           ? int(Flags1) : int(Flags1 & ~LinearAccessBit),
    Flags3 = is_lvalue<PlainObjectType>::value ? int(Flags2) : (int(Flags2) & ~LvalueBit),
    Flags = KeepsPacketAccess ? int(Flags3) : (int(Flags3) & ~PacketAccessBit)
  };
private:
  enum { Options }; // Expressions don't have Options
};
}

template<typename PlainObjectType, int MapOptions, typename StrideType> class Map
  : public MapBase<Map<PlainObjectType, MapOptions, StrideType> >
{
  public:

    typedef MapBase<Map> Base;
    EIGEN_DENSE_PUBLIC_INTERFACE(Map)

    typedef typename Base::PointerType PointerType;
#if EIGEN2_SUPPORT_STAGE <= STAGE30_FULL_EIGEN3_API
    typedef const Scalar* PointerArgType;
    inline PointerType cast_to_pointer_type(PointerArgType ptr) { return const_cast<PointerType>(ptr); }
#else
    typedef PointerType PointerArgType;
    EIGEN_DEVICE_FUNC
    inline PointerType cast_to_pointer_type(PointerArgType ptr) { return ptr; }
#endif

    EIGEN_DEVICE_FUNC
    inline Index innerStride() const
    {
      return StrideType::InnerStrideAtCompileTime != 0 ? m_stride.inner() : 1;
    }

    EIGEN_DEVICE_FUNC
    inline Index outerStride() const
    {
      return StrideType::OuterStrideAtCompileTime != 0 ? m_stride.outer()
           : IsVectorAtCompileTime ? this->size()
           : int(Flags)&RowMajorBit ? this->cols()
           : this->rows();
    }

    /** Constructor in the fixed-size case.
      *
      * \param dataPtr pointer to the array to map
      * \param a_stride optional Stride object, passing the strides.
      */
    EIGEN_DEVICE_FUNC
    inline Map(PointerArgType dataPtr, const StrideType& a_stride = StrideType())
      : Base(cast_to_pointer_type(dataPtr)), m_stride(a_stride)
    {
      PlainObjectType::Base::_check_template_params();
    }

    /** Constructor in the dynamic-size vector case.
      *
      * \param dataPtr pointer to the array to map
      * \param a_size the size of the vector expression
      * \param a_stride optional Stride object, passing the strides.
      */
    EIGEN_DEVICE_FUNC
    inline Map(PointerArgType dataPtr, Index a_size, const StrideType& a_stride = StrideType())
      : Base(cast_to_pointer_type(dataPtr), a_size), m_stride(a_stride)
    {
      PlainObjectType::Base::_check_template_params();
    }

    /** Constructor in the dynamic-size matrix case.
      *
      * \param dataPtr pointer to the array to map
      * \param nbRows the number of rows of the matrix expression
      * \param nbCols the number of columns of the matrix expression
      * \param a_stride optional Stride object, passing the strides.
      */
    EIGEN_DEVICE_FUNC
    inline Map(PointerArgType dataPtr, Index nbRows, Index nbCols, const StrideType& a_stride = StrideType())
      : Base(cast_to_pointer_type(dataPtr), nbRows, nbCols), m_stride(a_stride)
    {
      PlainObjectType::Base::_check_template_params();
    }

    EIGEN_INHERIT_ASSIGNMENT_OPERATORS(Map)

  protected:
    StrideType m_stride;
};


} // end namespace Eigen

#endif // EIGEN_MAP_H
