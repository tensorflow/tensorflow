// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008-2009 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_GENERAL_MATRIX_VECTOR_H
#define EIGEN_GENERAL_MATRIX_VECTOR_H

namespace Eigen {

namespace internal {

/* Optimized col-major matrix * vector product:
 * This algorithm processes 4 columns at onces that allows to both reduce
 * the number of load/stores of the result by a factor 4 and to reduce
 * the instruction dependency. Moreover, we know that all bands have the
 * same alignment pattern.
 *
 * Mixing type logic: C += alpha * A * B
 *  |  A  |  B  |alpha| comments
 *  |real |cplx |cplx | no vectorization
 *  |real |cplx |real | alpha is converted to a cplx when calling the run function, no vectorization
 *  |cplx |real |cplx | invalid, the caller has to do tmp: = A * B; C += alpha*tmp
 *  |cplx |real |real | optimal case, vectorization possible via real-cplx mul
 *
 * Accesses to the matrix coefficients follow the following logic:
 *
 * - if all columns have the same alignment then
 *   - if the columns have the same alignment as the result vector, then easy! (-> AllAligned case)
 *   - otherwise perform unaligned loads only (-> NoneAligned case)
 * - otherwise
 *   - if even columns have the same alignment then
 *     // odd columns are guaranteed to have the same alignment too
 *     - if even or odd columns have the same alignment as the result, then
 *       // for a register size of 2 scalars, this is guarantee to be the case (e.g., SSE with double)
 *       - perform half aligned and half unaligned loads (-> EvenAligned case)
 *     - otherwise perform unaligned loads only (-> NoneAligned case)
 *   - otherwise, if the register size is 4 scalars (e.g., SSE with float) then
 *     - one over 4 consecutive columns is guaranteed to be aligned with the result vector,
 *       perform simple aligned loads for this column and aligned loads plus re-alignment for the other. (-> FirstAligned case)
 *       // this re-alignment is done by the palign function implemented for SSE in Eigen/src/Core/arch/SSE/PacketMath.h
 *   - otherwise,
 *     // if we get here, this means the register size is greater than 4 (e.g., AVX with floats),
 *     // we currently fall back to the NoneAligned case
 *
 * The same reasoning apply for the transposed case.
 *
 * The last case (PacketSize>4) could probably be improved by generalizing the FirstAligned case, but since we do not support AVX yet...
 * One might also wonder why in the EvenAligned case we perform unaligned loads instead of using the aligned-loads plus re-alignment
 * strategy as in the FirstAligned case. The reason is that we observed that unaligned loads on a 8 byte boundary are not too slow
 * compared to unaligned loads on a 4 byte boundary.
 *
 */
template<typename Index, typename LhsScalar, typename LhsMapper, bool ConjugateLhs, typename RhsScalar, typename RhsMapper, bool ConjugateRhs, int Version>
struct general_matrix_vector_product<Index,LhsScalar,LhsMapper,ColMajor,ConjugateLhs,RhsScalar,RhsMapper,ConjugateRhs,Version>
{
  typedef typename scalar_product_traits<LhsScalar, RhsScalar>::ReturnType ResScalar;

enum {
  Vectorizable = packet_traits<LhsScalar>::Vectorizable && packet_traits<RhsScalar>::Vectorizable
              && int(packet_traits<LhsScalar>::size)==int(packet_traits<RhsScalar>::size),
  LhsPacketSize = Vectorizable ? packet_traits<LhsScalar>::size : 1,
  RhsPacketSize = Vectorizable ? packet_traits<RhsScalar>::size : 1,
  ResPacketSize = Vectorizable ? packet_traits<ResScalar>::size : 1
};

typedef typename packet_traits<LhsScalar>::type  _LhsPacket;
typedef typename packet_traits<RhsScalar>::type  _RhsPacket;
typedef typename packet_traits<ResScalar>::type  _ResPacket;

typedef typename conditional<Vectorizable,_LhsPacket,LhsScalar>::type LhsPacket;
typedef typename conditional<Vectorizable,_RhsPacket,RhsScalar>::type RhsPacket;
typedef typename conditional<Vectorizable,_ResPacket,ResScalar>::type ResPacket;

EIGEN_DONT_INLINE static void run(
  Index rows, Index cols,
  const LhsMapper& lhs,
  const RhsMapper& rhs,
        ResScalar* res, Index resIncr,
  RhsScalar alpha);
};

template<typename Index, typename LhsScalar, typename LhsMapper, bool ConjugateLhs, typename RhsScalar, typename RhsMapper, bool ConjugateRhs, int Version>
EIGEN_DONT_INLINE void general_matrix_vector_product<Index,LhsScalar,LhsMapper,ColMajor,ConjugateLhs,RhsScalar,RhsMapper,ConjugateRhs,Version>::run(
  Index rows, Index cols,
  const LhsMapper& lhs,
  const RhsMapper& rhs,
        ResScalar* res, Index resIncr,
  RhsScalar alpha)
{
  EIGEN_UNUSED_VARIABLE(resIncr);
  eigen_internal_assert(resIncr==1);
  #ifdef _EIGEN_ACCUMULATE_PACKETS
  #error _EIGEN_ACCUMULATE_PACKETS has already been defined
  #endif
  #define _EIGEN_ACCUMULATE_PACKETS(Alignment0,Alignment13,Alignment2) \
    pstore(&res[j], \
      padd(pload<ResPacket>(&res[j]), \
        padd( \
      padd(pcj.pmul(lhs0.template load<LhsPacket, Alignment0>(j),    ptmp0), \
      pcj.pmul(lhs1.template load<LhsPacket, Alignment13>(j),   ptmp1)),   \
      padd(pcj.pmul(lhs2.template load<LhsPacket, Alignment2>(j),    ptmp2), \
      pcj.pmul(lhs3.template load<LhsPacket, Alignment13>(j),   ptmp3)) )))

  typedef typename LhsMapper::VectorMapper LhsScalars;

  conj_helper<LhsScalar,RhsScalar,ConjugateLhs,ConjugateRhs> cj;
  conj_helper<LhsPacket,RhsPacket,ConjugateLhs,ConjugateRhs> pcj;
  if(ConjugateRhs)
    alpha = numext::conj(alpha);

  enum { AllAligned = 0, EvenAligned, FirstAligned, NoneAligned };
  const Index columnsAtOnce = 4;
  const Index peels = 2;
  const Index LhsPacketAlignedMask = LhsPacketSize-1;
  const Index ResPacketAlignedMask = ResPacketSize-1;
//  const Index PeelAlignedMask = ResPacketSize*peels-1;
  const Index size = rows;

  const Index lhsStride = lhs.stride();

  // How many coeffs of the result do we have to skip to be aligned.
  // Here we assume data are at least aligned on the base scalar type.
  Index alignedStart = internal::first_aligned(res,size);
  Index alignedSize = ResPacketSize>1 ? alignedStart + ((size-alignedStart) & ~ResPacketAlignedMask) : 0;
  const Index peeledSize = alignedSize - RhsPacketSize*peels - RhsPacketSize + 1;

  const Index alignmentStep = LhsPacketSize>1 ? (LhsPacketSize - lhsStride % LhsPacketSize) & LhsPacketAlignedMask : 0;
  Index alignmentPattern = alignmentStep==0 ? AllAligned
                       : alignmentStep==(LhsPacketSize/2) ? EvenAligned
                       : FirstAligned;

  // we cannot assume the first element is aligned because of sub-matrices
  const Index lhsAlignmentOffset = lhs.firstAligned(size);

  // find how many columns do we have to skip to be aligned with the result (if possible)
  Index skipColumns = 0;
  // if the data cannot be aligned (TODO add some compile time tests when possible, e.g. for floats)
  if( (lhsAlignmentOffset < 0) || (lhsAlignmentOffset == size) || (size_t(res)%sizeof(ResScalar)) )
  {
    alignedSize = 0;
    alignedStart = 0;
    alignmentPattern = NoneAligned;
  }
  else if(LhsPacketSize > 4)
  {
    // TODO: extend the code to support aligned loads whenever possible when LhsPacketSize > 4.
    // Currently, it seems to be better to perform unaligned loads anyway
    alignmentPattern = NoneAligned;
  }
  else if (LhsPacketSize>1)
  {
  //    eigen_internal_assert(size_t(firstLhs+lhsAlignmentOffset)%sizeof(LhsPacket)==0 || size<LhsPacketSize);

    while (skipColumns<LhsPacketSize &&
          alignedStart != ((lhsAlignmentOffset + alignmentStep*skipColumns)%LhsPacketSize))
      ++skipColumns;
    if (skipColumns==LhsPacketSize)
    {
      // nothing can be aligned, no need to skip any column
      alignmentPattern = NoneAligned;
      skipColumns = 0;
    }
    else
    {
      skipColumns = (std::min)(skipColumns,cols);
      // note that the skiped columns are processed later.
    }

    /*    eigen_internal_assert(  (alignmentPattern==NoneAligned)
                      || (skipColumns + columnsAtOnce >= cols)
                      || LhsPacketSize > size
                      || (size_t(firstLhs+alignedStart+lhsStride*skipColumns)%sizeof(LhsPacket))==0);*/
  }
  else if(Vectorizable)
  {
    alignedStart = 0;
    alignedSize = size;
    alignmentPattern = AllAligned;
  }

  const Index offset1 = (FirstAligned && alignmentStep==1?3:1);
  const Index offset3 = (FirstAligned && alignmentStep==1?1:3);

  Index columnBound = ((cols-skipColumns)/columnsAtOnce)*columnsAtOnce + skipColumns;
  for (Index i=skipColumns; i<columnBound; i+=columnsAtOnce)
  {
    RhsPacket ptmp0 = pset1<RhsPacket>(alpha*rhs(i, 0)),
              ptmp1 = pset1<RhsPacket>(alpha*rhs(i+offset1, 0)),
              ptmp2 = pset1<RhsPacket>(alpha*rhs(i+2, 0)),
              ptmp3 = pset1<RhsPacket>(alpha*rhs(i+offset3, 0));

    // this helps a lot generating better binary code
    const LhsScalars lhs0 = lhs.getVectorMapper(0, i+0),   lhs1 = lhs.getVectorMapper(0, i+offset1),
                     lhs2 = lhs.getVectorMapper(0, i+2),   lhs3 = lhs.getVectorMapper(0, i+offset3);

    if (Vectorizable)
    {
      /* explicit vectorization */
      // process initial unaligned coeffs
      for (Index j=0; j<alignedStart; ++j)
      {
        res[j] = cj.pmadd(lhs0(j), pfirst(ptmp0), res[j]);
        res[j] = cj.pmadd(lhs1(j), pfirst(ptmp1), res[j]);
        res[j] = cj.pmadd(lhs2(j), pfirst(ptmp2), res[j]);
        res[j] = cj.pmadd(lhs3(j), pfirst(ptmp3), res[j]);
      }

      if (alignedSize>alignedStart)
      {
        switch(alignmentPattern)
        {
          case AllAligned:
            for (Index j = alignedStart; j<alignedSize; j+=ResPacketSize)
              _EIGEN_ACCUMULATE_PACKETS(Aligned,Aligned,Aligned);
            break;
          case EvenAligned:
            for (Index j = alignedStart; j<alignedSize; j+=ResPacketSize)
              _EIGEN_ACCUMULATE_PACKETS(Aligned,Unaligned,Aligned);
            break;
          case FirstAligned:
          {
            Index j = alignedStart;
            if(peels>1)
            {
              LhsPacket A00, A01, A02, A03, A10, A11, A12, A13;
              ResPacket T0, T1;

              A01 = lhs1.template load<LhsPacket, Aligned>(alignedStart-1);
              A02 = lhs2.template load<LhsPacket, Aligned>(alignedStart-2);
              A03 = lhs3.template load<LhsPacket, Aligned>(alignedStart-3);

              for (; j<peeledSize; j+=peels*ResPacketSize)
              {
                A11 = lhs1.template load<LhsPacket, Aligned>(j-1+LhsPacketSize);  palign<1>(A01,A11);
                A12 = lhs2.template load<LhsPacket, Aligned>(j-2+LhsPacketSize);  palign<2>(A02,A12);
                A13 = lhs3.template load<LhsPacket, Aligned>(j-3+LhsPacketSize);  palign<3>(A03,A13);

                A00 = lhs0.template load<LhsPacket, Aligned>(j);
                A10 = lhs0.template load<LhsPacket, Aligned>(j+LhsPacketSize);
                T0  = pcj.pmadd(A00, ptmp0, pload<ResPacket>(&res[j]));
                T1  = pcj.pmadd(A10, ptmp0, pload<ResPacket>(&res[j+ResPacketSize]));

                T0  = pcj.pmadd(A01, ptmp1, T0);
                A01 = lhs1.template load<LhsPacket, Aligned>(j-1+2*LhsPacketSize);  palign<1>(A11,A01);
                T0  = pcj.pmadd(A02, ptmp2, T0);
                A02 = lhs2.template load<LhsPacket, Aligned>(j-2+2*LhsPacketSize);  palign<2>(A12,A02);
                T0  = pcj.pmadd(A03, ptmp3, T0);
                pstore(&res[j],T0);
                A03 = lhs3.template load<LhsPacket, Aligned>(j-3+2*LhsPacketSize);  palign<3>(A13,A03);
                T1  = pcj.pmadd(A11, ptmp1, T1);
                T1  = pcj.pmadd(A12, ptmp2, T1);
                T1  = pcj.pmadd(A13, ptmp3, T1);
                pstore(&res[j+ResPacketSize],T1);
              }
            }
            for (; j<alignedSize; j+=ResPacketSize)
              _EIGEN_ACCUMULATE_PACKETS(Aligned,Unaligned,Unaligned);
            break;
          }
          default:
            for (Index j = alignedStart; j<alignedSize; j+=ResPacketSize)
              _EIGEN_ACCUMULATE_PACKETS(Unaligned,Unaligned,Unaligned);
            break;
        }
      }
    } // end explicit vectorization

    /* process remaining coeffs (or all if there is no explicit vectorization) */
    for (Index j=alignedSize; j<size; ++j)
    {
      res[j] = cj.pmadd(lhs0(j), pfirst(ptmp0), res[j]);
      res[j] = cj.pmadd(lhs1(j), pfirst(ptmp1), res[j]);
      res[j] = cj.pmadd(lhs2(j), pfirst(ptmp2), res[j]);
      res[j] = cj.pmadd(lhs3(j), pfirst(ptmp3), res[j]);
    }
  }

  // process remaining first and last columns (at most columnsAtOnce-1)
  Index end = cols;
  Index start = columnBound;
  do
  {
    for (Index k=start; k<end; ++k)
    {
      RhsPacket ptmp0 = pset1<RhsPacket>(alpha*rhs(k, 0));
      const LhsScalars lhs0 = lhs.getVectorMapper(0, k);

      if (Vectorizable)
      {
        /* explicit vectorization */
        // process first unaligned result's coeffs
        for (Index j=0; j<alignedStart; ++j)
          res[j] += cj.pmul(lhs0(j), pfirst(ptmp0));
        // process aligned result's coeffs
        if (lhs0.template aligned<LhsPacket>(alignedStart))
          for (Index i = alignedStart;i<alignedSize;i+=ResPacketSize)
            pstore(&res[i], pcj.pmadd(lhs0.template load<LhsPacket, Aligned>(i), ptmp0, pload<ResPacket>(&res[i])));
        else
          for (Index i = alignedStart;i<alignedSize;i+=ResPacketSize)
            pstore(&res[i], pcj.pmadd(lhs0.template load<LhsPacket, Unaligned>(i), ptmp0, pload<ResPacket>(&res[i])));
      }

      // process remaining scalars (or all if no explicit vectorization)
      for (Index i=alignedSize; i<size; ++i)
        res[i] += cj.pmul(lhs0(i), pfirst(ptmp0));
    }
    if (skipColumns)
    {
      start = 0;
      end = skipColumns;
      skipColumns = 0;
    }
    else
      break;
  } while(Vectorizable);
  #undef _EIGEN_ACCUMULATE_PACKETS
}

/* Optimized row-major matrix * vector product:
 * This algorithm processes 4 rows at onces that allows to both reduce
 * the number of load/stores of the result by a factor 4 and to reduce
 * the instruction dependency. Moreover, we know that all bands have the
 * same alignment pattern.
 *
 * Mixing type logic:
 *  - alpha is always a complex (or converted to a complex)
 *  - no vectorization
 */
template<typename Index, typename LhsScalar, typename LhsMapper, bool ConjugateLhs, typename RhsScalar, typename RhsMapper, bool ConjugateRhs, int Version>
struct general_matrix_vector_product<Index,LhsScalar,LhsMapper,RowMajor,ConjugateLhs,RhsScalar,RhsMapper,ConjugateRhs,Version>
{
typedef typename scalar_product_traits<LhsScalar, RhsScalar>::ReturnType ResScalar;

enum {
  Vectorizable = packet_traits<LhsScalar>::Vectorizable && packet_traits<RhsScalar>::Vectorizable
              && int(packet_traits<LhsScalar>::size)==int(packet_traits<RhsScalar>::size),
  LhsPacketSize = Vectorizable ? packet_traits<LhsScalar>::size : 1,
  RhsPacketSize = Vectorizable ? packet_traits<RhsScalar>::size : 1,
  ResPacketSize = Vectorizable ? packet_traits<ResScalar>::size : 1
};

typedef typename packet_traits<LhsScalar>::type  _LhsPacket;
typedef typename packet_traits<RhsScalar>::type  _RhsPacket;
typedef typename packet_traits<ResScalar>::type  _ResPacket;

typedef typename conditional<Vectorizable,_LhsPacket,LhsScalar>::type LhsPacket;
typedef typename conditional<Vectorizable,_RhsPacket,RhsScalar>::type RhsPacket;
typedef typename conditional<Vectorizable,_ResPacket,ResScalar>::type ResPacket;

EIGEN_DONT_INLINE static void run(
  Index rows, Index cols,
  const LhsMapper& lhs,
  const RhsMapper& rhs,
        ResScalar* res, Index resIncr,
  ResScalar alpha);
};

template<typename Index, typename LhsScalar, typename LhsMapper, bool ConjugateLhs, typename RhsScalar, typename RhsMapper, bool ConjugateRhs, int Version>
EIGEN_DONT_INLINE void general_matrix_vector_product<Index,LhsScalar,LhsMapper,RowMajor,ConjugateLhs,RhsScalar,RhsMapper,ConjugateRhs,Version>::run(
  Index rows, Index cols,
  const LhsMapper& lhs,
  const RhsMapper& rhs,
  ResScalar* res, Index resIncr,
  ResScalar alpha)
{
  eigen_internal_assert(rhs.stride()==1);

  #ifdef _EIGEN_ACCUMULATE_PACKETS
  #error _EIGEN_ACCUMULATE_PACKETS has already been defined
  #endif

  #define _EIGEN_ACCUMULATE_PACKETS(Alignment0,Alignment13,Alignment2) {\
    RhsPacket b = rhs.getVectorMapper(j, 0).template load<RhsPacket, Aligned>(0);  \
    ptmp0 = pcj.pmadd(lhs0.template load<LhsPacket, Alignment0>(j), b, ptmp0); \
    ptmp1 = pcj.pmadd(lhs1.template load<LhsPacket, Alignment13>(j), b, ptmp1); \
    ptmp2 = pcj.pmadd(lhs2.template load<LhsPacket, Alignment2>(j), b, ptmp2); \
    ptmp3 = pcj.pmadd(lhs3.template load<LhsPacket, Alignment13>(j), b, ptmp3); }

  conj_helper<LhsScalar,RhsScalar,ConjugateLhs,ConjugateRhs> cj;
  conj_helper<LhsPacket,RhsPacket,ConjugateLhs,ConjugateRhs> pcj;

  typedef typename LhsMapper::VectorMapper LhsScalars;

  enum { AllAligned=0, EvenAligned=1, FirstAligned=2, NoneAligned=3 };
  const Index rowsAtOnce = 4;
  const Index peels = 2;
  const Index RhsPacketAlignedMask = RhsPacketSize-1;
  const Index LhsPacketAlignedMask = LhsPacketSize-1;
  const Index depth = cols;
  const Index lhsStride = lhs.stride();

  // How many coeffs of the result do we have to skip to be aligned.
  // Here we assume data are at least aligned on the base scalar type
  // if that's not the case then vectorization is discarded, see below.
  Index alignedStart = rhs.firstAligned(depth);
  Index alignedSize = RhsPacketSize>1 ? alignedStart + ((depth-alignedStart) & ~RhsPacketAlignedMask) : 0;
  const Index peeledSize = alignedSize - RhsPacketSize*peels - RhsPacketSize + 1;

  const Index alignmentStep = LhsPacketSize>1 ? (LhsPacketSize - lhsStride % LhsPacketSize) & LhsPacketAlignedMask : 0;
  Index alignmentPattern = alignmentStep==0 ? AllAligned
                           : alignmentStep==(LhsPacketSize/2) ? EvenAligned
                           : FirstAligned;

  // we cannot assume the first element is aligned because of sub-matrices
  const Index lhsAlignmentOffset = lhs.firstAligned(depth);
  const Index rhsAlignmentOffset = rhs.firstAligned(rows);

  // find how many rows do we have to skip to be aligned with rhs (if possible)
  Index skipRows = 0;
  // if the data cannot be aligned (TODO add some compile time tests when possible, e.g. for floats)
  if( (sizeof(LhsScalar)!=sizeof(RhsScalar))
      || (lhsAlignmentOffset < 0) || (lhsAlignmentOffset == depth)
      || (rhsAlignmentOffset < 0) || (rhsAlignmentOffset == rows))
  {
    alignedSize = 0;
    alignedStart = 0;
    alignmentPattern = NoneAligned;
  }
  else if(LhsPacketSize > 4)
  {
    // TODO: extend the code to support aligned loads whenever possible when LhsPacketSize > 4.
    alignmentPattern = NoneAligned;
  }
  else if (LhsPacketSize>1)
  {
  //    eigen_internal_assert(size_t(firstLhs+lhsAlignmentOffset)%sizeof(LhsPacket)==0  || depth<LhsPacketSize);

    while (skipRows<LhsPacketSize &&
           alignedStart != ((lhsAlignmentOffset + alignmentStep*skipRows)%LhsPacketSize))
      ++skipRows;
    if (skipRows==LhsPacketSize)
    {
      // nothing can be aligned, no need to skip any column
      alignmentPattern = NoneAligned;
      skipRows = 0;
    }
    else
    {
      skipRows = (std::min)(skipRows,Index(rows));
      // note that the skiped columns are processed later.
    }
    /*    eigen_internal_assert(  alignmentPattern==NoneAligned
                      || LhsPacketSize==1
                      || (skipRows + rowsAtOnce >= rows)
                      || LhsPacketSize > depth
                      || (size_t(firstLhs+alignedStart+lhsStride*skipRows)%sizeof(LhsPacket))==0);*/
  }
  else if(Vectorizable)
  {
    alignedStart = 0;
    alignedSize = depth;
    alignmentPattern = AllAligned;
  }

  const Index offset1 = (FirstAligned && alignmentStep==1?3:1);
  const Index offset3 = (FirstAligned && alignmentStep==1?1:3);

  Index rowBound = ((rows-skipRows)/rowsAtOnce)*rowsAtOnce + skipRows;
  for (Index i=skipRows; i<rowBound; i+=rowsAtOnce)
  {
    EIGEN_ALIGN_DEFAULT ResScalar tmp0 = ResScalar(0);
    ResScalar tmp1 = ResScalar(0), tmp2 = ResScalar(0), tmp3 = ResScalar(0);

    // this helps the compiler generating good binary code
    const LhsScalars lhs0 = lhs.getVectorMapper(i+0, 0),    lhs1 = lhs.getVectorMapper(i+offset1, 0),
                     lhs2 = lhs.getVectorMapper(i+2, 0),    lhs3 = lhs.getVectorMapper(i+offset3, 0);

    if (Vectorizable)
    {
      /* explicit vectorization */
      ResPacket ptmp0 = pset1<ResPacket>(ResScalar(0)), ptmp1 = pset1<ResPacket>(ResScalar(0)),
                ptmp2 = pset1<ResPacket>(ResScalar(0)), ptmp3 = pset1<ResPacket>(ResScalar(0));

      // process initial unaligned coeffs
      // FIXME this loop get vectorized by the compiler !
      for (Index j=0; j<alignedStart; ++j)
      {
        RhsScalar b = rhs(j, 0);
        tmp0 += cj.pmul(lhs0(j),b); tmp1 += cj.pmul(lhs1(j),b);
        tmp2 += cj.pmul(lhs2(j),b); tmp3 += cj.pmul(lhs3(j),b);
      }

      if (alignedSize>alignedStart)
      {
        switch(alignmentPattern)
        {
          case AllAligned:
            for (Index j = alignedStart; j<alignedSize; j+=RhsPacketSize)
              _EIGEN_ACCUMULATE_PACKETS(Aligned,Aligned,Aligned);
            break;
          case EvenAligned:
            for (Index j = alignedStart; j<alignedSize; j+=RhsPacketSize)
              _EIGEN_ACCUMULATE_PACKETS(Aligned,Unaligned,Aligned);
            break;
          case FirstAligned:
          {
            Index j = alignedStart;
            if (peels>1)
            {
              /* Here we proccess 4 rows with with two peeled iterations to hide
               * the overhead of unaligned loads. Moreover unaligned loads are handled
               * using special shift/move operations between the two aligned packets
               * overlaping the desired unaligned packet. This is *much* more efficient
               * than basic unaligned loads.
               */
              LhsPacket A01, A02, A03, A11, A12, A13;
              A01 = lhs1.template load<LhsPacket, Aligned>(alignedStart-1);
              A02 = lhs2.template load<LhsPacket, Aligned>(alignedStart-2);
              A03 = lhs3.template load<LhsPacket, Aligned>(alignedStart-3);

              for (; j<peeledSize; j+=peels*RhsPacketSize)
              {
                RhsPacket b = rhs.getVectorMapper(j, 0).template load<RhsPacket, Aligned>(0);
                A11 = lhs1.template load<LhsPacket, Aligned>(j-1+LhsPacketSize);  palign<1>(A01,A11);
                A12 = lhs2.template load<LhsPacket, Aligned>(j-2+LhsPacketSize);  palign<2>(A02,A12);
                A13 = lhs3.template load<LhsPacket, Aligned>(j-3+LhsPacketSize);  palign<3>(A03,A13);

                ptmp0 = pcj.pmadd(lhs0.template load<LhsPacket, Aligned>(j), b, ptmp0);
                ptmp1 = pcj.pmadd(A01, b, ptmp1);
                A01 = lhs1.template load<LhsPacket, Aligned>(j-1+2*LhsPacketSize);  palign<1>(A11,A01);
                ptmp2 = pcj.pmadd(A02, b, ptmp2);
                A02 = lhs2.template load<LhsPacket, Aligned>(j-2+2*LhsPacketSize);  palign<2>(A12,A02);
                ptmp3 = pcj.pmadd(A03, b, ptmp3);
                A03 = lhs3.template load<LhsPacket, Aligned>(j-3+2*LhsPacketSize);  palign<3>(A13,A03);

                b = rhs.getVectorMapper(j+RhsPacketSize, 0).template load<RhsPacket, Aligned>(0);
                ptmp0 = pcj.pmadd(lhs0.template load<LhsPacket, Aligned>(j+LhsPacketSize), b, ptmp0);
                ptmp1 = pcj.pmadd(A11, b, ptmp1);
                ptmp2 = pcj.pmadd(A12, b, ptmp2);
                ptmp3 = pcj.pmadd(A13, b, ptmp3);
              }
            }
            for (; j<alignedSize; j+=RhsPacketSize)
              _EIGEN_ACCUMULATE_PACKETS(Aligned,Unaligned,Unaligned);
            break;
          }
          default:
            for (Index j = alignedStart; j<alignedSize; j+=RhsPacketSize)
              _EIGEN_ACCUMULATE_PACKETS(Unaligned,Unaligned,Unaligned);
            break;
        }
        tmp0 += predux(ptmp0);
        tmp1 += predux(ptmp1);
        tmp2 += predux(ptmp2);
        tmp3 += predux(ptmp3);
      }
    } // end explicit vectorization

    // process remaining coeffs (or all if no explicit vectorization)
    // FIXME this loop get vectorized by the compiler !
    for (Index j=alignedSize; j<depth; ++j)
    {
      RhsScalar b = rhs(j, 0);
      tmp0 += cj.pmul(lhs0(j),b); tmp1 += cj.pmul(lhs1(j),b);
      tmp2 += cj.pmul(lhs2(j),b); tmp3 += cj.pmul(lhs3(j),b);
    }
    res[i*resIncr]            += alpha*tmp0;
    res[(i+offset1)*resIncr]  += alpha*tmp1;
    res[(i+2)*resIncr]        += alpha*tmp2;
    res[(i+offset3)*resIncr]  += alpha*tmp3;
  }

  // process remaining first and last rows (at most columnsAtOnce-1)
  Index end = rows;
  Index start = rowBound;
  do
  {
    for (Index i=start; i<end; ++i)
    {
      EIGEN_ALIGN_DEFAULT ResScalar tmp0 = ResScalar(0);
      ResPacket ptmp0 = pset1<ResPacket>(tmp0);
      const LhsScalars lhs0 = lhs.getVectorMapper(i, 0);
      // process first unaligned result's coeffs
      // FIXME this loop get vectorized by the compiler !
      for (Index j=0; j<alignedStart; ++j)
        tmp0 += cj.pmul(lhs0(j), rhs(j, 0));

      if (alignedSize>alignedStart)
      {
        // process aligned rhs coeffs
        if (lhs0.template aligned<LhsPacket>(alignedStart))
          for (Index j = alignedStart;j<alignedSize;j+=RhsPacketSize)
            ptmp0 = pcj.pmadd(lhs0.template load<LhsPacket, Aligned>(j), rhs.getVectorMapper(j, 0).template load<RhsPacket, Aligned>(0), ptmp0);
        else
          for (Index j = alignedStart;j<alignedSize;j+=RhsPacketSize)
            ptmp0 = pcj.pmadd(lhs0.template load<LhsPacket, Unaligned>(j), rhs.getVectorMapper(j, 0).template load<RhsPacket, Aligned>(0), ptmp0);
        tmp0 += predux(ptmp0);
      }

      // process remaining scalars
      // FIXME this loop get vectorized by the compiler !
      for (Index j=alignedSize; j<depth; ++j)
        tmp0 += cj.pmul(lhs0(j), rhs(j, 0));
      res[i*resIncr] += alpha*tmp0;
    }
    if (skipRows)
    {
      start = 0;
      end = skipRows;
      skipRows = 0;
    }
    else
      break;
  } while(Vectorizable);

  #undef _EIGEN_ACCUMULATE_PACKETS
}

} // end namespace internal

} // end namespace Eigen

#endif // EIGEN_GENERAL_MATRIX_VECTOR_H
