/*
 Copyright (c) 2011, Intel Corporation. All rights reserved.

 Redistribution and use in source and binary forms, with or without modification,
 are permitted provided that the following conditions are met:

 * Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.
 * Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.
 * Neither the name of Intel Corporation nor the names of its contributors may
   be used to endorse or promote products derived from this software without
   specific prior written permission.

 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
 ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

 ********************************************************************************
 *   Content : Eigen bindings to Intel(R) MKL
 *   MKL VML support for coefficient-wise unary Eigen expressions like a=b.sin()
 ********************************************************************************
*/

#ifndef EIGEN_ASSIGN_VML_H
#define EIGEN_ASSIGN_VML_H

namespace Eigen { 

namespace internal {

template<typename Op> struct vml_call
{ enum { IsSupported = 0 }; };

template<typename Dst, typename Src, typename UnaryOp>
class vml_assign_traits
{
  private:
    enum {
      DstHasDirectAccess = Dst::Flags & DirectAccessBit,
      SrcHasDirectAccess = Src::Flags & DirectAccessBit,

      StorageOrdersAgree = (int(Dst::IsRowMajor) == int(Src::IsRowMajor)),
      InnerSize = int(Dst::IsVectorAtCompileTime) ? int(Dst::SizeAtCompileTime)
                : int(Dst::Flags)&RowMajorBit ? int(Dst::ColsAtCompileTime)
                : int(Dst::RowsAtCompileTime),
      InnerMaxSize  = int(Dst::IsVectorAtCompileTime) ? int(Dst::MaxSizeAtCompileTime)
                    : int(Dst::Flags)&RowMajorBit ? int(Dst::MaxColsAtCompileTime)
                    : int(Dst::MaxRowsAtCompileTime),
      MaxSizeAtCompileTime = Dst::SizeAtCompileTime,

      MightEnableVml =  vml_call<UnaryOp>::IsSupported && StorageOrdersAgree && DstHasDirectAccess && SrcHasDirectAccess
                     && Src::InnerStrideAtCompileTime==1 && Dst::InnerStrideAtCompileTime==1,
      MightLinearize = MightEnableVml && (int(Dst::Flags) & int(Src::Flags) & LinearAccessBit),
      VmlSize = MightLinearize ? MaxSizeAtCompileTime : InnerMaxSize,
      LargeEnough = VmlSize==Dynamic || VmlSize>=EIGEN_MKL_VML_THRESHOLD,
      MayEnableVml = MightEnableVml && LargeEnough,
      MayLinearize = MayEnableVml && MightLinearize
    };
  public:
    enum {
      Traversal = MayLinearize ? LinearVectorizedTraversal
                : MayEnableVml ? InnerVectorizedTraversal
                : DefaultTraversal
    };
};

template<typename Derived1, typename Derived2, typename UnaryOp, int Traversal, int Unrolling,
         int VmlTraversal = vml_assign_traits<Derived1, Derived2, UnaryOp>::Traversal >
struct vml_assign_impl
  : assign_impl<Derived1, Eigen::CwiseUnaryOp<UnaryOp, Derived2>,Traversal,Unrolling,BuiltIn>
{
};

template<typename Derived1, typename Derived2, typename UnaryOp, int Traversal, int Unrolling>
struct vml_assign_impl<Derived1, Derived2, UnaryOp, Traversal, Unrolling, InnerVectorizedTraversal>
{
  typedef typename Derived1::Scalar Scalar;
  typedef typename Derived1::Index Index;
  static inline void run(Derived1& dst, const CwiseUnaryOp<UnaryOp, Derived2>& src)
  {
    // in case we want to (or have to) skip VML at runtime we can call:
    // assign_impl<Derived1,Eigen::CwiseUnaryOp<UnaryOp, Derived2>,Traversal,Unrolling,BuiltIn>::run(dst,src);
    const Index innerSize = dst.innerSize();
    const Index outerSize = dst.outerSize();
    for(Index outer = 0; outer < outerSize; ++outer) {
      const Scalar *src_ptr = src.IsRowMajor ?  &(src.nestedExpression().coeffRef(outer,0)) :
                                                &(src.nestedExpression().coeffRef(0, outer));
      Scalar *dst_ptr = dst.IsRowMajor ? &(dst.coeffRef(outer,0)) : &(dst.coeffRef(0, outer));
      vml_call<UnaryOp>::run(src.functor(), innerSize, src_ptr, dst_ptr );
    }
  }
};

template<typename Derived1, typename Derived2, typename UnaryOp, int Traversal, int Unrolling>
struct vml_assign_impl<Derived1, Derived2, UnaryOp, Traversal, Unrolling, LinearVectorizedTraversal>
{
  static inline void run(Derived1& dst, const CwiseUnaryOp<UnaryOp, Derived2>& src)
  {
    // in case we want to (or have to) skip VML at runtime we can call:
    // assign_impl<Derived1,Eigen::CwiseUnaryOp<UnaryOp, Derived2>,Traversal,Unrolling,BuiltIn>::run(dst,src);
    vml_call<UnaryOp>::run(src.functor(), dst.size(), src.nestedExpression().data(), dst.data() );
  }
};

// Macroses

#define EIGEN_MKL_VML_SPECIALIZE_ASSIGN(TRAVERSAL,UNROLLING) \
  template<typename Derived1, typename Derived2, typename UnaryOp> \
  struct assign_impl<Derived1, Eigen::CwiseUnaryOp<UnaryOp, Derived2>, TRAVERSAL, UNROLLING, Specialized>  {  \
    static inline void run(Derived1 &dst, const Eigen::CwiseUnaryOp<UnaryOp, Derived2> &src) { \
      vml_assign_impl<Derived1,Derived2,UnaryOp,TRAVERSAL,UNROLLING>::run(dst, src); \
    } \
  };

EIGEN_MKL_VML_SPECIALIZE_ASSIGN(DefaultTraversal,NoUnrolling)
EIGEN_MKL_VML_SPECIALIZE_ASSIGN(DefaultTraversal,CompleteUnrolling)
EIGEN_MKL_VML_SPECIALIZE_ASSIGN(DefaultTraversal,InnerUnrolling)
EIGEN_MKL_VML_SPECIALIZE_ASSIGN(LinearTraversal,NoUnrolling)
EIGEN_MKL_VML_SPECIALIZE_ASSIGN(LinearTraversal,CompleteUnrolling)
EIGEN_MKL_VML_SPECIALIZE_ASSIGN(InnerVectorizedTraversal,NoUnrolling)
EIGEN_MKL_VML_SPECIALIZE_ASSIGN(InnerVectorizedTraversal,CompleteUnrolling)
EIGEN_MKL_VML_SPECIALIZE_ASSIGN(InnerVectorizedTraversal,InnerUnrolling)
EIGEN_MKL_VML_SPECIALIZE_ASSIGN(LinearVectorizedTraversal,CompleteUnrolling)
EIGEN_MKL_VML_SPECIALIZE_ASSIGN(LinearVectorizedTraversal,NoUnrolling)
EIGEN_MKL_VML_SPECIALIZE_ASSIGN(SliceVectorizedTraversal,NoUnrolling)


#if !defined (EIGEN_FAST_MATH) || (EIGEN_FAST_MATH != 1)
#define  EIGEN_MKL_VML_MODE VML_HA
#else
#define  EIGEN_MKL_VML_MODE VML_LA
#endif

#define EIGEN_MKL_VML_DECLARE_UNARY_CALL(EIGENOP, VMLOP, EIGENTYPE, VMLTYPE)     \
  template<> struct vml_call< scalar_##EIGENOP##_op<EIGENTYPE> > {               \
    enum { IsSupported = 1 };                                                    \
    static inline void run( const scalar_##EIGENOP##_op<EIGENTYPE>& /*func*/,        \
                            int size, const EIGENTYPE* src, EIGENTYPE* dst) {    \
      VMLOP(size, (const VMLTYPE*)src, (VMLTYPE*)dst);                           \
    }                                                                            \
  };

#define EIGEN_MKL_VML_DECLARE_UNARY_CALL_LA(EIGENOP, VMLOP, EIGENTYPE, VMLTYPE)  \
  template<> struct vml_call< scalar_##EIGENOP##_op<EIGENTYPE> > {               \
    enum { IsSupported = 1 };                                                    \
    static inline void run( const scalar_##EIGENOP##_op<EIGENTYPE>& /*func*/,        \
                            int size, const EIGENTYPE* src, EIGENTYPE* dst) {    \
      MKL_INT64 vmlMode = EIGEN_MKL_VML_MODE;                                    \
      VMLOP(size, (const VMLTYPE*)src, (VMLTYPE*)dst, vmlMode);                  \
    }                                                                            \
  };

#define EIGEN_MKL_VML_DECLARE_POW_CALL(EIGENOP, VMLOP, EIGENTYPE, VMLTYPE)       \
  template<> struct vml_call< scalar_##EIGENOP##_op<EIGENTYPE> > {               \
    enum { IsSupported = 1 };                                                    \
    static inline void run( const scalar_##EIGENOP##_op<EIGENTYPE>& func,        \
                          int size, const EIGENTYPE* src, EIGENTYPE* dst) {      \
      EIGENTYPE exponent = func.m_exponent;                                      \
      MKL_INT64 vmlMode = EIGEN_MKL_VML_MODE;                                    \
      VMLOP(&size, (const VMLTYPE*)src, (const VMLTYPE*)&exponent,               \
                        (VMLTYPE*)dst, &vmlMode);                                \
    }                                                                            \
  };

#define EIGEN_MKL_VML_DECLARE_UNARY_CALLS_REAL(EIGENOP, VMLOP)                   \
  EIGEN_MKL_VML_DECLARE_UNARY_CALL(EIGENOP, vs##VMLOP, float, float)             \
  EIGEN_MKL_VML_DECLARE_UNARY_CALL(EIGENOP, vd##VMLOP, double, double)

#define EIGEN_MKL_VML_DECLARE_UNARY_CALLS_COMPLEX(EIGENOP, VMLOP)                \
  EIGEN_MKL_VML_DECLARE_UNARY_CALL(EIGENOP, vc##VMLOP, scomplex, MKL_Complex8)   \
  EIGEN_MKL_VML_DECLARE_UNARY_CALL(EIGENOP, vz##VMLOP, dcomplex, MKL_Complex16)

#define EIGEN_MKL_VML_DECLARE_UNARY_CALLS(EIGENOP, VMLOP)                        \
  EIGEN_MKL_VML_DECLARE_UNARY_CALLS_REAL(EIGENOP, VMLOP)                         \
  EIGEN_MKL_VML_DECLARE_UNARY_CALLS_COMPLEX(EIGENOP, VMLOP)


#define EIGEN_MKL_VML_DECLARE_UNARY_CALLS_REAL_LA(EIGENOP, VMLOP)                \
  EIGEN_MKL_VML_DECLARE_UNARY_CALL_LA(EIGENOP, vms##VMLOP, float, float)         \
  EIGEN_MKL_VML_DECLARE_UNARY_CALL_LA(EIGENOP, vmd##VMLOP, double, double)

#define EIGEN_MKL_VML_DECLARE_UNARY_CALLS_COMPLEX_LA(EIGENOP, VMLOP)             \
  EIGEN_MKL_VML_DECLARE_UNARY_CALL_LA(EIGENOP, vmc##VMLOP, scomplex, MKL_Complex8)  \
  EIGEN_MKL_VML_DECLARE_UNARY_CALL_LA(EIGENOP, vmz##VMLOP, dcomplex, MKL_Complex16)

#define EIGEN_MKL_VML_DECLARE_UNARY_CALLS_LA(EIGENOP, VMLOP)                     \
  EIGEN_MKL_VML_DECLARE_UNARY_CALLS_REAL_LA(EIGENOP, VMLOP)                      \
  EIGEN_MKL_VML_DECLARE_UNARY_CALLS_COMPLEX_LA(EIGENOP, VMLOP)


EIGEN_MKL_VML_DECLARE_UNARY_CALLS_LA(sin,  Sin)
EIGEN_MKL_VML_DECLARE_UNARY_CALLS_LA(asin, Asin)
EIGEN_MKL_VML_DECLARE_UNARY_CALLS_LA(cos,  Cos)
EIGEN_MKL_VML_DECLARE_UNARY_CALLS_LA(acos, Acos)
EIGEN_MKL_VML_DECLARE_UNARY_CALLS_LA(tan,  Tan)
EIGEN_MKL_VML_DECLARE_UNARY_CALLS_LA(atan,  Atan)
//EIGEN_MKL_VML_DECLARE_UNARY_CALLS(abs,  Abs)
EIGEN_MKL_VML_DECLARE_UNARY_CALLS_LA(exp,  Exp)
EIGEN_MKL_VML_DECLARE_UNARY_CALLS_LA(log,  Ln)
EIGEN_MKL_VML_DECLARE_UNARY_CALLS_LA(sqrt, Sqrt)

EIGEN_MKL_VML_DECLARE_UNARY_CALLS_REAL(square, Sqr)

// The vm*powx functions are not avaibale in the windows version of MKL.
#ifndef _WIN32
EIGEN_MKL_VML_DECLARE_POW_CALL(pow, vmspowx_, float, float)
EIGEN_MKL_VML_DECLARE_POW_CALL(pow, vmdpowx_, double, double)
EIGEN_MKL_VML_DECLARE_POW_CALL(pow, vmcpowx_, scomplex, MKL_Complex8)
EIGEN_MKL_VML_DECLARE_POW_CALL(pow, vmzpowx_, dcomplex, MKL_Complex16)
#endif

} // end namespace internal

} // end namespace Eigen

#endif // EIGEN_ASSIGN_VML_H
