// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008-2010 Gael Guennebaud <gael.guennebaud@inria.fr>
// Copyright (C) 2006-2008 Benoit Jacob <jacob.benoit.1@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_MACROS_H
#define EIGEN_MACROS_H

#define EIGEN_WORLD_VERSION 3
#define EIGEN_MAJOR_VERSION 2
#define EIGEN_MINOR_VERSION 90

#define EIGEN_VERSION_AT_LEAST(x,y,z) (EIGEN_WORLD_VERSION>x || (EIGEN_WORLD_VERSION>=x && \
                                      (EIGEN_MAJOR_VERSION>y || (EIGEN_MAJOR_VERSION>=y && \
                                                                 EIGEN_MINOR_VERSION>=z))))

// Compiler identification, EIGEN_COMP_*
/// \internal EIGEN_COMP_GNUC set to 1 for all compilers compatible with GCC
#ifdef __GNUC__
  #define EIGEN_COMP_GNUC 1
#else
  #define EIGEN_COMP_GNUC 0
#endif

/// \internal EIGEN_COMP_CLANG set to 1 if the compiler is clang (alias for __clang__)
#if defined(__clang__)
  #define EIGEN_COMP_CLANG 1
#else
  #define EIGEN_COMP_CLANG 0
#endif


/// \internal EIGEN_COMP_LLVM set to 1 if the compiler backend is llvm
#if defined(__llvm__)
  #define EIGEN_COMP_LLVM 1
#else
  #define EIGEN_COMP_LLVM 0
#endif

/// \internal EIGEN_COMP_ICC set to __INTEL_COMPILER if the compiler is Intel compiler, 0 otherwise
#if defined(__INTEL_COMPILER)
  #define EIGEN_COMP_ICC __INTEL_COMPILER
#else
  #define EIGEN_COMP_ICC 0
#endif

/// \internal EIGEN_COMP_MINGW set to 1 if the compiler is mingw
#if defined(__MINGW32__)
  #define EIGEN_COMP_MINGW 1
#else
  #define EIGEN_COMP_MINGW 0
#endif

/// \internal EIGEN_COMP_SUNCC set to 1 if the compiler is Solaris Studio
#if defined(__SUNPRO_CC)
  #define EIGEN_COMP_SUNCC 1
#else
  #define EIGEN_COMP_SUNCC 0
#endif

/// \internal EIGEN_COMP_MSVC set to _MSC_VER if the compiler is Microsoft Visual C++, 0 otherwise.
#if defined(_MSC_VER)
  #define EIGEN_COMP_MSVC _MSC_VER
#else
  #define EIGEN_COMP_MSVC 0
#endif

/// \internal EIGEN_COMP_MSVC_STRICT set to 1 if the compiler is really Microsoft Visual C++ and not ,e.g., ICC
#if EIGEN_COMP_MSVC && !(EIGEN_COMP_ICC)
  #define EIGEN_COMP_MSVC_STRICT 1
#else
  #define EIGEN_COMP_MSVC_STRICT 0
#endif

/// \internal EIGEN_COMP_IBM set to 1 if the compiler is IBM XL C++
#if defined(__IBMCPP__) || defined(__xlc__)
  #define EIGEN_COMP_IBM 1
#else
  #define EIGEN_COMP_IBM 0
#endif

/// \internal EIGEN_COMP_PGI set to 1 if the compiler is Portland Group Compiler
#if defined(__PGI)
  #define EIGEN_COMP_PGI 1
#else
  #define EIGEN_COMP_PGI 0
#endif

/// \internal EIGEN_COMP_ARM set to 1 if the compiler is ARM Compiler
#if defined(__CC_ARM) || defined(__ARMCC_VERSION)
  #define EIGEN_COMP_ARM 1
#else
  #define EIGEN_COMP_ARM 0
#endif


/// \internal EIGEN_GNUC_STRICT set to 1 if the compiler is really GCC and not a compatible compiler (e.g., ICC, clang, mingw, etc.)
#if EIGEN_COMP_GNUC && !(EIGEN_COMP_CLANG || EIGEN_COMP_CLANG || EIGEN_COMP_MINGW || EIGEN_COMP_PGI || EIGEN_COMP_IBM || EIGEN_COMP_ARM )
  #define EIGEN_COMP_GNUC_STRICT 1
#else
  #define EIGEN_COMP_GNUC_STRICT 0
#endif


#if EIGEN_COMP_GNUC
  #define EIGEN_GNUC_AT_LEAST(x,y) ((__GNUC__==x && __GNUC_MINOR__>=y) || __GNUC__>x)
  #define EIGEN_GNUC_AT_MOST(x,y)  ((__GNUC__==x && __GNUC_MINOR__<=y) || __GNUC__<x)
  #define EIGEN_GNUC_AT(x,y)       ( __GNUC__==x && __GNUC_MINOR__==y )
#else
  #define EIGEN_GNUC_AT_LEAST(x,y) 0
  #define EIGEN_GNUC_AT_MOST(x,y)  0
  #define EIGEN_GNUC_AT(x,y)       0
#endif

// FIXME: could probably be removed as we do not support gcc 3.x anymore
#if EIGEN_COMP_GNUC && (__GNUC__ <= 3)
#define EIGEN_GCC3_OR_OLDER 1
#else
#define EIGEN_GCC3_OR_OLDER 0
#endif


// Architecture identification, EIGEN_ARCH_*

#if defined(__x86_64__) || defined(_M_X64) || defined(__amd64)
  #define EIGEN_ARCH_x86_64 1
#else
  #define EIGEN_ARCH_x86_64 0
#endif

#if defined(__i386__) || defined(_M_IX86) || defined(_X86_) || defined(__i386)
  #define EIGEN_ARCH_i386 1
#else
  #define EIGEN_ARCH_i386 0
#endif

#if EIGEN_ARCH_x86_64 || EIGEN_ARCH_i386
  #define EIGEN_ARCH_i386_OR_x86_64 1
#else
  #define EIGEN_ARCH_i386_OR_x86_64 0
#endif

/// \internal EIGEN_ARCH_ARM set to 1 if the architecture is ARM
#if defined(__arm__)
  #define EIGEN_ARCH_ARM 1
#else
  #define EIGEN_ARCH_ARM 0
#endif

/// \internal EIGEN_ARCH_ARM64 set to 1 if the architecture is ARM64
#if defined(__aarch64__)
  #define EIGEN_ARCH_ARM64 1
#else
  #define EIGEN_ARCH_ARM64 0
#endif

#if EIGEN_ARCH_ARM || EIGEN_ARCH_ARM64
  #define EIGEN_ARCH_ARM_OR_ARM64 1
#else
  #define EIGEN_ARCH_ARM_OR_ARM64 0
#endif

/// \internal EIGEN_ARCH_MIPS set to 1 if the architecture is MIPS
#if defined(__mips__) || defined(__mips)
  #define EIGEN_ARCH_MIPS 1
#else
  #define EIGEN_ARCH_MIPS 0
#endif

/// \internal EIGEN_ARCH_SPARC set to 1 if the architecture is SPARC
#if defined(__sparc__) || defined(__sparc)
  #define EIGEN_ARCH_SPARC 1
#else
  #define EIGEN_ARCH_SPARC 0
#endif

/// \internal EIGEN_ARCH_IA64 set to 1 if the architecture is Intel Itanium
#if defined(__ia64__)
  #define EIGEN_ARCH_IA64 1
#else
  #define EIGEN_ARCH_IA64 0
#endif

/// \internal EIGEN_ARCH_PPC set to 1 if the architecture is PowerPC
#if defined(__powerpc__) || defined(__ppc__) || defined(_M_PPC)
  #define EIGEN_ARCH_PPC 1
#else
  #define EIGEN_ARCH_PPC 0
#endif



// Operating system identification, EIGEN_OS_*

/// \internal EIGEN_OS_UNIX set to 1 if the OS is a unix variant
#if defined(__unix__) || defined(__unix)
  #define EIGEN_OS_UNIX 1
#else
  #define EIGEN_OS_UNIX 0
#endif

/// \internal EIGEN_OS_LINUX set to 1 if the OS is based on Linux kernel
#if defined(__linux__)
  #define EIGEN_OS_LINUX 1
#else
  #define EIGEN_OS_LINUX 0
#endif

/// \internal EIGEN_OS_ANDROID set to 1 if the OS is Android
// note: ANDROID is defined when using ndk_build, __ANDROID__ is defined when using a standalone toolchain.
#if defined(__ANDROID__) || defined(ANDROID)
  #define EIGEN_OS_ANDROID 1
#else
  #define EIGEN_OS_ANDROID 0
#endif

/// \internal EIGEN_OS_GNULINUX set to 1 if the OS is GNU Linux and not Linux-based OS (e.g., not android)
#if defined(__gnu_linux__) && !(EIGEN_OS_ANDROID)
  #define EIGEN_OS_GNULINUX 1
#else
  #define EIGEN_OS_GNULINUX 0
#endif

/// \internal EIGEN_OS_BSD set to 1 if the OS is a BSD variant
#if defined(__FreeBSD__) || defined(__NetBSD__) || defined(__OpenBSD__) || defined(__bsdi__) || defined(__DragonFly__)
  #define EIGEN_OS_BSD 1
#else
  #define EIGEN_OS_BSD 0
#endif

/// \internal EIGEN_OS_MAC set to 1 if the OS is MacOS
#if defined(__APPLE__)
  #define EIGEN_OS_MAC 1
#else
  #define EIGEN_OS_MAC 0
#endif

/// \internal EIGEN_OS_QNX set to 1 if the OS is QNX
#if defined(__QNX__)
  #define EIGEN_OS_QNX 1
#else
  #define EIGEN_OS_QNX 0
#endif

/// \internal EIGEN_OS_WIN set to 1 if the OS is Windows based
#if defined(_WIN32)
  #define EIGEN_OS_WIN 1
#else
  #define EIGEN_OS_WIN 0
#endif

/// \internal EIGEN_OS_WIN64 set to 1 if the OS is Windows 64bits
#if defined(_WIN64)
  #define EIGEN_OS_WIN64 1
#else
  #define EIGEN_OS_WIN64 0
#endif

/// \internal EIGEN_OS_WINCE set to 1 if the OS is Windows CE
#if defined(_WIN32_WCE)
  #define EIGEN_OS_WINCE 1
#else
  #define EIGEN_OS_WINCE 0
#endif

/// \internal EIGEN_OS_CYGWIN set to 1 if the OS is Windows/Cygwin
#if defined(__CYGWIN__)
  #define EIGEN_OS_CYGWIN 1
#else
  #define EIGEN_OS_CYGWIN 0
#endif

/// \internal EIGEN_OS_WIN_STRICT set to 1 if the OS is really Windows and not some variants
#if EIGEN_OS_WIN && !( EIGEN_OS_WINCE || EIGEN_OS_CYGWIN )
  #define EIGEN_OS_WIN_STRICT 1
#else
  #define EIGEN_OS_WIN_STRICT 0
#endif




#if EIGEN_GNUC_AT_MOST(4,3) && !EIGEN_COMP_CLANG
  // see bug 89
  #define EIGEN_SAFE_TO_USE_STANDARD_ASSERT_MACRO 0
#else
  #define EIGEN_SAFE_TO_USE_STANDARD_ASSERT_MACRO 1
#endif

// 16 byte alignment is only useful for vectorization. Since it affects the ABI, we need to enable
// 16 byte alignment on all platforms where vectorization might be enabled. In theory we could always
// enable alignment, but it can be a cause of problems on some platforms, so we just disable it in
// certain common platform (compiler+architecture combinations) to avoid these problems.
// Only static alignment is really problematic (relies on nonstandard compiler extensions that don't
// work everywhere, for example don't work on GCC/ARM), try to keep heap alignment even
// when we have to disable static alignment.
#if EIGEN_COMP_GNUC && !(EIGEN_ARCH_i386_OR_x86_64 || EIGEN_ARCH_PPC || EIGEN_ARCH_IA64)
#define EIGEN_GCC_AND_ARCH_DOESNT_WANT_STACK_ALIGNMENT 1
#else
#define EIGEN_GCC_AND_ARCH_DOESNT_WANT_STACK_ALIGNMENT 0
#endif

// static alignment is completely disabled with GCC 3, Sun Studio, and QCC/QNX
#if !EIGEN_GCC_AND_ARCH_DOESNT_WANT_STACK_ALIGNMENT \
 && !EIGEN_GCC3_OR_OLDER \
 && !EIGEN_COMP_SUNCC \
 && !EIGEN_OS_QNX
  #define EIGEN_ARCH_WANTS_STACK_ALIGNMENT 1
#else
  #define EIGEN_ARCH_WANTS_STACK_ALIGNMENT 0
#endif

// Defined the boundary (in bytes) on which the data needs to be aligned. Note
// that unless EIGEN_ALIGN is defined and not equal to 0, the data may not be
// aligned at all regardless of the value of this #define.
#define EIGEN_ALIGN_BYTES 16

#ifdef EIGEN_DONT_ALIGN
  #ifndef EIGEN_DONT_ALIGN_STATICALLY
    #define EIGEN_DONT_ALIGN_STATICALLY
  #endif
  #define EIGEN_ALIGN 0
#elif !defined(EIGEN_DONT_VECTORIZE)
  #if defined(__AVX__)
    #undef EIGEN_ALIGN_BYTES
    #define EIGEN_ALIGN_BYTES 32
  #endif
  #define EIGEN_ALIGN 1
#else
  #define EIGEN_ALIGN 0
#endif

#define EIGEN_MAX_ALIGN_BYTES EIGEN_ALIGN_BYTES


// This macro can be used to prevent from macro expansion, e.g.:
//   std::max EIGEN_NOT_A_MACRO(a,b)
#define EIGEN_NOT_A_MACRO

// EIGEN_ALIGN_STATICALLY is the true test whether we want to align arrays on the stack or not. It takes into account both the user choice to explicitly disable
// alignment (EIGEN_DONT_ALIGN_STATICALLY) and the architecture config (EIGEN_ARCH_WANTS_STACK_ALIGNMENT). Henceforth, only EIGEN_ALIGN_STATICALLY should be used.
#if EIGEN_ARCH_WANTS_STACK_ALIGNMENT && !defined(EIGEN_DONT_ALIGN_STATICALLY)
  #define EIGEN_ALIGN_STATICALLY 1
#else
  #define EIGEN_ALIGN_STATICALLY 0
  #ifndef EIGEN_DISABLE_UNALIGNED_ARRAY_ASSERT
    #define EIGEN_DISABLE_UNALIGNED_ARRAY_ASSERT
  #endif
#endif

#ifdef EIGEN_DEFAULT_TO_ROW_MAJOR
#define EIGEN_DEFAULT_MATRIX_STORAGE_ORDER_OPTION Eigen::RowMajor
#else
#define EIGEN_DEFAULT_MATRIX_STORAGE_ORDER_OPTION Eigen::ColMajor
#endif

#ifndef EIGEN_DEFAULT_DENSE_INDEX_TYPE
#define EIGEN_DEFAULT_DENSE_INDEX_TYPE std::ptrdiff_t
#endif

// Cross compiler wrapper around LLVM's __has_builtin
#ifdef __has_builtin
#  define EIGEN_HAS_BUILTIN(x) __has_builtin(x)
#else
#  define EIGEN_HAS_BUILTIN(x) 0
#endif

// A Clang feature extension to determine compiler features.
// We use it to determine 'cxx_rvalue_references'
#ifndef __has_feature
# define __has_feature(x) 0
#endif

#if __cplusplus > 199711L
#define EIGEN_HAS_VARIADIC_TEMPLATES 1
#endif

// Does the compiler support const expressions?
#if __cplusplus > 199711L && !defined(__NVCC__) && !defined(GOOGLE_LIBCXX) && !defined(__APPLE__)
#define EIGEN_HAS_CONSTEXPR 1
#endif

/** Allows to disable some optimizations which might affect the accuracy of the result.
  * Such optimization are enabled by default, and set EIGEN_FAST_MATH to 0 to disable them.
  * They currently include:
  *   - single precision Cwise::sin() and Cwise::cos() when SSE vectorization is enabled.
  */
#ifndef EIGEN_FAST_MATH
#define EIGEN_FAST_MATH 1
#endif

#define EIGEN_DEBUG_VAR(x) std::cerr << #x << " = " << x << std::endl;

// concatenate two tokens
#define EIGEN_CAT2(a,b) a ## b
#define EIGEN_CAT(a,b) EIGEN_CAT2(a,b)

// convert a token to a string
#define EIGEN_MAKESTRING2(a) #a
#define EIGEN_MAKESTRING(a) EIGEN_MAKESTRING2(a)

// EIGEN_STRONG_INLINE is a stronger version of the inline, using __forceinline on MSVC,
// but it still doesn't use GCC's always_inline. This is useful in (common) situations where MSVC needs forceinline
// but GCC is still doing fine with just inline.
#if EIGEN_COMP_MSVC || EIGEN_COMP_ICC
#define EIGEN_STRONG_INLINE __forceinline
#else
#define EIGEN_STRONG_INLINE inline
#endif

// EIGEN_ALWAYS_INLINE is the stronget, it has the effect of making the function inline and adding every possible
// attribute to maximize inlining. This should only be used when really necessary: in particular,
// it uses __attribute__((always_inline)) on GCC, which most of the time is useless and can severely harm compile times.
// FIXME with the always_inline attribute,
// gcc 3.4.x reports the following compilation error:
//   Eval.h:91: sorry, unimplemented: inlining failed in call to 'const Eigen::Eval<Derived> Eigen::MatrixBase<Scalar, Derived>::eval() const'
//    : function body not available
#if EIGEN_GNUC_AT_LEAST(4,0)
#define EIGEN_ALWAYS_INLINE __attribute__((always_inline)) inline
#else
#define EIGEN_ALWAYS_INLINE EIGEN_STRONG_INLINE
#endif

#if EIGEN_COMP_GNUC
#define EIGEN_DONT_INLINE __attribute__((noinline))
#elif EIGEN_COMP_MSVC
#define EIGEN_DONT_INLINE __declspec(noinline)
#else
#define EIGEN_DONT_INLINE
#endif

#if EIGEN_COMP_GNUC
#define EIGEN_PERMISSIVE_EXPR __extension__
#else
#define EIGEN_PERMISSIVE_EXPR
#endif

#if EIGEN_COMP_GNUC
#define EIGEN_LIKELY(x) __builtin_expect((x), 1)
#define EIGEN_UNLIKELY(x) __builtin_expect((x), 0)
#else
#define EIGEN_LIKELY(x) (x)
#define EIGEN_UNLIKELY(x) (x)
#endif

// this macro allows to get rid of linking errors about multiply defined functions.
//  - static is not very good because it prevents definitions from different object files to be merged.
//           So static causes the resulting linked executable to be bloated with multiple copies of the same function.
//  - inline is not perfect either as it unwantedly hints the compiler toward inlining the function.
#define EIGEN_DECLARE_FUNCTION_ALLOWING_MULTIPLE_DEFINITIONS
#define EIGEN_DEFINE_FUNCTION_ALLOWING_MULTIPLE_DEFINITIONS inline

#ifdef NDEBUG
# ifndef EIGEN_NO_DEBUG
#  define EIGEN_NO_DEBUG
# endif
#endif

#if !defined(EIGEN_NO_CHECK) || (!defined(EIGEN_NO_DEBUG) && !EIGEN_SAFE_TO_USE_STANDARD_ASSERT_MACRO)
  // Custom assertion code that works regardless of the compilation mode.
  #include <cstdlib>   // for abort
  #include <iostream>  // for std::cerr

  namespace Eigen {
  namespace internal {
  // trivial function copying a bool. Must be EIGEN_DONT_INLINE, so we implement it after including Eigen headers.
  // see bug 89.
  namespace {
  EIGEN_DONT_INLINE bool copy_bool(bool b) { return b; }
  }
  inline void assert_fail(const char *condition, const char *function, const char *file, int line)
  {
    copy_bool(true);  // dummy call to avoid warnings about unused functions.
    std::cerr << "assertion failed: " << condition << " in function " << function << " at " << file << ":" << line << std::endl;
    abort();
  }
  }
  }
  #define eigen_internal_check(x) \
    do { \
      if(!Eigen::internal::copy_bool(x)) \
        Eigen::internal::assert_fail(EIGEN_MAKESTRING(x), __PRETTY_FUNCTION__, __FILE__, __LINE__); \
    } while(false)
#endif

#ifdef EIGEN_NO_CHECK
  #define eigen_check(x)
#else
  #define eigen_check(x) eigen_internal_check(x)
#endif

// eigen_plain_assert is where we implement the workaround for the assert() bug in GCC <= 4.3, see bug 89
#ifdef EIGEN_NO_DEBUG
  #define eigen_plain_assert(x)
#else
  #if EIGEN_SAFE_TO_USE_STANDARD_ASSERT_MACRO
    namespace Eigen {
    namespace internal {
    inline bool copy_bool(bool b) { return b; }
    }
    }
    #define eigen_plain_assert(x) assert(x)
  #else
    // work around bug 89
    #define eigen_plain_assert(x) eigen_internal_check(x)
  #endif
#endif

// eigen_assert can be overridden
#ifndef eigen_assert
#define eigen_assert(x) eigen_plain_assert(x)
#endif

#ifdef EIGEN_INTERNAL_DEBUGGING
#define eigen_internal_assert(x) eigen_assert(x)
#else
#define eigen_internal_assert(x)
#endif

#ifdef EIGEN_NO_DEBUG
#define EIGEN_ONLY_USED_FOR_DEBUG(x) (void)x
#else
#define EIGEN_ONLY_USED_FOR_DEBUG(x)
#endif

#ifndef EIGEN_NO_DEPRECATED_WARNING
  #if EIGEN_COMP_GNUC
    #define EIGEN_DEPRECATED __attribute__((deprecated))
  #elif (defined _MSC_VER)
    #define EIGEN_DEPRECATED __declspec(deprecated)
  #else
    #define EIGEN_DEPRECATED
  #endif
#else
  #define EIGEN_DEPRECATED
#endif

#if EIGEN_COMP_GNUC
#define EIGEN_UNUSED __attribute__((unused))
#else
#define EIGEN_UNUSED
#endif

// Suppresses 'unused variable' warnings.
namespace Eigen {
  namespace internal {
    template<typename T> void ignore_unused_variable(const T&) {}
  }
}
#define EIGEN_UNUSED_VARIABLE(var) Eigen::internal::ignore_unused_variable(var);

#if !defined(EIGEN_ASM_COMMENT)
  #if EIGEN_COMP_GNUC && (EIGEN_ARCH_i386_OR_x86_64 || EIGEN_ARCH_ARM_OR_ARM64)
    #define EIGEN_ASM_COMMENT(X)  asm("#" X)
  #else
    #define EIGEN_ASM_COMMENT(X)
  #endif
#endif

/* EIGEN_ALIGN_TO_BOUNDARY(n) forces data to be n-byte aligned. This is used to satisfy SIMD requirements.
 * However, we do that EVEN if vectorization (EIGEN_VECTORIZE) is disabled,
 * so that vectorization doesn't affect binary compatibility.
 *
 * If we made alignment depend on whether or not EIGEN_VECTORIZE is defined, it would be impossible to link
 * vectorized and non-vectorized code.
 */
#if (defined __CUDACC__)
  #define EIGEN_ALIGN_TO_BOUNDARY(n) __align__(n)
#elif EIGEN_COMP_GNUC || EIGEN_COMP_PGI || EIGEN_COMP_IBM || EIGEN_COMP_ARM
  #define EIGEN_ALIGN_TO_BOUNDARY(n) __attribute__((aligned(n)))
#elif EIGEN_COMP_MSVC
  #define EIGEN_ALIGN_TO_BOUNDARY(n) __declspec(align(n))
#elif EIGEN_COMP_SUNCC
  // FIXME not sure about this one:
  #define EIGEN_ALIGN_TO_BOUNDARY(n) __attribute__((aligned(n)))
#else
  #error Please tell me what is the equivalent of __attribute__((aligned(n))) for your compiler
#endif

#define EIGEN_ALIGN16 EIGEN_ALIGN_TO_BOUNDARY(16)
#define EIGEN_ALIGN32 EIGEN_ALIGN_TO_BOUNDARY(32)
#define EIGEN_ALIGN_DEFAULT EIGEN_ALIGN_TO_BOUNDARY(EIGEN_ALIGN_BYTES)
#define EIGEN_ALIGN_MAX EIGEN_ALIGN_DEFAULT

#if EIGEN_ALIGN_STATICALLY
#define EIGEN_USER_ALIGN_TO_BOUNDARY(n) EIGEN_ALIGN_TO_BOUNDARY(n)
#define EIGEN_USER_ALIGN16 EIGEN_ALIGN16
#define EIGEN_USER_ALIGN32 EIGEN_ALIGN32
#define EIGEN_USER_ALIGN_DEFAULT EIGEN_ALIGN_DEFAULT
#else
#define EIGEN_USER_ALIGN_TO_BOUNDARY(n)
#define EIGEN_USER_ALIGN16
#define EIGEN_USER_ALIGN32
#define EIGEN_USER_ALIGN_DEFAULT
#endif

#ifdef EIGEN_DONT_USE_RESTRICT_KEYWORD
  #define EIGEN_RESTRICT
#endif
#ifndef EIGEN_RESTRICT
  #define EIGEN_RESTRICT __restrict
#endif

#ifndef EIGEN_STACK_ALLOCATION_LIMIT
#define EIGEN_STACK_ALLOCATION_LIMIT 20000
#endif

#ifndef EIGEN_DEFAULT_IO_FORMAT
#ifdef EIGEN_MAKING_DOCS
// format used in Eigen's documentation
// needed to define it here as escaping characters in CMake add_definition's argument seems very problematic.
#define EIGEN_DEFAULT_IO_FORMAT Eigen::IOFormat(3, 0, " ", "\n", "", "")
#else
#define EIGEN_DEFAULT_IO_FORMAT Eigen::IOFormat()
#endif
#endif

// just an empty macro !
#define EIGEN_EMPTY

#if EIGEN_COMP_MSVC_STRICT
  #define EIGEN_INHERIT_ASSIGNMENT_EQUAL_OPERATOR(Derived) \
    using Base::operator =;
#elif EIGEN_COMP_CLANG // workaround clang bug (see http://forum.kde.org/viewtopic.php?f=74&t=102653)
  #define EIGEN_INHERIT_ASSIGNMENT_EQUAL_OPERATOR(Derived) \
    using Base::operator =; \
    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Derived& operator=(const Derived& other) { Base::operator=(other); return *this; } \
    template <typename OtherDerived> \
    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Derived& operator=(const DenseBase<OtherDerived>& other) { Base::operator=(other.derived()); return *this; }
#else
  #define EIGEN_INHERIT_ASSIGNMENT_EQUAL_OPERATOR(Derived) \
    using Base::operator =; \
    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Derived& operator=(const Derived& other) \
    { \
      Base::operator=(other); \
      return *this; \
    }
#endif

#define EIGEN_INHERIT_ASSIGNMENT_OPERATORS(Derived) EIGEN_INHERIT_ASSIGNMENT_EQUAL_OPERATOR(Derived)

/**
* Just a side note. Commenting within defines works only by documenting
* behind the object (via '!<'). Comments cannot be multi-line and thus
* we have these extra long lines. What is confusing doxygen over here is
* that we use '\' and basically have a bunch of typedefs with their
* documentation in a single line.
**/

#define EIGEN_GENERIC_PUBLIC_INTERFACE(Derived) \
  typedef typename Eigen::internal::traits<Derived>::Scalar Scalar; /*!< \brief Numeric type, e.g. float, double, int or std::complex<float>. */ \
  typedef typename Eigen::NumTraits<Scalar>::Real RealScalar; /*!< \brief The underlying numeric type for composed scalar types. \details In cases where Scalar is e.g. std::complex<T>, T were corresponding to RealScalar. */ \
  typedef typename Base::CoeffReturnType CoeffReturnType; /*!< \brief The return type for coefficient access. \details Depending on whether the object allows direct coefficient access (e.g. for a MatrixXd), this type is either 'const Scalar&' or simply 'Scalar' for objects that do not allow direct coefficient access. */ \
  typedef typename Eigen::internal::nested<Derived>::type Nested; \
  typedef typename Eigen::internal::traits<Derived>::StorageKind StorageKind; \
  typedef typename Eigen::internal::traits<Derived>::Index Index; \
  enum { RowsAtCompileTime = Eigen::internal::traits<Derived>::RowsAtCompileTime, \
        ColsAtCompileTime = Eigen::internal::traits<Derived>::ColsAtCompileTime, \
        Flags = Eigen::internal::traits<Derived>::Flags, \
        CoeffReadCost = Eigen::internal::traits<Derived>::CoeffReadCost, \
        SizeAtCompileTime = Base::SizeAtCompileTime, \
        MaxSizeAtCompileTime = Base::MaxSizeAtCompileTime, \
        IsVectorAtCompileTime = Base::IsVectorAtCompileTime };


#define EIGEN_DENSE_PUBLIC_INTERFACE(Derived) \
  typedef typename Eigen::internal::traits<Derived>::Scalar Scalar; /*!< \brief Numeric type, e.g. float, double, int or std::complex<float>. */ \
  typedef typename Eigen::NumTraits<Scalar>::Real RealScalar; /*!< \brief The underlying numeric type for composed scalar types. \details In cases where Scalar is e.g. std::complex<T>, T were corresponding to RealScalar. */ \
  typedef typename Base::PacketScalar PacketScalar; \
  typedef typename Base::CoeffReturnType CoeffReturnType; /*!< \brief The return type for coefficient access. \details Depending on whether the object allows direct coefficient access (e.g. for a MatrixXd), this type is either 'const Scalar&' or simply 'Scalar' for objects that do not allow direct coefficient access. */ \
  typedef typename Eigen::internal::nested<Derived>::type Nested; \
  typedef typename Eigen::internal::traits<Derived>::StorageKind StorageKind; \
  typedef typename Eigen::internal::traits<Derived>::Index Index; \
  enum { RowsAtCompileTime = Eigen::internal::traits<Derived>::RowsAtCompileTime, \
        ColsAtCompileTime = Eigen::internal::traits<Derived>::ColsAtCompileTime, \
        MaxRowsAtCompileTime = Eigen::internal::traits<Derived>::MaxRowsAtCompileTime, \
        MaxColsAtCompileTime = Eigen::internal::traits<Derived>::MaxColsAtCompileTime, \
        Flags = Eigen::internal::traits<Derived>::Flags, \
        CoeffReadCost = Eigen::internal::traits<Derived>::CoeffReadCost, \
        SizeAtCompileTime = Base::SizeAtCompileTime, \
        MaxSizeAtCompileTime = Base::MaxSizeAtCompileTime, \
        IsVectorAtCompileTime = Base::IsVectorAtCompileTime }; \
  using Base::derived; \
  using Base::const_cast_derived;


#define EIGEN_PLAIN_ENUM_MIN(a,b) (((int)a <= (int)b) ? (int)a : (int)b)
#define EIGEN_PLAIN_ENUM_MAX(a,b) (((int)a >= (int)b) ? (int)a : (int)b)

// EIGEN_SIZE_MIN_PREFER_DYNAMIC gives the min between compile-time sizes. 0 has absolute priority, followed by 1,
// followed by Dynamic, followed by other finite values. The reason for giving Dynamic the priority over
// finite values is that min(3, Dynamic) should be Dynamic, since that could be anything between 0 and 3.
#define EIGEN_SIZE_MIN_PREFER_DYNAMIC(a,b) (((int)a == 0 || (int)b == 0) ? 0 \
                           : ((int)a == 1 || (int)b == 1) ? 1 \
                           : ((int)a == Dynamic || (int)b == Dynamic) ? Dynamic \
                           : ((int)a <= (int)b) ? (int)a : (int)b)

// EIGEN_SIZE_MIN_PREFER_FIXED is a variant of EIGEN_SIZE_MIN_PREFER_DYNAMIC comparing MaxSizes. The difference is that finite values
// now have priority over Dynamic, so that min(3, Dynamic) gives 3. Indeed, whatever the actual value is
// (between 0 and 3), it is not more than 3.
#define EIGEN_SIZE_MIN_PREFER_FIXED(a,b)  (((int)a == 0 || (int)b == 0) ? 0 \
                           : ((int)a == 1 || (int)b == 1) ? 1 \
                           : ((int)a == Dynamic && (int)b == Dynamic) ? Dynamic \
                           : ((int)a == Dynamic) ? (int)b \
                           : ((int)b == Dynamic) ? (int)a \
                           : ((int)a <= (int)b) ? (int)a : (int)b)

// see EIGEN_SIZE_MIN_PREFER_DYNAMIC. No need for a separate variant for MaxSizes here.
#define EIGEN_SIZE_MAX(a,b) (((int)a == Dynamic || (int)b == Dynamic) ? Dynamic \
                           : ((int)a >= (int)b) ? (int)a : (int)b)

#define EIGEN_LOGICAL_XOR(a,b) (((a) || (b)) && !((a) && (b)))

#define EIGEN_IMPLIES(a,b) (!(a) || (b))

#define EIGEN_MAKE_CWISE_BINARY_OP(METHOD,FUNCTOR) \
  template<typename OtherDerived> \
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const CwiseBinaryOp<FUNCTOR<Scalar>, const Derived, const OtherDerived> \
  (METHOD)(const EIGEN_CURRENT_STORAGE_BASE_CLASS<OtherDerived> &other) const \
  { \
    return CwiseBinaryOp<FUNCTOR<Scalar>, const Derived, const OtherDerived>(derived(), other.derived()); \
  }

// the expression type of a cwise product
#define EIGEN_CWISE_PRODUCT_RETURN_TYPE(LHS,RHS) \
    CwiseBinaryOp< \
      internal::scalar_product_op< \
          typename internal::traits<LHS>::Scalar, \
          typename internal::traits<RHS>::Scalar \
      >, \
      const LHS, \
      const RHS \
    >

#endif // EIGEN_MACROS_H
