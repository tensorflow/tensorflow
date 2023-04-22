/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef TENSORFLOW_CORE_PLATFORM_PLATFORM_STRINGS_H_
#define TENSORFLOW_CORE_PLATFORM_PLATFORM_STRINGS_H_

// This header defines the macro TF_PLATFORM_STRINGS() which should be used
// once in each dynamically loadable TensorFlow module.  It embeds static
// strings into the compilation unit that allow TensorFlow to determine what
// compilation options were in effect when the compilation unit was built.  All
// compilation units within the same dynamically loadable library should be
// built with the same options (or at least, the strings should be embedded in
// the compilation unit built with the most restrictive options).

// The platform strings embedded into a binary may be retrieved with the
// GetPlatformStrings function.

// Rationale:
// We wish to load only those libraries that this CPU can execute.  For
// example, we should not load a library compiled with avx256 instructions on a
// CPU that cannot execute them.
//
// One might think that one could dlopen() the library, and call a routine that
// would return which cpu type it was compiled for.  Alas, this does not work,
// because at dlopen() time, a library containing C++ will execute constructors
// of class variables with static storage class.  Even code that looks
// innocuous may use optional platform-specific instructions.  For example,
// the fastest way to zero a region of memory might use optional instructions.
//
// One might think one could run a tool such as "objdump" to read flags from
// the libraries' headers, or perhaps disassemble each library to look for
// particular instructions.  Unfortunately, the desired flags are not present
// in the headers, and disassembly can be prohibitively slow ("objdump -d" is
// very slow, for example).  Moreover, a tool to examine the library may not
// be present on the system unless the user has installed special packages (for
// example, on Windows).
//
// Instead, we adopt a crude but straightforward solution:  We require
// developers to use the macro TF_PLATFORM_STRINGS() in their library, to
// embed the compilation options as constant strings.  The compiler's
// predefined macros pick which strings are included.  We then search for the
// strings in the files, and then dlopen() only those libraries that have or
// lack strings as needed.
//
// We adopt the approach of placing in the binary a fairly raw copy of the
// predefined macros, rather than trying to interpret them in complex ways at
// compile time.  This allows the loading binary to alter its interpretation of
// the strings without library developers having to recompile.

#include <stdio.h>

#include <string>
#include <vector>

// Aside from the header guard, the internal macros defined here have the form:
//   TF_PLAT_STR_*

// If a macro is removed from the list of tested macros, the major version in
// the following version number should be incremented, and the minor version
// set to zero.  Otherwise, if a macro is added to the list of tested macros,
// the minor number should be incremented.
#define TF_PLAT_STR_VERSION_ "1.0"

// Prefix of each option string indicator in the binary.
// After the prefix, such strings have the form:
//    [A-Za-z_0-9]=<value>
// followed by a terminating nul.  To simplify searching, this prefix is all
// ASCII, starts with a nul, and contains no character twice.
#define TF_PLAT_STR_MAGIC_PREFIX_ "\0S\\s\":^p*L}"

// A helper macro for TF_PLAT_STR_AS_STR_().
#define TF_PLAT_STR_STR_1_(x) #x

// Yield a constant string corresponding to x, after macro expansion.
#define TF_PLAT_STR_AS_STR_(x) TF_PLAT_STR_STR_1_(x)

// An empty definition to make lists more uniform.
#define TF_PLAT_STR_TERMINATOR_

// TF_PLAT_STR_(x) introduces a constant string indicating whether a
// particular compilation option has been turned on.
//
// In gcc and clang, we might imagine using something like
// #define TF_PLAT_STR_(x) \
//     (sizeof (#x) != sizeof (TF_PLAT_STR_AS_STR_ (x))? \
//      TF_PLAT_STR_MAGIC_PREFIX_ #x "=" TF_PLAT_STR_AS_STR_ (x) : \
//      TF_PLAT_STR_MAGIC_PREFIX_ #x "=0"),
// but some compilers (notably MSVC) place both "foo" and "bar" in the binary
// when presented with
//    (true?  "foo" : "bar")
// so we must use #if to select the strings we need, which is rather verbose.
#define TF_PLAT_STR_(x) TF_PLAT_STR_MAGIC_PREFIX_ #x "=" TF_PLAT_STR_AS_STR_(x)

// Include the #if machinery that sets the macros used below.
// platform_strings_computed.h can be generated by filtering this header file
// through:
// awk '
// header == "" { print; }
// /\*\// && header == "" {
//     print "// Generated from platform_strings.h.";
//     print "";
//     print "#ifndef TENSORFLOW_CORE_PLATFORM_PLATFORM_STRINGS_COMPUTED_H_";
//     print "#define TENSORFLOW_CORE_PLATFORM_PLATFORM_STRINGS_COMPUTED_H_";
//     print "";
//     header = 1;
// }
// /^#define TF_PLAT_STR_LIST_[a-zA-Z0-9_]*\(\) *\\$/ { active = 1; }
// /TF_PLAT_STR_TERMINATOR_/ { active = 0; }
// /^ *TF_PLAT_STR_[A-Za-z0-9_]* *\\$/ && active {
//     x = $0;
//     sub(/^ *TF_PLAT_STR_/, "", x);
//     sub(/ *\\$/, "", x);
//     printf ("#if defined(%s)\n", x);
//     printf ("#define TF_PLAT_STR_%s TF_PLAT_STR_(%s)\n", x, x);
//     printf ("#else\n");
//     printf ("#define TF_PLAT_STR_%s\n", x);
//     printf ("#endif\n");
// }
// END {
//     print "";
//     print "#endif  // TENSORFLOW_CORE_PLATFORM_PLATFORM_STRINGS_COMPUTED_H_";
// }'
#include "tensorflow/core/platform/platform_strings_computed.h"

// clang-format butchers the following lines.
// clang-format off

// x86_64 and x86_32 optional features.
#define TF_PLAT_STR_LIST___x86_64__()                                      \
        TF_PLAT_STR__M_IX86_FP                                             \
        TF_PLAT_STR__NO_PREFETCHW                                          \
        TF_PLAT_STR___3dNOW_A__                                            \
        TF_PLAT_STR___3dNOW__                                              \
        TF_PLAT_STR___ABM__                                                \
        TF_PLAT_STR___ADX__                                                \
        TF_PLAT_STR___AES__                                                \
        TF_PLAT_STR___AVX2__                                               \
        TF_PLAT_STR___AVX512BW__                                           \
        TF_PLAT_STR___AVX512CD__                                           \
        TF_PLAT_STR___AVX512DQ__                                           \
        TF_PLAT_STR___AVX512ER__                                           \
        TF_PLAT_STR___AVX512F__                                            \
        TF_PLAT_STR___AVX512IFMA__                                         \
        TF_PLAT_STR___AVX512PF__                                           \
        TF_PLAT_STR___AVX512VBMI__                                         \
        TF_PLAT_STR___AVX512VL__                                           \
        TF_PLAT_STR___AVX__                                                \
        TF_PLAT_STR___BMI2__                                               \
        TF_PLAT_STR___BMI__                                                \
        TF_PLAT_STR___CLFLUSHOPT__                                         \
        TF_PLAT_STR___CLZERO__                                             \
        TF_PLAT_STR___F16C__                                               \
        TF_PLAT_STR___FMA4__                                               \
        TF_PLAT_STR___FMA__                                                \
        TF_PLAT_STR___FP_FAST_FMA                                          \
        TF_PLAT_STR___FP_FAST_FMAF                                         \
        TF_PLAT_STR___FSGSBASE__                                           \
        TF_PLAT_STR___FXSR__                                               \
        TF_PLAT_STR___LWP__                                                \
        TF_PLAT_STR___LZCNT__                                              \
        TF_PLAT_STR___MMX__                                                \
        TF_PLAT_STR___MWAITX__                                             \
        TF_PLAT_STR___PCLMUL__                                             \
        TF_PLAT_STR___PKU__                                                \
        TF_PLAT_STR___POPCNT__                                             \
        TF_PLAT_STR___PRFCHW__                                             \
        TF_PLAT_STR___RDRND__                                              \
        TF_PLAT_STR___RDSEED__                                             \
        TF_PLAT_STR___RTM__                                                \
        TF_PLAT_STR___SHA__                                                \
        TF_PLAT_STR___SSE2_MATH__                                          \
        TF_PLAT_STR___SSE2__                                               \
        TF_PLAT_STR___SSE_MATH__                                           \
        TF_PLAT_STR___SSE__                                                \
        TF_PLAT_STR___SSE3__                                               \
        TF_PLAT_STR___SSE4A__                                              \
        TF_PLAT_STR___SSE4_1__                                             \
        TF_PLAT_STR___SSE4_2__                                             \
        TF_PLAT_STR___SSSE3__                                              \
        TF_PLAT_STR___TBM__                                                \
        TF_PLAT_STR___XOP__                                                \
        TF_PLAT_STR___XSAVEC__                                             \
        TF_PLAT_STR___XSAVEOPT__                                           \
        TF_PLAT_STR___XSAVES__                                             \
        TF_PLAT_STR___XSAVE__                                              \
        TF_PLAT_STR_TERMINATOR_

// PowerPC (64- and 32-bit) optional features.
#define TF_PLAT_STR_LIST___powerpc64__()                                   \
        TF_PLAT_STR__SOFT_DOUBLE                                           \
        TF_PLAT_STR__SOFT_FLOAT                                            \
        TF_PLAT_STR___ALTIVEC__                                            \
        TF_PLAT_STR___APPLE_ALTIVEC__                                      \
        TF_PLAT_STR___CRYPTO__                                             \
        TF_PLAT_STR___FLOAT128_HARDWARE__                                  \
        TF_PLAT_STR___FLOAT128_TYPE__                                      \
        TF_PLAT_STR___FP_FAST_FMA                                          \
        TF_PLAT_STR___FP_FAST_FMAF                                         \
        TF_PLAT_STR___HTM__                                                \
        TF_PLAT_STR___NO_FPRS__                                            \
        TF_PLAT_STR___NO_LWSYNC__                                          \
        TF_PLAT_STR___POWER8_VECTOR__                                      \
        TF_PLAT_STR___POWER9_VECTOR__                                      \
        TF_PLAT_STR___PPC405__                                             \
        TF_PLAT_STR___QUAD_MEMORY_ATOMIC__                                 \
        TF_PLAT_STR___RECIPF__                                             \
        TF_PLAT_STR___RECIP_PRECISION__                                    \
        TF_PLAT_STR___RECIP__                                              \
        TF_PLAT_STR___RSQRTEF__                                            \
        TF_PLAT_STR___RSQRTE__                                             \
        TF_PLAT_STR___TM_FENCE__                                           \
        TF_PLAT_STR___UPPER_REGS_DF__                                      \
        TF_PLAT_STR___UPPER_REGS_SF__                                      \
        TF_PLAT_STR___VEC__                                                \
        TF_PLAT_STR___VSX__                                                \
        TF_PLAT_STR_TERMINATOR_

// aarch64 and 32-bit arm optional features
#define TF_PLAT_STR_LIST___aarch64__()                                     \
        TF_PLAT_STR___ARM_ARCH                                             \
        TF_PLAT_STR___ARM_FEATURE_CLZ                                      \
        TF_PLAT_STR___ARM_FEATURE_CRC32                                    \
        TF_PLAT_STR___ARM_FEATURE_CRC32                                    \
        TF_PLAT_STR___ARM_FEATURE_CRYPTO                                   \
        TF_PLAT_STR___ARM_FEATURE_DIRECTED_ROUNDING                        \
        TF_PLAT_STR___ARM_FEATURE_DSP                                      \
        TF_PLAT_STR___ARM_FEATURE_FMA                                      \
        TF_PLAT_STR___ARM_FEATURE_IDIV                                     \
        TF_PLAT_STR___ARM_FEATURE_LDREX                                    \
        TF_PLAT_STR___ARM_FEATURE_NUMERIC_MAXMIN                           \
        TF_PLAT_STR___ARM_FEATURE_QBIT                                     \
        TF_PLAT_STR___ARM_FEATURE_QRDMX                                    \
        TF_PLAT_STR___ARM_FEATURE_SAT                                      \
        TF_PLAT_STR___ARM_FEATURE_SIMD32                                   \
        TF_PLAT_STR___ARM_FEATURE_UNALIGNED                                \
        TF_PLAT_STR___ARM_FP                                               \
        TF_PLAT_STR___ARM_NEON_FP                                          \
        TF_PLAT_STR___ARM_NEON__                                           \
        TF_PLAT_STR___ARM_WMMX                                             \
        TF_PLAT_STR___IWMMXT2__                                            \
        TF_PLAT_STR___IWMMXT__                                             \
        TF_PLAT_STR___VFP_FP__                                             \
        TF_PLAT_STR_TERMINATOR_

// Generic features, including indication of architecture and OS.
// The _M_* macros are defined by Visual Studio.
// It doesn't define __LITTLE_ENDIAN__ or __BYTE_ORDER__;
// Windows is assumed to be little endian.
#define TF_PLAT_STR_LIST___generic__()                                     \
        TF_PLAT_STR_TARGET_IPHONE_SIMULATOR                                \
        TF_PLAT_STR_TARGET_OS_IOS                                          \
        TF_PLAT_STR_TARGET_OS_IPHONE                                       \
        TF_PLAT_STR__MSC_VER                                               \
        TF_PLAT_STR__M_ARM                                                 \
        TF_PLAT_STR__M_ARM64                                               \
        TF_PLAT_STR__M_ARM_ARMV7VE                                         \
        TF_PLAT_STR__M_ARM_FP                                              \
        TF_PLAT_STR__M_IX86                                                \
        TF_PLAT_STR__M_X64                                                 \
        TF_PLAT_STR__WIN32                                                 \
        TF_PLAT_STR__WIN64                                                 \
        TF_PLAT_STR___ANDROID__                                            \
        TF_PLAT_STR___APPLE__                                              \
        TF_PLAT_STR___BYTE_ORDER__                                         \
        TF_PLAT_STR___CYGWIN__                                             \
        TF_PLAT_STR___FreeBSD__                                            \
        TF_PLAT_STR___LITTLE_ENDIAN__                                      \
        TF_PLAT_STR___NetBSD__                                             \
        TF_PLAT_STR___OpenBSD__                                            \
        TF_PLAT_STR_____MSYS__                                             \
        TF_PLAT_STR___aarch64__                                            \
        TF_PLAT_STR___alpha__                                              \
        TF_PLAT_STR___arm__                                                \
        TF_PLAT_STR___i386__                                               \
        TF_PLAT_STR___i686__                                               \
        TF_PLAT_STR___ia64__                                               \
        TF_PLAT_STR___linux__                                              \
        TF_PLAT_STR___mips32__                                             \
        TF_PLAT_STR___mips64__                                             \
        TF_PLAT_STR___powerpc64__                                          \
        TF_PLAT_STR___powerpc__                                            \
        TF_PLAT_STR___riscv___                                             \
        TF_PLAT_STR___s390x__                                              \
        TF_PLAT_STR___sparc64__                                            \
        TF_PLAT_STR___sparc__                                              \
        TF_PLAT_STR___x86_64__                                             \
        TF_PLAT_STR_TERMINATOR_

#if !defined(__x86_64__) && !defined(_M_X64) && \
    !defined(__i386__) && !defined(_M_IX86)
#undef TF_PLAT_STR_LIST___x86_64__
#define TF_PLAT_STR_LIST___x86_64__()
#endif
#if !defined(__powerpc64__) && !defined(__powerpc__)
#undef TF_PLAT_STR_LIST___powerpc64__
#define TF_PLAT_STR_LIST___powerpc64__()
#endif
#if !defined(__aarch64__) && !defined(_M_ARM64) && \
    !defined(__arm__) && !defined(_M_ARM)
#undef TF_PLAT_STR_LIST___aarch64__
#define TF_PLAT_STR_LIST___aarch64__()
#endif

// Macro to be used in each dynamically loadable library.
//
// The BSS global variable tf_cpu_option_global and the class
// instance tf_cpu_option_avoid_omit_class are needed to prevent
// compilers/linkers such as clang from omitting the static variable
// tf_cpu_option[], which would otherwise appear to be unused.  We cannot make
// tf_cpu_option[] global, because we then might get multiply-defined symbols
// if TF_PLAT_STR() is used twice in the same library.
// (tf_cpu_option_global doesn't see such errors because it is
// defined in BSS, so multiple definitions are combined by the linker.)  gcc's
// __attribute__((used)) is insufficient because it seems to be ignored by
// linkers.
#define TF_PLATFORM_STRINGS()                                                  \
    static const char tf_cpu_option[] =                                        \
        TF_PLAT_STR_MAGIC_PREFIX_ "TF_PLAT_STR_VERSION=" TF_PLAT_STR_VERSION_  \
        TF_PLAT_STR_LIST___x86_64__()                                          \
        TF_PLAT_STR_LIST___powerpc64__()                                       \
        TF_PLAT_STR_LIST___aarch64__()                                         \
        TF_PLAT_STR_LIST___generic__()                                         \
    ;                                                                          \
    const char *tf_cpu_option_global;                                          \
    namespace {                                                                \
    class TFCPUOptionHelper {                                                  \
     public:                                                                   \
      TFCPUOptionHelper() {                                                    \
        /* Compilers/linkers remove unused variables aggressively.  The */     \
        /* following gyrations subvert most such optimizations. */             \
        tf_cpu_option_global = tf_cpu_option;                                  \
        /* Nothing is printed because the string starts with a nul. */         \
        printf("%s%s", tf_cpu_option, "");                                     \
      }                                                                        \
    } tf_cpu_option_avoid_omit_class;                                          \
    }  /* anonymous namespace */
// clang-format on

namespace tensorflow {

class Status;

// Retrieves the platform strings from the file at the given path and appends
// them to the given vector. If the returned int is non-zero, an error occurred
// reading the file and vector may or may not be modified. The returned error
// code is suitable for use with strerror().
int GetPlatformStrings(const std::string& path,
                       std::vector<std::string>* found);

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PLATFORM_PLATFORM_STRINGS_H_
