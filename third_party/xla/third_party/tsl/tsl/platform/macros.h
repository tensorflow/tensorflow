/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_TSL_PLATFORM_MACROS_H_
#define TENSORFLOW_TSL_PLATFORM_MACROS_H_

// Compiler attributes
#if (defined(__GNUC__) || defined(__APPLE__)) && !defined(SWIG)
// Compiler supports GCC-style attributes
#define TF_ATTRIBUTE_NORETURN __attribute__((noreturn))
#define TF_ATTRIBUTE_ALWAYS_INLINE __attribute__((always_inline))
#define TF_ATTRIBUTE_NOINLINE __attribute__((noinline))
#define TF_ATTRIBUTE_UNUSED __attribute__((unused))
#define TF_ATTRIBUTE_COLD __attribute__((cold))
#define TF_ATTRIBUTE_WEAK __attribute__((weak))
#define TF_PACKED __attribute__((packed))
#define TF_MUST_USE_RESULT __attribute__((warn_unused_result))
#define TF_PRINTF_ATTRIBUTE(string_index, first_to_check) \
  __attribute__((__format__(__printf__, string_index, first_to_check)))
#define TF_SCANF_ATTRIBUTE(string_index, first_to_check) \
  __attribute__((__format__(__scanf__, string_index, first_to_check)))
#elif defined(_MSC_VER)
// Non-GCC equivalents
#define TF_ATTRIBUTE_NORETURN __declspec(noreturn)
#define TF_ATTRIBUTE_ALWAYS_INLINE __forceinline
#define TF_ATTRIBUTE_NOINLINE
#define TF_ATTRIBUTE_UNUSED
#define TF_ATTRIBUTE_COLD
#define TF_ATTRIBUTE_WEAK
#define TF_MUST_USE_RESULT
#define TF_PACKED
#define TF_PRINTF_ATTRIBUTE(string_index, first_to_check)
#define TF_SCANF_ATTRIBUTE(string_index, first_to_check)
#else
// Non-GCC equivalents
#define TF_ATTRIBUTE_NORETURN
#define TF_ATTRIBUTE_ALWAYS_INLINE
#define TF_ATTRIBUTE_NOINLINE
#define TF_ATTRIBUTE_UNUSED
#define TF_ATTRIBUTE_COLD
#define TF_ATTRIBUTE_WEAK
#define TF_MUST_USE_RESULT
#define TF_PACKED
#define TF_PRINTF_ATTRIBUTE(string_index, first_to_check)
#define TF_SCANF_ATTRIBUTE(string_index, first_to_check)
#endif

// Control visibility outside .so
#if defined(_WIN32)
#ifdef TF_COMPILE_LIBRARY
#define TF_EXPORT __declspec(dllexport)
#else
#define TF_EXPORT __declspec(dllimport)
#endif  // TF_COMPILE_LIBRARY
#else
#define TF_EXPORT __attribute__((visibility("default")))
#endif  // _WIN32

#ifdef __has_builtin
#define TF_HAS_BUILTIN(x) __has_builtin(x)
#else
#define TF_HAS_BUILTIN(x) 0
#endif

// C++11-style attributes (N2761)
#if defined(__has_cpp_attribute)
// Safely checks if an attribute is supported. Equivalent to
// ABSL_HAVE_CPP_ATTRIBUTE.
#define TF_HAS_CPP_ATTRIBUTE(n) __has_cpp_attribute(n)
#else
#define TF_HAS_CPP_ATTRIBUTE(n) 0
#endif

// [[clang::annotate("x")]] allows attaching custom strings (e.g. "x") to
// declarations (variables, functions, fields, etc.) for use by tools. They are
// represented in the Clang AST (as AnnotateAttr nodes) and in LLVM IR, but not
// in final output.
#if TF_HAS_CPP_ATTRIBUTE(clang::annotate)
#define TF_ATTRIBUTE_ANNOTATE(str) [[clang::annotate(str)]]
#else
#define TF_ATTRIBUTE_ANNOTATE(str)
#endif

// A variable declaration annotated with the `TF_CONST_INIT` attribute will
// not compile (on supported platforms) unless the variable has a constant
// initializer.
#if TF_HAS_CPP_ATTRIBUTE(clang::require_constant_initialization)
#define TF_CONST_INIT [[clang::require_constant_initialization]]
#else
#define TF_CONST_INIT
#endif

// Compilers can be told that a certain branch is not likely to be taken
// (for instance, a CHECK failure), and use that information in static
// analysis. Giving it this information can help it optimize for the
// common case in the absence of better information (ie.
// -fprofile-arcs).
#if TF_HAS_BUILTIN(__builtin_expect) || (defined(__GNUC__) && __GNUC__ >= 3)
#define TF_PREDICT_FALSE(x) (__builtin_expect(x, 0))
#define TF_PREDICT_TRUE(x) (__builtin_expect(!!(x), 1))
#else
#define TF_PREDICT_FALSE(x) (x)
#define TF_PREDICT_TRUE(x) (x)
#endif

// DEPRECATED: directly use the macro implementation instead.
// A macro to disallow the copy constructor and operator= functions
// This is usually placed in the private: declarations for a class.
#define TF_DISALLOW_COPY_AND_ASSIGN(TypeName) \
  TypeName(const TypeName&) = delete;         \
  void operator=(const TypeName&) = delete

// The TF_ARRAYSIZE(arr) macro returns the # of elements in an array arr.
//
// The expression TF_ARRAYSIZE(a) is a compile-time constant of type
// size_t.
#define TF_ARRAYSIZE(a)         \
  ((sizeof(a) / sizeof(*(a))) / \
   static_cast<size_t>(!(sizeof(a) % sizeof(*(a)))))

#if defined(__GXX_EXPERIMENTAL_CXX0X__) || __cplusplus >= 201103L || \
    (defined(_MSC_VER) && _MSC_VER >= 1900)
// Define this to 1 if the code is compiled in C++11 mode; leave it
// undefined otherwise.  Do NOT define it to 0 -- that causes
// '#ifdef LANG_CXX11' to behave differently from '#if LANG_CXX11'.
#define LANG_CXX11 1
#endif

#if defined(__clang__) && defined(LANG_CXX11) && defined(__has_warning)
#if __has_feature(cxx_attributes) && __has_warning("-Wimplicit-fallthrough")
#define TF_FALLTHROUGH_INTENDED [[clang::fallthrough]]  // NOLINT
#endif
#endif

#ifndef TF_FALLTHROUGH_INTENDED
#define TF_FALLTHROUGH_INTENDED \
  do {                          \
  } while (0)
#endif

namespace tsl {
namespace internal {
template <typename T>
void remove_unused_variable_compiler_warning(const T&){};
}  // namespace internal
}  // namespace tsl
#define TF_UNUSED_VARIABLE(x) \
  tensorflow::internal::remove_unused_variable_compiler_warning(x)

#endif  // TENSORFLOW_TSL_PLATFORM_MACROS_H_
