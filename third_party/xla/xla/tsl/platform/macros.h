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

#ifndef XLA_TSL_PLATFORM_MACROS_H_
#define XLA_TSL_PLATFORM_MACROS_H_

#include "absl/base/attributes.h"
#include "absl/base/macros.h"
#include "absl/base/optimization.h"

// Compiler attributes
#define TF_ATTRIBUTE_ALWAYS_INLINE ABSL_ATTRIBUTE_ALWAYS_INLINE
#define TF_ATTRIBUTE_NOINLINE ABSL_ATTRIBUTE_NOINLINE
#define TF_ATTRIBUTE_UNUSED ABSL_ATTRIBUTE_UNUSED
#define TF_PACKED ABSL_ATTRIBUTE_PACKED
#define TF_MUST_USE_RESULT ABSL_MUST_USE_RESULT

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

// [[clang::annotate("x")]] allows attaching custom strings (e.g. "x") to
// declarations (variables, functions, fields, etc.) for use by tools. They are
// represented in the Clang AST (as AnnotateAttr nodes) and in LLVM IR, but not
// in final output.
#if ABSL_HAVE_CPP_ATTRIBUTE(clang::annotate)
#define TF_ATTRIBUTE_ANNOTATE(str) [[clang::annotate(str)]]
#else
#define TF_ATTRIBUTE_ANNOTATE(str)
#endif

// A variable declaration annotated with the `TF_CONST_INIT` attribute will
// not compile (on supported platforms) unless the variable has a constant
// initializer.
#define TF_CONST_INIT ABSL_CONST_INIT

// Compilers can be told that a certain branch is not likely to be taken
// (for instance, a CHECK failure), and use that information in static
// analysis. Giving it this information can help it optimize for the
// common case in the absence of better information (ie.
// -fprofile-arcs).
#define TF_PREDICT_FALSE ABSL_PREDICT_FALSE
#define TF_PREDICT_TRUE ABSL_PREDICT_TRUE

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
#define TF_ARRAYSIZE ABSL_ARRAYSIZE

#if defined(__GXX_EXPERIMENTAL_CXX0X__) || __cplusplus >= 201103L || \
    (defined(_MSC_VER) && _MSC_VER >= 1900)
// Define this to 1 if the code is compiled in C++11 mode; leave it
// undefined otherwise.  Do NOT define it to 0 -- that causes
// '#ifdef LANG_CXX11' to behave differently from '#if LANG_CXX11'.
#define LANG_CXX11 1
#endif

#define TF_FALLTHROUGH_INTENDED ABSL_FALLTHROUGH_INTENDED

#endif  // XLA_TSL_PLATFORM_MACROS_H_
