/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

// This file provides some common support for 'registration' of e.g. ops and
// kernels. In particular, it relates to the REGISTER_OP (op registration) and
// REGISTER_KERNEL_BUILDER (kernel registration) macros.
//
// Note that there are two sides to 'registration':
//   - Definition (compile-time): making op and kernel definitions _available_.
//   - Usage (run-time): adding particular (available) definitions of ops and
//     kernels to the global OpRegistry / KernelRegistry, to be found when
//     constructing and executing graphs.
//
// Currently, definition and usage happen to be coupled together: all
// 'available' definitions (from the REGISTER_*' macros) are added to the global
// registries on startup / library load.

#ifndef TENSORFLOW_CORE_FRAMEWORK_REGISTRATION_REGISTRATION_H_
#define TENSORFLOW_CORE_FRAMEWORK_REGISTRATION_REGISTRATION_H_

#include <string.h>

#include <type_traits>
#include <utility>

#include "tensorflow/core/framework/registration/options.h"

#if !TF_OPTION_REGISTRATION_V2()

#ifdef SELECTIVE_REGISTRATION

// Experimental selective registration support to reduce binary size.
//
// To use selective registration, when building:
// 1. define SELECTIVE_REGISTRATION, e.g. in gcc by passing
//    -DSELECTIVE_REGISTRATION to compilation.
// 2. Provide ops_to_register.h. This file is not included in the repo and must
//    be placed by the user or a tool where the compiler can find it.  It must
//    define the constants and functions used in the macros below. The
//    functions should be defined as valid constexpr functions, so that they are
//    evaluated at compile time: this is needed to make symbols referenced by
//    un-registered objects unused, and therefore allow the linker to strip them
//    out.  See python/tools/print_selective_registration_header.py for a tool
//    that can be used to generate ops_to_register.h.
//
// ops_to_register.h should define macros for:
//   // Ops for which this is false will not be registered.
//   SHOULD_REGISTER_OP(op)
//   // If this is false, then no gradient ops are registered.
//   SHOULD_REGISTER_OP_GRADIENT
//   // Op kernel classes where this is false won't be registered.
//   SHOULD_REGISTER_OP_KERNEL(clz)
// The macros should be defined using constexprs.

#include "ops_to_register.h"

#if (!defined(SHOULD_REGISTER_OP) || !defined(SHOULD_REGISTER_OP_GRADIENT) || \
     !defined(SHOULD_REGISTER_OP_KERNEL))
static_assert(false, "ops_to_register.h must define SHOULD_REGISTER macros");
#endif
#else  // SELECTIVE_REGISTRATION
#define SHOULD_REGISTER_OP(op) true
#define SHOULD_REGISTER_OP_GRADIENT true
#define SHOULD_REGISTER_OP_KERNEL(clz) true
#endif  // SELECTIVE_REGISTRATION

#else  // ! TF_OPTION_REGISTRATION_V2()

#ifdef SELECTIVE_REGISTRATION
#error TF_OPTION_REGISTRATION_V2(): Compile-time selective registration is not supported
#endif

#endif  // ! TF_OPTION_REGISTRATION_V2()

namespace tensorflow {

// An InitOnStartupMarker is 'initialized' on program startup, purely for the
// side-effects of that initialization - the struct itself is empty. (The type
// is expected to be used to define globals.)
//
// The '<<' operator should be used in initializer expressions to specify what
// to run on startup. The following values are accepted:
//   - An InitOnStartupMarker. Example:
//      InitOnStartupMarker F();
//      InitOnStartupMarker const kInitF =
//        InitOnStartupMarker{} << F();
//   - Something to call, which returns an InitOnStartupMarker. Example:
//      InitOnStartupMarker const kInit =
//        InitOnStartupMarker{} << []() { G(); return
//
// See also: TF_INIT_ON_STARTUP_IF
struct InitOnStartupMarker {
  constexpr InitOnStartupMarker operator<<(InitOnStartupMarker) const {
    return *this;
  }

  template <typename T>
  constexpr InitOnStartupMarker operator<<(T&& v) const {
    return std::forward<T>(v)();
  }
};

// Conditional initializer expressions for InitOnStartupMarker:
//   TF_INIT_ON_STARTUP_IF(cond) << f
// If 'cond' is true, 'f' is evaluated (and called, if applicable) on startup.
// Otherwise, 'f' is *not evaluated*. Note that 'cond' is required to be a
// constant-expression, and so this approximates #ifdef.
//
// The implementation uses the ?: operator (!cond prevents evaluation of 'f').
// The relative precedence of ?: and << is significant; this effectively expands
// to (see extra parens):
//   !cond ? InitOnStartupMarker{} : (InitOnStartupMarker{} << f)
//
// Note that although forcing 'cond' to be a constant-expression should not
// affect binary size (i.e. the same optimizations should apply if it 'happens'
// to be one), it was found to be necessary (for a recent version of clang;
// perhaps an optimizer bug).
//
// The parens are necessary to hide the ',' from the preprocessor; it could
// otherwise act as a macro argument separator.
#define TF_INIT_ON_STARTUP_IF(cond)                \
  (::std::integral_constant<bool, !(cond)>::value) \
      ? ::tensorflow::InitOnStartupMarker{}        \
      : ::tensorflow::InitOnStartupMarker {}

// Wrapper for generating unique IDs (for 'anonymous' InitOnStartup definitions)
// using __COUNTER__. The new ID (__COUNTER__ already expanded) is provided as a
// macro argument.
//
// Usage:
//   #define M_IMPL(id, a, b) ...
//   #define M(a, b) TF_NEW_ID_FOR_INIT(M, a, b)
#define TF_NEW_ID_FOR_INIT_2(m, c, ...) m(c, __VA_ARGS__)
#define TF_NEW_ID_FOR_INIT_1(m, c, ...) TF_NEW_ID_FOR_INIT_2(m, c, __VA_ARGS__)
#define TF_NEW_ID_FOR_INIT(m, ...) \
  TF_NEW_ID_FOR_INIT_1(m, __COUNTER__, __VA_ARGS__)

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_FRAMEWORK_REGISTRATION_REGISTRATION_H_
