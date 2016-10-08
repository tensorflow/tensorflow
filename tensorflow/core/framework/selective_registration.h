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

#ifndef TENSORFLOW_FRAMEWORK_SELECTIVE_REGISTRATION_H_
#define TENSORFLOW_FRAMEWORK_SELECTIVE_REGISTRATION_H_

#include <string.h>

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
//    out.  See tools/print_required_ops/print_selective_registration_header.py
//    for a tool that can be used to generate ops_to_register.h.
#include "ops_to_register.h"

// ops_to_register should define macros for:
//
//   SHOULD_REGISTER_OP_KERNEL(clz)
//   SHOULD_REGISTER_OP(op)
//   SHOULD_REGISTER_OP_GRADIENT
//   # same as SHOULD_REGISTER_OP, but invoked from a non-constexpr location.
//   SHOULD_REGISTER_OP_NON_CONSTEXPR(op)
//
// Except for SHOULD_REGISTER_OP_NON_CONSTEXPR, the macros should be defined
// using constexprs. See selective_registration_util.h for some utilities that
// can be used.
#if (!defined(SHOULD_REGISTER_OP_KERNEL) || !defined(SHOULD_REGISTER_OP) || \
     !defined(SHOULD_REGISTER_OP_GRADIENT) ||                               \
     !defined(SHOULD_REGISTER_OP_NON_CONSTEXPR))
static_assert(false, "ops_to_register.h must define SHOULD_REGISTER macros");
#endif

#else
#define SHOULD_REGISTER_OP_KERNEL(clz) true
#define SHOULD_REGISTER_OP(op) true
#define SHOULD_REGISTER_OP_NON_CONSTEXPR(op) true
#define SHOULD_REGISTER_OP_GRADIENT true
#endif

#endif  // TENSORFLOW_FRAMEWORK_SELECTIVE_REGISTRATION_H_
