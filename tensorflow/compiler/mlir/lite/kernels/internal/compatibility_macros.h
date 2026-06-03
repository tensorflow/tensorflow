/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_MLIR_LITE_KERNELS_INTERNAL_COMPATIBILITY_MACROS_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_KERNELS_INTERNAL_COMPATIBILITY_MACROS_H_

#ifndef TFLITE_ABORT
#define TFLITE_ABORT abort()
#endif

#ifndef TFLITE_ASSERT_FALSE
#if defined(NDEBUG)
#define TFLITE_ASSERT_FALSE (static_cast<void>(0))
#else
#define TFLITE_ASSERT_FALSE TFLITE_ABORT
#endif
#endif

// LINT.IfChange

#ifndef TFLITE_DCHECK
#define TFLITE_DCHECK(condition) (condition) ? (void)0 : TFLITE_ASSERT_FALSE
#endif

#ifndef TFLITE_DCHECK_EQ
#define TFLITE_DCHECK_EQ(x, y) ((x) == (y)) ? (void)0 : TFLITE_ASSERT_FALSE
#endif

#ifndef TFLITE_DCHECK_NE
#define TFLITE_DCHECK_NE(x, y) ((x) != (y)) ? (void)0 : TFLITE_ASSERT_FALSE
#endif

#ifndef TFLITE_DCHECK_GE
#define TFLITE_DCHECK_GE(x, y) ((x) >= (y)) ? (void)0 : TFLITE_ASSERT_FALSE
#endif

#ifndef TFLITE_DCHECK_GT
#define TFLITE_DCHECK_GT(x, y) ((x) > (y)) ? (void)0 : TFLITE_ASSERT_FALSE
#endif

#ifndef TFLITE_DCHECK_LE
#define TFLITE_DCHECK_LE(x, y) ((x) <= (y)) ? (void)0 : TFLITE_ASSERT_FALSE
#endif

#ifndef TFLITE_DCHECK_LT
#define TFLITE_DCHECK_LT(x, y) ((x) < (y)) ? (void)0 : TFLITE_ASSERT_FALSE
#endif

// LINT.ThenChange(//tensorflow/lite/kernels/internal/compatibility.h)

// TFLITE_DEPRECATED()
//
// Duplicated from absl/base/macros.h to avoid pulling in that library.
// Marks a deprecated class, struct, enum, function, method and variable
// declarations. The macro argument is used as a custom diagnostic message (e.g.
// suggestion of a better alternative).
//
// Example:
//
//   class TFLITE_DEPRECATED("Use Bar instead") Foo {...};
//   TFLITE_DEPRECATED("Use Baz instead") void Bar() {...}
//
// Every usage of a deprecated entity will trigger a warning when compiled with
// clang's `-Wdeprecated-declarations` option. This option is turned off by
// default, but the warnings will be reported by clang-tidy.
#if defined(__clang__) && __cplusplus >= 201103L
#define TFLITE_DEPRECATED(message) __attribute__((deprecated(message)))
#endif

#ifndef TFLITE_DEPRECATED
#define TFLITE_DEPRECATED(message)
#endif

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_KERNELS_INTERNAL_COMPATIBILITY_MACROS_H_
