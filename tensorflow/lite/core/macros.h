/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
// This provides utility macros and functions that are inherently platform
// specific.
#ifndef TENSORFLOW_LITE_CORE_MACROS_H_
#define TENSORFLOW_LITE_CORE_MACROS_H_

#ifdef __has_builtin
#define TFLITE_HAS_BUILTIN(x) __has_builtin(x)
#else
#define TFLITE_HAS_BUILTIN(x) 0
#endif

#if (!defined(__NVCC__)) && (TFLITE_HAS_BUILTIN(__builtin_expect) || \
                             (defined(__GNUC__) && __GNUC__ >= 3))
#define TFLITE_EXPECT_FALSE(cond) __builtin_expect(cond, false)
#define TFLITE_EXPECT_TRUE(cond) __builtin_expect(!!(cond), true)
#else
#define TFLITE_EXPECT_FALSE(cond) (cond)
#define TFLITE_EXPECT_TRUE(cond) (cond)
#endif

#endif  // TENSORFLOW_LITE_CORE_MACROS_H_
