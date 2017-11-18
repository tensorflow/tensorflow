/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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
#ifndef THIRD_PARTY_TENSORFLOW_CONTRIB_LITE_KERNELS_OP_UTIL_H_
#define THIRD_PARTY_TENSORFLOW_CONTRIB_LITE_KERNELS_OP_UTIL_H_

#define TF_LITE_FATAL(msg)          \
  do {                              \
    fprintf(stderr, "%s\n", (msg)); \
    exit(1);                        \
  } while (0)
#define TF_LITE_ASSERT(x)        \
  do {                           \
    if (!(x)) TF_LITE_FATAL(#x); \
  } while (0)
#define TF_LITE_ASSERT_EQ(x, y)                            \
  do {                                                     \
    if ((x) != (y)) TF_LITE_FATAL(#x " didn't equal " #y); \
  } while (0)

#endif  // THIRD_PARTY_TENSORFLOW_CONTRIB_LITE_KERNELS_OP_UTIL_H_
