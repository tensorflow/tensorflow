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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_CPU_TEST_TARGET_TRIPLE_HELPER_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_CPU_TEST_TARGET_TRIPLE_HELPER_H_

#if defined(__aarch64__)
static const char kTargetCpuForHost[] = "aarch64";
static const char kTargetTripleForHost[] = "aarch64-unknown-linux-gnu";
#elif (defined(__powerpc__) || \
       defined(__ppc__) && (__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__))
static const char kTargetCpuForHost[] = "ppc";
static const char kTargetTripleForHost[] = "ppc64le-ibm-linux-gnu";
#elif defined(__s390x__)
static const char kTargetCpuForHost[] = "s390x";
static const char kTargetTripleForHost[] = "systemz-none-linux-gnu";
#else
static const char kTargetCpuForHost[] = "";
static const char kTargetTripleForHost[] = "x86_64-pc-linux";
#endif

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_CPU_TEST_TARGET_TRIPLE_HELPER_H_
