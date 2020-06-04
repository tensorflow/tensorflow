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

#ifndef TENSORFLOW_TEST_TARGET_TRIPLE_HELPER_H_
#define TENSORFLOW_TEST_TARGET_TRIPLE_HELPER_H_

#if (defined(__powerpc__) || defined(__ppc__) && (__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__))
static const std::string kTargetCpuForHost="ppc";
static const std::string kTargetTripleForHost="ppc64le-ibm-linux-gnu";
#else
static const std::string kTargetCpuForHost="";
static const std::string kTargetTripleForHost="x86_64-pc-linux";
#endif

#endif
