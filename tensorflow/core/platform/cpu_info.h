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

#ifndef TENSORFLOW_PLATFORM_CPU_INFO_H_
#define TENSORFLOW_PLATFORM_CPU_INFO_H_

#include <string>

// TODO(ahentz): This is not strictly required here but, for historical
// reasons, many people depend on cpu_info.h in order to use kLittleEndian.
#include "tensorflow/core/platform/byte_order.h"

#if defined(_MSC_VER)
#include "tensorflow/core/platform/windows/cpu_info.h"
#endif

namespace tensorflow {
namespace port {

// Returns an estimate of the number of schedulable CPUs for this
// process.  Usually, it's constant throughout the lifetime of a
// process, but it might change if the underlying cluster management
// software can change it dynamically.
int NumSchedulableCPUs();

// Mostly ISA related features that we care about
enum CPUFeature {
  // Do not change numeric assignments.
  MMX = 0,
  SSE = 1,
  SSE2 = 2,
  SSE3 = 3,
  SSSE3 = 4,
  SSE4_1 = 5,
  SSE4_2 = 6,
  CMOV = 7,
  CMPXCHG8B = 8,
  CMPXCHG16B = 9,
  POPCNT = 10,
  AES = 11,
  AVX = 12,
  RDRAND = 13,
  AVX2 = 14,
  FMA = 15,
  F16C = 16,
  PCLMULQDQ = 17,
  RDSEED = 18,
  ADX = 19,
  SMAP = 20,

  // Prefetch Vector Data Into Caches with Intent to Write and T1 Hint
  // http://www.felixcloutier.com/x86/PREFETCHWT1.html.
  // You probably want PREFETCHW instead.
  PREFETCHWT1 = 21,

  BMI1 = 22,
  BMI2 = 23,
  HYPERVISOR = 25,  // 0 when on a real CPU, 1 on (well-behaved) hypervisor.

  // Prefetch Data into Caches in Anticipation of a Write (3D Now!).
  // http://www.felixcloutier.com/x86/PREFETCHW.html
  PREFETCHW = 26,

  // AVX-512: 512-bit vectors (plus masking, etc.) in Knights Landing,
  // Skylake
  // Xeon, etc.; each of these entries is a different subset of
  // instructions,
  // various combinations of which occur on various CPU types.
  AVX512F = 27,        // Foundation
  AVX512CD = 28,       // Conflict detection
  AVX512ER = 29,       // Exponential and reciprocal
  AVX512PF = 30,       // Prefetching
  AVX512VL = 31,       // Shorter vector lengths
  AVX512BW = 32,       // Byte and word
  AVX512DQ = 33,       // Dword and qword
  AVX512VBMI = 34,     // Bit manipulation
  AVX512IFMA = 35,     // Integer multiply-add
  AVX512_4VNNIW = 36,  // Integer neural network
  AVX512_4FMAPS = 37,  // Floating point neural network
};

// Checks whether the current processor supports one of the features above.
// Checks CPU registers to return hardware capabilities.
bool TestCPUFeature(CPUFeature feature);

// Returns CPU Vendor string (i.e. 'GenuineIntel', 'AuthenticAMD', etc.)
std::string CPUVendorIDString();

// Returns CPU family.
int CPUFamily();

// Returns CPU model number.
int CPUModelNum();

// Returns nominal core processor cycles per second of each processor.
double NominalCPUFrequency();

}  // namespace port
}  // namespace tensorflow

#endif  // TENSORFLOW_PLATFORM_CPU_INFO_H_
