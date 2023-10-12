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

#ifndef TENSORFLOW_CORE_PLATFORM_CPU_INFO_H_
#define TENSORFLOW_CORE_PLATFORM_CPU_INFO_H_

#include <string>

// TODO(ahentz): This is not strictly required here but, for historical
// reasons, many people depend on cpu_info.h in order to use kLittleEndian.
#include "tensorflow/core/platform/byte_order.h"
#include "tsl/platform/cpu_info.h"

namespace tensorflow {
namespace port {
using tsl::port::Aarch64CPU;
using tsl::port::ADX;
using tsl::port::AES;
using tsl::port::AMX_BF16;
using tsl::port::AMX_INT8;
using tsl::port::AMX_TILE;
using tsl::port::AVX;
using tsl::port::AVX2;
using tsl::port::AVX512_4FMAPS;
using tsl::port::AVX512_4VNNIW;
using tsl::port::AVX512_BF16;
using tsl::port::AVX512_VNNI;
using tsl::port::AVX512BW;
using tsl::port::AVX512CD;
using tsl::port::AVX512DQ;
using tsl::port::AVX512ER;
using tsl::port::AVX512F;
using tsl::port::AVX512IFMA;
using tsl::port::AVX512PF;
using tsl::port::AVX512VBMI;
using tsl::port::AVX512VL;
using tsl::port::AVX_VNNI;
using tsl::port::BMI1;
using tsl::port::BMI2;
using tsl::port::CMOV;
using tsl::port::CMPXCHG16B;
using tsl::port::CMPXCHG8B;
using tsl::port::CPUFamily;
using tsl::port::CPUFeature;
using tsl::port::CPUIDNumSMT;
using tsl::port::CPUModelNum;
using tsl::port::CPUVendorIDString;
using tsl::port::F16C;
using tsl::port::FMA;
using tsl::port::GetCurrentCPU;
using tsl::port::HYPERVISOR;
using tsl::port::kUnknownCPU;
using tsl::port::MaxParallelism;
using tsl::port::MMX;
using tsl::port::NominalCPUFrequency;
using tsl::port::NumHyperthreadsPerCore;
using tsl::port::NumSchedulableCPUs;
using tsl::port::NumTotalCPUs;
using tsl::port::PCLMULQDQ;
using tsl::port::POPCNT;
using tsl::port::PREFETCHW;
using tsl::port::PREFETCHWT1;
using tsl::port::RDRAND;
using tsl::port::RDSEED;
using tsl::port::SMAP;
using tsl::port::SSE;
using tsl::port::SSE2;
using tsl::port::SSE3;
using tsl::port::SSE4_1;
using tsl::port::SSE4_2;
using tsl::port::SSSE3;
using tsl::port::TestAarch64CPU;
using tsl::port::TestCPUFeature;

}  // namespace port
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PLATFORM_CPU_INFO_H_
