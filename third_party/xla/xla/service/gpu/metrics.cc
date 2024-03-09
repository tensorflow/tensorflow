/* Copyright 2022 The OpenXLA Authors.

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

#include "xla/service/gpu/metrics.h"

#include <cstdint>

#include "tsl/lib/monitoring/counter.h"
#include "tsl/lib/monitoring/gauge.h"
#include "tsl/lib/monitoring/sampler.h"

namespace xla {
namespace {

auto* compile_time_usecs_histogram = tsl::monitoring::Sampler<1>::New(
    {"/xla/service/gpu/compile_time_usecs_histogram",
     "The wall-clock time spent on compiling the graphs in microseconds.",
     "phase"},
    // These exponential buckets cover the following range:
    // Minimum: 1 ms
    // Maximum: 1 ms * 2 ^ 24 == ~4.66 hours
    {tsl::monitoring::Buckets::Exponential(1000, 2, 25)});

auto* compiled_programs_count = tsl::monitoring::Counter<0>::New(
    "/xla/service/gpu/compiled_programs_count", "Number of compiled programs.");

auto* xla_device_binary_size = tsl::monitoring::Gauge<int64_t, 0>::New(
    "/xla/service/gpu/xla_device_binary_size",
    "The size of the XLA binary loaded onto the GPU device.");

}  // namespace

void RecordHloPassesDuration(const uint64_t time_usecs) {
  static auto* cell = compile_time_usecs_histogram->GetCell("hlo_passes");
  cell->Add(time_usecs);
}

void RecordHloToLlvmDuration(const uint64_t time_usecs) {
  static auto* cell = compile_time_usecs_histogram->GetCell("hlo_to_llvm");
  cell->Add(time_usecs);
}

void RecordLlvmPassesAndLlvmToPtxDuration(const uint64_t time_usecs) {
  // When 'llvm_to_ptx' was added, it mistakenly included both llvm
  // optimization and llvm to ptx compilation, and now changing it would
  // invalidate historical data.
  static auto* cell = compile_time_usecs_histogram->GetCell("llvm_to_ptx");
  cell->Add(time_usecs);
}

void RecordLlvmPassesDuration(const uint64_t time_usecs) {
  static auto* cell = compile_time_usecs_histogram->GetCell("llvm_passes");
  cell->Add(time_usecs);
}

void RecordLlvmToPtxDuration(const uint64_t time_usecs) {
  // 'llvm_to_ptx' is taken and can't be changed without invalidating
  // historical data.
  static auto* cell = compile_time_usecs_histogram->GetCell("llvm_to_ptx_only");
  cell->Add(time_usecs);
}

void RecordPtxToCubinDuration(const uint64_t time_usecs) {
  static auto* cell = compile_time_usecs_histogram->GetCell("ptx_to_cubin");
  cell->Add(time_usecs);
}

void IncrementCompiledProgramsCount() {
  compiled_programs_count->GetCell()->IncrementBy(1);
}

int64_t GetCompiledProgramsCount() {
  return compiled_programs_count->GetCell()->value();
}

void RecordXlaDeviceBinarySize(const int64_t size) {
  xla_device_binary_size->GetCell()->Set(size);
}

}  // namespace xla
