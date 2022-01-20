/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/gpu/metrics.h"

#include "tensorflow/core/lib/monitoring/sampler.h"

namespace xla {
namespace {

auto* compile_time_usecs_histogram = tensorflow::monitoring::Sampler<1>::New(
    {"/xla/service/gpu/compile_time_usecs_histogram",
     "The wall-clock time spent on compiling the graphs in microseconds.",
     "phase"},
    // These exponential buckets cover the following range:
    // Minimum: 1 ms
    // Maximum: 1 ms * 2 ^ 24 == ~4.66 hours
    {tensorflow::monitoring::Buckets::Exponential(1000, 2, 25)});

}  // namespace

void RecordHloPassesDuration(const uint64_t time_usecs) {
  static auto* cell = compile_time_usecs_histogram->GetCell("hlo_passes");
  cell->Add(time_usecs);
}

void RecordHloToLlvmDuration(const uint64_t time_usecs) {
  static auto* cell = compile_time_usecs_histogram->GetCell("hlo_to_llvm");
  cell->Add(time_usecs);
}

void RecordLlvmToPtxDuration(const uint64_t time_usecs) {
  static auto* cell = compile_time_usecs_histogram->GetCell("llvm_to_ptx");
  cell->Add(time_usecs);
}

}  // namespace xla
