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

#include "tensorflow/compiler/mlir/tfrt/jit/tf_jitrt.h"

#include <string>

#include "absl/time/time.h"
#include "tensorflow/core/lib/monitoring/counter.h"
#include "tensorflow/core/lib/monitoring/sampler.h"

namespace tensorflow {
namespace {

auto* compile_time_us = tensorflow::monitoring::Sampler<3>::New(
    {"/tensorflow/serving/jitrt/compile_time_us",
     "Distribution of wall-clock time (in microseconds) spent compiling "
     "`tf_jitrt` kernels.",
     "model_name", "kernel", "type"},
    // These exponential buckets cover the following range:
    // Minimum: 1 ms
    // Maximum: 1 ms * 2 ^ 24 == ~4.66 hours
    {tensorflow::monitoring::Buckets::Exponential(1000, 2, 25)});

auto* compile_counter = monitoring::Counter<2>::New(
    "/tensorflow/serving/jitrt/compilations",
    "Number of `tf_jitrt` kernel compilations. Useful for identifying "
    "excessive recompilations of specialized executables.",
    "model_name", "kernel");

}  // namespace

void RecordCompileTime(const std::string& model_name, const std::string& kernel,
                       std::optional<size_t> specialization,
                       absl::Duration compile_time) {
  auto* compile_time_cell = compile_time_us->GetCell(
      model_name, kernel,
      specialization.has_value() ? "specialized" : "default");
  compile_time_cell->Add(absl::ToInt64Microseconds(compile_time));

  auto* compile_counter_cell = compile_counter->GetCell(model_name, kernel);
  compile_counter_cell->IncrementBy(1);
}

}  // namespace tensorflow
