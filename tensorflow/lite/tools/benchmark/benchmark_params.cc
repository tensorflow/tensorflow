/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/tools/benchmark/benchmark_params.h"

#include <string>
#include <unordered_map>
#include <vector>

#include "tensorflow/lite/tools/benchmark/logging.h"

namespace tflite {
namespace benchmark {

void BenchmarkParam::AssertHasSameType(BenchmarkParam::ParamType a,
                                       BenchmarkParam::ParamType b) {
  TFLITE_BENCHMARK_CHECK(a == b) << "Type mismatch while accessing parameter.";
}

template <>
BenchmarkParam::ParamType BenchmarkParam::GetValueType<int32_t>() {
  return BenchmarkParam::ParamType::TYPE_INT32;
}

template <>
BenchmarkParam::ParamType BenchmarkParam::GetValueType<bool>() {
  return BenchmarkParam::ParamType::TYPE_BOOL;
}

template <>
BenchmarkParam::ParamType BenchmarkParam::GetValueType<float>() {
  return BenchmarkParam::ParamType::TYPE_FLOAT;
}

template <>
BenchmarkParam::ParamType BenchmarkParam::GetValueType<std::string>() {
  return BenchmarkParam::ParamType::TYPE_STRING;
}

void BenchmarkParams::AssertParamExists(const std::string& name) const {
  TFLITE_BENCHMARK_CHECK(HasParam(name)) << name << " was not found.";
}

}  // namespace benchmark
}  // namespace tflite
