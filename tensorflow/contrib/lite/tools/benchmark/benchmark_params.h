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

#ifndef TENSORFLOW_CONTRIB_LITE_TOOLS_BENCHMARK_BENCHMARK_PARAMS_H_
#define TENSORFLOW_CONTRIB_LITE_TOOLS_BENCHMARK_BENCHMARK_PARAMS_H_
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "tensorflow/contrib/lite/tools/benchmark/logging.h"

namespace tflite {
namespace benchmark {

template <typename T>
class TypedBenchmarkParam;

class BenchmarkParam {
 protected:
  enum class ParamType { TYPE_INT32, TYPE_FLOAT, TYPE_BOOL, TYPE_STRING };

 public:
  template <typename T>
  static std::unique_ptr<BenchmarkParam> Create(const T& default_value) {
    return std::unique_ptr<BenchmarkParam>(
        new TypedBenchmarkParam<T>(default_value));
  }

  template <typename T>
  TypedBenchmarkParam<T>* AsTyped() {
    AssertHasSameType(GetValueType<T>(), type_);
    return static_cast<TypedBenchmarkParam<T>*>(this);
  }
  virtual ~BenchmarkParam() {}
  BenchmarkParam(ParamType type) : type_(type) {}

 private:
  static void AssertHasSameType(ParamType a, ParamType b);
  template <typename T>
  static ParamType GetValueType();

  const ParamType type_;
};

template <typename T>
class TypedBenchmarkParam : public BenchmarkParam {
 public:
  TypedBenchmarkParam(const T& value)
      : BenchmarkParam(GetValueType<T>()), value_(value) {}
  void Set(const T& value) { value_ = value; }

  T Get() { return value_; }

 private:
  T value_;
};

class BenchmarkParams {
 public:
  void AddParam(const std::string& name,
                std::unique_ptr<BenchmarkParam> value) {
    params_[name] = std::move(value);
  }

  bool HasParam(const std::string& name) const {
    return params_.find(name) != params_.end();
  }

  template <typename T>
  void Set(const std::string& name, const T& value) {
    AssertParamExists(name);
    params_.at(name)->AsTyped<T>()->Set(value);
  }

  template <typename T>
  T Get(const std::string& name) const {
    AssertParamExists(name);
    return params_.at(name)->AsTyped<T>()->Get();
  }

 private:
  void AssertParamExists(const std::string& name) const;
  std::unordered_map<std::string, std::unique_ptr<BenchmarkParam>> params_;
};

}  // namespace benchmark
}  // namespace tflite
#endif  // TENSORFLOW_CONTRIB_LITE_TOOLS_BENCHMARK_BENCHMARK_PARAMS_H_
