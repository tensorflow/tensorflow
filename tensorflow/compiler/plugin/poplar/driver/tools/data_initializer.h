/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_DATA_INITIALIZER_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_DATA_INITIALIZER_H_

#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"

#include <memory>
#include <string>

namespace xla {

class Shape;

namespace poplarplugin {

enum class RandomType {
  UNIFORM,
  NORMAL,
};

class DataInitializer {
 public:
  // Creates a DataInitializer given the type string.
  static std::unique_ptr<DataInitializer> GetDataInitializer(
      const std::string& type_string);

  // Get the instance which is used for initializing synthetic data.
  static DataInitializer& GetSyntheticDataInitializer();

  // Gets the underlying initialization string.
  const std::string GetTypeString() const;

  // Get an initialised xla::Literal.
  StatusOr<Literal> GetData(const Shape& shape);

 protected:
  DataInitializer(const std::string& type_string);

  // A function which is used to create and put a single element into the
  // buffer.
  virtual void GetValue(char*& buffer, const PrimitiveType& type) = 0;

 private:
  std::string type_string_;
};

class RandomDataInitializer : public DataInitializer {
 public:
  RandomDataInitializer(const std::string& type_string, RandomType random_type);

 protected:
  void GetValue(char*& buffer, const PrimitiveType& type) override;

 private:
  RandomType random_type_;
  std::random_device random_device_;
  std::mt19937 generator_;
};

class ConstantDataInitializer : public DataInitializer {
 public:
  ConstantDataInitializer(const std::string& type_string);

 protected:
  void GetValue(char*& buffer, const PrimitiveType& type) override;

 private:
  int32 value_;
};

}  // namespace poplarplugin
}  // namespace xla

#endif
