/* Copyright 2016 Google Inc. All Rights Reserved.

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

#include "tensorflow/core/example/feature_util.h"

namespace tensorflow {

namespace internal {

::tensorflow::Feature& ExampleFeature(const string& name,
                                      ::tensorflow::Example* example) {
  ::tensorflow::Features* features = example->mutable_features();
  return (*features->mutable_feature())[name];
}

}  //  namespace internal

template <>
bool ExampleHasFeature<int64>(const string& name, const Example& example) {
  auto it = example.features().feature().find(name);
  return (it != example.features().feature().end()) &&
         (it->second.kind_case() == Feature::KindCase::kInt64List);
}

template <>
bool ExampleHasFeature<float>(const string& name, const Example& example) {
  auto it = example.features().feature().find(name);
  return (it != example.features().feature().end()) &&
         (it->second.kind_case() == Feature::KindCase::kFloatList);
}

template <>
bool ExampleHasFeature<string>(const string& name, const Example& example) {
  auto it = example.features().feature().find(name);
  return (it != example.features().feature().end()) &&
         (it->second.kind_case() == Feature::KindCase::kBytesList);
}

template <>
const protobuf::RepeatedField<int64>& GetFeatureValues<int64>(
    const string& name, const Example& example) {
  return example.features().feature().at(name).int64_list().value();
}

template <>
protobuf::RepeatedField<int64>* GetFeatureValues<int64>(const string& name,
                                                        Example* example) {
  return internal::ExampleFeature(name, example)
      .mutable_int64_list()
      ->mutable_value();
}

template <>
const protobuf::RepeatedField<float>& GetFeatureValues<float>(
    const string& name, const Example& example) {
  return example.features().feature().at(name).float_list().value();
}

template <>
protobuf::RepeatedField<float>* GetFeatureValues<float>(const string& name,
                                                        Example* example) {
  return internal::ExampleFeature(name, example)
      .mutable_float_list()
      ->mutable_value();
}

template <>
const protobuf::RepeatedPtrField<string>& GetFeatureValues<string>(
    const string& name, const Example& example) {
  return example.features().feature().at(name).bytes_list().value();
}

template <>
protobuf::RepeatedPtrField<string>* GetFeatureValues<string>(const string& name,
                                                             Example* example) {
  return internal::ExampleFeature(name, example)
      .mutable_bytes_list()
      ->mutable_value();
}

}  // namespace tensorflow
