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

#include "tensorflow/core/example/feature_util.h"

namespace tensorflow {

namespace internal {
Feature& ExampleFeature(const std::string& name, Example* example) {
  return *GetFeature(name, example);
}

}  // namespace internal

template <>
bool HasFeature<>(const std::string& key, const Features& features) {
  return (features.feature().find(key) != features.feature().end());
}

template <>
bool HasFeature<protobuf_int64>(const std::string& key,
                                const Features& features) {
  auto it = features.feature().find(key);
  return (it != features.feature().end()) &&
         (it->second.kind_case() == Feature::KindCase::kInt64List);
}

template <>
bool HasFeature<float>(const std::string& key, const Features& features) {
  auto it = features.feature().find(key);
  return (it != features.feature().end()) &&
         (it->second.kind_case() == Feature::KindCase::kFloatList);
}

template <>
bool HasFeature<std::string>(const std::string& key, const Features& features) {
  auto it = features.feature().find(key);
  return (it != features.feature().end()) &&
         (it->second.kind_case() == Feature::KindCase::kBytesList);
}

template <>
bool HasFeature<tstring>(const std::string& key, const Features& features) {
  auto it = features.feature().find(key);
  return (it != features.feature().end()) &&
         (it->second.kind_case() == Feature::KindCase::kBytesList);
}

bool HasFeatureList(const std::string& key,
                    const SequenceExample& sequence_example) {
  auto& feature_list = sequence_example.feature_lists().feature_list();
  return (feature_list.find(key) != feature_list.end());
}

template <>
const protobuf::RepeatedField<protobuf_int64>& GetFeatureValues<protobuf_int64>(
    const Feature& feature) {
  return feature.int64_list().value();
}

template <>
protobuf::RepeatedField<protobuf_int64>* GetFeatureValues<protobuf_int64>(
    Feature* feature) {
  return feature->mutable_int64_list()->mutable_value();
}

template <>
const protobuf::RepeatedField<float>& GetFeatureValues<float>(
    const Feature& feature) {
  return feature.float_list().value();
}

template <>
protobuf::RepeatedField<float>* GetFeatureValues<float>(Feature* feature) {
  return feature->mutable_float_list()->mutable_value();
}

template <>
const protobuf::RepeatedPtrField<std::string>& GetFeatureValues<tstring>(
    const Feature& feature) {
  return feature.bytes_list().value();
}

template <>
const protobuf::RepeatedPtrField<std::string>& GetFeatureValues<std::string>(
    const Feature& feature) {
  return feature.bytes_list().value();
}

template <>
protobuf::RepeatedPtrField<std::string>* GetFeatureValues<tstring>(
    Feature* feature) {
  return feature->mutable_bytes_list()->mutable_value();
}

template <>
protobuf::RepeatedPtrField<std::string>* GetFeatureValues<std::string>(
    Feature* feature) {
  return feature->mutable_bytes_list()->mutable_value();
}

const protobuf::RepeatedPtrField<Feature>& GetFeatureList(
    const std::string& key, const SequenceExample& sequence_example) {
  return sequence_example.feature_lists().feature_list().at(key).feature();
}

protobuf::RepeatedPtrField<Feature>* GetFeatureList(
    const std::string& feature_list_key, SequenceExample* sequence_example) {
  return (*sequence_example->mutable_feature_lists()
               ->mutable_feature_list())[feature_list_key]
      .mutable_feature();
}

template <>
void ClearFeatureValues<protobuf_int64>(Feature* feature) {
  feature->mutable_int64_list()->Clear();
}

template <>
void ClearFeatureValues<float>(Feature* feature) {
  feature->mutable_float_list()->Clear();
}

template <>
void ClearFeatureValues<std::string>(Feature* feature) {
  feature->mutable_bytes_list()->Clear();
}

template <>
void ClearFeatureValues<tstring>(Feature* feature) {
  feature->mutable_bytes_list()->Clear();
}

template <>
Features* GetFeatures<Features>(Features* proto) {
  return proto;
}

template <>
Features* GetFeatures<Example>(Example* proto) {
  return proto->mutable_features();
}

template <>
const Features& GetFeatures<Features>(const Features& proto) {
  return proto;
}

template <>
const Features& GetFeatures<Example>(const Example& proto) {
  return proto.features();
}

template <>
const protobuf::RepeatedField<protobuf_int64>& GetFeatureValues<protobuf_int64>(
    const Feature& feature);

template <>
protobuf::RepeatedField<protobuf_int64>* GetFeatureValues<protobuf_int64>(
    Feature* feature);

template <>
const protobuf::RepeatedField<float>& GetFeatureValues<float>(
    const Feature& feature);

template <>
protobuf::RepeatedField<float>* GetFeatureValues<float>(Feature* feature);

template <>
const protobuf::RepeatedPtrField<std::string>& GetFeatureValues<std::string>(
    const Feature& feature);

template <>
const protobuf::RepeatedPtrField<std::string>& GetFeatureValues<tstring>(
    const Feature& feature);

template <>
protobuf::RepeatedPtrField<std::string>* GetFeatureValues<std::string>(
    Feature* feature);

template <>
protobuf::RepeatedPtrField<std::string>* GetFeatureValues<tstring>(
    Feature* feature);

}  // namespace tensorflow
