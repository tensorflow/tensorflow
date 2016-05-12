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

// A set of lightweight wrappers which simplify access to Example features.
//
// Tensorflow Example proto uses associative maps on top of oneof fields.
// So accessing feature values is not very convenient.
//
// For example, to read a first value of integer feature "tag":
//   int id = example.features().feature().at("tag").int64_list().value(0)
//
// to add a value:
//   auto features = example->mutable_features();
//   (*features->mutable_feature())["tag"].mutable_int64_list()->add_value(id)
//
// For float features you have to use float_list, for string - bytes_list.
//
// To do the same with this library:
//   int id = GetFeatureValues<int64>("tag", example).Get(0);
//   GetFeatureValues<int64>("tag", &example)->Add(id);
//
// Modification of bytes features is slightly different:
//   auto tag = GetFeatureValues<string>("tag", example);
//   *tag->Add() = "lorem ipsum";
//
// To copy multiple values into a feature:
//   AppendFeatureValues({1,2,3}, "tag", &example);
//
// GetFeatureValues gives you access to underlying data - RepeatedField object
// (RepeatedPtrField for byte list). So refer to its documentation of
// RepeatedField for full list of supported methods.
//
// NOTE: It is also important to mention that due to the nature of oneof proto
// fields setting a feature of one type automatically clears all values stored
// as another type with the same feature name.

#ifndef TENSORFLOW_EXAMPLE_FEATURE_H_
#define TENSORFLOW_EXAMPLE_FEATURE_H_

#include <iterator>
#include <type_traits>

#include "tensorflow/core/example/example.pb.h"
#include "tensorflow/core/example/feature.pb.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

namespace internal {

// Returns a reference to a feature corresponding to the name.
// Note: it will create a new Feature if it is missing in the example.
::tensorflow::Feature& ExampleFeature(const string& name,
                                      ::tensorflow::Example* example);

// Specializations of RepeatedFieldTrait define a type of RepeatedField
// corresponding to a selected feature type.
template <typename FeatureType>
struct RepeatedFieldTrait;

template <>
struct RepeatedFieldTrait<int64> {
  using Type = protobuf::RepeatedField<int64>;
};

template <>
struct RepeatedFieldTrait<float> {
  using Type = protobuf::RepeatedField<float>;
};

template <>
struct RepeatedFieldTrait<string> {
  using Type = protobuf::RepeatedPtrField<string>;
};

// Specializations of FeatureTrait define a type of feature corresponding to a
// selected value type.
template <typename ValueType, class Enable = void>
struct FeatureTrait;

template <typename ValueType>
struct FeatureTrait<ValueType, typename std::enable_if<
                                   std::is_integral<ValueType>::value>::type> {
  using Type = int64;
};

template <typename ValueType>
struct FeatureTrait<
    ValueType,
    typename std::enable_if<std::is_floating_point<ValueType>::value>::type> {
  using Type = float;
};

template <typename T>
struct is_string
    : public std::integral_constant<
          bool,
          std::is_same<char*, typename std::decay<T>::type>::value ||
              std::is_same<const char*, typename std::decay<T>::type>::value> {
};

template <>
struct is_string<string> : std::true_type {};

template <>
struct is_string<::tensorflow::StringPiece> : std::true_type {};

template <typename ValueType>
struct FeatureTrait<
    ValueType, typename std::enable_if<is_string<ValueType>::value>::type> {
  using Type = string;
};

}  //  namespace internal

// Returns true if feature with the specified name belongs to the example proto.
// Doesn't check feature type. Note that specialized versions return false if
// the feature has a wrong type.
template <typename FeatureType = void>
bool ExampleHasFeature(const string& name, const Example& example) {
  return example.features().feature().find(name) !=
         example.features().feature().end();
}

// Base declaration of a family of template functions to return a read only
// repeated field corresponding to a feature with the specified name.
template <typename FeatureType>
const typename internal::RepeatedFieldTrait<FeatureType>::Type&
GetFeatureValues(const string& name, const Example& example);

// Base declaration of a family of template functions to return a mutable
// repeated field corresponding to a feature with the specified name.
template <typename FeatureType>
typename internal::RepeatedFieldTrait<FeatureType>::Type* GetFeatureValues(
    const string& name, Example* example);

// Copies elements from the range, defined by [first, last) into a feature.
template <typename IteratorType>
void AppendFeatureValues(IteratorType first, IteratorType last,
                         const string& name, Example* example) {
  using FeatureType = typename internal::FeatureTrait<
      typename std::iterator_traits<IteratorType>::value_type>::Type;
  std::copy(first, last, protobuf::RepeatedFieldBackInserter(
                             GetFeatureValues<FeatureType>(name, example)));
}

// Copies all elements from the container into a feature.
template <typename ContainerType>
void AppendFeatureValues(const ContainerType& container, const string& name,
                         Example* example) {
  using IteratorType = typename ContainerType::const_iterator;
  AppendFeatureValues<IteratorType>(container.begin(), container.end(), name,
                                    example);
}

// Copies all elements from the initializer list into a feature.
template <typename ValueType>
void AppendFeatureValues(std::initializer_list<ValueType> container,
                         const string& name, Example* example) {
  using IteratorType =
      typename std::initializer_list<ValueType>::const_iterator;
  AppendFeatureValues<IteratorType>(container.begin(), container.end(), name,
                                    example);
}

template <>
bool ExampleHasFeature<int64>(const string& name, const Example& example);

template <>
bool ExampleHasFeature<float>(const string& name, const Example& example);

template <>
bool ExampleHasFeature<string>(const string& name, const Example& example);

template <>
const protobuf::RepeatedField<int64>& GetFeatureValues<int64>(
    const string& name, const Example& example);

template <>
protobuf::RepeatedField<int64>* GetFeatureValues<int64>(const string& name,
                                                        Example* example);

template <>
const protobuf::RepeatedField<float>& GetFeatureValues<float>(
    const string& name, const Example& example);

template <>
protobuf::RepeatedField<float>* GetFeatureValues<float>(const string& name,
                                                        Example* example);

template <>
const protobuf::RepeatedPtrField<string>& GetFeatureValues<string>(
    const string& name, const Example& example);

template <>
protobuf::RepeatedPtrField<string>* GetFeatureValues<string>(const string& name,
                                                             Example* example);

}  // namespace tensorflow
#endif  // TENSORFLOW_EXAMPLE_FEATURE_H_
