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

// A set of lightweight wrappers which simplify access to Feature protos.
//
// TensorFlow Example proto uses associative maps on top of oneof fields.
// SequenceExample proto uses associative map of FeatureList.
// So accessing feature values is not very convenient.
//
// For example, to read a first value of integer feature "tag":
//   int id = example.features().feature().at("tag").int64_list().value(0);
//
// to add a value:
//   auto features = example->mutable_features();
//   (*features->mutable_feature())["tag"].mutable_int64_list()->add_value(id);
//
// For float features you have to use float_list, for string - bytes_list.
//
// To do the same with this library:
//   int id = GetFeatureValues<int64_t>("tag", example).Get(0);
//   GetFeatureValues<int64_t>("tag", &example)->Add(id);
//
// Modification of bytes features is slightly different:
//   auto tag = GetFeatureValues<std::string>("tag", &example);
//   *tag->Add() = "lorem ipsum";
//
// To copy multiple values into a feature:
//   AppendFeatureValues({1,2,3}, "tag", &example);
//
// GetFeatureValues gives you access to underlying data - RepeatedField object
// (RepeatedPtrField for byte list). So refer to its documentation of
// RepeatedField for full list of supported methods.
//
// NOTE: Due to the nature of oneof proto fields setting a feature of one type
// automatically clears all values stored as another type with the same feature
// key.
//
// This library also has tools to work with SequenceExample protos.
//
// To get a value from SequenceExample.context:
//   int id = GetFeatureValues<protobuf_int64>("tag", se.context()).Get(0);
// To add a value to the context:
//   GetFeatureValues<protobuf_int64>("tag", se.mutable_context())->Add(42);
//
// To add values to feature_lists:
//   AppendFeatureValues({4.0},
//                       GetFeatureList("images", &se)->Add());
//   AppendFeatureValues({5.0, 3.0},
//                       GetFeatureList("images", &se)->Add());
// This will create a feature list keyed as "images" with two features:
//   feature_lists {
//     feature_list {
//       key: "images"
//       value {
//         feature { float_list { value: [4.0] } }
//         feature { float_list { value: [5.0, 3.0] } }
//       }
//     }
//   }
// For string-valued features, note that the Append... and Set... functions
// support absl::string_view containers. This allows you to copy existing
// buffers into a Feature with only one copy:
//   std::vector<absl::string_view> image;
//   image.push_back(image_buffer);               // No copy.
//   SetFeatureValues(image, "image", &example);  // Copy.
//
// Functions exposed by this library:
//   HasFeature<[FeatureType]>(key, proto) -> bool
//     Returns true if a feature with the specified key, and optionally
//     FeatureType, belongs to the Features or Example proto.
//   HasFeatureList(key, sequence_example) -> bool
//     Returns true if SequenceExample has a feature_list with the key.
//
//   GetFeatureValues<FeatureType>(key, proto) -> RepeatedField<FeatureType>
//     Returns values for the specified key and the FeatureType.
//     Supported types for the proto: Example, Features.
//   GetFeatureList(key, sequence_example) -> RepeatedPtrField<Feature>
//     Returns Feature protos associated with a key.
//
//   AppendFeatureValues(begin, end, feature)
//   AppendFeatureValues(container or initializer_list, feature)
//     Copies values into a Feature.
//   AppendFeatureValues(begin, end, key, proto)
//   AppendFeatureValues(container or initializer_list, key, proto)
//     Copies values into Features and Example protos with the specified key.
//
//   ClearFeatureValues<FeatureType>(feature)
//     Clears the feature's repeated field of the given type.
//
//   SetFeatureValues(begin, end, feature)
//   SetFeatureValues(container or initializer_list, feature)
//     Clears a Feature, then copies values into it.
//   SetFeatureValues(begin, end, key, proto)
//   SetFeatureValues(container or initializer_list, key, proto)
//     Clears Features or Example protos with the specified key,
//     then copies values into them.
//
// Auxiliary functions, it is unlikely you'll need to use them directly:
//   GetFeatures(proto) -> Features
//     A convenience function to get Features proto.
//     Supported types for the proto: Example, Features.
//   GetFeature(key, proto) -> Feature
//     Returns a Feature proto for the specified key.
//     Supported types for the proto: Example, Features.
//   GetFeatureValues<FeatureType>(feature) -> RepeatedField<FeatureType>
//     Returns values of the feature for the FeatureType.

#ifndef TENSORFLOW_CORE_EXAMPLE_FEATURE_UTIL_H_
#define TENSORFLOW_CORE_EXAMPLE_FEATURE_UTIL_H_

#include <algorithm>
#include <iterator>
#include <string>
#include <type_traits>
#include <utility>

#include "absl/strings/string_view.h"
#include "tensorflow/core/example/example.pb.h"
#include "tensorflow/core/example/feature.pb.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/stringpiece.h"

// Must come after the import for absl::string_view.
#ifdef ABSL_HAVE_STD_STRING_VIEW
#include <string_view>
#endif

namespace tensorflow {
namespace internal {

// TODO(gorban): Update all clients in a followup CL.
// Returns a reference to a feature corresponding to the name.
// Note: it will create a new Feature if it is missing in the example.
ABSL_DEPRECATED("Use GetFeature instead.")
Feature& ExampleFeature(absl::string_view name, Example* example);

// Specializations of RepeatedFieldTrait define a type of RepeatedField
// corresponding to a selected feature type.
template <typename FeatureType>
struct RepeatedFieldTrait;

template <>
struct RepeatedFieldTrait<protobuf_int64> {
  using Type = protobuf::RepeatedField<protobuf_int64>;
};

template <>
struct RepeatedFieldTrait<float> {
  using Type = protobuf::RepeatedField<float>;
};

template <>
struct RepeatedFieldTrait<tstring> {
  using Type = protobuf::RepeatedPtrField<std::string>;
};

template <>
struct RepeatedFieldTrait<std::string> {
  using Type = protobuf::RepeatedPtrField<std::string>;
};

// Specializations of FeatureTrait define a type of feature corresponding to a
// selected value type.
template <typename ValueType, class Enable = void>
struct FeatureTrait;

template <typename ValueType>
struct FeatureTrait<ValueType, typename std::enable_if<
                                   std::is_integral<ValueType>::value>::type> {
  using Type = protobuf_int64;
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
struct is_string<std::string> : std::true_type {};

template <>
struct is_string<::tensorflow::StringPiece> : std::true_type {};

template <>
struct is_string<tstring> : std::true_type {};

template <typename ValueType>
struct FeatureTrait<
    ValueType, typename std::enable_if<is_string<ValueType>::value>::type> {
  using Type = std::string;
};

// Port of the C++20 `requires` expressions.
template <typename... T, typename F>
constexpr bool Requires(F) {
  return std::is_invocable<F, T...>::value;
}

struct NoneSuch {};

// True if the Feature map in a tf.Example supports heterogenous lookup.
// See https://abseil.io/tips/144.
inline constexpr bool kFeatureMapHasHeterogeneousLookup =
    Requires<decltype(Features::default_instance().feature())>(
        [](auto&& c) -> decltype(c.find(NoneSuch{})) {});

// Converts an `absl::string_view` into a string-type compatible for use in the
// protobuf library (e.g. as lookup keys in `proto2::Map` or as elements addable
// to a `proto2::RepeatedPtrField`) depending on the BUILD mode.
//
// NOTE: While the newest versions of `proto2::Map` support heterogenous lookup,
// it does so through `std::string_view`. If the type is just an alias (as noted
// by `ABSL_USES_STD_STRING_VIEW`) then nothing more needs to be done; however,
// when the type is not an alias an explicit conversion to is necessary.
//
// NOTE: This conversion is only necessary until the migration for protobuf to
// take a dependency on ABSL is complete.
inline auto ProtoMapKey(absl::string_view str) {
  if constexpr (kFeatureMapHasHeterogeneousLookup) {
#ifdef ABSL_USES_STD_STRING_VIEW
    return str;
#else
#ifdef ABSL_HAVE_STD_STRING_VIEW
    return std::string_view(str.data(), str.size());
#else
    return std::string(str);
#endif
#endif
  } else {
    return std::string(str);
  }
}

}  //  namespace internal

// Returns true if sequence_example has a feature_list with the specified key.
bool HasFeatureList(absl::string_view key,
                    const SequenceExample& sequence_example);

template <typename T>
struct TypeHasFeatures : std::false_type {};

template <>
struct TypeHasFeatures<SequenceExample> : std::true_type {};

template <>
struct TypeHasFeatures<Example> : std::true_type {};

template <>
struct TypeHasFeatures<Features> : std::true_type {};

// A family of template functions to return mutable Features proto from a
// container proto. Supported ProtoTypes: SequenceExample, Example, Features.
template <typename ProtoType>
typename std::enable_if<TypeHasFeatures<ProtoType>::value, Features*>::type
GetFeatures(ProtoType* proto);

template <>
Features* GetFeatures<Features>(Features* proto);
template <>
Features* GetFeatures<Example>(Example* proto);
template <>
Features* GetFeatures<SequenceExample>(SequenceExample* proto);

template <typename ProtoType>
typename std::enable_if<TypeHasFeatures<ProtoType>::value,
                        const Features&>::type
GetFeatures(const ProtoType& proto);

template <>
const Features& GetFeatures<Features>(const Features& proto);
template <>
const Features& GetFeatures<Example>(const Example& proto);
template <>
const Features& GetFeatures<SequenceExample>(const SequenceExample& proto);

// Base declaration of a family of template functions to return a read only
// repeated field of feature values.
template <typename FeatureType>
const typename internal::RepeatedFieldTrait<FeatureType>::Type&
GetFeatureValues(const Feature& feature);

template <>
const protobuf::RepeatedField<protobuf_int64>& GetFeatureValues<protobuf_int64>(
    const Feature& feature);
template <>
const protobuf::RepeatedField<float>& GetFeatureValues<float>(
    const Feature& feature);
template <>
const protobuf::RepeatedPtrField<std::string>& GetFeatureValues<tstring>(
    const Feature& feature);
template <>
const protobuf::RepeatedPtrField<std::string>& GetFeatureValues<std::string>(
    const Feature& feature);

// Returns a read only repeated field corresponding to a feature with the
// specified name and FeatureType. Supported ProtoTypes: SequenceExample,
// Example, Features.
template <typename FeatureType, typename ProtoType>
const typename internal::RepeatedFieldTrait<FeatureType>::Type&
GetFeatureValues(absl::string_view key, const ProtoType& proto) {
  return GetFeatureValues<FeatureType>(
      GetFeatures(proto).feature().at(internal::ProtoMapKey(key)));
}

// Returns a mutable repeated field of a feature values.
template <typename FeatureType>
typename internal::RepeatedFieldTrait<FeatureType>::Type* GetFeatureValues(
    Feature* feature);

template <>
protobuf::RepeatedField<protobuf_int64>* GetFeatureValues<protobuf_int64>(
    Feature* feature);
template <>
protobuf::RepeatedField<float>* GetFeatureValues<float>(Feature* feature);
template <>
protobuf::RepeatedPtrField<std::string>* GetFeatureValues<tstring>(
    Feature* feature);
template <>
protobuf::RepeatedPtrField<std::string>* GetFeatureValues<std::string>(
    Feature* feature);

// Returns a mutable repeated field corresponding to a feature with the
// specified name and FeatureType. Supported ProtoTypes: SequenceExample,
// Example, Features.
template <typename FeatureType, typename ProtoType>
typename internal::RepeatedFieldTrait<FeatureType>::Type* GetFeatureValues(
    absl::string_view key, ProtoType* proto) {
  ::tensorflow::Feature& feature =
      (*GetFeatures(proto)->mutable_feature())[internal::ProtoMapKey(key)];
  return GetFeatureValues<FeatureType>(&feature);
}

// Returns a read-only Feature proto for the specified key, throws
// std::out_of_range if the key is not found. Supported types for the proto:
// SequenceExample, Example, Features.
template <typename ProtoType>
const Feature& GetFeature(absl::string_view key, const ProtoType& proto) {
  return GetFeatures(proto).feature().at(internal::ProtoMapKey(key));
}

// Returns a read-only Feature proto for the specified key, returns nullptr
// if the key is not found. Supported types for the proto: SequenceExample,
// Example, Features.
template <typename ProtoType>
const Feature* MaybeGetFeature(absl::string_view key, const ProtoType& proto) {
  const protobuf::Map<std::string, Feature>& feature_map =
      GetFeatures(proto).feature();
  auto it = feature_map.find(internal::ProtoMapKey(key));

  if (it == feature_map.end()) {
    return nullptr;
  }

  return &it->second;
}

// Base declaration of a family of template functions to return a read only
// repeated field of feature values or nullptr.
template <typename FeatureType>
const typename internal::RepeatedFieldTrait<FeatureType>::Type*
MaybeGetFeatureValues(const Feature& feature);

template <>
const protobuf::RepeatedField<protobuf_int64>*
MaybeGetFeatureValues<protobuf_int64>(const Feature& feature);
template <>
const protobuf::RepeatedField<float>* MaybeGetFeatureValues<float>(
    const Feature& feature);
template <>
const protobuf::RepeatedPtrField<std::string>* MaybeGetFeatureValues<tstring>(
    const Feature& feature);
template <>
const protobuf::RepeatedPtrField<std::string>*
MaybeGetFeatureValues<std::string>(const Feature& feature);

// Returns a read only repeated field corresponding to a feature with the
// specified name and FeatureType. Supported ProtoTypes: SequenceExample,
// Example, Features.
template <typename FeatureType, typename ProtoType>
const typename internal::RepeatedFieldTrait<FeatureType>::Type*
MaybeGetFeatureValues(absl::string_view key, const ProtoType& proto) {
  const Feature* feature = MaybeGetFeature(key, proto);
  if (feature == nullptr) {
    return nullptr;
  }
  return &GetFeatureValues<FeatureType>(*feature);
}

// Returns a mutable Feature proto for the specified key, creates a new if
// necessary. Supported types for the proto: SequenceExample, Example, Features.
template <typename ProtoType>
Feature* GetFeature(absl::string_view key, ProtoType* proto) {
  return &(*GetFeatures(proto)->mutable_feature())[internal::ProtoMapKey(key)];
}

// Returns a repeated field with features corresponding to a feature_list key.
const protobuf::RepeatedPtrField<Feature>& GetFeatureList(
    absl::string_view key, const SequenceExample& sequence_example);

// Returns a mutable repeated field with features corresponding to a
// feature_list key. It will create a new FeatureList if necessary.
protobuf::RepeatedPtrField<Feature>* GetFeatureList(
    absl::string_view feature_list_key, SequenceExample* sequence_example);

template <typename IteratorType>
void AppendFeatureValues(IteratorType first, IteratorType last,
                         Feature* feature) {
  using FeatureType = typename internal::FeatureTrait<
      typename std::iterator_traits<IteratorType>::value_type>::Type;
  auto& values = *GetFeatureValues<FeatureType>(feature);
  values.Reserve(std::distance(first, last));
  for (auto it = first; it != last; ++it) {
    *values.Add() = *it;
  }
}

template <typename ValueType>
void AppendFeatureValues(std::initializer_list<ValueType> container,
                         Feature* feature) {
  using FeatureType = typename internal::FeatureTrait<ValueType>::Type;
  auto& values = *GetFeatureValues<FeatureType>(feature);
  values.Reserve(container.size());
  for (auto& elt : container) {
    *values.Add() = std::move(elt);
  }
}

namespace internal {

// HasSize<T>::value is true_type if T has a size() member.
template <typename T, typename = void>
struct HasSize : std::false_type {};

template <typename T>
struct HasSize<T, absl::void_t<decltype(std::declval<T>().size())>>
    : std::true_type {};

// Reserves the container's size, if a container.size() method exists.
template <typename ContainerType, typename RepeatedFieldType>
auto ReserveIfSizeAvailable(const ContainerType& container,
                            RepeatedFieldType& values) ->
    typename std::enable_if_t<HasSize<ContainerType>::value, void> {
  values.Reserve(container.size());
}

template <typename ContainerType, typename RepeatedFieldType>
auto ReserveIfSizeAvailable(const ContainerType& container,
                            RepeatedFieldType& values) ->
    typename std::enable_if_t<!HasSize<ContainerType>::value, void> {}

}  // namespace internal

template <typename ContainerType>
void AppendFeatureValues(const ContainerType& container, Feature* feature) {
  using IteratorType = typename ContainerType::const_iterator;
  using FeatureType = typename internal::FeatureTrait<
      typename std::iterator_traits<IteratorType>::value_type>::Type;
  auto* values = GetFeatureValues<FeatureType>(feature);
  internal::ReserveIfSizeAvailable(container, *values);
  // This is equivalent to std::copy into `values` with a
  // RepeatedFieldBackInserter, the difference is RFBI isn't compatible with
  // types that we want to convert (e.g. absl::string_view -> std::string).
  for (const auto& elt : container) {
    if constexpr (internal::is_string<FeatureType>::value) {
      *values->Add() = std::string(elt);
    } else {
      *values->Add() = elt;
    }
  }
}

// Copies elements from the range, defined by [first, last) into the feature
// obtainable from the (proto, key) combination.
template <typename IteratorType, typename ProtoType>
void AppendFeatureValues(IteratorType first, IteratorType last,
                         absl::string_view key, ProtoType* proto) {
  AppendFeatureValues(first, last, GetFeature(key, GetFeatures(proto)));
}

// Copies all elements from the container into a feature.
template <typename ContainerType, typename ProtoType>
void AppendFeatureValues(const ContainerType& container, absl::string_view key,
                         ProtoType* proto) {
  AppendFeatureValues<ContainerType>(container,
                                     GetFeature(key, GetFeatures(proto)));
}

// Copies all elements from the initializer list into a Feature contained by
// Features or Example proto.
template <typename ValueType, typename ProtoType>
void AppendFeatureValues(std::initializer_list<ValueType> container,
                         absl::string_view key, ProtoType* proto) {
  AppendFeatureValues<ValueType>(container,
                                 GetFeature(key, GetFeatures(proto)));
}

// Clears the feature's repeated field (int64, float, or string).
template <typename... FeatureType>
void ClearFeatureValues(Feature* feature);

template <>
void ClearFeatureValues<protobuf_int64>(Feature* feature);
template <>
void ClearFeatureValues<float>(Feature* feature);
template <>
void ClearFeatureValues<std::string>(Feature* feature);
template <>
void ClearFeatureValues<tstring>(Feature* feature);

// Clears the feature's repeated field (int64, float, or string). Copies
// elements from the range, defined by [first, last) into the feature's repeated
// field.
template <typename IteratorType>
void SetFeatureValues(IteratorType first, IteratorType last, Feature* feature) {
  using FeatureType = typename internal::FeatureTrait<
      typename std::iterator_traits<IteratorType>::value_type>::Type;
  ClearFeatureValues<FeatureType>(feature);
  AppendFeatureValues(first, last, feature);
}

// Clears the feature's repeated field (int64, float, or string). Copies all
// elements from the initializer list into the feature's repeated field.
template <typename ValueType>
void SetFeatureValues(std::initializer_list<ValueType> container,
                      Feature* feature) {
  using FeatureType = typename internal::FeatureTrait<ValueType>::Type;
  ClearFeatureValues<FeatureType>(feature);
  AppendFeatureValues(container, feature);
}

// Clears the feature's repeated field (int64, float, or string). Copies all
// elements from the container into the feature's repeated field.
template <typename ContainerType>
void SetFeatureValues(const ContainerType& container, Feature* feature) {
  using IteratorType = typename ContainerType::const_iterator;
  using FeatureType = typename internal::FeatureTrait<
      typename std::iterator_traits<IteratorType>::value_type>::Type;
  ClearFeatureValues<FeatureType>(feature);
  AppendFeatureValues(container, feature);
}

// Clears the feature's repeated field (int64, float, or string). Copies
// elements from the range, defined by [first, last) into the feature's repeated
// field.
template <typename IteratorType, typename ProtoType>
void SetFeatureValues(IteratorType first, IteratorType last,
                      absl::string_view key, ProtoType* proto) {
  SetFeatureValues(first, last, GetFeature(key, GetFeatures(proto)));
}

// Clears the feature's repeated field (int64, float, or string). Copies all
// elements from the container into the feature's repeated field.
template <typename ContainerType, typename ProtoType>
void SetFeatureValues(const ContainerType& container, absl::string_view key,
                      ProtoType* proto) {
  SetFeatureValues<ContainerType>(container,
                                  GetFeature(key, GetFeatures(proto)));
}

// Clears the feature's repeated field (int64, float, or string). Copies all
// elements from the initializer list into the feature's repeated field.
template <typename ValueType, typename ProtoType>
void SetFeatureValues(std::initializer_list<ValueType> container,
                      absl::string_view key, ProtoType* proto) {
  SetFeatureValues<ValueType>(container, GetFeature(key, GetFeatures(proto)));
}

// Returns true if a feature with the specified key belongs to the Features.
// The template parameter pack accepts zero or one template argument - which
// is FeatureType. If the FeatureType not specified (zero template arguments)
// the function will not check the feature type. Otherwise it will return false
// if the feature has a wrong type.
template <typename... FeatureType>
bool HasFeature(absl::string_view key, const Features& features);

template <>
bool HasFeature<>(absl::string_view key, const Features& features);
template <>
bool HasFeature<protobuf_int64>(absl::string_view key,
                                const Features& features);
template <>
bool HasFeature<float>(absl::string_view key, const Features& features);
template <>
bool HasFeature<std::string>(absl::string_view key, const Features& features);
template <>
bool HasFeature<tstring>(absl::string_view key, const Features& features);

// Returns true if a feature with the specified key belongs to the Example.
// Doesn't check feature type if used without FeatureType, otherwise the
// specialized versions return false if the feature has a wrong type.
template <typename... FeatureType>
bool HasFeature(absl::string_view key, const Example& example) {
  return HasFeature<FeatureType...>(key, GetFeatures(example));
}

// Returns true if a feature with the specified key belongs to the
// SequenceExample. Doesn't check feature type if used without FeatureType,
// otherwise the specialized versions return false if the feature has a wrong
// type.
template <typename... FeatureType>
bool HasFeature(absl::string_view key,
                const SequenceExample& sequence_example) {
  return HasFeature<FeatureType...>(key, GetFeatures(sequence_example));
}

// TODO(gorban): update all clients in a followup CL.
template <typename... FeatureType>
ABSL_DEPRECATED("Use HasFeature instead.")
bool ExampleHasFeature(absl::string_view key, const Example& example) {
  return HasFeature<FeatureType...>(key, example);
}

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_EXAMPLE_FEATURE_UTIL_H_
