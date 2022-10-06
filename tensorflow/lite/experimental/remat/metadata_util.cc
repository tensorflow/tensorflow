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
#include "tensorflow/lite/experimental/remat/metadata_util.h"

#include <string>
#include <utility>
#include <vector>

namespace {

// We serialize unsigneds as protobuf varints, i.e., in chunks of 7 bits each.
constexpr int kMod = (1 << 7);

void Serialize(std::string* out, uint32_t value) {
  for (; value >= kMod; value /= kMod) {
    out->push_back(value % kMod + kMod);
  }
  out->push_back(value);
}

bool Parse(const char** data, size_t* size, uint32_t* out) {
  *out = 0;
  uint32_t mul = 1;
  for (bool done = false; !done;
       mul *= kMod, done = !(**data & kMod), ++*data, --*size) {
    if (*size == 0) {
      return false;
    }
    *out += static_cast<unsigned char>(**data) % kMod * mul;
  }
  return true;
}

// Signed ints are zigzag-encoded as unsigned varints, [..., -2, -1, 0, 1, 2,
// ...] ->
// [..., 3, 1, 0, 2, 4, ...].
void Serialize(std::string* out, int32_t value) {
  Serialize(out, static_cast<uint32_t>(
                     value < 0 ? static_cast<uint32_t>(-(value + 1)) * 2 + 1
                               : static_cast<uint32_t>(value) * 2));
}

bool Parse(const char** data, size_t* size, int32_t* out) {
  uint32_t value = 0;
  if (!Parse(data, size, &value)) {
    return false;
  }
  const int32_t magnitude = value / 2;
  *out = (value % 2) ? (-magnitude - 1) : magnitude;
  return true;
}

// Pairs are serialized as the concatenation of their elements' serialization.
template <class First, class Second>
void Serialize(std::string* out, const std::pair<First, Second>& in) {
  Serialize(out, in.first);
  Serialize(out, in.second);
}

template <class First, class Second>
bool Parse(const char** data, size_t* size, std::pair<First, Second>* out) {
  return Parse(data, size, &(out->first)) && Parse(data, size, &(out->second));
}

// Vectors are serialized as the concetation of the serialization of their size
// and the the serializations of their elements.
template <class Value>
void Serialize(std::string* out, const std::vector<Value>& in) {
  Serialize(out, static_cast<uint32_t>(in.size()));
  for (const auto& val : in) {
    Serialize(out, val);
  }
}

template <class T>
bool Parse(const char** data, size_t* size, std::vector<T>* out) {
  uint32_t num_elems = 0;
  if (!Parse(data, size, &num_elems)) {
    return false;
  }
  out->assign(num_elems, T{});
  for (auto& elem : *out) {
    if (!Parse(data, size, &elem)) {
      return false;
    }
  }
  return true;
}

}  // namespace

namespace tflite {
std::string SerializeModelControlDependencies(
    const ModelControlDependencies& in) {
  std::string out;
  Serialize(&out, kModelControlDependenciesMetadataVersion);
  Serialize(&out, in);
  return out;
}

bool ParseModelControlDependencies(const char* data, size_t size,
                                   ModelControlDependencies* out) {
  out->clear();
  uint32_t version = 0;
  return Parse(&data, &size, &version) &&
         (version == kModelControlDependenciesMetadataVersion) &&
         Parse(&data, &size, out) && (size == 0);
}

}  // namespace tflite
