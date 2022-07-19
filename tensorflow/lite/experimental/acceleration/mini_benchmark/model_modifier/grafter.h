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
#ifndef TENSORFLOW_LITE_EXPERIMENTAL_ACCELERATION_MINIBENCHMARK_MODEL_MODIFIER_GRAFTER_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_ACCELERATION_MINIBENCHMARK_MODEL_MODIFIER_GRAFTER_H_

#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "flatbuffers/flatbuffers.h"  // from @flatbuffers
#include "flatbuffers/idl.h"  // from @flatbuffers
#include "flatbuffers/reflection_generated.h"  // from @flatbuffers

namespace tflite {
struct Model;
}  // namespace tflite

namespace tflite {
namespace acceleration {

// Combines the given models into one, using the FlatBufferBuilder.
//
// This is useful for constructing models that contain validation data and
// metrics.
//
// The model fields are handled as follows:
// - version is set to 3
// - operator codes are concatenated (no deduplication)
// - subgraphs are concatenated in order, rewriting operator and buffer indices
// to match the combined model. Subgraph names are set from 'subgraph_names'
// - description is taken from first model
// - buffers are concatenated
// - metadata buffer is left unset
// - metadata are concatenated
// - signature_defs are taken from the first model (as they refer to the main
// subgraph).
absl::Status CombineModels(flatbuffers::FlatBufferBuilder* fbb,
                           std::vector<const Model*> models,
                           std::vector<std::string> subgraph_names,
                           const reflection::Schema* schema);

// Convenience methods for copying flatbuffer Tables and Vectors.
//
// These are used by CombineModels above, but also needed for constructing
// validation subgraphs to be combined with models.
class FlatbufferHelper {
 public:
  FlatbufferHelper(flatbuffers::FlatBufferBuilder* fbb,
                   const reflection::Schema* schema);
  template <typename T>
  absl::Status CopyTableToVector(const std::string& name, const T* o,
                                 std::vector<flatbuffers::Offset<T>>* v) {
    auto copied = CopyTable(name, o);
    if (!copied.ok()) {
      return copied.status();
    }
    v->push_back(*copied);
    return absl::OkStatus();
  }
  template <typename T>
  absl::StatusOr<flatbuffers::Offset<T>> CopyTable(const std::string& name,
                                                   const T* o) {
    if (o == nullptr) return 0;
    const reflection::Object* def = FindObject(name);
    if (!def) {
      return absl::NotFoundError(
          absl::StrFormat("Type %s not found in schema", name));
    }
    // We want to use the general copying mechanisms that operate on
    // flatbuffers::Table pointers. Flatbuffer types are not directly
    // convertible to Table, as they inherit privately from table.
    // For type* -> Table*, use reinterpret cast.
    const flatbuffers::Table* ot =
        reinterpret_cast<const flatbuffers::Table*>(o);
    // For Offset<Table *> -> Offset<type>, rely on uoffset_t conversion to
    // any flatbuffers::Offset<T>.
    return flatbuffers::CopyTable(*fbb_, *schema_, *def, *ot).o;
  }
  template <typename int_type>
  flatbuffers::Offset<flatbuffers::Vector<int_type>> CopyIntVector(
      const flatbuffers::Vector<int_type>* from) {
    if (from == nullptr) {
      return 0;
    }
    std::vector<int_type> v{from->cbegin(), from->cend()};
    return fbb_->CreateVector(v);
  }
  const reflection::Object* FindObject(const std::string& name);

 private:
  flatbuffers::FlatBufferBuilder* fbb_;
  const reflection::Schema* schema_;
};

}  // namespace acceleration
}  // namespace tflite

#endif  // THIRD_PARTY_TENSORFLOW_LITE_EXPERIMENTAL_ACCELERATION_MINIBENCHMARK_MODEL_MODIFIER_GRAFTER_H_
