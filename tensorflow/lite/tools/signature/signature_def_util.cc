/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/tools/signature/signature_def_util.h"

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <utility>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "flatbuffers/buffer.h"  // from @flatbuffers
#include "flatbuffers/flatbuffer_builder.h"  // from @flatbuffers
#include "flatbuffers/flexbuffers.h"  // from @flatbuffers
#include "flatbuffers/vector.h"  // from @flatbuffers
#include "tensorflow/core/protobuf/meta_graph.pb.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace {

using SerializedSignatureDefMap = absl::flat_hash_map<std::string, std::string>;
using SignatureDefMap = std::map<std::string, tensorflow::SignatureDef>;

const Metadata* GetSignatureDefMetadata(const Model* model) {
  if (!model || !model->metadata()) {
    return nullptr;
  }
  for (const Metadata* metadata : *model->metadata()) {
    if (metadata && metadata->name() != nullptr &&
        absl::string_view(metadata->name()->c_str(),
                          metadata->name()->size()) ==
            kSignatureDefsMetadataName) {
      return metadata;
    }
  }
  return nullptr;
}

absl::Status ReadSignatureDefMap(const Model* model, const Metadata* metadata,
                                 SerializedSignatureDefMap* map) {
  if (!model || !metadata || !map) {
    return absl::InvalidArgumentError("Arguments must not be nullptr");
  }
  if (!model->buffers()) {
    return absl::InvalidArgumentError("Missing buffers vector in model");
  }
  if (metadata->buffer() >= model->buffers()->size()) {
    return absl::InternalError("Invalid buffer index in metadata");
  }
  const flatbuffers::Vector<uint8_t>* flatbuffer_data =
      model->buffers()->Get(metadata->buffer())->data();
  if (!flatbuffer_data || flatbuffer_data->size() < 3 ||
      !flexbuffers::VerifyBuffer(flatbuffer_data->data(),
                                 flatbuffer_data->size())) {
    return absl::InvalidArgumentError("Invalid flexbuffer data");
  }
  const auto signature_defs =
      flexbuffers::GetRoot(flatbuffer_data->data(), flatbuffer_data->size())
          .AsMap();
  for (size_t i = 0; i < signature_defs.size(); ++i) {
    const std::string key = signature_defs.Keys()[i].AsString().str();
    (*map)[key] = signature_defs.Values()[i].AsString().str();
  }
  return absl::OkStatus();
}

}  // namespace

absl::Status SetSignatureDefMap(const Model* model,
                                const SignatureDefMap& signature_def_map,
                                std::string* model_data_with_signature_defs) {
  if (!model || !model_data_with_signature_defs) {
    return absl::InvalidArgumentError("Arguments must not be nullptr");
  }
  if (signature_def_map.empty()) {
    return absl::InvalidArgumentError("signature_def_map should not be empty");
  }
  if (!model->buffers()) {
    return absl::InvalidArgumentError("Missing buffers vector in model");
  }
  flexbuffers::Builder fbb;
  const size_t start_map = fbb.StartMap();
  auto mutable_model = std::make_unique<ModelT>();
  model->UnPackTo(mutable_model.get(), nullptr);
  uint32_t buffer_id = mutable_model->buffers.size();
  const Metadata* metadata = GetSignatureDefMetadata(model);
  if (metadata) {
    buffer_id = metadata->buffer();
    if (buffer_id >= mutable_model->buffers.size()) {
      return absl::InternalError("Invalid buffer index in metadata");
    }
  } else {
    auto buffer = std::make_unique<BufferT>();
    mutable_model->buffers.emplace_back(std::move(buffer));
    auto sigdef_metadata = std::make_unique<MetadataT>();
    sigdef_metadata->buffer = buffer_id;
    sigdef_metadata->name = kSignatureDefsMetadataName;
    mutable_model->metadata.emplace_back(std::move(sigdef_metadata));
  }
  for (const auto& [key, signature_def] : signature_def_map) {
    fbb.String(key.c_str(), signature_def.SerializeAsString());
  }
  fbb.EndMap(start_map);
  fbb.Finish();
  mutable_model->buffers[buffer_id]->data = fbb.GetBuffer();
  flatbuffers::FlatBufferBuilder builder;
  flatbuffers::Offset<Model> packed_model =
      Model::Pack(builder, mutable_model.get());
  FinishModelBuffer(builder, packed_model);
  model_data_with_signature_defs->assign(
      reinterpret_cast<const char*>(builder.GetBufferPointer()),
      builder.GetSize());
  return absl::OkStatus();
}

bool HasSignatureDef(const Model* model, absl::string_view signature_key) {
  if (!model) {
    return false;
  }
  const Metadata* metadata = GetSignatureDefMetadata(model);
  if (!metadata) {
    return false;
  }
  if (!model->buffers()) {
    return false;
  }
  if (metadata->buffer() >= model->buffers()->size()) {
    return false;
  }
  const flatbuffers::Vector<uint8_t>* flatbuffer_data =
      model->buffers()->Get(metadata->buffer())->data();
  if (!flatbuffer_data || flatbuffer_data->size() < 3 ||
      !flexbuffers::VerifyBuffer(flatbuffer_data->data(),
                                 flatbuffer_data->size())) {
    return false;
  }
  const auto signature_defs =
      flexbuffers::GetRoot(flatbuffer_data->data(), flatbuffer_data->size())
          .AsMap();
  return !signature_defs[std::string(signature_key)].IsNull();
}

absl::Status GetSignatureDefMap(const Model* model,
                                SignatureDefMap* signature_def_map) {
  if (!model || !signature_def_map) {
    return absl::InvalidArgumentError("Arguments must not be nullptr");
  }
  SignatureDefMap retrieved_signature_def_map;
  const Metadata* metadata = GetSignatureDefMetadata(model);
  if (metadata) {
    SerializedSignatureDefMap signature_defs;
    absl::Status status = ReadSignatureDefMap(model, metadata, &signature_defs);
    if (!status.ok()) {
      return absl::Status(
          status.code(),
          absl::StrCat("Error reading signature def map: ", status.message()));
    }
    tensorflow::SignatureDef signature_def;
    for (const auto& [key, serialized_def] : signature_defs) {
      if (!signature_def.ParseFromString(serialized_def)) {
        return absl::InternalError(
            "Cannot parse signature def found in flatbuffer.");
      }
      retrieved_signature_def_map[key] = std::move(signature_def);
    }
    *signature_def_map = std::move(retrieved_signature_def_map);
  }
  return absl::OkStatus();
}

absl::Status ClearSignatureDefMap(const Model* model, std::string* model_data) {
  if (!model || !model_data) {
    return absl::InvalidArgumentError("Arguments must not be nullptr");
  }
  auto mutable_model = std::make_unique<ModelT>();
  model->UnPackTo(mutable_model.get(), nullptr);
  auto it = absl::c_find_if(mutable_model->metadata, [](const auto& m) {
    return m->name == kSignatureDefsMetadataName;
  });
  if (it != mutable_model->metadata.end()) {
    if ((*it)->buffer >= mutable_model->buffers.size()) {
      return absl::InternalError("Invalid buffer index in metadata");
    }
    mutable_model->buffers[(*it)->buffer]->data.clear();
    mutable_model->metadata.erase(it);
  }
  flatbuffers::FlatBufferBuilder builder;
  flatbuffers::Offset<Model> packed_model =
      Model::Pack(builder, mutable_model.get());
  FinishModelBuffer(builder, packed_model);
  model_data->assign(reinterpret_cast<const char*>(builder.GetBufferPointer()),
                     builder.GetSize());
  return absl::OkStatus();
}

}  // namespace tflite
