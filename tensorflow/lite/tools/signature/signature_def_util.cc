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

#include "absl/status/status.h"
#include "flatbuffers/flatbuffer_builder.h"  // from @flatbuffers
#include "flatbuffers/flexbuffers.h"  // from @flatbuffers
#include "flatbuffers/vector.h"  // from @flatbuffers
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tsl/platform/status.h"

namespace tflite {
namespace {

using tensorflow::Status;
using SerializedSignatureDefMap = std::map<std::string, std::string>;
using SignatureDefMap = std::map<std::string, tensorflow::SignatureDef>;

const Metadata* GetSignatureDefMetadata(const Model* model) {
  if (!model || !model->metadata()) {
    return nullptr;
  }
  for (int i = 0; i < model->metadata()->size(); ++i) {
    const Metadata* metadata = model->metadata()->Get(i);
    if (metadata->name()->str() == kSignatureDefsMetadataName) {
      return metadata;
    }
  }
  return nullptr;
}

Status ReadSignatureDefMap(const Model* model, const Metadata* metadata,
                           SerializedSignatureDefMap* map) {
  if (!model || !metadata || !map) {
    return tensorflow::errors::InvalidArgument("Arguments must not be nullptr");
  }
  const flatbuffers::Vector<uint8_t>* flatbuffer_data =
      model->buffers()->Get(metadata->buffer())->data();
  const auto signature_defs =
      flexbuffers::GetRoot(flatbuffer_data->data(), flatbuffer_data->size())
          .AsMap();
  for (int i = 0; i < signature_defs.Keys().size(); ++i) {
    const std::string key = signature_defs.Keys()[i].AsString().c_str();
    (*map)[key] = signature_defs[key].AsString().c_str();
  }
  return absl::OkStatus();
}

}  // namespace

Status SetSignatureDefMap(const Model* model,
                          const SignatureDefMap& signature_def_map,
                          std::string* model_data_with_signature_def) {
  if (!model || !model_data_with_signature_def) {
    return tensorflow::errors::InvalidArgument("Arguments must not be nullptr");
  }
  if (signature_def_map.empty()) {
    return tensorflow::errors::InvalidArgument(
        "signature_def_map should not be empty");
  }
  flexbuffers::Builder fbb;
  const size_t start_map = fbb.StartMap();
  auto mutable_model = std::make_unique<ModelT>();
  model->UnPackTo(mutable_model.get(), nullptr);
  int buffer_id = mutable_model->buffers.size();
  const Metadata* metadata = GetSignatureDefMetadata(model);
  if (metadata) {
    buffer_id = metadata->buffer();
  } else {
    auto buffer = std::make_unique<BufferT>();
    mutable_model->buffers.emplace_back(std::move(buffer));
    auto sigdef_metadata = std::make_unique<MetadataT>();
    sigdef_metadata->buffer = buffer_id;
    sigdef_metadata->name = kSignatureDefsMetadataName;
    mutable_model->metadata.emplace_back(std::move(sigdef_metadata));
  }
  for (const auto& entry : signature_def_map) {
    fbb.String(entry.first.c_str(), entry.second.SerializeAsString());
  }
  fbb.EndMap(start_map);
  fbb.Finish();
  mutable_model->buffers[buffer_id]->data = fbb.GetBuffer();
  flatbuffers::FlatBufferBuilder builder;
  auto packed_model = Model::Pack(builder, mutable_model.get());
  FinishModelBuffer(builder, packed_model);
  *model_data_with_signature_def =
      std::string(reinterpret_cast<const char*>(builder.GetBufferPointer()),
                  builder.GetSize());
  return absl::OkStatus();
}

bool HasSignatureDef(const Model* model, const std::string& signature_key) {
  if (!model) {
    return false;
  }
  const Metadata* metadata = GetSignatureDefMetadata(model);
  if (!metadata) {
    return false;
  }
  SerializedSignatureDefMap signature_defs;
  if (ReadSignatureDefMap(model, metadata, &signature_defs) !=
      absl::OkStatus()) {
    return false;
  }
  return (signature_defs.find(signature_key) != signature_defs.end());
}

Status GetSignatureDefMap(const Model* model,
                          SignatureDefMap* signature_def_map) {
  if (!model || !signature_def_map) {
    return tensorflow::errors::InvalidArgument("Arguments must not be nullptr");
  }
  SignatureDefMap retrieved_signature_def_map;
  const Metadata* metadata = GetSignatureDefMetadata(model);
  if (metadata) {
    SerializedSignatureDefMap signature_defs;
    auto status = ReadSignatureDefMap(model, metadata, &signature_defs);
    if (status != absl::OkStatus()) {
      return tensorflow::errors::Internal("Error reading signature def map: ",
                                          status.message());
    }
    for (const auto& entry : signature_defs) {
      tensorflow::SignatureDef signature_def;
      if (!signature_def.ParseFromString(entry.second)) {
        return tensorflow::errors::Internal(
            "Cannot parse signature def found in flatbuffer.");
      }
      retrieved_signature_def_map[entry.first] = signature_def;
    }
    *signature_def_map = retrieved_signature_def_map;
  }
  return absl::OkStatus();
}

Status ClearSignatureDefMap(const Model* model, std::string* model_data) {
  if (!model || !model_data) {
    return tensorflow::errors::InvalidArgument("Arguments must not be nullptr");
  }
  auto mutable_model = std::make_unique<ModelT>();
  model->UnPackTo(mutable_model.get(), nullptr);
  for (int id = 0; id < model->metadata()->size(); ++id) {
    const Metadata* metadata = model->metadata()->Get(id);
    if (metadata->name()->str() == kSignatureDefsMetadataName) {
      auto* buffers = &(mutable_model->buffers);
      buffers->erase(buffers->begin() + metadata->buffer());
      mutable_model->metadata.erase(mutable_model->metadata.begin() + id);
      break;
    }
  }
  flatbuffers::FlatBufferBuilder builder;
  auto packed_model = Model::Pack(builder, mutable_model.get());
  FinishModelBuffer(builder, packed_model);
  *model_data =
      std::string(reinterpret_cast<const char*>(builder.GetBufferPointer()),
                  builder.GetSize());
  return absl::OkStatus();
}

}  // namespace tflite
