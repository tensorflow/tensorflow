/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/c/checkpoint_reader.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/stringpiece.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/saved_tensor_slice_util.h"
#include "tensorflow/core/util/tensor_bundle/naming.h"
#include "tensorflow/core/util/tensor_bundle/tensor_bundle.h"

namespace tensorflow {
namespace checkpoint {

class TensorSliceReader;

CheckpointReader::CheckpointReader(const std::string& filename,
                                   TF_Status* status)
    : reader_(nullptr),
      v2_reader_(nullptr),
      var_to_shape_map_(nullptr),
      var_to_data_type_map_(nullptr) {
  // Depending on whether this is a V2 ckpt, initializes "reader_" or
  // "v2_reader_".
  std::vector<std::string> v2_path;
  if (Env::Default()->GetMatchingPaths(MetaFilename(filename), &v2_path).ok() &&
      !v2_path.empty()) {
    v2_reader_.reset(
        new BundleReader(Env::Default(), filename /* prefix to a V2 ckpt */));
    if (!v2_reader_->status().ok()) {
      tsl::Set_TF_Status_from_Status(status, v2_reader_->status());
      return;
    }
    auto result = BuildV2VarMaps();
    if (!v2_reader_->status().ok()) {
      tsl::Set_TF_Status_from_Status(status, v2_reader_->status());
      return;
    }
    if (result.first == nullptr || result.second == nullptr) {
      tsl::Set_TF_Status_from_Status(
          status,
          absl::DataLossError("Corrupt checkpoint entry in V2 variable map."));
      return;
    }
    var_to_shape_map_.swap(result.first);
    var_to_data_type_map_.swap(result.second);
  } else {
    reader_.reset(new TensorSliceReader(filename));
    if (!reader_->status().ok()) {
      tsl::Set_TF_Status_from_Status(status, reader_->status());
      return;
    }
    var_to_shape_map_.reset(
        new TensorSliceReader::VarToShapeMap(reader_->GetVariableToShapeMap()));
    var_to_data_type_map_.reset(new TensorSliceReader::VarToDataTypeMap(
        reader_->GetVariableToDataTypeMap()));
  }
}

bool CheckpointReader::HasTensor(const std::string& name) const {
  if (var_to_shape_map_ != nullptr) {
    return var_to_shape_map_->find(name) != var_to_shape_map_->end();
  }
  absl::MutexLock lock(reader_mu_);
  if (reader_ != nullptr) {
    return reader_->HasTensor(name, nullptr, nullptr);
  }
  return v2_reader_->Contains(name);
}

const TensorSliceReader::VarToShapeMap&
CheckpointReader::GetVariableToShapeMap() const {
  CHECK(var_to_shape_map_);
  return *var_to_shape_map_;
}

const TensorSliceReader::VarToDataTypeMap&
CheckpointReader::GetVariableToDataTypeMap() const {
  CHECK(var_to_data_type_map_);
  return *var_to_data_type_map_;
}

const std::string CheckpointReader::DebugString() const {
  absl::MutexLock lock(reader_mu_);
  if (reader_ != nullptr) return reader_->DebugString();
  return v2_reader_->DebugString();
}

void CheckpointReader::GetTensor(
    const std::string& name, std::unique_ptr<tensorflow::Tensor>* out_tensor,
    TF_Status* out_status) const {
  absl::MutexLock lock(reader_mu_);
  absl::Status status;
  if (reader_ != nullptr) {
    status = reader_->GetTensor(name, out_tensor);
  } else {
    tensorflow::DataType dtype;
    tensorflow::TensorShape shape;
    status = v2_reader_->LookupDtypeAndShape(name, &dtype, &shape);
    if (status.ok()) {
      out_tensor->reset(new Tensor(dtype, shape));
      status = v2_reader_->Lookup(name, out_tensor->get());
      if (!status.ok()) out_tensor->reset();
    }
  }
  if (!status.ok()) {
    tsl::Set_TF_Status_from_Status(out_status, status);
  }
}

std::pair<std::unique_ptr<TensorSliceReader::VarToShapeMap>,
          std::unique_ptr<TensorSliceReader::VarToDataTypeMap>>
CheckpointReader::BuildV2VarMaps() {
  CHECK(v2_reader_ != nullptr);
  CHECK(v2_reader_->status().ok());

  // First pass: filters out the entries of the slices.
  absl::flat_hash_set<std::string> filtered_keys;
  BundleEntryProto entry;
  v2_reader_->Seek(kHeaderEntryKey);
  if (!v2_reader_->Valid()) {
    return std::make_pair(nullptr, nullptr);
  }
  for (v2_reader_->Next(); v2_reader_->Valid(); v2_reader_->Next()) {
    if (!entry.ParseFromString(v2_reader_->value())) {
      return std::make_pair(nullptr, nullptr);
    }
    for (int i = 0; i < entry.slices_size(); ++i) {
      const auto& slice_proto = entry.slices(i);
      TensorSlice slice_spec;
      if (!TensorSlice::BuildTensorSlice(slice_proto, &slice_spec).ok()) {
        return std::make_pair(nullptr, nullptr);
      }
      if (!filtered_keys
               .insert(EncodeTensorNameSlice(
                   std::string(v2_reader_->key()) /* full var's name */,
                   slice_spec))
               .second) {
        return std::make_pair(nullptr, nullptr);
      }
    }
  }
  if (!v2_reader_->status().ok()) {
    return std::make_pair(nullptr, nullptr);
  }

  // Second pass: adds the entries, ignoring the filtered keys.
  std::unique_ptr<TensorSliceReader::VarToShapeMap> var_to_shape_map(
      new TensorSliceReader::VarToShapeMap);
  std::unique_ptr<TensorSliceReader::VarToDataTypeMap> var_to_data_type_map(
      new TensorSliceReader::VarToDataTypeMap);
  v2_reader_->Seek(kHeaderEntryKey);
  if (!v2_reader_->Valid()) {
    return std::make_pair(nullptr, nullptr);
  }
  for (v2_reader_->Next(); v2_reader_->Valid(); v2_reader_->Next()) {
    if (filtered_keys.count(v2_reader_->key()) > 0) continue;
    if (!entry.ParseFromString(v2_reader_->value())) {
      return std::make_pair(nullptr, nullptr);
    }
    TensorShape shape;
    if (!TensorShape::BuildTensorShape(entry.shape(), &shape).ok()) {
      return std::make_pair(nullptr, nullptr);
    }
    std::string key(v2_reader_->key());
    (*var_to_shape_map)[key] = std::move(shape);
    (*var_to_data_type_map)[key] = DataType(entry.dtype());
  }
  if (!v2_reader_->status().ok()) {
    return std::make_pair(nullptr, nullptr);
  }
  // The returned pointers are owned by the caller.
  return std::make_pair(std::move(var_to_shape_map),
                        std::move(var_to_data_type_map));
}

}  // namespace checkpoint
}  // namespace tensorflow
