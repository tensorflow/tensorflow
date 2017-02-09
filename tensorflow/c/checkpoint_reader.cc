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

#include <unordered_set>

#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/saved_tensor_slice_util.h"

namespace tensorflow {

namespace checkpoint {

class TensorSliceReader;

CheckpointReader::CheckpointReader(const string& filename,
                                   TF_Status* out_status)
    : reader_(nullptr), v2_reader_(nullptr), var_to_shape_map_ptr_(nullptr) {
  // Depending on whether this is a V2 ckpt, initializes "reader_" or
  // "v2_reader_".
  std::vector<string> v2_path;
  if (Env::Default()->GetMatchingPaths(MetaFilename(filename), &v2_path).ok() &&
      !v2_path.empty()) {
    v2_reader_ =
        new BundleReader(Env::Default(), filename /* prefix to a V2 ckpt */);
    if (!v2_reader_->status().ok()) {
      Set_TF_Status_from_Status(out_status, v2_reader_->status());
      return;
    }
    var_to_shape_map_ptr_ = BuildV2VarToShapeMap();
  } else {
    reader_ = new TensorSliceReader(filename);
    if (!reader_->status().ok()) {
      Set_TF_Status_from_Status(out_status, reader_->status());
      return;
    }
    var_to_shape_map_ptr_ =
        new TensorSliceReader::VarToShapeMap(reader_->GetVariableToShapeMap());
  }
}

CheckpointReader::~CheckpointReader() {
  delete var_to_shape_map_ptr_;
  delete reader_;
  delete v2_reader_;
}

bool CheckpointReader::HasTensor(const string& name) const {
  if (reader_ != nullptr) {
    return reader_->HasTensor(name, nullptr, nullptr);
  }
  return v2_reader_->Contains(name);
}

const TensorSliceReader::VarToShapeMap&
CheckpointReader::GetVariableToShapeMap() const {
  CHECK(var_to_shape_map_ptr_);
  return *var_to_shape_map_ptr_;
}

const string CheckpointReader::DebugString() const {
  if (reader_ != nullptr) return reader_->DebugString();
  return v2_reader_->DebugString();
}

void CheckpointReader::GetTensor(
    const string& name, std::unique_ptr<tensorflow::Tensor>* out_tensor,
    TF_Status* out_status) const {
  Status status;
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
    Set_TF_Status_from_Status(out_status, status);
  }
}

TensorSliceReader::VarToShapeMap* CheckpointReader::BuildV2VarToShapeMap() {
  CHECK(v2_reader_ != nullptr);
  CHECK(v2_reader_->status().ok());

  // First pass: filters out the entries of the slices.
  std::unordered_set<string> filtered_keys;
  BundleEntryProto entry;
  v2_reader_->Seek(kHeaderEntryKey);
  for (v2_reader_->Next(); v2_reader_->Valid(); v2_reader_->Next()) {
    CHECK(entry.ParseFromArray(v2_reader_->value().data(),
                               v2_reader_->value().size()))
        << entry.InitializationErrorString();
    for (int i = 0; i < entry.slices_size(); ++i) {
      const auto& slice_proto = entry.slices(i);
      CHECK(filtered_keys
                .insert(EncodeTensorNameSlice(
                    v2_reader_->key().ToString() /* full var's name */,
                    TensorSlice(slice_proto)))
                .second);
    }
  }

  // Second pass: adds the entries, ignoring the filtered keys.
  TensorSliceReader::VarToShapeMap* var_to_shape_map =
      new TensorSliceReader::VarToShapeMap;
  v2_reader_->Seek(kHeaderEntryKey);
  for (v2_reader_->Next(); v2_reader_->Valid(); v2_reader_->Next()) {
    if (filtered_keys.count(v2_reader_->key().ToString()) > 0) continue;
    CHECK(entry.ParseFromArray(v2_reader_->value().data(),
                               v2_reader_->value().size()))
        << entry.InitializationErrorString();
    (*var_to_shape_map)[v2_reader_->key().ToString()] =
        TensorShape(entry.shape());
  }
  return var_to_shape_map;  // Owned by caller.
}

}  // namespace checkpoint
}  // namespace tensorflow
