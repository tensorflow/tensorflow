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
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

namespace checkpoint {

class TensorSliceReader;

CheckpointReader::CheckpointReader(const string& filename,
                                   TF_Status* out_status)
    : reader_(nullptr), var_to_shape_map_ptr_(nullptr) {
  reader_ = new TensorSliceReader(filename);
  if (!reader_->status().ok()) {
    Set_TF_Status_from_Status(out_status, reader_->status());
  } else {
    var_to_shape_map_ptr_ =
        new TensorSliceReader::VarToShapeMap(reader_->GetVariableToShapeMap());
  }
}

CheckpointReader::~CheckpointReader() {
  delete var_to_shape_map_ptr_;
  delete reader_;
}

bool CheckpointReader::HasTensor(const string& name) const {
  return reader_->HasTensor(name, nullptr, nullptr);
}

const TensorSliceReader::VarToShapeMap&
CheckpointReader::GetVariableToShapeMap() const {
  CHECK(var_to_shape_map_ptr_);
  return *var_to_shape_map_ptr_;
}

const string CheckpointReader::DebugString() const {
  return reader_->DebugString();
}

}  // namespace checkpoint
}  // namespace tensorflow
