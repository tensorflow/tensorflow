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

#ifndef TENSORFLOW_C_CHECKPOINT_READER_H
#define TENSORFLOW_C_CHECKPOINT_READER_H

#include "tensorflow/c/tf_status_helper.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/tensor_slice_reader.h"

namespace tensorflow {

namespace checkpoint {

class TensorSliceReader;

// A wrapper around checkpoint::TensorSliceReader that is more easily SWIG
// wrapped for other languages.
class CheckpointReader {
 public:
  ~CheckpointReader();

  bool HasTensor(const string& name) const;
  const string DebugString() const;

  const TensorSliceReader::VarToShapeMap& GetVariableToShapeMap() const;

  void GetTensor(const string& name,
                 std::unique_ptr<tensorflow::Tensor>* out_tensor,
                 TF_Status* out_status) const {
    Status status = reader_->GetTensor(name, out_tensor);
    if (!status.ok()) {
      Set_TF_Status_from_Status(out_status, status);
    }
  }

  CheckpointReader(const string& filepattern, TF_Status* out_status);

 private:
  TensorSliceReader* reader_;                               // Owned
  TensorSliceReader::VarToShapeMap* var_to_shape_map_ptr_;  // Owned

  TF_DISALLOW_COPY_AND_ASSIGN(CheckpointReader);
};

}  // namespace checkpoint
}  // namespace tensorflow

#endif  // TENSORFLOW_C_CHECKPOINT_READER_H
