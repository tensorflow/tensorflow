/* Copyright 2015 Google Inc. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_UTIL_CHECKPOINT_READER_H
#define TENSORFLOW_CORE_UTIL_CHECKPOINT_READER_H

#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/tensor_slice_reader.h"

namespace tensorflow {

namespace checkpoint {

class TensorSliceReader;

// A wrapper around checkpoint::TensorSliceReader that is more easily SWIG
// wrapped for Python.
class CheckpointReader {
 public:
  ~CheckpointReader();

  static Status NewCheckpointReaderImpl(
      const string& filepattern, std::unique_ptr<CheckpointReader>* out_reader);
  bool HasTensor(const string& name) const;
  const string DebugString() const;

  const TensorSliceReader::VarToShapeMap& GetVariableToShapeMap() const;

  Status GetTensor(const string& name,
                   std::unique_ptr<tensorflow::Tensor>* out_tensor) const {
    return reader_->GetTensor(name, out_tensor);
  }

 protected:
  CheckpointReader(const string& filepattern, tensorflow::Status*);

 private:
  TensorSliceReader* reader_;                               // Owned
  TensorSliceReader::VarToShapeMap* var_to_shape_map_ptr_;  // Owned

  TF_DISALLOW_COPY_AND_ASSIGN(CheckpointReader);
};

Status NewCheckpointReader(const string& filepattern,
                           std::unique_ptr<CheckpointReader>* out_reader);

}  // namespace checkpoint
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_UTIL_CHECKPOINT_READER_H
