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

#include "tensorflow/python/util/py_checkpoint_reader.h"
#include "tensorflow/python/lib/core/py_func.h"

namespace tensorflow {

namespace checkpoint {

PyCheckpointReader::PyCheckpointReader(const string& filepattern,
                                       tensorflow::Status* status)
    : CheckpointReader(filepattern, status) {}

Status PyCheckpointReader::GetTensor(const string& name,
                                     PyObject** numpy_output) const {
  CHECK(numpy_output);
  std::unique_ptr<tensorflow::Tensor> output;
  TF_RETURN_IF_ERROR(CheckpointReader::GetTensor(name, &output));
  TF_RETURN_IF_ERROR(ConvertTensorToNdarray(*output.get(), numpy_output));
  return Status::OK();
}

Status PyCheckpointReader::NewPyCheckpointReaderImpl(
    const string& filepattern,
    std::unique_ptr<PyCheckpointReader>* out_reader) {
  tensorflow::Status status;
  std::unique_ptr<PyCheckpointReader> out(
      new PyCheckpointReader(filepattern, &status));
  if (status.ok()) {
    std::swap(*out_reader, out);
  }
  return status;
}

Status NewPyCheckpointReader(const string& filepattern,
                             std::unique_ptr<PyCheckpointReader>* out_reader) {
  return PyCheckpointReader::NewPyCheckpointReaderImpl(filepattern, out_reader);
}

}  // namespace checkpoint
}  // namespace tensorflow
