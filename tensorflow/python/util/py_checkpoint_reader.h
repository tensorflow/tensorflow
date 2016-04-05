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

#ifndef TENSORFLOW_PYTHON_UTIL_CHECKPOINT_READER_H
#define TENSORFLOW_PYTHON_UTIL_CHECKPOINT_READER_H

#include "tensorflow/core/util/checkpoint_reader.h"

#include <Python.h>

namespace tensorflow {

namespace checkpoint {

class TensorSliceReader;

// A wrapper around checkpoint::TensorSliceReader that is more easily SWIG
// wrapped for Python.
class PyCheckpointReader : public CheckpointReader {
 public:
  // Returns tensor value as numpy_array.
  Status GetTensor(const string& name, PyObject** numpy_output) const;

  static Status NewPyCheckpointReaderImpl(
      const string& filepattern,
      std::unique_ptr<PyCheckpointReader>* out_reader);

 private:
  PyCheckpointReader(const string& filepattern, Status* status);

  TF_DISALLOW_COPY_AND_ASSIGN(PyCheckpointReader);
};

Status NewPyCheckpointReader(const string& filepattern,
                             std::unique_ptr<PyCheckpointReader>* out_reader);

}  // namespace checkpoint
}  // namespace tensorflow

#endif  // TENSORFLOW_PYTHON_UTIL_CHECKPOINT_READER_H
