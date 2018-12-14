/* Copyright 2016 Google Inc. All Rights Reserved.

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

#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/kernels/fuzzing/fuzz_session.h"

namespace tensorflow {
namespace fuzzing {

class FuzzIdentity : public FuzzSession {
  SINGLE_INPUT_OP_BUILDER(DT_INT8, Identity);

  void FuzzImpl(const uint8_t* data, size_t size) final {
    Tensor input_tensor(tensorflow::DT_INT8,
                        TensorShape({static_cast<int64>(size)}));
    auto flat_tensor = input_tensor.flat<int8>();
    for (size_t i = 0; i < size; i++) {
      flat_tensor(i) = data[i];
    }

    Status s = RunInputs({{"input", input_tensor}});
    // Note:  For many ops, we don't care about this success -- but when
    // testing to make sure the harness actually works, it's useful.
    if (!s.ok()) {
      LOG(ERROR) << "Execution failed: " << s.error_message();
    }
  }
};

STANDARD_TF_FUZZ_FUNCTION(FuzzIdentity);

}  // end namespace fuzzing
}  // end namespace tensorflow
