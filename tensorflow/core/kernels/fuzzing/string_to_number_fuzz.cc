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

#include "tensorflow/cc/ops/parsing_ops.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/kernels/fuzzing/fuzz_session.h"

namespace tensorflow {
namespace fuzzing {

class FuzzStringToNumber : public FuzzStringInputOp {
  SINGLE_INPUT_OP_BUILDER(DT_STRING, StringToNumber);
};

// TODO(dga):  Generalize this to hit both the float and int
// variants of StringToNumber - requires an update to the
// plumbing code to specify the output dtype.
STANDARD_TF_FUZZ_FUNCTION(FuzzStringToNumber);

}  // end namespace fuzzing
}  // end namespace tensorflow
