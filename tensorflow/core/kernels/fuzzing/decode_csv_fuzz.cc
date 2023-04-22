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

class FuzzDecodeCsv : public FuzzStringInputOp {
  void BuildGraph(const Scope& scope) override {
    auto input =
        tensorflow::ops::Placeholder(scope.WithOpName("input"), DT_STRING);
    // For now, assume we want CSVs with 4 columns, as we need a refactoring
    // of the entire infrastructure to support the more complex usecase due to
    // the fact that graph generation and fuzzing data are at separate steps.
    InputList defaults = {Input("a"), Input("b"), Input("c"), Input("d")};
    (void)tensorflow::ops::DecodeCSV(scope.WithOpName("output"), input,
                                     defaults);
  }
};

STANDARD_TF_FUZZ_FUNCTION(FuzzDecodeCsv);

}  // end namespace fuzzing
}  // end namespace tensorflow
