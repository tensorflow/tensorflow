// Copyright 2022 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <gtest/gtest.h>
#include "fuzztest/fuzztest.h"
#include "tensorflow/cc/ops/math_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/security/fuzzing/cc/core/framework/tensor_domains.h"
#include "tensorflow/security/fuzzing/cc/fuzz_session.h"

namespace tensorflow {
namespace fuzzing {

// Creates FuzzIdentity class that wraps a single operation node session.
BINARY_INPUT_OP_FUZZER(DT_INT32, DT_INT32, MatMul);
// Setup up fuzzing test.
FUZZ_TEST_F(FuzzMatMul, Fuzz)
    .WithDomains(fuzzing::AnySmallValidNumericTensor(),
                 fuzzing::AnySmallValidNumericTensor());

}  // end namespace fuzzing
}  // end namespace tensorflow
