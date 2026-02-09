// Copyright 2025 TF.Text Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/shape_inference_testutil.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

TEST(WordpieceTokenizeWithOffsetsOpTest, ShapeFn) {
  // WordpieceTokenizeWithOffsets(input_values, vocab_lookup_table) ->
  //     [output_values, output_row_lengths, start_values, limit_values]
  ShapeInferenceTestOp op("WordpieceTokenizeWithOffsets");
  auto &attr = *op.node_def.mutable_attr();

  attr["output_row_partition_type"].set_s("row_lengths");
  INFER_OK(op, "?;?", "[?];[?];[?];[?]");
  INFER_OK(op, "[?];?", "[?];[d0_0];[?];[?]");
  INFER_OK(op, "[?];[]", "[?];[d0_0];[?];[?]");
  INFER_OK(op, "[5];?", "[?];[d0_0];[?];[?]");
  INFER_OK(op, "[5];[]", "[?];[d0_0];[?];[?]");
  INFER_ERROR("Shape must be rank 1 but is rank 0", op, "[];?");
  INFER_ERROR("Shape must be rank 1 but is rank 2", op, "[1,2];?");
  INFER_ERROR("Shape must be rank 0 but is rank 1", op, "?;[1]");

  attr["output_row_partition_type"].set_s("row_splits");
  INFER_OK(op, "?;?", "[?];[?];[?];[?]");
  INFER_OK(op, "[?];?", "[?];[?];[?];[?]");
  INFER_OK(op, "[?];[]", "[?];[?];[?];[?]");
  INFER_OK(op, "[5];?", "[?];[6];[?];[?]");
  INFER_OK(op, "[5];[]", "[?];[6];[?];[?]");
}

}  // namespace
}  // namespace tensorflow
