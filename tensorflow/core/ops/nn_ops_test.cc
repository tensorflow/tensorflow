/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");

You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference_testutil.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {

TEST(ArrayOpsTest, TopK_ShapeFn) {
  std::unique_ptr<NodeDef> def_storage(new NodeDef);
  NodeDef* def = def_storage.get();
  auto set_k = [def](int k) {
    TF_CHECK_OK(NodeDefBuilder("test", "Pack")
                    .Input({{"a", 0, DT_FLOAT}})
                    .Attr("k", k)
                    .Finalize(def));
  };
  const char op[] = "TopK";

  set_k(20);
  // With known input, each output is an unknown shape.
  INFER_OK_WITH_DEF(op, def, "?", "?;?");
  // With vector input, each output is [k].
  INFER_OK_WITH_DEF(op, def, "[20]", "[20];[20]");
  INFER_OK_WITH_DEF(op, def, "[21]", "[20];[20]");

  // With input rank 3, each output is the two first 2 dims of input, plus k.
  INFER_OK_WITH_DEF(op, def, "[1,?,21]", "[d0_0,d0_1,20];[d0_0,d0_1,20]");
  // With input rank 4, each output is the two first 3 dims of input, plus k.
  INFER_OK_WITH_DEF(op, def, "[1,?,21,?]",
                    "[d0_0,d0_1,d0_2,20];[d0_0,d0_1,d0_2,20]");

  INFER_ERROR_WITH_DEF("Shape must be at least rank 1 but is rank 0", op, def,
                       "[]");
  INFER_ERROR_WITH_DEF("input must have last dimension >= k = 20 but is 1", op,
                       def, "[1]");
  INFER_ERROR_WITH_DEF("input must have last dimension >= k = 20 but is 4", op,
                       def, "[1,2,3,4]");
  set_k(-1);
  INFER_ERROR_WITH_DEF("Need k >= 0, got -1", op, def, "[1,2,3,4]");
}

TEST(ArrayOpsTest, TopKV2_ShapeFn) {
  std::vector<const Tensor*> in_tensors{nullptr, nullptr};
  const char op[] = "TopKV2";

  Tensor k_t;
  in_tensors[1] = &k_t;

  k_t = test::AsScalar<int32>(20);
  // With known input, each output is an unknown shape.
  INFER_OK_WITH_TENSORS(op, "?;[]", in_tensors, "?;?");
  // With vector input, each output is [k].
  INFER_OK_WITH_TENSORS(op, "[20];[]", in_tensors, "[20];[20]");

  // With input rank 3, each output is the two first 2 dims of input, plus k.
  INFER_OK_WITH_TENSORS(op, "[1,?,21];[]", in_tensors,
                        "[d0_0,d0_1,20];[d0_0,d0_1,20]");
  // With input rank 4, each output is the two first 3 dims of input, plus k.
  INFER_OK_WITH_TENSORS(op, "[1,?,21,?];[]", in_tensors,
                        "[d0_0,d0_1,d0_2,20];[d0_0,d0_1,d0_2,20]");

  INFER_ERROR_WITH_TENSORS("Shape must be at least rank 1 but is rank 0", op,
                           "[];[]", in_tensors);
  INFER_ERROR_WITH_TENSORS("input must have last dimension >= k = 20 but is 1",
                           op, "[1];[]", in_tensors);
  INFER_ERROR_WITH_TENSORS("input must have last dimension >= k = 20 but is 4",
                           op, "[1,2,3,4];[]", in_tensors);
  k_t = test::AsScalar<int32>(-1);
  INFER_ERROR_WITH_TENSORS(
      "Dimension size, given by scalar input 1, must be non-negative but is -1",
      op, "[1,2,3,4];[]", in_tensors);
}

}  // end namespace tensorflow
