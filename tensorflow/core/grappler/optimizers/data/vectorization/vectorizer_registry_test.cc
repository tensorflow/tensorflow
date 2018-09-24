/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/grappler/optimizers/data/vectorization/vectorizer_registry.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace grappler {
namespace vectorization_utils {

class TestVectorizer : public Vectorizer {
 public:
  Status Vectorize(const NodeDef& node, gtl::ArraySlice<string> inputs,
                   FunctionDef* outer_scope,
                   std::map<string, string>* conversion_map) override {
    return Status::OK();
  }
};

REGISTER_VECTORIZER("test_op", TestVectorizer);

TEST(TestVectorizer, TestTestVectorizer) {
  EXPECT_EQ(VectorizerRegistry::Global()->Get("nonexistent"), nullptr);

  auto vectorizer = VectorizerRegistry::Global()->Get("test_op");
  EXPECT_NE(vectorizer, nullptr);

  FunctionDef function;
  NodeDef node;
  std::map<string, string> conversion_map;
  EXPECT_TRUE(vectorizer->Vectorize(node, {}, &function, &conversion_map).ok());
}

}  // namespace vectorization_utils
}  // namespace grappler
}  // namespace tensorflow
