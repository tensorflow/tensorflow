/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/data/service/test_util.h"

#include "tensorflow/core/data/standalone.h"
#include "tensorflow/core/kernels/data/dataset_test_base.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace data {
namespace test_util {

TEST(TestUtil, MapTestCase) {
  GraphDefTestCase test_case;
  TF_ASSERT_OK(map_test_case(&test_case));
  standalone::Dataset::Params params;
  std::unique_ptr<standalone::Dataset> dataset;
  TF_ASSERT_OK(
      standalone::Dataset::FromGraph(params, test_case.graph_def, &dataset));

  std::unique_ptr<standalone::Iterator> iterator;
  TF_ASSERT_OK(dataset->MakeIterator(&iterator));

  bool end_of_input = false;

  std::vector<std::vector<Tensor>> result;
  while (!end_of_input) {
    std::vector<tensorflow::Tensor> outputs;
    TF_ASSERT_OK(iterator->GetNext(&outputs, &end_of_input));
    if (!end_of_input) {
      result.push_back(outputs);
    }
  }
  ASSERT_EQ(result.size(), test_case.output.size());
  for (int i = 0; i < result.size(); ++i) {
    TF_EXPECT_OK(DatasetOpsTestBase::ExpectEqual(result[i], test_case.output[i],
                                                 /*compare_order=*/true));
  }
}

}  // namespace test_util
}  // namespace data
}  // namespace tensorflow
