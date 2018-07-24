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

#include <numeric>
#include <vector>

#include "tensorflow/compiler/xla/tests/client_library_test_base.h"
#include "tensorflow/core/lib/core/errors.h"

namespace xla {
namespace {

class IotaTest : public ClientLibraryTestBase {
 public:
  explicit IotaTest(se::Platform* platform = nullptr)
      : ClientLibraryTestBase(platform) {}
  template <typename T>
  std::vector<T> GetExpected(const int64 num_elements) {
    std::vector<T> result(num_elements);
    std::iota(result.begin(), result.end(), 0);
    return result;
  }
};

TEST_F(IotaTest, SimpleR1) {
  for (int num_elements = 1; num_elements < 10000001; num_elements *= 10) {
    {
      XlaBuilder builder(TestName() + "_f32");
      IotaGen(&builder, F32, num_elements);
      ComputeAndCompareR1<float>(&builder, GetExpected<float>(num_elements), {},
                                 ErrorSpec{0.0001});
    }
    {
      XlaBuilder builder(TestName() + "_u32");
      IotaGen(&builder, U32, num_elements);
      ComputeAndCompareR1<uint32>(&builder, GetExpected<uint32>(num_elements),
                                  {});
    }
    {
      XlaBuilder builder(TestName() + "_s32");
      IotaGen(&builder, S32, num_elements);
      ComputeAndCompareR1<int32>(&builder, GetExpected<int32>(num_elements),
                                 {});
    }
  }
}

}  // namespace
}  // namespace xla
