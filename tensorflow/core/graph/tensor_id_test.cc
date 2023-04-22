/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/graph/tensor_id.h"
#include <vector>
#include "tensorflow/core/lib/random/simple_philox.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"

namespace tensorflow {
namespace {

string ParseHelper(const string& n) { return ParseTensorName(n).ToString(); }

TEST(TensorIdTest, ParseTensorName) {
  EXPECT_EQ(ParseHelper("W1"), "W1:0");
  EXPECT_EQ(ParseHelper("W1:0"), "W1:0");
  EXPECT_EQ(ParseHelper("weights:0"), "weights:0");
  EXPECT_EQ(ParseHelper("W1:1"), "W1:1");
  EXPECT_EQ(ParseHelper("W1:17"), "W1:17");
  EXPECT_EQ(ParseHelper("xyz1_17"), "xyz1_17:0");
  EXPECT_EQ(ParseHelper("^foo"), "^foo");
}

uint32 Skewed(random::SimplePhilox* rnd, int max_log) {
  const uint32 space = 1 << (rnd->Rand32() % (max_log + 1));
  return rnd->Rand32() % space;
}

void BM_ParseTensorName(::testing::benchmark::State& state) {
  const int arg = state.range(0);
  random::PhiloxRandom philox(301, 17);
  random::SimplePhilox rnd(&philox);
  std::vector<string> names;
  for (int i = 0; i < 100; i++) {
    string name;
    switch (arg) {
      case 0: {  // Generate random names
        size_t len = Skewed(&rnd, 4);
        while (name.size() < len) {
          name += rnd.OneIn(4) ? '0' : 'a';
        }
        if (rnd.OneIn(3)) {
          strings::StrAppend(&name, ":", rnd.Uniform(12));
        }
        break;
      }
      case 1:
        name = "W1";
        break;
      case 2:
        name = "t0003";
        break;
      case 3:
        name = "weights";
        break;
      case 4:
        name = "weights:17";
        break;
      case 5:
        name = "^weights";
        break;
      default:
        LOG(FATAL) << "Unexpected arg";
        break;
    }
    names.push_back(name);
  }

  TensorId id;
  int index = 0;
  int sum = 0;
  for (auto s : state) {
    id = ParseTensorName(names[index++ % names.size()]);
    sum += id.second;
  }
  VLOG(2) << sum;  // Prevent compiler from eliminating loop body
}
BENCHMARK(BM_ParseTensorName)->Arg(0)->Arg(1)->Arg(2)->Arg(3)->Arg(4)->Arg(5);

TEST(TensorIdTest, IsTensorIdControl) {
  string input = "^foo";
  TensorId tensor_id = ParseTensorName(input);
  EXPECT_TRUE(IsTensorIdControl(tensor_id));

  input = "foo";
  tensor_id = ParseTensorName(input);
  EXPECT_FALSE(IsTensorIdControl(tensor_id));

  input = "foo:2";
  tensor_id = ParseTensorName(input);
  EXPECT_FALSE(IsTensorIdControl(tensor_id));
}

TEST(TensorIdTest, PortZero) {
  for (string input : {"foo", "foo:0"}) {
    TensorId tensor_id = ParseTensorName(input);
    EXPECT_EQ("foo", tensor_id.node());
    EXPECT_EQ(0, tensor_id.index());
  }
}

}  // namespace
}  // namespace tensorflow
