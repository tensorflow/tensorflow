/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/stream_executor/dnn.h"

#include "tensorflow/tsl/platform/test.h"

namespace stream_executor {
namespace {

TEST(StreamTest, AlgorithmDescToString) {
  dnn::AlgorithmDesc desc(17, {{12, 1}, {1, 0}, {3, 1}}, 0);
  EXPECT_EQ(desc.ToString(), "eng17{k1=0,k3=1,k12=1}");
}

}  // namespace
}  // namespace stream_executor
