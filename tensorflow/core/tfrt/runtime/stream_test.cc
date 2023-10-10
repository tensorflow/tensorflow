/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/tfrt/runtime/stream.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/memory/memory.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/tfrt/saved_model/saved_model_testutil.h"
#include "tensorflow/core/tfrt/utils/thread_pool.h"
#include "tsl/platform/env.h"
#include "tsl/platform/statusor.h"

namespace tensorflow {
namespace tfrt_stub {
namespace {

using ::tensorflow::test::AsTensor;
using ::testing::AnyOf;
using ::testing::ElementsAreArray;
using ::testing::Pair;
using ::testing::UnorderedElementsAre;

class TestStreamInterface : public StreamControllerInterface {
 public:
  TestStreamInterface() : StreamControllerInterface("test_address") {}
};

const bool kUnused = []() {
  GetGlobalStreamInterfaceFactory().RegisterController(
      []() { return std::make_unique<TestStreamInterface>(); });
  return true;
}();

TEST(StreamTest, Simple) {
  StreamCallbackId callback_id(1234);
  StepId step_id(5678);

  std::vector<absl::flat_hash_map<std::string, tensorflow::Tensor>> outputs;

  {
    TF_ASSERT_OK_AND_ASSIGN(
        auto scoped_stream_callback,
        GetGlobalStreamCallbackRegistry().Register(
            "test_model", callback_id, step_id,
            [&](absl::flat_hash_map<std::string, tensorflow::Tensor> arg) {
              outputs.push_back(std::move(arg));
            }));

    std::vector<absl::flat_hash_map<std::string, tensorflow::Tensor>> expected =
        {{{"a", AsTensor<int32_t>({100})}, {"b", AsTensor<int32_t>({200})}},
         {{"c", AsTensor<int32_t>({300})}}};
    auto thread = absl::WrapUnique(tsl::Env::Default()->StartThread(
        tsl::ThreadOptions(), "fake_stream_client", [&]() {
          for (const auto& map : expected) {
            TfThreadPool thread_pool(/*name=*/"test", /*num_threads=*/4);
            CHECK_OK(GetGlobalStreamCallbackRegistry().Invoke(
                &thread_pool, callback_id, step_id, {map, absl::Now()}));
          }
        }));
  }

  EXPECT_EQ(outputs.size(), 2);
  EXPECT_THAT(GetTfTensorData<int32_t>(outputs[0]["a"]),
              ElementsAreArray({100}));
  EXPECT_THAT(GetTfTensorData<int32_t>(outputs[0]["b"]),
              ElementsAreArray({200}));
  EXPECT_THAT(GetTfTensorData<int32_t>(outputs[1]["c"]),
              ElementsAreArray({300}));
}

TEST(StreamTest, MultipleWriters) {
  StreamCallbackId callback_id(1234);
  StepId step_id(5678);

  std::vector<absl::flat_hash_map<std::string, std::vector<int32_t>>> outputs;

  {
    TfThreadPool thread_pool(/*name=*/"test", /*num_threads=*/4);
    TF_ASSERT_OK_AND_ASSIGN(
        auto scoped_stream_callback,
        GetGlobalStreamCallbackRegistry().Register(
            "test_model", callback_id, step_id,
            [&](absl::flat_hash_map<std::string, tensorflow::Tensor> arg) {
              absl::flat_hash_map<std::string, std::vector<int32_t>> out;
              for (const auto& p : arg) {
                out[p.first] = GetTfTensorData<int32_t>(p.second);
              }
              outputs.push_back(std::move(out));
            }));

    std::vector<absl::flat_hash_map<std::string, tensorflow::Tensor>> expected =
        {{{"a", AsTensor<int32_t>({100})}, {"b", AsTensor<int32_t>({200})}},
         {{"c", AsTensor<int32_t>({300})}}};

    for (const auto& p : expected) {
      tsl::Env::Default()->SchedClosure([&, callback_id, step_id, p]() {
        TfThreadPool thread_pool(/*name=*/"test", /*num_threads=*/4);
        // The stream callback may be dropped early, and in that case we ignore
        // the error.
        GetGlobalStreamCallbackRegistry()
            .Invoke(&thread_pool, callback_id, step_id, {p, absl::Now()})
            .IgnoreError();
      });
    }

    absl::SleepFor(absl::Microseconds(100));
  }

  LOG(INFO) << "StreamCallback receives " << outputs.size() << " outputs.";

  for (const auto& output : outputs) {
    EXPECT_THAT(
        output,
        AnyOf(UnorderedElementsAre(Pair("a", ElementsAreArray({100})),
                                   Pair("b", ElementsAreArray({200}))),
              UnorderedElementsAre(Pair("c", ElementsAreArray({300})))));
  }
}

}  // namespace
}  // namespace tfrt_stub
}  // namespace tensorflow
