/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <numeric>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/types/span.h"
#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/experimental/genai/genai_ops.h"
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/util.h"

namespace tflite {
namespace {

using ::testing::ElementsAreArray;
using ::testing::TestWithParam;

enum class TestType {
  kSharedKV = 0,
  kPingPongKV = 1,
};

class ExternalKVSingleOpModel : public SingleOpModel {
 public:
  ExternalKVSingleOpModel(const TensorData& k_cache, const TensorData& v_cache,
                          const TensorData& pos_tensor,
                          const TensorData& k_slice, const TensorData& v_slice,
                          TestType test_type)
      : test_type_(test_type) {
    k_cache_in_ = AddInput(k_cache);
    v_cache_in_ = AddInput(v_cache);
    position_ = AddInput(pos_tensor);
    k_slice_ = AddInput(k_slice);
    v_slice_ = AddInput(v_slice);
    k_cache_out_ = AddOutput(k_cache);
    v_cache_out_ = AddOutput(v_cache);

    auto get_padded_cache_size = [&](const TensorData& cache) {
      size_t size = static_cast<size_t>(std::accumulate(
          cache.shape.begin(), cache.shape.end(), 1.0, std::multiplies<>()));
      // This added padding is to ensure that we can get an aligned buffer
      // for the cache.
      return size + kDefaultTensorAlignment;
    };

    SetCustomOp("EKV_Cache", {}, ops::custom::Register_EXTERNAL_KV_CACHE);
    BuildInterpreter({GetShape(k_cache_in_), GetShape(v_cache_in_),
                      GetShape(position_), GetShape(k_slice_),
                      GetShape(v_slice_)});
    k_cache_1_.resize(get_padded_cache_size(k_cache), 0.0);
    v_cache_1_.resize(get_padded_cache_size(v_cache), 0.0);
    k_cache_2_.resize(get_padded_cache_size(k_cache), 0.0);
    v_cache_2_.resize(get_padded_cache_size(v_cache), 0.0);
  }

  TfLiteStatus Run(absl::Span<const int32_t> position,
                   absl::Span<const float> k_slice,
                   absl::Span<const float> v_slice) {
    if (test_type_ == TestType::kSharedKV) {
      TF_LITE_ENSURE_STATUS(SharedBufferPrepare());
    } else {
      TF_LITE_ENSURE_STATUS(PingPongBufferPrepare());
    }
    PopulateTensor(position_, position);
    PopulateTensor(k_slice_, k_slice);
    PopulateTensor(v_slice_, v_slice);

    return SingleOpModel::Invoke();
  };

  std::vector<float> GetKCache() { return ExtractVector<float>(k_cache_out_); }

  std::vector<float> GetVCache() { return ExtractVector<float>(v_cache_out_); }

 protected:
  TfLiteStatus SharedBufferPrepare() {
    TF_LITE_ENSURE_STATUS(
        SetCustomAllocationFromCache(k_cache_1_, k_cache_in_));
    TF_LITE_ENSURE_STATUS(
        SetCustomAllocationFromCache(v_cache_1_, v_cache_in_));
    TF_LITE_ENSURE_STATUS(
        SetCustomAllocationFromCache(k_cache_1_, k_cache_out_));
    TF_LITE_ENSURE_STATUS(
        SetCustomAllocationFromCache(v_cache_1_, v_cache_out_));
    return interpreter_->AllocateTensors();
  }

  TfLiteStatus PingPongBufferPrepare() {
    // The ping-pong buffer is useful for cases where source and destination
    // buffers cannot be the same (e.g., GPU).
    std::vector<float>* input_k_caches;
    std::vector<float>* input_v_caches;
    std::vector<float>* output_k_caches;
    std::vector<float>* output_v_caches;
    if (kv_flop_) {
      input_k_caches = &k_cache_1_;
      input_v_caches = &v_cache_1_;
      output_k_caches = &k_cache_2_;
      output_v_caches = &v_cache_2_;
    } else {
      input_k_caches = &k_cache_2_;
      input_v_caches = &v_cache_2_;
      output_k_caches = &k_cache_1_;
      output_v_caches = &v_cache_1_;
    }
    kv_flop_ = !kv_flop_;

    TF_LITE_ENSURE_STATUS(
        SetCustomAllocationFromCache(*input_k_caches, k_cache_in_));
    TF_LITE_ENSURE_STATUS(
        SetCustomAllocationFromCache(*input_v_caches, v_cache_in_));
    TF_LITE_ENSURE_STATUS(
        SetCustomAllocationFromCache(*output_k_caches, k_cache_out_));
    TF_LITE_ENSURE_STATUS(
        SetCustomAllocationFromCache(*output_v_caches, v_cache_out_));
    return interpreter_->AllocateTensors();
  }

  TfLiteStatus SetCustomAllocationFromCache(std::vector<float>& cache,
                                            int tensor_index) {
    size_t total_bytes = cache.size() * sizeof(float);
    size_t required_number_of_bytes = total_bytes - kDefaultTensorAlignment;
    void* original_buffer = static_cast<void*>(cache.data());
    void* aligned_buffer =
        std::align(kDefaultTensorAlignment, required_number_of_bytes,
                   original_buffer, total_bytes);
    if (aligned_buffer == nullptr ||
        reinterpret_cast<intptr_t>(aligned_buffer) % kDefaultTensorAlignment !=
            0) {
      return kTfLiteError;
    }
    TfLiteCustomAllocation allocation = {.data = aligned_buffer,
                                         .bytes = required_number_of_bytes};
    return interpreter_->SetCustomAllocationForTensor(tensor_index, allocation);
  };

  int position_;
  int k_cache_in_;
  int v_cache_in_;
  int k_slice_;
  int v_slice_;
  int k_cache_out_;
  int v_cache_out_;
  TestType test_type_;
  std::vector<float> k_cache_1_;
  std::vector<float> v_cache_1_;
  std::vector<float> k_cache_2_;
  std::vector<float> v_cache_2_;
  bool kv_flop_ = true;
};

class EKVCacheTest : public TestWithParam<TestType> {};

TEST_P(EKVCacheTest, SingleSliceUpdateTest) {
  ExternalKVSingleOpModel m(
      {TensorType_FLOAT32, {1, 3, 2, 2}}, {TensorType_FLOAT32, {1, 3, 2, 2}},
      {TensorType_INT32, {1}}, {TensorType_FLOAT32, {1, 1, 2, 2}},
      {TensorType_FLOAT32, {1, 1, 2, 2}}, GetParam());
  {
    ASSERT_EQ(m.Run(/*position=*/{0}, /*k_slice=*/{10, 11, 12, 13},
                    /*v_slice=*/{20, 21, 22, 23}),
              kTfLiteOk);

    std::vector<float> k = m.GetKCache();
    ASSERT_THAT(k, ElementsAreArray({10, 11, 12, 13, 0, 0, 0, 0, 0, 0, 0, 0}));

    std::vector<float> v = m.GetVCache();
    ASSERT_THAT(v, ElementsAreArray({20, 21, 22, 23, 0, 0, 0, 0, 0, 0, 0, 0}));
  }
  {
    ASSERT_EQ(m.Run(/*position=*/{2}, /*k_slice=*/{50, 51, 52, 53},
                    /*v_slice=*/{60, 61, 62, 63}),
              kTfLiteOk);

    std::vector<float> k = m.GetKCache();
    ASSERT_THAT(k,
                ElementsAreArray({10, 11, 12, 13, 0, 0, 0, 0, 50, 51, 52, 53}));

    std::vector<float> v = m.GetVCache();
    ASSERT_THAT(v,
                ElementsAreArray({20, 21, 22, 23, 0, 0, 0, 0, 60, 61, 62, 63}));
  }
  {
    ASSERT_EQ(m.Run(/*position=*/{1}, /*k_slice=*/{70, 71, 72, 73},
                    /*v_slice=*/{80, 81, 82, 83}),
              kTfLiteOk);

    std::vector<float> k = m.GetKCache();
    ASSERT_THAT(
        k, ElementsAreArray({10, 11, 12, 13, 70, 71, 72, 73, 50, 51, 52, 53}));

    std::vector<float> v = m.GetVCache();
    ASSERT_THAT(
        v, ElementsAreArray({20, 21, 22, 23, 80, 81, 82, 83, 60, 61, 62, 63}));
  }
  {
    ASSERT_EQ(m.Run(/*position=*/{1}, /*k_slice=*/{1, 2, 3, 4},
                    /*v_slice=*/{1, 2, 3, 4}),
              kTfLiteOk);

    std::vector<float> k = m.GetKCache();
    ASSERT_THAT(k,
                ElementsAreArray({10, 11, 12, 13, 1, 2, 3, 4, 50, 51, 52, 53}));

    std::vector<float> v = m.GetVCache();
    ASSERT_THAT(v,
                ElementsAreArray({20, 21, 22, 23, 1, 2, 3, 4, 60, 61, 62, 63}));
  }
}

TEST_P(EKVCacheTest, MultipleSliceUpdateTest) {
  ExternalKVSingleOpModel m(
      {TensorType_FLOAT32, {1, 3, 2, 2}}, {TensorType_FLOAT32, {1, 3, 2, 2}},
      {TensorType_INT32, {2}}, {TensorType_FLOAT32, {1, 2, 2, 2}},
      {TensorType_FLOAT32, {1, 2, 2, 2}}, GetParam());
  {
    ASSERT_EQ(m.Run(/*position=*/{0, 1}, /*k_slice=*/{1, 1, 1, 1, 2, 2, 2, 2},
                    /*v_slice=*/{5, 5, 5, 5, 6, 6, 6, 6}),
              kTfLiteOk);

    std::vector<float> k = m.GetKCache();
    ASSERT_THAT(k, ElementsAreArray({1, 1, 1, 1, 2, 2, 2, 2, 0, 0, 0, 0}));

    std::vector<float> v = m.GetVCache();
    ASSERT_THAT(v, ElementsAreArray({5, 5, 5, 5, 6, 6, 6, 6, 0, 0, 0, 0}));
  }
  {
    ASSERT_EQ(
        m.Run(/*position=*/{1, 2}, /*k_slice=*/{10, 10, 10, 10, 11, 11, 11, 11},
              /*v_slice=*/{20, 20, 20, 20, 21, 21, 21, 21}),
        kTfLiteOk);

    std::vector<float> k = m.GetKCache();
    ASSERT_THAT(k,
                ElementsAreArray({1, 1, 1, 1, 10, 10, 10, 10, 11, 11, 11, 11}));

    std::vector<float> v = m.GetVCache();
    ASSERT_THAT(v,
                ElementsAreArray({5, 5, 5, 5, 20, 20, 20, 20, 21, 21, 21, 21}));
  }
}

TEST_P(EKVCacheTest, FailsOnOutOfBoundPosition) {
  ExternalKVSingleOpModel m(
      {TensorType_FLOAT32, {1, 3, 2, 2}}, {TensorType_FLOAT32, {1, 3, 2, 2}},
      {TensorType_INT32, {1}}, {TensorType_FLOAT32, {1, 1, 2, 2}},
      {TensorType_FLOAT32, {1, 1, 2, 2}}, GetParam());
  ASSERT_EQ(m.Run(/*position=*/{3}, /*k_slice=*/{1, 2, 3, 4},
                  /*v_slice=*/{1, 2, 3, 4}),
            kTfLiteError);
}

INSTANTIATE_TEST_SUITE_P(EKVCacheTest, EKVCacheTest,
                         testing::Values(TestType::kSharedKV,
                                         TestType::kPingPongKV));

}  // namespace
}  // namespace tflite
