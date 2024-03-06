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

#include "tensorflow/lite/experimental/genai/kvcache.h"

#include <vector>

#include <gtest/gtest.h>
#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace {

class SimpleCacheOpModel : public SingleOpModel {
 public:
  SimpleCacheOpModel(const TensorData& k_tensor, const TensorData& v_tensor) {
    k_ = AddInput(k_tensor);
    v_ = AddInput(v_tensor);
    kfull_ = AddOutput(k_tensor.type);
    vfull_ = AddOutput(v_tensor.type);
    SetCustomOp("KV_Cache", {}, ops::custom::Register_KV_CACHE);

    BuildInterpreter({GetShape(k_), GetShape(v_)});
  }

  void SetKey(const std::vector<float>& data) { PopulateTensor(k_, data); }
  void SetValue(const std::vector<float>& data) { PopulateTensor(v_, data); }
  void ResizeKey(const std::vector<int>& dims) {
    interpreter_->ResizeInputTensor(k_, dims);
  }
  void ResizeValue(const std::vector<int>& dims) {
    interpreter_->ResizeInputTensor(v_, dims);
  }
  std::vector<float> GetFullK() {
    const auto output = ExtractVector<float>(kfull_);
    return output;
  }

  std::vector<float> GetFullV() {
    const auto output = ExtractVector<float>(vfull_);
    return output;
  }

  TfLiteStatus ReAllocate() { return interpreter_->AllocateTensors(); }

 protected:
  int k_;
  int v_;
  int kfull_;
  int vfull_;
  int idx_;
};

TEST(SimpleCacheOp1Test, BasicTest) {
  SimpleCacheOpModel m({TensorType_FLOAT32, {1, 2, 2, 3}},
                       {TensorType_FLOAT32, {1, 2, 2, 3}});

  m.SetKey({{1, 0, -6, 2, 4, 3, 1, 0, -6, 2, 4, 3}});
  m.SetValue({{4, 2, -4, 2, 4, 2, 4, 2, -4, 2, 4, 2}});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  std::vector<float> fullk = m.GetFullK();
  std::vector<float> fullv = m.GetFullV();
  ASSERT_EQ(fullk.size(), 2 * 3 * 1024);
  ASSERT_EQ(fullv.size(), 2 * 3 * 1024);
}

TEST(SimpleCacheOp2Test, AddToCache) {
  SimpleCacheOpModel m({TensorType_FLOAT32, {1, 2, 2, 3}},
                       {TensorType_FLOAT32, {1, 2, 2, 3}});

  std::vector<float> key = {1, 5, -6, 2, 4, 3, 1, 5, -6, 2, 4, 3};
  m.SetKey(key);
  std::vector<float> value = {4, 2, -4, 2, 4, 2, 4, 2, -4, 2, 4, 2};
  m.SetValue(value);
  const int key_size = 2 * 3;
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  std::vector<float> fullk = m.GetFullK();
  std::vector<float> fullv = m.GetFullV();
  for (int i = 0; i < key.size(); ++i) {
    ASSERT_EQ(fullk[i], key[i]);
    ASSERT_EQ(fullv[i], value[i]);
  }
  for (int i = key.size(); i < fullk.size(); ++i) {
    ASSERT_EQ(fullk[i], 0.);
    ASSERT_EQ(fullv[i], 0.);
  }

  ASSERT_EQ(fullk.size(), 2 * 3 * 1024);
  ASSERT_EQ(fullv.size(), 2 * 3 * 1024);

  for (int i = 0; i < 510; i++) {
    ASSERT_EQ(m.Invoke(), kTfLiteOk);
  }

  fullk = m.GetFullK();
  fullv = m.GetFullV();

  for (int i = 0; i < 1022 * key_size; ++i) {
    ASSERT_NE(fullv[i], 0);
  }

  for (int i = 1022 * key_size; i < fullk.size(); ++i) {
    ASSERT_EQ(fullv[i], 0);
  }
}

TEST(SimpleCacheOp2Test, ShiftSlotsInCache) {
  SimpleCacheOpModel m({TensorType_FLOAT32, {1, 2, 2, 3}},
                       {TensorType_FLOAT32, {1, 2, 2, 3}});

  std::vector<float> key = {1, 5, -6, 2, 4, 3, 2, 6, -7, 3, 5, 4};
  m.SetKey(key);
  std::vector<float> value = {4, 2, -4, 2, 4, 2, 9, 8, -9, 8, 9, 1};
  m.SetValue(value);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  std::vector<float> fullk = m.GetFullK();
  std::vector<float> fullv = m.GetFullV();
  for (int i = 0; i < key.size(); ++i) {
    ASSERT_EQ(fullk[i], key[i]);
    ASSERT_EQ(fullv[i], value[i]);
  }
  for (int i = key.size(); i < fullk.size(); ++i) {
    ASSERT_EQ(fullk[i], 0.);
    ASSERT_EQ(fullv[i], 0.);
  }
  ASSERT_EQ(fullk.size(), 2 * 3 * 1024);
  ASSERT_EQ(fullv.size(), 2 * 3 * 1024);

  // Now fill up the cache
  for (int i = 0; i < 511; i++) {
    ASSERT_EQ(m.Invoke(), kTfLiteOk);
  }

  fullk = m.GetFullK();
  fullv = m.GetFullV();

  for (int i = 0; i < fullk.size(); ++i) {
    ASSERT_NE(fullk[i], 0);
    ASSERT_NE(fullv[i], 0);
  }

  for (int j = 0; j < 6; ++j) {
    int idxfull = fullk.size() - 6 + j;
    int idx = 6 + j;
    ASSERT_EQ(fullk[idxfull], key[idx]);
    ASSERT_EQ(fullv[idxfull], value[idx]);
  }

  std::vector<float> key2 = {1, 1, 1, 1, 1, 1, 7, 7, 7, 7, 7, 7};
  m.SetKey(key2);
  std::vector<float> value2 = {8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9};
  m.SetValue(value2);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);

  fullk = m.GetFullK();
  fullv = m.GetFullV();

  for (int j = 0; j < 12; ++j) {
    int idxfull = fullk.size() - 12 + j;
    ASSERT_EQ(fullk[idxfull], key2[j]);
    ASSERT_EQ(fullv[idxfull], value2[j]);
  }

  // Resize to a single entry. Add to a full cache and verify
  // the cached contents.
  m.ResizeKey({1, 1, 2, 3});
  m.ResizeValue({1, 1, 2, 3});
  m.ReAllocate();

  std::vector<float> key3 = {4, 4, 4, 4, 4, 4};
  m.SetKey(key3);
  std::vector<float> value3 = {2, 2, 2, 2, 2, 2};
  m.SetValue(value3);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);

  fullk = m.GetFullK();
  fullv = m.GetFullV();

  for (int j = 0; j < 6; ++j) {
    int idxfull = fullk.size() - 6 + j;
    ASSERT_EQ(fullk[idxfull], key3[j]);
    ASSERT_EQ(fullv[idxfull], value3[j]);
  }
}

}  // namespace
}  // namespace tflite
