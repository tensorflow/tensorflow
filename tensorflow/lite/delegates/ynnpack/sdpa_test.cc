/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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

#include <cmath>
#include <memory>
#include <vector>

#include <gtest/gtest.h>
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/delegates/ynnpack/ynnpack_delegate.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/test_util.h"

namespace tflite {
namespace ynnpack {
namespace {

class SdpaModel : public SingleOpModel {
 public:
  SdpaModel(const TensorData& q_tensor, const TensorData& k_tensor,
            const TensorData& v_tensor, const TensorData& output_tensor) {
    q_id_ = AddInput(q_tensor);
    k_id_ = AddInput(k_tensor);
    v_id_ = AddInput(v_tensor);
    out_id_ = AddOutput(output_tensor);

    SetCustomOp("odml.scaled_dot_product_attention", {},
                /*register_fn=*/[]() -> TfLiteRegistration* {
                  static TfLiteRegistration reg = {
                      /*init=*/nullptr,
                      /*free=*/nullptr,
                      /*prepare=*/nullptr,
                      /*invoke=*/nullptr,
                      /*profiling_string=*/nullptr,
                      /*builtin_code=*/0,
                      "odml.scaled_dot_product_attention",
                      /*version=*/1};
                  return &reg;
                });

    BuildInterpreter({GetShape(q_id_), GetShape(k_id_), GetShape(v_id_)});
  }

  void SetQuery(const std::vector<float>& data) {
    PopulateTensor<float>(q_id_, data);
  }
  void SetKey(const std::vector<float>& data) {
    PopulateTensor<float>(k_id_, data);
  }
  void SetValue(const std::vector<float>& data) {
    PopulateTensor<float>(v_id_, data);
  }

  TfLiteStatus ApplyCustomDelegate(TfLiteDelegate* delegate) {
    return interpreter_->ModifyGraphWithDelegate(delegate);
  }

  TfLiteStatus ResizeSequence(int new_seq_len) {
    TF_LITE_ENSURE_STATUS(
        interpreter_->ResizeInputTensor(q_id_, {1, new_seq_len, 1, 4}));
    TF_LITE_ENSURE_STATUS(
        interpreter_->ResizeInputTensor(k_id_, {1, new_seq_len, 1, 4}));
    TF_LITE_ENSURE_STATUS(
        interpreter_->ResizeInputTensor(v_id_, {1, new_seq_len, 1, 4}));
    return interpreter_->AllocateTensors();
  }

  std::vector<float> GetOutput() { return ExtractVector<float>(out_id_); }

  int query_id() const { return q_id_; }
  int key_id() const { return k_id_; }
  int value_id() const { return v_id_; }

 private:
  int q_id_;
  int k_id_;
  int v_id_;
  int out_id_;
};

TEST(YNNPackDelegateSdpaTest, BasicAttentionDelegation) {
  // Shape: [B, S, H, D] = [1, 2, 1, 4]
  SdpaModel model(
      {TensorType_FLOAT32, {1, 2, 1, 4}}, {TensorType_FLOAT32, {1, 2, 1, 4}},
      {TensorType_FLOAT32, {1, 2, 1, 4}}, {TensorType_FLOAT32, {1, 2, 1, 4}});

  // Apply YNNPACK Delegate
  TfLiteYNNPackDelegateOptions options = TfLiteYNNPackDelegateOptionsDefault();
  options.num_threads = 1;
  auto delegate = std::unique_ptr<TfLiteDelegate, void (*)(TfLiteDelegate*)>(
      TfLiteYNNPackDelegateCreate(&options), TfLiteYNNPackDelegateDelete);

  ASSERT_EQ(model.ApplyCustomDelegate(delegate.get()), kTfLiteOk);

  model.SetQuery({1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f});
  model.SetKey({1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f});
  model.SetValue({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f});

  ASSERT_EQ(model.Invoke(), kTfLiteOk);

  auto output = model.GetOutput();
  EXPECT_EQ(output.size(), 8);
  for (float val : output) {
    EXPECT_FALSE(std::isnan(val));
  }
}

class SdpaModelWithParams : public SingleOpModel {
 public:
  SdpaModelWithParams(const TensorData& q_tensor, const TensorData& k_tensor,
                      const TensorData& v_tensor, const TensorData& p_tensor,
                      const TensorData& output_tensor) {
    q_id_ = AddInput(q_tensor);
    k_id_ = AddInput(k_tensor);
    v_id_ = AddInput(v_tensor);
    p_id_ = AddInput(p_tensor);
    out_id_ = AddOutput(output_tensor);

    SetCustomOp("odml.scaled_dot_product_attention", {},
                /*register_fn=*/[]() -> TfLiteRegistration* {
                  static TfLiteRegistration reg = {
                      /*init=*/nullptr,
                      /*free=*/nullptr,
                      /*prepare=*/nullptr,
                      /*invoke=*/nullptr,
                      /*profiling_string=*/nullptr,
                      /*builtin_code=*/0,
                      "odml.scaled_dot_product_attention",
                      /*version=*/1};
                  return &reg;
                });

    BuildInterpreter(
        {GetShape(q_id_), GetShape(k_id_), GetShape(v_id_), GetShape(p_id_)});
  }

  void SetQuery(const std::vector<float>& data) {
    PopulateTensor<float>(q_id_, data);
  }
  void SetKey(const std::vector<float>& data) {
    PopulateTensor<float>(k_id_, data);
  }
  void SetValue(const std::vector<float>& data) {
    PopulateTensor<float>(v_id_, data);
  }
  void SetParams(const std::vector<int32_t>& data) {
    PopulateTensor<int32_t>(p_id_, data);
  }

  TfLiteStatus ApplyCustomDelegate(TfLiteDelegate* delegate) {
    return interpreter_->ModifyGraphWithDelegate(delegate);
  }

  std::vector<float> GetOutput() { return ExtractVector<float>(out_id_); }

 private:
  int q_id_;
  int k_id_;
  int v_id_;
  int p_id_;
  int out_id_;
};

TEST(YNNPackDelegateSdpaTest, DynamicSequenceLengthDelegation) {
  // Query: [B, S_q, H, D] = [1, 1, 1, 4]
  // Key/Value: [B, S_kv, H, D] = [1, 4, 1, 4] (full capacity = 4 tokens)
  // Params: [1] (int32 tensor representing active tokens count)
  SdpaModelWithParams model(
      {TensorType_FLOAT32, {1, 1, 1, 4}}, {TensorType_FLOAT32, {1, 4, 1, 4}},
      {TensorType_FLOAT32, {1, 4, 1, 4}}, {TensorType_INT32, {1}},
      {TensorType_FLOAT32, {1, 1, 1, 4}});

  TfLiteYNNPackDelegateOptions options = TfLiteYNNPackDelegateOptionsDefault();
  options.num_threads = 1;
  auto delegate = std::unique_ptr<TfLiteDelegate, void (*)(TfLiteDelegate*)>(
      TfLiteYNNPackDelegateCreate(&options), TfLiteYNNPackDelegateDelete);

  ASSERT_EQ(model.ApplyCustomDelegate(delegate.get()), kTfLiteOk);

  // Set Query to unit vector [1, 0, 0, 0]
  model.SetQuery({1.0f, 0.0f, 0.0f, 0.0f});

  // Set Key tokens: tokens 0 and 1 are unit vector [1, 0, 0, 0]
  // Tokens 2 and 3 are HUGE vectors [1000, 0, 0, 0] that would dominate
  // attention if active!
  model.SetKey({
      1.0f, 0.0f, 0.0f, 0.0f,     // Token 0
      1.0f, 0.0f, 0.0f, 0.0f,     // Token 1
      1000.0f, 0.0f, 0.0f, 0.0f,  // Token 2 (padding)
      1000.0f, 0.0f, 0.0f, 0.0f   // Token 3 (padding)
  });

  // Set Value tokens: tokens 0 and 1 are [10, 20, 30, 40]
  // Tokens 2 and 3 are massive numerical outliers [9999, 9999, 9999, 9999]
  model.SetValue({
      10.0f, 20.0f, 30.0f, 40.0f,          // Token 0
      10.0f, 20.0f, 30.0f, 40.0f,          // Token 1
      9999.0f, 9999.0f, 9999.0f, 9999.0f,  // Token 2 (padding)
      9999.0f, 9999.0f, 9999.0f, 9999.0f   // Token 3 (padding)
  });

  // Step 1: Set active tokens = 2. Notice tokens 2 and 3 should be ignored!
  model.SetParams({2});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  auto output_step1 = model.GetOutput();
  ASSERT_EQ(output_step1.size(), 4);
  EXPECT_NEAR(output_step1[0], 10.0f, 1e-4);
  EXPECT_NEAR(output_step1[1], 20.0f, 1e-4);
  EXPECT_NEAR(output_step1[2], 30.0f, 1e-4);
  EXPECT_NEAR(output_step1[3], 40.0f, 1e-4);

  // Step 2: Dynamically change active tokens = 1 (without modifying graph or
  // re-allocating!)
  model.SetParams({1});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  auto output_step2 = model.GetOutput();
  ASSERT_EQ(output_step2.size(), 4);
  EXPECT_NEAR(output_step2[0], 10.0f, 1e-4);
  EXPECT_NEAR(output_step2[1], 20.0f, 1e-4);
  EXPECT_NEAR(output_step2[2], 30.0f, 1e-4);
  EXPECT_NEAR(output_step2[3], 40.0f, 1e-4);
}

TEST(YNNPackDelegateSdpaTest, ActualDynamicSequenceResizing) {
  // Start with shape: [B, S, H, D] = [1, 2, 1, 4]
  SdpaModel model(
      {TensorType_FLOAT32, {1, 2, 1, 4}}, {TensorType_FLOAT32, {1, 2, 1, 4}},
      {TensorType_FLOAT32, {1, 2, 1, 4}}, {TensorType_FLOAT32, {1, 2, 1, 4}});

  // Apply YNNPACK Delegate with static_shape = false for dynamic resizing
  TfLiteYNNPackDelegateOptions options = TfLiteYNNPackDelegateOptionsDefault();
  options.num_threads = 1;
  options.static_shape = false;
  auto delegate = std::unique_ptr<TfLiteDelegate, void (*)(TfLiteDelegate*)>(
      TfLiteYNNPackDelegateCreate(&options), TfLiteYNNPackDelegateDelete);

  ASSERT_EQ(model.ApplyCustomDelegate(delegate.get()), kTfLiteOk);

  // Run 1: S = 2
  model.SetQuery({1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f});
  model.SetKey({1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f});
  model.SetValue({1.0f, 2.0f, 3.0f, 4.0f, 1.0f, 2.0f, 3.0f, 4.0f});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  auto output1 = model.GetOutput();
  ASSERT_EQ(output1.size(), 8);
  EXPECT_NEAR(output1[0], 1.0f, 1e-4);
  EXPECT_NEAR(output1[4], 1.0f, 1e-4);

  // Run 2: Dynamically resize input tensor sequence length from 2 to 4
  ASSERT_EQ(model.ResizeSequence(4), kTfLiteOk);
  model.SetQuery({
      1.0f, 0.0f, 0.0f, 0.0f,  // Token 0
      1.0f, 0.0f, 0.0f, 0.0f,  // Token 1
      1.0f, 0.0f, 0.0f, 0.0f,  // Token 2
      1.0f, 0.0f, 0.0f, 0.0f   // Token 3
  });
  model.SetKey({
      1.0f, 0.0f, 0.0f, 0.0f,  // Token 0
      1.0f, 0.0f, 0.0f, 0.0f,  // Token 1
      1.0f, 0.0f, 0.0f, 0.0f,  // Token 2
      1.0f, 0.0f, 0.0f, 0.0f   // Token 3
  });
  model.SetValue({
      10.0f, 20.0f, 30.0f, 40.0f,  // Token 0
      10.0f, 20.0f, 30.0f, 40.0f,  // Token 1
      10.0f, 20.0f, 30.0f, 40.0f,  // Token 2
      10.0f, 20.0f, 30.0f, 40.0f   // Token 3
  });
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  auto output2 = model.GetOutput();
  ASSERT_EQ(output2.size(), 16);  // 1 * 4 * 1 * 4 = 16 elements
  for (size_t i = 0; i < 4; ++i) {
    EXPECT_NEAR(output2[i * 4 + 0], 10.0f, 1e-4);
    EXPECT_NEAR(output2[i * 4 + 1], 20.0f, 1e-4);
    EXPECT_NEAR(output2[i * 4 + 2], 30.0f, 1e-4);
    EXPECT_NEAR(output2[i * 4 + 3], 40.0f, 1e-4);
  }

  // Run 3: Dynamically resize input tensor sequence length from 4 down to 1
  ASSERT_EQ(model.ResizeSequence(1), kTfLiteOk);
  model.SetQuery({1.0f, 0.0f, 0.0f, 0.0f});
  model.SetKey({1.0f, 0.0f, 0.0f, 0.0f});
  model.SetValue({100.0f, 200.0f, 300.0f, 400.0f});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  auto output3 = model.GetOutput();
  ASSERT_EQ(output3.size(), 4);  // 1 * 1 * 1 * 4 = 4 elements
  EXPECT_NEAR(output3[0], 100.0f, 1e-4);
  EXPECT_NEAR(output3[1], 200.0f, 1e-4);
  EXPECT_NEAR(output3[2], 300.0f, 1e-4);
  EXPECT_NEAR(output3[3], 400.0f, 1e-4);
}

}  // namespace
}  // namespace ynnpack
}  // namespace tflite
