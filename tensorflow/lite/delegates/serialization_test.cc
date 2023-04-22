/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/delegates/serialization.h"

#include <cstdint>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/util.h"

namespace tflite {
namespace delegates {
namespace {

void EmptyReportError(TfLiteContext* context, const char* format, ...) {}

class SerializationTest : public ::testing::Test {
 protected:
  void TearDown() override {
    for (auto* owned_array : owned_arrays_) {
      TfLiteIntArrayFree(owned_array);
    }
  }

  std::string getSerializationDir() {
    auto from_env = ::testing::TempDir();
    if (!from_env.empty()) {
      return from_env;
    }
    return "";
  }

  // Unique num_tensors creates unique context fingerprint for testing.
  TfLiteContext GenerateTfLiteContext(int num_tensors) {
    owned_tensor_vecs_.emplace_back();
    auto& tensors_vec = owned_tensor_vecs_.back();
    for (int i = 0; i < num_tensors; ++i) {
      tensors_vec.emplace_back();
      auto& tensor = tensors_vec.back();
      tensor.bytes = i + 1;
    }

    TfLiteContext context;
    context.tensors_size = num_tensors;
    context.tensors = tensors_vec.data();
    context.ReportError = EmptyReportError;
    return context;
  }

  TfLiteDelegateParams GenerateTfLiteDelegateParams(int num_nodes,
                                                    int num_input_tensors,
                                                    int num_output_tensors) {
    // Create a dummy execution plan.
    auto* nodes_to_replace = TfLiteIntArrayCreate(num_nodes);
    auto* input_tensors = TfLiteIntArrayCreate(num_input_tensors);
    auto* output_tensors = TfLiteIntArrayCreate(num_output_tensors);
    owned_arrays_.push_back(nodes_to_replace);
    owned_arrays_.push_back(input_tensors);
    owned_arrays_.push_back(output_tensors);
    for (int i = 0; i < num_nodes; ++i) {
      nodes_to_replace->data[i] = i;
    }
    for (int i = 0; i < num_input_tensors; ++i) {
      input_tensors->data[i] = i + 2;
    }
    for (int i = 0; i < num_output_tensors; ++i) {
      output_tensors->data[i] = i + 3;
    }

    TfLiteDelegateParams params;
    params.input_tensors = input_tensors;
    params.output_tensors = output_tensors;
    params.nodes_to_replace = nodes_to_replace;
    return params;
  }

  std::vector<TfLiteIntArray*> owned_arrays_;
  std::vector<std::vector<TfLiteTensor>> owned_tensor_vecs_;
};

TEST_F(SerializationTest, StrFingerprint) {
  std::vector<int> data_1 = {1, 2, 3, 4};
  std::vector<int> data_1_equivalent = {1, 2, 3, 4};
  std::vector<int> data_2 = {2, 4, 6, 8};

  auto fingerprint_1 =
      StrFingerprint(data_1.data(), data_1.size() * sizeof(int));
  auto fingerprint_1_equivalent = StrFingerprint(
      data_1_equivalent.data(), data_1_equivalent.size() * sizeof(int));
  auto fingerprint_2 =
      StrFingerprint(data_2.data(), data_2.size() * sizeof(int));

  EXPECT_EQ(fingerprint_1, fingerprint_1_equivalent);
  EXPECT_NE(fingerprint_1, fingerprint_2);
}

TEST_F(SerializationTest, DelegateEntryFingerprint) {
  const std::string model_token = "mobilenet";
  const std::string dir = "/test/dir";
  const std::string delegate1 = "gpu";
  const std::string delegate2 = "nnapi";
  TfLiteContext context1 = GenerateTfLiteContext(/*num_tensors*/ 20);
  TfLiteContext context2 = GenerateTfLiteContext(/*num_tensors*/ 30);

  SerializationParams serialization_params = {model_token.c_str(), dir.c_str()};
  Serialization serialization(serialization_params);

  // Different contexts yield different keys.
  auto entry1 = serialization.GetEntryForDelegate(delegate1.c_str(), &context1);
  auto entry2 = serialization.GetEntryForDelegate(delegate1.c_str(), &context2);
  ASSERT_NE(entry1.GetFingerprint(), entry2.GetFingerprint());

  // Different custom_keys yield different keys.
  auto entry3 = serialization.GetEntryForDelegate(delegate2.c_str(), &context1);
  ASSERT_NE(entry1.GetFingerprint(), entry3.GetFingerprint());

  // Same fingerprint across serialization runs.
  Serialization serialization2(serialization_params);
  auto entry2_retry =
      serialization2.GetEntryForDelegate(delegate1.c_str(), &context2);
  ASSERT_EQ(entry2.GetFingerprint(), entry2_retry.GetFingerprint());
}

TEST_F(SerializationTest, KernelEntryFingerprint) {
  const std::string model_token = "mobilenet";
  const std::string dir = "/test/dir";
  const std::string delegate = "gpu";
  SerializationParams serialization_params = {model_token.c_str(), dir.c_str()};
  Serialization serialization(serialization_params);

  TfLiteContext ref_context = GenerateTfLiteContext(/*num_tensors*/ 30);
  TfLiteDelegateParams ref_partition = GenerateTfLiteDelegateParams(
      /*num_nodes=*/3, /*num_input_tensors=*/4, /*num_output_tensors=*/2);
  auto ref_entry = serialization.GetEntryForKernel(
      delegate.c_str(), &ref_context, &ref_partition);

  // Different inputs to delegated partition => different fingerprint.
  TfLiteDelegateParams diff_input_partition = GenerateTfLiteDelegateParams(
      /*num_nodes=*/3, /*num_input_tensors=*/3, /*num_output_tensors=*/2);
  ASSERT_NE(ref_entry.GetFingerprint(),
            serialization
                .GetEntryForKernel(delegate.c_str(), &ref_context,
                                   &diff_input_partition)
                .GetFingerprint());

  // Different outputs from delegated partition => different fingerprint.
  TfLiteDelegateParams diff_output_partition = GenerateTfLiteDelegateParams(
      /*num_nodes=*/3, /*num_input_tensors=*/4, /*num_output_tensors=*/3);
  ASSERT_NE(ref_entry.GetFingerprint(),
            serialization
                .GetEntryForKernel(delegate.c_str(), &ref_context,
                                   &diff_output_partition)
                .GetFingerprint());

  // Different nodes from delegated partition => different fingerprint.
  TfLiteDelegateParams diff_nodes_partition = GenerateTfLiteDelegateParams(
      /*num_nodes=*/4, /*num_input_tensors=*/4, /*num_output_tensors=*/2);
  ASSERT_NE(ref_entry.GetFingerprint(),
            serialization
                .GetEntryForKernel(delegate.c_str(), &ref_context,
                                   &diff_nodes_partition)
                .GetFingerprint());

  // Different contexts, same partition.
  TfLiteContext other_context = GenerateTfLiteContext(/*num_tensors*/ 60);
  ASSERT_NE(
      ref_entry.GetFingerprint(),
      serialization
          .GetEntryForKernel(delegate.c_str(), &other_context, &ref_partition)
          .GetFingerprint());

  // Same values across runs.
  ASSERT_EQ(
      ref_entry.GetFingerprint(),
      serialization
          .GetEntryForKernel(delegate.c_str(), &ref_context, &ref_partition)
          .GetFingerprint());

  // Same value from a new Serialization instance.
  Serialization serialization2(serialization_params);
  ASSERT_EQ(
      ref_entry.GetFingerprint(),
      serialization
          .GetEntryForKernel(delegate.c_str(), &ref_context, &ref_partition)
          .GetFingerprint());
}

TEST_F(SerializationTest, ModelTokenFingerprint) {
  std::string model_token1 = "model1";
  std::string model_token2 = "model2";
  const std::string dir = "/test/dir";
  const std::string delegate = "gpu";
  TfLiteContext context = GenerateTfLiteContext(/*num_tensors*/ 20);
  TfLiteDelegateParams partition = GenerateTfLiteDelegateParams(
      /*num_nodes=*/2, /*num_input_tensors=*/3, /*num_output_tensors=*/1);

  SerializationParams serialization_params1 = {model_token1.c_str(),
                                               dir.c_str()};
  Serialization serialization1(serialization_params1);
  auto entry1 =
      serialization1.GetEntryForKernel(delegate.c_str(), &context, &partition);
  SerializationParams serialization_params2 = {model_token2.c_str(),
                                               dir.c_str()};
  Serialization serialization2(serialization_params2);
  auto entry2 =
      serialization2.GetEntryForKernel(delegate.c_str(), &context, &partition);

  // Same params, but different model tokens.
  ASSERT_NE(entry1.GetFingerprint(), entry2.GetFingerprint());

  // Serialization Dir shouldn't matter for fingerprint values.
  std::string serialization_dir2 = "/another/dir";
  SerializationParams serialization_params3 = {model_token1.c_str(),
                                               serialization_dir2.c_str()};
  Serialization serialization3(serialization_params3);
  auto entry3 =
      serialization3.GetEntryForKernel(delegate.c_str(), &context, &partition);
  ASSERT_EQ(entry1.GetFingerprint(), entry3.GetFingerprint());
}

TEST_F(SerializationTest, SerializationData) {
  // Sample data to store in serialization.
  float value1 = 456.24;
  float value2 = 678.23;
  std::string model_token = "model1";
  std::string test_dir = getSerializationDir();
  const std::string fake_dir = "/test/dir";

  // Dummy context.
  TfLiteContext context = GenerateTfLiteContext(/*num_tensors*/ 30);
  TfLiteDelegateParams partition = GenerateTfLiteDelegateParams(
      /*num_nodes=*/2, /*num_input_tensors=*/3, /*num_output_tensors=*/1);

  SerializationParams serialization_params = {model_token.c_str(),
                                              test_dir.c_str()};
  Serialization serialization(serialization_params);

  {
    std::string custom_str1 = "test1";

    // Set data.
    auto entry1 =
        serialization.GetEntryForKernel(custom_str1, &context, &partition);
    ASSERT_EQ(entry1.SetData(&context, reinterpret_cast<const char*>(&value1),
                             sizeof(value1)),
              kTfLiteOk);

    // Same key instance should be able to read the data back.
    std::string read_back1 = "this string should be cleared";
    ASSERT_EQ(entry1.GetData(&context, &read_back1), kTfLiteOk);
    auto* retrieved_data1 = reinterpret_cast<float*>(&(read_back1[0]));
    ASSERT_FLOAT_EQ(*retrieved_data1, value1);

    // Equivalent key from same serialization should be able to read the same
    // data back.
    auto entry2 =
        serialization.GetEntryForKernel(custom_str1, &context, &partition);
    std::string read_back2;
    ASSERT_EQ(entry2.GetData(&context, &read_back2), kTfLiteOk);
    auto* retrieved_data2 = reinterpret_cast<float*>(&(read_back2[0]));
    ASSERT_FLOAT_EQ(*retrieved_data2, value1);
  }

  {
    std::string custom_str2 = "test2";

    // Trying to read data without setting should result in a 'cache miss'.
    auto entry3 =
        serialization.GetEntryForKernel(custom_str2, &context, &partition);
    std::string read_back3;
    ASSERT_EQ(entry3.GetData(&context, &read_back3),
              kTfLiteDelegateDataNotFound);
    // Now insert data.
    ASSERT_EQ(entry3.SetData(&context, reinterpret_cast<const char*>(&value2),
                             sizeof(value2)),
              kTfLiteOk);

    // Equivalent key from different serialization with same caching dir & model
    // token should read back the data.
    Serialization serialization2(serialization_params);
    std::string read_back4;
    auto entry4 =
        serialization2.GetEntryForKernel(custom_str2, &context, &partition);
    ASSERT_EQ(entry4.GetData(&context, &read_back4), kTfLiteOk);
    auto* retrieved_data = reinterpret_cast<float*>(&(read_back4[0]));
    ASSERT_FLOAT_EQ(*retrieved_data, value2);

    // Same key, but different dir shouldn't find data.
    SerializationParams new_params = {model_token.c_str(), fake_dir.c_str()};
    Serialization serialization3(new_params);
    auto entry5 =
        serialization3.GetEntryForKernel(custom_str2, &context, &partition);
    std::string read_back5;
    ASSERT_EQ(entry5.GetData(&context, &read_back5),
              kTfLiteDelegateDataNotFound);
  }
}

TEST_F(SerializationTest, CachingDelegatedNodes) {
  std::string model_token = "model1";
  std::string test_dir = getSerializationDir();
  SerializationParams serialization_params = {model_token.c_str(),
                                              test_dir.c_str()};
  Serialization serialization(serialization_params);
  TfLiteContext context = GenerateTfLiteContext(/*num_tensors*/ 30);
  const std::string test_delegate_id = "dummy_delegate";

  std::vector<int> nodes_to_delegate = {2, 3, 4, 7};
  TfLiteIntArray* nodes_to_delegate_array =
      ConvertVectorToTfLiteIntArray(nodes_to_delegate);
  std::vector<int> empty_nodes = {};
  TfLiteIntArray* empty_nodes_array =
      ConvertVectorToTfLiteIntArray(empty_nodes);

  {
    ASSERT_EQ(SaveDelegatedNodes(&context, &serialization, test_delegate_id,
                                 nodes_to_delegate_array),
              kTfLiteOk);
  }
  {
    TfLiteIntArray* read_back_array;
    ASSERT_EQ(GetDelegatedNodes(&context, &serialization, "unknown_delegate",
                                &read_back_array),
              kTfLiteDelegateDataNotFound);
    ASSERT_EQ(GetDelegatedNodes(&context, &serialization, test_delegate_id,
                                &read_back_array),
              kTfLiteOk);
    ASSERT_EQ(TfLiteIntArrayEqual(nodes_to_delegate_array, read_back_array), 1);
    TfLiteIntArrayFree(read_back_array);
  }
  {
    ASSERT_EQ(SaveDelegatedNodes(&context, &serialization, test_delegate_id,
                                 empty_nodes_array),
              kTfLiteOk);
    TfLiteIntArray* read_back_array;
    ASSERT_EQ(GetDelegatedNodes(&context, &serialization, test_delegate_id,
                                &read_back_array),
              kTfLiteOk);
    ASSERT_EQ(read_back_array->size, 0);
    TfLiteIntArrayFree(read_back_array);
  }
  {
    // nullptr invalid.
    ASSERT_EQ(
        SaveDelegatedNodes(&context, &serialization, test_delegate_id, nullptr),
        kTfLiteError);
    ASSERT_EQ(
        GetDelegatedNodes(&context, &serialization, test_delegate_id, nullptr),
        kTfLiteError);
  }

  TfLiteIntArrayFree(nodes_to_delegate_array);
  TfLiteIntArrayFree(empty_nodes_array);
}

}  // namespace
}  // namespace delegates
}  // namespace tflite
