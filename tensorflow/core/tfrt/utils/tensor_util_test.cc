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
#include "tensorflow/core/tfrt/utils/tensor_util.h"

#include <complex>
#include <memory>
#include <numeric>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "llvm/ADT/SmallVector.h"
#include "tensorflow/core/runtime_fallback/kernel/kernel_fallback_tensor.h"
#include "tfrt/cpp_tests/test_util.h"  // from @tf_runtime
#include "tfrt/host_context/concurrent_work_queue.h"  // from @tf_runtime
#include "tfrt/host_context/host_allocator.h"  // from @tf_runtime
#include "tfrt/host_context/host_context.h"  // from @tf_runtime
#include "tfrt/support/forward_decls.h"  // from @tf_runtime
#include "tfrt/tensor/dense_host_tensor.h"  // from @tf_runtime
#include "tfrt/tensor/dense_host_tensor_view.h"  // from @tf_runtime
#include "tfrt/tensor/scalar_host_tensor.h"  // from @tf_runtime
#include "tfrt/tensor/string_host_tensor.h"  // from @tf_runtime

namespace tfrt {
namespace {

TEST(TensorUtilTest, DHTToTFTensor) {
  std::unique_ptr<HostContext> host = CreateHostContext();
  auto dht = *DenseHostTensor::CreateUninitialized(
      TensorMetadata(DType(DType::I32), {2, 2}), host.get());
  auto view = MutableDHTArrayView<int32_t>(&dht);
  std::iota(view.begin(), view.end(), 1);
  auto tf_tensor = *TFRTTensorToTFTensor(dht);
  EXPECT_THAT(tf_tensor.shape().dim_sizes(), testing::ElementsAre(2, 2));
  EXPECT_EQ(tf_tensor.data(), dht.data());
}

TEST(TensorUtilTest, SHTToTFTensor) {
  std::unique_ptr<HostContext> host = CreateHostContext();
  auto sht = *StringHostTensor::CreateUninitialized(
      TensorMetadata(DType(DType::String), TensorShape(1)), host.get());
  sht.strings().front() = "hello";
  auto tf_tensor = *TFRTTensorToTFTensor(sht);
  EXPECT_EQ(tf_tensor.NumElements(), 1);
  EXPECT_EQ(tf_tensor.flat<tensorflow::tstring>()(0), "hello");
}

TEST(TensorUtilTest, KNFBToTFTensor) {
  std::unique_ptr<HostContext> host = CreateHostContext();
  tensorflow::Tensor original_tf_tensor(0.1f);
  tensorflow::KernelFallbackTensor knfbt(original_tf_tensor);
  auto tf_tensor = *TFRTTensorToTFTensor(knfbt);
  EXPECT_EQ(tf_tensor.NumElements(), 1);
  EXPECT_EQ(tf_tensor.data(), original_tf_tensor.data());
}

TEST(TensorUtilTest, ScalarHostTensorToTFTensor) {
  std::unique_ptr<HostContext> host = CreateHostContext();
  ScalarHostTensor<int32_t> t(TensorShape({2, 2}), 1);
  auto tf_tensor = TFRTTensorToTFTensor(t);
  ASSERT_FALSE(!tf_tensor);
  ASSERT_EQ(tf_tensor->dtype(), tensorflow::DT_INT32);
  ASSERT_THAT(tf_tensor->shape().dim_sizes(), testing::ElementsAre(2, 2));

  const auto* data = tf_tensor->flat<int32_t>().data();
  EXPECT_THAT(std::vector<int32_t>(data, data + 4),
              testing::ElementsAre(1, 1, 1, 1));
}

TEST(TensorUtilTest, TFRTTensorToTFTensorUnsupported) {
  std::unique_ptr<HostContext> host = CreateHostContext();

  class UnsupportedTensor : public Tensor,
                            public TensorTraits<UnsupportedTensor> {
   public:
    explicit UnsupportedTensor(const TensorMetadata& metadata)
        : Tensor(metadata) {}

    void Print(llvm::raw_ostream& os) const override {}

    static const char* name() { return "UnsupportedTensor"; }
  };

  UnsupportedTensor t(TensorMetadata(DType{DType::I32}, /*shape=*/{}));
  EXPECT_FALSE(TFRTTensorToTFTensor(t));
}

TEST(TensorUtilTest, TFTensorToTFRTTensorHandle) {
  std::unique_ptr<HostContext> host = CreateHostContext();
  tensorflow::Tensor tf_tensor(0.2f);
  auto handle = TFTensorToTFRTTensorHandle(tf_tensor, host.get());
  ASSERT_TRUE(handle->GetAsyncTensor()->IsAvailable());
  auto converted_tf_tensor =
      *TFRTTensorToTFTensor(handle->GetAsyncTensor()->get<Tensor>());
  EXPECT_EQ(converted_tf_tensor.NumElements(), 1);
  EXPECT_EQ(converted_tf_tensor.data(), tf_tensor.data());
}

std::unique_ptr<HostContext> CreateDefaultHostContext() {
  return std::make_unique<HostContext>([](const DecodedDiagnostic&) {},
                                       CreateMallocAllocator(),
                                       CreateSingleThreadedWorkQueue());
}

template <typename T, size_t N>
tensorflow::Tensor CreateTFTensor(const T (&data)[N],
                                  const tensorflow::TensorShape& shape) {
  tensorflow::Tensor tensor(tensorflow::DataTypeToEnum<T>::value, shape);
  std::copy(std::begin(data), std::end(data), tensor.flat<T>().data());
  return tensor;
}

TEST(TensorUtilTest, TFTensorAndTensorHandleBasic) {
  auto host = CreateDefaultHostContext();

  bool i1_data[] = {true, false, true, false};
  int8_t i8_data[] = {1, 2, 3, 4};
  int16_t i16_data[] = {1, 2, 3, 4};
  int32_t i32_data[] = {1, 2, 3, 4};
  int64_t i64_data[] = {1, 2, 3, 4};
  uint8_t ui8_data[] = {1, 2, 3, 4};
  uint16_t ui16_data[] = {1, 2, 3, 4};
  uint32_t ui32_data[] = {1, 2, 3, 4};
  uint64_t ui64_data[] = {1, 2, 3, 4};
  Eigen::half f16_data[] = {Eigen::half(1.0f), Eigen::half(2.0f),
                            Eigen::half(3.0f), Eigen::half(4.0f)};
  Eigen::bfloat16 bf16_data[] = {Eigen::bfloat16(1.0f), Eigen::bfloat16(2.0f),
                                 Eigen::bfloat16(3.0f), Eigen::bfloat16(4.0f)};
  float f32_data[] = {1.0f, 2.0f, 3.0f, 4.0f};
  double f64_data[] = {1.0f, 2.0f, 3.0f, 4.0f};
  std::complex<float> complex64_data[] = {
      {1.0f, 1.0f}, {2.0f, 2.0f}, {3.0f, 3.0f}, {4.0f, 4.0f}};
  std::complex<double> complex128_data[] = {
      {1.0f, 1.0f}, {2.0f, 2.0f}, {3.0f, 3.0f}, {4.0f, 4.0f}};

  tensorflow::TensorShape shape({2, 2});

  tensorflow::Tensor tensors[] = {CreateTFTensor(i1_data, shape),
                                  CreateTFTensor(i8_data, shape),
                                  CreateTFTensor(i16_data, shape),
                                  CreateTFTensor(i32_data, shape),
                                  CreateTFTensor(i64_data, shape),
                                  CreateTFTensor(ui8_data, shape),
                                  CreateTFTensor(ui16_data, shape),
                                  CreateTFTensor(ui32_data, shape),
                                  CreateTFTensor(ui64_data, shape),
                                  CreateTFTensor(f16_data, shape),
                                  CreateTFTensor(bf16_data, shape),
                                  CreateTFTensor(f32_data, shape),
                                  CreateTFTensor(f64_data, shape),
                                  CreateTFTensor(complex64_data, shape),
                                  CreateTFTensor(complex128_data, shape)};

  for (const auto& tf_tensor : tensors) {
    TF_ASSERT_OK_AND_ASSIGN(
        auto tfrt_tensor_handle,
        CreateTensorHandleFromTFTensor(tf_tensor, host.get()));

    TF_ASSERT_OK_AND_ASSIGN(auto tf_tensor_out,
                            CreateTFTensorFromTensorHandle(tfrt_tensor_handle));

    ASSERT_EQ(tf_tensor.dtype(), tf_tensor_out.dtype());
    EXPECT_EQ(tf_tensor.tensor_data(), tf_tensor_out.tensor_data())
        << "Tensor conversion failed for dtype: "
        << tensorflow::DataType_Name(tf_tensor.dtype());
  }
}

TEST(TensorUtilTest, TFTensorAndTensorHandleString) {
  auto host = CreateDefaultHostContext();

  tensorflow::Tensor tf_tensor(tensorflow::DT_STRING,
                               tensorflow::TensorShape({2}));
  auto tf_data = tf_tensor.flat<tensorflow::tstring>();
  tf_data(0).assign("test");
  tf_data(1).assign("string");

  TF_ASSERT_OK_AND_ASSIGN(
      auto tfrt_tensor_handle,
      CreateTensorHandleFromTFTensor(tf_tensor, host.get()));

  TF_ASSERT_OK_AND_ASSIGN(auto tf_tensor_2,
                          CreateTFTensorFromTensorHandle(tfrt_tensor_handle));

  auto tf_data_2 = tf_tensor_2.flat<tensorflow::tstring>();
  EXPECT_EQ(tf_data_2(0), "test");
  EXPECT_EQ(tf_data_2(1), "string");
}

TEST(TensorUtilTest, TFTensorToDHT) {
  std::unique_ptr<HostContext> host = CreateHostContext();
  tensorflow::Tensor tf_tensor(0.2f);
  auto dht = ConvertTfTensorToDHT(tf_tensor);
  ASSERT_FALSE(!dht);
  llvm::SmallVector<int32_t, 4> dims;
  EXPECT_EQ(dht->NumElements(), 1);
  EXPECT_EQ(tf_tensor.data(), dht->data());
}

}  // namespace
}  // namespace tfrt
