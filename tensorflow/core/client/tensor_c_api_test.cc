#include "tensorflow/core/public/tensor_c_api.h"

#include <gtest/gtest.h>
#include "tensorflow/core/public/tensor.h"

using tensorflow::Tensor;
using tensorflow::TensorShape;

namespace tensorflow {
bool TF_Tensor_DecodeStrings(TF_Tensor* src, Tensor* dst, TF_Status* status);
TF_Tensor* TF_Tensor_EncodeStrings(const Tensor& src);
}  // namespace tensorflow

TEST(CApi, Status) {
  TF_Status* s = TF_NewStatus();
  EXPECT_EQ(TF_OK, TF_GetCode(s));
  EXPECT_EQ(tensorflow::string(), TF_Message(s));
  TF_SetStatus(s, TF_CANCELLED, "cancel");
  EXPECT_EQ(TF_CANCELLED, TF_GetCode(s));
  EXPECT_EQ(tensorflow::string("cancel"), TF_Message(s));
  TF_DeleteStatus(s);
}

static void Deallocator(void* data, size_t, void* arg) {
  tensorflow::cpu_allocator()->DeallocateRaw(data);
  *reinterpret_cast<bool*>(arg) = true;
}

TEST(CApi, Tensor) {
  float* values =
      reinterpret_cast<float*>(tensorflow::cpu_allocator()->AllocateRaw(
          EIGEN_MAX_ALIGN_BYTES, 6 * sizeof(float)));
  tensorflow::int64 dims[] = {2, 3};
  bool deallocator_called = false;
  TF_Tensor* t = TF_NewTensor(TF_FLOAT, dims, 2, values, sizeof(values),
                              &Deallocator, &deallocator_called);
  EXPECT_FALSE(deallocator_called);
  EXPECT_EQ(TF_FLOAT, TF_TensorType(t));
  EXPECT_EQ(2, TF_NumDims(t));
  EXPECT_EQ(dims[0], TF_Dim(t, 0));
  EXPECT_EQ(dims[1], TF_Dim(t, 1));
  EXPECT_EQ(sizeof(values), TF_TensorByteSize(t));
  EXPECT_EQ(static_cast<void*>(values), TF_TensorData(t));
  TF_DeleteTensor(t);
  EXPECT_TRUE(deallocator_called);
}

static void TestEncodeDecode(int line,
                             const std::vector<tensorflow::string>& data) {
  const tensorflow::int64 n = data.size();
  for (std::vector<tensorflow::int64> dims :
       std::vector<std::vector<tensorflow::int64>>{
           {n}, {1, n}, {n, 1}, {n / 2, 2}}) {
    // Create C++ Tensor
    Tensor src(tensorflow::DT_STRING, TensorShape(dims));
    for (tensorflow::int64 i = 0; i < src.NumElements(); i++) {
      src.flat<tensorflow::string>()(i) = data[i];
    }
    TF_Tensor* dst = TF_Tensor_EncodeStrings(src);

    // Convert back to a C++ Tensor and ensure we get expected output.
    TF_Status* status = TF_NewStatus();
    Tensor output;
    ASSERT_TRUE(TF_Tensor_DecodeStrings(dst, &output, status)) << line;
    ASSERT_EQ(TF_OK, TF_GetCode(status)) << line;
    ASSERT_EQ(src.NumElements(), output.NumElements()) << line;
    for (tensorflow::int64 i = 0; i < src.NumElements(); i++) {
      ASSERT_EQ(data[i], output.flat<tensorflow::string>()(i)) << line;
    }

    TF_DeleteStatus(status);
    TF_DeleteTensor(dst);
  }
}

TEST(CApi, TensorEncodeDecodeStrings) {
  TestEncodeDecode(__LINE__, {});
  TestEncodeDecode(__LINE__, {"hello"});
  TestEncodeDecode(__LINE__,
                   {"the", "quick", "brown", "fox", "jumped", "over"});

  tensorflow::string big(1000, 'a');
  TestEncodeDecode(__LINE__, {"small", big, "small2"});
}

TEST(CApi, SessionOptions) {
  TF_SessionOptions* opt = TF_NewSessionOptions();
  TF_DeleteSessionOptions(opt);
}

// TODO(jeff,sanjay): Session tests
// . Create and delete
// . Extend graph
// . Run
