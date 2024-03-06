/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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
#include <memory>
#include <numeric>
#include <vector>

#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/shape_inference_testutil.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {
Status MakeNodeDef(DataType dtype, NodeDef* node_def) {
  return NodeDefBuilder("fingerprint", "Fingerprint")
      .Input(FakeInput(dtype))
      .Input(FakeInput(DT_STRING))
      .Finalize(node_def);
}

class FingerprintOpTest : public OpsTestBase {
 protected:
  Status MakeFingerprintOp(Tensor* tensor) {
    return MakeFingerprintOp(tensor, "farmhash64");
  }

  Status MakeFingerprintOp(Tensor* data, const string& method) {
    TF_RETURN_IF_ERROR(MakeNodeDef(data->dtype(), node_def()));
    TF_RETURN_IF_ERROR(InitOp());

    inputs_.clear();
    inputs_.push_back(TensorValue(data));

    method_ = Tensor(DT_STRING, TensorShape{});
    method_.scalar<tstring>()() = method;
    inputs_.push_back(TensorValue(&method_));
    return absl::OkStatus();
  }

  Tensor batch_dims_;
  Tensor method_;
};

TEST_F(FingerprintOpTest, Empty) {
  Tensor tensor(DT_UINT8, {0});

  TF_ASSERT_OK(MakeFingerprintOp(&tensor));
  TF_ASSERT_OK(RunOpKernel());
  EXPECT_EQ(GetOutput(0)->shape(), (TensorShape{0, 8}));
  EXPECT_EQ(GetOutput(0)->tensor_data(), "");
}

// This test detects changes in fingerprint method.
TEST_F(FingerprintOpTest, GoldenValue) {
  Tensor tensor(DT_UINT8, {1, 3, 4, 5, 6, 7});
  auto buffer = tensor.flat<uint8>();
  std::iota(buffer.data(), buffer.data() + buffer.size(),
            static_cast<uint8>(47));

  TF_ASSERT_OK(MakeFingerprintOp(&tensor));
  TF_ASSERT_OK(RunOpKernel());
  EXPECT_EQ(GetOutput(0)->shape(), (TensorShape{1, 8}));
  EXPECT_EQ(GetOutput(0)->tensor_data(), "\x2d\x90\xdf\x03\x79\x36\x3c\x43");
}

// String types have a different compute path. This test detects changes in this
// special-case handling.
TEST_F(FingerprintOpTest, StringGoldenValue) {
  Tensor data(DT_STRING, {1, 2, 2});
  auto buffer = data.flat<tstring>();
  buffer(0).resize(10);
  buffer(1).resize(7);
  buffer(2).resize(0);
  buffer(3).resize(19);
  std::iota(&buffer(0)[0], &buffer(0)[0] + buffer(0).size(), 0);
  std::iota(&buffer(1)[0], &buffer(1)[0] + buffer(1).size(), 7);
  std::iota(&buffer(2)[0], &buffer(2)[0] + buffer(2).size(), 71);
  std::iota(&buffer(3)[0], &buffer(3)[0] + buffer(3).size(), 41);

  TF_ASSERT_OK(MakeFingerprintOp(&data));
  TF_ASSERT_OK(RunOpKernel());
  ASSERT_EQ(GetOutput(0)->shape(), (TensorShape{1, 8}));
  EXPECT_EQ(GetOutput(0)->tensor_data(), "\x92\x43\x28\x52\xa3\x7c\x48\x18");

  // When each batch item has exactly one string, Fingerprint op avoids
  // double-fingerprint. Adding a test to detect any change in this logic.
  ASSERT_TRUE(data.CopyFrom(data, TensorShape{4}));
  TF_ASSERT_OK(MakeFingerprintOp(&data));
  TF_ASSERT_OK(RunOpKernel());
  ASSERT_EQ(GetOutput(0)->shape(), (TensorShape{4, 8}));
  EXPECT_EQ(GetOutput(0)->tensor_data(),
            "\xea\xff\xd6\xb2\xb2\x4d\x70\x9b"
            "\x6e\x9d\xed\x21\xc6\x4a\x61\x52"
            "\x4f\x40\x90\x2f\x3b\x6a\xe1\x9a"
            "\x0d\x9b\x7f\x63\x23\x14\x1c\xb8");
}

TEST_F(FingerprintOpTest, Collision) {
  const TensorShape shape = {1, 2, 4, 6};
  for (DataType dtype : kRealNumberTypes) {
    const int64_t size = shape.num_elements() * DataTypeSize(dtype);

    Tensor tensor(dtype, shape);
    auto buffer = tensor.bit_casted_shaped<uint8, 1>({size});
    buffer.setRandom();

    TF_ASSERT_OK(MakeFingerprintOp(&tensor));
    TF_ASSERT_OK(RunOpKernel());
    const Tensor fingerprint0 = *GetOutput(0);

    // Alter a byte value in the buffer.
    const int offset = buffer(0) % buffer.size();
    buffer(offset) = ~buffer(offset);

    TF_ASSERT_OK(MakeFingerprintOp(&tensor));
    TF_ASSERT_OK(RunOpKernel());
    const Tensor fingerprint1 = *GetOutput(0);

    EXPECT_NE(fingerprint0.tensor_data(), fingerprint1.tensor_data());
  }
}

TEST_F(FingerprintOpTest, CollisionString) {
  constexpr int64_t size = 256;

  Tensor tensor(DT_STRING, {1});
  auto& input = tensor.vec<tstring>()(0);
  input.resize(size);

  TTypes<uint8>::UnalignedFlat buffer(reinterpret_cast<uint8*>(&input[0]),
                                      input.size());
  buffer.setRandom();

  TF_ASSERT_OK(MakeFingerprintOp(&tensor));
  TF_ASSERT_OK(RunOpKernel());
  const Tensor fingerprint0 = *GetOutput(0);

  // Alter a byte value in the buffer.
  const int offset = buffer(0) % buffer.size();
  buffer(offset) = ~buffer(offset);

  TF_ASSERT_OK(MakeFingerprintOp(&tensor));
  TF_ASSERT_OK(RunOpKernel());
  const Tensor fingerprint1 = *GetOutput(0);

  EXPECT_NE(fingerprint0.tensor_data(), fingerprint1.tensor_data());
}

TEST_F(FingerprintOpTest, CompareBytesAndString) {
  Tensor pods_tensor(DT_FLOAT, {4, 64});
  Tensor strings_tensor(DT_STRING, {4});

  auto pods = pods_tensor.matrix<float>();
  pods.setRandom();

  auto strings = strings_tensor.vec<tstring>();
  for (int64_t i = 0; i < strings.size(); ++i) {
    strings(i).assign(reinterpret_cast<const char*>(&pods(i, 0)),
                      pods.dimension(1) * sizeof(pods(i, 0)));
  }

  TF_ASSERT_OK(MakeFingerprintOp(&pods_tensor));
  TF_ASSERT_OK(RunOpKernel());
  Tensor pods_fingerprints = *GetOutput(0);

  TF_ASSERT_OK(MakeFingerprintOp(&strings_tensor));
  TF_ASSERT_OK(RunOpKernel());
  Tensor strings_fingerprints = *GetOutput(0);

  EXPECT_EQ(pods_fingerprints.tensor_data(),
            strings_fingerprints.tensor_data());
}

TEST_F(FingerprintOpTest, SupportedMethods) {
  Tensor tensor(DT_STRING, TensorShape{1});
  TF_ASSERT_OK(MakeFingerprintOp(&tensor, "unsupported_method"));

  const Status status = RunOpKernel();
  EXPECT_FALSE(status.ok());
  EXPECT_NE(status.message().find("unsupported_method"), string::npos);
}

TEST_F(FingerprintOpTest, SupportedTypes) {
  Tensor input(DT_RESOURCE, TensorShape{1});
  EXPECT_FALSE(MakeFingerprintOp(&input).ok());
}

TEST(FingerprintOpShapeFnTest, MethodKnownStatically) {
  ShapeInferenceTestOp op("Fingerprint");

  Tensor method(DT_STRING, TensorShape{});
  method.scalar<tstring>()() = "farmhash64";
  op.input_tensors.assign({nullptr, &method});

  TF_ASSERT_OK(MakeNodeDef(DT_UINT8, &op.node_def));
  INFER_OK(op, "?;?", "[?,8]");
  INFER_ERROR("must be at least rank 1", op, "[];?");
  INFER_OK(op, "[?];?", "[d0_0,8]");
  INFER_OK(op, "[1,?];?", "[d0_0,8]");
  INFER_OK(op, "[?,2,3];?", "[d0_0,8]");
}

TEST(FingerprintOpShapeFnTest, MethodUnknownStatically) {
  ShapeInferenceTestOp op("Fingerprint");

  TF_ASSERT_OK(MakeNodeDef(DT_FLOAT, &op.node_def));
  INFER_OK(op, "?;?", "[?,?]");
  INFER_ERROR("must be at least rank 1", op, "[];?");
  INFER_OK(op, "[?];?", "[d0_0,?]");
  INFER_OK(op, "[1,?];?", "[d0_0,?]");
  INFER_OK(op, "[?,2,3];?", "[d0_0,?]");
}

TEST(FingerprintOpShapeFnTest, InvalidMethod) {
  ShapeInferenceTestOp op("Fingerprint");

  // When `method` shape is known statically.
  INFER_ERROR("must be rank 0", op, "[1];[1]");

  // When `method` shape is unknown statically.
  Tensor method(DT_STRING, TensorShape{1});
  method.vec<tstring>()(0) = "farmhash64";
  op.input_tensors.assign({nullptr, &method});
  INFER_ERROR("must be rank 0", op, "?;?");

  method = Tensor(DT_STRING, TensorShape{});
  method.scalar<tstring>()() = "unsupported_method";
  op.input_tensors.assign({nullptr, &method});
  INFER_ERROR("unsupported_method", op, "?;?");
}
}  // namespace
}  // namespace tensorflow
