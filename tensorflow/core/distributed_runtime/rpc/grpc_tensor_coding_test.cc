/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/distributed_runtime/rpc/grpc_tensor_coding.h"

#include "grpcpp/support/byte_buffer.h"
#include "grpcpp/support/slice.h"
#include "absl/status/status.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/protobuf/worker.pb.h"

namespace tensorflow {

class GrpcTensorCodingTest : public ::testing::Test {
 public:
  void Validate(const Tensor& t, bool is_dead) {
    // Check by encoding to a ByteBuffer
    ::grpc::ByteBuffer buf;
    absl::Status s = grpc::EncodeTensorToByteBuffer(is_dead, t, false, &buf);
    TF_EXPECT_OK(s);

    // Make a string
    std::vector<::grpc::Slice> slices;
    (void)buf.Dump(&slices);
    string tmp;
    for (const auto& s : slices) {
      tmp.append(reinterpret_cast<const char*>(s.begin()), s.size());
    }

    RecvTensorResponse response;
    EXPECT_TRUE(response.ParseFromString(tmp));
    EXPECT_EQ(response.is_dead(), is_dead);

    Tensor result_tensor;
    EXPECT_TRUE(result_tensor.FromProto(response.tensor()));
    EXPECT_EQ(t.dtype(), result_tensor.dtype());
    EXPECT_EQ(t.shape().DebugString(), result_tensor.shape().DebugString());
    EXPECT_EQ(t.DebugString(), result_tensor.DebugString());
  }

  template <typename T>
  void DoTest(DataType dt) {
    gtl::InlinedVector<T, 4> v;
    for (int elems = 0; elems <= 10000; elems++) {
      if (elems < 100 || (elems % 1000 == 0)) {
        Tensor a(dt, TensorShape({1, static_cast<int64_t>(v.size())}));
        test::FillValues<T>(&a, v);
        Validate(a, (elems == 0));
      }
      v.push_back(static_cast<T>(elems));
    }
  }
  void DoTestForStrings(DataType dt) {
    absl::InlinedVector<tstring, 4UL> v;
    for (int elems = 0; elems <= 10000; elems++) {
      if (elems < 100 || (elems % 1000 == 0)) {
        Tensor a(dt, TensorShape({1, static_cast<int64_t>(v.size())}));
        test::FillValues<tstring>(&a, v);
        Validate(a, (elems == 0));
      }
      v.push_back(strings::StrCat("This is string ", elems));
    }
  }
};

TEST_F(GrpcTensorCodingTest, Simple) {
  DoTest<float>(DT_FLOAT);
  DoTest<double>(DT_DOUBLE);
  DoTest<int32>(DT_INT32);
  DoTest<uint16>(DT_UINT16);
  DoTest<uint8>(DT_UINT8);
  DoTest<int16>(DT_INT16);
  DoTest<int8>(DT_INT8);
  DoTest<complex64>(DT_COMPLEX64);
  DoTest<complex128>(DT_COMPLEX128);
  DoTest<int64_t>(DT_INT64);
  DoTest<bool>(DT_BOOL);
  DoTest<qint8>(DT_QINT8);
  DoTest<quint8>(DT_QUINT8);
  DoTest<qint16>(DT_QINT16);
  DoTest<quint16>(DT_QUINT16);
  DoTest<qint32>(DT_QINT32);
  DoTest<bfloat16>(DT_BFLOAT16);
  DoTest<Eigen::half>(DT_HALF);
}

TEST_F(GrpcTensorCodingTest, StringTensor) { DoTestForStrings(DT_STRING); }

TEST_F(GrpcTensorCodingTest, LargeTensor) {
  Tensor t(DT_INT8, TensorShape({1, 1 + (1LL << 31)}));
  ::grpc::ByteBuffer buf;
  absl::Status s = grpc::EncodeTensorToByteBuffer(/*is_dead=*/false, t,
                                                  /*require_ack=*/false, &buf);
  EXPECT_EQ(s.code(), absl::StatusCode::kInternal);
}

}  // namespace tensorflow
