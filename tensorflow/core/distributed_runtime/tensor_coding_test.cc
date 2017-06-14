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

#include "tensorflow/core/distributed_runtime/tensor_coding.h"

#include "tensorflow/core/framework/device_base.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"
#include "tensorflow/core/protobuf/worker.pb.h"
#include "tensorflow/core/public/session_options.h"

namespace tensorflow {

class DummyDevice : public DeviceBase {
 public:
  explicit DummyDevice(Env* env) : DeviceBase(env) {
    attr_.set_device_type("CPU");
  }

  const DeviceAttributes& attributes() const override { return attr_; }

  Allocator* GetAllocator(AllocatorAttributes attr) override {
    return cpu_allocator();
  }

 private:
  DeviceAttributes attr_;
};

class StringSource : public TensorResponse::Source {
 public:
  explicit StringSource(const string* s, int block_size)
      : s_(s), stream_(nullptr), block_size_(block_size) {}
  ~StringSource() override { DeleteStream(); }

  protobuf::io::ZeroCopyInputStream* contents() override {
    DeleteStream();
    stream_ = new (&space_)
        protobuf::io::ArrayInputStream(s_->data(), s_->size(), block_size_);
    return stream_;
  }

  void DeleteStream() {
    if (stream_) {
      stream_->~ArrayInputStream();
    }
  }

 private:
  const string* s_;
  protobuf::io::ArrayInputStream* stream_;
  char space_[sizeof(protobuf::io::ArrayInputStream)];
  int block_size_;
};

class TensorResponseTest : public ::testing::Test {
 public:
  void Validate(const Tensor& src, bool is_dead, bool use_tensor_content) {
    RecvTensorResponse proto;
    proto.set_is_dead(is_dead);
    proto.set_send_start_micros(123456);
    if (use_tensor_content) {
      src.AsProtoTensorContent(proto.mutable_tensor());
    } else {
      src.AsProtoField(proto.mutable_tensor());
    }
    string encoded;
    proto.AppendToString(&encoded);

    StringSource source(&encoded, 1024);

    TensorResponse response;
    DummyDevice cpu_device(Env::Default());
    response.InitAlloc(&cpu_device, AllocatorAttributes());
    for (int i = 0; i < 2; i++) {  // Twice so we exercise reuse of "response"
      Status s = response.ParseFrom(&source);
      EXPECT_TRUE(s.ok());

      const RecvTensorResponse& meta = response.metadata();
      EXPECT_EQ(meta.is_dead(), is_dead);
      EXPECT_EQ(meta.send_start_micros(), 123456);

      const Tensor& result = response.tensor();
      EXPECT_EQ(result.dtype(), src.dtype());
      EXPECT_EQ(result.shape().DebugString(), src.shape().DebugString());
      EXPECT_EQ(result.DebugString(), src.DebugString());
    }
  }

  template <typename T>
  void DoTest(DataType dt) {
    gtl::InlinedVector<T, 4> v;
    LOG(ERROR) << "DT: " << static_cast<int>(dt);
    for (int elems = 0; elems <= 10000; elems++) {
      if (elems < 100 || (elems % 1000 == 0)) {
        Tensor a(dt, TensorShape({1, static_cast<int64>(v.size())}));
        test::FillValues<T>(&a, v);
        Validate(a, (elems == 0), true);
      }
      v.push_back(static_cast<T>(elems));
    }
  }
  void DoTestForStrings(DataType dt) {
    gtl::InlinedVector<string, 4> v;
    LOG(ERROR) << "DT: string";
    for (int elems = 0; elems <= 10000; elems++) {
      if (elems < 100 || (elems % 1000 == 0)) {
        Tensor a(dt, TensorShape({1, static_cast<int64>(v.size())}));
        test::FillValues<string>(&a, v);
        Validate(a, (elems == 0), true);
      }
      v.push_back(strings::StrCat("This is string ", elems));
    }
  }
};

TEST_F(TensorResponseTest, Simple) {
  DoTest<float>(DT_FLOAT);
  DoTest<double>(DT_DOUBLE);
  DoTest<int32>(DT_INT32);
  DoTest<uint16>(DT_UINT16);
  DoTest<uint8>(DT_UINT8);
  DoTest<int16>(DT_INT16);
  DoTest<int8>(DT_INT8);
  DoTest<complex64>(DT_COMPLEX64);
  DoTest<complex128>(DT_COMPLEX128);
  DoTest<int64>(DT_INT64);
  DoTest<bool>(DT_BOOL);
  DoTest<qint8>(DT_QINT8);
  DoTest<quint8>(DT_QUINT8);
  DoTest<qint16>(DT_QINT16);
  DoTest<quint16>(DT_QUINT16);
  DoTest<qint32>(DT_QINT32);
  DoTest<bfloat16>(DT_BFLOAT16);
  DoTest<Eigen::half>(DT_HALF);
}

TEST_F(TensorResponseTest, StringTensor) { DoTestForStrings(DT_STRING); }

string MakeFloatTensorTestCase(int num_elems) {
  std::vector<int8> v(num_elems);
  for (int i = 0; i < num_elems; i++) {
    v[i] = i % 10;
  }
  Tensor src(DT_INT8, TensorShape({1, static_cast<int64>(v.size())}));
  test::FillValues<int8>(&src, v);

  RecvTensorResponse proto;
  proto.set_is_dead(false);
  proto.set_send_start_micros(123456);
  src.AsProtoTensorContent(proto.mutable_tensor());
  string encoded;
  proto.AppendToString(&encoded);
  return encoded;
}

static void BM_TensorResponse(int iters, int arg) {
  testing::StopTiming();
  string encoded = MakeFloatTensorTestCase(arg);
  DummyDevice cpu_device(Env::Default());
  testing::StartTiming();
  while (--iters > 0) {
    TensorResponse response;
    response.InitAlloc(&cpu_device, AllocatorAttributes());
    StringSource source(&encoded, -1);
    Status s = response.ParseFrom(&source);
    if (iters == 1) {
      testing::SetLabel(
          strings::StrCat("Bytes: ", response.tensor().TotalBytes()));
    }
  }
}
BENCHMARK(BM_TensorResponse)->Arg(0)->Arg(1000)->Arg(100000);

static void BM_TensorViaTensorProto(int iters, int arg) {
  testing::StopTiming();
  string encoded = MakeFloatTensorTestCase(arg);
  testing::StartTiming();
  while (--iters > 0) {
    RecvTensorResponse r;
    r.ParseFromString(encoded);
    Tensor t;
    CHECK(t.FromProto(r.tensor()));
    if (iters == 1) {
      testing::SetLabel(strings::StrCat("Bytes: ", t.TotalBytes()));
    }
  }
}
BENCHMARK(BM_TensorViaTensorProto)->Arg(0)->Arg(1000)->Arg(100000);

}  // namespace tensorflow
