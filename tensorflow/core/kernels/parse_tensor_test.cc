/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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
#include <vector>
#include <string>

#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/framework/register_types.h"

namespace tensorflow {
namespace {

class SerializeTensorOpTest : public OpsTestBase {
 protected:
  template <typename T>
  void MakeOp() {
    TF_ASSERT_OK(
        NodeDefBuilder("myop", "SerializeTensor")
            .Input(FakeInput(DataTypeToEnum<T>::value))
            .Finalize(node_def()));
    TF_ASSERT_OK(InitOp());
  }
};

#define REGISTER_TEST(T, Shape, Mapping)                                 \
  TEST_F(SerializeTensorOpTest, SerializeTensorOpTest_##T) {             \
    MakeOp<T>();                                                         \
    AddInput<T>((Shape), (Mapping));                                     \
    TF_ASSERT_OK(RunOpKernel());                                         \
    Tensor* serialize_output = GetOutput(0);                             \
    const Tensor& input = GetInput(0);                                   \
    NodeDef parse;                                                       \
    TF_ASSERT_OK(NodeDefBuilder("parse", "ParseTensor")                  \
                     .Input(FakeInput(DT_STRING))                        \
                     .Attr("out_type", DataTypeToEnum<T>::value)         \
                     .Finalize(&parse));                                 \
    std::unique_ptr<Device> device(                                      \
        DeviceFactory::NewDevice("CPU", {}, "/job:a/replica:0/task:0")); \
    gtl::InlinedVector<TensorValue, 4> inputs;                           \
    inputs.push_back({nullptr, serialize_output});                       \
    Status status;                                                       \
    std::unique_ptr<OpKernel> op(CreateOpKernel(                         \
        DEVICE_CPU, device.get(), cpu_allocator(), parse,                \
        TF_GRAPH_DEF_VERSION, &status));                                 \
    TF_EXPECT_OK(status);                                                \
    OpKernelContext::Params params;                                      \
    params.device = device.get();                                        \
    params.inputs = &inputs;                                             \
    params.frame_iter = FrameAndIter(0, 0);                              \
    params.op_kernel = op.get();                                         \
    std::vector<AllocatorAttributes> attrs;                              \
    test::SetOutputAttrs(&params, &attrs);                               \
    OpKernelContext ctx(&params);                                        \
    op->Compute(&ctx);                                                   \
    TF_EXPECT_OK(status);                                                \
    Tensor* parse_output = ctx.mutable_output(0);                        \
    test::ExpectTensorEqual<T>(*parse_output, input);                    \
  }

#define REGISTER_TEST_REAL(T)         \
  REGISTER_TEST(T, TensorShape({10}), \
                [](int x) -> T { return static_cast<T>(x + 10.); })

#define REGISTER_TEST_COMPLEX(T)              \
  REGISTER_TEST(T, TensorShape({10}),         \
                ([](int x) -> T { return { x + 10., x + 9. }; }))

#define REGISTER_TEST_BOOL(T)         \
  REGISTER_TEST(T, TensorShape({10}), \
                [](int x) -> T { return static_cast<T>(x % 2); })

#define REGISTER_TEST_STRING(T)      \
  REGISTER_TEST(T, TensorShape({2}), \
                [](int x) -> T { return std::to_string(x); })

using Eigen::half;
REGISTER_TEST_REAL(half)
REGISTER_TEST_REAL(float)
REGISTER_TEST_REAL(double)

REGISTER_TEST_REAL(int64)
REGISTER_TEST_REAL(int32)
REGISTER_TEST_REAL(int16)
REGISTER_TEST_REAL(int8)
REGISTER_TEST_REAL(uint16)
REGISTER_TEST_REAL(uint8)

REGISTER_TEST_COMPLEX(complex64)
REGISTER_TEST_COMPLEX(complex128)
REGISTER_TEST_BOOL(bool)
using std::string;
REGISTER_TEST_STRING(string)

}  // namespace
}  // namespace tensorflow
