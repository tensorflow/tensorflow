/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/framework/memory_types.h"

#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {

class DummyKernel : public OpKernel {
 public:
  explicit DummyKernel(tensorflow::OpKernelConstruction* context)
      : OpKernel(context) {}
  void Compute(tensorflow::OpKernelContext* context) override {}
};

REGISTER_OP("HostMemoryTest")
    .Input("a: float")
    .Input("b: float")
    .Input("c: T")
    .Input("d: N * string")
    .Input("e: Tlist")
    .Input("f: Rlist")
    .Output("o: N * T")
    .Output("p: N * T")
    .Output("r: Tlist")
    .Attr("T: type")
    .Attr("N: int")
    .Attr("Tlist: list(type)")
    .Attr("Rlist: list(type)");
REGISTER_KERNEL_BUILDER(Name("HostMemoryTest").Device(DEVICE_CPU), DummyKernel);
REGISTER_KERNEL_BUILDER(Name("HostMemoryTest")
                            .Device(DEVICE_GPU)
                            .HostMemory("b")
                            .HostMemory("d")
                            .HostMemory("e")
                            .HostMemory("p"),
                        DummyKernel);

TEST(MemoryTypesForNode, Simple) {
  NodeDef node_def;
  TF_ASSERT_OK(NodeDefBuilder("test", "HostMemoryTest")
                   .Input(FakeInput())
                   .Input(FakeInput())
                   .Input(FakeInput(DT_BOOL))
                   .Input(FakeInput(3))
                   .Input(FakeInput({DT_INT32, DT_FLOAT, DT_INT32}))
                   .Input(FakeInput({DT_RESOURCE, DT_STRING, DT_RESOURCE}))
                   .Finalize(&node_def));
  AddNodeAttr("_input_hostmem", {0}, &node_def);
  AddNodeAttr("_output_hostmem", {6, 7}, &node_def);

  MemoryTypeVector input, output;

  TF_EXPECT_OK(MemoryTypesForNode(OpRegistry::Global(), DEVICE_CPU, node_def,
                                  &input, &output));
  // a:float, b:bool, c:3*string, d:(int32, float, int32),
  // e:(resource, string, resource)
  EXPECT_EQ(
      MemoryTypeVector({HOST_MEMORY, DEVICE_MEMORY, DEVICE_MEMORY, HOST_MEMORY,
                        HOST_MEMORY, HOST_MEMORY, DEVICE_MEMORY, DEVICE_MEMORY,
                        DEVICE_MEMORY, HOST_MEMORY, HOST_MEMORY, HOST_MEMORY}),
      input);
  // o:3*bool, p:(int32, float, int32)
  EXPECT_EQ(MemoryTypeVector({DEVICE_MEMORY, DEVICE_MEMORY, DEVICE_MEMORY,
                              DEVICE_MEMORY, DEVICE_MEMORY, DEVICE_MEMORY,
                              HOST_MEMORY, HOST_MEMORY, DEVICE_MEMORY}),
            output);

  TF_EXPECT_OK(MemoryTypesForNode(OpRegistry::Global(), DEVICE_GPU, node_def,
                                  &input, &output));
  EXPECT_EQ(
      MemoryTypeVector({HOST_MEMORY, HOST_MEMORY, DEVICE_MEMORY, HOST_MEMORY,
                        HOST_MEMORY, HOST_MEMORY, HOST_MEMORY, HOST_MEMORY,
                        HOST_MEMORY, HOST_MEMORY, HOST_MEMORY, HOST_MEMORY}),
      input);
  EXPECT_EQ(MemoryTypeVector({DEVICE_MEMORY, DEVICE_MEMORY, DEVICE_MEMORY,
                              HOST_MEMORY, HOST_MEMORY, HOST_MEMORY,
                              HOST_MEMORY, HOST_MEMORY, DEVICE_MEMORY}),
            output);
}

}  // namespace tensorflow
