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

#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/image_ops.h"
#include "tensorflow/cc/ops/nn_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/common_runtime/kernel_benchmark_testlib.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/conv_ops_gpu.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"
#include "tensorflow/core/public/session.h"

namespace tensorflow {
namespace {
class DepthwiseConvOpTest : public OpsTestBase {
 protected:
  enum class Device { CPU, GPU };

  template <typename T>
  void Run(Device device) {
    if (device == Device::GPU) {
      SetDevice(DEVICE_GPU,
                std::unique_ptr<tensorflow::Device>(DeviceFactory::NewDevice(
                    "GPU", {}, "/job:a/replica:0/task:0")));
    }
    DataType dtype = DataTypeToEnum<T>::value;
    TF_EXPECT_OK(NodeDefBuilder("depthwise_conv2d", "DepthwiseConv2dNative")
                     .Input(FakeInput(dtype))
                     .Input(FakeInput(dtype))
                     .Attr("T", dtype)
                     .Attr("strides", {1, 1, 1, 1})
                     .Attr("padding", "SAME")
                     .Finalize(node_def()));
    TF_EXPECT_OK(InitOp());
    const int depth = 2;
    const int image_width = 2;
    const int image_height = 3;
    const int batch_count = 1;
    // The image matrix is ('first/second' channel):
    // | 1/2  |  3/4  |
    // | 5/6  |  7/8  |
    // | 9/10 | 11/12 |
    Tensor image(dtype, {batch_count, image_height, image_width, depth});
    test::FillValues<T>(&image, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});

    // The filter matrix is:
    // | 1/2 |  7/8  | 13/14 |
    // | 3/4 |  9/10 | 15/16 |
    // | 5/6 | 11/12 | 17/18 |
    const int filter_size = 3;
    const int filter_count = 1;
    Tensor filter(dtype, {filter_size, filter_size, depth, filter_count});
    test::FillValues<T>(&filter, {1, 2, 7, 8, 13, 14, 3, 4, 9, 10, 15, 16, 5, 6,
                                  11, 12, 17, 18});

    AddInputFromArray<T>(image.shape(), image.flat<T>());
    AddInputFromArray<T>(filter.shape(), filter.flat<T>());
    TF_ASSERT_OK(RunOpKernel());

    // We're sliding two 3x3 filters across the 3x2 image, with accesses outside
    // the input set to zero because we're using the 'SAME' padding mode.
    // This means we should end up with this matrix:
    // | 105/150 | 183/95  |
    // | 235/312 | 357/178 |
    // | 187/234 | 261/121 |
    Tensor expected(dtype, image.shape());
    test::FillValues<T>(&expected, {228, 300, 132, 180, 482, 596, 266, 344, 372,
                                    452, 180, 236});
    const Tensor& output = *GetOutput(0);
    // TODO(csigg): This should happen as part of GetOutput.
    TF_EXPECT_OK(device_->Sync());
    test::ExpectTensorNear<T>(expected, output, 1e-5);
  }
};

TEST_F(DepthwiseConvOpTest, DepthwiseConvFloatCpu) { Run<float>(Device::CPU); }
TEST_F(DepthwiseConvOpTest, DepthwiseConvDoubleCpu) {
  Run<double>(Device::CPU);
}
TEST_F(DepthwiseConvOpTest, DepthwiseConvHalfCpu) {
  Run<Eigen::half>(Device::CPU);
}

#ifdef GOOGLE_CUDA
TEST_F(DepthwiseConvOpTest, DepthwiseConvFloatGpu) { Run<float>(Device::GPU); }
TEST_F(DepthwiseConvOpTest, DepthwiseConvDoubleGpu) {
  Run<double>(Device::GPU);
}
TEST_F(DepthwiseConvOpTest, DepthwiseConvHalfGpu) {
  Run<Eigen::half>(Device::GPU);
}
#endif

}  // namespace
}  // namespace tensorflow
