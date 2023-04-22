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

#include <algorithm>
#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include "absl/synchronization/notification.h"
#include "tensorflow/compiler/jit/flags.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/device_base.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/notification.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/util/port.h"

namespace tensorflow {
namespace {

static bool Initialized = [] {
  tensorflow::GetXlaDeviceFlags()->tf_xla_enable_xla_devices = true;
  return true;
}();

class UnaryOpsCompositionTest : public OpsTestBase {
 protected:
  template <typename T>
  void RunComposedOp(const std::vector<string> op_names, T input_scalar_value,
                     T expected_scalar_value) {
    string xla_device_name =
        tensorflow::IsGoogleCudaEnabled() ? DEVICE_XLA_GPU : DEVICE_XLA_CPU;
    SetDevice(DeviceType(xla_device_name),
              std::unique_ptr<tensorflow::Device>(DeviceFactory::NewDevice(
                  xla_device_name, {}, "/job:a/replica:0/task:0")));

    TF_ASSERT_OK(NodeDefBuilder("unary_op_composition", "_UnaryOpsComposition")
                     .Input(FakeInput(DataTypeToEnum<T>::v()))
                     .Attr("T", DataTypeToEnum<T>::v())
                     .Attr("op_names", op_names)
                     .Finalize(node_def()));
    TF_ASSERT_OK(InitOp());

    // We're using an XLA device here which allocates XlaTensors.  We can't
    // inspect XlaTensors directly so we create the input on the host and copy
    // it over to the XLA device.  We do the inverse on the output.

    TensorShape shape({});

    AllocatorAttributes host_alloc_attrs;
    host_alloc_attrs.set_gpu_compatible(true);
    host_alloc_attrs.set_on_host(true);
    Allocator* cpu_allocator = device_->GetAllocator(host_alloc_attrs);

    DataType dtype = DataTypeToEnum<T>::value;

    Tensor input_on_host(cpu_allocator, dtype, shape);
    test::FillValues<T>(&input_on_host, {input_scalar_value});

    Tensor* input = AddInput(dtype, shape);

    DeviceContext* device_context =
        device_->tensorflow_gpu_device_info()->default_context;

    TF_CHECK_OK(device_context->CopyCPUTensorToDeviceSync(&input_on_host,
                                                          device_, input));

    TF_ASSERT_OK(RunOpKernel());

    Tensor expected_tensor(cpu_allocator, dtype, shape);
    test::FillValues<T>(&expected_tensor, {expected_scalar_value});

    Tensor* output = GetOutput(0);
    Tensor output_on_host(cpu_allocator, output->dtype(), output->shape());

    TF_CHECK_OK(device_context->CopyDeviceTensorToCPUSync(
        output, "output 0", device_, &output_on_host));

    test::ExpectClose(expected_tensor, output_on_host, /*atol=*/1e-5,
                      /*rtol=*/1e-5);
  }
};

TEST_F(UnaryOpsCompositionTest, Compose_Sqrt_Sqrt_F) {
  RunComposedOp<float>({"Sqrt", "Sqrt"}, 81.0, 3.0);
}

TEST_F(UnaryOpsCompositionTest, Compose_Sqrt_Sqrt_D) {
  RunComposedOp<double>({"Sqrt", "Sqrt"}, 81.0, 3.0);
}

TEST_F(UnaryOpsCompositionTest, Compose_Sqrt_Sin_F) {
  RunComposedOp<float>({"Sqrt", "Sin"}, 81.0, std::sin(9.0f));
}

TEST_F(UnaryOpsCompositionTest, Compose_Cos_Acos_F) {
  RunComposedOp<float>({"Cos", "Acos"}, 0.5, std::acos(std::cos(0.5f)));
}

TEST_F(UnaryOpsCompositionTest, Compose_Tanh_Relu_F) {
  RunComposedOp<float>({"Tanh", "Relu"}, 0.5, std::max(0.0f, std::tanh(0.5f)));
}

TEST_F(UnaryOpsCompositionTest, Compose_Tanh_Relu_D) {
  RunComposedOp<double>({"Tanh", "Relu"}, 0.5, std::max(0.0, std::tanh(0.5)));
}

TEST_F(UnaryOpsCompositionTest, Compose_Tanh_Relu6_F) {
  RunComposedOp<float>({"Relu6"}, 11.0f, 6.0f);
}
}  // namespace
}  // end namespace tensorflow
