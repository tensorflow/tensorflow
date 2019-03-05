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

#include <string>
#include <vector>

#include "absl/algorithm/container.h"
#include "tensorflow/cc/ops/array_ops.h"
#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/nn_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/public/session.h"

namespace tensorflow {
namespace {
Status GetTestDevice(Session* session, string* test_device) {
  std::vector<DeviceAttributes> devices;
  TF_RETURN_IF_ERROR(session->ListDevices(&devices));

  bool found_cpu = absl::c_any_of(devices, [&](const DeviceAttributes& device) {
    return device.device_type() == "CPU";
  });

  bool found_gpu = absl::c_any_of(devices, [&](const DeviceAttributes& device) {
    return device.device_type() == "GPU";
  });

  if (!found_gpu && !found_cpu) {
    return errors::Internal("Expected at least one CPU or GPU!");
  }

  *test_device = found_gpu ? "GPU" : "CPU";
  VLOG(2) << "Using test device " << *test_device;
  return Status::OK();
}

void FillZeros(Tensor* tensor) {
  auto flat = tensor->flat<float>();
  for (int i = 0; i < flat.size(); i++) {
    flat.data()[i] = 0.0f;
  }
}

// This tests check that the implementation outputs from FusedBatchnorm
// training, reserve_space_{1|2}, are what we assume them to be in the TF/XLA
// lowering.
//
// If this test starts failing then it doesn't indicate that TF/cudnn have
// violated their contract, but it indicates that we need to update the TF/XLA
// lowering for FusedBatchnorm training to match the new implementation defined
// behavior.
TEST(FusedBatchnormReserveSpaceTest, Test) {
  using ::tensorflow::ops::Const;
  using ::tensorflow::ops::FusedBatchNorm;

  std::unique_ptr<tensorflow::Session> session(
      tensorflow::NewSession(tensorflow::SessionOptions{}));

  string test_device;
  TF_ASSERT_OK(GetTestDevice(session.get(), &test_device));

  Scope root = tensorflow::Scope::NewRootScope();
  Output input = ops::Placeholder(root.WithOpName("input"), DT_FLOAT);

  Tensor scale_data(DT_FLOAT, TensorShape({10}));
  FillZeros(&scale_data);
  Output scale =
      Const(root.WithOpName("scale"), Input::Initializer(scale_data));

  Tensor offset_data(DT_FLOAT, TensorShape({10}));
  FillZeros(&offset_data);
  Output offset =
      Const(root.WithOpName("offset"), Input::Initializer(offset_data));

  Tensor mean_data(DT_FLOAT, TensorShape({0}));
  Output mean = Const(root.WithOpName("offset"), Input::Initializer(mean_data));

  Tensor variance_data(DT_FLOAT, TensorShape({0}));
  Output variance =
      Const(root.WithOpName("variance"), Input::Initializer(variance_data));

  string tf_device = absl::StrCat("/device:", test_device, ":0");
  string xla_device = absl::StrCat("/device:XLA_", test_device, ":0");

  FusedBatchNorm fused_batch_norm_tf(
      root.WithOpName("fused_batch_norm_tf").WithDevice(tf_device), input,
      scale, offset, mean, variance, FusedBatchNorm::Attrs{}.IsTraining(true));
  FusedBatchNorm fused_batch_norm_xla(
      root.WithOpName("fused_batch_norm_xla").WithDevice(xla_device), input,
      scale, offset, mean, variance, FusedBatchNorm::Attrs{}.IsTraining(true));

  tensorflow::GraphDef graph;
  TF_ASSERT_OK(root.ToGraphDef(&graph));

  TF_ASSERT_OK(session->Create(graph));

  Tensor input_data(DT_FLOAT, TensorShape({10, 10, 10, 10}));
  auto flat_input = input_data.flat<float>();
  for (int i = 0; i < flat_input.size(); i++) {
    flat_input.data()[i] = (i - 5) / 1000.0f;
  }

  std::vector<Tensor> results;
  TF_ASSERT_OK(session->Run({{"input", input_data}},
                            {fused_batch_norm_tf.reserve_space_1.name(),
                             fused_batch_norm_xla.reserve_space_1.name(),
                             fused_batch_norm_tf.reserve_space_2.name(),
                             fused_batch_norm_xla.reserve_space_2.name()},
                            {}, &results));

  test::ExpectClose(results[0], results[1], /*atol=*/1e-4);
  test::ExpectClose(results[2], results[3], /*atol=*/1e-4);
}
}  // namespace
}  // namespace tensorflow
