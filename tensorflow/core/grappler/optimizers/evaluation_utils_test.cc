#include "tensorflow/core/platform/cpu_info.h"
/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#define EIGEN_USE_THREADS

#include "third_party/eigen3/unsupported/Eigen/CXX11/ThreadPool"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/grappler/optimizers/evaluation_utils.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace grappler {

TEST(EvaluationUtilsTest, DeviceSimple_BasicProperties) {
  DeviceSimple dsimple;
  ASSERT_TRUE(dsimple.has_eigen_cpu_device());
  EXPECT_EQ(dsimple.eigen_cpu_device()->numThreads(),
            port::NumSchedulableCPUs());
  const Eigen::ThreadPoolInterface* pool =
      dsimple.eigen_cpu_device()->getPool();
  ASSERT_NE(pool, nullptr);
}

TEST(EvaluationUtilsTest, DeviceSimple_MakeTensorFromProto) {
  DeviceSimple dsimple;

  TensorProto proto;
  Tensor tensor;
  EXPECT_FALSE(dsimple.MakeTensorFromProto(proto, {}, &tensor).ok());

  Tensor original(tensorflow::DT_INT16, TensorShape{4, 2});
  original.flat<int16>().setRandom();

  original.AsProtoTensorContent(&proto);
  TF_ASSERT_OK(dsimple.MakeTensorFromProto(proto, {}, &tensor));

  ASSERT_EQ(tensor.dtype(), original.dtype());
  ASSERT_EQ(tensor.shape(), original.shape());

  auto buf0 = original.flat<int16>();
  auto buf1 = tensor.flat<int16>();
  ASSERT_EQ(buf0.size(), buf1.size());
  for (int i = 0; i < buf0.size(); ++i) {
    EXPECT_EQ(buf0(i), buf1(i));
  }
}
}  // namespace grappler
}  // namespace tensorflow
