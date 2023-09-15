/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/runtime_fallback/util/tensor_util.h"

#include <cstddef>
#include <memory>

#include <gmock/gmock.h>
#include "Eigen/Core"  // from @eigen_archive  // IWYU pragma: keep
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/platform/protobuf.h"  // IWYU pragma: keep
#include "tensorflow/core/platform/test.h"
#include "tfrt/dtype/dtype.h"  // from @tf_runtime
#include "tfrt/host_context/concurrent_work_queue.h"  // from @tf_runtime
#include "tfrt/host_context/diagnostic.h"  // from @tf_runtime
#include "tfrt/host_context/host_allocator.h"  // from @tf_runtime
#include "tfrt/host_context/host_buffer.h"  // from @tf_runtime
#include "tfrt/host_context/host_context.h"  // from @tf_runtime
#include "tfrt/tensor/dense_host_tensor.h"  // from @tf_runtime
#include "tfrt/tensor/string_host_tensor.h"  // from @tf_runtime
#include "tfrt/tensor/tensor_metadata.h"  // from @tf_runtime
#include "tfrt/tensor/tensor_shape.h"  // from @tf_runtime

namespace tensorflow {
namespace tfd {
namespace {

using ::tensorflow::protobuf::TextFormat;

std::unique_ptr<tfrt::HostContext> CreateTestHostContext() {
  return std::make_unique<tfrt::HostContext>(
      [](const tfrt::DecodedDiagnostic&) {}, tfrt::CreateMallocAllocator(),
      tfrt::CreateSingleThreadedWorkQueue());
}

TEST(TensorUtilTest, MoveHostBufferToTfTensorOk) {
  // 4 int32
  ssize_t size = 4 * 4;
  auto host_context = CreateTestHostContext();
  tfrt::TensorMetadata metadata(tfrt::DType(tfrt::DType::I32),
                                llvm::ArrayRef(4));
  auto* ptr =
      host_context->AllocateBytes(size, /*alignment=*/EIGEN_MAX_ALIGN_BYTES);
  tfrt::DenseHostTensor dht(
      metadata, tfrt::HostBuffer::CreateFromExternal(
                    ptr, size, [&host_context](void* ptr, size_t size) {
                      host_context->DeallocateBytes(ptr, size);
                    }));
  tensorflow::Tensor tensor =
      MoveHostBufferToTfTensor(dht.ReleaseBuffer(), dht.dtype(), dht.shape());
  EXPECT_EQ(tensor.dtype(), DT_INT32);
  EXPECT_EQ(tensor.NumElements(), 4);
}

TEST(TensorUtilTest, CopyShtToTfTensorOk) {
  auto host_context = CreateTestHostContext();
  tfrt::TensorShape shape(1);
  auto sht =
      tfrt::StringHostTensor::CreateUninitialized(shape, host_context.get());
  sht->strings()[0] = "Tensorflow runtime";
  tensorflow::Tensor tensor = CopyShtToTfTensor(*sht);
  EXPECT_THAT(reinterpret_cast<char*>(tensor.data()),
              ::testing::HasSubstr("Tensorflow runtime"));
}

TEST(TensorUtilTest, GetTfShapeOk) {
  tfrt::TensorShape shape{{2, 3}};
  tensorflow::TensorShape tf_shape = GetTfShape(shape);
  EXPECT_EQ(tf_shape.dims(), 2);
  EXPECT_EQ(tf_shape.dim_size(0), 2);
  EXPECT_EQ(tf_shape.dim_size(1), 3);
}

TEST(TensorUtilTest, GetTensorMetadataOk) {
  tensorflow::TensorProto tensor_pb;
  ASSERT_TRUE(TextFormat::ParseFromString(
      R"pb(
        dtype: DT_BOOL
        tensor_shape {
          dim:
          [ { size: 3 }
            , { size: 2 }]
        }
        bool_val: [ false, false, false, false, false, false ]
      )pb",
      &tensor_pb));
  tensorflow::Tensor tensor;
  ASSERT_TRUE(tensor.FromProto(tensor_pb));
  tfrt::TensorMetadata metadata = GetTensorMetadata(tensor);
  EXPECT_EQ(metadata.dtype, tfrt::DType::I1);
  EXPECT_EQ(metadata.shape, tfrt::TensorShape({3, 2}));
}
}  // namespace
}  // namespace tfd
}  // namespace tensorflow
