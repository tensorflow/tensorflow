/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/c/experimental/saved_model/core/saved_model_utils.h"

#include <string.h>

#include <memory>
#include <vector>

#include "tensorflow/c/eager/immediate_execution_tensor_handle.h"
#include "tensorflow/c/experimental/saved_model/core/revived_types/constant.h"
#include "tensorflow/c/tensor_interface.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/eager/context.h"
#include "tensorflow/core/framework/numeric_types.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/bfloat16/bfloat16.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace {

// Converts a tensorflow::DatatypeSet to std::vector<DataType>.
// This is needed for GTest's ::testing::ValuesIn, since
// DataTypeSet doesn't fullfill all the constraints of an STL-like iterable.
std::vector<DataType> DataTypeSetToVector(DataTypeSet set) {
  std::vector<DataType> result;
  result.reserve(set.size());
  for (DataType dt : set) {
    result.push_back(dt);
  }
  return result;
}

// Returns a vector of shapes intended to be "interesting" test cases.
std::vector<std::vector<int64>> InterestingShapes() {
  std::vector<std::vector<int64>> interesting_shapes;
  interesting_shapes.push_back({});             // Scalar
  interesting_shapes.push_back({10});           // 1D Vector
  interesting_shapes.push_back({3, 3});         // 2D Matrix
  interesting_shapes.push_back({1, 4, 6, 10});  // Higher Dimension Tensor
  return interesting_shapes;
}

// Fills a numeric tensor with `value`.
void FillNumericTensor(Tensor* tensor, int8 value) {
  switch (tensor->dtype()) {
#define CASE(type)                                    \
  case DataTypeToEnum<type>::value: {                 \
    const auto& flattened = tensor->flat<type>();     \
    for (int i = 0; i < tensor->NumElements(); ++i) { \
      flattened(i) = value;                           \
    }                                                 \
    break;                                            \
  }
    TF_CALL_INTEGRAL_TYPES(CASE);
    TF_CALL_double(CASE);
    TF_CALL_float(CASE);
#undef CASE
    default:
      CHECK(false) << "Unsupported data type: "
                   << DataTypeString(tensor->dtype());
      break;
  }
}

// Checks the underlying data is equal for the buffers for two numeric tensors.
// Note: The caller must ensure to check that the dtypes and sizes of the
// underlying buffers are the same before calling this.
void CheckBufferDataIsEqual(DataType dtype, int64 num_elements, void* a,
                            void* b) {
  switch (dtype) {
#define CASE(type)                               \
  case DataTypeToEnum<type>::value: {            \
    type* typed_a = static_cast<type*>(a);       \
    type* typed_b = static_cast<type*>(b);       \
    for (int64 i = 0; i < num_elements; ++i) {   \
      if (DataTypeIsFloating(dtype)) {           \
        EXPECT_FLOAT_EQ(typed_a[i], typed_b[i]); \
      } else {                                   \
        EXPECT_EQ(typed_a[i], typed_b[i]);       \
      }                                          \
    }                                            \
    break;                                       \
  }
    TF_CALL_INTEGRAL_TYPES(CASE);
    TF_CALL_double(CASE);
    TF_CALL_float(CASE);
#undef CASE
    default:
      CHECK(false) << "Unsupported data type: " << DataTypeString(dtype);
  }
}

class ConstantTest : public ::testing::TestWithParam<
                         std::tuple<DataType, std::vector<int64>, bool>> {
 public:
  ConstantTest()
      : device_mgr_(std::make_unique<StaticDeviceMgr>(DeviceFactory::NewDevice(
            "CPU", {}, "/job:localhost/replica:0/task:0"))),
        ctx_(new EagerContext(
            SessionOptions(),
            tensorflow::ContextDevicePlacementPolicy::DEVICE_PLACEMENT_SILENT,
            tensorflow::ContextMirroringPolicy::MIRRORING_NONE,
            /* async= */ false,
            /* lazy_copy_function_remote_inputs= */ false, device_mgr_.get(),
            /* device_mgr_owned= */ false, /* rendezvous= */ nullptr,
            /* custom_kernel_creator= */ nullptr,
            /* cluster_flr= */ nullptr)) {}

  EagerContext* context() { return ctx_.get(); }

 private:
  std::unique_ptr<StaticDeviceMgr> device_mgr_;
  EagerContextPtr ctx_;
};

// Basic sanity check that roundtripping a Tensor->Tensorproto->Constant
// preserves values.
TEST_P(ConstantTest, CreateConstantSuccessful) {
  // Get test parameters
  auto& test_params = GetParam();
  DataType dtype = std::get<0>(test_params);
  TensorShape shape(std::get<1>(test_params));
  bool tensorproto_use_tensor_content = std::get<2>(test_params);

  // Construct a Tensor with the given dtype + shape
  Tensor expected(dtype, shape);
  FillNumericTensor(&expected, 42);

  // Serialize it to a Tensorproto
  TensorProto proto;
  if (tensorproto_use_tensor_content) {
    expected.AsProtoTensorContent(&proto);
  } else {
    expected.AsProtoField(&proto);
  }

  // Revival should succeed w/o errors
  std::unique_ptr<Constant> revived;
  TF_EXPECT_OK(internal::TensorProtoToConstant(context(), proto, &revived));

  // The revived tensorhandle should have the exact same dtype, shape, +
  // approx equivalent data to the original.
  ImmediateExecutionTensorHandle* handle = revived->handle();
  Status status;
  AbstractTensorPtr revived_tensor(handle->Resolve(&status));
  TF_EXPECT_OK(status) << "Failed to convert tensorhandle to tensor";
  EXPECT_EQ(revived_tensor->Type(), expected.dtype());
  EXPECT_EQ(revived_tensor->NumElements(), expected.NumElements());
  EXPECT_EQ(revived_tensor->NumDims(), expected.dims());
  for (int i = 0; i < expected.dims(); ++i) {
    EXPECT_EQ(revived_tensor->Dim(i), expected.dim_size(i));
  }

  CheckBufferDataIsEqual(expected.dtype(), expected.NumElements(),
                         revived_tensor->Data(), expected.data());
}

// Test against combinations of tensors that are
// 1. Varying dtypes
// 2. Varying shapes
// 3. TensorProto serialized using tensor_content vs repeated type
INSTANTIATE_TEST_SUITE_P(
    ConstantIntegerDtypesTest, ConstantTest,
    ::testing::Combine(
        ::testing::ValuesIn(DataTypeSetToVector(kDataTypeIsInteger)),
        ::testing::ValuesIn(InterestingShapes()),
        ::testing::Values(false, true)));

INSTANTIATE_TEST_SUITE_P(
    ConstantFloatingDtypesTest, ConstantTest,
    ::testing::Combine(::testing::Values(DT_FLOAT, DT_DOUBLE),
                       ::testing::ValuesIn(InterestingShapes()),
                       ::testing::Values(false, true)));

}  // namespace
}  // namespace tensorflow
