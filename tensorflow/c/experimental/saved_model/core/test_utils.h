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

#ifndef TENSORFLOW_C_EXPERIMENTAL_SAVED_MODEL_CORE_TEST_UTILS_H_
#define TENSORFLOW_C_EXPERIMENTAL_SAVED_MODEL_CORE_TEST_UTILS_H_

#include <memory>
#include <vector>

#include "absl/types/span.h"
#include "tensorflow/c/eager/immediate_execution_context.h"
#include "tensorflow/c/eager/immediate_execution_tensor_handle.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/eager/context.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace testing {

// Creates a DeviceMgr suitable for local tests.
std::unique_ptr<StaticDeviceMgr> CreateTestingDeviceMgr();

// Creates an EagerContext suitable for local tests. Does not take ownership
// of `device_mgr`.
EagerContextPtr CreateTestingEagerContext(DeviceMgr* device_mgr);

// Converts a tensorflow::DatatypeSet to std::vector<DataType>.
// This is useful for tests using GTest's ::testing::ValuesIn, since
// DataTypeSet doesn't fullfill all the constraints of an STL-like iterable.
std::vector<DataType> DataTypeSetToVector(DataTypeSet set);

// Returns a vector of shapes intended to be "interesting" test cases.
// Currently, this returns scalar, 1D vector, 2D matrix, and a 4D tensor shapes
std::vector<std::vector<int64>> InterestingShapes();

// Returns a TensorHandle of `dtype` and `shape`, filled with `value`.
// `dtype` must be an integer dtype, float, or double.
// If a TensorHandle cannot be created successfully, this function will
// CHECK fail. This should only be used for testing purposes.
ImmediateTensorHandlePtr CreateTensorHandle(ImmediateExecutionContext* ctx,
                                            DataType dtype,
                                            absl::Span<const int64> shape,
                                            int8 value);

// Fills a numeric tensor's buffer with `value`.
// dtype must be any integer dtype, float or double.
void FillNumericTensorBuffer(DataType dtype, size_t num_elements, void* buffer,
                             int8 value);

// Checks the underlying data is equal for the buffers for two numeric tensors.
// Note: The caller must ensure to check that the dtypes and sizes of the
// underlying buffers are the same before calling this.
// dtype must be any integer dtype, float, or double.
void CheckBufferDataIsEqual(DataType dtype, int64 num_elements, void* a,
                            void* b);

// Converts a TensorHandle to a Tensor, and dies if unsuccessful. This should
// only be used for testing purposes.
AbstractTensorPtr TensorHandleToTensor(ImmediateExecutionTensorHandle* handle);

}  // namespace testing
}  // namespace tensorflow

#endif  // TENSORFLOW_C_EXPERIMENTAL_SAVED_MODEL_CORE_TEST_UTILS_H_
