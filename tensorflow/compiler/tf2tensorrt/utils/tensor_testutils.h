
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

#ifndef TENSORFLOW_COMPILER_TF2TENSORRT_UTILS_TENSOR_TESTUTILS_H_
#define TENSORFLOW_COMPILER_TF2TENSORRT_UTILS_TENSOR_TESTUTILS_H_

#if GOOGLE_CUDA && GOOGLE_TENSORRT

#include <memory>
#include "tensorflow/compiler/tf2tensorrt/utils/trt_testutils.h"
#include "tensorflow/core/common_runtime/gpu/gpu_managed_allocator.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_testutil.h"

namespace tensorflow {
namespace tensorrt {
// This class manages an Allocator of 'AllocatorType' and provides templated
// methods for creating Tensor objects using that allocator. GpuManagedAllocator
// should be used in functional unit tests.
template <typename AllocatorType = GpuManagedAllocator>
class TensorFactory {
 public:
  TensorFactory() : allocator_(std::make_unique<AllocatorType>()) {}

  // Constructs a flat tensor with 'vals'.
  template <typename T>
  Tensor AsTensor(gtl::ArraySlice<T> vals) {  // non-absl ok
    Tensor ret(allocator_.get(), DataTypeToEnum<T>::value,
               {static_cast<int64>(vals.size())});
    std::copy_n(vals.data(), vals.size(), ret.flat<T>().data());
    return ret;
  }

  // Constructs a tensor of "shape" with values "vals".
  template <typename T>
  Tensor AsTensor(gtl::ArraySlice<T> vals,  // non-absl ok
                  const TensorShape& shape) {
    Tensor ret(allocator_.get(), DataTypeToEnum<T>::value,
               {static_cast<int64>(vals.size())});
    CHECK(ret.CopyFrom(AsTensor(vals), shape));
    return ret;
  }

  // Constructs a Tensor with given values 'vals' of shape 'input_dims' and type
  // 'tf_type'.
  template <typename T>
  Tensor AsTensor(const std::vector<T>& vals,
                  const std::vector<int>& input_dims, DataType tf_type) {
    Tensor ret(allocator_.get(), tf_type, {static_cast<int64>(vals.size())});
    if (tf_type == DT_FLOAT) {
      auto conv_vals = convert::CastVector<T, float>(vals);
      std::copy_n(conv_vals.data(), conv_vals.size(), ret.flat<float>().data());
    } else if (tf_type == DT_HALF) {
      auto conv_vals = convert::CastVector<T, Eigen::half>(vals);
      std::copy_n(conv_vals.data(), conv_vals.size(),
                  ret.flat<Eigen::half>().data());
    } else if (tf_type == DT_INT32) {
      auto conv_vals = convert::CastVector<T, int32>(vals);
      std::copy_n(conv_vals.data(), conv_vals.size(), ret.flat<int32>().data());
    } else {
      LOG(FATAL) << "Cannot create tensor with type "
                 << DataTypeString(tf_type);
    }
    TensorShape shape;
    TF_EXPECT_OK(TensorShapeUtils::MakeShape(input_dims, &shape));
    CHECK(ret.CopyFrom(ret, shape));
    return ret;
  }

  // Constructs a flat tensor of the given size and fills with the given value.
  template <typename T>
  Tensor ConstructTensor(int data_size, const T& value = T()) {
    std::vector<T> values(data_size, value);
    return AsTensor<T>(values);
  }

  // Constructs a flat Tensor of the  given size and Datatype, and fills with
  // the given value.
  template <typename T>
  Tensor ConstructTensor(int data_size, const T& value, DataType tf_type) {
    std::vector<T> values(data_size, value);
    return AsTensor<T>(values, {data_size}, tf_type);
  }

 private:
  std::unique_ptr<Allocator> allocator_;
};
}  // namespace tensorrt
}  // namespace tensorflow

#endif  // GOOGLE_CUDA && GOOGLE_TENSORRT
#endif  // TENSORFLOW_COMPILER_TF2TENSORRT_UTILS_TENSOR_TESTUTILS_H_
