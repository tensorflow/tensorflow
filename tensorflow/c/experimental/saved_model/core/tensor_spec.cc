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

#include "tensorflow/c/experimental/saved_model/core/tensor_spec.h"

#include <cstdint>
#include <initializer_list>
#include <utility>

#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/protobuf/struct.pb.h"

namespace tensorflow {

TensorSpec::TensorSpec()
    : shape_(std::initializer_list<int64_t>()), dtype_(DT_FLOAT) {}

TensorSpec::TensorSpec(PartialTensorShape shape, DataType dtype)
    : shape_(std::move(shape)), dtype_(dtype) {}

TensorSpec::TensorSpec(const TensorSpecProto& proto)
    : shape_(proto.shape()), dtype_(proto.dtype()) {}

const PartialTensorShape& TensorSpec::shape() const { return shape_; }

DataType TensorSpec::dtype() const { return dtype_; }

}  // namespace tensorflow
