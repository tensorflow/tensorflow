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

#include "tensorflow/core/framework/tensor_util.h"

#include <vector>
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/stringpiece.h"

namespace tensorflow {
namespace tensor {

Tensor DeepCopy(const Tensor& other) {
  Tensor tmp = Tensor(other.dtype(), other.shape());
  if (DataTypeCanUseMemcpy(other.dtype())) {
    StringPiece other_data = other.tensor_data();

    // We use StringPiece as a convenient map over the tensor buffer,
    // but we cast the type to get to the underlying buffer to do the
    // copy.
    StringPiece tmp_data = tmp.tensor_data();
    memcpy(const_cast<char*>(tmp_data.data()), other_data.data(),
           other_data.size());
  } else {
    CHECK_EQ(DT_STRING, other.dtype());
    tmp.flat<string>() = other.flat<string>();
  }
  return tmp;
}

Tensor Concat(const gtl::ArraySlice<Tensor>& tensors) {
  CHECK_GT(tensors.size(), size_t{0});
  int64 total_dim0_size = 0;
  for (const Tensor& tensor : tensors) {
    CHECK_GT(tensor.dims(), 0);
    total_dim0_size += tensor.dim_size(0);
  }
  TensorShape shape = tensors[0].shape();
  shape.set_dim(0, total_dim0_size);
  Tensor result = Tensor(tensors[0].dtype(), shape);

  // We use StringPiece as a convenient map over the tensor buffer,
  // but we cast the type to get to the underlying buffer to do the
  // copy.
  StringPiece to_data = result.tensor_data();

  if (DataTypeCanUseMemcpy(result.dtype())) {
    int64 offset = 0;
    for (const Tensor& tensor : tensors) {
      StringPiece from_data = tensor.tensor_data();
      CHECK_LE(offset + from_data.size(), to_data.size());
      memcpy(const_cast<char*>(to_data.data()) + offset, from_data.data(),
             from_data.size());

      offset += from_data.size();
    }
  } else {
    CHECK_EQ(DT_STRING, result.dtype());
    string* to_strings =
        reinterpret_cast<string*>(const_cast<char*>(to_data.data()));

    int64 offset = 0;
    for (const Tensor& tensor : tensors) {
      auto from_strings = tensor.flat<string>();
      CHECK_LE(offset + tensor.NumElements(), result.NumElements());
      for (int i = 0; i < tensor.NumElements(); ++i) {
        to_strings[offset + i] = from_strings(i);
      }

      offset += tensor.NumElements();
    }
  }

  return result;
}

std::vector<Tensor> Split(const Tensor& tensor,
                          const gtl::ArraySlice<int64>& sizes) {
  CHECK_GT(tensor.dims(), 0);
  int64 total_size = 0;
  for (int64 size : sizes) {
    total_size += size;
  }
  CHECK_EQ(total_size, tensor.dim_size(0));

  std::vector<Tensor> result;

  StringPiece from_data = tensor.tensor_data();

  if (DataTypeCanUseMemcpy(tensor.dtype())) {
    int64 offset = 0;
    for (int64 size : sizes) {
      TensorShape shape = tensor.shape();
      shape.set_dim(0, size);
      result.emplace_back(tensor.dtype(), shape);
      Tensor* split = &result[result.size() - 1];

      // We use StringPiece as a convenient map over the tensor buffer,
      // but we cast the type to get to the underlying buffer to do the
      // copy.
      StringPiece to_data = split->tensor_data();
      CHECK_LE(offset + to_data.size(), from_data.size());
      memcpy(const_cast<char*>(to_data.data()), from_data.data() + offset,
             to_data.size());

      offset += to_data.size();
    }
  } else {
    CHECK_EQ(DT_STRING, tensor.dtype());
    auto from_strings = tensor.flat<string>();

    int64 offset = 0;
    for (int64 size : sizes) {
      TensorShape shape = tensor.shape();
      shape.set_dim(0, size);
      result.emplace_back(tensor.dtype(), shape);
      Tensor& split = result[result.size() - 1];
      string* to_strings = reinterpret_cast<string*>(
          const_cast<char*>(split.tensor_data().data()));

      CHECK_LE(offset + split.NumElements(), tensor.NumElements());
      for (int i = 0; i < split.NumElements(); ++i) {
        to_strings[i] = from_strings(offset + i);
      }

      offset += split.NumElements();
    }
  }

  return result;
}

}  // namespace tensor
}  // namespace tensorflow
