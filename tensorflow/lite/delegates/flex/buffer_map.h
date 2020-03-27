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
#ifndef TENSORFLOW_LITE_DELEGATES_FLEX_BUFFER_MAP_H_
#define TENSORFLOW_LITE_DELEGATES_FLEX_BUFFER_MAP_H_

#include <map>

#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/lite/c/common.h"

namespace tflite {
namespace flex {

// Maps a TF Lite tensor index into a TensorFlow tensor.
//
// The TF Lite interpreter assigns integer indices to each of its tensors, but
// the Flex delegate deals in terms of TensorFlow tensors. This class maps
// from indices to tensors and allows the creation of new tensors to be
// associated with a given index.
class BufferMap {
 public:
  BufferMap();
  ~BufferMap();

  // Returns true if the given 'tensor_index' has a corresponding
  // tensorflow::Tensor.
  bool HasTensor(int tensor_index) const;

  // Returns true if the given 'tensor_index' has a corresponding
  // tensorflow::Tensor *and* the content is owned by TensorFlow (that is, the
  // mapping was added by SetFromTensorFlow()).
  bool IsTensorFlowTensor(int tensor_index) const;

  // Returns the tensorflow::Tensor associated with the given 'tensor_index'.
  // Precondition: HasTensor() is true.
  tensorflow::Tensor GetTensor(int tensor_index) const;

  // Associates the given tensorflow::Tensor with the given 'tensor_index'.
  // Note that TensorFlow Tensors share data buffers, so this method is only a
  // shallow copy.
  void SetFromTensorFlow(int tensor_index, tensorflow::Tensor tensor);

  // Same as above but creates a new tensorflow::Tensor with a copy of the
  // given TfLiteTensor's data.
  void SetFromTfLite(int tensor_index, const TfLiteTensor* tensor);

 private:
  // Mapping from TL Lite tensor ID to TensorFlow's Tensor. All tensors that
  // are inputs or outputs of a subgraph will be added here, irrespective of
  // whether their data are managed by TF Lite or TensorFlow.
  std::map<int, tensorflow::Tensor> id_to_tensor_;
  // A list of tensors that are completely managed by TensorFlow. Most of the
  // time, TF Lite will populate tensors that are inputs to subgraphs, while
  // TensorFlow will populate output tensors. Occasionally, however, an input
  // tensor is coming from a previous subgraph and could have been populated by
  // TensorFlow. This set keeps track of all input or output tensors that have
  // been populated by tensorflow.
  std::set<int> owned_by_tf_;
};

}  // namespace flex
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_FLEX_BUFFER_MAP_H_
