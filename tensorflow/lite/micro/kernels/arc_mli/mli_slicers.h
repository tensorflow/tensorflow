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

#ifndef TENSORFLOW_LITE_MICRO_KERNELS_ARC_MLI_SLICERS_H_
#define TENSORFLOW_LITE_MICRO_KERNELS_ARC_MLI_SLICERS_H_

#include "mli_api.h"  // NOLINT
namespace tflite {
namespace ops {
namespace micro {

class TensorSlicer {
 public:
  TensorSlicer(const mli_tensor* full_tensor, int slice_dim, int slice_size,
               int padding_pre = 0, int padding_post = 0, int overlap = 0,
               bool interleave_mode = false);
  ~TensorSlicer() = default;

  void Next();
  bool Done();
  int GetPaddingPre();
  int GetPaddingPost();

  mli_tensor* Sub();

  // Default constructor is deleted
  TensorSlicer() = delete;

 private:
  const mli_tensor* full_tensor_;
  mli_tensor sub_tensor_;
  mli_sub_tensor_cfg sub_cfg_;
  bool done_;
  int sliceDim_;
  int pad_pre_, pad_post_, overlap_;
  int actual_padding_pre, actual_padding_post;

  void ComputeSubTensor();
};

}  // namespace micro
}  // namespace ops
}  // namespace tflite
#endif  // TENSORFLOW_LITE_MICRO_KERNELS_ARC_MLI_SLICERS_H_
