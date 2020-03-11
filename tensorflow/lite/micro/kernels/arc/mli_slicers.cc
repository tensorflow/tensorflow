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

#include "mli_slicers.h"

#define MAX(A,B) (((A) > (B))? (A): (B))
#define MIN(A,B) (((A) > (B))? (B): (A)) 

namespace tflite {
namespace ops {
namespace micro {

TensorSlicer::TensorSlicer(const mli_tensor* full_tensor, int slice_dim, int slice_size, int padding_pre, int padding_post, int overlap)
  : full_tensor_(full_tensor)
  , sliceDim_(slice_dim)
  , pad_pre_(padding_pre)
  , pad_post_(padding_post)
  , overlap_(overlap)
  , subtsr_cfg_{ {0, 0}, static_cast<uint8_t>(slice_dim + 1), static_cast<uint8_t>(slice_size) }
  , sub_tensor_{0}
  , done_(false){

  ComputeSubTensor();
}

void TensorSlicer::ComputeSubTensor(void) {
  // subtsr_cfg_ is used to keep track of the itteration.
  // A copy is created to update it with the correct clipping and padding for the current slice
  mli_point_to_subtsr_cfg cfg_new = subtsr_cfg_;
  // add clipping of first_out_dim_size to not exceed total size in that dimensions
  // add padding logic

  // begin and end spans the complete input region including padding areas.
  const int begin = (int)subtsr_cfg_.start_coord[1] - pad_pre_;
  // end is clipped to the end of the full input region. this is needed for cases where the last slice is smaller than the rest.
  const int end = MIN(begin + subtsr_cfg_.first_out_dim_size + overlap_, full_tensor_->shape[sliceDim_] + pad_post_);
  // The start coordinate of the subtensor is clipped to zero
  cfg_new.start_coord[sliceDim_] = MAX(begin, 0);
  // and the stop coordinate is clipped to the size of the full tensor
  const int stop_coord = MIN(end, full_tensor_->shape[sliceDim_]);
  // compute the size of the subtensor
  cfg_new.first_out_dim_size = stop_coord - cfg_new.start_coord[sliceDim_];

  // compute the padding configuration for the current slice.
  actual_padding_pre = cfg_new.start_coord[sliceDim_] - begin;
  actual_padding_post = end - stop_coord;

  mli_hlp_point_to_subtensor(full_tensor_, &cfg_new, &sub_tensor_);
}
void TensorSlicer::Next(void){
  // TODO make generic for any number of dimensions.
  subtsr_cfg_.start_coord[1]+= subtsr_cfg_.first_out_dim_size;
  if (subtsr_cfg_.start_coord[1] >= full_tensor_->shape[1]) {
    subtsr_cfg_.start_coord[1] = 0;
    subtsr_cfg_.start_coord[0]++;
    if (subtsr_cfg_.start_coord[0] >= full_tensor_->shape[0]) {
      done_ = true;
    }
  }
  if (!done_) ComputeSubTensor();
}

bool TensorSlicer::Done(void) {
  return done_;
}

int TensorSlicer::GetPaddingPre(void) {
  return actual_padding_pre;
}

int TensorSlicer::GetPaddingPost(void) {
  return actual_padding_post;
}

mli_tensor* TensorSlicer::Sub(void) {
  return &sub_tensor_;
}

}  // namespace micro
}  // namespace ops
}  // namespace tflite
