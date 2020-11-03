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

#include "mli_slicers.h"  // NOLINT

#include <algorithm>

namespace tflite {
namespace ops {
namespace micro {

TensorSlicer::TensorSlicer(const mli_tensor* full_tensor, int slice_dim,
                           int slice_size, int padding_pre, int padding_post,
                           int overlap, bool interleave_mode)
    : full_tensor_(full_tensor),
      sub_tensor_{},
      sub_cfg_{},
      done_(false),
      sliceDim_(slice_dim),
      pad_pre_(padding_pre),
      pad_post_(padding_post),
      overlap_(overlap) {
  /* In the interleave mode, the slicing happens from the deepest dimension up
  to the slice_dim for example in an HWC layout this can mode can be used to
  slice in the C dimenstion. in this mode the data is not contiguous in memory
  anymore */
  if (interleave_mode) {
    for (int i = 0; i < static_cast<int>(full_tensor->rank); i++) {
      if (i > slice_dim) {
        sub_cfg_.size[i] = 1;
      } else if (i == slice_dim) {
        sub_cfg_.size[i] = slice_size;
      } else {
        sub_cfg_.size[i] = full_tensor->shape[i];
      }
    }
    sub_cfg_.sub_tensor_rank = full_tensor->rank;

  } else {
    /* In the not interleaved mode, the slicing happens from the outer most
    dimension up to the slice_dim for example in an HWC layout this mode can be
    used to slice in the H dimension. in this mode the data of the slice is
    still contiguous in memory (if that was the case in the input tensor */
    for (int i = 0; i < static_cast<int>(full_tensor->rank); i++) {
      if (i < slice_dim) {
        sub_cfg_.size[i] = 1;
      } else if (i == slice_dim) {
        sub_cfg_.size[i] = slice_size;
      } else {
        sub_cfg_.size[i] = full_tensor->shape[i];
      }
    }
    sub_cfg_.sub_tensor_rank = full_tensor->rank - slice_dim;
  }

  ComputeSubTensor();
}

void TensorSlicer::ComputeSubTensor(void) {
  // subtsr_cfg_ is used to keep track of the iteration.
  // A copy is created to update it with the correct clipping and padding for
  // the current slice
  mli_sub_tensor_cfg cfg_new = sub_cfg_;

  // begin and end spans the complete input region including padding areas.
  const int begin = (int)sub_cfg_.offset[sliceDim_] - pad_pre_;
  // end is clipped to the end of the full input region. this is needed for
  // cases where the last slice is smaller than the rest.
  const int end = std::min(begin + sub_cfg_.size[sliceDim_] + overlap_,
                           full_tensor_->shape[sliceDim_] + pad_post_);
  // The start coordinate of the subtensor is clipped to zero
  cfg_new.offset[sliceDim_] = std::max(begin, 0);
  // and the stop coordinate is clipped to the size of the full tensor
  const int stop_coord =
      std::min(end, static_cast<int>(full_tensor_->shape[sliceDim_]));
  // compute the size of the subtensor
  cfg_new.size[sliceDim_] = stop_coord - cfg_new.offset[sliceDim_];

  // compute the padding configuration for the current slice.
  actual_padding_pre = cfg_new.offset[sliceDim_] - begin;
  actual_padding_post = end - stop_coord;

  mli_hlp_create_subtensor(full_tensor_, &cfg_new, &sub_tensor_);
}

void TensorSlicer::Next(void) {
  for (int i = full_tensor_->rank - 1; i >= 0; i--) {
    sub_cfg_.offset[i] += sub_cfg_.size[i];
    if (sub_cfg_.offset[i] >= full_tensor_->shape[i]) {
      // wrap
      sub_cfg_.offset[i] = 0;
      // and continue to the next dimension, if no next dimension we are done.
      if (i == 0) done_ = true;
      continue;
    } else {
      // carry is false, so break from the loop
      break;
    }
  }

  if (!done_) ComputeSubTensor();
}

bool TensorSlicer::Done(void) { return done_; }

int TensorSlicer::GetPaddingPre(void) { return actual_padding_pre; }

int TensorSlicer::GetPaddingPost(void) { return actual_padding_post; }

mli_tensor* TensorSlicer::Sub(void) { return &sub_tensor_; }

}  // namespace micro
}  // namespace ops
}  // namespace tflite
