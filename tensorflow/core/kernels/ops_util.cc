/* Copyright 2015 Google Inc. All Rights Reserved.

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

#include <cmath>

#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/util/padding.h"

namespace tensorflow {

Status Get2dOutputSize(const int in_height, const int in_width,
                       int filter_height, int filter_width, int row_stride,
                       int col_stride, Padding padding, int* new_height,
                       int* new_width, int* pad_rows, int* pad_cols) {
  int pad_bottom_unused, pad_right_unused;
  return Get2dOutputSizeVerbose(
      in_height, in_width, filter_height, filter_width, row_stride, col_stride,
      padding, new_height, new_width, pad_rows, &pad_bottom_unused, pad_cols,
      &pad_right_unused);
}

Status Get2dOutputSizeVerbose(const int in_height, const int in_width,
                              int filter_height, int filter_width,
                              int row_stride, int col_stride, Padding padding,
                              int* new_height, int* new_width, int* pad_top,
                              int* pad_bottom, int* pad_left, int* pad_right) {
  // Cannot have strides larger than the patch size.
  if (row_stride > filter_height || col_stride > filter_width) {
    return errors::InvalidArgument(
        "stride must be less than or equal to kernel size");
  }
  switch (padding) {
    case Padding::VALID:
      *new_height = ceil((in_height - filter_height + 1.f) /
                         static_cast<float>(row_stride));
      *new_width = ceil((in_width - filter_width + 1.f) /
                        static_cast<float>(col_stride));
      *pad_top = 0;
      *pad_bottom = 0;
      *pad_left = 0;
      *pad_right = 0;
      break;
    case Padding::SAME:
      *new_height = ceil(in_height / static_cast<float>(row_stride));
      *new_width = ceil(in_width / static_cast<float>(col_stride));
      // Calculate padding for top/bottom/left/right, spilling any excess
      // padding to bottom and right.
      const int pad_needed_height =
          (*new_height - 1) * row_stride + filter_height - in_height;
      *pad_top = pad_needed_height / 2;
      CHECK_GE(pad_needed_height, 0);
      *pad_bottom = pad_needed_height - *pad_top;

      const int pad_needed_width =
          (*new_width - 1) * col_stride + filter_width - in_width;
      *pad_left = pad_needed_width / 2;
      CHECK_GE(pad_needed_width, 0);
      *pad_right = pad_needed_width - *pad_left;
      break;
  }
  if (*new_height < 0 || *new_width < 0) {
    return errors::InvalidArgument("computed output size would be negative");
  }
  return Status::OK();
}

Eigen::PaddingType BrainPadding2EigenPadding(Padding padding) {
  switch (padding) {
    case Padding::VALID:
      return Eigen::PADDING_VALID;
    case Padding::SAME:
      return Eigen::PADDING_SAME;
  }
  return Eigen::PADDING_SAME;  // Prevent compiler warning about missing return
}

Status GetBroadcastSize(const int index, const int in_size, const int ksize,
                        const int stride, const int pad_size, int* bindex,
                        int* bsize) {
  // Cannot have strides larger than the patch size.
  if (stride > ksize) {
    return errors::InvalidArgument(
        "stride must be less than or equal to kernel size");
  }
  // Cannot have index beyond the input size.
  if (index * stride > in_size) {
    return errors::InvalidArgument(
        "index * stride must be less than or equal to input size");
  }
  *bindex = index * stride;
  *bsize = ksize;
  if (*bindex < pad_size) {
    // If the current index is in the padding area, start broadcast  from index
    // 0 with broadcast size reduced by padding size.
    *bsize = ksize + *bindex - pad_size;
    *bindex = 0;
  } else {
    // Otherwise, start broadcast from current index reduced by padding size.
    *bindex -= pad_size;
  }
  if (*bindex + ksize > in_size) {
    *bsize = std::min((in_size - *bindex), ksize);
  }
  return Status::OK();
}

string SanitizeThreadSuffix(string suffix) {
  string clean;
  for (int i = 0; i < suffix.size(); ++i) {
    const char ch = suffix[i];
    if ((ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') ||
        (ch >= '0' && ch <= '9') || ch == '_' || ch == '-') {
      clean += ch;
    } else {
      clean += '_';
    }
  }
  return clean;
}

}  // namespace tensorflow
