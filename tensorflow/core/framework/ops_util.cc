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

#include <algorithm>
#include <cmath>

#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/ops_util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/util/padding.h"

namespace tensorflow {

Eigen::PaddingType BrainPadding2EigenPadding(Padding padding) {
  switch (padding) {
    case Padding::VALID:
      return Eigen::PADDING_VALID;
    case Padding::SAME:
      return Eigen::PADDING_SAME;
    case Padding::EXPLICIT:
      LOG(FATAL) << "Eigen does not have explicit padding enum "  // Crash OK
                    "value";
  }
  return Eigen::PADDING_SAME;  // Prevent compiler warning about missing return
}

Status GetBroadcastSize(const int index, const int in_size, const int ksize,
                        const int stride, const int pad_size, int* bindex,
                        int* bsize) {
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
  return OkStatus();
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
