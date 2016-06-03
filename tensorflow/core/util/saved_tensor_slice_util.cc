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

#include "tensorflow/core/util/saved_tensor_slice_util.h"

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/strings/ordered_code.h"

namespace tensorflow {

namespace checkpoint {

const char kSavedTensorSlicesKey[] = "";

string EncodeTensorNameSlice(const string& name, const TensorSlice& slice) {
  string buffer;
  // All the tensor slice keys will start with a 0
  tensorflow::strings::OrderedCode::WriteNumIncreasing(&buffer, 0);
  tensorflow::strings::OrderedCode::WriteString(&buffer, name);
  tensorflow::strings::OrderedCode::WriteNumIncreasing(&buffer, slice.dims());
  for (int d = 0; d < slice.dims(); ++d) {
    // A trivial extent (meaning we take EVERYTHING) will default to -1 for both
    // start and end. These will be properly parsed.
    tensorflow::strings::OrderedCode::WriteSignedNumIncreasing(&buffer,
                                                               slice.start(d));
    tensorflow::strings::OrderedCode::WriteSignedNumIncreasing(&buffer,
                                                               slice.length(d));
  }
  return buffer;
}

Status DecodeTensorNameSlice(const string& code, string* name,
                             tensorflow::TensorSlice* slice) {
  StringPiece src(code);
  uint64 x;
  if (!tensorflow::strings::OrderedCode::ReadNumIncreasing(&src, &x)) {
    return errors::Internal("Failed to parse the leading number: src = ", src);
  }
  if (x != 0) {
    return errors::Internal(
        "The leading number should always be 0 for any valid key: src = ", src);
  }
  if (!tensorflow::strings::OrderedCode::ReadString(&src, name)) {
    return errors::Internal("Failed to parse the tensor name: src = ", src);
  }
  if (!tensorflow::strings::OrderedCode::ReadNumIncreasing(&src, &x)) {
    return errors::Internal("Failed to parse the tensor rank: src = ", src);
  }
  if (x == 0) {
    return errors::Internal("Expecting positive rank of the tensor, got ", x,
                            ", src = ", src);
  }
  if (x >= kint32max) {
    return errors::Internal("Too many elements ", x);
  }
  slice->SetFullSlice(x);
  for (int d = 0; d < static_cast<int32>(x); ++d) {
    // We expected 2x integers
    int64 start, length;
    if (!tensorflow::strings::OrderedCode::ReadSignedNumIncreasing(&src,
                                                                   &start)) {
      return errors::Internal("Failed to parse start: src = ", src);
    }
    if (!tensorflow::strings::OrderedCode::ReadSignedNumIncreasing(&src,
                                                                   &length)) {
      return errors::Internal("Failed to parse length: src = ", src);
    }
    if (length >= 0) {
      // a non-trivial extent
      slice->set_start(d, start);
      slice->set_length(d, length);
    }
  }
  return Status::OK();
}

}  // namespace checkpoint

}  // namespace tensorflow
