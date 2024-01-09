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

// A class to manage slices of a tensor. You can "register" set of slices for a
// tensor and then "query" if we have data for a given slice.

#ifndef TENSORFLOW_CORE_UTIL_TENSOR_SLICE_SET_H_
#define TENSORFLOW_CORE_UTIL_TENSOR_SLICE_SET_H_

#include <string>  // for string
#include <unordered_map>
#include <utility>
#include <vector>

#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_slice.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"       // for Status
#include "tensorflow/core/lib/core/stringpiece.h"  // for StringPiece
#include "tensorflow/core/platform/types.h"        // for int64

namespace tensorflow {

namespace checkpoint {

class TensorSliceSet {
 public:
  TensorSliceSet(const TensorShape& shape, DataType type);
  virtual ~TensorSliceSet();

  const TensorShape& shape() const { return shape_; }
  DataType type() const { return type_; }

  // Register a new slice for the tensor. The "tag" is an arbitrary string
  // associated with the slice (in one application it denotes the name of the
  // file that contains the slice); the "data" points to the data of the tensor
  // slice (it can be a nullptr).
  Status Register(const TensorSlice& slice, const string& tag);

  // Alternative way of querying about a new slice: instead of copying the
  // data, it returns a list of meta data about the stored slices that will
  // supply data for the slice.
  bool QueryMeta(
      const TensorSlice& slice,
      std::vector<std::pair<tensorflow::TensorSlice, string>>* results) const;

  struct SliceInfo {
    TensorSlice slice;
    const string tag;
    int64_t num_floats;
  };

  // Returns the map from slice string to SliceInfo.
  const std::unordered_map<string, SliceInfo>& Slices() const {
    return slices_;
  }

 private:
  const TensorShape shape_;
  const DataType type_;
  // We maintain a mapping from the slice string to the slice information.
  std::unordered_map<string, SliceInfo> slices_;

  // Minimal slice which contains all presented slices. Used for speeding up
  // overlap check when slices are being added consequently.
  TensorSlice slices_hull_;
};

// Registers "slice" in the TensorSliceSet stored in "tensor_slices", under key
// "name".  Other arguments are used for validations.  Does not modify the map
// or its values on non-OK.
// REQUIRES: tensor_slices != nullptr
Status RegisterTensorSlice(
    const string& name, const TensorShape& shape, DataType type,
    const string& tag, const TensorSlice& slice,
    std::unordered_map<string, TensorSliceSet*>* tensor_slices);

}  // namespace checkpoint

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_UTIL_TENSOR_SLICE_SET_H_
