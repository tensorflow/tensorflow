// A class to manage slices of a tensor. You can "register" set of slices for a
// tensor and then "query" if we have data for a given slice.

// TODO(yangke): consider moving it to a more private place so that we don't
// need to expose the API.

#ifndef TENSORFLOW_UTIL_TENSOR_SLICE_SET_H_
#define TENSORFLOW_UTIL_TENSOR_SLICE_SET_H_

#include <string>  // for string
#include <unordered_map>

#include "tensorflow/core/framework/tensor_slice.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/stringpiece.h"  // for StringPiece
#include "tensorflow/core/platform/port.h"         // for int64
#include "tensorflow/core/public/status.h"         // for Status
#include "tensorflow/core/public/tensor_shape.h"

namespace tensorflow {

namespace checkpoint {

class TensorSliceSet {
 public:
  TensorSliceSet(const TensorShape& shape, DataType type);
  virtual ~TensorSliceSet();

  const TensorShape& shape() const { return shape_; }
  const DataType type() const { return type_; }

  // Register a new slice for the tensor. The "tag" is an arbitrary string
  // associated with the slice (in one application it denotes the name of the
  // file that contains the slice); the "data" points to the data of the tensor
  // slice (it can be a nullptr).
  // We don't take the ownership of "data" and the caller needs to make sure
  // the data is always available during the life time of the tensor slice set
  // if it is not nullptr.
  Status Register(const TensorSlice& slice, const string& tag,
                  const float* data);

  // Query about a new slice: checks if we have data for "slice" and if we have
  // the data and "data" is not nullptr, fill "data" with the slice data. The
  // caller needs to make sure "data" point to a large eough buffer.
  // TODO(yangke): avoid unnecessary copying by using a core::RefCounted
  // pointer.
  bool Query(const TensorSlice& slice, float* data) const;

  // Alternative way of querying about a new slice: instead of copying the
  // data, it returns a list of meta data about the stored slices that will
  // supply data for the slice.
  bool QueryMeta(
      const TensorSlice& slice,
      std::vector<std::pair<tensorflow::TensorSlice, string>>* results) const;

 private:
  const TensorShape shape_;
  const DataType type_;
  struct SliceInfo {
    TensorSlice slice;
    const string tag;
    const float* data;
    int64 num_floats;
  };
  // We maintain a mapping from the slice string to the slice information.
  std::unordered_map<string, SliceInfo> slices_;
};

}  // namespace checkpoint

}  // namespace tensorflow

#endif  // TENSORFLOW_UTIL_TENSOR_SLICE_SET_H_
