#ifndef TENSORFLOW_GRAPH_TENSOR_ID_H_
#define TENSORFLOW_GRAPH_TENSOR_ID_H_

#include <string>

#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/strings/strcat.h"

namespace tensorflow {

// Identifier for a tensor within a step.
// first == operation_name, second == output_index
// Note: does not own backing storage for name.
struct TensorId : public std::pair<StringPiece, int> {
  typedef std::pair<StringPiece, int> Base;

  // Inherit the set of constructors.
  using Base::pair;

  string ToString() const { return strings::StrCat(first, ":", second); }
};

TensorId ParseTensorName(const string& name);
TensorId ParseTensorName(StringPiece name);

}  // namespace tensorflow

#endif  // TENSORFLOW_GRAPH_TENSOR_ID_H_
