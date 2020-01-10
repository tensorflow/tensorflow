#ifndef TENSORFLOW_UTIL_PADDING_H_
#define TENSORFLOW_UTIL_PADDING_H_

// This file contains helper routines to deal with padding in various ops and
// kernels.

#include <string>

#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/public/status.h"

namespace tensorflow {

// Padding: the padding we apply to the input tensor along the rows and columns
// dimensions. This is usually used to make sure that the spatial dimensions do
// not shrink when we progress with convolutions. Two types of padding are
// supported:
//   VALID: No padding is carried out.
//   SAME: The pad value is computed so that the output will have the same
//         dimensions as the input.
// The padded area is zero-filled.
enum Padding {
  VALID = 1,  // No padding.
  SAME = 2,   // Input and output layers have the same size.
};

// Return the string containing the list of valid padding types, that can be
// used as an Attr() in REGISTER_OP.
string GetPaddingAttrString();

// Specialization to parse an attribute directly into a Padding enum.
Status GetNodeAttr(const NodeDef& node_def, const string& attr_name,
                   Padding* value);

}  // end namespace tensorflow

#endif  // TENSORFLOW_UTIL_PADDING_H_
