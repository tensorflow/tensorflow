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

#ifndef TENSORFLOW_UTIL_PADDING_H_
#define TENSORFLOW_UTIL_PADDING_H_

// This file contains helper routines to deal with padding in various ops and
// kernels.

#include <string>

#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/lib/core/status.h"

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
Status GetNodeAttr(const NodeDef& node_def, StringPiece attr_name,
                   Padding* value);

}  // end namespace tensorflow

#endif  // TENSORFLOW_UTIL_PADDING_H_
