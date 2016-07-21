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

// Functions to read and write images in GIF format.
//
// The advantage over image/codec/png{enc,dec}ocder.h is that this library
// supports both 8 and 16 bit images.
//
// The decoding routine accepts binary image data as a StringPiece.  These are
// implicitly constructed from strings or char* so they're completely
// transparent to the caller.  They're also very cheap to construct so this
// doesn't introduce any additional overhead.
//
// The primary benefit of StringPieces being, in this case, that APIs already
// returning StringPieces (e.g., Bigtable Scanner) or Cords (e.g., IOBuffer;
// only when they're flat, though) or protocol buffer fields typed to either of
// these can be decoded without copying the data into a C++ string.

#ifndef TENSORFLOW_CORE_LIB_GIF_GIF_IO_H_
#define TENSORFLOW_CORE_LIB_GIF_GIF_IO_H_

#include <functional>
#include <string>
#include <utility>
#include <vector>

#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace gif {

uint8* Decode(const void* srcdata, int datasize,
              std::function<uint8*(int, int, int, int)> allocate_output);

}  // namespace gif
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_LIB_GIF_GIF_IO_H_
