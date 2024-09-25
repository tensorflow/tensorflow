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

// This file declares the functions and structures for memory I/O with libjpeg
// These functions are not meant to be used directly, see jpeg_mem.h instead.

#ifndef TENSORFLOW_CORE_LIB_JPEG_JPEG_HANDLE_H_
#define TENSORFLOW_CORE_LIB_JPEG_JPEG_HANDLE_H_

#include "tensorflow/core/platform/jpeg.h"
#include "tensorflow/core/platform/tstring.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace jpeg {

// Handler for fatal JPEG library errors: clean up & return
void CatchError(j_common_ptr cinfo);

typedef struct {
  struct jpeg_destination_mgr pub;
  JOCTET *buffer;
  int bufsize;
  int datacount;
  tstring *dest;
} MemDestMgr;

typedef struct {
  struct jpeg_source_mgr pub;
  const unsigned char *data;
  unsigned long int datasize;
  bool try_recover_truncated_jpeg;
} MemSourceMgr;

void SetSrc(j_decompress_ptr cinfo, const void *data,
            unsigned long int datasize, bool try_recover_truncated_jpeg);

// JPEG destination: we will store all the data in a buffer "buffer" of total
// size "bufsize", if the buffer overflows, we will be in trouble.
void SetDest(j_compress_ptr cinfo, void *buffer, int bufsize);
// Same as above, except that buffer is only used as a temporary structure and
// is emptied into "destination" as soon as it fills up.
void SetDest(j_compress_ptr cinfo, void *buffer, int bufsize,
             tstring *destination);

}  // namespace jpeg
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_LIB_JPEG_JPEG_HANDLE_H_
