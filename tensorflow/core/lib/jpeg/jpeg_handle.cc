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

// This file implements a memory destination for libjpeg
// The design is very similar to jdatadst.c in libjpeg
// These functions are not meant to be used directly, see jpeg_mem.h instead.
// We are filling out stubs required by jpeglib, those stubs are private to
// the implementation, we are just making available JPGMemSrc, JPGMemDest

#include "tensorflow/core/lib/jpeg/jpeg_handle.h"

#include <setjmp.h>
#include <stddef.h>

#include "tensorflow/core/platform/logging.h"

namespace tensorflow {
namespace jpeg {

void CatchError(j_common_ptr cinfo) {
  (*cinfo->err->output_message)(cinfo);
  jmp_buf *jpeg_jmpbuf = reinterpret_cast<jmp_buf *>(cinfo->client_data);
  jpeg_destroy(cinfo);
  longjmp(*jpeg_jmpbuf, 1);
}

// *****************************************************************************
// *****************************************************************************
// *****************************************************************************
// Destination functions

// -----------------------------------------------------------------------------
void MemInitDestination(j_compress_ptr cinfo) {
  MemDestMgr *dest = reinterpret_cast<MemDestMgr *>(cinfo->dest);
  VLOG(1) << "Initializing buffer=" << dest->bufsize << " bytes";
  dest->pub.next_output_byte = dest->buffer;
  dest->pub.free_in_buffer = dest->bufsize;
  dest->datacount = 0;
  if (dest->dest) {
    dest->dest->clear();
  }
}

// -----------------------------------------------------------------------------
boolean MemEmptyOutputBuffer(j_compress_ptr cinfo) {
  MemDestMgr *dest = reinterpret_cast<MemDestMgr *>(cinfo->dest);
  VLOG(1) << "Writing " << dest->bufsize << " bytes";
  if (dest->dest) {
    dest->dest->append(reinterpret_cast<char *>(dest->buffer), dest->bufsize);
  }
  dest->pub.next_output_byte = dest->buffer;
  dest->pub.free_in_buffer = dest->bufsize;
  return TRUE;
}

// -----------------------------------------------------------------------------
void MemTermDestination(j_compress_ptr cinfo) {
  MemDestMgr *dest = reinterpret_cast<MemDestMgr *>(cinfo->dest);
  VLOG(1) << "Writing " << dest->bufsize - dest->pub.free_in_buffer << " bytes";
  if (dest->dest) {
    dest->dest->append(reinterpret_cast<char *>(dest->buffer),
                       dest->bufsize - dest->pub.free_in_buffer);
    VLOG(1) << "Total size= " << dest->dest->size();
  }
  dest->datacount = dest->bufsize - dest->pub.free_in_buffer;
}

// -----------------------------------------------------------------------------
void SetDest(j_compress_ptr cinfo, void *buffer, int bufsize) {
  SetDest(cinfo, buffer, bufsize, nullptr);
}

// -----------------------------------------------------------------------------
void SetDest(j_compress_ptr cinfo, void *buffer, int bufsize,
             tstring *destination) {
  MemDestMgr *dest;
  if (cinfo->dest == nullptr) {
    cinfo->dest = reinterpret_cast<struct jpeg_destination_mgr *>(
        (*cinfo->mem->alloc_small)(reinterpret_cast<j_common_ptr>(cinfo),
                                   JPOOL_PERMANENT, sizeof(MemDestMgr)));
  }

  dest = reinterpret_cast<MemDestMgr *>(cinfo->dest);
  dest->bufsize = bufsize;
  dest->buffer = static_cast<JOCTET *>(buffer);
  dest->dest = destination;
  dest->pub.init_destination = MemInitDestination;
  dest->pub.empty_output_buffer = MemEmptyOutputBuffer;
  dest->pub.term_destination = MemTermDestination;
}

// *****************************************************************************
// *****************************************************************************
// *****************************************************************************
// Source functions

// -----------------------------------------------------------------------------
void MemInitSource(j_decompress_ptr cinfo) {
  MemSourceMgr *src = reinterpret_cast<MemSourceMgr *>(cinfo->src);
  src->pub.next_input_byte = src->data;
  src->pub.bytes_in_buffer = src->datasize;
}

// -----------------------------------------------------------------------------
// We emulate the same error-handling as fill_input_buffer() from jdatasrc.c,
// for coherency's sake.
boolean MemFillInputBuffer(j_decompress_ptr cinfo) {
  static const JOCTET kEOIBuffer[2] = {0xff, JPEG_EOI};
  MemSourceMgr *src = reinterpret_cast<MemSourceMgr *>(cinfo->src);
  if (src->pub.bytes_in_buffer == 0 && src->pub.next_input_byte == src->data) {
    // empty file -> treated as an error.
    ERREXIT(cinfo, JERR_INPUT_EMPTY);
    return FALSE;
  } else if (src->pub.bytes_in_buffer) {
    // if there's still some data left, it's probably corrupted
    return src->try_recover_truncated_jpeg ? TRUE : FALSE;
  } else if (src->pub.next_input_byte != kEOIBuffer &&
             src->try_recover_truncated_jpeg) {
    // In an attempt to recover truncated files, we insert a fake EOI
    WARNMS(cinfo, JWRN_JPEG_EOF);
    src->pub.next_input_byte = kEOIBuffer;
    src->pub.bytes_in_buffer = 2;
    return TRUE;
  } else {
    // We already inserted a fake EOI and it wasn't enough, so this time
    // it's really an error.
    ERREXIT(cinfo, JERR_FILE_READ);
    return FALSE;
  }
}

// -----------------------------------------------------------------------------
void MemTermSource(j_decompress_ptr cinfo) {}

// -----------------------------------------------------------------------------
void MemSkipInputData(j_decompress_ptr cinfo, long jump) {
  MemSourceMgr *src = reinterpret_cast<MemSourceMgr *>(cinfo->src);
  if (jump < 0) {
    return;
  }
  if (jump > src->pub.bytes_in_buffer) {
    src->pub.bytes_in_buffer = 0;
    (void)MemFillInputBuffer(cinfo);  // warn with a fake EOI or error
  } else {
    src->pub.bytes_in_buffer -= jump;
    src->pub.next_input_byte += jump;
  }
}

// -----------------------------------------------------------------------------
void SetSrc(j_decompress_ptr cinfo, const void *data,
            unsigned long int datasize, bool try_recover_truncated_jpeg) {
  MemSourceMgr *src;

  cinfo->src = reinterpret_cast<struct jpeg_source_mgr *>(
      (*cinfo->mem->alloc_small)(reinterpret_cast<j_common_ptr>(cinfo),
                                 JPOOL_PERMANENT, sizeof(MemSourceMgr)));

  src = reinterpret_cast<MemSourceMgr *>(cinfo->src);
  src->pub.init_source = MemInitSource;
  src->pub.fill_input_buffer = MemFillInputBuffer;
  src->pub.skip_input_data = MemSkipInputData;
  src->pub.resync_to_restart = jpeg_resync_to_restart;
  src->pub.term_source = MemTermSource;
  src->data = reinterpret_cast<const unsigned char *>(data);
  src->datasize = datasize;
  src->pub.bytes_in_buffer = 0;
  src->pub.next_input_byte = nullptr;
  src->try_recover_truncated_jpeg = try_recover_truncated_jpeg;
}

}  // namespace jpeg
}  // namespace tensorflow
