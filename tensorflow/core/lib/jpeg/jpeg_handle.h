// This file declares the functions and structures for memory I/O with libjpeg
// These functions are not meant to be used directly, see jpeg_mem.h isntead.

#ifndef TENSORFLOW_LIB_JPEG_JPEG_HANDLE_H_
#define TENSORFLOW_LIB_JPEG_JPEG_HANDLE_H_

extern "C" {
#include "external/jpeg_archive/jpeg-9a/jinclude.h"
#include "external/jpeg_archive/jpeg-9a/jpeglib.h"
#include "external/jpeg_archive/jpeg-9a/jerror.h"
#include "external/jpeg_archive/jpeg-9a/transupp.h"  // for rotations
}

#include "tensorflow/core/platform/port.h"

namespace tensorflow {
namespace jpeg {

// Handler for fatal JPEG library errors: clean up & return
void CatchError(j_common_ptr cinfo);

typedef struct {
  struct jpeg_destination_mgr pub;
  JOCTET *buffer;
  int bufsize;
  int datacount;
  string *dest;
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
             string *destination);

}  // namespace jpeg
}  // namespace tensorflow

#endif  // TENSORFLOW_LIB_JPEG_JPEG_HANDLE_H_
