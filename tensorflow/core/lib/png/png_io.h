// Functions to read and write images in PNG format.
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

#ifndef TENSORFLOW_LIB_PNG_PNG_IO_H_
#define TENSORFLOW_LIB_PNG_PNG_IO_H_

#include <string>
#include <utility>
#include <vector>

#include "tensorflow/core/lib/core/stringpiece.h"
#include "external/png_archive/libpng-1.2.53/png.h"

namespace tensorflow {
namespace png {

// Handy container for decoding informations and struct pointers
struct DecodeContext {
  const uint8* data;
  int         data_left;
  png_structp png_ptr;
  png_infop   info_ptr;
  png_uint_32 width, height;
  int         num_passes;
  int         color_type;
  int         bit_depth;
  int channels;
  bool need_to_synthesize_16;
  bool error_condition;
  DecodeContext() : png_ptr(NULL), info_ptr(NULL) {}
};

bool DecodeHeader(StringPiece png_string, int* width, int* height,
                  int* components, int* channel_bit_depth,
                  std::vector<std::pair<string, string> >* metadata);

// Sample usage for reading PNG:
//
// string png_string;  /* fill with input PNG format data */
// DecodeContext context;
// CHECK(CommonInitDecode(png_string, 3 /*RGB*/, 8 /*uint8*/, &context));
// char* image_buffer = new char[3*context.width*context.height];
// CHECK(CommonFinishDecode(bit_cast<png_byte*>(image_buffer),
//       3*context.width /*stride*/, &context));
//
// desired_channels may be 0 to detected it from the input.

bool CommonInitDecode(StringPiece png_string, int desired_channels,
                      int desired_channel_bits, DecodeContext* context);

bool CommonFinishDecode(png_bytep data, int row_bytes, DecodeContext* context);

// Normally called automatically from CommonFinishDecode.  If CommonInitDecode
// is called but not CommonFinishDecode, call this to clean up.  Safe to call
// extra times.
void CommonFreeDecode(DecodeContext* context);

// Sample usage for writing PNG:
//
// uint16* image_buffer = new uint16[width*height];  /* fill with pixels */
// string png_string;
// CHECK(WriteImageToBuffer(image_buffer, width, height, 2*width /*stride*/,
//       1 /*gray*/, 16 /*uint16*/, &png_string, NULL));
//
// compression is in [-1,9], where 0 is fast and weak compression, 9 is slow
// and strong, and -1 is the zlib default.

bool WriteImageToBuffer(
    const void* image, int width, int height, int row_bytes, int num_channels,
    int channel_bits, int compression, string* png_string,
    const std::vector<std::pair<string, string> >* metadata);

}  // namespace png
}  // namespace tensorflow

#endif  // TENSORFLOW_LIB_PNG_PNG_IO_H_
