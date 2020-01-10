// Functions to read and write images in PNG format.

#include <string.h>
#include <sys/types.h>
#include <string>
#include <utility>
#include <vector>
// NOTE(skal): we don't '#include <setjmp.h>' before png/png.h as it otherwise
// provokes a compile error. We instead let png.h include what is needed.

#include "external/png_archive/libpng-1.2.53/png.h"
#include "tensorflow/core/lib/core/casts.h"
#include "tensorflow/core/lib/png/png_io.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/port.h"  // endian

namespace tensorflow {
namespace png {

////////////////////////////////////////////////////////////////////////////////
// Encode an 8- or 16-bit rgb/grayscale image to PNG string
////////////////////////////////////////////////////////////////////////////////

namespace {

#define PTR_INC(type, ptr, del) \
  (ptr = reinterpret_cast<type*>(reinterpret_cast<char*>(ptr) + (del)))
#define CPTR_INC(type, ptr, del)                                            \
  (ptr = reinterpret_cast<const type*>(reinterpret_cast<const char*>(ptr) + \
                                       (del)))

// Convert from 8 bit components to 16. This works in-place.
static void Convert8to16(const uint8* p8, int num_comps, int p8_row_bytes,
                         int width, int height, uint16* p16,
                         int p16_row_bytes) {
  // Adjust pointers to copy backwards
  width *= num_comps;
  CPTR_INC(uint8, p8, (height - 1) * p8_row_bytes + (width - 1) * sizeof(*p8));
  PTR_INC(uint16, p16,
          (height - 1) * p16_row_bytes + (width - 1) * sizeof(*p16));
  int bump8 = width * sizeof(*p8) - p8_row_bytes;
  int bump16 = width * sizeof(*p16) - p16_row_bytes;
  for (; height-- != 0;
       CPTR_INC(uint8, p8, bump8), PTR_INC(uint16, p16, bump16)) {
    for (int w = width; w-- != 0; --p8, --p16) {
      uint pix = *p8;
      pix |= pix << 8;
      *p16 = static_cast<uint16>(pix);
    }
  }
}

#undef PTR_INC
#undef CPTR_INC

void ErrorHandler(png_structp png_ptr, png_const_charp msg) {
  DecodeContext* const ctx = bit_cast<DecodeContext*>(png_get_io_ptr(png_ptr));
  ctx->error_condition = true;
  // To prevent log spam, errors are logged as VLOG(1) instead of ERROR.
  VLOG(1) << "PNG error: " << msg;
  longjmp(png_jmpbuf(png_ptr), 1);
}

void WarningHandler(png_structp png_ptr, png_const_charp msg) {
  LOG(WARNING) << "PNG warning: " << msg;
}

void StringReader(png_structp png_ptr, png_bytep data, png_size_t length) {
  DecodeContext* const ctx = bit_cast<DecodeContext*>(png_get_io_ptr(png_ptr));
  if (static_cast<png_size_t>(ctx->data_left) < length) {
    if (!ctx->error_condition) {
      VLOG(1) << "PNG read decoding error";
      ctx->error_condition = true;
    }
    memset(data, 0, length);
  } else {
    memcpy(data, ctx->data, length);
    ctx->data += length;
    ctx->data_left -= length;
  }
}

void StringWriter(png_structp png_ptr, png_bytep data, png_size_t length) {
  string* const s = bit_cast<string*>(png_get_io_ptr(png_ptr));
  s->append(bit_cast<const char*>(data), length);
}

void StringWriterFlush(png_structp png_ptr) {}

char* check_metadata_string(const string& s) {
  const char* const c_string = s.c_str();
  const size_t length = s.size();
  if (strlen(c_string) != length) {
    LOG(WARNING) << "Warning! Metadata contains \\0 character(s).";
  }
  return const_cast<char*>(c_string);
}

}  // namespace

// We move CommonInitDecode() and CommonFinishDecode()
// out of the CommonDecode() template to save code space.
void CommonFreeDecode(DecodeContext* context) {
  if (context->png_ptr) {
    png_destroy_read_struct(&context->png_ptr,
                            context->info_ptr ? &context->info_ptr : NULL, 0);
    context->png_ptr = nullptr;
    context->info_ptr = nullptr;
  }
}

bool DecodeHeader(StringPiece png_string, int* width, int* height,
                  int* components, int* channel_bit_depth,
                  std::vector<std::pair<string, string> >* metadata) {
  DecodeContext context;
  // Ask for 16 bits even if there may be fewer.  This assures that sniffing
  // the metadata will succeed in all cases.
  //
  // TODO(skal): CommonInitDecode() mixes the operation of sniffing the
  // metadata with setting up the data conversions.  These should be separated.
  constexpr int kDesiredNumChannels = 1;
  constexpr int kDesiredChannelBits = 16;
  if (!CommonInitDecode(png_string, kDesiredNumChannels, kDesiredChannelBits,
                        &context)) {
    return false;
  }
  CHECK_NOTNULL(width);
  *width = static_cast<int>(context.width);
  CHECK_NOTNULL(height);
  *height = static_cast<int>(context.height);
  if (components != NULL) {
    switch (context.color_type) {
      case PNG_COLOR_TYPE_PALETTE:
        *components = (context.info_ptr->valid & PNG_INFO_tRNS) ? 4 : 3;
        break;
      case PNG_COLOR_TYPE_GRAY:
        *components = 1;
        break;
      case PNG_COLOR_TYPE_GRAY_ALPHA:
        *components = 2;
        break;
      case PNG_COLOR_TYPE_RGB:
        *components = 3;
        break;
      case PNG_COLOR_TYPE_RGB_ALPHA:
        *components = 4;
        break;
      default:
        *components = 0;
        break;
    }
  }
  if (channel_bit_depth != NULL) {
    *channel_bit_depth = context.bit_depth;
  }
  if (metadata != NULL) {
    metadata->clear();
    for (int i = 0; i < context.info_ptr->num_text; i++) {
      const png_text& text = context.info_ptr->text[i];
      metadata->push_back(std::make_pair(text.key, text.text));
    }
  }
  CommonFreeDecode(&context);
  return true;
}

bool CommonInitDecode(StringPiece png_string, int desired_channels,
                      int desired_channel_bits, DecodeContext* context) {
  CHECK(desired_channel_bits == 8 || desired_channel_bits == 16)
      << "desired_channel_bits = " << desired_channel_bits;
  CHECK(0 <= desired_channels && desired_channels <= 4) << "desired_channels = "
                                                        << desired_channels;
  context->error_condition = false;
  context->channels = desired_channels;
  context->png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, context,
                                            ErrorHandler, WarningHandler);
  if (!context->png_ptr) {
    VLOG(1) << ": DecodePNG <- png_create_read_struct failed";
    return false;
  }
  if (setjmp(png_jmpbuf(context->png_ptr))) {
    VLOG(1) << ": DecodePNG error trapped.";
    CommonFreeDecode(context);
    return false;
  }
  context->info_ptr = png_create_info_struct(context->png_ptr);
  if (!context->info_ptr || context->error_condition) {
    VLOG(1) << ": DecodePNG <- png_create_info_struct failed";
    CommonFreeDecode(context);
    return false;
  }
  context->data = bit_cast<const uint8*>(png_string.data());
  context->data_left = png_string.size();
  png_set_read_fn(context->png_ptr, context, StringReader);
  png_read_info(context->png_ptr, context->info_ptr);
  png_get_IHDR(context->png_ptr, context->info_ptr, &context->width,
               &context->height, &context->bit_depth, &context->color_type, 0,
               0, 0);
  if (context->error_condition) {
    VLOG(1) << ": DecodePNG <- error during header parsing.";
    CommonFreeDecode(context);
    return false;
  }
  if (context->width <= 0 || context->height <= 0) {
    VLOG(1) << ": DecodePNG <- invalid dimensions";
    CommonFreeDecode(context);
    return false;
  }
  if (context->channels == 0) {  // Autodetect number of channels
    context->channels = context->info_ptr->channels;
  }
  const bool has_tRNS = (context->info_ptr->valid & PNG_INFO_tRNS) != 0;
  const bool has_alpha = (context->color_type & PNG_COLOR_MASK_ALPHA) != 0;
  if ((context->channels & 1) == 0) {  // We desire alpha
    if (has_alpha) {                   // There is alpha
    } else if (has_tRNS) {
      png_set_tRNS_to_alpha(context->png_ptr);  // Convert transparency to alpha
    } else {
      png_set_add_alpha(context->png_ptr, (1 << context->bit_depth) - 1,
                        PNG_FILLER_AFTER);
    }
  } else {                                    // We don't want alpha
    if (has_alpha || has_tRNS) {              // There is alpha
      png_set_strip_alpha(context->png_ptr);  // Strip alpha
    }
  }

  // If we only want 8 bits, but are given 16, strip off the LS 8 bits
  if (context->bit_depth > 8 && desired_channel_bits <= 8)
    png_set_strip_16(context->png_ptr);

  context->need_to_synthesize_16 =
      (context->bit_depth <= 8 && desired_channel_bits == 16);

  png_set_packing(context->png_ptr);
  context->num_passes = png_set_interlace_handling(context->png_ptr);
  png_read_update_info(context->png_ptr, context->info_ptr);

#ifdef IS_LITTLE_ENDIAN
  if (desired_channel_bits > 8) png_set_swap(context->png_ptr);
#endif  // IS_LITTLE_ENDIAN

  // convert palette to rgb(a) if needs be.
  if (context->color_type == PNG_COLOR_TYPE_PALETTE)
    png_set_palette_to_rgb(context->png_ptr);

  // handle grayscale case for source or destination
  const bool want_gray = (context->channels < 3);
  const bool is_gray = !(context->color_type & PNG_COLOR_MASK_COLOR);
  if (is_gray) {  // upconvert gray to 8-bit if needed.
    if (context->bit_depth < 8) png_set_gray_1_2_4_to_8(context->png_ptr);
  }
  if (want_gray) {  // output is grayscale
    if (!is_gray)
      png_set_rgb_to_gray(context->png_ptr, 1, 0.299, 0.587);  // 601, JPG
  } else {  // output is rgb(a)
    if (is_gray)
      png_set_gray_to_rgb(context->png_ptr);  // Enable gray -> RGB conversion
  }
  return true;
}

bool CommonFinishDecode(png_bytep data, int row_bytes, DecodeContext* context) {
  CHECK_NOTNULL(data);

  // we need to re-set the jump point so that we trap the errors
  // within *this* function (and not CommonInitDecode())
  if (setjmp(png_jmpbuf(context->png_ptr))) {
    VLOG(1) << ": DecodePNG error trapped.";
    CommonFreeDecode(context);
    return false;
  }
  // png_read_row() takes care of offsetting the pointer based on interlacing
  for (int p = 0; p < context->num_passes; ++p) {
    png_bytep row = data;
    for (int h = context->height; h-- != 0; row += row_bytes) {
      png_read_row(context->png_ptr, row, NULL);
    }
  }

  context->info_ptr->valid |= PNG_INFO_IDAT;
  png_read_end(context->png_ptr, context->info_ptr);

  // Clean up.
  const bool ok = !context->error_condition;
  CommonFreeDecode(context);

  // Synthesize 16 bits from 8 if requested.
  if (context->need_to_synthesize_16)
    Convert8to16(bit_cast<uint8*>(data), context->channels, row_bytes,
                 context->width, context->height, bit_cast<uint16*>(data),
                 row_bytes);
  return ok;
}

bool WriteImageToBuffer(
    const void* image, int width, int height, int row_bytes, int num_channels,
    int channel_bits, int compression, string* png_string,
    const std::vector<std::pair<string, string> >* metadata) {
  CHECK_NOTNULL(image);
  CHECK_NOTNULL(png_string);
  // Although this case is checked inside png.cc and issues an error message,
  // that error causes memory corruption.
  if (width == 0 || height == 0) return false;

  png_string->resize(0);
  png_infop info_ptr = NULL;
  png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL,
                                                ErrorHandler, WarningHandler);
  if (png_ptr == NULL) return false;
  if (setjmp(png_jmpbuf(png_ptr))) {
    png_destroy_write_struct(&png_ptr, info_ptr ? &info_ptr : NULL);
    return false;
  }
  info_ptr = png_create_info_struct(png_ptr);
  if (info_ptr == NULL) {
    png_destroy_write_struct(&png_ptr, NULL);
    return false;
  }

  int color_type = -1;
  switch (num_channels) {
    case 1:
      color_type = PNG_COLOR_TYPE_GRAY;
      break;
    case 2:
      color_type = PNG_COLOR_TYPE_GRAY_ALPHA;
      break;
    case 3:
      color_type = PNG_COLOR_TYPE_RGB;
      break;
    case 4:
      color_type = PNG_COLOR_TYPE_RGB_ALPHA;
      break;
    default:
      png_destroy_write_struct(&png_ptr, &info_ptr);
      return false;
  }

  png_set_write_fn(png_ptr, png_string, StringWriter, StringWriterFlush);
  if (compression < 0) compression = Z_DEFAULT_COMPRESSION;
  png_set_compression_level(png_ptr, compression);
  png_set_compression_mem_level(png_ptr, MAX_MEM_LEVEL);
  // There used to be a call to png_set_filter here turning off filtering
  // entirely, but it produced pessimal compression ratios.  I'm not sure
  // why it was there.
  png_set_IHDR(png_ptr, info_ptr, width, height, channel_bits, color_type,
               PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT,
               PNG_FILTER_TYPE_DEFAULT);
  // If we have metadata write to it.
  if (metadata && !metadata->empty()) {
    std::vector<png_text> text;
    for (const auto& pair : *metadata) {
      png_text txt;
      txt.compression = PNG_TEXT_COMPRESSION_NONE;
      txt.key = check_metadata_string(pair.first);
      txt.text = check_metadata_string(pair.second);
      text.push_back(txt);
    }
    png_set_text(png_ptr, info_ptr, &text[0], text.size());
  }

  png_write_info(png_ptr, info_ptr);
#ifdef IS_LITTLE_ENDIAN
  if (channel_bits > 8) png_set_swap(png_ptr);
#endif  // IS_LITTLE_ENDIAN

  png_byte* row = reinterpret_cast<png_byte*>(const_cast<void*>(image));
  for (; height--; row += row_bytes) png_write_row(png_ptr, row);
  png_write_end(png_ptr, NULL);

  png_destroy_write_struct(&png_ptr, &info_ptr);
  return true;
}

}  // namespace png
}  // namespace tensorflow
