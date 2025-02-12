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

// This file defines functions to compress and uncompress JPEG data
// to and from memory, as well as some direct manipulations of JPEG string

#include "tensorflow/core/lib/jpeg/jpeg_mem.h"

#include <setjmp.h>
#include <string.h>

#include <algorithm>
#include <functional>
#include <memory>
#include <ostream>
#include <string>
#include <utility>

#include "jpeglib.h"  // from @libjpeg_turbo
#include "tensorflow/core/lib/jpeg/jpeg_handle.h"
#include "tensorflow/core/platform/dynamic_annotations.h"
#include "tensorflow/core/platform/jpeg.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/tstring.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace jpeg {

// -----------------------------------------------------------------------------
// Decompression

namespace {

enum JPEGErrors {
  JPEGERRORS_OK,
  JPEGERRORS_UNEXPECTED_END_OF_DATA,
  JPEGERRORS_BAD_PARAM
};

// Prevent bad compiler behavior in ASAN mode by wrapping most of the
// arguments in a struct.
class FewerArgsForCompiler {
 public:
  FewerArgsForCompiler(int datasize, const UncompressFlags& flags,
                       int64_t* nwarn,
                       std::function<uint8*(int, int, int)> allocate_output)
      : datasize_(datasize),
        flags_(flags),
        pnwarn_(nwarn),
        allocate_output_(std::move(allocate_output)),
        height_read_(0),
        height_(0),
        stride_(0) {
    if (pnwarn_ != nullptr) *pnwarn_ = 0;
  }

  const int datasize_;
  const UncompressFlags flags_;
  int64_t* const pnwarn_;
  std::function<uint8*(int, int, int)> allocate_output_;
  int height_read_;  // number of scanline lines successfully read
  int height_;
  int stride_;
};

// Check whether the crop window is valid, assuming crop is true.
bool IsCropWindowValid(const UncompressFlags& flags, int input_image_width,
                       int input_image_height) {
  // Crop window is valid only if it is non zero and all the window region is
  // within the original image.
  return flags.crop_width > 0 && flags.crop_height > 0 && flags.crop_x >= 0 &&
         flags.crop_y >= 0 &&
         flags.crop_y + flags.crop_height <= input_image_height &&
         flags.crop_x + flags.crop_width <= input_image_width;
}

#ifdef FUZZING_BUILD_MODE_UNSAFE_FOR_PRODUCTION
// If in fuzzing mode, don't print any error message as that slows down fuzzing.
// See also http://llvm.org/docs/LibFuzzer.html#fuzzer-friendly-build-mode
void no_print(j_common_ptr cinfo) {}
#endif

uint8* UncompressLow(const void* srcdata, FewerArgsForCompiler* argball) {
  // unpack the argball
  const int datasize = argball->datasize_;
  const auto& flags = argball->flags_;
  const int ratio = flags.ratio;
  int components = flags.components;
  int stride = flags.stride;              // may be 0
  int64_t* const nwarn = argball->pnwarn_;  // may be NULL

  // Can't decode if the ratio is not recognized by libjpeg
  if ((ratio != 1) && (ratio != 2) && (ratio != 4) && (ratio != 8)) {
    return nullptr;
  }

  // Channels must be autodetect, grayscale, or rgb.
  if (!(components == 0 || components == 1 || components == 3)) {
    return nullptr;
  }

  // if empty image, return
  if (datasize == 0 || srcdata == nullptr) return nullptr;

  // Declare temporary buffer pointer here so that we can free on error paths
  JSAMPLE* tempdata = nullptr;

  // Initialize libjpeg structures to have a memory source
  // Modify the usual jpeg error manager to catch fatal errors.
  JPEGErrors error = JPEGERRORS_OK;
  struct jpeg_decompress_struct cinfo;
  struct jpeg_error_mgr jerr;
  cinfo.err = jpeg_std_error(&jerr);
  jerr.error_exit = CatchError;

#ifdef FUZZING_BUILD_MODE_UNSAFE_FOR_PRODUCTION
  jerr.output_message = no_print;
#endif

  jmp_buf jpeg_jmpbuf;
  cinfo.client_data = &jpeg_jmpbuf;
  if (setjmp(jpeg_jmpbuf)) {
    delete[] tempdata;
    return nullptr;
  }

  jpeg_create_decompress(&cinfo);
  SetSrc(&cinfo, srcdata, datasize, flags.try_recover_truncated_jpeg);
  jpeg_read_header(&cinfo, TRUE);

  // Set components automatically if desired, autoconverting cmyk to rgb.
  if (components == 0) components = std::min(cinfo.num_components, 3);

  // set grayscale and ratio parameters
  switch (components) {
    case 1:
      cinfo.out_color_space = JCS_GRAYSCALE;
      break;
    case 3:
      if (cinfo.jpeg_color_space == JCS_CMYK ||
          cinfo.jpeg_color_space == JCS_YCCK) {
        // Always use cmyk for output in a 4 channel jpeg. libjpeg has a
        // built-in decoder.  We will further convert to rgb below.
        cinfo.out_color_space = JCS_CMYK;
      } else {
        cinfo.out_color_space = JCS_RGB;
      }
      break;
    default:
      LOG(ERROR) << " Invalid components value " << components << std::endl;
      jpeg_destroy_decompress(&cinfo);
      return nullptr;
  }
  cinfo.do_fancy_upsampling = boolean(flags.fancy_upscaling);
  cinfo.scale_num = 1;
  cinfo.scale_denom = ratio;
  cinfo.dct_method = flags.dct_method;

  // Determine the output image size before attempting decompress to prevent
  // OOM'ing during the decompress
  jpeg_calc_output_dimensions(&cinfo);

  int64_t total_size = static_cast<int64_t>(cinfo.output_height) *
                       static_cast<int64_t>(cinfo.output_width) *
                       static_cast<int64_t>(cinfo.num_components);
  // Some of the internal routines do not gracefully handle ridiculously
  // large images, so fail fast.
  if (cinfo.output_width <= 0 || cinfo.output_height <= 0) {
    LOG(ERROR) << "Invalid image size: " << cinfo.output_width << " x "
               << cinfo.output_height;
    jpeg_destroy_decompress(&cinfo);
    return nullptr;
  }
  if (total_size >= (1LL << 29)) {
    LOG(ERROR) << "Image too large: " << total_size;
    jpeg_destroy_decompress(&cinfo);
    return nullptr;
  }

  jpeg_start_decompress(&cinfo);

  JDIMENSION target_output_width = cinfo.output_width;
  JDIMENSION target_output_height = cinfo.output_height;
  JDIMENSION skipped_scanlines = 0;
#if defined(LIBJPEG_TURBO_VERSION)
  if (flags.crop) {
    // Update target output height and width based on crop window.
    target_output_height = flags.crop_height;
    target_output_width = flags.crop_width;

    // So far, cinfo holds the original input image information.
    if (!IsCropWindowValid(flags, cinfo.output_width, cinfo.output_height)) {
      LOG(ERROR) << "Invalid crop window: x=" << flags.crop_x
                 << ", y=" << flags.crop_y << ", w=" << target_output_width
                 << ", h=" << target_output_height
                 << " for image_width: " << cinfo.output_width
                 << " and image_height: " << cinfo.output_height;
      jpeg_destroy_decompress(&cinfo);
      return nullptr;
    }

    // Update cinfo.output_width. It is tricky that cinfo.output_width must
    // fall on an Minimum Coded Unit (MCU) boundary; if it doesn't, then it will
    // be moved left to the nearest MCU boundary, and width will be increased
    // accordingly. Therefore, the final cinfo.crop_width might differ from the
    // given flags.crop_width. Please see libjpeg library for details.
    JDIMENSION crop_width = flags.crop_width;
    JDIMENSION crop_x = flags.crop_x;
    jpeg_crop_scanline(&cinfo, &crop_x, &crop_width);

    // Update cinfo.output_scanline.
    skipped_scanlines = jpeg_skip_scanlines(&cinfo, flags.crop_y);
    CHECK_EQ(skipped_scanlines, flags.crop_y);
  }
#endif

  // check for compatible stride
  const int min_stride = target_output_width * components * sizeof(JSAMPLE);
  if (stride == 0) {
    stride = min_stride;
  } else if (stride < min_stride) {
    LOG(ERROR) << "Incompatible stride: " << stride << " < " << min_stride;
    jpeg_destroy_decompress(&cinfo);
    return nullptr;
  }

  // Remember stride and height for use in Uncompress
  argball->height_ = target_output_height;
  argball->stride_ = stride;

#if !defined(LIBJPEG_TURBO_VERSION)
  uint8* dstdata = nullptr;
  if (flags.crop) {
    dstdata = new JSAMPLE[stride * target_output_height];
  } else {
    dstdata = argball->allocate_output_(target_output_width,
                                        target_output_height, components);
  }
#else
  uint8* dstdata = argball->allocate_output_(target_output_width,
                                             target_output_height, components);
#endif
  if (dstdata == nullptr) {
    jpeg_destroy_decompress(&cinfo);
    return nullptr;
  }
  JSAMPLE* output_line = static_cast<JSAMPLE*>(dstdata);

  // jpeg_read_scanlines requires the buffers to be allocated based on
  // cinfo.output_width, but the target image width might be different if crop
  // is enabled and crop_width is not MCU aligned. In this case, we need to
  // realign the scanline output to achieve the exact cropping.  Notably, only
  // cinfo.output_width needs to fall on MCU boundary, while cinfo.output_height
  // has no such constraint.
  const bool need_realign_cropped_scanline =
      (target_output_width != cinfo.output_width);
  const bool use_cmyk = (cinfo.out_color_space == JCS_CMYK);

  if (use_cmyk) {
    // Temporary buffer used for CMYK -> RGB conversion.
    tempdata = new JSAMPLE[cinfo.output_width * 4];
  } else if (need_realign_cropped_scanline) {
    // Temporary buffer used for MCU-aligned scanline data.
    tempdata = new JSAMPLE[cinfo.output_width * components];
  }

  // If there is an error reading a line, this aborts the reading.
  // Save the fraction of the image that has been read.
  argball->height_read_ = target_output_height;

  // These variables are just to avoid repeated computation in the loop.
  const int max_scanlines_to_read = skipped_scanlines + target_output_height;
  const int mcu_align_offset =
      (cinfo.output_width - target_output_width) * (use_cmyk ? 4 : components);
  while (cinfo.output_scanline < max_scanlines_to_read) {
    int num_lines_read = 0;
    if (use_cmyk) {
      num_lines_read = jpeg_read_scanlines(&cinfo, &tempdata, 1);
      if (num_lines_read > 0) {
        // Convert CMYK to RGB if scanline read succeeded.
        for (size_t i = 0; i < target_output_width; ++i) {
          int offset = 4 * i;
          if (need_realign_cropped_scanline) {
            // Align the offset for MCU boundary.
            offset += mcu_align_offset;
          }
          const int c = tempdata[offset + 0];
          const int m = tempdata[offset + 1];
          const int y = tempdata[offset + 2];
          const int k = tempdata[offset + 3];
          int r, g, b;
          if (cinfo.saw_Adobe_marker) {
            r = (k * c) / 255;
            g = (k * m) / 255;
            b = (k * y) / 255;
          } else {
            r = (255 - k) * (255 - c) / 255;
            g = (255 - k) * (255 - m) / 255;
            b = (255 - k) * (255 - y) / 255;
          }
          output_line[3 * i + 0] = r;
          output_line[3 * i + 1] = g;
          output_line[3 * i + 2] = b;
        }
      }
    } else if (need_realign_cropped_scanline) {
      num_lines_read = jpeg_read_scanlines(&cinfo, &tempdata, 1);
      if (num_lines_read > 0) {
        memcpy(output_line, tempdata + mcu_align_offset, min_stride);
      }
    } else {
      num_lines_read = jpeg_read_scanlines(&cinfo, &output_line, 1);
    }
    // Handle error cases
    if (num_lines_read == 0) {
      LOG(ERROR) << "Premature end of JPEG data. Stopped at line "
                 << cinfo.output_scanline - skipped_scanlines << "/"
                 << target_output_height;
      if (!flags.try_recover_truncated_jpeg) {
        argball->height_read_ = cinfo.output_scanline - skipped_scanlines;
        error = JPEGERRORS_UNEXPECTED_END_OF_DATA;
      } else {
        for (size_t line = cinfo.output_scanline; line < max_scanlines_to_read;
             ++line) {
          if (line == 0) {
            // If even the first line is missing, fill with black color
            memset(output_line, 0, min_stride);
          } else {
            // else, just replicate the line above.
            memcpy(output_line, output_line - stride, min_stride);
          }
          output_line += stride;
        }
        argball->height_read_ =
            target_output_height;  // consider all lines as read
        // prevent error-on-exit in libjpeg:
        cinfo.output_scanline = max_scanlines_to_read;
      }
      break;
    }
    DCHECK_EQ(num_lines_read, 1);
    TF_ANNOTATE_MEMORY_IS_INITIALIZED(output_line, min_stride);
    output_line += stride;
  }
  delete[] tempdata;
  tempdata = nullptr;

#if defined(LIBJPEG_TURBO_VERSION)
  if (flags.crop && cinfo.output_scanline < cinfo.output_height) {
    // Skip the rest of scanlines, required by jpeg_destroy_decompress.
    jpeg_skip_scanlines(&cinfo,
                        cinfo.output_height - flags.crop_y - flags.crop_height);
    // After this, cinfo.output_height must be equal to cinfo.output_height;
    // otherwise, jpeg_destroy_decompress would fail.
  }
#endif

  // Convert the RGB data to RGBA, with alpha set to 0xFF to indicate
  // opacity.
  // RGBRGBRGB... --> RGBARGBARGBA...
  if (components == 4) {
    // Start on the last line.
    JSAMPLE* scanlineptr = static_cast<JSAMPLE*>(
        dstdata + static_cast<int64_t>(target_output_height - 1) * stride);
    const JSAMPLE kOpaque = -1;  // All ones appropriate for JSAMPLE.
    const int right_rgb = (target_output_width - 1) * 3;
    const int right_rgba = (target_output_width - 1) * 4;

    for (int y = target_output_height; y-- > 0;) {
      // We do all the transformations in place, going backwards for each row.
      const JSAMPLE* rgb_pixel = scanlineptr + right_rgb;
      JSAMPLE* rgba_pixel = scanlineptr + right_rgba;
      scanlineptr -= stride;
      for (int x = target_output_width; x-- > 0;
           rgba_pixel -= 4, rgb_pixel -= 3) {
        // We copy the 3 bytes at rgb_pixel into the 4 bytes at rgba_pixel
        // The "a" channel is set to be opaque.
        rgba_pixel[3] = kOpaque;
        rgba_pixel[2] = rgb_pixel[2];
        rgba_pixel[1] = rgb_pixel[1];
        rgba_pixel[0] = rgb_pixel[0];
      }
    }
  }

  switch (components) {
    case 1:
      if (cinfo.output_components != 1) {
        error = JPEGERRORS_BAD_PARAM;
      }
      break;
    case 3:
    case 4:
      if (cinfo.out_color_space == JCS_CMYK) {
        if (cinfo.output_components != 4) {
          error = JPEGERRORS_BAD_PARAM;
        }
      } else {
        if (cinfo.output_components != 3) {
          error = JPEGERRORS_BAD_PARAM;
        }
      }
      break;
    default:
      // will never happen, should be caught by the previous switch
      LOG(ERROR) << "Invalid components value " << components << std::endl;
      jpeg_destroy_decompress(&cinfo);
      return nullptr;
  }

  // save number of warnings if requested
  if (nwarn != nullptr) {
    *nwarn = cinfo.err->num_warnings;
  }

  // Handle errors in JPEG
  switch (error) {
    case JPEGERRORS_OK:
      jpeg_finish_decompress(&cinfo);
      break;
    case JPEGERRORS_UNEXPECTED_END_OF_DATA:
    case JPEGERRORS_BAD_PARAM:
      jpeg_abort(reinterpret_cast<j_common_ptr>(&cinfo));
      break;
    default:
      LOG(ERROR) << "Unhandled case " << error;
      break;
  }

#if !defined(LIBJPEG_TURBO_VERSION)
  // TODO(tanmingxing): delete all these code after migrating to libjpeg_turbo
  // for Windows.
  if (flags.crop) {
    // Update target output height and width based on crop window.
    target_output_height = flags.crop_height;
    target_output_width = flags.crop_width;

    // cinfo holds the original input image information.
    if (!IsCropWindowValid(flags, cinfo.output_width, cinfo.output_height)) {
      LOG(ERROR) << "Invalid crop window: x=" << flags.crop_x
                 << ", y=" << flags.crop_y << ", w=" << target_output_width
                 << ", h=" << target_output_height
                 << " for image_width: " << cinfo.output_width
                 << " and image_height: " << cinfo.output_height;
      delete[] dstdata;
      jpeg_destroy_decompress(&cinfo);
      return nullptr;
    }

    const uint8* full_image = dstdata;
    dstdata = argball->allocate_output_(target_output_width,
                                        target_output_height, components);
    if (dstdata == nullptr) {
      delete[] full_image;
      jpeg_destroy_decompress(&cinfo);
      return nullptr;
    }

    const int full_image_stride = stride;
    // Update stride and hight for crop window.
    const int min_stride = target_output_width * components * sizeof(JSAMPLE);
    if (flags.stride == 0) {
      stride = min_stride;
    }
    argball->height_ = target_output_height;
    argball->stride_ = stride;

    if (argball->height_read_ > target_output_height) {
      argball->height_read_ = target_output_height;
    }
    const int crop_offset = flags.crop_x * components * sizeof(JSAMPLE);
    const uint8* full_image_ptr = full_image + flags.crop_y * full_image_stride;
    uint8* crop_image_ptr = dstdata;
    for (int i = 0; i < argball->height_read_; i++) {
      memcpy(crop_image_ptr, full_image_ptr + crop_offset, min_stride);
      crop_image_ptr += stride;
      full_image_ptr += full_image_stride;
    }
    delete[] full_image;
  }
#endif

  jpeg_destroy_decompress(&cinfo);
  return dstdata;
}

}  // anonymous namespace

// -----------------------------------------------------------------------------
//  We do the apparently silly thing of packing 5 of the arguments
//  into a structure that is then passed to another routine
//  that does all the work.  The reason is that we want to catch
//  fatal JPEG library errors with setjmp/longjmp, and g++ and
//  associated libraries aren't good enough to guarantee that 7
//  parameters won't get clobbered by the longjmp.  So we help
//  it out a little.
uint8* Uncompress(const void* srcdata, int datasize,
                  const UncompressFlags& flags, int64_t* nwarn,
                  std::function<uint8*(int, int, int)> allocate_output) {
  FewerArgsForCompiler argball(datasize, flags, nwarn,
                               std::move(allocate_output));
  uint8* const dstdata = UncompressLow(srcdata, &argball);

  const float fraction_read =
      argball.height_ == 0
          ? 1.0
          : (static_cast<float>(argball.height_read_) / argball.height_);
  if (dstdata == nullptr ||
      fraction_read < std::min(1.0f, flags.min_acceptable_fraction)) {
    // Major failure, none or too-partial read returned; get out
    return nullptr;
  }

  // If there was an error in reading the jpeg data,
  // set the unread pixels to black
  if (argball.height_read_ != argball.height_) {
    const int first_bad_line = argball.height_read_;
    uint8* start = dstdata + first_bad_line * argball.stride_;
    const int nbytes = (argball.height_ - first_bad_line) * argball.stride_;
    memset(static_cast<void*>(start), 0, nbytes);
  }

  return dstdata;
}

uint8* Uncompress(const void* srcdata, int datasize,
                  const UncompressFlags& flags, int* pwidth, int* pheight,
                  int* pcomponents, int64_t* nwarn) {
  uint8* buffer = nullptr;
  uint8* result =
      Uncompress(srcdata, datasize, flags, nwarn,
                 [=, &buffer](int width, int height, int components) {
                   if (pwidth != nullptr) *pwidth = width;
                   if (pheight != nullptr) *pheight = height;
                   if (pcomponents != nullptr) *pcomponents = components;
                   buffer = new uint8[height * width * components];
                   return buffer;
                 });
  if (!result) delete[] buffer;
  return result;
}

// ----------------------------------------------------------------------------
// Computes image information from jpeg header.
// Returns true on success; false on failure.
bool GetImageInfo(const void* srcdata, int datasize, int* width, int* height,
                  int* components) {
  // Init in case of failure
  if (width) *width = 0;
  if (height) *height = 0;
  if (components) *components = 0;

  // If empty image, return
  if (datasize == 0 || srcdata == nullptr) return false;

  // Initialize libjpeg structures to have a memory source
  // Modify the usual jpeg error manager to catch fatal errors.
  struct jpeg_decompress_struct cinfo;
  struct jpeg_error_mgr jerr;
  jmp_buf jpeg_jmpbuf;
  cinfo.err = jpeg_std_error(&jerr);
  cinfo.client_data = &jpeg_jmpbuf;
  jerr.error_exit = CatchError;
  if (setjmp(jpeg_jmpbuf)) {
    return false;
  }

  // set up, read header, set image parameters, save size
  jpeg_create_decompress(&cinfo);
  SetSrc(&cinfo, srcdata, datasize, false);

  jpeg_read_header(&cinfo, TRUE);
  jpeg_calc_output_dimensions(&cinfo);
  if (width) *width = cinfo.output_width;
  if (height) *height = cinfo.output_height;
  if (components) *components = cinfo.output_components;

  jpeg_destroy_decompress(&cinfo);

  return true;
}

// -----------------------------------------------------------------------------
// Compression

namespace {
bool CompressInternal(const uint8* srcdata, int width, int height,
                      const CompressFlags& flags, tstring* output) {
  if (output == nullptr) {
    LOG(ERROR) << "Output buffer is null: ";
    return false;
  }

  output->clear();
  const int components = (static_cast<int>(flags.format) & 0xff);

  int64_t total_size =
      static_cast<int64_t>(width) * static_cast<int64_t>(height);
  // Some of the internal routines do not gracefully handle ridiculously
  // large images, so fail fast.
  if (width <= 0 || height <= 0) {
    LOG(ERROR) << "Invalid image size: " << width << " x " << height;
    return false;
  }
  if (total_size >= (1LL << 29)) {
    LOG(ERROR) << "Image too large: " << total_size;
    return false;
  }

  int in_stride = flags.stride;
  if (in_stride == 0) {
    in_stride = width * (static_cast<int>(flags.format) & 0xff);
  } else if (in_stride < width * components) {
    LOG(ERROR) << "Incompatible input stride";
    return false;
  }

  JOCTET* buffer = nullptr;

  // NOTE: for broader use xmp_metadata should be made a Unicode string
  CHECK(srcdata != nullptr);
  CHECK(output != nullptr);
  // This struct contains the JPEG compression parameters and pointers to
  // working space
  struct jpeg_compress_struct cinfo;
  // This struct represents a JPEG error handler.
  struct jpeg_error_mgr jerr;
  jmp_buf jpeg_jmpbuf;  // recovery point in case of error

  // Step 1: allocate and initialize JPEG compression object
  // Use the usual jpeg error manager.
  cinfo.err = jpeg_std_error(&jerr);
  cinfo.client_data = &jpeg_jmpbuf;
  jerr.error_exit = CatchError;
  if (setjmp(jpeg_jmpbuf)) {
    output->clear();
    delete[] buffer;
    return false;
  }

  jpeg_create_compress(&cinfo);

  // Step 2: specify data destination
  // We allocate a buffer of reasonable size. If we have a small image, just
  // estimate the size of the output using the number of bytes of the input.
  // If this is getting too big, we will append to the string by chunks of 1MB.
  // This seems like a reasonable compromise between performance and memory.
  int bufsize = std::min(width * height * components, 1 << 20);
  buffer = new JOCTET[bufsize];
  SetDest(&cinfo, buffer, bufsize, output);

  // Step 3: set parameters for compression
  cinfo.image_width = width;
  cinfo.image_height = height;
  switch (components) {
    case 1:
      cinfo.input_components = 1;
      cinfo.in_color_space = JCS_GRAYSCALE;
      break;
    case 3:
    case 4:
      cinfo.input_components = 3;
      cinfo.in_color_space = JCS_RGB;
      break;
    default:
      LOG(ERROR) << " Invalid components value " << components << std::endl;
      output->clear();
      delete[] buffer;
      return false;
  }
  jpeg_set_defaults(&cinfo);
  if (flags.optimize_jpeg_size) cinfo.optimize_coding = TRUE;

  cinfo.density_unit = flags.density_unit;  // JFIF code for pixel size units:
                                            // 1 = in, 2 = cm
  cinfo.X_density = flags.x_density;        // Horizontal pixel density
  cinfo.Y_density = flags.y_density;        // Vertical pixel density
  jpeg_set_quality(&cinfo, flags.quality, TRUE);

  if (flags.progressive) {
    jpeg_simple_progression(&cinfo);
  }

  if (!flags.chroma_downsampling) {
    // Turn off chroma subsampling (it is on by default).  For more details on
    // chroma subsampling, see http://en.wikipedia.org/wiki/Chroma_subsampling.
    for (int i = 0; i < cinfo.num_components; ++i) {
      cinfo.comp_info[i].h_samp_factor = 1;
      cinfo.comp_info[i].v_samp_factor = 1;
    }
  }

  jpeg_start_compress(&cinfo, TRUE);

  // Embed XMP metadata if any
  if (!flags.xmp_metadata.empty()) {
    // XMP metadata is embedded in the APP1 tag of JPEG and requires this
    // namespace header string (null-terminated)
    const string name_space = "http://ns.adobe.com/xap/1.0/";
    const int name_space_length = name_space.size();
    const int metadata_length = flags.xmp_metadata.size();
    const int packet_length = metadata_length + name_space_length + 1;
    std::unique_ptr<JOCTET[]> joctet_packet(new JOCTET[packet_length]);

    for (int i = 0; i < name_space_length; i++) {
      // Conversion char --> JOCTET
      joctet_packet[i] = name_space[i];
    }
    joctet_packet[name_space_length] = 0;  // null-terminate namespace string

    for (int i = 0; i < metadata_length; i++) {
      // Conversion char --> JOCTET
      joctet_packet[i + name_space_length + 1] = flags.xmp_metadata[i];
    }
    jpeg_write_marker(&cinfo, JPEG_APP0 + 1, joctet_packet.get(),
                      packet_length);
  }

  // JSAMPLEs per row in image_buffer
  std::unique_ptr<JSAMPLE[]> row_temp(
      new JSAMPLE[width * cinfo.input_components]);
  while (cinfo.next_scanline < cinfo.image_height) {
    JSAMPROW row_pointer[1];  // pointer to JSAMPLE row[s]
    const uint8* r = &srcdata[cinfo.next_scanline * in_stride];
    uint8* p = static_cast<uint8*>(row_temp.get());
    switch (flags.format) {
      case FORMAT_RGBA: {
        for (int i = 0; i < width; ++i, p += 3, r += 4) {
          p[0] = r[0];
          p[1] = r[1];
          p[2] = r[2];
        }
        row_pointer[0] = row_temp.get();
        break;
      }
      case FORMAT_ABGR: {
        for (int i = 0; i < width; ++i, p += 3, r += 4) {
          p[0] = r[3];
          p[1] = r[2];
          p[2] = r[1];
        }
        row_pointer[0] = row_temp.get();
        break;
      }
      default: {
        row_pointer[0] = reinterpret_cast<JSAMPLE*>(const_cast<JSAMPLE*>(r));
      }
    }
    CHECK_EQ(jpeg_write_scanlines(&cinfo, row_pointer, 1), 1u);
  }
  jpeg_finish_compress(&cinfo);

  // release JPEG compression object
  jpeg_destroy_compress(&cinfo);
  delete[] buffer;
  return true;
}

}  // anonymous namespace

// -----------------------------------------------------------------------------

bool Compress(const void* srcdata, int width, int height,
              const CompressFlags& flags, tstring* output) {
  return CompressInternal(static_cast<const uint8*>(srcdata), width, height,
                          flags, output);
}

tstring Compress(const void* srcdata, int width, int height,
                 const CompressFlags& flags) {
  tstring temp;
  CompressInternal(static_cast<const uint8*>(srcdata), width, height, flags,
                   &temp);
  // If CompressInternal fails, temp will be empty.
  return temp;
}

}  // namespace jpeg
}  // namespace tensorflow
