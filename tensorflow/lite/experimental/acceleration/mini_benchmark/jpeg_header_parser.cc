/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/jpeg_header_parser.h"

#include <cstdint>
#include <string>

#include "tensorflow/lite/core/c/c_api_types.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/minimal_logging.h"
#include "tensorflow/lite/string_util.h"

namespace tflite {
namespace acceleration {
namespace decode_jpeg_kernel {

/*

JPEG file overall file structure

SOI Marker                 FFD8
Marker XX size=SSSS        FFXX	SSSS	DDDD......
Marker YY size=TTTT        FFYY	TTTT	DDDD......
SOFn marker with the info we want
SOS Marker size=UUUU       FFDA	UUUU	DDDD....
Image stream               I I I I....
EOI Marker                 FFD9

The first marker is either APP0 (JFIF format) or APP1 (EXIF format)

We support only JFIF images
*/

namespace {

using MarkerId = uint16_t;

void AsWord(int value, char* msb, char* lsb) {
  *msb = static_cast<char>(value >> 8);
  *lsb = static_cast<char>(value);
}

// JFIF spec at
// https://www.ecma-international.org/publications-and-standards/technical-reports/ecma-tr-98/
// Marker definition summary at
// http://lad.dsc.ufcg.edu.br/multimidia/jpegmarker.pdf
// Overall JPEG File structure with discussion of the supported number of
// channels per format
// https://docs.oracle.com/javase/8/docs/api/javax/imageio/metadata/doc-files/jpeg_metadata.html
//

class JfifHeaderParser {
 public:
  explicit JfifHeaderParser(const tflite::StringRef& jpeg_image_data)
      : jpeg_image_data_(jpeg_image_data), offset_(0) {
    if (!IsJpegImage(jpeg_image_data_)) {
      is_valid_image_buffer_ = false;
      validation_error_message_ = "Not a valid JPEG image.";
    } else if (!IsJfifImage(jpeg_image_data_)) {
      is_valid_image_buffer_ = false;
      validation_error_message_ = "Image is not in JFIF format.";
      return;
    } else {
      is_valid_image_buffer_ = true;
    }
  }

#define ENSURE_READ_STATUS(a)                           \
  do {                                                  \
    const TfLiteStatus s = (a);                         \
    if (s != kTfLiteOk) {                               \
      return {s, "Error trying to parse JPEG header."}; \
    }                                                   \
  } while (0)

  Status ReadJpegHeader(JpegHeader* result) {
    if (!is_valid_image_buffer_) {
      return {kTfLiteError, validation_error_message_};
    }

    Status move_to_sof_status = MoveToStartOfFrameMarker();
    if (move_to_sof_status.code != kTfLiteOk) {
      return move_to_sof_status;
    }

    ENSURE_READ_STATUS(SkipBytes(2));  // skipping marker length
    char precision;
    ENSURE_READ_STATUS(ReadByte(&precision));
    uint16_t height;
    ENSURE_READ_STATUS(ReadWord(&height));
    uint16_t width;
    ENSURE_READ_STATUS(ReadWord(&width));
    char num_of_components;
    ENSURE_READ_STATUS(ReadByte(&num_of_components));

    if (num_of_components != 1 && num_of_components != 3) {
      return {kTfLiteError,
              "A JFIF image without App14 marker doesn't support a number of "
              "components = " +
                  std::to_string(static_cast<int>(num_of_components))};
    }

    result->width = width;
    result->height = height;
    result->channels = num_of_components;
    result->bits_per_sample = precision;

    return {kTfLiteOk, ""};
  }

  Status ApplyHeaderToImage(const JpegHeader& new_header,
                            std::string& write_to) {
    if (!is_valid_image_buffer_) {
      return {kTfLiteError, validation_error_message_};
    }

    Status move_to_sof_status = MoveToStartOfFrameMarker();
    if (move_to_sof_status.code != kTfLiteOk) {
      return move_to_sof_status;
    }
    ENSURE_READ_STATUS(SkipBytes(2));  // skipping marker length

    if (!HasData(6)) {
      return {kTfLiteError,
              "Invalid SOF marker, image buffer ends before end of marker"};
    }

    char header[6];
    header[0] = static_cast<char>(new_header.bits_per_sample);
    AsWord(new_header.height, header + 1, header + 2);
    AsWord(new_header.width, header + 3, header + 4);
    header[5] = static_cast<char>(new_header.channels);

    write_to.clear();
    write_to.append(jpeg_image_data_.str, offset_);
    write_to.append(header, 6);

    ENSURE_READ_STATUS(SkipBytes(6));
    if (HasData()) {
      write_to.append(jpeg_image_data_.str + offset_,
                      jpeg_image_data_.len - offset_);
    }

    return {kTfLiteOk, ""};
  }

 private:
  const tflite::StringRef jpeg_image_data_;
  // Using int for consistency with the size in StringRef
  int offset_;
  bool is_valid_image_buffer_;
  std::string validation_error_message_;

  // Moves to the begin of the first SOF marker
  Status MoveToStartOfFrameMarker() {
    const MarkerId kStartOfStreamMarkerId = 0xFFDA;  // Start of image data

    offset_ = 0;
    ENSURE_READ_STATUS(SkipBytes(4));  // skipping SOI and APP0 marker IDs
    ENSURE_READ_STATUS(SkipCurrentMarker());  // skipping APP0
    MarkerId curr_marker_id;
    // We need at least 2 bytes for the marker ID and 2 for the length
    while (HasData(/*min_data_size=*/4)) {
      ENSURE_READ_STATUS(ReadWord(&curr_marker_id));
      // We are breaking at the first met SOF marker. This won't generate
      // results inconsistent with LibJPEG because only
      // image with a single SOF marker are successfully parsed by it.
      // LibJPEG fails if more than one marker is found in the header (see
      // https://github.com/libjpeg-turbo/libjpeg-turbo/blob/main/jerror.h#L121
      // and
      // https://github.com/libjpeg-turbo/libjpeg-turbo/blob/main/jdmarker.c#L264-L265
      if (IsStartOfFrameMarkerId(curr_marker_id)) {
        break;
      }
      if (curr_marker_id == kStartOfStreamMarkerId) {
        return {kTfLiteError, "Error trying to parse JPEG header."};
      }
      ENSURE_READ_STATUS(SkipCurrentMarker());
    }

    return {kTfLiteOk, ""};
  }

#undef ENSURE_READ_STATUS

  bool HasData(int min_data_size = 1) {
    return offset_ <= jpeg_image_data_.len - min_data_size;
  }

  TfLiteStatus SkipBytes(int bytes) {
    if (!HasData(bytes)) {
      TFLITE_LOG(TFLITE_LOG_WARNING,
                 "Trying to move out of image boundaries from offset %d, "
                 "skipping %d bytes",
                 offset_, bytes);
      return kTfLiteError;
    }

    offset_ += bytes;

    return kTfLiteOk;
  }

  TfLiteStatus ReadByte(char* result) {
    if (!HasData()) {
      return kTfLiteError;
    }

    *result = jpeg_image_data_.str[offset_];

    return SkipBytes(1);
  }

  TfLiteStatus ReadWord(uint16_t* result) {
    TF_LITE_ENSURE_STATUS(ReadWordAt(jpeg_image_data_, offset_, result));
    return SkipBytes(2);
  }

  TfLiteStatus SkipCurrentMarker() {
    // We just read the marker ID so we are on top of the marker len
    uint16_t full_marker_len;
    TF_LITE_ENSURE_STATUS(ReadWord(&full_marker_len));
    if (full_marker_len <= 2) {
      TFLITE_LOG(TFLITE_LOG_WARNING,
                 "Invalid marker length %d read at offset %X", full_marker_len,
                 offset_);
      return kTfLiteError;
    }

    // The marker len includes the 2 bytes of marker length
    return SkipBytes(full_marker_len - 2);
  }

  static TfLiteStatus ReadWordAt(const tflite::StringRef& jpeg_image_data,
                                 int read_offset, uint16_t* result) {
    if (read_offset < 0 || read_offset + 2 > jpeg_image_data.len) {
      return kTfLiteError;
    }
    // Cast to unsigned since char can be signed.
    const unsigned char* buf =
        reinterpret_cast<const unsigned char*>(jpeg_image_data.str);

    *result = (buf[read_offset] << 8) + buf[read_offset + 1];

    return kTfLiteOk;
  }

  static bool IsJpegImage(const tflite::StringRef& jpeg_image_data) {
    const MarkerId kStartOfImageMarkerId = 0xFFD8;
    const MarkerId kEndOfImageMarkerId = 0xFFD9;

    MarkerId soi_marker_id;
    MarkerId eoi_marker_id;
    if (ReadWordAt(jpeg_image_data, 0, &soi_marker_id) != kTfLiteOk) {
      return false;
    }
    if (ReadWordAt(jpeg_image_data, jpeg_image_data.len - 2, &eoi_marker_id) !=
        kTfLiteOk) {
      return false;
    }

    return (soi_marker_id == kStartOfImageMarkerId) &&
           (eoi_marker_id == kEndOfImageMarkerId);
  }

  static bool IsJfifImage(const tflite::StringRef& jpeg_image_data) {
    const MarkerId kApp0MarkerId = 0xFFE0;  // First marker in JIFF image

    MarkerId app_marker_id;
    if ((ReadWordAt(jpeg_image_data, 2, &app_marker_id) != kTfLiteOk) ||
        (app_marker_id != kApp0MarkerId)) {
      return false;
    }

    // Checking Jfif identifier string "JFIF\0" in APP0 Marker
    const std::string kJfifIdString{"JFIF\0", 5};

    // The ID starts after SOI (2 bytes), APP0 marker IDs (2 bytes) and 2 other
    // bytes with APP0 marker length
    const int KJfifIdStringStartOffset = 6;

    if (KJfifIdStringStartOffset + kJfifIdString.size() >=
        jpeg_image_data.len) {
      TFLITE_LOG(TFLITE_LOG_WARNING,
                 "Invalid image, reached end of data at offset while "
                 "parsing APP0 header");
      return false;
    }

    const std::string actualImgId(
        jpeg_image_data.str + KJfifIdStringStartOffset, kJfifIdString.size());
    if (kJfifIdString != actualImgId) {
      TFLITE_LOG(TFLITE_LOG_WARNING, "Invalid image, invalid APP0 header");

      return false;
    }

    return true;
  }

  static bool IsStartOfFrameMarkerId(MarkerId marker_id) {
    return 0xFFC0 <= marker_id && marker_id < 0xFFCF;
  }
};

}  // namespace
Status ReadJpegHeader(const tflite::StringRef& jpeg_image_data,
                      JpegHeader* header) {
  JfifHeaderParser parser(jpeg_image_data);

  return parser.ReadJpegHeader(header);
}

Status BuildImageWithNewHeader(const tflite::StringRef& orig_jpeg_image_data,
                               const JpegHeader& new_header,
                               std::string& new_image_data) {
  JfifHeaderParser parser(orig_jpeg_image_data);

  return parser.ApplyHeaderToImage(new_header, new_image_data);
}

}  // namespace decode_jpeg_kernel
}  // namespace acceleration
}  // namespace tflite
