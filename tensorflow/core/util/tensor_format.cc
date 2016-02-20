#include "tensorflow/core/util/tensor_format.h"

namespace tensorflow {

string GetConvnetDataFormatAttrString() {
  return "data_format: { 'NHWC', 'NCHW' } = 'NHWC' ";
}

string ToString(TensorFormat format) {
  switch (format) {
    case FORMAT_NHWC:
      return "NHWC";
    case FORMAT_NCHW:
      return "NCHW";
    default:
      LOG(FATAL) << "Invalid Format: " << static_cast<int32>(format);
      return "INVALID_FORMAT";
  }
}

bool FormatFromString(const string& format_str, TensorFormat* format) {
  if (format_str == "NHWC") {
    *format = FORMAT_NHWC;
    return true;
  } else if (format_str == "NCHW") {
    *format = FORMAT_NCHW;
    return true;
  }
  return false;
}

}  // namespace tensorflow
