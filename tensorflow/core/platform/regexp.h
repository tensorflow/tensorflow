#ifndef TENSORFLOW_PLATFORM_REGEXP_H_
#define TENSORFLOW_PLATFORM_REGEXP_H_

#include "tensorflow/core/platform/port.h"

#if defined(PLATFORM_GOOGLE) || defined(PLATFORM_GOOGLE_ANDROID)
#include "third_party/re2/re2.h"
namespace tensorflow {
typedef ::StringPiece RegexpStringPiece;
}  // namespace tensorflow

#else

#include "external/re2/re2/re2.h"
namespace tensorflow {
typedef re2::StringPiece RegexpStringPiece;
}  // namespace tensorflow

#endif

namespace tensorflow {

// Conversion to/from the appropriate StringPiece type for using in RE2
inline RegexpStringPiece ToRegexpStringPiece(tensorflow::StringPiece sp) {
  return RegexpStringPiece(sp.data(), sp.size());
}
inline tensorflow::StringPiece FromRegexpStringPiece(RegexpStringPiece sp) {
  return tensorflow::StringPiece(sp.data(), sp.size());
}

}  // namespace tensorflow

#endif  // TENSORFLOW_PLATFORM_REGEXP_H_
