// Helper routines for encoding/decoding tensor contents.
#ifndef TENSORFLOW_PLATFORM_TENSOR_CODING_H_
#define TENSORFLOW_PLATFORM_TENSOR_CODING_H_

#include <string>
#include "tensorflow/core/lib/core/refcount.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/platform/port.h"

#ifdef PLATFORM_GOOGLE
#include "tensorflow/core/platform/google/cord_coding.h"
#endif

namespace tensorflow {
namespace port {

// Store src contents in *out.  If backing memory for src is shared with *out,
// will ref obj during the call and will arrange to unref obj when no
// longer needed.
void AssignRefCounted(StringPiece src, core::RefCounted* obj, string* out);

// Copy contents of src to dst[0,src.size()-1].
inline void CopyToArray(const string& src, char* dst) {
  memcpy(dst, src.data(), src.size());
}

// Store encoding of strings[0..n-1] in *out.
void EncodeStringList(const string* strings, int64 n, string* out);

// Decode n strings from src and store in strings[0..n-1].
// Returns true if successful, false on parse error.
bool DecodeStringList(const string& src, string* strings, int64 n);

// Assigns base[0..bytes-1] to *s
void CopyFromArray(string* s, const char* base, size_t bytes);

}  // namespace port
}  // namespace tensorflow

#endif  // TENSORFLOW_PLATFORM_TENSOR_CODING_H_
