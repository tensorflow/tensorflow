/* Copyright 2015 Google Inc. All Rights Reserved.

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

#ifndef TENSORFLOW_LIB_STRINGS_NUMBERS_H_
#define TENSORFLOW_LIB_STRINGS_NUMBERS_H_

#include <string>

#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace strings {

// ----------------------------------------------------------------------
// FastIntToBufferLeft()
//    These are intended for speed.
//
//    All functions take the output buffer as an arg.  FastInt() uses
//    at most 22 bytes, FastTime() uses exactly 30 bytes.  They all
//    return a pointer to the beginning of the output, which is the same as
//    the beginning of the input buffer.
//
//    NOTE: In 64-bit land, sizeof(time_t) is 8, so it is possible
//    to pass to FastTimeToBuffer() a time whose year cannot be
//    represented in 4 digits. In this case, the output buffer
//    will contain the string "Invalid:<value>"
// ----------------------------------------------------------------------

// Previously documented minimums -- the buffers provided must be at least this
// long, though these numbers are subject to change:
//     Int32, UInt32:                   12 bytes
//     Int64, UInt64, Int, Uint:        22 bytes
//     Time:                            30 bytes
// Use kFastToBufferSize rather than hardcoding constants.
static const int kFastToBufferSize = 32;

// ----------------------------------------------------------------------
// FastInt32ToBufferLeft()
// FastUInt32ToBufferLeft()
// FastInt64ToBufferLeft()
// FastUInt64ToBufferLeft()
//
// These functions convert their numeric argument to an ASCII
// representation of the numeric value in base 10, with the
// representation being left-aligned in the buffer.  The caller is
// responsible for ensuring that the buffer has enough space to hold
// the output.  The buffer should typically be at least kFastToBufferSize
// bytes.
//
// Returns a pointer to the end of the string (i.e. the null character
// terminating the string).
// ----------------------------------------------------------------------

char* FastInt32ToBufferLeft(int32 i, char* buffer);    // at least 12 bytes
char* FastUInt32ToBufferLeft(uint32 i, char* buffer);  // at least 12 bytes
char* FastInt64ToBufferLeft(int64 i, char* buffer);    // at least 22 bytes
char* FastUInt64ToBufferLeft(uint64 i, char* buffer);  // at least 22 bytes

// Required buffer size for DoubleToBuffer is kFastToBufferSize.
// Required buffer size for FloatToBuffer is kFastToBufferSize.
char* DoubleToBuffer(double i, char* buffer);
char* FloatToBuffer(float i, char* buffer);

// Convert a 64-bit fingerprint value to an ASCII representation.
string FpToString(Fprint fp);

// Attempt to parse a fingerprint in the form encoded by FpToString.  If
// successful, stores the fingerprint in *fp and returns true.  Otherwise,
// returns false.
bool StringToFp(const string& s, Fprint* fp);

// Convert a 64-bit fingerprint value to an ASCII representation that
// is terminated by a '\0'.
// Buf must point to an array of at least kFastToBufferSize characters
StringPiece Uint64ToHexString(uint64 v, char* buf);

// Attempt to parse a uint64 in the form encoded by FastUint64ToHexString.  If
// successful, stores the value in *v and returns true.  Otherwise,
// returns false.
bool HexStringToUint64(const StringPiece& s, uint64* v);

// Convert strings to 32bit integer values.
// Leading and trailing spaces are allowed.
// Return false with overflow or invalid input.
bool safe_strto32(StringPiece str, int32* value);

// Convert strings to unsigned 32bit integer values.
// Leading and trailing spaces are allowed.
// Return false with overflow or invalid input.
bool safe_strtou32(StringPiece str, uint32* value);

// Convert strings to 64bit integer values.
// Leading and trailing spaces are allowed.
// Return false with overflow or invalid input.
bool safe_strto64(StringPiece str, int64* value);

// Convert strings to unsigned 64bit integer values.
// Leading and trailing spaces are allowed.
// Return false with overflow or invalid input.
bool safe_strtou64(StringPiece str, uint64* value);

// Convert strings to floating point values.
// Leading and trailing spaces are allowed.
// Values may be rounded on over- and underflow.
bool safe_strtof(const char* str, float* value);

// Convert strings to double precision floating point values.
// Leading and trailing spaces are allowed.
// Values may be rounded on over- and underflow.
bool safe_strtod(const char* str, double* value);

// Converts from an int64 representing a number of bytes to a
// human readable string representing the same number.
// e.g. 12345678 -> "11.77MiB".
string HumanReadableNumBytes(int64 num_bytes);

}  // namespace strings
}  // namespace tensorflow

#endif  // TENSORFLOW_LIB_STRINGS_NUMBERS_H_
