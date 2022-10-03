/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_TSL_PLATFORM_CTSTRING_H_
#define TENSORFLOW_TSL_PLATFORM_CTSTRING_H_

#include <stdint.h>
#include <stdlib.h>

#include "tensorflow/tsl/platform/ctstring_internal.h"

// Initialize a new tstring.  This must be called before using any function
// below.
inline void TF_TString_Init(TF_TString *str);
// Deallocate a tstring.
inline void TF_TString_Dealloc(TF_TString *str);

// Resizes `str' to `new_size'.  This function will appropriately grow or shrink
// the string buffer to fit a `new_size' string.  Grown regions of the string
// will be initialized with `c'.
inline char *TF_TString_Resize(TF_TString *str, size_t new_size, char c);
// Similar to TF_TString_Resize, except the newly allocated regions will remain
// uninitialized.  This is useful if you plan on overwriting the newly grown
// regions immediately after allocation; doing so will elide a superfluous
// initialization of the new buffer.
inline char *TF_TString_ResizeUninitialized(TF_TString *str, size_t new_size);
// Reserves a string buffer with a capacity of at least `new_cap'.
// Reserve will not change the size, or the contents of the existing
// string.  This is useful if you have a rough idea of `str's upperbound in
// size, and want to avoid allocations as you append to `str'. It should not be
// considered safe to write in the region between size and capacity; explicitly
// resize before doing so.
inline void TF_TString_Reserve(TF_TString *str, size_t new_cap);
// Similar to TF_TString_Reserve, except that we ensure amortized growth, i.e.
// that we grow the capacity by at least a constant factor >1.
inline void TF_TString_ReserveAmortized(TF_TString *str, size_t new_cap);

// Returns the size of the string.
inline size_t TF_TString_GetSize(const TF_TString *str);
// Returns the capacity of the string buffer.  It should not be considered safe
// to write in the region between size and capacity---call Resize or
// ResizeUninitialized before doing so.
inline size_t TF_TString_GetCapacity(const TF_TString *str);
// Returns the underlying type of the tstring:
// TF_TSTR_SMALL:
//    Small string optimization; the contents of strings
//    less than 22-bytes are stored in the TF_TString struct. This avoids any
//    heap allocations.
// TF_TSTR_LARGE:
//    Heap allocated string.
// TF_TSTR_OFFSET: (currently unused)
//    An offset defined string.  The string buffer begins at an internally
//    defined little-endian offset from `str'; i.e. GetDataPointer() = str +
//    offset.  This type is useful for memory mapping or reading string tensors
//    directly from file, without the need to deserialize the data.  For
//    security reasons, it is imperative that OFFSET based string tensors are
//    validated before use, or are from a trusted source.
// TF_TSTR_VIEW:
//    A view into an unowned character string.
//
// NOTE:
//    VIEW and OFFSET types are immutable, so any modifcation via Append,
//    AppendN, or GetMutableDataPointer of a VIEW/OFFSET based tstring will
//    result in a conversion to an owned type (SMALL/LARGE).
inline TF_TString_Type TF_TString_GetType(const TF_TString *str);

// Returns a const char pointer to the start of the underlying string. The
// underlying character buffer may not be null-terminated.
inline const char *TF_TString_GetDataPointer(const TF_TString *str);
// Returns a char pointer to a mutable representation of the underlying string.
// In the case of VIEW and OFFSET types, `src' is converted to an owned type
// (SMALL/LARGE).  The underlying character buffer may not be null-terminated.
inline char *TF_TString_GetMutableDataPointer(TF_TString *str);

// Sets `dst' as a VIEW type to `src'.  `dst' will not take ownership of `src'.
// It is the user's responsibility to ensure that the lifetime of `src' exceeds
// `dst'.  Any mutations to `dst' via Append, AppendN, or GetMutableDataPointer,
// will result in a copy into an owned SMALL or LARGE type, and will not modify
// `src'.
inline void TF_TString_AssignView(TF_TString *dst, const char *src,
                                  size_t size);

// Appends `src' onto `dst'.  If `dst' is a VIEW or OFFSET type, it will first
// be converted to an owned LARGE or SMALL type.  `dst' should not point to
// memory owned by `src'.
inline void TF_TString_Append(TF_TString *dst, const TF_TString *src);
inline void TF_TString_AppendN(TF_TString *dst, const char *src, size_t size);

// Copy/Move/Assign semantics
//
//        | src     | dst          | complexity
// Copy   | *       |  SMALL/LARGE | fixed/O(size)
// Assign | SMALL   |  SMALL       | fixed
// Assign | OFFSET  |  VIEW        | fixed
// Assign | VIEW    |  VIEW        | fixed
// Assign | LARGE   |  LARGE       | O(size)
// Move   | *       |  same as src | fixed

// Copies `src' to `dst'. `dst' will be an owned type (SMALL/LARGE). `src'
// should not point to memory owned by `dst'.
inline void TF_TString_Copy(TF_TString *dst, const char *src, size_t size);
// Assigns a `src' tstring to `dst'.  An OFFSET `src' type will yield a `VIEW'
// `dst'.  LARGE `src' types will be copied to a new buffer; all other `src'
// types will incur a fixed cost.
inline void TF_TString_Assign(TF_TString *dst, const TF_TString *src);
// Moves a `src' tstring to `dst'.  Moving a LARGE `src' to `dst' will result in
// a valid but unspecified `src'.  This function incurs a fixed cost for all
// inputs.
inline void TF_TString_Move(TF_TString *dst, TF_TString *src);

#endif  // TENSORFLOW_TSL_PLATFORM_CTSTRING_H_
