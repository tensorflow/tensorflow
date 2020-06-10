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

#ifndef TENSORFLOW_CORE_PLATFORM_CTSTRING_INTERNAL_H_
#define TENSORFLOW_CORE_PLATFORM_CTSTRING_INTERNAL_H_

#include <limits.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#if (defined(__BYTE_ORDER__) && defined(__ORDER_LITTLE_ENDIAN__) && \
     __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__) ||                  \
    defined(_WIN32)
#define TF_TSTRING_LITTLE_ENDIAN 1
#elif defined(__BYTE_ORDER__) && defined(__ORDER_BIG_ENDIAN__) && \
    __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
#define TF_TSTRING_LITTLE_ENDIAN 0
#else
#error "Unable to detect endianness."
#endif

#if defined(__clang__) || \
    (defined(__GNUC__) && \
     ((__GNUC__ == 4 && __GNUC_MINOR__ >= 8) || __GNUC__ >= 5))
static inline uint32_t TF_swap32(uint32_t host_int) {
  return __builtin_bswap32(host_int);
}

#elif defined(_MSC_VER)
static inline uint32_t TF_swap32(uint32_t host_int) {
  return _byteswap_ulong(host_int);
}

#elif defined(__APPLE__)
static inline uint32_t TF_swap32(uint32_t host_int) {
  return OSSwapInt32(host_int);
}

#else
static inline uint32_t TF_swap32(uint32_t host_int) {
#if defined(__GLIBC__)
  return bswap_32(host_int);
#else   // defined(__GLIBC__)
  return (((host_int & uint32_t{0xFF}) << 24) |
          ((host_int & uint32_t{0xFF00}) << 8) |
          ((host_int & uint32_t{0xFF0000}) >> 8) |
          ((host_int & uint32_t{0xFF000000}) >> 24));
#endif  // defined(__GLIBC__)
}
#endif

#if TF_TSTRING_LITTLE_ENDIAN
#define TF_le32toh(x) TF_swap32(x)
#else  // TF_TSTRING_LITTLE_ENDIAN
#define TF_le32toh(x) x
#endif  // TF_TSTRING_LITTLE_ENDIAN

static inline size_t TF_align16(size_t i) { return (i + 0xF) & ~0xF; }

static inline size_t TF_max(size_t a, size_t b) { return a > b ? a : b; }
static inline size_t TF_min(size_t a, size_t b) { return a < b ? a : b; }

typedef enum TF_TString_Type {  // NOLINT
  TF_TSTR_SMALL = 0x00,
  TF_TSTR_LARGE = 0x01,
  TF_TSTR_OFFSET = 0x02,
  TF_TSTR_VIEW = 0x03,
  TF_TSTR_TYPE_MASK = 0x03
} TF_TString_Type;

typedef struct TF_TString_Large {  // NOLINT
  size_t size;
  size_t cap;
  char *ptr;
} TF_TString_Large;

typedef struct TF_TString_Offset {  // NOLINT
  uint32_t size;
  uint32_t offset;
  uint32_t count;
} TF_TString_Offset;

typedef struct TF_TString_View {  // NOLINT
  size_t size;
  const char *ptr;
} TF_TString_View;

typedef struct TF_TString_Raw {  // NOLINT
  uint8_t raw[24];
} TF_TString_Raw;

typedef union TF_TString_Union {  // NOLINT
  TF_TString_Large large;
  TF_TString_Offset offset;
  TF_TString_View view;
  TF_TString_Raw raw;
} TF_TString_Union;

enum {
  TF_TString_SmallCapacity =
      (sizeof(TF_TString_Union) - sizeof(/* null delim */ char) -
       sizeof(/* uint8_t size */ uint8_t)),
};

typedef struct TF_TString_Small {  // NOLINT
  uint8_t size;
  char str[TF_TString_SmallCapacity + sizeof(/* null delim */ char)];
} TF_TString_Small;

typedef struct TF_TString {  // NOLINT
  union {
    // small conflicts with '#define small char' in RpcNdr.h for MSVC, so we use
    // smll instead.
    TF_TString_Small smll;
    TF_TString_Large large;
    TF_TString_Offset offset;
    TF_TString_View view;
    TF_TString_Raw raw;
  } u;
} TF_TString;

// TODO(dero): Fix for OSS, and add C only build test.
// _Static_assert(CHAR_BIT == 8);
// _Static_assert(sizeof(TF_TString) == 24);

extern inline TF_TString_Type TF_TString_GetType(const TF_TString *str) {
  return (TF_TString_Type)(str->u.raw.raw[0] & TF_TSTR_TYPE_MASK);  // NOLINT
}

// XXX(dero): For the big-endian case, this function could potentially be more
// performant and readable by always storing the string size as little-endian
// and always byte-swapping on big endian, resulting in a simple 'bswap'+'shr'
// (for architectures that have a bswap op).
static inline size_t TF_TString_ToActualSizeT(size_t size) {
#if TF_TSTRING_LITTLE_ENDIAN
  return size >> 2;
#else   // TF_TSTRING_LITTLE_ENDIAN
  // 0xFF000000 or 0xFF00000000000000 depending on platform
  static const size_t mask = ~((~(size_t)0) >> 8);

  return (((mask << 2) & size) >> 2) | (~mask & size);
#endif  // TF_TSTRING_LITTLE_ENDIAN
}

static inline size_t TF_TString_ToInternalSizeT(size_t size,
                                                TF_TString_Type type) {
#if TF_TSTRING_LITTLE_ENDIAN
  return (size << 2) | type;
#else   // TF_TSTRING_LITTLE_ENDIAN
  // 0xFF000000 or 0xFF00000000000000 depending on platform
  static const size_t mask = ~((~(size_t)0) >> 8);

  return (mask & (size << 2)) | (~mask & size) |
         ((size_t)type << ((sizeof(size_t) - 1) * 8));  // NOLINT
#endif  // TF_TSTRING_LITTLE_ENDIAN
}

extern inline void TF_TString_Init(TF_TString *str) {
  str->u.smll.size = 0;
  str->u.smll.str[0] = '\0';
}

extern inline void TF_TString_Dealloc(TF_TString *str) {
  if (TF_TString_GetType(str) == TF_TSTR_LARGE &&
      str->u.large.ptr != NULL) {  // NOLINT
    free(str->u.large.ptr);
    TF_TString_Init(str);
  }
}

extern inline size_t TF_TString_GetSize(const TF_TString *str) {
  switch (TF_TString_GetType(str)) {
    case TF_TSTR_SMALL:
      return str->u.smll.size >> 2;
    case TF_TSTR_LARGE:
      return TF_TString_ToActualSizeT(str->u.large.size);
    case TF_TSTR_OFFSET:
      return TF_le32toh(str->u.offset.size) >> 2;
    case TF_TSTR_VIEW:
      return TF_TString_ToActualSizeT(str->u.view.size);
    default:
      return 0;  // Unreachable.
  }
}

extern inline size_t TF_TString_GetCapacity(const TF_TString *str) {
  switch (TF_TString_GetType(str)) {
    case TF_TSTR_SMALL:
      return TF_TString_SmallCapacity;
    case TF_TSTR_LARGE:
      return str->u.large.cap;
    case TF_TSTR_OFFSET:
    case TF_TSTR_VIEW:
    default:
      return 0;
  }
}

extern inline const char *TF_TString_GetDataPointer(const TF_TString *str) {
  switch (TF_TString_GetType(str)) {
    case TF_TSTR_SMALL:
      return str->u.smll.str;
    case TF_TSTR_LARGE:
      return str->u.large.ptr;
    case TF_TSTR_OFFSET:
      return (const char *)str + str->u.offset.offset;  // NOLINT
    case TF_TSTR_VIEW:
      return str->u.view.ptr;
    default:
      // Unreachable.
      return NULL;  // NOLINT
  }
}

extern inline char *TF_TString_ResizeUninitialized(TF_TString *str,
                                                   size_t new_size) {
  size_t curr_size = TF_TString_GetSize(str);
  size_t copy_size = TF_min(new_size, curr_size);

  TF_TString_Type curr_type = TF_TString_GetType(str);
  const char *curr_ptr = TF_TString_GetDataPointer(str);

  // Case: SMALL/LARGE/VIEW/OFFSET -> SMALL
  if (new_size <= TF_TString_SmallCapacity) {
    str->u.smll.size = (uint8_t)((new_size << 2) | TF_TSTR_SMALL);  // NOLINT
    str->u.smll.str[new_size] = '\0';

    if (curr_type != TF_TSTR_SMALL && copy_size) {
      memcpy(str->u.smll.str, curr_ptr, copy_size);
    }

    if (curr_type == TF_TSTR_LARGE) {
      free((void *)curr_ptr);  // NOLINT
    }

    // We do not clear out the newly excluded region.

    return str->u.smll.str;
  }

  // Case: SMALL/LARGE/VIEW/OFFSET -> LARGE
  size_t new_cap;
  size_t curr_cap = TF_TString_GetCapacity(str);
  // We assume SIZE_MAX % 16 == 0.
  size_t curr_cap_x2 = curr_cap >= SIZE_MAX / 2 ? SIZE_MAX - 1 : curr_cap * 2;

  if (new_size < curr_size && new_size < curr_cap / 2) {
    // TODO(dero): Replace with shrink_to_fit flag.
    new_cap = TF_align16(curr_cap / 2 + 1) - 1;
  } else if (new_size > curr_cap_x2) {
    new_cap = TF_align16(new_size + 1) - 1;
  } else if (new_size > curr_cap) {
    new_cap = TF_align16(curr_cap_x2 + 1) - 1;
  } else {
    new_cap = curr_cap;
  }

  char *new_ptr;
  if (new_cap == curr_cap) {
    new_ptr = str->u.large.ptr;
  } else if (curr_type == TF_TSTR_LARGE) {
    new_ptr = (char *)realloc(str->u.large.ptr, new_cap + 1);  // NOLINT
  } else {
    new_ptr = (char *)malloc(new_cap + 1);  // NOLINT
    if (copy_size) {
      memcpy(new_ptr, curr_ptr, copy_size);
    }
  }

  str->u.large.size = TF_TString_ToInternalSizeT(new_size, TF_TSTR_LARGE);
  str->u.large.ptr = new_ptr;
  str->u.large.ptr[new_size] = '\0';
  str->u.large.cap = new_cap;

  return str->u.large.ptr;
}

extern inline char *TF_TString_GetMutableDataPointer(TF_TString *str) {
  switch (TF_TString_GetType(str)) {
    case TF_TSTR_SMALL:
      return str->u.smll.str;
    case TF_TSTR_OFFSET:
    case TF_TSTR_VIEW:
      // Convert OFFSET/VIEW to SMALL/LARGE
      TF_TString_ResizeUninitialized(str, TF_TString_GetSize(str));
      return (TF_TString_GetType(str) == TF_TSTR_SMALL) ? str->u.smll.str
                                                        : str->u.large.ptr;
    case TF_TSTR_LARGE:
      return str->u.large.ptr;
    default:
      // Unreachable.
      return NULL;  // NOLINT
  }
}

extern inline void TF_TString_Reserve(TF_TString *str, size_t new_cap) {
  TF_TString_Type curr_type = TF_TString_GetType(str);

  if (new_cap <= TF_TString_SmallCapacity) {
    // We do nothing, we let Resize/GetMutableDataPointer handle the
    // conversion to SMALL from VIEW/OFFSET when the need arises.
    // In the degenerate case, where new_cap <= TF_TString_SmallCapacity,
    // curr_size > TF_TString_SmallCapacity, and the type is VIEW/OFFSET, we
    // defer the malloc to Resize/GetMutableDataPointer.
    return;
  }

  if (curr_type == TF_TSTR_LARGE && new_cap <= str->u.large.cap) {
    // We handle reduced cap in resize.
    return;
  }

  // Case: VIEW/OFFSET -> LARGE or grow an existing LARGE type
  size_t curr_size = TF_TString_GetSize(str);
  const char *curr_ptr = TF_TString_GetDataPointer(str);

  // Since VIEW and OFFSET types are read-only, their capacity is effectively 0.
  // So we make sure we have enough room in the VIEW and OFFSET cases.
  new_cap = TF_align16(TF_max(new_cap, curr_size) + 1) - 1;

  if (curr_type == TF_TSTR_LARGE) {
    str->u.large.ptr =
        (char *)realloc(str->u.large.ptr, new_cap + 1);  // NOLINT
  } else {
    // Convert to Large
    char *new_ptr = (char *)malloc(new_cap + 1);  // NOLINT
    memcpy(new_ptr, curr_ptr, curr_size);

    str->u.large.size = TF_TString_ToInternalSizeT(curr_size, TF_TSTR_LARGE);
    str->u.large.ptr = new_ptr;
    str->u.large.ptr[curr_size] = '\0';
  }

  str->u.large.cap = new_cap;
}

extern inline char *TF_TString_Resize(TF_TString *str, size_t new_size,
                                      char c) {
  size_t curr_size = TF_TString_GetSize(str);
  char *cstr = TF_TString_ResizeUninitialized(str, new_size);

  if (new_size > curr_size) {
    memset(cstr + curr_size, c, new_size - curr_size);
  }

  return cstr;
}

extern inline void TF_TString_AssignView(TF_TString *dst, const char *src,
                                         size_t size) {
  TF_TString_Dealloc(dst);

  dst->u.view.size = TF_TString_ToInternalSizeT(size, TF_TSTR_VIEW);
  dst->u.view.ptr = src;
}

extern inline void TF_TString_AppendN(TF_TString *dst, const char *src,
                                      size_t src_size) {
  if (!src_size) return;

  size_t dst_size = TF_TString_GetSize(dst);

  char *dst_c = TF_TString_ResizeUninitialized(dst, dst_size + src_size);

  memcpy(dst_c + dst_size, src, src_size);
}

extern inline void TF_TString_Append(TF_TString *dst, const TF_TString *src) {
  const char *src_c = TF_TString_GetDataPointer(src);
  size_t size = TF_TString_GetSize(src);

  TF_TString_AppendN(dst, src_c, size);
}

extern inline void TF_TString_Copy(TF_TString *dst, const char *src,
                                   size_t size) {
  char *dst_c = TF_TString_ResizeUninitialized(dst, size);

  if (size) memcpy(dst_c, src, size);
}

extern inline void TF_TString_Assign(TF_TString *dst, const TF_TString *src) {
  if (dst == src) return;

  TF_TString_Dealloc(dst);

  switch (TF_TString_GetType(src)) {
    case TF_TSTR_SMALL:
    case TF_TSTR_VIEW:
      *dst = *src;
      return;
    case TF_TSTR_LARGE: {
      const char *src_c = TF_TString_GetDataPointer(src);
      size_t size = TF_TString_GetSize(src);

      TF_TString_Copy(dst, src_c, size);
    }
      return;
    case TF_TSTR_OFFSET: {
      const char *src_c = TF_TString_GetDataPointer(src);
      size_t size = TF_TString_GetSize(src);

      TF_TString_AssignView(dst, src_c, size);
    }
      return;
    default:
      return;  // Unreachable.
  }
}

extern inline void TF_TString_Move(TF_TString *dst, TF_TString *src) {
  if (dst == src) return;

  TF_TString_Dealloc(dst);

  switch (TF_TString_GetType(src)) {
    case TF_TSTR_SMALL:
    case TF_TSTR_VIEW:
      *dst = *src;
      return;
    case TF_TSTR_LARGE:
      *dst = *src;
      TF_TString_Init(src);
      return;
    case TF_TSTR_OFFSET: {
      const char *src_c = TF_TString_GetDataPointer(src);
      size_t size = TF_TString_GetSize(src);

      TF_TString_AssignView(dst, src_c, size);
    }
      return;
    default:
      return;  // Unreachable.
  }
}

#endif  // TENSORFLOW_CORE_PLATFORM_CTSTRING_INTERNAL_H_
