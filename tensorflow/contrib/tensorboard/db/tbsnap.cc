/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

/// \brief SQLite extension for Snappy compression
///
/// Snappy a compression library that trades ratio for speed, almost going a
/// tenth as fast as memcpy().
///
/// This extension adds the following native functions:
///
/// - snap(value: NULL|BLOB|TEXT) -> NULL|BLOB
///
///   Applies Snappy compression. If value is NULL, then NULL is returned. If
///   value is TEXT and BLOB, then it is compressed and the result is a BLOB.
///   An uncompressed byte is prepended to indicate the original type.
///
/// - unsnap(value: NULL|BLOB) -> NULL|TEXT|BLOB
///
///   Decompresses value created by snap(). If value is NULL, then NULL is
///   returned. If value is empty, then an empty blob is returned. Otherwise
///   the original type is restored from the first byte and the remaining ones
///   are decompressed.
///
/// These functions are deterministic so they can be used for all purposes,
/// including INDEX. Please note that SQLite currently does not currently
/// perform common sub-expression optimization for pure functions when
/// compiling queries.
///
/// If your SQLite environment isn't universally UTF8, please file an issue
/// with the TensorBoard team letting us know. While this implementation should
/// work, its performance could be improved to avoid superfluous TEXT coding.

#include "sqlite3ext.h"
#include "snappy.h"

namespace {
SQLITE_EXTENSION_INIT1

void snap(sqlite3_context* ctx, int argc, sqlite3_value** argv) {
  const char* data;
  int type = sqlite3_value_type(argv[0]);
  switch (type) {
    case SQLITE_NULL:
      return;
    case SQLITE_BLOB:
      data = reinterpret_cast<const char*>(sqlite3_value_blob(argv[0]));
      break;
    case SQLITE_TEXT:
      data = reinterpret_cast<const char*>(sqlite3_value_text(argv[0]));
      break;
    default:
      sqlite3_result_error(ctx, "snap() takes NULL|BLOB|TEXT", -1);
      sqlite3_result_error_code(ctx, SQLITE_MISMATCH);
      return;
  }
  int size = sqlite3_value_bytes(argv[0]);
  if (size <= 0) {
    char result[] = {static_cast<char>(type)};
    sqlite3_result_blob(ctx, result, sizeof(result), SQLITE_TRANSIENT);
    return;
  }
  size_t output_size =
      snappy::MaxCompressedLength(static_cast<size_t>(size)) + 1;
  if (output_size >
      sqlite3_limit(sqlite3_context_db_handle(ctx), SQLITE_LIMIT_LENGTH, -1)) {
    sqlite3_result_error_toobig(ctx);
    return;
  }
  char* output =
      static_cast<char*>(sqlite3_malloc(static_cast<int>(output_size)));
  if (output == nullptr) {
    sqlite3_result_error_nomem(ctx);
    return;
  }
  *output++ = static_cast<char>(type), --output_size;
  snappy::RawCompress(data, static_cast<size_t>(size), output, &output_size);
  sqlite3_result_blob(ctx, output - 1, static_cast<int>(output_size + 1),
                      sqlite3_free);
}

void unsnap(sqlite3_context* ctx, int argc, sqlite3_value** argv) {
  int type = sqlite3_value_type(argv[0]);
  if (type == SQLITE_NULL) return;
  if (type != SQLITE_BLOB) {
    sqlite3_result_error(ctx, "unsnap() takes NULL|BLOB", -1);
    sqlite3_result_error_code(ctx, SQLITE_MISMATCH);
    return;
  }
  int size = sqlite3_value_bytes(argv[0]);
  const char* blob = reinterpret_cast<const char*>(sqlite3_value_blob(argv[0]));
  if (size <= 0) {
    sqlite3_result_zeroblob(ctx, 0);
    return;
  }
  type = static_cast<int>(*blob++), --size;
  if (type != SQLITE_BLOB && type != SQLITE_TEXT) {
    sqlite3_result_error(ctx, "unsnap() first byte is invalid type", -1);
    sqlite3_result_error_code(ctx, SQLITE_CORRUPT);
    return;
  }
  if (size == 0) {
    if (type == SQLITE_TEXT) {
      sqlite3_result_text(ctx, "", 0, SQLITE_STATIC);
    } else {
      sqlite3_result_zeroblob(ctx, 0);
    }
    return;
  }
  size_t output_size;
  if (!snappy::GetUncompressedLength(blob, static_cast<size_t>(size),
                                     &output_size)) {
    sqlite3_result_error(ctx, "snappy parse error", -1);
    sqlite3_result_error_code(ctx, SQLITE_CORRUPT);
    return;
  }
  if (output_size >
      sqlite3_limit(sqlite3_context_db_handle(ctx), SQLITE_LIMIT_LENGTH, -1)) {
    sqlite3_result_error_toobig(ctx);
    return;
  }
  char* output =
      static_cast<char*>(sqlite3_malloc(static_cast<int>(output_size)));
  if (output == nullptr) {
    sqlite3_result_error_nomem(ctx);
    return;
  }
  if (!snappy::RawUncompress(blob, static_cast<size_t>(size), output)) {
    sqlite3_result_error(ctx, "snappy message corruption", -1);
    sqlite3_result_error_code(ctx, SQLITE_CORRUPT);
    return;
  }
  if (type == SQLITE_TEXT) {
    sqlite3_result_text(ctx, output, static_cast<int>(output_size),
                        sqlite3_free);
  } else {
    sqlite3_result_blob(ctx, output, static_cast<int>(output_size),
                        sqlite3_free);
  }
}

int init(sqlite3* db, const char** pzErrMsg, const sqlite3_api_routines* pApi) {
  SQLITE_EXTENSION_INIT2(pApi);
  int rc;

  rc = sqlite3_create_function_v2(
      db,
      "snap",                              // zFunctionName
      1,                                   // nArg
      SQLITE_UTF8 | SQLITE_DETERMINISTIC,  // eTextRep
      nullptr,                             // pApp
      snap,                                // xFunc
      nullptr,                             // xStep
      nullptr,                             // xFinal
      nullptr                              // xDestroy
  );
  if (rc != SQLITE_OK) {
    *pzErrMsg = "oh snap()";
    return rc;
  }

  rc = sqlite3_create_function_v2(
      db,
      "unsnap",                            // zFunctionName
      1,                                   // nArg
      SQLITE_UTF8 | SQLITE_DETERMINISTIC,  // eTextRep
      nullptr,                             // pApp
      unsnap,                              // xFunc
      nullptr,                             // xStep
      nullptr,                             // xFinal
      nullptr                              // xDestroy
  );
  if (rc != SQLITE_OK) {
    *pzErrMsg = "oh unsnap()";
    return rc;
  }

  return SQLITE_OK;
}

}  // namespace

extern "C" {

#if defined(TF_SQLITE3_AUTO_EXTENSION)
extern int sqlite3_tbsnap_status = sqlite3_auto_extension(init);
#else

#if defined(_MSC_VER) || defined(__MINGW32__)
#define EXPORT __declspec(dllexport)
#else
#define EXPORT __attribute__((visibility("default")))
#endif

// SQLite deduces this function name from "libtbsnap.so".
EXPORT extern int sqlite3_tbsnap_init(sqlite3* db, const char** pzErrMsg,
                                      const sqlite3_api_routines* pApi) {
  return init(db, pzErrMsg, pApi);
}

#endif

}  // extern "C"
