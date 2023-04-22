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
/// FUNCTIONS
///
/// - snap(value: BLOB|TEXT) -> BLOB
/// - snap(value: NULL|INT|REAL) -> value
///
///   Applies Snappy compression. If value is TEXT or BLOB, then it is
///   compressed and a BLOB is returned with a byte prepended to indicate the
///   original type. Other types are returned as-is.
///
/// - unsnap(value: BLOB) -> TEXT|BLOB
/// - unsnap(value: TEXT) -> SQLITE_MISMATCH
/// - unsnap(value: NULL|INT|REAL) -> value
///
///   Decompresses value created by snap(). If value is empty, then an empty
///   blob is returned. Otherwise the original type is restored from the first
///   byte and the remaining ones are decompressed. TEXT is not allowed as an
///   input type. Remaining types are returned as-is.
///
/// PERFORMANCE CONSIDERATIONS
///
/// These functions are deterministic. This means SQLite ≥3.8.3 will factor
/// them out of inner loops when constant arguments are provided. In SQLite
/// ≥3.15.0 they can be used in the WHERE clause of partial indexes. Currently
/// there is no support for common sub-expression elimination.
///
/// SQLite environments that aren't universally UTF8 will work, but should
/// encounter superfluous charset transcodings; as this implementation encodes
/// only UTF8 TEXT for the sake of simplicity. Contributions are welcome that
/// register multiple sister functions for the various charsets, which use the
/// higher order bits of the type byte to indicate encoding.
///
/// SUPPORT MATRIX
///
/// - 3.20.0 (2016-05-18) What FOSS TensorFlow uses
/// - 3.13.0 (2016-05-18) What Google uses c. 2017-12
/// - 3.8.2  (2013-12-06) Used by Ubuntu 14.04
///
/// MANUAL COMPILATION
///
/// $ sudo apt-get install libsqlite3-dev libsnappy-dev
/// $ c++ -shared --std=c++11 -o libsnapfn.so -fPIC snapfn.cc -lsnappy
///
/// $ sqlite3
/// sqlite> .load libsnapfn.so
/// sqlite> select hex(snap('aaaaaaaaaaaaaaaaa'));
/// 031100613E0100
/// sqlite> select unsnap(x'031100613E0100');
/// aaaaaaaaaaaaaaaaa
///
/// $ python
/// >>> import sqlite3
/// >>> db = sqlite3.connect(':memory:')
/// >>> db.enable_load_extension(True)
/// >>> db.execute('select load_extension("libsnapfn.so")')
/// >>> db.enable_load_extension(False)
/// >>> db.execute('select hex(snap("aaaaaaaaaaaaaaaaa"))').fetchone()[0]
/// u'031100613E0100'

#include "sqlite3ext.h"
#include "snappy.h"

SQLITE_EXTENSION_INIT1

static void snap(sqlite3_context* ctx, int /*argc*/, sqlite3_value** argv) {
  const char* data;
  int type = sqlite3_value_type(argv[0]);
  switch (type) {
    case SQLITE_NULL:
      return;
    case SQLITE_INTEGER:
      sqlite3_result_int64(ctx, sqlite3_value_int64(argv[0]));
      return;
    case SQLITE_FLOAT:
      sqlite3_result_double(ctx, sqlite3_value_double(argv[0]));
      return;
    case SQLITE_BLOB:
      data = reinterpret_cast<const char*>(sqlite3_value_blob(argv[0]));
      break;
    case SQLITE_TEXT:
      data = reinterpret_cast<const char*>(sqlite3_value_text(argv[0]));
      break;
    default:
      sqlite3_result_error(ctx, "snap() invalid type", -1);
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
      static_cast<size_t>(sqlite3_limit(sqlite3_context_db_handle(ctx),
                                        SQLITE_LIMIT_LENGTH, -1))) {
    sqlite3_result_error_toobig(ctx);
    return;
  }
  auto output =
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

static void unsnap(sqlite3_context* ctx, int /*argc*/, sqlite3_value** argv) {
  int type = sqlite3_value_type(argv[0]);
  switch (type) {
    case SQLITE_NULL:
      return;
    case SQLITE_INTEGER:
      sqlite3_result_int64(ctx, sqlite3_value_int64(argv[0]));
      return;
    case SQLITE_FLOAT:
      sqlite3_result_double(ctx, sqlite3_value_double(argv[0]));
      return;
    case SQLITE_BLOB:
      break;
    default:
      sqlite3_result_error(ctx, "unsnap() invalid type", -1);
      sqlite3_result_error_code(ctx, SQLITE_MISMATCH);
      return;
  }
  int size = sqlite3_value_bytes(argv[0]);
  auto blob = reinterpret_cast<const char*>(sqlite3_value_blob(argv[0]));
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
      static_cast<size_t>(sqlite3_limit(sqlite3_context_db_handle(ctx),
                                        SQLITE_LIMIT_LENGTH, -1))) {
    sqlite3_result_error_toobig(ctx);
    return;
  }
  auto output =
      static_cast<char*>(sqlite3_malloc(static_cast<int>(output_size)));
  if (output == nullptr) {
    sqlite3_result_error_nomem(ctx);
    return;
  }
  if (!snappy::RawUncompress(blob, static_cast<size_t>(size), output)) {
    sqlite3_result_error(ctx, "snappy message corruption", -1);
    sqlite3_result_error_code(ctx, SQLITE_CORRUPT);
    sqlite3_free(output);
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

extern "C" {

#ifndef SQLITE_DETERMINISTIC
#define SQLITE_DETERMINISTIC 0
#endif

#ifndef SQLITE_CALLBACK
#define SQLITE_CALLBACK
#endif

SQLITE_CALLBACK int sqlite3_snapfn_init(sqlite3* db, const char** /*pzErrMsg*/,
                                        const sqlite3_api_routines* pApi) {
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
    return rc;
  }

  return SQLITE_OK;
}

}  // extern "C"
