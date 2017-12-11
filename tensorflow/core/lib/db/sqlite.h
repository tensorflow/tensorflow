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
#ifndef TENSORFLOW_CORE_LIB_DB_SQLITE_H_
#define TENSORFLOW_CORE_LIB_DB_SQLITE_H_

#include <cstddef>
#include <memory>
#include <utility>

#include "sqlite3.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

class SqliteStatement;

/// \brief SQLite connection object.
///
/// This class is a thin wrapper around `sqlite3` that makes it easier
/// and safer to use SQLite in the TensorFlow C++ codebase. It removes
/// deprecated APIs, improves the safety of others, adds helpers, and
/// pretends UTF16 doesn't exist.
///
/// Instances are thread safe, with the exception of Close().
class Sqlite {
 public:
  /// \brief Opens SQLite database file.
  ///
  /// The `uri` parameter can be a filename, or a proper URI like
  /// `file:/tmp/tf.sqlite?mode=ro&cache=private`. It can also be
  /// `file::memory:` for testing.
  ///
  /// See https://sqlite.org/c3ref/open.html
  static xla::StatusOr<std::shared_ptr<Sqlite>> Open(const string& uri);

  /// \brief Makes tensorflow::Status for SQLite result code.
  ///
  /// See https://sqlite.org/rescode.html
  static Status MakeStatus(int resultCode);

  /// \brief Destroys object and frees resources.
  ///
  /// This will free the underlying object if Close was not called. If
  /// an error code is returned then it will be logged.
  ///
  /// Note: Unlike Close() this destructor maps to sqlite3_close_v2(),
  /// which is lax about ordering and GC friendly.
  ~Sqlite();

  /// \brief Frees underlying SQLite object.
  ///
  /// Unlike the destructor, all SqliteStatement objects must be closed
  /// beforehand. This is a no-op if already closed
  Status Close();

  /// \brief Enables WAL mode with less fsync or log a warning.
  ///
  /// The synchronous pragma is only set to NORMAL if WAL mode was
  /// successfully enabled. This must be called immediately after
  /// creating the object.
  void UseWriteAheadLogWithReducedDurabilityIfPossible();

  /// \brief Creates SQLite statement.
  ///
  /// Call result.status() to determine whether or not this operation
  /// failed. It is also possible to punt the error checking to after
  /// the values have been binded and Step() or ExecuteWriteQuery() is
  /// called.
  SqliteStatement Prepare(const string& sql);

 private:
  explicit Sqlite(sqlite3* db, const string& uri);
  sqlite3* db_;
  string uri_;
  TF_DISALLOW_COPY_AND_ASSIGN(Sqlite);
};

/// \brief SQLite prepared statement cursor object.
///
/// This class tracks error state internally, like Status::Update.
///
/// Instances of this class are not thread safe.
class SqliteStatement {
 public:
  /// \brief Constructs empty statement that should be assigned later.
  SqliteStatement() : stmt_(nullptr), error_(SQLITE_OK) {}

  /// \brief Empties object and finalizes statement if needed.
  ~SqliteStatement() { CloseOrLog(); }

  /// \brief Move constructor, after which <other> should not be used.
  SqliteStatement(SqliteStatement&& other);

  /// \brief Move assignment, after which <other> should not be used.
  SqliteStatement& operator=(SqliteStatement&& other);

  /// \brief Returns true if statement is not empty.
  explicit operator bool() const { return stmt_ != nullptr; }

  /// \brief Returns SQLite result code state.
  ///
  /// This will be SQLITE_OK unless an error happened. If multiple
  /// errors happened, only the first error code will be returned.
  int error() const { return error_; }

  /// \brief Returns error() as a tensorflow::Status.
  Status status() const;

  /// \brief Finalize statement object.
  ///
  /// Please note that the destructor can also do this. This method is
  /// a no-op if already closed.
  Status Close();

  /// \brief Executes query and/or fetches next row.
  ///
  /// `isDone` will always be set to true unless SQLITE_ROW is returned
  /// by the underlying API. If status() is already in an error state,
  /// then this method is a no-op and the existing status is returned.
  Status Step(bool* isDone);

  /// \brief Executes query that returns no data.
  ///
  /// This helper calls Step(), ensures SQLITE_DONE was returned, then
  /// resets the statement and clears the bindings. If status() is
  /// already in an error state, then this method is a no-op and the
  /// existing status is returned.
  Status StepAndReset();

  /// \brief Resets statement so it can be executed again.
  ///
  /// - Resets the prepared statement
  /// - Sets all Bind*() values to NULL
  ///
  /// Support for calling sqlite3_reset() and sqlite3_clear_bindings()
  /// independently may be added in the future if a compelling use case
  /// can be demonstrated.
  void Reset();

  /// \brief Binds signed 64-bit integer to 1-indexed query parameter.
  void BindInt(int parameter, int64 value) {
    Update(sqlite3_bind_int64(stmt_, parameter, value));
  }
  void BindInt(const string& parameter, int64 value) {
    BindInt(GetParameterIndex(parameter), value);
  }

  /// \brief Binds double to 1-indexed query parameter.
  void BindDouble(int parameter, double value) {
    Update(sqlite3_bind_double(stmt_, parameter, value));
  }
  void BindDouble(const string& parameter, double value) {
    BindDouble(GetParameterIndex(parameter), value);
  }

  /// \brief Copies UTF-8 text to 1-indexed query parameter.
  ///
  /// If NUL characters are present, they will still go in the DB and
  /// be successfully retrieved by ColumnString(); however, the
  /// behavior of these values with SQLite functions is undefined.
  void BindText(int parameter, const string& text) {
    Update(sqlite3_bind_text64(stmt_, parameter, text.data(), text.size(),
                               SQLITE_TRANSIENT, SQLITE_UTF8));
  }
  void BindText(const string& parameter, const string& text) {
    BindText(GetParameterIndex(parameter), text);
  }

  /// \brief Copies binary data to 1-indexed query parameter.
  void BindBlob(int parameter, const string& blob) {
    Update(sqlite3_bind_blob64(stmt_, parameter, blob.data(), blob.size(),
                               SQLITE_TRANSIENT));
  }
  void BindBlob(const string& parameter, const string& blob) {
    BindBlob(GetParameterIndex(parameter), blob);
  }

  /// \brief Binds UTF-8 text to 1-indexed query parameter.
  ///
  /// The contents of `text` must not be changed or freed until Reset()
  /// or Close() is called.
  ///
  /// If NUL characters are present, they will still go in the DB and
  /// be successfully retrieved by ColumnString(); however, the
  /// behavior of these values with SQLite functions is undefined.
  void BindTextUnsafe(int parameter, const string& text) {
    Update(sqlite3_bind_text64(stmt_, parameter, text.data(), text.size(),
                               SQLITE_STATIC, SQLITE_UTF8));
  }
  void BindTextUnsafe(const string& parameter, const string& text) {
    BindTextUnsafe(GetParameterIndex(parameter), text);
  }

  /// \brief Binds binary data to 1-indexed query parameter.
  ///
  /// The contents of `blob` must not be changed or freed until Reset()
  /// or Close() is called.
  void BindBlobUnsafe(int parameter, const string& blob) {
    Update(sqlite3_bind_blob64(stmt_, parameter, blob.data(), blob.size(),
                               SQLITE_STATIC));
  }
  void BindBlobUnsafe(const string& parameter, const string& text) {
    BindBlobUnsafe(GetParameterIndex(parameter), text);
  }

  /// \brief Returns number of columns in result set.
  int ColumnCount() TF_MUST_USE_RESULT { return sqlite3_column_count(stmt_); }

  /// \brief Returns type of 0-indexed column value in row data.
  ///
  /// Please note that SQLite is dynamically typed and the type of a
  /// particular column can vary from row to row.
  int ColumnType(int column) TF_MUST_USE_RESULT {
    return sqlite3_column_type(stmt_, column);
  }

  /// \brief Returns 0-indexed column from row result coerced as an integer.
  int64 ColumnInt(int column) TF_MUST_USE_RESULT {
    return sqlite3_column_int64(stmt_, column);
  }

  /// \brief Returns 0-indexed column from row result coerced as a double.
  double ColumnDouble(int column) TF_MUST_USE_RESULT {
    return sqlite3_column_double(stmt_, column);
  }

  /// \brief Copies 0-indexed column from row result coerced as a string.
  ///
  /// NULL values are returned as empty string. This method should be
  /// used for both BLOB and TEXT columns. See also: ColumnType().
  string ColumnString(int column) TF_MUST_USE_RESULT {
    auto data = sqlite3_column_blob(stmt_, column);
    if (data == nullptr) {
      return "";
    }
    return {static_cast<const char*>(data),
            static_cast<size_t>(ColumnSize(column))};
  }

  /// \brief Returns pointer to binary data at 0-indexed column.
  ///
  /// The returned memory will be mutated or freed the next time
  /// Step() or Reset() is called. No NUL terminator is added. See
  /// ColumnSize(). Please note that an empty BLOB is NULL.
  const char* ColumnStringUnsafe(int column) TF_MUST_USE_RESULT {
    return static_cast<const char*>(sqlite3_column_blob(stmt_, column));
  }

  /// \brief Returns number of bytes stored at 0-indexed column.
  int ColumnSize(int column) TF_MUST_USE_RESULT {
    return sqlite3_column_bytes(stmt_, column);
  }

 private:
  friend Sqlite;
  SqliteStatement(sqlite3_stmt* stmt, int error,
                  std::unique_ptr<string> prepare_error_sql)
      : stmt_(stmt),
        error_(error),
        prepare_error_sql_(std::move(prepare_error_sql)) {}
  void CloseOrLog();

  void Update(int rc) {
    if (TF_PREDICT_FALSE(rc != SQLITE_OK)) {
      if (error_ == SQLITE_OK) {
        error_ = rc;
      }
    }
  }

  int GetParameterIndex(const string& parameter) {
    // Each call to this function requires O(n) strncmp().
    int index = sqlite3_bind_parameter_index(stmt_, parameter.c_str());
    if (TF_PREDICT_FALSE(index == 0)) {
      Update(SQLITE_NOTFOUND);
    }
    return index;
  }

  sqlite3_stmt* stmt_;
  int error_;
  std::unique_ptr<string> prepare_error_sql_;

  TF_DISALLOW_COPY_AND_ASSIGN(SqliteStatement);
};

inline SqliteStatement::SqliteStatement(SqliteStatement&& other)
    : stmt_(other.stmt_),
      error_(other.error_),
      prepare_error_sql_(std::move(other.prepare_error_sql_)) {
  other.stmt_ = nullptr;
  other.error_ = SQLITE_OK;
}

inline SqliteStatement& SqliteStatement::operator=(SqliteStatement&& other) {
  if (&other != this) {
    CloseOrLog();
    stmt_ = other.stmt_;
    error_ = other.error_;
    prepare_error_sql_ = std::move(other.prepare_error_sql_);
    other.stmt_ = nullptr;
    other.error_ = SQLITE_OK;
  }
  return *this;
}

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_LIB_DB_SQLITE_H_
