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

#include <mutex>

#include "sqlite3.h"
#include "tensorflow/core/lib/core/refcount.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/platform/types.h"

/// TensorFlow SQLite Veneer
///
/// - Memory safety
/// - Less boilerplate
/// - Removes deprecated stuff
/// - Pretends UTF16 doesn't exist
/// - Transaction compile-time safety
/// - Statically loads our native extensions
/// - Error reporting via tensorflow::Status et al.
///
/// SQLite>=3.8.2 needs to be supported until April 2019, which is when
/// Ubuntu 14.04 LTS becomes EOL.

namespace tensorflow {

class SqliteLock;
class SqliteStatement;
class SqliteTransaction;

/// \brief SQLite connection object.
///
/// The SQLite connection is closed automatically by the destructor.
/// Reference counting ensures that happens after its statements are
/// destructed.
///
/// Instances are reference counted and can be shared between threads.
/// This class offers the same thread safety behaviors as the SQLite
/// API itself.
///
/// This veneer uses auto-commit mode by default, which means a 4ms
/// fsync() happens after every write unless a SqliteTransaction is
/// used or WAL mode is enabled beforehand.
class TF_LOCKABLE Sqlite : public core::RefCounted {
 public:
  /// \brief Closes SQLite connection, which can take milliseconds.
  virtual ~Sqlite();

  /// \brief Opens SQLite database file.
  ///
  /// Most users will want to set flags to SQLITE_OPEN_READWRITE |
  /// SQLITE_OPEN_CREATE. There are many other open flags; here are
  /// notes on a few of them:
  ///
  /// - SQLITE_OPEN_READONLY: Allowed if no WAL journal is active.
  /// - SQLITE_OPEN_SHAREDCACHE: Will be ignored because this veneer
  ///   doesn't support the unlock notify API.
  /// - SQLITE_OPEN_NOMUTEX: Means access to this connection MUST be
  ///   serialized by the caller in accordance with the same contracts
  ///   implemented by this API.
  ///
  /// This function sets PRAGMA values from TF_SQLITE_* environment
  /// variables. See sqlite.cc to learn more.
  static Status Open(const string& path, int flags, Sqlite** db);

  /// \brief Creates SQLite statement.
  ///
  /// This routine should never fail if sql is valid and does not
  /// reference tables. When tables are referenced, system calls are
  /// needed which can take microseconds. When the schema changes, this
  /// routine will retry automatically and then possibly fail.
  ///
  /// The returned statement holds a reference to this object.
  Status Prepare(const StringPiece& sql, SqliteStatement* stmt);
  SqliteStatement PrepareOrDie(const StringPiece& sql);

  /// \brief Returns extended result code of last error.
  ///
  /// If the most recent API call was successful, the result is
  /// undefined. The legacy result code can be obtained by saying
  /// errcode() & 0xff.
  int errcode() const TF_EXCLUSIVE_LOCKS_REQUIRED(this) {
    return sqlite3_extended_errcode(db_);
  }

  /// \brief Returns pointer to current error message state.
  const char* errmsg() const TF_EXCLUSIVE_LOCKS_REQUIRED(this) {
    return sqlite3_errmsg(db_);
  }

  /// \brief Returns rowid assigned to last successful insert.
  int64 last_insert_rowid() const TF_EXCLUSIVE_LOCKS_REQUIRED(this) {
    return sqlite3_last_insert_rowid(db_);
  }

  /// \brief Returns number of rows directly changed by last write.
  int64 changes() const TF_EXCLUSIVE_LOCKS_REQUIRED(this) {
    return sqlite3_changes(db_);
  }

 private:
  friend class SqliteLock;
  friend class SqliteStatement;
  friend class SqliteTransaction;

  Sqlite(sqlite3* db, sqlite3_stmt* begin, sqlite3_stmt* commit,
         sqlite3_stmt* rollback) noexcept
      : db_(db), begin_(begin), commit_(commit), rollback_(rollback) {}

  sqlite3* const db_;
  sqlite3_stmt* const begin_;
  sqlite3_stmt* const commit_;
  sqlite3_stmt* const rollback_;
  bool is_in_transaction_ = false;

  TF_DISALLOW_COPY_AND_ASSIGN(Sqlite);
};

/// \brief SQLite prepared statement.
///
/// Instances can only be shared between threads if caller serializes
/// access from first Bind*() to *Reset().
///
/// When reusing a statement in a loop, be certain to not have jumps
/// betwixt Bind*() and *Reset().
class SqliteStatement {
 public:
  /// \brief Initializes an empty statement to be assigned later.
  SqliteStatement() noexcept = default;

  /// \brief Finalizes statement.
  ///
  /// This can take milliseconds if it was blocking the Sqlite
  /// connection object from being freed.
  ~SqliteStatement() {
    sqlite3_finalize(stmt_);
    if (db_ != nullptr) db_->Unref();
  }

  /// \brief Returns true if statement is initialized.
  explicit operator bool() const { return stmt_ != nullptr; }

  /// \brief Returns SQL text from when this query was prepared.
  const char* sql() const { return sqlite3_sql(stmt_); }

  /// \brief Number of bytes bound since last *Reset().
  uint64 size() { return size_; }

  /// \brief Executes query for fetching arbitrary rows.
  ///
  /// `is_done` will always be set to true unless SQLITE_ROW is
  /// returned by the underlying API. If status() is already in an
  /// error state, then this method is a no-op and the existing status
  /// is returned.
  ///
  /// The OrDie version returns `!is_done` which, if true, indicates a
  /// row is available.
  ///
  /// This statement should be Reset() or destructed when finished with
  /// the result.
  Status Step(bool* is_done);
  bool StepOrDie() TF_MUST_USE_RESULT;

  /// \brief Executes query when only one row is desired.
  ///
  /// If a row isn't returned, an internal error Status is returned
  /// that won't be reflected in the connection error state.
  ///
  /// This statement should be Reset() or destructed when finished with
  /// the result.
  Status StepOnce();
  const SqliteStatement& StepOnceOrDie();

  /// \brief Executes query, ensures zero rows returned, then Reset().
  ///
  /// If a row is returned, an internal error Status is returned that
  /// won't be reflected in the connection error state.
  Status StepAndReset();
  void StepAndResetOrDie();

  /// \brief Resets statement so it can be executed again.
  ///
  /// Implementation note: This method diverges from canonical API
  /// behavior by calling sqlite3_clear_bindings() in addition to
  /// sqlite3_reset(). That makes the veneer safer; we haven't found a
  /// super compelling reason yet to call them independently.
  void Reset();

  /// \brief Binds signed 64-bit integer to 1-indexed query parameter.
  void BindInt(int parameter, int64 value) {
    Update(sqlite3_bind_int64(stmt_, parameter, value), parameter);
    size_ += sizeof(int64);
  }
  void BindInt(const char* parameter, int64 value) {
    BindInt(GetParameterIndex(parameter), value);
  }

  /// \brief Binds double to 1-indexed query parameter.
  void BindDouble(int parameter, double value) {
    Update(sqlite3_bind_double(stmt_, parameter, value), parameter);
    size_ += sizeof(double);
  }
  void BindDouble(const char* parameter, double value) {
    BindDouble(GetParameterIndex(parameter), value);
  }

  /// \brief Copies UTF-8 text to 1-indexed query parameter.
  ///
  /// If NUL characters are present, they will still go in the DB and
  /// be successfully retrieved by ColumnString(); however, the
  /// behavior of these values with SQLite functions is undefined.
  ///
  /// When using the unsafe methods, the data must not be changed or
  /// freed until this statement is Reset() or finalized.
  void BindText(int parameter, const StringPiece& text) {
    Update(sqlite3_bind_text64(stmt_, parameter, text.data(), text.size(),
                               SQLITE_TRANSIENT, SQLITE_UTF8),
           parameter);
    size_ += text.size();
  }
  void BindText(const char* parameter, const StringPiece& text) {
    BindText(GetParameterIndex(parameter), text);
  }
  void BindTextUnsafe(int parameter, const StringPiece& text) {
    Update(sqlite3_bind_text64(stmt_, parameter, text.data(), text.size(),
                               SQLITE_STATIC, SQLITE_UTF8),
           parameter);
    size_ += text.size();
  }
  void BindTextUnsafe(const char* parameter, const StringPiece& text) {
    BindTextUnsafe(GetParameterIndex(parameter), text);
  }

  /// \brief Copies binary data to 1-indexed query parameter.
  ///
  /// When using the unsafe methods, the data must not be changed or
  /// freed until this statement is Reset() or finalized.
  void BindBlob(int parameter, const StringPiece& blob) {
    Update(sqlite3_bind_blob64(stmt_, parameter, blob.data(), blob.size(),
                               SQLITE_TRANSIENT),
           parameter);
    size_ += blob.size();
  }
  void BindBlob(const char* parameter, const StringPiece& blob) {
    BindBlob(GetParameterIndex(parameter), blob);
  }
  void BindBlobUnsafe(int parameter, const StringPiece& blob) {
    Update(sqlite3_bind_blob64(stmt_, parameter, blob.data(), blob.size(),
                               SQLITE_STATIC),
           parameter);
    size_ += blob.size();
  }
  void BindBlobUnsafe(const char* parameter, const StringPiece& text) {
    BindBlobUnsafe(GetParameterIndex(parameter), text);
  }

  /// \brief Returns number of columns in result set.
  int ColumnCount() const TF_MUST_USE_RESULT {
    return sqlite3_column_count(stmt_);
  }

  /// \brief Returns type of 0-indexed column value in row data.
  ///
  /// Please note that SQLite is dynamically typed and the type of a
  /// particular column can vary from row to row.
  int ColumnType(int column) const TF_MUST_USE_RESULT {
    return sqlite3_column_type(stmt_, column);
  }

  /// \brief Returns 0-indexed column from row result coerced as an integer.
  int64 ColumnInt(int column) const TF_MUST_USE_RESULT {
    return sqlite3_column_int64(stmt_, column);
  }

  /// \brief Returns 0-indexed column from row result coerced as a double.
  double ColumnDouble(int column) const TF_MUST_USE_RESULT {
    return sqlite3_column_double(stmt_, column);
  }

  /// \brief Copies 0-indexed column from row result coerced as a string.
  ///
  /// NULL values are returned as empty string. This method should be
  /// used for both BLOB and TEXT columns. See also: ColumnType().
  string ColumnString(int column) const TF_MUST_USE_RESULT {
    auto data = sqlite3_column_blob(stmt_, column);
    if (data == nullptr) return "";
    return {static_cast<const char*>(data),
            static_cast<size_t>(ColumnSize(column))};
  }

  /// \brief Returns pointer to binary data at 0-indexed column.
  ///
  /// Empty values are returned as NULL. The returned memory will no
  /// longer be valid the next time Step() or Reset() is called. No NUL
  /// terminator is added.
  StringPiece ColumnStringUnsafe(int column) const TF_MUST_USE_RESULT {
    return {static_cast<const char*>(sqlite3_column_blob(stmt_, column)),
            static_cast<size_t>(ColumnSize(column))};
  }

  /// \brief Returns number of bytes stored at 0-indexed column.
  int ColumnSize(int column) const TF_MUST_USE_RESULT {
    return sqlite3_column_bytes(stmt_, column);
  }

  /// \brief Move constructor, after which <other> is reset to empty.
  SqliteStatement(SqliteStatement&& other) noexcept
      : db_(other.db_), stmt_(other.stmt_), bind_error_(other.bind_error_) {
    other.db_ = nullptr;
    other.stmt_ = nullptr;
    other.bind_error_ = SQLITE_OK;
  }

  /// \brief Move assignment, after which <other> is reset to empty.
  SqliteStatement& operator=(SqliteStatement&& other) noexcept {
    if (&other != this) {
      if (db_ != nullptr) db_->Unref();
      if (stmt_ != nullptr) sqlite3_finalize(stmt_);
      db_ = other.db_;
      stmt_ = other.stmt_;
      bind_error_ = other.bind_error_;
      size_ = other.size_;
      other.db_ = nullptr;
      other.stmt_ = nullptr;
      other.bind_error_ = SQLITE_OK;
      other.size_ = 0;
    }
    return *this;
  }

 private:
  friend class Sqlite;

  SqliteStatement(Sqlite* db, sqlite3_stmt* stmt) noexcept
      : db_(db), stmt_(stmt) {
    db_->Ref();
  }

  void Update(int rc, int parameter) {
    // Binding strings can fail if they exceed length limit.
    if (TF_PREDICT_FALSE(rc != SQLITE_OK)) {
      if (bind_error_ == SQLITE_OK) {
        bind_error_ = rc;
        bind_error_parameter_ = parameter;
      }
    }
  }

  int GetParameterIndex(const char* parameter) {
    int index = sqlite3_bind_parameter_index(stmt_, parameter);
    DCHECK(index > 0);  // OK to compile away since it'll fail again
    return index;
  }

  Sqlite* db_ = nullptr;
  sqlite3_stmt* stmt_ = nullptr;
  int bind_error_ = SQLITE_OK;
  int bind_error_parameter_ = 0;
  uint64 size_ = 0;

  TF_DISALLOW_COPY_AND_ASSIGN(SqliteStatement);
};

/// \brief Reentrant SQLite connection object lock
///
/// This is a no-op if SQLITE_OPEN_NOMUTEX was used.
class TF_SCOPED_LOCKABLE SqliteLock {
 public:
  explicit SqliteLock(Sqlite& db) TF_EXCLUSIVE_LOCK_FUNCTION(db)
      : mutex_(sqlite3_db_mutex(db.db_)) {
    sqlite3_mutex_enter(mutex_);
  }
  SqliteLock(Sqlite& db, std::try_to_lock_t) TF_EXCLUSIVE_LOCK_FUNCTION(db)
      : mutex_(sqlite3_db_mutex(db.db_)) {
    if (TF_PREDICT_FALSE(sqlite3_mutex_try(mutex_) != SQLITE_OK)) {
      is_locked_ = false;
    }
  }
  ~SqliteLock() TF_UNLOCK_FUNCTION() {
    if (is_locked_) sqlite3_mutex_leave(mutex_);
  }
  explicit operator bool() const { return is_locked_; }

 private:
  sqlite3_mutex* const mutex_;
  bool is_locked_ = true;
  TF_DISALLOW_COPY_AND_ASSIGN(SqliteLock);
};
#define SqliteLock(x) static_assert(0, "sqlite_lock_decl_missing_name");

/// \brief SQLite transaction scope.
///
/// This class acquires an exclusive lock on the connection object (if
/// mutexes weren't disabled) and runs BEGIN / ROLLBACK automatically.
/// Unlike SqliteLock this scope is non-reentrant. To avoid program
/// crashes, business logic should use the TF_EXCLUSIVE_LOCK_FUNCTION and
/// TF_LOCKS_EXCLUDED annotations as much as possible.
class TF_SCOPED_LOCKABLE SqliteTransaction {
 public:
  /// \brief Locks db and begins deferred transaction.
  ///
  /// This will crash if a transaction is already active.
  explicit SqliteTransaction(Sqlite& db) TF_EXCLUSIVE_LOCK_FUNCTION(db);

  /// \brief Runs ROLLBACK and unlocks.
  ~SqliteTransaction() TF_UNLOCK_FUNCTION();

  /// \brief Commits transaction.
  ///
  /// If this is successful, a new transaction will be started, which
  /// is rolled back when exiting the scope.
  Status Commit();

 private:
  void Begin();
  Sqlite* const db_;

  TF_DISALLOW_COPY_AND_ASSIGN(SqliteTransaction);
};

#define SQLITE_EXCLUSIVE_TRANSACTIONS_REQUIRED(...) \
  TF_EXCLUSIVE_LOCKS_REQUIRED(__VA_ARGS__)
#define SQLITE_TRANSACTIONS_EXCLUDED(...) TF_LOCKS_EXCLUDED(__VA_ARGS__)

inline SqliteStatement Sqlite::PrepareOrDie(const StringPiece& sql) {
  SqliteStatement stmt;
  TF_CHECK_OK(Prepare(sql, &stmt));
  return stmt;
}

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_LIB_DB_SQLITE_H_
