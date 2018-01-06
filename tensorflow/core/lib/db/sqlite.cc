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
#include "tensorflow/core/lib/db/sqlite.h"

#include "tensorflow/core/lib/strings/stringprintf.h"

extern "C" int sqlite3_snapfn_init(sqlite3*, const char**, const void*);

namespace tensorflow {
namespace {

error::Code GetTfErrorCode(int code) {
  // See: https://sqlite.org/rescode.html
  switch (code & 0xff) {
    case SQLITE_OK:    // Successful result
    case SQLITE_ROW:   // Step has another row ready
    case SQLITE_DONE:  // Step has finished executing
      return error::OK;
    case SQLITE_ABORT:  // Callback routine requested an abort
      return error::ABORTED;
    case SQLITE_READONLY:  // Attempt to write a readonly database
    case SQLITE_MISMATCH:  // Data type mismatch
      return error::FAILED_PRECONDITION;
    case SQLITE_MISUSE:    // Library used incorrectly
    case SQLITE_INTERNAL:  // Internal logic error in SQLite
      return error::INTERNAL;
    case SQLITE_RANGE:  // 2nd parameter to sqlite3_bind out of range
      return error::OUT_OF_RANGE;
    case SQLITE_CANTOPEN:    // Unable to open the database file
    case SQLITE_CONSTRAINT:  // Abort due to constraint violation
    case SQLITE_NOTFOUND:    // Unknown opcode or statement parameter name
    case SQLITE_NOTADB:      // File opened that is not a database file
      return error::INVALID_ARGUMENT;
    case SQLITE_CORRUPT:  // The database disk image is malformed
      return error::DATA_LOSS;
    case SQLITE_AUTH:  // Authorization denied
    case SQLITE_PERM:  // Access permission denied
      return error::PERMISSION_DENIED;
    case SQLITE_FULL:    // Insertion failed because database is full
    case SQLITE_TOOBIG:  // String or BLOB exceeds size limit
    case SQLITE_NOLFS:   // Uses OS features not supported on host
      return error::RESOURCE_EXHAUSTED;
    case SQLITE_BUSY:      // The database file is locked
    case SQLITE_LOCKED:    // A table in the database is locked
    case SQLITE_PROTOCOL:  // Database lock protocol error
    case SQLITE_NOMEM:     // Out of heap or perhaps lookaside memory
      return error::UNAVAILABLE;
    case SQLITE_INTERRUPT:  // Operation terminated by sqlite3_interrupt
      return error::CANCELLED;
    case SQLITE_ERROR:   // SQL error or missing database
    case SQLITE_IOERR:   // Some kind of disk I/O error occurred
    case SQLITE_SCHEMA:  // The database schema changed
    default:
      return error::UNKNOWN;
  }
}

template <typename... Args>
Status PrintfStatus(int rc, const char* fmt, Args&&... args) {
  return {GetTfErrorCode(rc),
          strings::Printf(fmt, std::forward<Args>(args)...)};
}

Status AsStatus(Sqlite* db, int rc) EXCLUSIVE_LOCKS_REQUIRED(*db) {
  if (TF_PREDICT_TRUE(rc == SQLITE_OK)) return Status::OK();
  return {GetTfErrorCode(rc), db->errmsg()};
}

sqlite3_stmt* PrepareRawOrDie(sqlite3* db, const char* sql) {
  sqlite3_stmt* stmt = nullptr;
  int rc = sqlite3_prepare_v2(db, sql, -1, &stmt, nullptr);
  CHECK_EQ(SQLITE_OK, rc) << sql;
  return stmt;
}

Status SetEnvPragmaActual(Sqlite* db, const char* pragma, const char* var) {
  const char* value = std::getenv(var);
  if (value == nullptr || *value == '\0') return Status::OK();
  for (const char* p = value; *p != '\0'; ++p) {
    if (!(('0' <= *p && *p <= '9') || *p == '-' ||
          ('A' <= *p && *p <= 'Z') ||
          ('a' <= *p && *p <= 'z'))) {
      return errors::InvalidArgument("Illegal character");
    }
  }
  // We can't use Bind*() for pragmas.
  auto stmt = db->Prepare(strings::StrCat("PRAGMA ", pragma, "=", value));
  TF_RETURN_IF_ERROR(stmt.status());
  bool unused_done;
  return stmt.ValueOrDie().Step(&unused_done);
}

Status EnvPragma(Sqlite* db, const char* pragma, const char* var) {
  TF_RETURN_WITH_CONTEXT_IF_ERROR(SetEnvPragmaActual(db, pragma, var),
                                  "getenv(", var, ")");
  return Status::OK();
}

}  // namespace

/* static */
xla::StatusOr<std::shared_ptr<Sqlite>> Sqlite::Open(string path, int flags) {
  flags |= SQLITE_OPEN_PRIVATECACHE;
  sqlite3* sqlite = nullptr;
  int rc = sqlite3_open_v2(path.c_str(), &sqlite, flags, nullptr);
  if (rc != SQLITE_OK) {
    return PrintfStatus(rc, "Sqlite::Open(%s) failed: %s", path.c_str(),
                        sqlite3_errstr(rc));
  }
  CHECK_EQ(SQLITE_OK, sqlite3_extended_result_codes(sqlite, 1));
  CHECK_EQ(SQLITE_OK, sqlite3_snapfn_init(sqlite, nullptr, nullptr));
  // Prepare these tiny privileged statements for SqliteTransaction
  // so it can do less work, particularly in its constructor, per
  // Google C++ Style.
  sqlite3_stmt* begin = PrepareRawOrDie(sqlite, "BEGIN");
  sqlite3_stmt* commit = PrepareRawOrDie(sqlite, "COMMIT");
  sqlite3_stmt* rollback = PrepareRawOrDie(sqlite, "ROLLBACK");
  auto r = std::shared_ptr<Sqlite>(
      new Sqlite(sqlite, std::move(path), begin, commit, rollback));
  r->self_ = std::weak_ptr<Sqlite>(r);
  Sqlite* db = r.get();
  // TensorFlow is designed to work well in all SQLite modes. However
  // users might find tuning some these pragmas rewarding, depending on
  // various considerations.
  TF_RETURN_IF_ERROR(EnvPragma(db, "secure_delete", "TF_SQLITE_SECURE_DELETE"));
  TF_RETURN_IF_ERROR(EnvPragma(db, "page_size", "TF_SQLITE_PAGE_SIZE"));
  TF_RETURN_IF_ERROR(EnvPragma(db, "journal_mode", "TF_SQLITE_JOURNAL_MODE"));
  TF_RETURN_IF_ERROR(EnvPragma(db, "synchronous", "TF_SQLITE_SYNCHRONOUS"));
  TF_RETURN_IF_ERROR(EnvPragma(db, "mmap_size", "TF_SQLITE_MMAP_SIZE"));
  TF_RETURN_IF_ERROR(EnvPragma(db, "locking_mode", "TF_SQLITE_LOCKING_MODE"));
  TF_RETURN_IF_ERROR(EnvPragma(db, "cache_size", "TF_SQLITE_CACHE_SIZE"));
  TF_RETURN_IF_ERROR(EnvPragma(db, "auto_vacuum", "TF_SQLITE_AUTO_VACUUM"));
  return r;
}

Sqlite::~Sqlite() {
  sqlite3_finalize(rollback_);
  sqlite3_finalize(commit_);
  sqlite3_finalize(begin_);
  CHECK_EQ(SQLITE_OK, sqlite3_close(db_));
}

xla::StatusOr<SqliteStatement> Sqlite::Prepare(const StringPiece& sql) {
  SqliteLock lock(*this);
  sqlite3_stmt* stmt = nullptr;
  int rc = sqlite3_prepare_v2(db_, sql.data(), static_cast<int>(sql.size()),
                              &stmt, nullptr);
  if (rc != SQLITE_OK) {
    return PrintfStatus(rc, "Prepare() failed: %s: %.*s", errmsg(), sql.size(),
                        sql.data());
  }
  return SqliteStatement(stmt, self_.lock());
}

Status SqliteStatement::Step(bool* is_done) {
  DCHECK(stmt_ != nullptr);
  if (TF_PREDICT_FALSE(bind_error_ != SQLITE_OK)) {
    *is_done = true;
    return PrintfStatus(bind_error_, "Bind(%d) failed: %s: %s",
                        bind_error_parameter_, sqlite3_errstr(bind_error_),
                        sql());
  }
  SqliteLock lock(*db_);
  int rc = sqlite3_step(stmt_);
  switch (rc) {
    case SQLITE_ROW:
      *is_done = false;
      return Status::OK();
    case SQLITE_DONE:
      *is_done = true;
      return Status::OK();
    default:
      *is_done = true;
      return PrintfStatus(rc, "Step() failed: %s: %s", db_->errmsg(), sql());
  }
}

bool SqliteStatement::StepOrDie() {
  bool is_done;
  TF_CHECK_OK(Step(&is_done));
  return !is_done;
}

Status SqliteStatement::StepOnce() {
  bool is_done;
  TF_RETURN_IF_ERROR(Step(&is_done));
  if (TF_PREDICT_FALSE(is_done)) {
    return errors::Internal("No rows returned: ", sql());
  }
  return Status::OK();
}

const SqliteStatement& SqliteStatement::StepOnceOrDie() {
  TF_CHECK_OK(StepOnce());
  return *this;
}

Status SqliteStatement::StepAndReset() {
  bool is_done;
  Status s = Step(&is_done);
  if (TF_PREDICT_FALSE(s.ok() && !is_done)) {
    s = errors::Internal("Unexpected row: ", sql());
  }
  Reset();
  return s;
}

void SqliteStatement::StepAndResetOrDie() { TF_CHECK_OK(StepAndReset()); }

void SqliteStatement::Reset() {
  if (TF_PREDICT_TRUE(stmt_ != nullptr)) {
    sqlite3_reset(stmt_);
    sqlite3_clear_bindings(stmt_);
  }
  bind_error_ = SQLITE_OK;
  size_ = 0;
}

SqliteTransaction::SqliteTransaction(Sqlite& db) : db_(&db) {
  sqlite3_mutex_enter(sqlite3_db_mutex(db_->db_));
  CHECK(!db_->is_in_transaction_);
  db_->is_in_transaction_ = true;
  Begin();
}

SqliteTransaction::~SqliteTransaction() {
  // Rollback should only return an error if there's no transaction.
  // Since the API performs auto-rollbacks in some cases, we ignore.
  sqlite3_step(db_->rollback_);
  sqlite3_reset(db_->rollback_);
  sqlite3_reset(db_->begin_);
  db_->is_in_transaction_ = false;
  sqlite3_mutex_leave(sqlite3_db_mutex(db_->db_));
}

void SqliteTransaction::Begin() {
  // This shouldn't allocate memory or perform I/O. All it does is
  // execute OP_AutoCommit(0, 0) a.k.a. BEGIN DEFERRED which flips
  // the sqlite3::autoCommit bit.
  if (sqlite3_step(db_->begin_) != SQLITE_DONE) {
    // It shouldn't be possible for this to fail since we already
    // performed the reentrancy check.
    LOG(FATAL) << "BEGIN failed: " << sqlite3_errmsg(db_->db_);
  }
}

Status SqliteTransaction::Commit() {
  int rc = sqlite3_step(db_->commit_);
  if (rc != SQLITE_DONE) {
    return PrintfStatus(rc, "COMMIT failed: %s", sqlite3_errmsg(db_->db_));
  }
  sqlite3_reset(db_->commit_);
  sqlite3_reset(db_->begin_);
  Begin();
  return Status::OK();
}

}  // namespace tensorflow
