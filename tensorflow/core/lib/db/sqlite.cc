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

#include "absl/status/status.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/strings/stringprintf.h"

extern "C" int sqlite3_snapfn_init(sqlite3*, const char**, const void*);

namespace tensorflow {
namespace {

absl::StatusCode GetTfErrorCode(int code) {
  // See: https://sqlite.org/rescode.html
  switch (code & 0xff) {
    case SQLITE_OK:    // Successful result
    case SQLITE_ROW:   // Step has another row ready
    case SQLITE_DONE:  // Step has finished executing
      return absl::StatusCode::kOk;
    case SQLITE_ABORT:  // Callback routine requested an abort
      return absl::StatusCode::kAborted;
    case SQLITE_READONLY:  // Attempt to write a readonly database
    case SQLITE_MISMATCH:  // Data type mismatch
      return absl::StatusCode::kFailedPrecondition;
    case SQLITE_MISUSE:    // Library used incorrectly
    case SQLITE_INTERNAL:  // Internal logic error in SQLite
      return absl::StatusCode::kInternal;
    case SQLITE_RANGE:  // 2nd parameter to sqlite3_bind out of range
      return absl::StatusCode::kOutOfRange;
    case SQLITE_CANTOPEN:    // Unable to open the database file
    case SQLITE_CONSTRAINT:  // Abort due to constraint violation
    case SQLITE_NOTFOUND:    // Unknown opcode or statement parameter name
    case SQLITE_NOTADB:      // File opened that is not a database file
      return absl::StatusCode::kInvalidArgument;
    case SQLITE_CORRUPT:  // The database disk image is malformed
      return absl::StatusCode::kDataLoss;
    case SQLITE_AUTH:  // Authorization denied
    case SQLITE_PERM:  // Access permission denied
      return absl::StatusCode::kPermissionDenied;
    case SQLITE_FULL:    // Insertion failed because database is full
    case SQLITE_TOOBIG:  // String or BLOB exceeds size limit
    case SQLITE_NOLFS:   // Uses OS features not supported on host
      return absl::StatusCode::kResourceExhausted;
    case SQLITE_BUSY:      // The database file is locked
    case SQLITE_LOCKED:    // A table in the database is locked
    case SQLITE_PROTOCOL:  // Database lock protocol error
    case SQLITE_NOMEM:     // Out of heap or perhaps lookaside memory
      return absl::StatusCode::kUnavailable;
    case SQLITE_INTERRUPT:  // Operation terminated by sqlite3_interrupt
      return absl::StatusCode::kCancelled;
    case SQLITE_ERROR:   // SQL error or missing database
    case SQLITE_IOERR:   // Some kind of disk I/O error occurred
    case SQLITE_SCHEMA:  // The database schema changed
    default:
      return absl::StatusCode::kUnknown;
  }
}

template <typename... Args>
Status PrintfStatus(int rc, const char* fmt, Args&&... args) {
  return {GetTfErrorCode(rc),
          strings::Printf(fmt, std::forward<Args>(args)...)};
}

sqlite3_stmt* PrepareRawOrDie(sqlite3* db, const char* sql) {
  sqlite3_stmt* stmt = nullptr;
  int rc = sqlite3_prepare_v2(db, sql, -1, &stmt, nullptr);
  CHECK_EQ(SQLITE_OK, rc) << sql;
  return stmt;
}

Status SetPragma(Sqlite* db, const char* pragma, const StringPiece& value) {
  if (value.empty()) return OkStatus();
  for (auto p = value.begin(); p < value.end(); ++p) {
    if (!(('0' <= *p && *p <= '9') || ('A' <= *p && *p <= 'Z') ||
          ('a' <= *p && *p <= 'z') || *p == '-')) {
      return errors::InvalidArgument("Illegal pragma character");
    }
  }
  SqliteStatement stmt;
  TF_RETURN_IF_ERROR(  // We can't use Bind*() pragma statements.
      db->Prepare(strings::StrCat("PRAGMA ", pragma, "=", value), &stmt));
  bool unused_done;
  return stmt.Step(&unused_done);
}

const StringPiece GetEnv(const char* var) {
  const char* val = std::getenv(var);
  return (val == nullptr) ? StringPiece() : StringPiece(val);
}

Status EnvPragma(Sqlite* db, const char* pragma, const char* var) {
  TF_RETURN_WITH_CONTEXT_IF_ERROR(SetPragma(db, pragma, GetEnv(var)), "getenv(",
                                  var, ")");
  return OkStatus();
}

}  // namespace

/* static */
Status Sqlite::Open(const string& path, int flags, Sqlite** db) {
  flags |= SQLITE_OPEN_PRIVATECACHE;
  flags |= SQLITE_OPEN_URI;
  sqlite3* sqlite = nullptr;
  int rc = sqlite3_open_v2(path.c_str(), &sqlite, flags, nullptr);
  if (rc != SQLITE_OK) {
    *db = nullptr;
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
  *db = new Sqlite(sqlite, begin, commit, rollback);
  Status s = OkStatus();
  // Up until 2016 the default SQLite page_size was 1024. This ensures
  // the new default regardless of linkage unless configured otherwise.
  s.Update(SetPragma(*db, "page_size", "4096"));
  // TensorFlow is designed to work well in all SQLite modes. However
  // users might find tuning some these pragmas rewarding, depending on
  // various considerations. Pragmas are set on a best-effort basis and
  // might be ignored.
  s.Update(EnvPragma(*db, "secure_delete", "TF_SQLITE_SECURE_DELETE"));
  s.Update(EnvPragma(*db, "page_size", "TF_SQLITE_PAGE_SIZE"));
  s.Update(EnvPragma(*db, "journal_mode", "TF_SQLITE_JOURNAL_MODE"));
  s.Update(EnvPragma(*db, "synchronous", "TF_SQLITE_SYNCHRONOUS"));
  s.Update(EnvPragma(*db, "mmap_size", "TF_SQLITE_MMAP_SIZE"));
  s.Update(EnvPragma(*db, "locking_mode", "TF_SQLITE_LOCKING_MODE"));
  s.Update(EnvPragma(*db, "cache_size", "TF_SQLITE_CACHE_SIZE"));
  s.Update(EnvPragma(*db, "auto_vacuum", "TF_SQLITE_AUTO_VACUUM"));
  DCHECK((*db)->RefCountIsOne());
  if (!s.ok()) {
    (*db)->Unref();
    *db = nullptr;
  }
  return s;
}

Sqlite::~Sqlite() {
  sqlite3_finalize(rollback_);
  sqlite3_finalize(commit_);
  sqlite3_finalize(begin_);
  CHECK_EQ(SQLITE_OK, sqlite3_close(db_));
}

Status Sqlite::Prepare(const StringPiece& sql, SqliteStatement* stmt) {
  SqliteLock lock(*this);
  sqlite3_stmt* ps = nullptr;
  int rc = sqlite3_prepare_v2(db_, sql.data(), static_cast<int>(sql.size()),
                              &ps, nullptr);
  if (rc != SQLITE_OK) {
    *stmt = SqliteStatement();
    return PrintfStatus(rc, "Prepare() failed: [%d] %s: %.*s", rc, errmsg(),
                        sql.size(), sql.data());
  }
  *stmt = SqliteStatement(this, ps);
  return OkStatus();
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
      return OkStatus();
    case SQLITE_DONE:
      *is_done = true;
      return OkStatus();
    default:
      *is_done = true;
      return PrintfStatus(rc, "Step() failed: [%d] %s: %s", rc, db_->errmsg(),
                          sql());
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
  return OkStatus();
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
    return PrintfStatus(rc, "COMMIT failed: [%d] %s", rc,
                        sqlite3_errmsg(db_->db_));
  }
  sqlite3_reset(db_->commit_);
  sqlite3_reset(db_->begin_);
  Begin();
  return OkStatus();
}

}  // namespace tensorflow
