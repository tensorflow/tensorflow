/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "lmdb.h"
#include "tensorflow/core/framework/reader_op_kernel.h"
#include "tensorflow/core/framework/reader_base.h"
#include "tensorflow/core/lib/core/errors.h"

#include <sys/stat.h>

namespace tensorflow {

inline void MDB_CHECK(int mdb_status) {
  CHECK_EQ(mdb_status, MDB_SUCCESS) << mdb_strerror(mdb_status);
}

class LMDBReader : public ReaderBase {
 public:
  LMDBReader(const string& node_name, Env* env)
      : ReaderBase(strings::StrCat("LMDBReader '", node_name, "'")),
        env_(env),
        mdb_env_(nullptr),
        mdb_dbi_(0),
        mdb_txn_(nullptr),
        mdb_cursor_(nullptr) {}

  Status OnWorkStartedLocked() override {
    MDB_CHECK(mdb_env_create(&mdb_env_));
    int flags = MDB_RDONLY | MDB_NOTLS;

    // Check if the LMDB filename is actually a file instead of a directory.
    // If so, set appropriate flags so we can open it.
    struct stat source_stat;
    if (stat(current_work().c_str(), &source_stat) == 0 &&
        (source_stat.st_mode & S_IFREG)) {
      flags |= MDB_NOSUBDIR;
    }

    MDB_CHECK(mdb_env_open(mdb_env_, current_work().c_str(), flags, 0664));
    MDB_CHECK(mdb_txn_begin(mdb_env_, nullptr, MDB_RDONLY, &mdb_txn_));
    MDB_CHECK(mdb_dbi_open(mdb_txn_, nullptr, 0, &mdb_dbi_));

    return Status::OK();
  }

  Status OnWorkFinishedLocked() override {
    if (mdb_env_ != nullptr) {
      if (mdb_cursor_) {
        mdb_cursor_close(mdb_cursor_);
      }
      mdb_txn_abort(mdb_txn_);
      mdb_dbi_close(mdb_env_, mdb_dbi_);
      mdb_env_close(mdb_env_);
      mdb_env_ = nullptr;
    }
    return Status::OK();
  }

  Status ReadLocked(string* key, string* value, bool* produced,
                    bool* at_end) override {
    if (mdb_cursor_ == nullptr) {
      MDB_CHECK(mdb_cursor_open(mdb_txn_, mdb_dbi_, &mdb_cursor_));
      if (Seek(MDB_FIRST) == false) {
        *at_end = true;
        return Status::OK();
      }
    }
    else {
      if (Seek(MDB_NEXT) == false) {
        *at_end = true;
        return Status::OK();
      }
    }
    *key = string(static_cast<const char*>(mdb_key_.mv_data),
                  mdb_key_.mv_size);
    *value = string(static_cast<const char*>(mdb_value_.mv_data),
                    mdb_value_.mv_size);
    *produced = true;
    return Status::OK();
  }

  Status ResetLocked() override {
    CHECK_EQ(Seek(MDB_FIRST), true);
    return ReaderBase::ResetLocked();
  }

 private:
  bool Seek(MDB_cursor_op op) {
    CHECK_NOTNULL(mdb_cursor_);
    int mdb_status = mdb_cursor_get(mdb_cursor_, &mdb_key_, &mdb_value_, op);
    if (mdb_status == MDB_NOTFOUND) {
      return false;
    } else {
      MDB_CHECK(mdb_status);
      return true;
    }
  }

  Env* const env_;
  MDB_env* mdb_env_;
  MDB_dbi mdb_dbi_;

  MDB_txn* mdb_txn_;
  MDB_cursor* mdb_cursor_;
  MDB_val mdb_key_, mdb_value_;
};

class LMDBReaderOp : public ReaderOpKernel {
 public:
  explicit LMDBReaderOp(OpKernelConstruction* context)
      : ReaderOpKernel(context) {
    Env* env = context->env();
    SetReaderFactory([this, env]() {
      return new LMDBReader(name(), env);
    });
  }
};

REGISTER_KERNEL_BUILDER(Name("LMDBReader").Device(DEVICE_CPU),
                        LMDBReaderOp);

}
