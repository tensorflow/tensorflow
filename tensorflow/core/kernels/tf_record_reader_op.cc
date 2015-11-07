// See docs in ../ops/io_ops.cc.

#include <memory>
#include "tensorflow/core/framework/reader_op_kernel.h"
#include "tensorflow/core/kernels/reader_base.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/io/record_reader.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/public/env.h"

namespace tensorflow {

class TFRecordReader : public ReaderBase {
 public:
  TFRecordReader(const string& node_name, Env* env)
      : ReaderBase(strings::StrCat("TFRecordReader '", node_name, "'")),
        env_(env),
        offset_(0) {}

  Status OnWorkStartedLocked() override {
    offset_ = 0;
    RandomAccessFile* file = nullptr;
    TF_RETURN_IF_ERROR(env_->NewRandomAccessFile(current_work(), &file));
    file_.reset(file);
    reader_.reset(new io::RecordReader(file));
    return Status::OK();
  }

  Status OnWorkFinishedLocked() override {
    reader_.reset(nullptr);
    file_.reset(nullptr);
    return Status::OK();
  }

  Status ReadLocked(string* key, string* value, bool* produced,
                    bool* at_end) override {
    *key = strings::StrCat(current_work(), ":", offset_);
    Status status = reader_->ReadRecord(&offset_, value);
    if (errors::IsOutOfRange(status)) {
      *at_end = true;
      return Status::OK();
    }
    if (!status.ok()) return status;
    *produced = true;
    return Status::OK();
  }

  Status ResetLocked() override {
    offset_ = 0;
    reader_.reset(nullptr);
    file_.reset(nullptr);
    return ReaderBase::ResetLocked();
  }

  // TODO(josh11b): Implement serializing and restoring the state.

 private:
  Env* const env_;
  uint64 offset_;
  std::unique_ptr<RandomAccessFile> file_;
  std::unique_ptr<io::RecordReader> reader_;
};

class TFRecordReaderOp : public ReaderOpKernel {
 public:
  explicit TFRecordReaderOp(OpKernelConstruction* context)
      : ReaderOpKernel(context) {
    Env* env = context->env();
    SetReaderFactory([this, env]() { return new TFRecordReader(name(), env); });
  }
};

REGISTER_KERNEL_BUILDER(Name("TFRecordReader").Device(DEVICE_CPU),
                        TFRecordReaderOp);

}  // namespace tensorflow
