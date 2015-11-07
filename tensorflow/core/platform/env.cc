#include "tensorflow/core/public/env.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/protobuf.h"

namespace tensorflow {

Env::~Env() {}

RandomAccessFile::~RandomAccessFile() {}

WritableFile::~WritableFile() {}

Thread::~Thread() {}

EnvWrapper::~EnvWrapper() {}

Status ReadFileToString(Env* env, const string& fname, string* data) {
  data->clear();
  RandomAccessFile* file;
  Status s = env->NewRandomAccessFile(fname, &file);
  if (!s.ok()) {
    return s;
  }
  int64 offset = 0;
  static const int kBufferSize = 8192;
  char* space = new char[kBufferSize];
  while (true) {
    StringPiece fragment;
    s = file->Read(offset, kBufferSize, &fragment, space);
    if (!s.ok()) {
      if (errors::IsOutOfRange(s)) {  // No more bytes, but not an error
        s = Status::OK();
        data->append(fragment.data(), fragment.size());
      }
      break;
    }
    offset += fragment.size();
    data->append(fragment.data(), fragment.size());
    if (fragment.empty()) {
      break;
    }
  }
  delete[] space;
  delete file;
  return s;
}

Status WriteStringToFile(Env* env, const string& fname,
                         const StringPiece& data) {
  WritableFile* file;
  Status s = env->NewWritableFile(fname, &file);
  if (!s.ok()) {
    return s;
  }
  s = file->Append(data);
  if (s.ok()) {
    s = file->Close();
  }
  delete file;
  return s;
}

// A ZeroCopyInputStream on a RandomAccessFile.
namespace {
class FileStream : public ::tensorflow::protobuf::io::ZeroCopyInputStream {
 public:
  explicit FileStream(RandomAccessFile* file) : file_(file), pos_(0) {}

  void BackUp(int count) override { pos_ -= count; }
  bool Skip(int count) override {
    pos_ += count;
    return true;
  }
  int64 ByteCount() const override { return pos_; }
  Status status() const { return status_; }

  bool Next(const void** data, int* size) override {
    StringPiece result;
    Status s = file_->Read(pos_, kBufSize, &result, scratch_);
    if (result.empty()) {
      status_ = s;
      return false;
    }
    pos_ += result.size();
    *data = result.data();
    *size = result.size();
    return true;
  }

 private:
  static const int kBufSize = 512 << 10;

  RandomAccessFile* file_;
  int64 pos_;
  Status status_;
  char scratch_[kBufSize];
};

}  // namespace

Status ReadBinaryProto(Env* env, const string& fname,
                       ::tensorflow::protobuf::MessageLite* proto) {
  RandomAccessFile* file;
  auto s = env->NewRandomAccessFile(fname, &file);
  if (!s.ok()) {
    return s;
  }
  std::unique_ptr<RandomAccessFile> file_holder(file);
  std::unique_ptr<FileStream> stream(new FileStream(file));

  // TODO(jiayq): the following coded stream is for debugging purposes to allow
  // one to parse arbitrarily large messages for MessageLite. One most likely
  // doesn't want to put protobufs larger than 64MB on Android, so we should
  // eventually remove this and quit loud when a large protobuf is passed in.
  ::tensorflow::protobuf::io::CodedInputStream coded_stream(stream.get());
  // Total bytes hard limit / warning limit are set to 1GB and 512MB
  // respectively.
  coded_stream.SetTotalBytesLimit(1024LL << 20, 512LL << 20);

  if (!proto->ParseFromCodedStream(&coded_stream)) {
    s = stream->status();
    if (s.ok()) {
      s = Status(error::DATA_LOSS, "Parse error");
    }
  }
  return s;
}

}  // namespace tensorflow
