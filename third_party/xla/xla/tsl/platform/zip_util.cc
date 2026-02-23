/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/tsl/platform/zip_util.h"

#include <algorithm>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/cleanup/cleanup.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "third_party/libzip/lib/zip.h"
#include "third_party/libzip/zipconf.h"
#include "google/protobuf/io/zero_copy_stream.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/file_statistics.h"
#include "xla/tsl/platform/ram_file_system.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/zip.h"
#include "tsl/platform/file_system.h"  // IWYU pragma: keep
#include "tsl/platform/path.h"

namespace tsl {
namespace zip {
namespace {

absl::Status ValidateZipEntryPath(absl::string_view entry) {
  if (tsl::io::IsAbsolutePath(entry)) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Entry '", entry, "' is an absolute path, which is disallowed."));
  }
  std::string cleaned_path = tsl::io::CleanPath(entry);
  if (cleaned_path == ".." || absl::StartsWith(cleaned_path, "../")) {
    return absl::InvalidArgumentError(
        absl::StrCat("Entry '", entry,
                     "' attempts to traverse above archive root via '..'. "
                     "This is disallowed for security reasons."));
  }
  return absl::OkStatus();
}

// Implementation of ZeroCopyInputStream that reads from a zip file entry
// using libzip.
class LibzipInputStream : public google::protobuf::io::ZeroCopyInputStream {
 public:
  explicit LibzipInputStream(zip_file_t* file) : file_(file) {}
  ~LibzipInputStream() override {
    if (file_) {
      zip_fclose(file_);
    }
  }

  bool Next(const void** data, int* size) override {
    if (buffer_offset_ < buffer_size_) {
      // There are unused bytes in buffer_, likely from BackUp().
      *data = buffer_ + buffer_offset_;
      *size = buffer_size_ - buffer_offset_;
      position_ += *size;
      buffer_offset_ = buffer_size_;
      return true;
    }

    zip_int64_t bytes_read = zip_fread(file_, buffer_, kBufferSize);
    if (bytes_read < 0) {
      buffer_size_ = 0;
      buffer_offset_ = 0;
      return false;  // error
    }
    if (bytes_read == 0) {
      buffer_size_ = 0;
      buffer_offset_ = 0;
      return false;  // EOF
    }
    buffer_size_ = bytes_read;
    buffer_offset_ = bytes_read;
    *data = buffer_;
    *size = bytes_read;
    position_ += bytes_read;
    return true;
  }

  void BackUp(int count) override {
    buffer_offset_ -= count;
    position_ -= count;
  }

  bool Skip(int count) override {
    while (count > 0) {
      const void* data;
      int size;
      if (!Next(&data, &size)) {
        return false;
      }
      if (size > count) {
        BackUp(size - count);
        return true;
      }
      count -= size;
    }
    return true;
  }

  int64_t ByteCount() const override { return position_; }

 private:
  static constexpr int kBufferSize = 1024 * 64;
  zip_file_t* file_;
  char buffer_[kBufferSize];
  int buffer_size_ = 0;    // size of valid data in buffer_
  int buffer_offset_ = 0;  // offset of next byte to return in buffer_
  int64_t position_ = 0;
};

// Implementation of ZipArchive that uses libzip to read from a zip archive.
class LibzipArchive : public ZipArchive {
 public:
  explicit LibzipArchive(std::shared_ptr<zip_t> archive)
      : archive_(std::move(archive)) {}

  absl::StatusOr<std::vector<std::string>> GetEntries() override {
    std::vector<std::string> files;
    zip_int64_t num_entries = zip_get_num_entries(archive_.get(), 0);
    if (num_entries < 0) {
      return absl::InternalError(
          "Failed to get number of entries in zip archive.");
    }

    for (zip_int64_t i = 0; i < num_entries; ++i) {
      const char* name = zip_get_name(archive_.get(), i, 0);
      if (name == nullptr) {
        return absl::InternalError("Failed to get name of zip entry.");
      }
      files.push_back(name);
    }
    return files;
  }

  absl::StatusOr<std::string> GetContents(absl::string_view entry) override {
    TF_RETURN_IF_ERROR(ValidateZipEntryPath(entry));
    struct zip_stat st;
    zip_stat_init(&st);
    if (zip_stat(archive_.get(), entry.data(), 0, &st) != 0) {
      return absl::NotFoundError(absl::StrCat("Entry not found: ", entry));
    }

    if (!(st.valid & ZIP_STAT_SIZE)) {
      return absl::InternalError("Could not determine size of entry.");
    }

    zip_file_t* file = zip_fopen(archive_.get(), entry.data(), 0);
    if (file == nullptr) {
      return absl::InternalError(absl::StrCat("Failed to open entry: ", entry));
    }
    absl::Cleanup closer = [file] { zip_fclose(file); };

    std::string content;
    content.resize(st.size);
    zip_int64_t bytes_read = zip_fread(file, &content[0], st.size);

    if (bytes_read < 0 || static_cast<zip_uint64_t>(bytes_read) != st.size) {
      return absl::InternalError(
          absl::StrCat("Failed to read whole entry: ", entry));
    }

    return content;
  }

  absl::StatusOr<std::unique_ptr<tsl::RandomAccessFile>> Open(
      absl::string_view entry) override {
    TF_RETURN_IF_ERROR(ValidateZipEntryPath(entry));
    TF_ASSIGN_OR_RETURN(std::string content, GetContents(entry));
    return std::make_unique<RamRandomAccessFile>(
        std::string(entry), std::make_shared<std::string>(std::move(content)));
  }

  absl::StatusOr<std::unique_ptr<google::protobuf::io::ZeroCopyInputStream>>
  GetZeroCopyInputStream(absl::string_view entry) override {
    TF_RETURN_IF_ERROR(ValidateZipEntryPath(entry));
    struct zip_stat st;
    zip_stat_init(&st);
    if (zip_stat(archive_.get(), entry.data(), 0, &st) != 0) {
      return absl::NotFoundError(absl::StrCat("Entry not found: ", entry));
    }
    zip_file_t* file = zip_fopen(archive_.get(), entry.data(), 0);
    if (file == nullptr) {
      return absl::InternalError(absl::StrCat("Failed to open entry: ", entry));
    }
    return std::make_unique<LibzipInputStream>(file);
  }

 private:
  std::shared_ptr<zip_t> archive_;
};

// Context for tsl_source_callback. This holds the state for reading from
// a zip archive file using tsl::Env.
struct TslSourceContext {
  explicit TslSourceContext(const std::string& path)
      : filename(path),
        env(Env::Default()),
        file(nullptr),
        size(0),
        position(0),
        status(absl::OkStatus()) {
    zip_error_init(&error);
  }
  ~TslSourceContext() { zip_error_fini(&error); }
  std::string filename;
  Env* env;
  std::unique_ptr<RandomAccessFile> file;
  uint64_t size;
  uint64_t position;
  zip_error_t error;
  absl::Status status;
};

// Callback for libzip custom source. This allows libzip to read from a zip
// archive file using tsl::Env.
zip_int64_t tsl_source_callback(void* userdata, void* data, zip_uint64_t len,
                                zip_source_cmd_t cmd) {
  TslSourceContext* ctx = static_cast<TslSourceContext*>(userdata);
  switch (cmd) {
    case ZIP_SOURCE_OPEN: {
      if (ctx->file) {
        return 0;
      }
      std::unique_ptr<RandomAccessFile> f;
      absl::Status s = ctx->env->NewRandomAccessFile(ctx->filename, &f);
      if (!s.ok()) {
        ctx->status = s;
        zip_error_set(&ctx->error, ZIP_ER_OPEN, ENOENT);
        return -1;
      }
      ctx->file = std::move(f);
      ctx->position = 0;
      return 0;
    }
    case ZIP_SOURCE_READ: {
      if (!ctx->file) {
        zip_error_set(&ctx->error, ZIP_ER_INVAL, 0);
        return -1;
      }
      absl::string_view result_sv;
      size_t to_read = std::min(len, ctx->size - ctx->position);
      if (to_read == 0) {
        return 0;
      }
      absl::Status s =
          ctx->file->Read(ctx->position, result_sv,
                          absl::MakeSpan(static_cast<char*>(data), to_read));
      if (!s.ok() && !absl::IsOutOfRange(s)) {
        // Read error
        ctx->status = s;
        zip_error_set(&ctx->error, ZIP_ER_READ, EIO);
        return -1;
      }
      if (result_sv.data() != data) {
        memcpy(data, result_sv.data(), result_sv.size());
      }
      ctx->position += result_sv.size();
      return result_sv.size();
    }
    case ZIP_SOURCE_CLOSE:
      ctx->file.reset();
      return 0;
    case ZIP_SOURCE_STAT: {
      if (len < sizeof(zip_stat_t)) {
        return -1;
      }
      zip_stat_t* st = static_cast<zip_stat_t*>(data);

      tsl::FileStatistics stat;
      absl::Status s = ctx->env->Stat(ctx->filename, &stat);
      if (!s.ok()) {
        ctx->status = s;
        zip_error_set(&ctx->error, ZIP_ER_READ, EIO);  // stat failed
        return -1;
      }
      ctx->size = stat.length;
      st->valid |= ZIP_STAT_SIZE | ZIP_STAT_MTIME;
      st->size = stat.length;
      st->mtime = stat.mtime_nsec / 1000000000;
      return 0;
    }
    case ZIP_SOURCE_ERROR:
      return zip_error_to_data(&ctx->error, data, len);
    case ZIP_SOURCE_FREE:
      delete ctx;
      return 0;
    case ZIP_SOURCE_SEEK: {
      zip_source_args_seek_t* args =
          ZIP_SOURCE_GET_ARGS(zip_source_args_seek_t, data, len, &ctx->error);
      if (!args) {
        return -1;
      }
      zip_int64_t offset = args->offset;
      int whence = args->whence;
      zip_uint64_t new_pos;
      switch (whence) {
        case SEEK_SET:
          new_pos = offset;
          break;
        case SEEK_CUR:
          new_pos = ctx->position + offset;
          break;
        case SEEK_END:
          new_pos = ctx->size + offset;
          break;
        default:
          zip_error_set(&ctx->error, ZIP_ER_INVAL, 0);
          return -1;
      }
      if (new_pos > ctx->size) {
        zip_error_set(&ctx->error, ZIP_ER_INVAL, 0);
        return -1;
      }
      ctx->position = new_pos;
      return 0;
    }
    case ZIP_SOURCE_TELL:
      return ctx->position;
    case ZIP_SOURCE_SUPPORTS:
      return zip_source_make_command_bitmap(
          ZIP_SOURCE_OPEN, ZIP_SOURCE_READ, ZIP_SOURCE_CLOSE, ZIP_SOURCE_STAT,
          ZIP_SOURCE_ERROR, ZIP_SOURCE_FREE, ZIP_SOURCE_SEEK, ZIP_SOURCE_TELL,
          -1);
    default:
      zip_error_set(&ctx->error, ZIP_ER_OPNOTSUPP, 0);
      return -1;
  }
}

}  // namespace

absl::StatusOr<std::unique_ptr<ZipArchive>> OpenArchiveWithTsl(
    absl::string_view path_sv) {
  std::string path(path_sv);
  // Create a context for the tsl_source_callback. This context will be
  // passed to the callback by libzip.
  // When the zip archive is closed (e.g., via zip_discard), libzip will
  // free the zip source, which in turn will call tsl_source_callback with
  // ZIP_SOURCE_FREE, and the callback will delete this context.
  TslSourceContext* ctx = new TslSourceContext(path);
  // Create a libzip source from our callback. libzip will use this source
  // to read the zip archive file via tsl::Env.
  zip_source_t* source =
      zip_source_function_create(tsl_source_callback, ctx, &ctx->error);
  if (source == nullptr) {
    auto status = absl::InternalError(
        absl::StrCat("Failed to create zip source for ", path, ": ",
                     zip_error_strerror(&ctx->error)));
    delete ctx;
    return status;
  }
  // If zip_open_from_source() fails, it does not take ownership of `source`,
  // so we must free it. If it succeeds, it takes ownership of `source`.
  // We use a cleanup to free `source` in case of failure and cancel it in
  // case of success.
  absl::Cleanup source_cleanup = [source] { zip_source_free(source); };

  // Open the zip archive from the source. ZIP_RDONLY ensures that we only
  // read from the archive. If zip_open_from_source() succeeds, it takes
  // ownership of `source`.
  zip_t* archive = zip_open_from_source(source, ZIP_RDONLY, &ctx->error);
  if (archive == nullptr) {
    if (!ctx->status.ok()) {
      return ctx->status;
    }
    if (zip_error_code_zip(&ctx->error) == ZIP_ER_NOZIP) {
      return absl::InvalidArgumentError(
          absl::StrCat("File is not a valid zip archive: ", path));
    }
    return absl::InternalError(
        absl::StrCat("Failed to open zip archive ", path,
                     " from source: ", zip_error_strerror(&ctx->error)));
  }
  std::move(source_cleanup).Cancel();
  // Wrap archive in a shared_ptr with a custom deleter to call zip_discard
  // when we are done with it.
  std::shared_ptr<zip_t> shared_archive(archive, [](zip_t* z) {
    if (z) {
      zip_discard(z);
    }
  });
  return std::make_unique<LibzipArchive>(std::move(shared_archive));
}

}  // namespace zip
}  // namespace tsl
