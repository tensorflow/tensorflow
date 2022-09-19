/* Copyright 2015 Google Inc. All Rights Reserved.

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

#include "tensorflow/tsl/platform/windows/windows_file_system.h"

#include <Shlwapi.h>
#include <Windows.h>
#include <direct.h>
#include <errno.h>
#include <fcntl.h>
#include <io.h>
#undef StrCat
#include <stdio.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>

#include "tensorflow/core/protobuf/error_codes.pb.h"
#include "tensorflow/tsl/platform/env.h"
#include "tensorflow/tsl/platform/errors.h"
#include "tensorflow/tsl/platform/file_system_helper.h"
#include "tensorflow/tsl/platform/logging.h"
#include "tensorflow/tsl/platform/strcat.h"
#include "tensorflow/tsl/platform/windows/error_windows.h"
#include "tensorflow/tsl/platform/windows/wide_char.h"

// TODO(mrry): Prevent this Windows.h #define from leaking out of our headers.
#undef DeleteFile

namespace tsl {

using ::tsl::errors::IOError;

namespace {

// RAII helpers for HANDLEs
const auto CloseHandleFunc = [](HANDLE h) { ::CloseHandle(h); };
typedef std::unique_ptr<void, decltype(CloseHandleFunc)> UniqueCloseHandlePtr;

inline Status IOErrorFromWindowsError(const string& context) {
  auto last_error = ::GetLastError();
  return IOError(
      context + string(" : ") + internal::WindowsGetLastErrorMessage(),
      last_error);
}

// PLEASE NOTE: hfile is expected to be an async handle
// (i.e. opened with FILE_FLAG_OVERLAPPED)
SSIZE_T pread(HANDLE hfile, char* src, size_t num_bytes, uint64_t offset) {
  assert(num_bytes <= std::numeric_limits<DWORD>::max());
  OVERLAPPED overlapped = {0};
  ULARGE_INTEGER offset_union;
  offset_union.QuadPart = offset;

  overlapped.Offset = offset_union.LowPart;
  overlapped.OffsetHigh = offset_union.HighPart;
  overlapped.hEvent = ::CreateEvent(NULL, TRUE, FALSE, NULL);

  if (NULL == overlapped.hEvent) {
    return -1;
  }

  SSIZE_T result = 0;

  unsigned long bytes_read = 0;
  DWORD last_error = ERROR_SUCCESS;

  BOOL read_result = ::ReadFile(hfile, src, static_cast<DWORD>(num_bytes),
                                &bytes_read, &overlapped);
  if (TRUE == read_result) {
    result = bytes_read;
  } else if ((FALSE == read_result) &&
             ((last_error = GetLastError()) != ERROR_IO_PENDING)) {
    result = (last_error == ERROR_HANDLE_EOF) ? 0 : -1;
  } else {
    if (ERROR_IO_PENDING ==
        last_error) {  // Otherwise bytes_read already has the result.
      BOOL overlapped_result =
          ::GetOverlappedResult(hfile, &overlapped, &bytes_read, TRUE);
      if (FALSE == overlapped_result) {
        result = (::GetLastError() == ERROR_HANDLE_EOF) ? 0 : -1;
      } else {
        result = bytes_read;
      }
    }
  }

  ::CloseHandle(overlapped.hEvent);

  return result;
}

// read() based random-access
class WindowsRandomAccessFile : public RandomAccessFile {
 private:
  string filename_;
  HANDLE hfile_;

 public:
  WindowsRandomAccessFile(const string& fname, HANDLE hfile)
      : filename_(fname), hfile_(hfile) {}
  ~WindowsRandomAccessFile() override {
    if (hfile_ != NULL && hfile_ != INVALID_HANDLE_VALUE) {
      ::CloseHandle(hfile_);
    }
  }

  Status Name(StringPiece* result) const override {
    *result = filename_;
    return Status::OK();
  }

  Status Read(uint64 offset, size_t n, StringPiece* result,
              char* scratch) const override {
    Status s;
    char* dst = scratch;
    while (n > 0 && s.ok()) {
      size_t requested_read_length;
      if (n > std::numeric_limits<DWORD>::max()) {
        requested_read_length = std::numeric_limits<DWORD>::max();
      } else {
        requested_read_length = n;
      }
      SSIZE_T r = pread(hfile_, dst, requested_read_length, offset);
      if (r > 0) {
        offset += r;
        dst += r;
        n -= r;
      } else if (r == 0) {
        s = Status(error::OUT_OF_RANGE, "Read fewer bytes than requested");
      } else if (errno == EINTR || errno == EAGAIN) {
        // Retry
      } else {
        s = IOError(filename_, errno);
      }
    }
    *result = StringPiece(scratch, dst - scratch);
    return s;
  }

#if defined(TF_CORD_SUPPORT)
  Status Read(uint64 offset, size_t n, absl::Cord* cord) const override {
    if (n == 0) {
      return Status::OK();
    }
    if (n < 0) {
      return errors::InvalidArgument(
          "Attempting to read ", n,
          " bytes. You cannot read a negative number of bytes.");
    }

    char* scratch = new char[n];
    if (scratch == nullptr) {
      return errors::ResourceExhausted("Unable to allocate ", n,
                                       " bytes for file reading.");
    }

    StringPiece tmp;
    Status s = Read(offset, n, &tmp, scratch);

    absl::Cord tmp_cord = absl::MakeCordFromExternal(
        absl::string_view(static_cast<char*>(scratch), tmp.size()),
        [scratch](absl::string_view) { delete[] scratch; });
    cord->Append(tmp_cord);
    return s;
  }
#endif
};

class WindowsWritableFile : public WritableFile {
 private:
  string filename_;
  HANDLE hfile_;

 public:
  WindowsWritableFile(const string& fname, HANDLE hFile)
      : filename_(fname), hfile_(hFile) {}

  ~WindowsWritableFile() override {
    if (hfile_ != NULL && hfile_ != INVALID_HANDLE_VALUE) {
      WindowsWritableFile::Close();
    }
  }

  Status Append(StringPiece data) override {
    DWORD bytes_written = 0;
    DWORD data_size = static_cast<DWORD>(data.size());
    BOOL write_result =
        ::WriteFile(hfile_, data.data(), data_size, &bytes_written, NULL);
    if (FALSE == write_result) {
      return IOErrorFromWindowsError("Failed to WriteFile: " + filename_);
    }

    assert(size_t(bytes_written) == data.size());
    return Status::OK();
  }

#if defined(TF_CORD_SUPPORT)
  // \brief Append 'data' to the file.
  Status Append(const absl::Cord& cord) override {
    for (const auto& chunk : cord.Chunks()) {
      DWORD bytes_written = 0;
      DWORD data_size = static_cast<DWORD>(chunk.size());
      BOOL write_result =
          ::WriteFile(hfile_, chunk.data(), data_size, &bytes_written, NULL);
      if (FALSE == write_result) {
        return IOErrorFromWindowsError("Failed to WriteFile: " + filename_);
      }

      assert(size_t(bytes_written) == chunk.size());
    }
    return Status::OK();
  }
#endif

  Status Tell(int64* position) override {
    Status result = Flush();
    if (!result.ok()) {
      return result;
    }

    *position = SetFilePointer(hfile_, 0, NULL, FILE_CURRENT);

    if (*position == INVALID_SET_FILE_POINTER) {
      return IOErrorFromWindowsError("Tell(SetFilePointer) failed for: " +
                                     filename_);
    }

    return Status::OK();
  }

  Status Close() override {
    assert(INVALID_HANDLE_VALUE != hfile_);

    Status result = Flush();
    if (!result.ok()) {
      return result;
    }

    if (FALSE == ::CloseHandle(hfile_)) {
      return IOErrorFromWindowsError("CloseHandle failed for: " + filename_);
    }

    hfile_ = INVALID_HANDLE_VALUE;
    return Status::OK();
  }

  Status Flush() override {
    if (FALSE == ::FlushFileBuffers(hfile_)) {
      return IOErrorFromWindowsError("FlushFileBuffers failed for: " +
                                     filename_);
    }
    return Status::OK();
  }

  Status Name(StringPiece* result) const override {
    *result = filename_;
    return Status::OK();
  }

  Status Sync() override { return Flush(); }
};

class WinReadOnlyMemoryRegion : public ReadOnlyMemoryRegion {
 private:
  const std::string filename_;
  HANDLE hfile_;
  HANDLE hmap_;

  const void* const address_;
  const uint64 length_;

 public:
  WinReadOnlyMemoryRegion(const std::string& filename, HANDLE hfile,
                          HANDLE hmap, const void* address, uint64 length)
      : filename_(filename),
        hfile_(hfile),
        hmap_(hmap),
        address_(address),
        length_(length) {}

  ~WinReadOnlyMemoryRegion() {
    BOOL ret = ::UnmapViewOfFile(address_);
    assert(ret);

    ret = ::CloseHandle(hmap_);
    assert(ret);

    ret = ::CloseHandle(hfile_);
    assert(ret);
  }

  const void* data() override { return address_; }
  uint64 length() override { return length_; }
};

}  // namespace

Status WindowsFileSystem::NewRandomAccessFile(
    const string& fname, TransactionToken* token,
    std::unique_ptr<RandomAccessFile>* result) {
  string translated_fname = TranslateName(fname);
  std::wstring ws_translated_fname = Utf8ToWideChar(translated_fname);
  result->reset();

  // Open the file for read-only random access
  // Open in async mode which makes Windows allow more parallelism even
  // if we need to do sync I/O on top of it.
  DWORD file_flags = FILE_ATTRIBUTE_READONLY | FILE_FLAG_OVERLAPPED;
  // Shared access is necessary for tests to pass
  // almost all tests would work with a possible exception of fault_injection.
  DWORD share_mode = FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE;

  HANDLE hfile =
      ::CreateFileW(ws_translated_fname.c_str(), GENERIC_READ, share_mode, NULL,
                    OPEN_EXISTING, file_flags, NULL);

  if (INVALID_HANDLE_VALUE == hfile) {
    string context = "NewRandomAccessFile failed to Create/Open: " + fname;
    return IOErrorFromWindowsError(context);
  }

  result->reset(new WindowsRandomAccessFile(translated_fname, hfile));
  return Status::OK();
}

Status WindowsFileSystem::NewWritableFile(
    const string& fname, TransactionToken* token,
    std::unique_ptr<WritableFile>* result) {
  string translated_fname = TranslateName(fname);
  std::wstring ws_translated_fname = Utf8ToWideChar(translated_fname);
  result->reset();

  DWORD share_mode = FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE;
  HANDLE hfile =
      ::CreateFileW(ws_translated_fname.c_str(), GENERIC_WRITE, share_mode,
                    NULL, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);

  if (INVALID_HANDLE_VALUE == hfile) {
    string context = "Failed to create a NewWriteableFile: " + fname;
    return IOErrorFromWindowsError(context);
  }

  result->reset(new WindowsWritableFile(translated_fname, hfile));
  return Status::OK();
}

Status WindowsFileSystem::NewAppendableFile(
    const string& fname, TransactionToken* token,
    std::unique_ptr<WritableFile>* result) {
  string translated_fname = TranslateName(fname);
  std::wstring ws_translated_fname = Utf8ToWideChar(translated_fname);
  result->reset();

  DWORD share_mode = FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE;
  HANDLE hfile =
      ::CreateFileW(ws_translated_fname.c_str(), GENERIC_WRITE, share_mode,
                    NULL, OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);

  if (INVALID_HANDLE_VALUE == hfile) {
    string context = "Failed to create a NewAppendableFile: " + fname;
    return IOErrorFromWindowsError(context);
  }

  UniqueCloseHandlePtr file_guard(hfile, CloseHandleFunc);

  DWORD file_ptr = ::SetFilePointer(hfile, NULL, NULL, FILE_END);
  if (INVALID_SET_FILE_POINTER == file_ptr) {
    string context = "Failed to create a NewAppendableFile: " + fname;
    return IOErrorFromWindowsError(context);
  }

  result->reset(new WindowsWritableFile(translated_fname, hfile));
  file_guard.release();

  return Status::OK();
}

Status WindowsFileSystem::NewReadOnlyMemoryRegionFromFile(
    const string& fname, TransactionToken* token,
    std::unique_ptr<ReadOnlyMemoryRegion>* result) {
  string translated_fname = TranslateName(fname);
  std::wstring ws_translated_fname = Utf8ToWideChar(translated_fname);
  result->reset();
  Status s = Status::OK();

  // Open the file for read-only
  DWORD file_flags = FILE_ATTRIBUTE_READONLY;

  // Open in async mode which makes Windows allow more parallelism even
  // if we need to do sync I/O on top of it.
  file_flags |= FILE_FLAG_OVERLAPPED;

  DWORD share_mode = FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE;
  HANDLE hfile =
      ::CreateFileW(ws_translated_fname.c_str(), GENERIC_READ, share_mode, NULL,
                    OPEN_EXISTING, file_flags, NULL);

  if (INVALID_HANDLE_VALUE == hfile) {
    return IOErrorFromWindowsError(
        "NewReadOnlyMemoryRegionFromFile failed to Create/Open: " + fname);
  }

  UniqueCloseHandlePtr file_guard(hfile, CloseHandleFunc);

  // Use mmap when virtual address-space is plentiful.
  uint64_t file_size;
  s = GetFileSize(translated_fname, &file_size);
  if (s.ok()) {
    // Will not map empty files
    if (file_size == 0) {
      return IOError(
          "NewReadOnlyMemoryRegionFromFile failed to map empty file: " + fname,
          EINVAL);
    }

    HANDLE hmap = ::CreateFileMappingA(hfile, NULL, PAGE_READONLY,
                                       0,  // Whole file at its present length
                                       0,
                                       NULL);  // Mapping name

    if (!hmap) {
      string context =
          "Failed to create file mapping for "
          "NewReadOnlyMemoryRegionFromFile: " +
          fname;
      return IOErrorFromWindowsError(context);
    }

    UniqueCloseHandlePtr map_guard(hmap, CloseHandleFunc);

    const void* mapped_region =
        ::MapViewOfFileEx(hmap, FILE_MAP_READ,
                          0,  // High DWORD of access start
                          0,  // Low DWORD
                          file_size,
                          NULL);  // Let the OS choose the mapping

    if (!mapped_region) {
      string context =
          "Failed to MapViewOfFile for "
          "NewReadOnlyMemoryRegionFromFile: " +
          fname;
      return IOErrorFromWindowsError(context);
    }

    result->reset(new WinReadOnlyMemoryRegion(fname, hfile, hmap, mapped_region,
                                              file_size));

    map_guard.release();
    file_guard.release();
  }

  return s;
}

Status WindowsFileSystem::FileExists(const string& fname,
                                     TransactionToken* token) {
  constexpr int kOk = 0;
  std::wstring ws_translated_fname = Utf8ToWideChar(TranslateName(fname));
  if (_waccess(ws_translated_fname.c_str(), kOk) == 0) {
    return Status::OK();
  }
  return errors::NotFound(fname, " not found");
}

Status WindowsFileSystem::GetChildren(const string& dir,
                                      TransactionToken* token,
                                      std::vector<string>* result) {
  string translated_dir = TranslateName(dir);
  std::wstring ws_translated_dir = Utf8ToWideChar(translated_dir);
  result->clear();

  std::wstring pattern = ws_translated_dir;
  if (!pattern.empty() && pattern.back() != '\\' && pattern.back() != '/') {
    pattern += L"\\*";
  } else {
    pattern += L'*';
  }

  WIN32_FIND_DATAW find_data;
  HANDLE find_handle = ::FindFirstFileW(pattern.c_str(), &find_data);
  if (find_handle == INVALID_HANDLE_VALUE) {
    string context = "FindFirstFile failed for: " + translated_dir;
    return IOErrorFromWindowsError(context);
  }

  do {
    string file_name = WideCharToUtf8(find_data.cFileName);
    const StringPiece basename = file_name;
    if (basename != "." && basename != "..") {
      result->push_back(file_name);
    }
  } while (::FindNextFileW(find_handle, &find_data));

  if (!::FindClose(find_handle)) {
    string context = "FindClose failed for: " + translated_dir;
    return IOErrorFromWindowsError(context);
  }

  return Status::OK();
}

Status WindowsFileSystem::DeleteFile(const string& fname,
                                     TransactionToken* token) {
  Status result;
  std::wstring file_name = Utf8ToWideChar(fname);
  if (_wunlink(file_name.c_str()) != 0) {
    result = IOError("Failed to delete a file: " + fname, errno);
  }
  return result;
}

Status WindowsFileSystem::CreateDir(const string& name,
                                    TransactionToken* token) {
  Status result;
  std::wstring ws_name = Utf8ToWideChar(name);
  if (ws_name.empty()) {
    return errors::AlreadyExists(name);
  }
  if (_wmkdir(ws_name.c_str()) != 0) {
    result = IOError("Failed to create a directory: " + name, errno);
  }
  return result;
}

Status WindowsFileSystem::DeleteDir(const string& name,
                                    TransactionToken* token) {
  Status result;
  std::wstring ws_name = Utf8ToWideChar(name);
  if (_wrmdir(ws_name.c_str()) != 0) {
    result = IOError("Failed to remove a directory: " + name, errno);
  }
  return result;
}

Status WindowsFileSystem::GetFileSize(const string& fname,
                                      TransactionToken* token, uint64* size) {
  string translated_fname = TranslateName(fname);
  std::wstring ws_translated_dir = Utf8ToWideChar(translated_fname);
  Status result;
  WIN32_FILE_ATTRIBUTE_DATA attrs;
  if (TRUE == ::GetFileAttributesExW(ws_translated_dir.c_str(),
                                     GetFileExInfoStandard, &attrs)) {
    ULARGE_INTEGER file_size;
    file_size.HighPart = attrs.nFileSizeHigh;
    file_size.LowPart = attrs.nFileSizeLow;
    *size = file_size.QuadPart;
  } else {
    string context = "Can not get size for: " + fname;
    result = IOErrorFromWindowsError(context);
  }
  return result;
}

Status WindowsFileSystem::IsDirectory(const string& fname,
                                      TransactionToken* token) {
  TF_RETURN_IF_ERROR(FileExists(fname));
  std::wstring ws_translated_fname = Utf8ToWideChar(TranslateName(fname));
  if (PathIsDirectoryW(ws_translated_fname.c_str())) {
    return Status::OK();
  }
  return Status(tsl::error::FAILED_PRECONDITION, "Not a directory");
}

Status WindowsFileSystem::RenameFile(const string& src, const string& target,
                                     TransactionToken* token) {
  // rename() is not capable of replacing the existing file as on Linux
  // so use OS API directly
  std::wstring ws_translated_src = Utf8ToWideChar(TranslateName(src));
  std::wstring ws_translated_target = Utf8ToWideChar(TranslateName(target));

  // Calling MoveFileExW with the MOVEFILE_REPLACE_EXISTING flag can fail if
  // another process has a handle to the file that it didn't close yet. On the
  // other hand, calling DeleteFileW + MoveFileExW will work in that scenario
  // because it allows the process to keep using the old handle while also
  // creating a new handle for the new file.
  WIN32_FIND_DATAW find_file_data;
  HANDLE target_file_handle =
      ::FindFirstFileW(ws_translated_target.c_str(), &find_file_data);
  if (target_file_handle != INVALID_HANDLE_VALUE) {
    if (!::DeleteFileW(ws_translated_target.c_str())) {
      ::FindClose(target_file_handle);
      return IOErrorFromWindowsError(
          strings::StrCat("Failed to rename: ", src, " to: ", target));
    }
    ::FindClose(target_file_handle);
  }

  if (!::MoveFileExW(ws_translated_src.c_str(), ws_translated_target.c_str(),
                     0)) {
    return IOErrorFromWindowsError(
        strings::StrCat("Failed to rename: ", src, " to: ", target));
  }

  return Status::OK();
}

Status WindowsFileSystem::GetMatchingPaths(const string& pattern,
                                           TransactionToken* token,
                                           std::vector<string>* results) {
  // NOTE(mrry): The existing implementation of FileSystem::GetMatchingPaths()
  // does not handle Windows paths containing backslashes correctly. Since
  // Windows APIs will accept forward and backslashes equivalently, we
  // convert the pattern to use forward slashes exclusively. Note that this
  // is not ideal, since the API expects backslash as an escape character,
  // but no code appears to rely on this behavior.
  string converted_pattern(pattern);
  std::replace(converted_pattern.begin(), converted_pattern.end(), '\\', '/');
  TF_RETURN_IF_ERROR(internal::GetMatchingPaths(this, Env::Default(),
                                                converted_pattern, results));
  for (string& result : *results) {
    std::replace(result.begin(), result.end(), '/', '\\');
  }
  return Status::OK();
}

bool WindowsFileSystem::Match(const string& filename, const string& pattern) {
  std::wstring ws_path(Utf8ToWideChar(filename));
  std::wstring ws_pattern(Utf8ToWideChar(pattern));
  return PathMatchSpecW(ws_path.c_str(), ws_pattern.c_str()) == TRUE;
}

Status WindowsFileSystem::Stat(const string& fname, TransactionToken* token,
                               FileStatistics* stat) {
  Status result;
  struct _stat64 sbuf;
  std::wstring ws_translated_fname = Utf8ToWideChar(TranslateName(fname));
  if (_wstat64(ws_translated_fname.c_str(), &sbuf) != 0) {
    result = IOError(fname, errno);
  } else {
    stat->mtime_nsec = sbuf.st_mtime * 1e9;
    stat->length = sbuf.st_size;
    stat->is_directory = IsDirectory(fname).ok();
  }
  return result;
}

}  // namespace tsl
