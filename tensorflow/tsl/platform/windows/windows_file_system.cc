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
#include <tchar.h>
#include <stdio.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>
#include <shellapi.h>
#include <shlobj.h>
#include <Shobjidl.h>
#include <Combaseapi.h>

#pragma comment(lib, "shell32.lib")
#pragma comment(lib, "ole32.lib")

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
    return OkStatus();
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
      return OkStatus();
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
    return OkStatus();
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
    return OkStatus();
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

    return OkStatus();
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
    return OkStatus();
  }

  Status Flush() override {
    if (FALSE == ::FlushFileBuffers(hfile_)) {
      return IOErrorFromWindowsError("FlushFileBuffers failed for: " +
                                     filename_);
    }
    return OkStatus();
  }

  Status Name(StringPiece* result) const override {
    *result = filename_;
    return OkStatus();
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

#define MAX_LONGPATH_LENGTH 400

static std::wstring GetUncPathName(const std::wstring& path) {
  static WCHAR wcPath[MAX_LONGPATH_LENGTH];

  // boundary case check
  if (path.size() >= MAX_LONGPATH_LENGTH) {
    string context = "ERROR: GetUncPathName cannot handle path size >= " + std::to_string(MAX_LONGPATH_LENGTH) + ", " + WideCharToUtf8(path);
    LOG(ERROR) << context;
    return std::wstring(path);
  }

  auto rcode = GetFullPathNameW(path.c_str(), MAX_LONGPATH_LENGTH, wcPath, NULL);
  LOG(INFO) << "GetUncPathName GetFullPathNameW for " << WideCharToUtf8(path) << " => rcode=" << rcode << " , " << WideCharToUtf8(std::wstring(wcPath));
  std::wstring ws_final_path(wcPath);
  std::wstring uncPath;
  if (wcPath[0] == '\\' && wcPath[1] == '\\' && wcPath[2] == '?' && wcPath[3] == '\\') {
    uncPath = ws_final_path;
  } else {
    uncPath = L"\\\\?\\" + ws_final_path;
  }

  return uncPath;
}

static std::wstring GetUncPathName(const std::string& path) {
  return GetUncPathName(Utf8ToWideChar(path));
}

static std::wstring GetSymbolicLinkTarget(const std::wstring& linkPath) {

  WCHAR path[MAX_LONGPATH_LENGTH];

  std::wstring uncLinkPath = GetUncPathName(linkPath);

  HANDLE hFile = ::CreateFileW( uncLinkPath.c_str(),
    GENERIC_READ,
    FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,  
    0,
    OPEN_EXISTING,
    FILE_ATTRIBUTE_READONLY | FILE_FLAG_OVERLAPPED,
    0);

  if (INVALID_HANDLE_VALUE != hFile) {
    auto rcode = GetFinalPathNameByHandleW(hFile, path, MAX_LONGPATH_LENGTH, FILE_NAME_NORMALIZED);
    ::CloseHandle(hFile);
    if (rcode) {
      return std::wstring(path, path + rcode);
    }
  } else {
    DWORD dwErr = GetLastError();
    LOG(ERROR) << "ERROR: GetSymbolicLinkTarget cannot open file for " << WideCharToUtf8(uncLinkPath).c_str() << " GetLastError: " << dwErr << "\n";
  }

  return uncLinkPath;
}

Status WindowsFileSystem::NewRandomAccessFile(
    const string& fname, TransactionToken* token,
    std::unique_ptr<RandomAccessFile>* result) {
  string translated_fname = TranslateName(fname);
  std::wstring ws_translated_fname = Utf8ToWideChar(translated_fname);
  std::wstring ws_final_fname = GetSymbolicLinkTarget(ws_translated_fname);
  LOG(INFO) << "WindowsFileSystem::NewRandomAccessFile 0 => " << WideCharToUtf8(ws_final_fname).c_str() << "\n";
  result->reset();

  // Open the file for read-only random access
  // Open in async mode which makes Windows allow more parallelism even
  // if we need to do sync I/O on top of it.
  DWORD file_flags = FILE_ATTRIBUTE_READONLY | FILE_FLAG_OVERLAPPED;
  // Shared access is necessary for tests to pass
  // almost all tests would work with a possible exception of fault_injection.
  DWORD share_mode = FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE;

  HANDLE hfile =
      ::CreateFileW(ws_final_fname.c_str(), GENERIC_READ, share_mode, NULL,
                    OPEN_EXISTING, file_flags, NULL);

  if (INVALID_HANDLE_VALUE == hfile) {
    string context = "NewRandomAccessFile failed to Create/Open: " + fname;
    return IOErrorFromWindowsError(context);
  }

  result->reset(new WindowsRandomAccessFile(translated_fname, hfile));
  return OkStatus();
}

Status WindowsFileSystem::NewWritableFile(
    const string& fname, TransactionToken* token,
    std::unique_ptr<WritableFile>* result) {
  std::wstring ws_final_fname = GetUncPathName(TranslateName(fname));
  LOG(INFO) << "WindowsFileSystem::NewWritableFile => " << WideCharToUtf8(ws_final_fname).c_str() << "\n";
  result->reset();

  DWORD share_mode = FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE;
  HANDLE hfile =
      ::CreateFileW(ws_final_fname.c_str(), GENERIC_WRITE, share_mode,
                    NULL, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);

  if (INVALID_HANDLE_VALUE == hfile) {
    string context = "Failed to create a NewWriteableFile: " + fname;
    return IOErrorFromWindowsError(context);
  }

  result->reset(new WindowsWritableFile(WideCharToUtf8(ws_final_fname), hfile));
  return OkStatus();
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

  return OkStatus();
}

Status WindowsFileSystem::NewReadOnlyMemoryRegionFromFile(
    const string& fname, TransactionToken* token,
    std::unique_ptr<ReadOnlyMemoryRegion>* result) {
  string translated_fname = TranslateName(fname);
  std::wstring ws_translated_fname = Utf8ToWideChar(translated_fname);
  result->reset();
  Status s = OkStatus();

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
    return OkStatus();
  }
  return errors::NotFound(fname, " not found");
}

Status WindowsFileSystem::GetChildren(const string& dir,
                                      TransactionToken* token,
                                      std::vector<string>* result) {
  std::wstring ws_fname_final = GetUncPathName(TranslateName(dir));
  result->clear();

  std::wstring pattern = ws_fname_final;
  if (!pattern.empty() && pattern.back() != '\\' && pattern.back() != '/') {
    pattern += L"\\*";
  } else {
    pattern += L'*';
  }

  WIN32_FIND_DATAW find_data;
  HANDLE find_handle = ::FindFirstFileW(pattern.c_str(), &find_data);
  if (find_handle == INVALID_HANDLE_VALUE) {
    string context = "FindFirstFile failed for: " + dir;
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
    string context = "FindClose failed for: " + dir;
    return IOErrorFromWindowsError(context);
  }

  return OkStatus();
}

Status WindowsFileSystem::DeleteFile(const string& fname,
                                     TransactionToken* token) {
  Status result;
  std::wstring ws_fname_final = GetUncPathName(TranslateName(fname));
  LOG(INFO) << "WindowsFileSystem::DeleteFile => " << WideCharToUtf8(ws_fname_final).c_str() << "\n";
  if (_wunlink(ws_fname_final.c_str()) != 0) {
  //if (_wremove(file_name.c_str()) != 0) {
    // while (int fh1 = _wopen(file_name.c_str(), _O_RDONLY == 0) {
    //   _close( fh1 );
    // }
    result = IOError("Failed to delete a file: " + fname, errno);
  }
  // ::Sleep(100); // allow actual deletion to finish
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

static std::string windows_exec(const char* cmd) {
    std::array<char, 128> buffer;
    std::string result;
    std::unique_ptr<FILE, decltype(&_pclose)> pipe(_popen(cmd, "r"), _pclose);
    if (!pipe) {
        throw std::runtime_error("popen() failed!");
    }
    while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
        result += buffer.data();
    }
    return result;
}

/*
static bool _DeleteDirectory(LPCWSTR lpszDir, bool noRecycleBin = true)
{
    int len = wcslen(lpszDir);
    WCHAR* pszFrom = new WCHAR[len+2]; //4 to handle wide char
    //_tcscpy(pszFrom, lpszDir); //todo:remove warning//;//convet wchar to char*
    wcscpy_s (pszFrom, len+2, lpszDir);
    pszFrom[len] = 0;
    pszFrom[len+1] = 0;

    SHFILEOPSTRUCTW fileop;
    fileop.hwnd   = NULL;    // no status display
    fileop.wFunc  = FO_DELETE;  // delete operation
    fileop.pFrom  = pszFrom;  // source file name as double null terminated string
    fileop.pTo    = NULL;    // no destination needed
    fileop.fFlags = FOF_NOCONFIRMATION|FOF_SILENT;  // do not prompt the user

    if(!noRecycleBin)
        fileop.fFlags |= FOF_ALLOWUNDO;

    fileop.fAnyOperationsAborted = FALSE;
    fileop.lpszProgressTitle     = NULL;
    fileop.hNameMappings         = NULL;

    int ret = SHFileOperationW(&fileop); //SHFileOperation returns zero if successful; otherwise nonzero 
    // int ret = 0;
    delete [] pszFrom;  
    return (0 == ret);
}
*/

Status WindowsFileSystem::DeleteDir(const string& name,
                                    TransactionToken* token) {
  Status result;
  // std::wstring ws_name = Utf8ToWideChar(name);
  // if (_wrmdir(ws_name.c_str()) != 0) {
  //   result = IOError("Failed to remove a directory: " + name, errno);
  // }
  WIN32_FIND_DATAW ffd;
  LARGE_INTEGER filesize;
  // HANDLE hFind = FindFirstFileW(ws_name.c_str(), &ffd);
  LOG(INFO) << "WindowsFileSystem::DeleteDir => " << name.c_str() << "\n";

  std::wstring ws1 = GetUncPathName(TranslateName(name));
  std::string s2( ws1.begin(), ws1.end() );
  string cmd1 = std::string("dir ") + s2;
  LOG(INFO) << cmd1 << " -> " << windows_exec(cmd1.c_str()) << "\n";
  // string cmd2 = cmd1 + "\\..";
  // std::cout << cmd2 << " -> " << windows_exec(cmd2.c_str()) << "\n";
  // cmd2 = "cd";
  // std::cout << cmd2 << " -> " << windows_exec(cmd2.c_str()) << "\n";

  // if (ffd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)
  // {
  //     _tprintf(TEXT("  [%s]   <DIR>\n"), ffd.cFileName);
  // }
  // else
  // {
  //     filesize.LowPart = ffd.nFileSizeLow;
  //     filesize.HighPart = ffd.nFileSizeHigh;
  //     _tprintf(TEXT("  %s   %ld bytes\n"), ffd.cFileName, filesize.QuadPart);
  // }

  if (RemoveDirectoryW (ws1.c_str()) == 0) {
  // if (RemoveDirectoryW (ws1.c_str()) == 0) {
  // if (_DeleteDirectory (ws1.c_str()) == 0) {
    DWORD lastError = ::GetLastError();
    LOG(ERROR) << "RemoveDirectoryW FAILED !!! " << cmd1 << " -> " << windows_exec(cmd1.c_str()) << "\n" << "lastError=" << lastError << "\n"; 
    result = IOError("Failed to remove a directory: " + name, lastError);
  }
  return result;
}


/// <summary>Deletes a directory and everything in it</summary>
/// <param name="path">Path of the directory that will be deleted</param>
int _DelDirRecursive(const std::wstring &path) {
  std::vector<std::wstring::value_type> doubleNullTerminatedPath;
  std::copy(path.begin(), path.end(), std::back_inserter(doubleNullTerminatedPath));
  doubleNullTerminatedPath.push_back(L'\0');
  doubleNullTerminatedPath.push_back(L'\0');
 
  SHFILEOPSTRUCTW fileOperation;
  fileOperation.wFunc = FO_DELETE;
  fileOperation.pFrom = &doubleNullTerminatedPath[0];
  fileOperation.fFlags = FOF_NO_UI | FOF_NOCONFIRMATION;
 
  int result = SHFileOperationW(&fileOperation);
  return result;
}

// See https://github.com/microsoft/Windows-classic-samples/blob/main/Samples/Win7Samples/winui/shell/appplatform/fileoperations/FileOperationSample.cpp
HRESULT CreateAndInitializeFileOperation(REFIID riid, void **ppv)
{
    *ppv = NULL;
    // Create the IFileOperation object
    IFileOperation *pfo;
    HRESULT hr = CoCreateInstance(__uuidof(FileOperation), NULL, CLSCTX_ALL, IID_PPV_ARGS(&pfo));
    if (SUCCEEDED(hr))
    {
        // Set the operation flags.  Turn off  all UI
        // from being shown to the user during the
        // operation.  This includes error, confirmation
        // and progress dialogs.
        hr = pfo->SetOperationFlags(FOF_NO_UI);
        if (SUCCEEDED(hr))
        {
            hr = pfo->QueryInterface(riid, ppv);
        }
        pfo->Release();
    }
    return hr;
}

int _DelDirRecursive2(const std::wstring &dirname) {
  IShellItem *pSI;
  // Creates and initializes a Shell item object from a parsing name
  // When this method returns successfully, pSI contains the interface pointer requested in REFIID. This is typically IShellItem or IShellItem2.
  HRESULT hr = SHCreateItemFromParsingName(dirname.c_str(), NULL, IID_IShellItem, (void**) &pSI);
  if(SUCCEEDED(hr)) {
    IFileOperation *pfo;
    HRESULT hr = CreateAndInitializeFileOperation(IID_PPV_ARGS(&pfo));
    if (SUCCEEDED(hr))
    {
        hr = pfo->DeleteItem(pSI, NULL);
        if (SUCCEEDED(hr))
        {
          hr = pfo->PerformOperations();
        } else {
          LOG(ERROR) << "_DelDirRecursive2 => DeleteItem FAILED for " << WideCharToUtf8(dirname).c_str() << "\n";
        }
        pfo->Release();
    } else {
      LOG(ERROR) << "_DelDirRecursive2 => CreateAndInitializeFileOperation FAILED for " << WideCharToUtf8(dirname).c_str() << "\n";
    }
  }
  return (int)hr;
}


/// <summary>Automatically closes a search handle upon destruction</summary>
class SearchHandleScope {
 
  /// <summary>Initializes a new search handle closer</summary>
  /// <param name="searchHandle">Search handle that will be closed on destruction</param>
  public: SearchHandleScope(HANDLE searchHandle) :
    searchHandle(searchHandle) {}
 
  /// <summary>Closes the search handle</summary>
  public: ~SearchHandleScope() {
    ::FindClose(this->searchHandle);
  }
 
  /// <summary>Search handle that will be closed when the instance is destroyed</summary>
  private: HANDLE searchHandle;
 
};
 
/// <summary>Recursively deletes the specified directory and all its contents</summary>
/// <param name="path">Absolute path of the directory that will be deleted</param>
/// <remarks>
///   The path must not be terminated with a path separator.
/// </remarks>
void recursiveDeleteDirectory(const std::wstring &path) {
  static const std::wstring allFilesMask(L"\\*");
 
  WIN32_FIND_DATAW findData;
 
  // First, delete the contents of the directory, recursively for subdirectories
  std::wstring searchMask = path + allFilesMask;
  HANDLE searchHandle = ::FindFirstFileExW(
    searchMask.c_str(), FindExInfoBasic, &findData, FindExSearchNameMatch, nullptr, 0
  );
  if(searchHandle == INVALID_HANDLE_VALUE) {
    DWORD lastError = ::GetLastError();
    if(lastError != ERROR_FILE_NOT_FOUND) { // or ERROR_NO_MORE_FILES, ERROR_NOT_FOUND?
      throw std::runtime_error("Could not start directory enumeration");
    }
  }
 
  // Did this directory have any contents? If so, delete them first
  if(searchHandle != INVALID_HANDLE_VALUE) {
    SearchHandleScope scope(searchHandle);
    for(;;) {
 
      // Do not process the obligatory '.' and '..' directories
      if(findData.cFileName[0] != '.') {
        bool isDirectory = 
          ((findData.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) != 0) ||
          ((findData.dwFileAttributes & FILE_ATTRIBUTE_REPARSE_POINT) != 0);
 
        // Subdirectories need to be handled by deleting their contents first
        std::wstring filePath = path + L'\\' + findData.cFileName;
        if(isDirectory) {
          recursiveDeleteDirectory(filePath);
        } else {
          BOOL result = ::DeleteFileW(filePath.c_str());
          if(result == FALSE) {
            DWORD lastError = ::GetLastError();
            std::string s2( filePath.begin(), filePath.end() );
            string cmd1 = std::string("dir ") + s2;
            // std::cout << cmd1 << " -> " << windows_exec(cmd1.c_str()) << "\n";
            LOG(ERROR) << cmd1 << " , DeleteFileW FAIL -> " << windows_exec(cmd1.c_str()) << "\n" << "<-- lastError=" << lastError << "\n";
            throw std::runtime_error("Could not delete file");
          }
        }
      }
 
      // Advance to the next file in the directory
      BOOL result = ::FindNextFileW(searchHandle, &findData);
      if(result == FALSE) {
        DWORD lastError = ::GetLastError();
        if(lastError != ERROR_NO_MORE_FILES) {
          throw std::runtime_error("Error enumerating directory");
        }
        break; // All directory contents enumerated and deleted
      }
 
    } // for
  }
 
  // The directory is empty, we can now safely remove it
  BOOL result = ::RemoveDirectoryW(path.c_str());
  if(result == FALSE) {
    DWORD lastError = ::GetLastError();
    std::string s2( path.begin(), path.end() );
    string cmd1 = std::string("dir ") + s2;
    // std::cout << cmd1 << " -> " << windows_exec(cmd1.c_str()) << "\n";
    LOG(ERROR) << cmd1 << " , RemoveDirectoryW FAIL -> " << windows_exec(cmd1.c_str()) << "\n" << "<-- lastError=" << lastError << "\n";
    // result = IOError("Failed to remove a directory 818: " + dirname, errno);
    throw std::runtime_error("Could not remove directory");
  }
}


Status WindowsFileSystem::DeleteRecursively(const std::string& dirname,
                                        TransactionToken* token,
                                        int64_t* undeleted_files,
                                        int64_t* undeleted_dirs) {
  Status result;
  LOG(INFO) << "WindowsFileSystem::DeleteRecursively => " << dirname.c_str() << "\n";

  std::wstring ws1 = GetUncPathName(TranslateName(dirname));
  // DEBUG
  // std::string s2( ws1.begin(), ws1.end() );
  // string cmd1 = std::string("dir ") + s2;
  // std::cout << cmd1 << " -> " << windows_exec(cmd1.c_str()) << "\n";


  // IShellItem *pSI;
  // HRESULT hr = SHCreateItemFromParsingName(ws1.c_str(), NULL, IID_IShellItem, (void**) &pSI);
  // if(SUCCEEDED(hr)) {

  // }

  // if (RemoveDirectoryW (ws1.c_str()) == 0) {
  // if (_DeleteDirectory (ws1.c_str()) == 0) {
  // if (_DelDirRecursive2 (ws1) != 0) {
  //   std::cout << cmd1 << " FAIL -> " << windows_exec(cmd1.c_str()) << "\n" << "errno=" << errno << "\n";
  //   result = IOError("Failed to remove a directory: " + dirname, errno);
  // }
  // Sleep(1000);
  // std::cout << cmd1 << " PASS 1 -> " << windows_exec(cmd1.c_str()) << "\n" << "errno=" << errno << "\n";
  // DeleteDir(dirname, NULL);
  // std::cout << cmd1 << " PASS 2 -> " << windows_exec(cmd1.c_str()) << "\n" << "errno=" << errno << "\n";

  std::string dirname_final( ws1.begin(), ws1.end() );
  return FileSystem::DeleteRecursively(dirname_final, token, undeleted_files, undeleted_dirs);
  // ALTERNATE
  // recursiveDeleteDirectory(ws1);
  // result = OkStatus();
  // return result;
}

Status WindowsFileSystem::GetFileSize(const string& fname,
                                      TransactionToken* token, uint64* size) {
  string translated_fname = TranslateName(fname);
  std::wstring ws_translated_fname = Utf8ToWideChar(translated_fname);
  std::wstring ws_final_fname = GetSymbolicLinkTarget(ws_translated_fname);
  Status result;
  WIN32_FILE_ATTRIBUTE_DATA attrs;
  LOG(INFO) << "WindowsFileSystem::GetFileSize 0 => " << WideCharToUtf8(ws_final_fname).c_str();
  // std::cout << "WindowsFileSystem::GetFileSize 1 => " << fname << "\n";
  if (TRUE == ::GetFileAttributesExW(ws_final_fname.c_str(),
                                     GetFileExInfoStandard, &attrs)) {
    ULARGE_INTEGER file_size;
    file_size.HighPart = attrs.nFileSizeHigh;
    file_size.LowPart = attrs.nFileSizeLow;
    *size = file_size.QuadPart;
    LOG(INFO) << "WindowsFileSystem::GetFileSize 1 => " << (*size) << "\n";
  } else {
    LOG(INFO) << "WindowsFileSystem::GetFileSize 2 FAILED !!!\n";
    string context = "Can not get size for: " + fname;
    result = IOErrorFromWindowsError(context);
  }
  return result;
}

Status WindowsFileSystem::IsDirectory(const string& fname,
                                      TransactionToken* token) {
  std::wstring ws_translated_fname = Utf8ToWideChar(TranslateName(fname));
  std::wstring ws_final_fname = GetSymbolicLinkTarget(ws_translated_fname);
  std::string str_final_fname( ws_final_fname.begin(), ws_final_fname.end() );
  TF_RETURN_IF_ERROR(FileExists(str_final_fname));
  if (PathIsDirectoryW(ws_final_fname.c_str())) {
    return OkStatus();
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

  return OkStatus();
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
  LOG(INFO) << "GetMatchingPaths pattern=" << pattern;
  string converted_pattern(pattern);
  std::replace(converted_pattern.begin(), converted_pattern.end(), '\\', '/');
  TF_RETURN_IF_ERROR(internal::GetMatchingPaths(this, Env::Default(),
                                                converted_pattern, results));
  for (string& result : *results) {
    std::replace(result.begin(), result.end(), '/', '\\');
  }
  return OkStatus();
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
