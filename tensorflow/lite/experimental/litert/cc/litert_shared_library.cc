// Copyright 2025 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "tensorflow/lite/experimental/litert/cc/litert_shared_library.h"

#include <dlfcn.h>
#if defined(_GNU_SOURCE) && !defined(__ANDROID__) && !defined(__APPLE__)
#define LITERT_IMPLEMENT_SHARED_LIBRARY_INFO 1
#include <link.h>
#endif

#include <ostream>
#include <string>
#include <utility>

#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/cc/litert_expected.h"

namespace litert {

SharedLibrary::~SharedLibrary() noexcept { Close(); }

SharedLibrary::SharedLibrary(SharedLibrary&& other) noexcept
    : handle_kind_(other.handle_kind_),
      path_(std::move(other.path_)),
      handle_(other.handle_) {
  other.handle_kind_ = HandleKind::kInvalid;
  other.handle_ = nullptr;
}

SharedLibrary& SharedLibrary::operator=(SharedLibrary&& other) noexcept {
  Close();
  handle_kind_ = other.handle_kind_;
  path_ = std::move(other.path_);
  handle_ = other.handle_;
  other.handle_kind_ = HandleKind::kInvalid;
  other.handle_ = nullptr;
  return *this;
}

void SharedLibrary::Close() noexcept {
  if (handle_kind_ == HandleKind::kPath) {
    dlclose(handle_);
  }
  handle_kind_ = HandleKind::kInvalid;
  path_.clear();
}

absl::string_view SharedLibrary::DlError() noexcept {
  const char* error = dlerror();
  if (!error) {
    return {};
  }
  return error;
}

Expected<SharedLibrary> SharedLibrary::LoadImpl(
    SharedLibrary::HandleKind handle_kind, absl::string_view path,
    RtldFlags flags) {
  SharedLibrary lib;
  switch (handle_kind) {
    case HandleKind::kInvalid:
      return Error(kLiteRtStatusErrorDynamicLoading,
                   "This is a logic error. LoadImpl should not be called with "
                   "HandleKind::kInvalid");
    case HandleKind::kPath:
      if (path.empty()) {
        return Error(kLiteRtStatusErrorDynamicLoading,
                     "Cannot not load shared library: empty path.");
      }
      lib.path_ = path;
      lib.handle_ = dlopen(lib.Path().c_str(), flags);
      if (!lib.handle_) {
        return Error(kLiteRtStatusErrorDynamicLoading,
                     absl::StrFormat("Could not load shared library %s: %s.",
                                     lib.path_, DlError()));
      }
      break;
    case HandleKind::kRtldNext:
      lib.handle_ = RTLD_NEXT;
      break;
    case HandleKind::kRtldDefault:
      lib.handle_ = RTLD_DEFAULT;
      break;
  }
  lib.handle_kind_ = handle_kind;
  return lib;
}

Expected<void*> SharedLibrary::LookupSymbolImpl(const char* symbol_name) const {
  void* symbol = dlsym(handle_, symbol_name);
  if (!symbol) {
    return Error(kLiteRtStatusErrorDynamicLoading,
                 absl::StrFormat("Could not load symbol %s: %s.", symbol_name,
                                 DlError()));
  }
  return symbol;
}

std::ostream& operator<<(std::ostream& os, const SharedLibrary& lib) {
  static constexpr absl::string_view kHeader = "/// DLL Info ///\n";
  static constexpr absl::string_view kFooter = "////////////////\n";

  if (lib.handle_ == nullptr) {
    os << kHeader << "Handle is nullptr.\n" << kFooter;
    return os;
  }

  os << kHeader;
#if LITERT_IMPLEMENT_SHARED_LIBRARY_INFO
  if (Lmid_t dl_ns_idx; dlinfo(lib.handle_, RTLD_DI_LMID, &dl_ns_idx) != 0) {
    os << "Error getting lib namespace index: " << dlerror() << ".\n";
  } else {
    os << "LIB NAMESPACE INDEX: " << dl_ns_idx << "\n";
  }

  if (link_map* lm; dlinfo(lib.handle_, RTLD_DI_LINKMAP, &lm) != 0) {
    os << "Error getting linked objects: " << dlerror() << ".\n";
  } else {
    os << "LINKED OBJECTS:\n";
    // Rewind to the start of the linked list.
    const link_map* link = lm;
    while (link->l_prev) {
      link = link->l_prev;
    }
    // Print all list elements
    for (; link != nullptr; link = link->l_next) {
      os << (link != lm ? "   " : "***") << link->l_name << "\n";
    }
  }
#else
  os << "Unsupported platform.\n";
#endif
  return os << kFooter;
}

}  // namespace litert
