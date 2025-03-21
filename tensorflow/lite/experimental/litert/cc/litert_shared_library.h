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

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_LITERT_CC_LITERT_SHARED_LIBRARY_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_LITERT_CC_LITERT_SHARED_LIBRARY_H_

#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || \
    defined(__NT__) || defined(_WIN64)
#define LITERT_WINDOWS_OS 1
#endif

#if !LITERT_WINDOWS_OS
#include <dlfcn.h>
#endif

#include <ostream>
#include <string>

#include "absl/strings/string_view.h"
#include "tensorflow/lite/experimental/litert/cc/litert_expected.h"
#include "tensorflow/lite/experimental/litert/cc/litert_macros.h"

namespace litert {

struct RtldFlags {
  int flags;

  static constexpr struct NextTag {
  } kNext;
  static constexpr struct DefaultTag {
  } kDefault;

  // NOLINTNEXTLINE(*-explicit-constructor): we want this to be passed as flags.
  operator int() { return flags; }

  static constexpr RtldFlags Lazy() {
    return {
#if defined(RTLD_LAZY)
        RTLD_LAZY
#endif
    };
  }
  static constexpr RtldFlags Now() {
    return {
#if defined(RTLD_NOW)
        RTLD_NOW
#endif
    };
  }
  static constexpr RtldFlags Default() { return Lazy().Local().DeepBind(); }
  constexpr RtldFlags& Global() {
#if defined(RTLD_GLOBAL)
    flags |= RTLD_GLOBAL;
#endif
    return *this;
  }
  constexpr RtldFlags& Local() {
#if defined(RTLD_LOCAL)
    flags |= RTLD_LOCAL;
#endif
    return *this;
  }
  constexpr RtldFlags& NoDelete() {
#if defined(RTLD_NODELETE)
    flags |= RTLD_NODELETE;
#endif
    return *this;
  }
  constexpr RtldFlags& NoLoad() {
#if defined(RTLD_NOLOAD)
    flags |= RTLD_NOLOAD;
#endif
    return *this;
  }
  constexpr RtldFlags& DeepBind() {
#if defined(RTLD_DEEPBIND)
    flags |= RTLD_DEEPBIND;
#endif
    return *this;
  }
};

// Wraps a dynamically loaded shared library to offer RAII semantics.
class SharedLibrary {
 public:
  SharedLibrary() = default;
  SharedLibrary(const SharedLibrary&) = delete;
  SharedLibrary& operator=(const SharedLibrary&) = delete;
  SharedLibrary(SharedLibrary&&) noexcept;
  SharedLibrary& operator=(SharedLibrary&&) noexcept;
  ~SharedLibrary() noexcept;

  // Loads the library at the given path.
  static Expected<SharedLibrary> Load(absl::string_view path,
                                      RtldFlags flags) noexcept {
    return LoadImpl(HandleKind::kPath, path, flags);
  }

  // Loads the library as the RTLD_NEXT special handle.
  static Expected<SharedLibrary> Load(RtldFlags::NextTag) noexcept {
    return LoadImpl(HandleKind::kRtldNext, "", RtldFlags{});
  }

  // Loads the library as the RTLD_DEFAULT special handle.
  static Expected<SharedLibrary> Load(RtldFlags::DefaultTag) noexcept {
    return LoadImpl(HandleKind::kRtldDefault, "", RtldFlags{});
  }

  // Gets the last shared library operation error if there was one.
  //
  // If there was no error, returns an empty view.
  static absl::string_view DlError() noexcept;

  friend std::ostream& operator<<(std::ostream& os, const SharedLibrary& lib);

  bool Loaded() const noexcept { return handle_kind_ != HandleKind::kInvalid; }

  // Unloads the shared library.
  //
  // Note: this is automatically done when the object is destroyed.
  void Close() noexcept;

  // Looks up a symbol in the shared library.
  //
  // Note: This takes a `char*` because the underlying system call requires a
  // null terminated string which a string view doesn't guarantee.
  template <class T>
  Expected<T> LookupSymbol(const char* symbol) const noexcept {
    static_assert(std::is_pointer_v<T>,
                  "The template parameter should always be a pointer.");
    LITERT_ASSIGN_OR_RETURN(void* const raw_symbol, LookupSymbolImpl(symbol));
    return reinterpret_cast<T>(raw_symbol);
  }

  // Returns the loaded library path.
  const std::string& Path() const noexcept { return path_; }

  // Returns the underlying shared library handle.
  //
  // Warning: some special handle value may be NULL. Do not rely on this value
  // to check whether a library is loaded or not.
  const void* Handle() const noexcept { return handle_; }
  void* Handle() noexcept { return handle_; }

 private:
  enum class HandleKind { kInvalid, kPath, kRtldNext, kRtldDefault };

  static Expected<SharedLibrary> LoadImpl(HandleKind handle_kind,
                                          absl::string_view path,
                                          RtldFlags flags);

  Expected<void*> LookupSymbolImpl(const char* symbol) const;

  HandleKind handle_kind_ = HandleKind::kInvalid;
  std::string path_;
  void* handle_ = nullptr;
};

}  // namespace litert

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LITERT_CC_LITERT_SHARED_LIBRARY_H_
