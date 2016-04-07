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

#include "tensorflow/core/platform/file_system.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/lib/gtl/stl_util.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/protobuf.h"

namespace tensorflow {

FileSystem::~FileSystem() {}

string FileSystem::TranslateName(const string& name) const { return name; }

RandomAccessFile::~RandomAccessFile() {}

WritableFile::~WritableFile() {}

FileSystemRegistry::~FileSystemRegistry() {}

class FileSystemRegistryImpl : public FileSystemRegistry {
 public:
  void Register(const string& scheme, Factory factory) override;
  FileSystem* Lookup(const string& scheme) override;

 private:
  mutable mutex mu_;
  mutable std::unordered_map<string, FileSystem*> registry_ GUARDED_BY(mu_);
};

FileSystemRegistry* GlobalFileSystemRegistry() {
  static FileSystemRegistry* registry = new FileSystemRegistryImpl;
  return registry;
}

void FileSystemRegistryImpl::Register(const string& scheme,
                                      FileSystemRegistry::Factory factory) {
  mutex_lock lock(mu_);
  QCHECK(!gtl::FindOrNull(registry_, scheme)) << "File factory for " << scheme
                                              << " already registered";
  registry_[scheme] = factory();
}

FileSystem* FileSystemRegistryImpl::Lookup(const string& scheme) {
  mutex_lock lock(mu_);
  auto fs_ptr = gtl::FindOrNull(registry_, scheme);
  if (!fs_ptr) {
    return nullptr;
  }
  return *fs_ptr;
}

string GetSchemeFromURI(const string& name) {
  auto colon_loc = name.find(":");
  if (colon_loc != string::npos) {
    return name.substr(0, colon_loc);
  }
  return "";
}

string GetNameFromURI(const string& name) {
  string scheme = GetSchemeFromURI(name);
  if (scheme == "") {
    return name;
  }
  // Skip the 'scheme:' portion.
  StringPiece filename{name.data() + scheme.length() + 1,
                       name.length() - scheme.length() - 1};
  // If the URI confirmed to scheme://filename, skip the two '/'s and return
  // filename. Otherwise return the original 'name', and leave it up to the
  // implementations to handle the full URI.
  if (filename[0] == '/' && filename[1] == '/') {
    return filename.substr(2).ToString();
  }
  return name;
}

}  // namespace tensorflow
