// Copyright 2024 Google LLC.
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

#include "tensorflow/lite/experimental/lrt/qnn/load_sdk.h"

#include <dlfcn.h>
#include <link.h>

#include <iostream>
#include <string>

#include "absl/strings/string_view.h"

namespace qnn::load {

void DumpDlInfo(void* handle) {
  std::cerr << "--- Dyn Load Info ---\n";

  Lmid_t dl_ns_idx;
  if (0 != ::dlinfo(handle, RTLD_DI_LMID, &dl_ns_idx)) {
    return;
  }

  std::string dl_origin;
  dl_origin.resize(245);
  if (0 != ::dlinfo(handle, RTLD_DI_ORIGIN, dl_origin.data())) {
    return;
  }

  link_map* lm;
  if (0 != ::dlinfo(handle, RTLD_DI_LINKMAP, &lm)) {
    return;
  }

  std::cerr << "DL namespace: " << dl_ns_idx << "\n";
  std::cerr << "DL origin: " << dl_origin << "\n";

  std::cerr << "loaded objects:\n";

  auto* forward = lm->l_next;
  auto* backward = lm->l_prev;

  while (forward != nullptr) {
    std::cerr << "  " << forward->l_name << "\n";
    forward = forward->l_next;
  }

  std::cerr << "*** " << lm->l_name << "\n";

  while (backward != nullptr) {
    std::cerr << "  " << backward->l_name << "\n";
    backward = backward->l_prev;
  }
}

void* LoadSO(absl::string_view so_path) {
  void* lib_handle =
      ::dlopen(so_path.data(), RTLD_NOW | RTLD_LOCAL | RTLD_DEEPBIND);
  if (lib_handle == nullptr) {
    std::cerr << "Failed to load so at path: " << so_path
              << " with err: " << ::dlerror() << "\n";
  }
  return lib_handle;
}

}  // namespace qnn::load
