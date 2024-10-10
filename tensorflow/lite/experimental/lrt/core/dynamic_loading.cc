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

#include "tensorflow/lite/experimental/lrt/core/dynamic_loading.h"

#include <dlfcn.h>
#ifndef __ANDROID__
#include <link.h>
#endif

#include <iostream>
#include <sstream>
#include <string>

#include "absl/strings/string_view.h"
#include "tensorflow/lite/experimental/lrt/c/lite_rt_common.h"
#include "tensorflow/lite/experimental/lrt/core/logging.h"

namespace lrt {

// TODO make this return a status and have handle be output param.
LrtStatus OpenLib(absl::string_view so_path, void** lib_handle) {
#ifdef __ANDROID__
  void* res = ::dlopen(so_path.data(), RTLD_NOW | RTLD_LOCAL);
#else
  void* res = ::dlopen(so_path.data(), RTLD_NOW | RTLD_LOCAL | RTLD_DEEPBIND);
#endif

  if (res == nullptr) {
    LITE_RT_LOG(LRT_ERROR,
                "Failed to load .so at path: %s, with error:\n\t %s\n", so_path,
                ::dlerror());

    return kLrtStatusDynamicLoadErr;
  }
  *lib_handle = res;
  return kLrtStatusOk;
}

void DumpLibInfo(void* lib_handle) {
#ifndef __ANDROID__
  std::stringstream dump;
  dump << "\n--- Lib Info ---\n";

  Lmid_t dl_ns_idx;
  if (0 != ::dlinfo(lib_handle, RTLD_DI_LMID, &dl_ns_idx)) {
    return;
  }

  std::string dl_origin;
  dl_origin.resize(512);
  if (0 != ::dlinfo(lib_handle, RTLD_DI_ORIGIN, dl_origin.data())) {
    return;
  }

  link_map* lm;
  if (0 != ::dlinfo(lib_handle, RTLD_DI_LINKMAP, &lm)) {
    return;
  }

  dump << "Lib Namespace: " << dl_ns_idx << "\n";
  dump << "Lib Origin: " << dl_origin << "\n";

  dump << "loaded objects:\n";

  auto* forward = lm->l_next;
  auto* backward = lm->l_prev;

  while (forward != nullptr) {
    dump << "  " << forward->l_name << "\n";
    forward = forward->l_next;
  }

  dump << "***" << lm->l_name << "\n";

  while (backward != nullptr) {
    dump << "  " << backward->l_name << "\n";
    backward = backward->l_prev;
  }

  dump << "\n";
  std::cerr << dump.str();
#endif
}

}  // namespace lrt
