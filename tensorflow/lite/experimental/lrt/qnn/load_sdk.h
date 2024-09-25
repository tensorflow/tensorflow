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

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_LRT_QNN_LOAD_SDK_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_LRT_QNN_LOAD_SDK_H_

#include <dlfcn.h>
#include <stdlib.h>

#include <iostream>

#include "absl/strings/string_view.h"
#include "third_party/qairt/include/QNN/QnnCommon.h"
#include "third_party/qairt/include/QNN/QnnInterface.h"

#ifndef QNN_SDK_LIB_HTP

// If path not provided, check current directory.
constexpr absl::string_view kLibQnnHtpSo = "libQnnHtp.so";
#else

constexpr absl::string_view kLibQnnHtpSo = QNN_SDK_LIB_HTP;
#endif

namespace qnn::load {

//
// QNN Specific Data and Types
//

// This is one of two qnns symbol that needs sym. It is used to populate
// pointers to related available qnn functions.
constexpr char kLibQnnGetProvidersSymbol[] = "QnnInterface_getProviders";

// Type definition for the QnnInterface_getProviders symbol.
typedef Qnn_ErrorHandle_t (*QnnInterfaceGetProvidersFn_t)(
    const QnnInterface_t*** provider_list, uint32_t* num_providers);

//
// Wrappers for Dynamic Linking
//

// Loads (qnn) shared library at given path, returning handle on success
// and nullptr on failure.
void* LoadSO(absl::string_view so_path);

// Dumps info relavant to dynamic loading of given loaded so handle.
void DumpDlInfo(void* lib_handle);

// Resolves a named symbol from given loaded so handle of type SymbolT. Returns
// nullptr on failure.
template <class SymbolT>
inline static SymbolT ResolveQnnSymbol(void* lib_handle,
                                       absl::string_view symbol) {
  SymbolT ptr = (SymbolT)::dlsym(lib_handle, symbol.data());
  if (ptr == nullptr) {
    std::cerr << "Failed to resolve symbol: " << symbol << " with err "
              << ::dlerror() << "\n";
  }
  return ptr;
}

}  // namespace qnn::load

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LRT_QNN_LOAD_SDK_H_
