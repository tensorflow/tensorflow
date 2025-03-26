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

#include "tensorflow/lite/experimental/litert/c/litert_accelerator_compilation_options.h"

#include <cstring>
#include <memory>
#include <string>
#include <utility>

#include "tensorflow/lite/experimental/litert/c/litert_common.h"

struct LiteRtAcceleratorCompilationOptionsT {
  LiteRtApiVersion payload_version;
  std::string payload_identifier;
  std::unique_ptr<void, void (*)(void*)> payload_data;
  LiteRtAcceleratorCompilationOptionsT* next = nullptr;

  LiteRtAcceleratorCompilationOptionsT(const LiteRtApiVersion& payload_version_,
                                       std::string payload_identifier_,
                                       void* payload_data_,
                                       void (*payload_destructor_)(void*))
      : payload_version(payload_version_),
        payload_identifier(std::move(payload_identifier_)),
        payload_data(payload_data_, payload_destructor_) {}
};

LiteRtStatus LiteRtCreateAcceleratorCompilationOptions(
    const LiteRtApiVersion* payload_version, const char* payload_identifier,
    void* payload_data, void (*payload_destructor)(void*),
    LiteRtAcceleratorCompilationOptions* options) {
  if (!payload_version || !payload_identifier || !payload_data ||
      !payload_destructor || !options) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *options = new LiteRtAcceleratorCompilationOptionsT(
      *payload_version, std::string(payload_identifier), payload_data,
      payload_destructor);
  return kLiteRtStatusOk;
}

void LiteRtDestroyAcceleratorCompilationOptions(
    LiteRtAcceleratorCompilationOptions options) {
  while (options) {
    LiteRtAcceleratorCompilationOptions next = options->next;
    delete options;
    options = next;
  }
}

LiteRtStatus LiteRtGetAcceleratorCompilationOptionsVersion(
    LiteRtAcceleratorCompilationOptions options,
    LiteRtApiVersion* payload_version) {
  if (!options || !payload_version) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *payload_version = options->payload_version;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetAcceleratorCompilationOptionsIdentifier(
    LiteRtAcceleratorCompilationOptions options,
    const char** payload_identifier) {
  if (!options || !payload_identifier) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *payload_identifier = options->payload_identifier.c_str();
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetAcceleratorCompilationOptionsData(
    LiteRtAcceleratorCompilationOptions options, void** payload_data) {
  if (!options || !payload_data) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *payload_data = options->payload_data.get();
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtFindAcceleratorCompilationOptionsData(
    LiteRtAcceleratorCompilationOptions options, const char* payload_identifier,
    LiteRtApiVersion* payload_version, void** payload_data) {
  if (!options || !payload_identifier || !payload_version || !payload_data) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  while (options) {
    if (!strcmp(options->payload_identifier.c_str(), payload_identifier)) {
      *payload_version = options->payload_version;
      *payload_data = options->payload_data.get();
      return kLiteRtStatusOk;
    } else {
      options = options->next;
    }
  }
  return kLiteRtStatusErrorNotFound;
}

LiteRtStatus LiteRtGetNextAcceleratorCompilationOptions(
    LiteRtAcceleratorCompilationOptions* options) {
  if (!options || !*options) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *options = (*options)->next;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtAppendAcceleratorCompilationOptions(
    LiteRtAcceleratorCompilationOptions* options,
    LiteRtAcceleratorCompilationOptions appended_options) {
  if (!options || !appended_options) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  while (*options) {
    options = &((*options)->next);
  }
  *options = appended_options;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtPopAcceleratorCompilationOptions(
    LiteRtAcceleratorCompilationOptions* options) {
  if (!options) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  LiteRtAcceleratorCompilationOptions* last = options;
  while ((*last)->next) {
    last = &(*last)->next;
  }
  if (*last) {
    LiteRtDestroyAcceleratorCompilationOptions(*last);
    *last = nullptr;
  }
  return kLiteRtStatusOk;
}
