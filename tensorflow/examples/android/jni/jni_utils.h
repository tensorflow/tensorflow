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

#ifndef ORG_TENSORFLOW_JNI_JNI_UTILS_H_  // NOLINT
#define ORG_TENSORFLOW_JNI_JNI_UTILS_H_  // NOLINT

#include <jni.h>
#include <string>
#include <vector>

#include "tensorflow/core/platform/types.h"

namespace google {
namespace protobuf {
class MessageLite;
}  // google
}  // protobuf

class AAssetManager;

bool PortableReadFileToProto(const std::string& file_name,
                             ::google::protobuf::MessageLite* proto);

void ReadFileToProto(AAssetManager* const asset_manager,
    const char* const filename, google::protobuf::MessageLite* message);

void ReadFileToString(AAssetManager* const asset_manager,
    const char* const filename, std::string* str);

void ReadFileToVector(AAssetManager* const asset_manager,
    const char* const filename, std::vector<std::string>* str_vector);

void WriteProtoToFile(const char* const filename,
                      const google::protobuf::MessageLite& message);

#endif  // ORG_TENSORFLOW_JNI_JNI_UTILS_H_
