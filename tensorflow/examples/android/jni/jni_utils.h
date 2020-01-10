#ifndef ORG_TENSORFLOW_JNI_JNI_UTILS_H_  // NOLINT
#define ORG_TENSORFLOW_JNI_JNI_UTILS_H_  // NOLINT

#include <jni.h>
#include <string>
#include <vector>

#include "tensorflow/core/platform/port.h"

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

#endif  // ORG_TENSORFLOW_JNI_JNI_UTILS_H_
