#include "tensorflow/core/platform/protobuf.h"

namespace tensorflow {

bool ParseProtoUnlimited(protobuf::Message* proto, const string& serialized) {
  return ParseProtoUnlimited(proto, serialized.data(), serialized.size());
}

bool ParseProtoUnlimited(protobuf::Message* proto, const void* serialized,
                         size_t size) {
  protobuf::io::CodedInputStream coded_stream(
      reinterpret_cast<const uint8*>(serialized), size);
  coded_stream.SetTotalBytesLimit(INT_MAX, INT_MAX);
  return proto->ParseFromCodedStream(&coded_stream);
}

}  // namespace tensorflow
