#ifndef TENSORFLOW_CONTRIB_SEASTAR_SEASTAR_MESSAGE_H_
#define TENSORFLOW_CONTRIB_SEASTAR_SEASTAR_MESSAGE_H_

#include "tensorflow/core/framework/tensor_shape.h"

namespace tensorflow {

// message for recv tensor response
struct SeastarMessage {
  bool is_dead_;
  DataType data_type_;
  TensorShape tensor_shape_;
  uint64_t tensor_bytes_;

  // |is_dead|...
  // |    1B |...
  // ...|data_type|tensor_shape|tensor_bytes|tensor_buffer
  // ...|   XB    |    XB      |    8B      |...

  static const size_t kIsDeadStartIndex = 0;
  static const size_t kDataTypeStartIndex =
      kIsDeadStartIndex + sizeof(is_dead_);
  static const size_t kTensorShapeStartIndex =
      kDataTypeStartIndex + sizeof(data_type_);
  static const size_t kTensorBytesStartIndex =
      kTensorShapeStartIndex + sizeof(TensorShape);
  static const size_t kTensorBufferStartIndex =
      kTensorBytesStartIndex + sizeof(tensor_bytes_);
  static const size_t kMessageTotalBytes = kTensorBufferStartIndex;
  static const size_t kSeastarMessageBufferSize = kMessageTotalBytes;
  static void SerializeMessage(const SeastarMessage& rm, char* data);
  static void DeserializeMessage(SeastarMessage* rm, const char* data);
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CONTRIB_SEASTAR_SEASTAR_MESSAGE_H_
