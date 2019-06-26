#include "tensorflow/contrib/seastar/seastar_message.h"

namespace tensorflow {

void SeastarMessage::DeserializeMessage(SeastarMessage* sm,
                                        const char* message) {
  memcpy(&sm->is_dead_, &message[kIsDeadStartIndex], sizeof(sm->is_dead_));
  memcpy(&sm->data_type_, &message[kDataTypeStartIndex],
         sizeof(sm->data_type_));
  memcpy(&sm->tensor_shape_, &message[kTensorShapeStartIndex],
         sizeof(sm->tensor_shape_));
  memcpy(&sm->tensor_bytes_, &message[kTensorBytesStartIndex],
         sizeof(sm->tensor_bytes_));
}

void SeastarMessage::SerializeMessage(const SeastarMessage& sm, char* message) {
  memcpy(&message[kIsDeadStartIndex], &sm.is_dead_, sizeof(sm.is_dead_));
  memcpy(&message[kDataTypeStartIndex], &sm.data_type_, sizeof(sm.data_type_));
  memcpy(&message[kTensorShapeStartIndex], &sm.tensor_shape_,
         sizeof(sm.tensor_shape_));
  memcpy(&message[kTensorBytesStartIndex], &sm.tensor_bytes_,
         sizeof(sm.tensor_bytes_));
}

}  // namespace tensorflow
