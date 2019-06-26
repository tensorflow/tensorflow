#ifndef TENSORFLOW_CONTRIB_SEASTAR_SEASTAR_TENSOR_CODING_H_
#define TENSORFLOW_CONTRIB_SEASTAR_SEASTAR_TENSOR_CODING_H_

#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
struct SeastarBuf {
  uint64_t len_ = 0;
  char* data_ = nullptr;
  bool owned_ = true;
};

class SeastarTensorResponse {
 public:
  virtual ~SeastarTensorResponse() {}

  void SetIsDead(bool is_dead) { is_dead_ = is_dead; }
  bool GetIsDead() const { return is_dead_; }

  // for dst device
  void InitAlloc(Device* d, const AllocatorAttributes& aa);
  Allocator* GetAlloc() { return allocator_; }
  AllocatorAttributes GetAllocAttributes() { return alloc_attrs_; }
  Device* GetDevice() const { return device_; }
  bool GetOnHost() const { return on_host_; }

  void SetTensor(const Tensor& tensor) { tensor_ = tensor; }
  const Tensor& GetTensor() const { return tensor_; }

  TensorProto& GetTensorProto() { return tensor_proto_; }

  void Clear();

  void SetDataType(DataType data_type) { data_type_ = data_type; }
  DataType GetDataType() { return data_type_; }

 private:
  bool is_dead_ = false;
  bool on_host_ = false;

  // for dst device
  Device* device_ = nullptr;
  AllocatorAttributes alloc_attrs_;
  Allocator* allocator_ = nullptr;

  Tensor tensor_;
  TensorProto tensor_proto_;
  DataType data_type_;
};
}  // namespace tensorflow

#endif  // TENSORFLOW_CONTRIB_SEASTAR_SEASTAR_TENSOR_CODING_H_
