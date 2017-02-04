/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/c/c_api.h"

#include <algorithm>
#include <limits>
#include <memory>
#include <vector>

#ifndef __ANDROID__
#include "tensorflow/cc/saved_model/loader.h"
#endif
#include "tensorflow/core/common_runtime/shape_refiner.h"
#include "tensorflow/core/framework/log_memory.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/lib/core/coding.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/mem.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/public/version.h"

// The implementation below is at the top level instead of the
// brain namespace because we are defining 'extern "C"' functions.
using tensorflow::error::Code;
using tensorflow::errors::InvalidArgument;
using tensorflow::gtl::ArraySlice;
using tensorflow::AllocationDescription;
using tensorflow::DataType;
using tensorflow::Env;
using tensorflow::Graph;
using tensorflow::GraphDef;
using tensorflow::mutex;
using tensorflow::mutex_lock;
using tensorflow::NameRangeMap;
using tensorflow::NameRangesForNode;
using tensorflow::NewSession;
using tensorflow::Node;
using tensorflow::NodeDef;
using tensorflow::NodeBuilder;
using tensorflow::OpDef;
using tensorflow::OpRegistry;
using tensorflow::PartialTensorShape;
using tensorflow::Reset;
using tensorflow::RunMetadata;
using tensorflow::RunOptions;
using tensorflow::Session;
using tensorflow::SessionOptions;
using tensorflow::Status;
using tensorflow::Tensor;
using tensorflow::TensorBuffer;
using tensorflow::TensorId;
using tensorflow::TensorShape;
using tensorflow::TensorShapeProto;

extern "C" {

// --------------------------------------------------------------------------
const char* TF_Version() { return TF_VERSION_STRING; }

// --------------------------------------------------------------------------
size_t TF_DataTypeSize(TF_DataType dt) {
  return static_cast<size_t>(
      tensorflow::DataTypeSize(static_cast<DataType>(dt)));
}

// --------------------------------------------------------------------------
struct TF_Status {
  Status status;
};

TF_Status* TF_NewStatus() { return new TF_Status; }

void TF_DeleteStatus(TF_Status* s) { delete s; }

void TF_SetStatus(TF_Status* s, TF_Code code, const char* msg) {
  s->status = Status(static_cast<Code>(code), tensorflow::StringPiece(msg));
}

TF_Code TF_GetCode(const TF_Status* s) {
  return static_cast<TF_Code>(s->status.code());
}

const char* TF_Message(const TF_Status* s) {
  return s->status.error_message().c_str();
}

// --------------------------------------------------------------------------

namespace {
class TF_ManagedBuffer : public TensorBuffer {
 public:
  void* data_;
  size_t len_;
  void (*deallocator_)(void* data, size_t len, void* arg);
  void* deallocator_arg_;

  ~TF_ManagedBuffer() override {
    (*deallocator_)(data_, len_, deallocator_arg_);
  }

  void* data() const override { return data_; }
  size_t size() const override { return len_; }
  TensorBuffer* root_buffer() override { return this; }
  void FillAllocationDescription(AllocationDescription* proto) const override {
    tensorflow::int64 rb = size();
    proto->set_requested_bytes(rb);
    proto->set_allocator_name(tensorflow::cpu_allocator()->Name());
  }
};

void* allocate_tensor(const char* operation, size_t len) {
  void* data =
      tensorflow::cpu_allocator()->AllocateRaw(EIGEN_MAX_ALIGN_BYTES, len);
  if (tensorflow::LogMemory::IsEnabled()) {
    tensorflow::LogMemory::RecordRawAllocation(
        operation, tensorflow::LogMemory::EXTERNAL_TENSOR_ALLOCATION_STEP_ID,
        len, data, tensorflow::cpu_allocator());
  }
  return data;
}

void deallocate_buffer(void* data, size_t len, void* arg) {
  if (tensorflow::LogMemory::IsEnabled()) {
    tensorflow::LogMemory::RecordRawDeallocation(
        "TensorFlow C Api",
        tensorflow::LogMemory::EXTERNAL_TENSOR_ALLOCATION_STEP_ID, data,
        tensorflow::cpu_allocator(), false);
  }
  tensorflow::cpu_allocator()->DeallocateRaw(data);
}

Status MessageToBuffer(const tensorflow::protobuf::Message& in,
                       TF_Buffer* out) {
  if (out->data != nullptr) {
    return InvalidArgument("Passing non-empty TF_Buffer is invalid.");
  }
  const auto proto_size = in.ByteSize();
  void* buf = tensorflow::port::Malloc(proto_size);
  in.SerializeToArray(buf, proto_size);
  out->data = buf;
  out->length = proto_size;
  out->data_deallocator = [](void* data, size_t length) {
    tensorflow::port::Free(data);
  };
  return Status::OK();
}

}  // namespace

struct TF_Tensor {
  TF_DataType dtype;
  TensorShape shape;
  TensorBuffer* buffer;
};

TF_Tensor* TF_AllocateTensor(TF_DataType dtype, const int64_t* dims,
                             int num_dims, size_t len) {
  void* data = allocate_tensor("TF_AllocateTensor", len);
  return TF_NewTensor(dtype, dims, num_dims, data, len, deallocate_buffer,
                      nullptr);
}

TF_Tensor* TF_NewTensor(TF_DataType dtype, const int64_t* dims, int num_dims,
                        void* data, size_t len,
                        void (*deallocator)(void* data, size_t len, void* arg),
                        void* deallocator_arg) {
  std::vector<tensorflow::int64> dimvec(num_dims);
  for (int i = 0; i < num_dims; ++i) {
    dimvec[i] = static_cast<tensorflow::int64>(dims[i]);
  }

  TF_ManagedBuffer* buf = new TF_ManagedBuffer;
  buf->len_ = len;
  if (reinterpret_cast<intptr_t>(data) % EIGEN_MAX_ALIGN_BYTES != 0) {
    // Copy the data into a buffer that satisfies Eigen's alignment
    // requirements.
    buf->data_ = allocate_tensor("TF_NewTensor", len);
    std::memcpy(buf->data_, data, len);
    buf->deallocator_ = deallocate_buffer;
    buf->deallocator_arg_ = nullptr;
    // Free the original buffer.
    deallocator(data, len, deallocator_arg);
  } else {
    buf->data_ = data;
    buf->deallocator_ = deallocator;
    buf->deallocator_arg_ = deallocator_arg;
  }
  return new TF_Tensor{dtype, TensorShape(dimvec), buf};
}

void TF_DeleteTensor(TF_Tensor* t) {
  t->buffer->Unref();
  delete t;
}

TF_DataType TF_TensorType(const TF_Tensor* t) { return t->dtype; }
int TF_NumDims(const TF_Tensor* t) { return t->shape.dims(); }
int64_t TF_Dim(const TF_Tensor* t, int dim_index) {
  return static_cast<int64_t>(t->shape.dim_size(dim_index));
}
size_t TF_TensorByteSize(const TF_Tensor* t) { return t->buffer->size(); }
void* TF_TensorData(const TF_Tensor* t) { return t->buffer->data(); }

// --------------------------------------------------------------------------
size_t TF_StringEncode(const char* src, size_t src_len, char* dst,
                       size_t dst_len, TF_Status* status) {
  const size_t sz = TF_StringEncodedSize(src_len);
  if (sz < src_len) {
    status->status = InvalidArgument("src string is too large to encode");
    return 0;
  }
  if (dst_len < sz) {
    status->status =
        InvalidArgument("dst_len (", dst_len, ") too small to encode a ",
                        src_len, "-byte string");
    return 0;
  }
  dst = tensorflow::core::EncodeVarint64(dst, src_len);
  memcpy(dst, src, src_len);
  return sz;
}

size_t TF_StringDecode(const char* src, size_t src_len, const char** dst,
                       size_t* dst_len, TF_Status* status) {
  tensorflow::uint64 len64 = 0;
  const char* p = tensorflow::core::GetVarint64Ptr(src, src + src_len, &len64);
  if (p == nullptr) {
    status->status =
        InvalidArgument("invalid string encoding or truncated src buffer");
    return 0;
  }
  if (len64 > std::numeric_limits<size_t>::max()) {
    status->status =
        InvalidArgument("encoded string is ", len64,
                        "-bytes, which is too large for this architecture");
    return 0;
  }
  *dst = p;
  *dst_len = static_cast<size_t>(len64);
  return static_cast<size_t>(p - src) + *dst_len;
}

size_t TF_StringEncodedSize(size_t len) {
  return static_cast<size_t>(tensorflow::core::VarintLength(len)) + len;
}

// --------------------------------------------------------------------------
struct TF_SessionOptions {
  SessionOptions options;
};
TF_SessionOptions* TF_NewSessionOptions() { return new TF_SessionOptions; }
void TF_DeleteSessionOptions(TF_SessionOptions* opt) { delete opt; }

void TF_SetTarget(TF_SessionOptions* options, const char* target) {
  options->options.target = target;
}

void TF_SetConfig(TF_SessionOptions* options, const void* proto,
                  size_t proto_len, TF_Status* status) {
  if (!options->options.config.ParseFromArray(proto, proto_len)) {
    status->status = InvalidArgument("Unparseable ConfigProto");
  }
}
// --------------------------------------------------------------------------
TF_Buffer* TF_NewBuffer() { return new TF_Buffer{nullptr, 0, nullptr}; }

TF_Buffer* TF_NewBufferFromString(const void* proto, size_t proto_len) {
  void* copy = tensorflow::port::Malloc(proto_len);
  memcpy(copy, proto, proto_len);

  TF_Buffer* buf = new TF_Buffer;
  buf->data = copy;
  buf->length = proto_len;
  buf->data_deallocator = [](void* data, size_t length) {
    tensorflow::port::Free(data);
  };
  return buf;
}

void TF_DeleteBuffer(TF_Buffer* buffer) {
  if (buffer->data_deallocator != nullptr) {
    (*buffer->data_deallocator)(const_cast<void*>(buffer->data),
                                buffer->length);
  }
  delete buffer;
}

TF_Buffer TF_GetBuffer(TF_Buffer* buffer) { return *buffer; }

// --------------------------------------------------------------------------
struct TF_DeprecatedSession {
  Session* session;
};

TF_DeprecatedSession* TF_NewDeprecatedSession(const TF_SessionOptions* opt,
                                              TF_Status* status) {
  Session* session;
  status->status = NewSession(opt->options, &session);
  if (status->status.ok()) {
    return new TF_DeprecatedSession({session});
  } else {
    DCHECK_EQ(nullptr, session);
    return NULL;
  }
}

void TF_CloseDeprecatedSession(TF_DeprecatedSession* s, TF_Status* status) {
  status->status = s->session->Close();
}

void TF_DeleteDeprecatedSession(TF_DeprecatedSession* s, TF_Status* status) {
  status->status = Status::OK();
  delete s->session;
  delete s;
}

void TF_ExtendGraph(TF_DeprecatedSession* s, const void* proto,
                    size_t proto_len, TF_Status* status) {
  GraphDef g;
  if (!tensorflow::ParseProtoUnlimited(&g, proto, proto_len)) {
    status->status = InvalidArgument("Invalid GraphDef");
    return;
  }
  status->status = s->session->Extend(g);
}

static void DeleteArray(void* data, size_t size, void* arg) {
  DCHECK_EQ(data, arg);
  delete[] reinterpret_cast<char*>(arg);
}

}  // end extern "C"

namespace tensorflow {
namespace {

// Reset helper for converting character arrays to string vectors.
void TF_Reset_Helper(const TF_SessionOptions* opt, const char** containers,
                     int ncontainers, TF_Status* status) {
  std::vector<tensorflow::string> container_names(ncontainers);
  for (int i = 0; i < ncontainers; ++i) {
    container_names[i] = containers[i];
  }

  status->status = Reset(opt->options, container_names);
}

}  // namespace
}  // namespace tensorflow

extern "C" {

void TF_Reset(const TF_SessionOptions* opt, const char** containers,
              int ncontainers, TF_Status* status) {
  tensorflow::TF_Reset_Helper(opt, containers, ncontainers, status);
}

}  // end extern "C"

namespace tensorflow {

// Non-static for testing.
bool TF_Tensor_DecodeStrings(TF_Tensor* src, Tensor* dst, TF_Status* status) {
  const tensorflow::int64 num_elements = src->shape.num_elements();
  const char* input = reinterpret_cast<const char*>(TF_TensorData(src));
  const size_t src_size = TF_TensorByteSize(src);
  if (static_cast<tensorflow::int64>(src_size / sizeof(tensorflow::uint64)) <
      num_elements) {
    status->status = InvalidArgument(
        "Malformed TF_STRING tensor; too short to hold number of elements");
    return false;
  }
  const char* data_start = input + sizeof(tensorflow::uint64) * num_elements;
  const char* limit = input + src_size;

  *dst = Tensor(static_cast<DataType>(src->dtype), src->shape);
  auto dstarray = dst->flat<tensorflow::string>();
  for (tensorflow::int64 i = 0; i < num_elements; ++i) {
    tensorflow::uint64 offset =
        reinterpret_cast<const tensorflow::uint64*>(input)[i];
    if (static_cast<ptrdiff_t>(offset) >= (limit - data_start)) {
      status->status = InvalidArgument("Malformed TF_STRING tensor; element ",
                                       i, " out of range");
      return false;
    }
    size_t len;
    const char* p;
    const char* srcp = data_start + offset;
    TF_StringDecode(srcp, limit - srcp, &p, &len, status);
    if (!status->status.ok()) {
      return false;
    }
    dstarray(i).assign(p, len);
  }
  return true;
}

// Non-static for testing.
TF_Tensor* TF_Tensor_EncodeStrings(const Tensor& src) {
  // Compute bytes needed for encoding.
  size_t size = 0;
  const auto& srcarray = src.flat<tensorflow::string>();
  for (int i = 0; i < srcarray.size(); ++i) {
    const tensorflow::string& s = srcarray(i);
    // uint64 starting_offset, TF_StringEncode-d string.
    size += sizeof(tensorflow::uint64) + TF_StringEncodedSize(s.size());
  }

  // Encode all strings.
  char* base = new char[size];
  char* data_start = base + sizeof(tensorflow::uint64) * srcarray.size();
  char* dst = data_start;  // Where next string is encoded.
  size_t dst_len = size - static_cast<size_t>(data_start - base);
  tensorflow::uint64* offsets = reinterpret_cast<tensorflow::uint64*>(base);
  TF_Status status;
  for (int i = 0; i < srcarray.size(); ++i) {
    *offsets = (dst - data_start);
    offsets++;
    const tensorflow::string& s = srcarray(i);
    size_t consumed =
        TF_StringEncode(s.data(), s.size(), dst, dst_len, &status);
    CHECK(status.status.ok());
    dst += consumed;
    dst_len -= consumed;
  }
  CHECK_EQ(dst, base + size);

  auto dims = src.shape().dim_sizes();
  std::vector<tensorflow::int64> dimvec(dims.size());
  for (size_t i = 0; i < dims.size(); ++i) {
    dimvec[i] = dims[i];
  }
  static_assert(sizeof(int64_t) == sizeof(tensorflow::int64),
                "64-bit int types should match in size");
  return TF_NewTensor(TF_STRING,
                      reinterpret_cast<const int64_t*>(dimvec.data()),
                      dimvec.size(), base, size, DeleteArray, base);
}

class TensorCApi {
 public:
  static TensorBuffer* Buffer(const Tensor& tensor) { return tensor.buf_; }
  static Tensor MakeTensor(TF_DataType type, const TensorShape& shape,
                           TensorBuffer* buf) {
    return Tensor(static_cast<DataType>(type), shape, buf);
  }
};

// Create an empty tensor of type 'dtype'. 'shape' can be arbitrary, but has to
// result in a zero-sized tensor.
static TF_Tensor* EmptyTensor(TF_DataType dtype, const TensorShape& shape) {
  static char empty;
  tensorflow::int64 nelems = 1;
  std::vector<tensorflow::int64> dims;
  for (int i = 0; i < shape.dims(); ++i) {
    dims.push_back(shape.dim_size(i));
    nelems *= shape.dim_size(i);
  }
  CHECK_EQ(nelems, 0);
  static_assert(sizeof(int64_t) == sizeof(tensorflow::int64),
                "64-bit int types should match in size");
  return TF_NewTensor(dtype, reinterpret_cast<const int64_t*>(dims.data()),
                      shape.dims(), reinterpret_cast<void*>(&empty), 0,
                      [](void*, size_t, void*) {}, nullptr);
}

// Helpers for loading a TensorFlow plugin (a .so file).
Status LoadLibrary(const char* library_filename, void** result,
                   const void** buf, size_t* len);

}  // namespace tensorflow

static void TF_Run_Setup(int noutputs, TF_Tensor** c_outputs,
                         TF_Status* status) {
  status->status = Status::OK();
  for (int i = 0; i < noutputs; ++i) {
    c_outputs[i] = NULL;
  }
}

static bool TF_Run_Inputs(
    TF_Tensor* const* c_inputs,
    std::vector<std::pair<tensorflow::string, Tensor>>* input_pairs,
    TF_Status* status) {
  const int ninputs = input_pairs->size();
  for (int i = 0; i < ninputs; ++i) {
    TF_Tensor* src = c_inputs[i];
    if (c_inputs[i]->dtype != TF_STRING) {
      (*input_pairs)[i].second = tensorflow::TensorCApi::MakeTensor(
          src->dtype, src->shape, src->buffer);
    } else if (!tensorflow::TF_Tensor_DecodeStrings(
                   src, &(*input_pairs)[i].second, status)) {
      // TF_STRING tensors require copying since Tensor class expects
      // a sequence of string objects.
      return false;
    }
  }
  return true;
}

static void TF_Run_Helper(
    Session* session, const char* handle, const TF_Buffer* run_options,
    // Input tensors
    const std::vector<std::pair<tensorflow::string, Tensor>>& input_pairs,
    // Output tensors
    const std::vector<tensorflow::string>& output_tensor_names,
    TF_Tensor** c_outputs,
    // Target nodes
    const std::vector<tensorflow::string>& target_oper_names,
    TF_Buffer* run_metadata, TF_Status* status) {
  const int noutputs = output_tensor_names.size();
  std::vector<Tensor> outputs(noutputs);
  Status result;

  if (handle == nullptr) {
    RunOptions run_options_proto;
    if (run_options != nullptr &&
        !run_options_proto.ParseFromArray(run_options->data,
                                          run_options->length)) {
      status->status = InvalidArgument("Unparseable RunOptions proto");
      return;
    }
    if (run_metadata != nullptr && run_metadata->data != nullptr) {
      status->status =
          InvalidArgument("Passing non-empty run_metadata is invalid.");
      return;
    }

    RunMetadata run_metadata_proto;
    result = session->Run(run_options_proto, input_pairs, output_tensor_names,
                          target_oper_names, &outputs, &run_metadata_proto);

    // Serialize back to upstream client, who now owns the new buffer
    if (run_metadata != nullptr) {
      status->status = MessageToBuffer(run_metadata_proto, run_metadata);
      if (!status->status.ok()) return;
    }
  } else {
    // NOTE(zongheng): PRun does not support RunOptions yet.
    result = session->PRun(handle, input_pairs, output_tensor_names, &outputs);
  }
  if (!result.ok()) {
    status->status = result;
    return;
  }

  // Store results in c_outputs[]
  for (int i = 0; i < noutputs; ++i) {
    const Tensor& src = outputs[i];
    if (!src.IsInitialized() || src.NumElements() == 0) {
      c_outputs[i] = tensorflow::EmptyTensor(
          static_cast<TF_DataType>(src.dtype()), src.shape());
      continue;
    }
    if (src.dtype() != tensorflow::DT_STRING) {
      // Share the underlying buffer.
      TensorBuffer* buf = tensorflow::TensorCApi::Buffer(src);
      buf->Ref();
      c_outputs[i] = new TF_Tensor{static_cast<TF_DataType>(src.dtype()),
                                   src.shape(), buf};
    } else {
      c_outputs[i] = tensorflow::TF_Tensor_EncodeStrings(src);
    }
  }
}

extern "C" {

void TF_Run(TF_DeprecatedSession* s, const TF_Buffer* run_options,
            // Input tensors
            const char** c_input_names, TF_Tensor** c_inputs, int ninputs,
            // Output tensors
            const char** c_output_names, TF_Tensor** c_outputs, int noutputs,
            // Target nodes
            const char** c_target_oper_names, int ntargets,
            TF_Buffer* run_metadata, TF_Status* status) {
  TF_Run_Setup(noutputs, c_outputs, status);
  std::vector<std::pair<tensorflow::string, Tensor>> input_pairs(ninputs);
  if (!TF_Run_Inputs(c_inputs, &input_pairs, status)) return;
  for (int i = 0; i < ninputs; ++i) {
    input_pairs[i].first = c_input_names[i];
  }
  std::vector<tensorflow::string> output_names(noutputs);
  for (int i = 0; i < noutputs; ++i) {
    output_names[i] = c_output_names[i];
  }
  std::vector<tensorflow::string> target_oper_names(ntargets);
  for (int i = 0; i < ntargets; ++i) {
    target_oper_names[i] = c_target_oper_names[i];
  }
  TF_Run_Helper(s->session, nullptr, run_options, input_pairs, output_names,
                c_outputs, target_oper_names, run_metadata, status);
}

void TF_PRunSetup(TF_DeprecatedSession* s,
                  // Input names
                  const char** c_input_names, int ninputs,
                  // Output names
                  const char** c_output_names, int noutputs,
                  // Target nodes
                  const char** c_target_oper_names, int ntargets,
                  const char** handle, TF_Status* status) {
  status->status = Status::OK();

  std::vector<tensorflow::string> input_names(ninputs);
  std::vector<tensorflow::string> output_names(noutputs);
  std::vector<tensorflow::string> target_oper_names(ntargets);
  for (int i = 0; i < ninputs; ++i) {
    input_names[i] = c_input_names[i];
  }
  for (int i = 0; i < noutputs; ++i) {
    output_names[i] = c_output_names[i];
  }
  for (int i = 0; i < ntargets; ++i) {
    target_oper_names[i] = c_target_oper_names[i];
  }
  tensorflow::string new_handle;
  Status result;
  result = s->session->PRunSetup(input_names, output_names, target_oper_names,
                                 &new_handle);
  if (result.ok()) {
    char* buf = new char[new_handle.size() + 1];
    memcpy(buf, new_handle.c_str(), new_handle.size() + 1);
    *handle = buf;
  } else {
    status->status = result;
  }
}

void TF_PRun(TF_DeprecatedSession* s, const char* handle,
             // Input tensors
             const char** c_input_names, TF_Tensor** c_inputs, int ninputs,
             // Output tensors
             const char** c_output_names, TF_Tensor** c_outputs, int noutputs,
             // Target nodes
             const char** c_target_oper_names, int ntargets,
             TF_Status* status) {
  TF_Run_Setup(noutputs, c_outputs, status);
  std::vector<std::pair<tensorflow::string, Tensor>> input_pairs(ninputs);
  if (!TF_Run_Inputs(c_inputs, &input_pairs, status)) return;
  for (int i = 0; i < ninputs; ++i) {
    input_pairs[i].first = c_input_names[i];
  }

  std::vector<tensorflow::string> output_names(noutputs);
  for (int i = 0; i < noutputs; ++i) {
    output_names[i] = c_output_names[i];
  }
  std::vector<tensorflow::string> target_oper_names(ntargets);
  for (int i = 0; i < ntargets; ++i) {
    target_oper_names[i] = c_target_oper_names[i];
  }
  TF_Run_Helper(s->session, handle, nullptr, input_pairs, output_names,
                c_outputs, target_oper_names, nullptr, status);
}

struct TF_Library {
  void* lib_handle;
  TF_Buffer op_list;
};

TF_Library* TF_LoadLibrary(const char* library_filename, TF_Status* status) {
  TF_Library* lib_handle = new TF_Library;
  status->status = tensorflow::LoadLibrary(
      library_filename, &lib_handle->lib_handle, &lib_handle->op_list.data,
      &lib_handle->op_list.length);
  if (!status->status.ok()) {
    delete lib_handle;
    return nullptr;
  }
  return lib_handle;
}

TF_Buffer TF_GetOpList(TF_Library* lib_handle) { return lib_handle->op_list; }

void TF_DeleteLibraryHandle(TF_Library* lib_handle) {
  tensorflow::port::Free(const_cast<void*>(lib_handle->op_list.data));
  delete lib_handle;
}

TF_Buffer* TF_GetAllOpList() {
  std::vector<tensorflow::OpDef> op_defs;
  tensorflow::OpRegistry::Global()->GetRegisteredOps(&op_defs);
  tensorflow::OpList op_list;
  for (const auto& op : op_defs) {
    *(op_list.add_op()) = op;
  }
  TF_Buffer* ret = TF_NewBuffer();
  MessageToBuffer(op_list, ret);
  return ret;
}

}  // end extern "C"

// --------------------------------------------------------------------------
// New Graph and Session API

// Structures -----------------------------------------------------------------

extern "C" {

struct TF_Graph {
  TF_Graph()
      : graph(OpRegistry::Global()),
        refiner(graph.op_registry()),
        num_sessions(0),
        delete_requested(false) {}
  mutex mu;
  Graph graph GUARDED_BY(mu);

  // Runs shape inference.
  tensorflow::ShapeRefiner refiner GUARDED_BY(mu);

  // Maps from name of an operation to the Node* in 'graph'.
  std::unordered_map<tensorflow::string, Node*> name_map GUARDED_BY(mu);

  // TF_Graph may only / must be deleted when
  //   num_sessions == 0 && delete_requested == true

  // num_sessions incremented by TF_NewSession, and decremented by
  // TF_DeleteSession.
  int num_sessions GUARDED_BY(mu);
  bool delete_requested GUARDED_BY(mu);  // set true by TF_DeleteGraph
};

struct TF_OperationDescription {
  TF_OperationDescription(TF_Graph* g, const char* op_type,
                          const char* node_name)
      : node_builder(node_name, op_type, g->graph.op_registry()), graph(g) {}

  NodeBuilder node_builder;
  TF_Graph* graph;
  std::vector<tensorflow::string> colocation_constraints;
};

struct TF_Operation {
  Node node;
};

struct TF_Session {
  TF_Session(Session* s, TF_Graph* g)
      : session(s), graph(g), last_num_graph_nodes(0) {}
  Session* session;
  TF_Graph* graph;
  mutex mu;
  int last_num_graph_nodes;
};

}  // end extern "C"

// Helper functions -----------------------------------------------------------

namespace {

TF_Operation* ToOperation(Node* node) {
  return static_cast<TF_Operation*>(static_cast<void*>(node));
}

tensorflow::string OutputName(const TF_Output& output) {
  return tensorflow::strings::StrCat(output.oper->node.name(), ":",
                                     output.index);
}

const tensorflow::AttrValue* GetAttrValue(TF_Operation* oper,
                                          const char* attr_name,
                                          TF_Status* status) {
  const tensorflow::AttrValue* attr =
      tensorflow::AttrSlice(oper->node.def()).Find(attr_name);
  if (attr == nullptr) {
    status->status =
        InvalidArgument("Operation has no attr named '", attr_name, "'.");
  }
  return attr;
}

}  // namespace

// Shape functions -----------------------------------------------------------

void TF_GraphSetTensorShape(TF_Graph* graph, TF_Output output,
                            const int64_t* dims, const int num_dims,
                            TF_Status* status) {
  Node* node = &output.oper->node;

  mutex_lock l(graph->mu);
  // Set the shape.
  tensorflow::shape_inference::InferenceContext* ic =
      graph->refiner.GetContext(node);
  if (ic == nullptr) {
    status->status =
        InvalidArgument("Node ", node->name(), " was not found in the graph");
    return;
  }

  std::vector<tensorflow::shape_inference::DimensionHandle> dim_vec;
  for (int i = 0; i < num_dims; ++i) {
    dim_vec.push_back(ic->MakeDim(dims[i]));
  }

  tensorflow::shape_inference::ShapeHandle new_shape = ic->MakeShape(dim_vec);
  status->status = graph->refiner.SetShape(node, output.index, new_shape);
}

int TF_GraphGetTensorNumDims(TF_Graph* graph, TF_Output output,
                             TF_Status* status) {
  Node* node = &output.oper->node;

  mutex_lock l(graph->mu);
  tensorflow::shape_inference::InferenceContext* ic =
      graph->refiner.GetContext(node);
  if (ic == nullptr) {
    status->status =
        InvalidArgument("Node ", node->name(), " was not found in the graph");
    return -1;
  }

  tensorflow::shape_inference::ShapeHandle shape = ic->output(output.index);

  // Unknown rank means the number of dimensions is -1.
  if (!ic->RankKnown(shape)) {
    return -1;
  }

  return ic->Rank(shape);
}

void TF_GraphGetTensorShape(TF_Graph* graph, TF_Output output, int64_t* dims,
                            int num_dims, TF_Status* status) {
  Node* node = &output.oper->node;

  mutex_lock l(graph->mu);
  tensorflow::shape_inference::InferenceContext* ic =
      graph->refiner.GetContext(node);
  if (ic == nullptr) {
    status->status =
        InvalidArgument("Node ", node->name(), " was not found in the graph");
    return;
  }

  tensorflow::shape_inference::ShapeHandle shape = ic->output(output.index);

  int rank = -1;
  if (ic->RankKnown(shape)) {
    rank = ic->Rank(shape);
  }

  if (num_dims != rank) {
    status->status = InvalidArgument("Expected rank is ", num_dims,
                                     " but actual rank is ", rank);
    return;
  }

  if (num_dims == 0) {
    // Output shape is a scalar.
    return;
  }

  // Rank is greater than 0, so fill in the values, if known, and
  // -1 for unknown values.
  for (int i = 0; i < num_dims; ++i) {
    auto dim = ic->Dim(shape, i);
    tensorflow::int64 value = -1;
    if (ic->ValueKnown(dim)) {
      value = ic->Value(dim);
    }
    dims[i] = value;
  }
}

// TF_OperationDescription functions ------------------------------------------

extern "C" {

TF_OperationDescription* TF_NewOperation(TF_Graph* graph, const char* op_type,
                                         const char* oper_name) {
  mutex_lock l(graph->mu);
  return new TF_OperationDescription(graph, op_type, oper_name);
}

void TF_SetDevice(TF_OperationDescription* desc, const char* device) {
  desc->node_builder.Device(device);
}

void TF_AddInput(TF_OperationDescription* desc, TF_Output input) {
  desc->node_builder.Input(&input.oper->node, input.index);
}

void TF_AddInputList(TF_OperationDescription* desc, const TF_Output* inputs,
                     int num_inputs) {
  std::vector<NodeBuilder::NodeOut> input_list;
  input_list.reserve(num_inputs);
  for (int i = 0; i < num_inputs; ++i) {
    input_list.emplace_back(&inputs[i].oper->node, inputs[i].index);
  }
  desc->node_builder.Input(input_list);
}

void TF_AddControlInput(TF_OperationDescription* desc, TF_Operation* input) {
  desc->node_builder.ControlInput(&input->node);
}

void TF_ColocateWith(TF_OperationDescription* desc, TF_Operation* op) {
  desc->colocation_constraints.emplace_back(tensorflow::strings::StrCat(
      tensorflow::kColocationGroupPrefix, op->node.name()));
}

void TF_SetAttrString(TF_OperationDescription* desc, const char* attr_name,
                      const void* value, size_t length) {
  tensorflow::StringPiece s(static_cast<const char*>(value), length);
  desc->node_builder.Attr(attr_name, s);
}

void TF_SetAttrStringList(TF_OperationDescription* desc, const char* attr_name,
                          const void* const* values, const size_t* lengths,
                          int num_values) {
  std::vector<tensorflow::StringPiece> v;
  v.reserve(num_values);
  for (int i = 0; i < num_values; ++i) {
    v.emplace_back(static_cast<const char*>(values[i]), lengths[i]);
  }
  desc->node_builder.Attr(attr_name, v);
}

void TF_SetAttrInt(TF_OperationDescription* desc, const char* attr_name,
                   int64_t value) {
  static_assert(sizeof(int64_t) == sizeof(tensorflow::int64),
                "64-bit int types should match in size");
  desc->node_builder.Attr(attr_name, static_cast<tensorflow::int64>(value));
}

void TF_SetAttrIntList(TF_OperationDescription* desc, const char* attr_name,
                       const int64_t* values, int num_values) {
  static_assert(sizeof(int64_t) == sizeof(tensorflow::int64),
                "64-bit int types should match in size");
  desc->node_builder.Attr(
      attr_name,
      ArraySlice<const tensorflow::int64>(
          reinterpret_cast<const tensorflow::int64*>(values), num_values));
}

void TF_SetAttrFloat(TF_OperationDescription* desc, const char* attr_name,
                     float value) {
  desc->node_builder.Attr(attr_name, value);
}

void TF_SetAttrFloatList(TF_OperationDescription* desc, const char* attr_name,
                         const float* values, int num_values) {
  desc->node_builder.Attr(attr_name,
                          ArraySlice<const float>(values, num_values));
}

void TF_SetAttrBool(TF_OperationDescription* desc, const char* attr_name,
                    unsigned char value) {
  desc->node_builder.Attr(attr_name, static_cast<bool>(value));
}

void TF_SetAttrBoolList(TF_OperationDescription* desc, const char* attr_name,
                        const unsigned char* values, int num_values) {
  std::unique_ptr<bool[]> b(new bool[num_values]);
  for (int i = 0; i < num_values; ++i) {
    b[i] = values[i];
  }
  desc->node_builder.Attr(attr_name,
                          ArraySlice<const bool>(b.get(), num_values));
}

void TF_SetAttrType(TF_OperationDescription* desc, const char* attr_name,
                    TF_DataType value) {
  desc->node_builder.Attr(attr_name, static_cast<DataType>(value));
}

void TF_SetAttrTypeList(TF_OperationDescription* desc, const char* attr_name,
                        const TF_DataType* values, int num_values) {
  desc->node_builder.Attr(
      attr_name, ArraySlice<const DataType>(
                     reinterpret_cast<const DataType*>(values), num_values));
}

void TF_SetAttrShape(TF_OperationDescription* desc, const char* attr_name,
                     const int64_t* dims, int num_dims) {
  PartialTensorShape shape;
  if (num_dims >= 0) {
    static_assert(sizeof(int64_t) == sizeof(tensorflow::int64),
                  "64-bit int types should match in size");
    shape = PartialTensorShape(ArraySlice<tensorflow::int64>(
        reinterpret_cast<const tensorflow::int64*>(dims), num_dims));
  }
  desc->node_builder.Attr(attr_name, shape);
}

void TF_SetAttrShapeList(TF_OperationDescription* desc, const char* attr_name,
                         const int64_t* const* dims, const int* num_dims,
                         int num_shapes) {
  std::vector<PartialTensorShape> shapes;
  shapes.reserve(num_shapes);
  for (int i = 0; i < num_shapes; ++i) {
    if (num_dims[i] < 0) {
      shapes.emplace_back();
    } else {
      static_assert(sizeof(int64_t) == sizeof(tensorflow::int64),
                    "64-bit int types should match in size");
      shapes.emplace_back(ArraySlice<tensorflow::int64>(
          reinterpret_cast<const tensorflow::int64*>(dims[i]), num_dims[i]));
    }
  }
  desc->node_builder.Attr(attr_name, shapes);
}

void TF_SetAttrTensorShapeProto(TF_OperationDescription* desc,
                                const char* attr_name, const void* proto,
                                size_t proto_len, TF_Status* status) {
  // shape.ParseFromArray takes an int as length, this function takes size_t,
  // make sure there is no information loss.
  if (proto_len > std::numeric_limits<int>::max()) {
    status->status = InvalidArgument(
        "proto_len (", proto_len,
        " bytes) is too large to be parsed by the protocol buffer library");
    return;
  }
  TensorShapeProto shape;
  if (shape.ParseFromArray(proto, static_cast<int>(proto_len))) {
    desc->node_builder.Attr(attr_name, shape);
    status->status = Status::OK();
  } else {
    status->status = InvalidArgument("Unparseable TensorShapeProto");
  }
}

void TF_SetAttrTensorShapeProtoList(TF_OperationDescription* desc,
                                    const char* attr_name,
                                    const void* const* protos,
                                    const size_t* proto_lens, int num_shapes,
                                    TF_Status* status) {
  std::vector<TensorShapeProto> shapes;
  shapes.resize(num_shapes);
  for (int i = 0; i < num_shapes; ++i) {
    if (proto_lens[i] > std::numeric_limits<int>::max()) {
      status->status = InvalidArgument(
          "length of element ", i, " in the list (", proto_lens[i],
          " bytes) is too large to be parsed by the protocol buffer library");
      return;
    }
    if (!shapes[i].ParseFromArray(protos[i], static_cast<int>(proto_lens[i]))) {
      status->status =
          InvalidArgument("Unparseable TensorShapeProto at index ", i);
      return;
    }
  }
  desc->node_builder.Attr(attr_name, shapes);
  status->status = Status::OK();
}

void TF_SetAttrTensor(TF_OperationDescription* desc, const char* attr_name,
                      TF_Tensor* value, TF_Status* status) {
  status->status = Status::OK();
  Tensor t;
  bool ok = true;

  if (value->dtype != TF_STRING) {
    t = tensorflow::TensorCApi::MakeTensor(value->dtype, value->shape,
                                           value->buffer);
  } else {
    // TF_STRING tensors require copying since Tensor class expects
    // a sequence of string objects.
    ok = tensorflow::TF_Tensor_DecodeStrings(value, &t, status);
  }

  if (ok) desc->node_builder.Attr(attr_name, t);
}

void TF_SetAttrTensorList(TF_OperationDescription* desc, const char* attr_name,
                          TF_Tensor* const* values, int num_values,
                          TF_Status* status) {
  status->status = Status::OK();
  std::vector<Tensor> t;
  t.reserve(num_values);
  bool ok = true;

  for (int i = 0; i < num_values && ok; ++i) {
    if (values[i]->dtype != TF_STRING) {
      t.emplace_back(tensorflow::TensorCApi::MakeTensor(
          values[i]->dtype, values[i]->shape, values[i]->buffer));
    } else {
      t.emplace_back(::tensorflow::DT_STRING);
      // TF_STRING tensors require copying since Tensor class expects
      // a sequence of string objects.
      ok = tensorflow::TF_Tensor_DecodeStrings(values[i], &t.back(), status);
    }
  }

  if (ok) desc->node_builder.Attr(attr_name, t);
}

void TF_SetAttrValueProto(TF_OperationDescription* desc, const char* attr_name,
                          const void* proto, size_t proto_len,
                          TF_Status* status) {
  tensorflow::AttrValue attr_value;
  if (attr_value.ParseFromArray(proto, proto_len)) {
    desc->node_builder.Attr(attr_name, attr_value);
    status->status = Status::OK();
  } else {
    status->status = InvalidArgument("Unparseable AttrValue proto");
  }
}

TF_Operation* TF_FinishOperation(TF_OperationDescription* desc,
                                 TF_Status* status) {
  Node* ret = nullptr;
  mutex_lock l(desc->graph->mu);

  if (desc->graph->name_map.count(desc->node_builder.node_name())) {
    status->status = InvalidArgument("Duplicate node name in graph: '",
                                     desc->node_builder.node_name(), "'");
  } else {
    std::sort(desc->colocation_constraints.begin(),
              desc->colocation_constraints.end());
    desc->node_builder.Attr(tensorflow::kColocationAttrName,
                            desc->colocation_constraints);
    status->status = desc->node_builder.Finalize(&desc->graph->graph, &ret);

    if (status->status.ok()) {
      // Run shape inference function for newly added node.
      //
      // TODO(b/28152992): Enable returning the result of this
      // code-path once we have converted all python shape functions
      // to call their C++ versions.
      desc->graph->refiner.AddNode(ret);

      // Add the node to the name-to-node mapping.
      desc->graph->name_map[ret->name()] = ret;
    }
  }

  delete desc;

  return ToOperation(ret);
}

// TF_Operation functions
// ----------------------------------------------------------

const char* TF_OperationName(TF_Operation* oper) {
  return oper->node.name().c_str();
}

const char* TF_OperationOpType(TF_Operation* oper) {
  return oper->node.type_string().c_str();
}

const char* TF_OperationDevice(TF_Operation* oper) {
  return oper->node.def().device().c_str();
}

int TF_OperationNumOutputs(TF_Operation* oper) {
  return oper->node.num_outputs();
}

TF_DataType TF_OperationOutputType(TF_Output oper_out) {
  return static_cast<TF_DataType>(
      oper_out.oper->node.output_type(oper_out.index));
}

int TF_OperationOutputListLength(TF_Operation* oper, const char* arg_name,
                                 TF_Status* status) {
  NameRangeMap name_ranges;
  status->status = NameRangesForNode(oper->node.def(), oper->node.op_def(),
                                     nullptr, &name_ranges);
  if (!status->status.ok()) return -1;
  auto iter = name_ranges.find(arg_name);
  if (iter == name_ranges.end()) {
    status->status = InvalidArgument("Input arg '", arg_name, "' not found");
    return -1;
  }
  return iter->second.second - iter->second.first;
}

int TF_OperationNumInputs(TF_Operation* oper) {
  return oper->node.num_inputs();
}

TF_DataType TF_OperationInputType(TF_Input oper_in) {
  return static_cast<TF_DataType>(oper_in.oper->node.input_type(oper_in.index));
}

int TF_OperationInputListLength(TF_Operation* oper, const char* arg_name,
                                TF_Status* status) {
  NameRangeMap name_ranges;
  status->status = NameRangesForNode(oper->node.def(), oper->node.op_def(),
                                     &name_ranges, nullptr);
  if (!status->status.ok()) return -1;
  auto iter = name_ranges.find(arg_name);
  if (iter == name_ranges.end()) {
    status->status = InvalidArgument("Input arg '", arg_name, "' not found");
    return -1;
  }
  return iter->second.second - iter->second.first;
}

TF_Output TF_OperationInput(TF_Input oper_in) {
  const tensorflow::Edge* edge;
  Status s = oper_in.oper->node.input_edge(oper_in.index, &edge);
  if (!s.ok()) {
    return {nullptr, -1};
  }

  return {ToOperation(edge->src()), edge->src_output()};
}

int TF_OperationOutputNumConsumers(TF_Output oper_out) {
  int count = 0;
  for (const auto* edge : oper_out.oper->node.out_edges()) {
    if (edge->src_output() == oper_out.index) {
      ++count;
    }
  }
  return count;
}

int TF_OperationOutputConsumers(TF_Output oper_out, TF_Input* consumers,
                                int max_consumers) {
  int count = 0;
  for (const auto* edge : oper_out.oper->node.out_edges()) {
    if (edge->src_output() == oper_out.index) {
      if (count < max_consumers) {
        consumers[count] = {ToOperation(edge->dst()), edge->dst_input()};
      }
      ++count;
    }
  }
  return count;
}

int TF_OperationNumControlInputs(TF_Operation* oper) {
  return oper->node.in_edges().size() - oper->node.num_inputs();
}

int TF_OperationGetControlInputs(TF_Operation* oper,
                                 TF_Operation** control_inputs,
                                 int max_control_inputs) {
  int count = 0;
  for (const auto* edge : oper->node.in_edges()) {
    if (edge->IsControlEdge()) {
      if (count < max_control_inputs) {
        control_inputs[count] = ToOperation(edge->src());
      }
      ++count;
    }
  }
  return count;
}

int TF_OperationNumControlOutputs(TF_Operation* oper) {
  int count = 0;
  for (const auto* edge : oper->node.out_edges()) {
    if (edge->IsControlEdge()) {
      ++count;
    }
  }
  return count;
}

int TF_OperationGetControlOutputs(TF_Operation* oper,
                                  TF_Operation** control_outputs,
                                  int max_control_outputs) {
  int count = 0;
  for (const auto* edge : oper->node.out_edges()) {
    if (edge->IsControlEdge()) {
      if (count < max_control_outputs) {
        control_outputs[count] = ToOperation(edge->dst());
      }
      ++count;
    }
  }
  return count;
}

TF_AttrMetadata TF_OperationGetAttrMetadata(TF_Operation* oper,
                                            const char* attr_name,
                                            TF_Status* status) {
  TF_AttrMetadata metadata;
  const auto* attr = GetAttrValue(oper, attr_name, status);
  if (!status->status.ok()) return metadata;
  switch (attr->value_case()) {
#define SINGLE_CASE(kK, attr_type, size_expr) \
  case tensorflow::AttrValue::kK:             \
    metadata.is_list = 0;                     \
    metadata.list_size = -1;                  \
    metadata.type = attr_type;                \
    metadata.total_size = size_expr;          \
    break;

    SINGLE_CASE(kS, TF_ATTR_STRING, attr->s().length());
    SINGLE_CASE(kI, TF_ATTR_INT, -1);
    SINGLE_CASE(kF, TF_ATTR_FLOAT, -1);
    SINGLE_CASE(kB, TF_ATTR_BOOL, -1);
    SINGLE_CASE(kType, TF_ATTR_TYPE, -1);
    SINGLE_CASE(kShape, TF_ATTR_SHAPE,
                attr->shape().unknown_rank() ? -1 : attr->shape().dim_size());
    SINGLE_CASE(kTensor, TF_ATTR_TENSOR, -1);
#undef SINGLE_CASE

    case tensorflow::AttrValue::kList:
      metadata.is_list = 1;
      metadata.list_size = 0;
      metadata.total_size = -1;
#define LIST_CASE(field, attr_type, ...)              \
  if (attr->list().field##_size() > 0) {              \
    metadata.type = attr_type;                        \
    metadata.list_size = attr->list().field##_size(); \
    __VA_ARGS__;                                      \
    break;                                            \
  }

      LIST_CASE(s, TF_ATTR_STRING, metadata.total_size = 0;
                for (int i = 0; i < attr->list().s_size();
                     ++i) { metadata.total_size += attr->list().s(i).size(); });
      LIST_CASE(i, TF_ATTR_INT);
      LIST_CASE(f, TF_ATTR_FLOAT);
      LIST_CASE(b, TF_ATTR_BOOL);
      LIST_CASE(type, TF_ATTR_TYPE);
      LIST_CASE(shape, TF_ATTR_SHAPE, metadata.total_size = 0;
                for (int i = 0; i < attr->list().shape_size(); ++i) {
                  const auto& s = attr->list().shape(i);
                  metadata.total_size += s.unknown_rank() ? 0 : s.dim_size();
                });
      LIST_CASE(tensor, TF_ATTR_TENSOR);
      LIST_CASE(tensor, TF_ATTR_FUNC);
#undef LIST_CASE
      // All lists empty, determine the type from the OpDef.
      if (metadata.list_size == 0) {
        for (int i = 0; i < oper->node.op_def().attr_size(); ++i) {
          const auto& a = oper->node.op_def().attr(i);
          if (a.name().compare(attr_name) != 0) continue;
          const tensorflow::string& typestr = a.type();
          if (typestr == "list(string)") {
            metadata.type = TF_ATTR_STRING;
          } else if (typestr == "list(int)") {
            metadata.type = TF_ATTR_INT;
          } else if (typestr == "list(float)") {
            metadata.type = TF_ATTR_FLOAT;
          } else if (typestr == "list(bool)") {
            metadata.type = TF_ATTR_BOOL;
          } else if (typestr == "list(type)") {
            metadata.type = TF_ATTR_TYPE;
          } else if (typestr == "list(shape)") {
            metadata.type = TF_ATTR_SHAPE;
          } else if (typestr == "list(tensor)") {
            metadata.type = TF_ATTR_TENSOR;
          } else if (typestr == "list(func)") {
            metadata.type = TF_ATTR_FUNC;
          } else {
            status->status = InvalidArgument(
                "Attribute '", attr_name,
                "' has an empty value of an unrecognized type '", typestr, "'");
            return metadata;
          }
        }
      }
      break;

    case tensorflow::AttrValue::kPlaceholder:
      metadata.is_list = 0;
      metadata.list_size = -1;
      metadata.type = TF_ATTR_PLACEHOLDER;
      metadata.total_size = -1;
      break;

    case tensorflow::AttrValue::kFunc:
      metadata.is_list = 0;
      metadata.list_size = -1;
      metadata.type = TF_ATTR_FUNC;
      metadata.total_size = -1;
      break;

    case tensorflow::AttrValue::VALUE_NOT_SET:
      status->status =
          InvalidArgument("Attribute '", attr_name, "' has no value set");
      break;
  }
  return metadata;
}

void TF_OperationGetAttrString(TF_Operation* oper, const char* attr_name,
                               void* value, size_t max_length,
                               TF_Status* status) {
  const auto* attr = GetAttrValue(oper, attr_name, status);
  if (!status->status.ok()) return;
  if (attr->value_case() != tensorflow::AttrValue::kS) {
    status->status =
        InvalidArgument("Attribute '", attr_name, "' is not a string");
    return;
  }
  if (max_length <= 0) {
    return;
  }
  const auto& s = attr->s();
  std::memcpy(value, s.data(), std::min<size_t>(s.length(), max_length));
}

void TF_OperationGetAttrStringList(TF_Operation* oper, const char* attr_name,
                                   void** values, size_t* lengths,
                                   int max_values, void* storage,
                                   size_t storage_size, TF_Status* status) {
  const auto* attr = GetAttrValue(oper, attr_name, status);
  if (!status->status.ok()) return;
  if (attr->value_case() != tensorflow::AttrValue::kList) {
    status->status =
        InvalidArgument("Value for '", attr_name, "' is not a list");
    return;
  }
  const auto len = std::min(max_values, attr->list().s_size());
  char* p = static_cast<char*>(storage);
  for (int i = 0; i < len; ++i) {
    const tensorflow::string& s = attr->list().s(i);
    values[i] = p;
    lengths[i] = s.size();
    if ((p + s.size()) > (static_cast<char*>(storage) + storage_size)) {
      status->status = InvalidArgument(
          "Not enough storage to hold the requested list of strings");
      return;
    }
    memcpy(values[i], s.data(), s.size());
    p += s.size();
  }
}

#define DEFINE_GETATTR(func, c_type, cpp_type, list_field)                     \
  void func(TF_Operation* oper, const char* attr_name, c_type* value,          \
            TF_Status* status) {                                               \
    cpp_type v;                                                                \
    status->status = tensorflow::GetNodeAttr(oper->node.def(), attr_name, &v); \
    *value = static_cast<c_type>(v);                                           \
  }                                                                            \
  void func##List(TF_Operation* oper, const char* attr_name, c_type* values,   \
                  int max_values, TF_Status* status) {                         \
    const auto* attr = GetAttrValue(oper, attr_name, status);                  \
    if (!status->status.ok()) return;                                          \
    if (attr->value_case() != tensorflow::AttrValue::kList) {                  \
      status->status =                                                         \
          InvalidArgument("Value for '", attr_name, "' is not a list.");       \
      return;                                                                  \
    }                                                                          \
    const auto len = std::min(max_values, attr->list().list_field##_size());   \
    for (int i = 0; i < len; ++i) {                                            \
      values[i] = static_cast<c_type>(attr->list().list_field(i));             \
    }                                                                          \
  }
DEFINE_GETATTR(TF_OperationGetAttrInt, int64_t, tensorflow::int64, i);
DEFINE_GETATTR(TF_OperationGetAttrFloat, float, float, f);
DEFINE_GETATTR(TF_OperationGetAttrBool, unsigned char, bool, b);
DEFINE_GETATTR(TF_OperationGetAttrType, TF_DataType, DataType, type);
#undef DEFINE_GETATTR

void TF_OperationGetAttrShape(TF_Operation* oper, const char* attr_name,
                              int64_t* value, int num_dims, TF_Status* status) {
  PartialTensorShape shape;
  status->status = tensorflow::GetNodeAttr(oper->node.def(), attr_name, &shape);
  if (!status->status.ok()) return;
  auto len = std::min(shape.dims(), num_dims);
  for (int i = 0; i < len; ++i) {
    value[i] = shape.dim_size(i);
  }
}

void TF_OperationGetAttrShapeList(TF_Operation* oper, const char* attr_name,
                                  int64_t** values, int* num_dims,
                                  int max_values, int64_t* storage,
                                  int storage_size, TF_Status* status) {
  std::vector<PartialTensorShape> shapes;
  status->status =
      tensorflow::GetNodeAttr(oper->node.def(), attr_name, &shapes);
  if (!status->status.ok()) return;
  auto len = std::min(static_cast<int>(shapes.size()), max_values);
  int64_t* p = storage;
  int storage_left = storage_size;
  for (int i = 0; i < len; ++i) {
    // shapes[i].dims() == -1 for shapes with an unknown rank.
    int64_t n = shapes[i].dims();
    num_dims[i] = n;
    values[i] = p;
    if (n < 0) {
      continue;
    }
    if (storage_left < n) {
      status->status = InvalidArgument(
          "Not enough storage to hold the requested list of shapes");
      return;
    }
    storage_left -= n;
    for (int j = 0; j < n; ++j, ++p) {
      *p = shapes[i].dim_size(j);
    }
  }
}

void TF_OperationGetAttrTensorShapeProto(TF_Operation* oper,
                                         const char* attr_name,
                                         TF_Buffer* value, TF_Status* status) {
  const auto* attr = GetAttrValue(oper, attr_name, status);
  if (!status->status.ok()) return;
  if (attr->value_case() != tensorflow::AttrValue::kShape) {
    status->status =
        InvalidArgument("Value for '", attr_name, "' is not a shape.");
    return;
  }
  status->status = MessageToBuffer(attr->shape(), value);
}

void TF_OperationGetAttrTensorShapeProtoList(TF_Operation* oper,
                                             const char* attr_name,
                                             TF_Buffer** values, int max_values,
                                             TF_Status* status) {
  const auto* attr = GetAttrValue(oper, attr_name, status);
  if (!status->status.ok()) return;
  if (attr->value_case() != tensorflow::AttrValue::kList) {
    status->status =
        InvalidArgument("Value for '", attr_name, "' is not a list");
    return;
  }
  const auto len = std::min(max_values, attr->list().shape_size());
  for (int i = 0; i < len; ++i) {
    values[i] = TF_NewBuffer();
    status->status = MessageToBuffer(attr->list().shape(i), values[i]);
    if (!status->status.ok()) {
      // Delete everything allocated to far, the operation has failed.
      for (int j = 0; j <= i; ++j) {
        TF_DeleteBuffer(values[j]);
      }
      return;
    }
  }
}

void TF_OperationGetAttrTensor(TF_Operation* oper, const char* attr_name,
                               TF_Tensor** value, TF_Status* status) {
  *value = nullptr;
  Tensor t;
  status->status = tensorflow::GetNodeAttr(oper->node.def(), attr_name, &t);
  if (!status->status.ok()) return;
  *value = new TF_Tensor{static_cast<TF_DataType>(t.dtype()), t.shape(),
                         tensorflow::TensorCApi::Buffer(t)};
  (*value)->buffer->Ref();
}

void TF_OperationGetAttrTensorList(TF_Operation* oper, const char* attr_name,
                                   TF_Tensor** values, int max_values,
                                   TF_Status* status) {
  std::vector<Tensor> ts;
  status->status = tensorflow::GetNodeAttr(oper->node.def(), attr_name, &ts);
  if (!status->status.ok()) return;
  const auto len = std::min(max_values, static_cast<int>(ts.size()));
  for (int i = 0; i < len; ++i) {
    const Tensor& t = ts[i];
    values[i] = new TF_Tensor{static_cast<TF_DataType>(t.dtype()), t.shape(),
                              tensorflow::TensorCApi::Buffer(t)};
    values[i]->buffer->Ref();
  }
}

void TF_OperationGetAttrValueProto(TF_Operation* oper, const char* attr_name,
                                   TF_Buffer* output_attr_value,
                                   TF_Status* status) {
  const auto* attr = GetAttrValue(oper, attr_name, status);
  if (!status->status.ok()) return;
  status->status = MessageToBuffer(*attr, output_attr_value);
}

void TF_OperationToNodeDef(TF_Operation* oper, TF_Buffer* output_node_def,
                           TF_Status* status) {
  status->status = MessageToBuffer(oper->node.def(), output_node_def);
}

// TF_Graph functions ---------------------------------------------------------

TF_Graph* TF_NewGraph() { return new TF_Graph; }

void TF_DeleteGraph(TF_Graph* g) {
  g->mu.lock();
  g->delete_requested = true;
  const bool del = g->num_sessions == 0;
  g->mu.unlock();
  if (del) delete g;
}

TF_Operation* TF_GraphOperationByName(TF_Graph* graph, const char* oper_name) {
  mutex_lock l(graph->mu);
  auto iter = graph->name_map.find(oper_name);
  if (iter == graph->name_map.end()) {
    return nullptr;
  } else {
    return ToOperation(iter->second);
  }
}

TF_Operation* TF_GraphNextOperation(TF_Graph* graph, size_t* pos) {
  if (*pos == 0) {
    // Advance past the first sentinel nodes in every graph (the source & sink).
    *pos += 2;
  } else {
    // Advance to the next node.
    *pos += 1;
  }

  mutex_lock l(graph->mu);
  while (*pos < static_cast<size_t>(graph->graph.num_node_ids())) {
    Node* node = graph->graph.FindNodeId(*pos);
    // FindNodeId() returns nullptr for nodes that have been deleted.
    // We aren't currently allowing nodes to be deleted, but it is safer
    // to still check.
    if (node != nullptr) return ToOperation(node);
    *pos += 1;
  }

  // No more nodes.
  return nullptr;
}

void TF_GraphToGraphDef(TF_Graph* graph, TF_Buffer* output_graph_def,
                        TF_Status* status) {
  GraphDef def;
  {
    mutex_lock l(graph->mu);
    graph->graph.ToGraphDef(&def);
  }
  status->status = MessageToBuffer(def, output_graph_def);
}

struct TF_ImportGraphDefOptions {
  tensorflow::ImportGraphDefOptions opts;
};

TF_ImportGraphDefOptions* TF_NewImportGraphDefOptions() {
  return new TF_ImportGraphDefOptions;
}
void TF_DeleteImportGraphDefOptions(TF_ImportGraphDefOptions* opts) {
  delete opts;
}
void TF_ImportGraphDefOptionsSetPrefix(TF_ImportGraphDefOptions* opts,
                                       const char* prefix) {
  opts->opts.prefix = prefix;
}

namespace {

TensorId ToTensorId(const TF_Output& output) {
  return TensorId(output.oper->node.name(), output.index);
}

}  // namespace

void TF_ImportGraphDefOptionsAddInputMapping(TF_ImportGraphDefOptions* opts,
                                             const char* src_name,
                                             int src_index, TF_Output dst) {
  opts->opts.input_map[TensorId(src_name, src_index)] = ToTensorId(dst);
}

extern void TF_ImportGraphDefOptionsAddControlDependency(
    TF_ImportGraphDefOptions* opts, TF_Operation* oper) {
  opts->opts.control_dependencies.push_back(oper->node.name());
}

void TF_ImportGraphDefOptionsAddReturnOutput(TF_ImportGraphDefOptions* opts,
                                             const char* oper_name, int index) {
  opts->opts.return_tensors.push_back({oper_name, index});
}

int TF_ImportGraphDefOptionsNumReturnOutputs(
    const TF_ImportGraphDefOptions* opts) {
  return opts->opts.return_tensors.size();
}

static void GraphImportGraphDefLocked(TF_Graph* graph, const GraphDef& def,
                                      const TF_ImportGraphDefOptions* opts,
                                      TF_Output* return_outputs,
                                      int num_return_outputs, TF_Status* status)
    EXCLUSIVE_LOCKS_REQUIRED(graph->mu) {
  if (num_return_outputs != opts->opts.return_tensors.size()) {
    status->status = InvalidArgument("Expected 'num_return_outputs' to be ",
                                     opts->opts.return_tensors.size(), ", got ",
                                     num_return_outputs);
    return;
  }
  if (num_return_outputs > 0 && return_outputs == nullptr) {
    status->status = InvalidArgument(
        "'return_outputs' must be preallocated to length ", num_return_outputs);
    return;
  }
  const int last_node_id = graph->graph.num_node_ids();
  std::vector<std::pair<Node*, int>> return_outputs_vec;
  status->status = tensorflow::ImportGraphDef(
      opts->opts, def, &graph->graph, &graph->refiner, &return_outputs_vec);
  if (!status->status.ok()) return;
  for (int i = last_node_id; i < graph->graph.num_node_ids(); ++i) {
    auto* node = graph->graph.FindNodeId(i);
    if (node != nullptr) graph->name_map[node->name()] = node;
  }
  DCHECK_EQ(return_outputs_vec.size(), num_return_outputs);
  for (int i = 0; i < num_return_outputs; ++i) {
    return_outputs[i].oper = ToOperation(return_outputs_vec[i].first);
    return_outputs[i].index = return_outputs_vec[i].second;
  }
}

void TF_GraphImportGraphDefWithReturnOutputs(
    TF_Graph* graph, const TF_Buffer* graph_def,
    const TF_ImportGraphDefOptions* opts, TF_Output* return_outputs,
    int num_return_outputs, TF_Status* status) {
  GraphDef def;
  if (!def.ParseFromArray(graph_def->data, graph_def->length)) {
    status->status = InvalidArgument("Invalid GraphDef");
    return;
  }
  mutex_lock l(graph->mu);
  GraphImportGraphDefLocked(graph, def, opts, return_outputs,
                            num_return_outputs, status);
}

void TF_GraphImportGraphDef(TF_Graph* graph, const TF_Buffer* graph_def,
                            const TF_ImportGraphDefOptions* options,
                            TF_Status* status) {
  TF_GraphImportGraphDefWithReturnOutputs(graph, graph_def, options, nullptr, 0,
                                          status);
}

// TF_Session functions ----------------------------------------------

TF_Session* TF_NewSession(TF_Graph* graph, const TF_SessionOptions* opt,
                          TF_Status* status) {
  Session* session;
  status->status = NewSession(opt->options, &session);
  if (status->status.ok()) {
    if (graph != nullptr) {
      mutex_lock l(graph->mu);
      graph->num_sessions += 1;
    }
    return new TF_Session(session, graph);
  } else {
    DCHECK_EQ(nullptr, session);
    return NULL;
  }
}

#ifndef __ANDROID__
TF_Session* TF_LoadSessionFromSavedModel(
    const TF_SessionOptions* session_options, const TF_Buffer* run_options,
    const char* export_dir, const char* const* tags, int tags_len,
    TF_Graph* graph, TF_Buffer* meta_graph_def, TF_Status* status) {
  mutex_lock l(graph->mu);

  if (!graph->name_map.empty()) {
    status->status = InvalidArgument("Graph is non-empty.");
    return nullptr;
  }

  RunOptions run_options_proto;
  if (run_options != nullptr &&
      !run_options_proto.ParseFromArray(run_options->data,
                                        run_options->length)) {
    status->status = InvalidArgument("Unparseable RunOptions proto");
    return nullptr;
  }

  std::unordered_set<tensorflow::string> tag_set;
  for (int i = 0; i < tags_len; i++) {
    tag_set.insert(tensorflow::string(tags[i]));
  }

  tensorflow::SavedModelBundle bundle;
  status->status =
      tensorflow::LoadSavedModel(session_options->options, run_options_proto,
                                 export_dir, tag_set, &bundle);
  if (!status->status.ok()) return nullptr;

  // Create a TF_Graph from the MetaGraphDef. This is safe as long as Session
  // extends using GraphDefs. The Graph instance is different, but equivalent
  // to the one used to create the session.
  //
  // TODO(jhseu): When Session is modified to take Graphs instead of
  // GraphDefs, return the Graph generated in LoadSavedModel().
  TF_ImportGraphDefOptions* import_opts = TF_NewImportGraphDefOptions();
  GraphImportGraphDefLocked(graph, bundle.meta_graph_def.graph_def(),
                            import_opts, nullptr, 0, status);
  TF_DeleteImportGraphDefOptions(import_opts);
  if (TF_GetCode(status) != TF_OK) return nullptr;

  if (meta_graph_def != nullptr) {
    status->status = MessageToBuffer(bundle.meta_graph_def, meta_graph_def);
    if (!status->status.ok()) return nullptr;
  }

  TF_Session* session = new TF_Session(bundle.session.release(), graph);

  graph->num_sessions += 1;
  session->last_num_graph_nodes = graph->graph.num_node_ids();
  return session;
}
#endif  // __ANDROID__

void TF_CloseSession(TF_Session* s, TF_Status* status) {
  status->status = s->session->Close();
}

void TF_DeleteSession(TF_Session* s, TF_Status* status) {
  status->status = Status::OK();
  TF_Graph* const graph = s->graph;
  if (graph != nullptr) {
    graph->mu.lock();
    graph->num_sessions -= 1;
    const bool del = graph->delete_requested && graph->num_sessions == 0;
    graph->mu.unlock();
    if (del) delete graph;
  }
  delete s->session;
  delete s;
}

// TODO(josh11b,mrry): Change Session to be able to use a Graph*
// directly, instead of requiring us to serialize to a GraphDef and
// call Session::Extend().
static bool ExtendSessionGraphHelper(TF_Session* session, TF_Status* status) {
  if (session->graph != nullptr) {
    mutex_lock session_lock(session->mu);
    session->graph->mu.lock();
    const Graph& graph = session->graph->graph;
    const auto num_nodes = graph.num_node_ids();
    if (session->last_num_graph_nodes < num_nodes) {
      GraphDef graph_def;
      graph_def.mutable_versions()->CopyFrom(graph.versions());
      // Fill graph_def with nodes with ids in the range
      // [session->last_num_graph_nodes, num_nodes), that is the nodes
      // added since the last TF_SessionRun() call.
      for (auto id = session->last_num_graph_nodes; id < num_nodes; ++id) {
        Node* const node = graph.FindNodeId(id);
        if (node != nullptr && node->IsOp()) {
          NodeDef* const node_def = graph_def.add_node();
          *node_def = node->def();
        }
      }
      session->graph->mu.unlock();
      // TODO(josh11b): Also send the function library if needed.
      status->status = session->session->Extend(graph_def);
      if (!status->status.ok()) {
        // Contract is we always delete input_values[i].
        return false;
      }
      // Note: session->session is not modified if Extend() fails, so
      // we only set last_num_graph_nodes if it succeeds.
      session->last_num_graph_nodes = num_nodes;
    } else {
      session->graph->mu.unlock();
    }
  }
  return true;
}

void TF_SessionRun(TF_Session* session, const TF_Buffer* run_options,
                   const TF_Output* inputs, TF_Tensor* const* input_values,
                   int ninputs, const TF_Output* outputs,
                   TF_Tensor** output_values, int noutputs,
                   const TF_Operation* const* target_opers, int ntargets,
                   TF_Buffer* run_metadata, TF_Status* status) {
  // TODO(josh11b,mrry): Change Session to be able to use a Graph*
  // directly, instead of requiring us to serialize to a GraphDef and
  // call Session::Extend().
  if (!ExtendSessionGraphHelper(session, status)) {
    return;
  }

  TF_Run_Setup(noutputs, output_values, status);

  // Convert from TF_Output and TF_Tensor to a string and Tensor.
  std::vector<std::pair<tensorflow::string, Tensor>> input_pairs(ninputs);
  if (!TF_Run_Inputs(input_values, &input_pairs, status)) return;
  for (int i = 0; i < ninputs; ++i) {
    input_pairs[i].first = OutputName(inputs[i]);
  }

  // Convert from TF_Output to string names.
  std::vector<tensorflow::string> output_names(noutputs);
  for (int i = 0; i < noutputs; ++i) {
    output_names[i] = OutputName(outputs[i]);
  }

  // Convert from TF_Operation* to string names.
  std::vector<tensorflow::string> target_names(ntargets);
  for (int i = 0; i < ntargets; ++i) {
    target_names[i] = target_opers[i]->node.name();
  }

  // Actually run.
  TF_Run_Helper(session->session, nullptr, run_options, input_pairs,
                output_names, output_values, target_names, run_metadata,
                status);
}

void TF_SessionPRunSetup(TF_Session* session, const TF_Output* inputs,
                         int ninputs, const TF_Output* outputs, int noutputs,
                         const TF_Operation* const* target_opers, int ntargets,
                         const char** handle, TF_Status* status) {
  if (!ExtendSessionGraphHelper(session, status)) {
    return;
  }

  std::vector<tensorflow::string> input_names(ninputs);
  for (int i = 0; i < ninputs; ++i) {
    input_names[i] = OutputName(inputs[i]);
  }

  std::vector<tensorflow::string> output_names(noutputs);
  for (int i = 0; i < noutputs; ++i) {
    output_names[i] = OutputName(outputs[i]);
  }

  std::vector<tensorflow::string> target_names(ntargets);
  for (int i = 0; i < ntargets; ++i) {
    target_names[i] = target_opers[i]->node.name();
  }

  tensorflow::string new_handle;
  status->status = session->session->PRunSetup(input_names, output_names,
                                               target_names, &new_handle);
  if (status->status.ok()) {
    char* buf = new char[new_handle.size() + 1];
    memcpy(buf, new_handle.c_str(), new_handle.size() + 1);
    *handle = buf;
  }
}

void TF_SessionPRun(TF_Session* session, const char* handle,
                    const TF_Output* inputs, TF_Tensor* const* input_values,
                    int ninputs, const TF_Output* outputs,
                    TF_Tensor** output_values, int noutputs,
                    const TF_Operation* const* target_opers, int ntargets,
                    TF_Status* status) {
  // TODO(josh11b,mrry): Change Session to be able to use a Graph*
  // directly, instead of requiring us to serialize to a GraphDef and
  // call Session::Extend().
  if (!ExtendSessionGraphHelper(session, status)) {
    return;
  }

  TF_Run_Setup(noutputs, output_values, status);

  // Convert from TF_Output and TF_Tensor to a string and Tensor.
  std::vector<std::pair<tensorflow::string, Tensor>> input_pairs(ninputs);
  if (!TF_Run_Inputs(input_values, &input_pairs, status)) return;
  for (int i = 0; i < ninputs; ++i) {
    input_pairs[i].first = OutputName(inputs[i]);
  }

  // Convert from TF_Output to string names.
  std::vector<tensorflow::string> output_names(noutputs);
  for (int i = 0; i < noutputs; ++i) {
    output_names[i] = OutputName(outputs[i]);
  }

  // Convert from TF_Operation* to string names.
  std::vector<tensorflow::string> target_names(ntargets);
  for (int i = 0; i < ntargets; ++i) {
    target_names[i] = target_opers[i]->node.name();
  }

  TF_Run_Helper(session->session, handle, nullptr, input_pairs, output_names,
                output_values, target_names, nullptr, status);
}

}  // end extern "C"
