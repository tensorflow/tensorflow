#include "tensorflow/core/public/tensor_c_api.h"

#include <memory>

#include "tensorflow/core/lib/core/coding.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/platform/port.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/public/status.h"
#include "tensorflow/core/public/tensor.h"
#include "tensorflow/core/public/tensor_shape.h"

// The implementation below is at the top level instead of the
// brain namespace because we are defining 'extern "C"' functions.
using tensorflow::error::Code;
using tensorflow::errors::InvalidArgument;
using tensorflow::gtl::ArraySlice;
using tensorflow::AllocationDescription;
using tensorflow::Status;
using tensorflow::DataType;
using tensorflow::Env;
using tensorflow::GraphDef;
using tensorflow::NewSession;
using tensorflow::Session;
using tensorflow::Tensor;
using tensorflow::TensorBuffer;
using tensorflow::SessionOptions;
using tensorflow::TensorShape;

extern "C" {

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

void deallocate_realigned_buffer(void* data, size_t len, void* arg) {
  tensorflow::cpu_allocator()->DeallocateRaw(data);
}
}  // namespace

struct TF_Tensor {
  TF_DataType dtype;
  TensorShape shape;
  TensorBuffer* buffer;
};

TF_Tensor* TF_NewTensor(TF_DataType dtype, tensorflow::int64* dims,
                        int num_dims, void* data, size_t len,
                        void (*deallocator)(void* data, size_t len, void* arg),
                        void* deallocator_arg) {
  std::vector<tensorflow::int64> dimvec(num_dims);
  for (int i = 0; i < num_dims; i++) {
    dimvec[i] = dims[i];
  }

  TF_ManagedBuffer* buf = new TF_ManagedBuffer;
  buf->len_ = len;
  if (reinterpret_cast<intptr_t>(data) % EIGEN_MAX_ALIGN_BYTES != 0) {
    // Copy the data into a buffer that satisfies Eigen's alignment
    // requirements.
    buf->data_ =
        tensorflow::cpu_allocator()->AllocateRaw(EIGEN_MAX_ALIGN_BYTES, len);
    std::memcpy(buf->data_, data, len);
    buf->deallocator_ = deallocate_realigned_buffer;
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
tensorflow::int64 TF_Dim(const TF_Tensor* t, int dim_index) {
  return t->shape.dim_size(dim_index);
}
size_t TF_TensorByteSize(const TF_Tensor* t) { return t->buffer->size(); }
void* TF_TensorData(const TF_Tensor* t) { return t->buffer->data(); }

// --------------------------------------------------------------------------
struct TF_SessionOptions {
  SessionOptions options;
};
TF_SessionOptions* TF_NewSessionOptions() { return new TF_SessionOptions; }
void TF_DeleteSessionOptions(TF_SessionOptions* opt) { delete opt; }

void TF_SetTarget(TF_SessionOptions* options, const char* target) {
  options->options.target = target;
}

void TF_SetConfig(TF_SessionOptions* options, const char* config,
                  size_t config_len, TF_Status* status) {
  if (!options->options.config.ParseFromArray(config, config_len)) {
    status->status =
        tensorflow::errors::InvalidArgument("Unparseable ConfigProto");
  }
}

// --------------------------------------------------------------------------
struct TF_Session {
  Session* session;
};

TF_Session* TF_NewSession(const TF_SessionOptions* opt, TF_Status* status) {
  Session* session;
  status->status = NewSession(opt->options, &session);
  if (status->status.ok()) {
    return new TF_Session({session});
  } else {
    DCHECK_EQ(nullptr, session);
    return NULL;
  }
}

void TF_CloseSession(TF_Session* s, TF_Status* status) {
  status->status = s->session->Close();
}

void TF_DeleteSession(TF_Session* s, TF_Status* status) {
  status->status = Status::OK();
  delete s->session;
  delete s;
}

void TF_ExtendGraph(TF_Session* s, const void* proto, size_t proto_len,
                    TF_Status* status) {
  GraphDef g;
  if (!tensorflow::ParseProtoUnlimited(&g, proto, proto_len)) {
    status->status = tensorflow::errors::InvalidArgument("Invalid GraphDef");
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
  for (tensorflow::int64 i = 0; i < num_elements; i++) {
    tensorflow::uint64 offset =
        reinterpret_cast<const tensorflow::uint64*>(input)[i];
    tensorflow::uint64 len;
    const char* p;
    if (static_cast<ptrdiff_t>(offset) >= (limit - data_start) ||
        !(p = tensorflow::core::GetVarint64Ptr(data_start + offset, limit,
                                               &len)) ||
        (static_cast<ptrdiff_t>(len) > (limit - p))) {
      status->status = InvalidArgument("Malformed TF_STRING tensor; element ",
                                       i, " out of range");
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
  for (int i = 0; i < srcarray.size(); i++) {
    const tensorflow::string& s = srcarray(i);
    // uint64 starting_offset, varint64 length, string contents
    size += sizeof(tensorflow::uint64) +
            tensorflow::core::VarintLength(s.size()) + s.size();
  }

  // Encode all strings.
  char* base = new char[size];
  char* data_start = base + sizeof(tensorflow::uint64) * srcarray.size();
  char* dst = data_start;  // Where next string is encoded.
  tensorflow::uint64* offsets = reinterpret_cast<tensorflow::uint64*>(base);
  for (int i = 0; i < srcarray.size(); i++) {
    const tensorflow::string& s = srcarray(i);
    *offsets = (dst - data_start);
    offsets++;
    dst = tensorflow::core::EncodeVarint64(dst, s.size());
    memcpy(dst, s.data(), s.size());
    dst += s.size();
  }
  CHECK_EQ(dst, base + size);

  auto dims = src.shape().dim_sizes();
  std::vector<tensorflow::int64> dimvec(dims.size());
  for (size_t i = 0; i < dims.size(); i++) {
    dimvec[i] = dims[i];
  }
  return TF_NewTensor(TF_STRING, dimvec.data(), dimvec.size(), base, size,
                      DeleteArray, base);
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
  return TF_NewTensor(dtype, dims.data(), shape.dims(),
                      reinterpret_cast<void*>(&empty), 0,
                      [](void*, size_t, void*) {}, nullptr);
}

}  // namespace tensorflow

extern "C" {

void TF_Run(TF_Session* s,
            // Input tensors
            const char** c_input_names, TF_Tensor** c_inputs, int ninputs,
            // Output tensors
            const char** c_output_tensor_names, TF_Tensor** c_outputs,
            int noutputs,
            // Target nodes
            const char** c_target_node_names, int ntargets, TF_Status* status) {
  status->status = Status::OK();
  for (int i = 0; i < noutputs; i++) {
    c_outputs[i] = NULL;
  }

  // Initialize inputs.
  std::vector<std::pair<tensorflow::string, Tensor>> inputs(ninputs);
  bool ok = true;
  for (int i = 0; i < ninputs; i++) {
    TF_Tensor* src = c_inputs[i];
    if (ok) {
      inputs[i].first = c_input_names[i];
      if (c_inputs[i]->dtype != TF_STRING) {
        inputs[i].second = tensorflow::TensorCApi::MakeTensor(
            src->dtype, src->shape, src->buffer);
      } else {
        // TF_STRING tensors require copying since Tensor class expects
        // a sequence of string objects.
        ok =
            tensorflow::TF_Tensor_DecodeStrings(src, &inputs[i].second, status);
        // Must keep looping through all inputs even if there is an error
        // so that TF_DeleteTensor() is called unconditionally on all inputs.
      }
    }
    TF_DeleteTensor(src);
  }
  if (!ok) {
    return;
  }

  std::vector<tensorflow::string> output_tensor_names(noutputs);
  std::vector<Tensor> outputs(noutputs);
  std::vector<tensorflow::string> target_node_names(ntargets);
  for (int i = 0; i < noutputs; i++) {
    output_tensor_names[i] = c_output_tensor_names[i];
  }
  for (int i = 0; i < ntargets; i++) {
    target_node_names[i] = c_target_node_names[i];
  }
  Status result =
      s->session->Run(inputs, output_tensor_names, target_node_names, &outputs);
  if (!result.ok()) {
    status->status = result;
    return;
  }

  // Store results in c_outputs[]
  for (int i = 0; i < noutputs; i++) {
    const Tensor& src = outputs[i];
    if (!src.IsInitialized()) {
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

}  // end extern "C"
