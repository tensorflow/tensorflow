/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/distributed_runtime/rpc/grpc_tensor_coding.h"
#include "grpc++/support/byte_buffer.h"
#include "grpc++/support/slice.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_reference.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/lib/io/proto_encode_helper.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/protobuf/worker.pb.h"

namespace tensorflow {
namespace grpc {

static void do_nothing(void* raw) {}
static void unref_tensorreference(void* raw) {
  TensorReference* ref = static_cast<TensorReference*>(raw);
  ref->Unref();
  delete ref;
}

void EncodeRecvTensorResponseToByteBuffer(const RecvTensorResponse& proto,
                                          ::grpc::ByteBuffer* result) {
  size_t len = proto.ByteSize();
  gpr_slice s = gpr_slice_malloc(len);
  proto.SerializeWithCachedSizesToArray(
      reinterpret_cast<uint8*>(GPR_SLICE_START_PTR(s)));
  ::grpc::Slice slice(s, ::grpc::Slice::STEAL_REF);
  *result = ::grpc::ByteBuffer(&slice, 1);
}

// We generate a RecvTensorResponse protocol buffer encoding into "*result",
// but where possible, we share the underlying Tensor buffer for "val", to
// avoid an extra copy.
//
// We hand-encode the protocol buffer data in the following order, as follows:
//
// Let R be a RecvTensorResponse object we want to encode, logically
// constructed by filling in data from "is_dead" and "val" and filling
// in a few other fields as well.
//
// (Letters here are used in the code to refer back to which part of the
//  encoding the code is generating).
//
// A:   <protocol buffer encoding of fields except R.tensor()>
// B1:  <tag encoding for RecvTensorResponse::tensor>
// B2:  <varint32 length of R.tensor() sub message>
// C:   <protocol buffer encoding of R.tensor() except for
//          R.tensor().tensor_content()>
// D1:  <tag encoding for TensorProto::tensor_content>
// D2:  <varint32 length of R.tensor().tensor_content() data>
// E:   <actual data for val's representation>
//
// If the tensor data is up to "kLargeTensorBytes", then A
// through E will all be encoded into "*result" in a single gpr_slice.
//
// If the tensor data is larger than "kLargeTensorBytes", then A through
// D2 will be encoded in one gpr_slice, and E will be encoded in a second
// gpr_slice that points to the backing store for the tensor data, to avoid
// copying the tensor data (and the gpr_slice setup will be arrange so as
// to dereference the underlying tensor data buffer when it is no longer
// needed in the "*result" ByteBuffer).
static int VarLengthEncodingSize(uint32 tag, size_t bytes) {
  return core::VarintLength(tag << 3) + core::VarintLength(bytes) + bytes;
}

// Returns an upper bound in bytes of the protocol buffer encoding of
// the "skeleton" of "val" (all the data needed for dtype and the shape,
// but not the actual contents of "val").
static int SkeletonEncodingSizeUpperBound(const Tensor& val) {
  static const int kVarintMax64 = 10;  // Max length of varint64 encoding
  const int ndims = val.shape().dims();
  return (2 * kVarintMax64) +           // dtype
         (ndims * (4 * kVarintMax64));  // Shape: 4 varints per dim
}

// Encode the skeleton for "val" (the encoded TensorProto contents
// (dtype and shape, but not the actual data) into "*e".  The backing
// store for "*e" must be of appropriate size to hold this encoding.
static void EncodeSkeleton(const Tensor& val, io::ProtoEncodeHelper* e) {
  // Encode val.dtype()
  e->WriteUint64(TensorProto::kDtypeFieldNumber, val.dtype());

  // Compute length of val.shape() proto encoding
  const int ndims = val.shape().dims();
  int tensor_shape_bytes = 0;
  for (int d = 0; d < ndims; d++) {
    int64 dim_size = val.shape().dim_size(d);
    tensor_shape_bytes +=
        2 +  // TensorShapeProto dim tag + varintlength of submessage
        1 +  // TensorShapeProto_Dim::kSizeFieldNumber
        core::VarintLength(dim_size);
  }

  if (tensor_shape_bytes > 0) {
    e->WriteVarlengthBeginning(TensorProto::kTensorShapeFieldNumber,
                               tensor_shape_bytes);
    // Encode val.shape()
    for (int d = 0; d < ndims; d++) {
      int64 dim_size = val.shape().dim_size(d);
      int64 dim_varlen = 1 +  // TensorShapeProto_Dim::kSizeFieldNumber
                         core::VarintLength(dim_size);
      e->WriteVarlengthBeginning(TensorShapeProto::kDimFieldNumber, dim_varlen);
      e->WriteUint64(TensorShapeProto_Dim::kSizeFieldNumber, dim_size);
    }
  }

#ifndef NDEBUG
  {
    // Debug-mode only check to make sure the encoding above is
    // identical to the auto-generated protocol buffer encoding.
    TensorProto skeleton;
    skeleton.set_dtype(val.dtype());
    val.shape().AsProto(skeleton.mutable_tensor_shape());
    string tensor_except_contents;  // tensor() field except contents
    skeleton.AppendToString(&tensor_except_contents);
    TensorProto skeleton2;
    skeleton2.ParseFromString(string(e->data(), e->size()));
    string out;
    skeleton.AppendToString(&out);
    DCHECK_EQ(tensor_except_contents, out) << skeleton.DebugString() << " vs\n"
                                           << skeleton2.DebugString();
  }
#endif
}

void EncodeTensorToByteBuffer(bool is_dead, const Tensor& val,
                              ::grpc::ByteBuffer* result) {
  const int kLargeTensorBytes = 1024;
  RecvTensorResponse response;
  if (is_dead) {
    response.set_is_dead(is_dead);
  }
  response.set_send_start_micros(Env::Default()->NowMicros());
  if (!DataTypeCanUseMemcpy(val.dtype())) {
    // Straightforward but slow path for complicated kinds of tensor data
    // TODO(jeff,sanjay): If this becomes an issue, we could
    // go directly from val -> ByteBuffer, with some effort.
    val.AsProtoTensorContent(response.mutable_tensor());

    // Encode full protocol buffer to a ByteBuffer
    EncodeRecvTensorResponseToByteBuffer(response, result);
  } else {
    // skeleton is the encoded TensorProto contents (dtype and shape), but
    // not the actual data
    gtl::InlinedVector<char, 128> skeleton(SkeletonEncodingSizeUpperBound(val));
    io::ProtoEncodeHelper e_skeleton(skeleton.data(), skeleton.size());
    EncodeSkeleton(val, &e_skeleton);

    StringPiece tdata = val.tensor_data();
    uint32 overall_tensor_proto_bytesize =
        (e_skeleton.size() +
         VarLengthEncodingSize(TensorProto::kTensorContentFieldNumber,
                               tdata.size()));
    string header;  // All of RecvTensorRequest except the tensor() field
    response.AppendToString(&header);

    size_t expected_size =
        (header.size() +
         VarLengthEncodingSize(RecvTensorResponse::kTensorFieldNumber,
                               overall_tensor_proto_bytesize));
    // If "tensor_data_is_large == false", we copy the tensor data to the
    // end of the buffer we are preparing that holds the rest of the
    // RecvTensorResponse protocol buffer.
    //
    // If "tensor_data_is_large == true", we arrange to share the backing
    // store of the data by creating a slice that also points to the
    // backing store, with appropriate reference counts to keep the
    // backing store alive as needed.
    bool tensor_data_is_large = (tdata.size() > kLargeTensorBytes);
    size_t encoder_size = expected_size - tdata.size();

    // Encode all but the actual "tdata", but including the tag and
    // varlength header for the "tdata"
    gtl::InlinedVector<char, 1024> space(encoder_size);
    io::ProtoEncodeHelper e(space.data(), space.size());
    // (A)
    e.WriteRawBytes(header);

    // (B1) & (B2)
    e.WriteVarlengthBeginning(RecvTensorResponse::kTensorFieldNumber,
                              overall_tensor_proto_bytesize);
    // (C)
    e.WriteRawBytes(StringPiece(e_skeleton.data(), e_skeleton.size()));
    // (D1) & (D2)
    e.WriteVarlengthBeginning(TensorProto::kTensorContentFieldNumber,
                              tdata.size());

    // All but the tensor backing store are serialized now

    // Now allocate memory and put into the ByteBuffer
    ::grpc::Slice slices[3];
    int num_slices = 0;
    {
      size_t slice_len = e.size() + (tensor_data_is_large ? 0 : tdata.size());
      gpr_slice s0 = gpr_slice_malloc(slice_len);
      memcpy(GPR_SLICE_START_PTR(s0), e.data(), e.size());
      if (!tensor_data_is_large) {
        // (E)
        memcpy(GPR_SLICE_START_PTR(s0) + e.size(), tdata.data(), tdata.size());
      }
      slices[0] = ::grpc::Slice(s0, ::grpc::Slice::STEAL_REF);
      num_slices += 1;
    }

    if (tensor_data_is_large) {
      // Encode the actual tensor data by pointing to the backing store,
      // and add a special zero-length slice that is really a TensorReference
      // object that we will destroy when we are done.
      //
      // TODO(jeff): Note that this approach relies on the fact that
      // slices are destroyed in the order in which they are added to
      // the ByteBuffer.  In principle, these could be broken by future
      // hypothetical grpc_slice-related changes (e.g. the
      // implementation could decide to destroy 0-length slices
      // eagerly).  In practice, this does not happen with the current
      // implementation, and the gpr_slice interface at the moment does
      // not allow us to do the Tensor-unreferencing in the right way
      // (since the Tensor pointer is different than the backing store
      // array pointer).
      //
      // TODO(jeff,sanjay): switch to using new
      // gsr_slice_new_with_user_data interface that allows for
      // different pointers for the data and the argument to the
      // destroy function once that has been added integrated into grpc
      // (see https://github.com/grpc/grpc/pull/7488)

      // (E) Encode tensor data, but by sharing backing store

      // TODO(jeff,sanjay): It'd be nice to avoid this TensorReference
      // allocation, and instead get our hands on the underlying
      // TensorBuffer object and just directly ref it here and unref
      // it in unref_tensorreference.
      TensorReference* ref = new TensorReference(val);
      gpr_slice s1 = gpr_slice_new(
          const_cast<void*>(static_cast<const void*>(tdata.data())),
          tdata.size(), do_nothing);
      slices[1] = ::grpc::Slice(s1, ::grpc::Slice::STEAL_REF);

      gpr_slice s2 = gpr_slice_new(ref, 0, unref_tensorreference);
      slices[2] = ::grpc::Slice(s2, ::grpc::Slice::STEAL_REF);
      num_slices += 2;
    }
    size_t total_bytes = 0;
    for (int i = 0; i < num_slices; i++) {
      total_bytes += slices[i].size();
    }
    CHECK_EQ(total_bytes, expected_size);

    *result = ::grpc::ByteBuffer(&slices[0], num_slices);
  }
}

}  // namespace grpc
}  // namespace tensorflow
