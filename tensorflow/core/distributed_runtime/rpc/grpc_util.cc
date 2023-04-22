/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/distributed_runtime/rpc/grpc_util.h"
#include "tensorflow/core/distributed_runtime/tensor_coding.h"
#include "tensorflow/core/lib/random/random.h"

namespace tensorflow {

namespace {

double GenerateUniformRandomNumber() {
  return random::New64() * (1.0 / std::numeric_limits<uint64>::max());
}

double GenerateUniformRandomNumberBetween(double a, double b) {
  if (a == b) return a;
  DCHECK_LT(a, b);
  return a + GenerateUniformRandomNumber() * (b - a);
}

}  // namespace

int64 ComputeBackoffMicroseconds(int current_retry_attempt, int64_t min_delay,
                                 int64_t max_delay) {
  DCHECK_GE(current_retry_attempt, 0);

  // This function with the constants below is calculating:
  //
  // (0.4 * min_delay) + (random[0.6,1.0] * min_delay * 1.3^retries)
  //
  // Note that there is an extra truncation that occurs and is documented in
  // comments below.
  constexpr double kBackoffBase = 1.3;
  constexpr double kBackoffRandMult = 0.4;

  // This first term does not vary with current_retry_attempt or a random
  // number. It exists to ensure the final term is >= min_delay
  const double first_term = kBackoffRandMult * min_delay;

  // This is calculating min_delay * 1.3^retries
  double uncapped_second_term = min_delay;
  while (current_retry_attempt > 0 &&
         uncapped_second_term < max_delay - first_term) {
    current_retry_attempt--;
    uncapped_second_term *= kBackoffBase;
  }
  // Note that first_term + uncapped_second_term can exceed max_delay here
  // because of the final multiply by kBackoffBase.  We fix that problem with
  // the min() below.
  double second_term = std::min(uncapped_second_term, max_delay - first_term);

  // This supplies the random jitter to ensure that retried don't cause a
  // thundering herd problem.
  second_term *=
      GenerateUniformRandomNumberBetween(1.0 - kBackoffRandMult, 1.0);

  return std::max(static_cast<int64>(first_term + second_term), min_delay);
}

::grpc::Status GrpcMaybeUnparseProto(const protobuf::Message& src,
                                     grpc::ByteBuffer* dst) {
  bool own_buffer;
  return ::grpc::GenericSerialize<::grpc::ProtoBufferWriter,
                                  protobuf::Message>(src, dst, &own_buffer);
}

// GrpcMaybeUnparseProto from a string simply copies the string to the
// ByteBuffer.
::grpc::Status GrpcMaybeUnparseProto(const string& src, grpc::ByteBuffer* dst) {
  ::grpc::Slice s(src.data(), src.size());
  ::grpc::ByteBuffer buffer(&s, 1);
  dst->Swap(&buffer);
  return ::grpc::Status::OK;
}

bool GrpcMaybeParseProto(::grpc::ByteBuffer* src, protobuf::Message* dst) {
  ::grpc::ProtoBufferReader reader(src);
  return dst->ParseFromZeroCopyStream(&reader);
}

// Overload of GrpcParseProto so we can decode a TensorResponse without
// extra copying.  This overload is used by the RPCState class in
// grpc_state.h.
bool GrpcMaybeParseProto(::grpc::ByteBuffer* src, TensorResponse* dst) {
  ::tensorflow::GrpcByteSource byte_source(src);
  auto s = dst->ParseFrom(&byte_source);
  return s.ok();
}

// GrpcMaybeParseProto simply copies bytes into the string.
bool GrpcMaybeParseProto(grpc::ByteBuffer* src, string* dst) {
  dst->clear();
  dst->reserve(src->Length());
  std::vector<::grpc::Slice> slices;
  if (!src->Dump(&slices).ok()) {
    return false;
  }
  for (const ::grpc::Slice& s : slices) {
    dst->append(reinterpret_cast<const char*>(s.begin()), s.size());
  }
  return true;
}

// GrpcMaybeParseProto simply copies bytes into the tstring.
bool GrpcMaybeParseProto(grpc::ByteBuffer* src, tstring* dst) {
  dst->clear();
  dst->reserve(src->Length());
  std::vector<::grpc::Slice> slices;
  if (!src->Dump(&slices).ok()) {
    return false;
  }
  for (const ::grpc::Slice& s : slices) {
    dst->append(reinterpret_cast<const char*>(s.begin()), s.size());
  }
  return true;
}

}  // namespace tensorflow
