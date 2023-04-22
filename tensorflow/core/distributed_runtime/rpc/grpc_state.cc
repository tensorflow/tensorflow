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
#include "tensorflow/core/distributed_runtime/rpc/grpc_state.h"

#include "absl/strings/str_format.h"

namespace tensorflow {

const char* ToString(UntypedStreamingRPCState::Tag::TagType tag_type) {
  switch (tag_type) {
    case UntypedStreamingRPCState::Tag::TagType::kCallStarted:
      return "kCallStarted";
    case UntypedStreamingRPCState::Tag::TagType::kRequestWriteCompleted:
      return "kRequestWriteCompleted";
    case UntypedStreamingRPCState::Tag::TagType::kResponseReadCompleted:
      return "kResponseReadCompleted";
    case UntypedStreamingRPCState::Tag::TagType::kCallFinished:
      return "kCallFinished";
  }
}

UntypedStreamingRPCState::Tag::Tag(UntypedStreamingRPCState* streaming_state,
                                   Tag::TagType type)
    : streaming_state_(streaming_state), type_(type) {}

void UntypedStreamingRPCState::Tag::OnCompleted(bool ok) {
  switch (type_) {
    case TagType::kCallStarted:
      streaming_state_->CallStarted(ok);
      break;
    case TagType::kRequestWriteCompleted:
      streaming_state_->RequestWriteCompleted(ok);
      break;
    case TagType::kResponseReadCompleted:
      streaming_state_->ResponseReadCompleted(ok);
      break;
    case TagType::kCallFinished:
      streaming_state_->CallFinished(ok);
      break;
  }
  streaming_state_->Unref();  // Ref acquired when tag was handed to grpc.
}

void Exchange::Complete(Status status) {
  if (status.ok()) {
    if (!GrpcMaybeParseProto(&response_buf_, response_)) {
      status.Update(errors::Internal("could not parse rpc response"));
    }
  }
  VLOG(3) << "Completing exchange " << DebugString() << " with "
          << status.ToString();
  cb_(status);
}

std::ostream& operator<<(std::ostream& os, const Exchange::State& state) {
  os << ToString(state);
  return os;
}

const char* ToString(Exchange::State state) {
  switch (state) {
    case Exchange::State::kExchangeCreated:
      return "ExchangeCreated";
    case Exchange::State::kRequestWriteIssued:
      return "RequestWriteIssued";
    case Exchange::State::kRequestWriteCompleted:
      return "RequestWriteCompleted";
    case Exchange::State::kResponseReadIssued:
      return "ResponseReadIssued";
  }
}

string Exchange::DebugString() const {
  return absl::StrFormat("%p@%s_%s", this, ToString(state_), debug_string_);
}

void ExchangeQueue::Emplace(const ::grpc::ByteBuffer& request_buf,
                            protobuf::Message* response, StatusCallback cb,
                            string debug_string) {
  exchanges_.emplace(exchanges_.end(), request_buf, response, std::move(cb),
                     debug_string);
}

Exchange* ExchangeQueue::GetReadyForRequestWriting() {
  CheckInvariants();
  if (!call_started_) {
    return nullptr;
  }

  // TODO(iga): Optimize to avoid linear search.
  for (Exchange& e : exchanges_) {
    if (e.state() == Exchange::State::kExchangeCreated) {
      return &e;
    } else if (e.state() == Exchange::State::kRequestWriteIssued) {
      return nullptr;
    }
  }
  return nullptr;
}

Exchange* ExchangeQueue::GetReadyForResponseReading() {
  CheckInvariants();
  if (!call_started_) {
    // We should never ask for response reading when call has not
    // been started, but it does not hurt to defensively check here anyway.
    return nullptr;
  }
  if (exchanges_.empty()) {
    return nullptr;
  }
  Exchange& e = exchanges_[0];
  if (e.state() == Exchange::State::kRequestWriteCompleted) {
    return &e;
  }
  return nullptr;
}

void ExchangeQueue::MarkRequestWriteCompleted() {
  CheckInvariants();
  // TODO(iga): Optimize to avoid linear search.
  for (Exchange& e : exchanges_) {
    if (e.state() == Exchange::State::kRequestWriteIssued) {
      e.MarkRequestWriteCompleted();
    }
  }
  CheckInvariants();
}

Exchange& ExchangeQueue::GetFront() {
  CheckInvariants();
  return exchanges_.front();
}

void ExchangeQueue::PopFront() {
  CheckInvariants();
  exchanges_.pop_front();
}

string ExchangeQueue::DebugString() const {
  return absl::StrJoin(exchanges_, ", ", [](string* out, const Exchange& e) {
    out->append(e.DebugString());
  });
}

void ExchangeQueue::Swap(ExchangeQueue* other) {
  exchanges_.swap(other->exchanges_);
  std::swap(call_started_, other->call_started_);
}

void ExchangeQueue::CompleteAll(Status status) {
  for (Exchange& exchange : exchanges_) {
    exchange.Complete(status);
  }
}

namespace {
std::set<std::pair<Exchange::State, Exchange::State>>*
GetPossibleTransitions() {
  std::set<std::pair<Exchange::State, Exchange::State>>* s =
      new std::set<std::pair<Exchange::State, Exchange::State>>();
  // Regular state transitions
  s->emplace(Exchange::State::kExchangeCreated,
             Exchange::State::kRequestWriteIssued);
  s->emplace(Exchange::State::kRequestWriteIssued,
             Exchange::State::kRequestWriteCompleted);
  s->emplace(Exchange::State::kRequestWriteCompleted,
             Exchange::State::kResponseReadIssued);
  // Self transitions. Possible when several exchanges can be in
  // the same state.
  s->emplace(Exchange::State::kExchangeCreated,
             Exchange::State::kExchangeCreated);
  s->emplace(Exchange::State::kRequestWriteCompleted,
             Exchange::State::kRequestWriteCompleted);
  // Skip transitions. Possible when there are no exchanges in a
  // certain state.
  s->emplace(Exchange::State::kExchangeCreated,
             Exchange::State::kRequestWriteCompleted);
  s->emplace(Exchange::State::kExchangeCreated,
             Exchange::State::kResponseReadIssued);
  s->emplace(Exchange::State::kRequestWriteIssued,
             Exchange::State::kResponseReadIssued);
  return s;
}
}  // namespace

void ExchangeQueue::CheckInvariants() {
  static std::set<std::pair<Exchange::State, Exchange::State>>*
      possible_transitions = GetPossibleTransitions();

  if (!VLOG_IS_ON(5)) {
    return;
  }

  for (int i = 1, end = exchanges_.size(); i < end; ++i) {
    const Exchange& e0 = exchanges_[i - 1];
    const Exchange& e1 = exchanges_[i];
    // The first exchange in the pair is the one that arrived later and is
    // behind in processing.
    auto p = std::make_pair(e1.state(), e0.state());
    if (possible_transitions->find(p) == possible_transitions->end()) {
      LOG(FATAL)
          << "Found an impossible state transition in the exchange queue: "
          << p.first << " -> " << p.second;
    }
  }
}

}  // namespace tensorflow
