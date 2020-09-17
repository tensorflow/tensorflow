// Copyright (c) 2020, XMOS Ltd, All rights reserved
#include "tensorflow/lite/micro/kernels/xcore/xcore_planning.h"

#include "tensorflow/lite/micro/kernels/xcore/xcore_dispatcher.h"

namespace tflite {
namespace ops {
namespace micro {
namespace xcore {

//*****************************
//*****************************
//*****************************
// RowColRegionArray
//*****************************
//*****************************
//*****************************

RowColRegionArray::RowColRegionArray()
    : next_(0), size_(0), regions_(nullptr) {}

void RowColRegionArray::Init(TfLiteContext *ctx, size_t size) {
  assert(regions_ == nullptr);

  size_ = size;
  regions_ = reinterpret_cast<RowColRegion *>(
      ctx->AllocatePersistentBuffer(ctx, sizeof(RowColRegion) * size_));
}

const RowColRegion &RowColRegionArray::operator[](int i) {
  assert(i < size_);
  return regions_[i];
}

void RowColRegionArray::Append(const RowColRegion &region) {
  assert(next_ < size_);
  regions_[next_] = std::move(region);
  next_++;
}

size_t RowColRegionArray::GetSize() { return next_; }

//*****************************
//*****************************
//*****************************
// ChannelGroupArray
//*****************************
//*****************************
//*****************************

ChannelGroupArray::ChannelGroupArray()
    : next_(0), size_(0), chan_groups_(nullptr) {}

void ChannelGroupArray::Init(TfLiteContext *ctx, size_t size) {
  assert(chan_groups_ == nullptr);

  size_ = size;
  chan_groups_ = reinterpret_cast<ChannelGroup *>(
      ctx->AllocatePersistentBuffer(ctx, sizeof(ChannelGroup) * size_));
}

const ChannelGroup &ChannelGroupArray::operator[](int i) {
  assert(i < GetSize());

  return chan_groups_[i];
}

void ChannelGroupArray::Append(const ChannelGroup &changrp) {
  assert(next_ < size_);
  chan_groups_[next_] = std::move(changrp);
  next_++;
}

size_t ChannelGroupArray::GetSize() { return next_; }

//*****************************
//*****************************
//*****************************
// ExecutionPlan
//*****************************
//*****************************
//*****************************
ExecutionPlan::ExecutionPlan() : n_threads_(0), bias_scratch_offset_(0) {}

void ExecutionPlan::SetWeightsScratchSize(size_t size) {
  // NOTE: Weights assumes to start at scratch offset 0
  //        so we do not need to store it
  bias_scratch_offset_ = size;
}
size_t ExecutionPlan::GetWeightsScratchSize() { return bias_scratch_offset_; }
size_t ExecutionPlan::GetWeightsScratchOffset() { return 0; }

void ExecutionPlan::SetBiasScratchSize(size_t size) {
  // NOTE: size is ignored for now because it is a constant
}
size_t ExecutionPlan::GetBiasScratchSize() { return bso_changrp_bytes; }
size_t ExecutionPlan::GetBiasScratchOffset() { return bias_scratch_offset_; }

}  // namespace xcore
}  // namespace micro
}  // namespace ops
}  // namespace tflite