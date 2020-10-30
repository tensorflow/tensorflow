// Copyright (c) 2020, XMOS Ltd, All rights reserved
#ifndef XCORE_PLANNING_H_
#define XCORE_PLANNING_H_
#include <cassert>
#include <cstdint>
#include <iostream>

#include "tensorflow/lite/c/common.h"

namespace tflite {
namespace ops {
namespace micro {
namespace xcore {

constexpr size_t changrp_len = (16);
constexpr size_t bso_changrp_len = (7 * changrp_len);
constexpr size_t bso_changrp_bytes = (bso_changrp_len * 2);

typedef struct RowColRegion {
  int32_t top;
  int32_t left;
  int32_t rows;
  int32_t cols;
} RowColRegion;

class RowColRegionArray {
 public:
  RowColRegionArray();
  void Init(TfLiteContext *ctx, size_t size);
  const RowColRegion &operator[](int i);
  void Append(const RowColRegion &region);

  size_t GetSize();

 private:
  int32_t next_;
  int32_t size_;
  RowColRegion *regions_;
};

typedef struct ChannelGroup {
  int32_t index;
  int32_t start;
  int32_t size;
} ChannelGroup;

class ChannelGroupArray {
 public:
  ChannelGroupArray();
  void Init(TfLiteContext *ctx, size_t size);
  const ChannelGroup &operator[](int i);
  void Append(const ChannelGroup &changrp);
  size_t GetSize();

 private:
  int32_t next_;
  int32_t size_;
  ChannelGroup *chan_groups_;
};

class ExecutionPlan {
 public:
  ExecutionPlan();
  ~ExecutionPlan() {}

  void SetNumThreads(int32_t n_threads) { n_threads_ = n_threads; }
  size_t GetNumThreads() { return n_threads_; }

  void SetWeightsScratchSize(size_t size);
  size_t GetWeightsScratchSize();
  size_t GetWeightsScratchOffset();

  void SetBiasScratchSize(size_t size);
  size_t GetBiasScratchSize();
  size_t GetBiasScratchOffset();

  RowColRegionArray regions;
  ChannelGroupArray changrps;

 private:
  size_t n_threads_;
  size_t bias_scratch_offset_;
  size_t bias_scratch_size_;
};

}  // namespace xcore
}  // namespace micro
}  // namespace ops
}  // namespace tflite

#endif  // XCORE_PLANNING_H_