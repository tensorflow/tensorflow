//==============================================================================
//
//  Copyright (c) Qualcomm Innovation Center, Inc.
//  All Rights Reserved.
//
//==============================================================================
#ifndef TENSORFLOW_LITE_DELEGATES_HEXAGON_BACKEND_UTILS_H_
#define TENSORFLOW_LITE_DELEGATES_HEXAGON_BACKEND_UTILS_H_

// constexpr config values
constexpr int kSleepMinLatency = 40;
constexpr int kSleepLowLatency = 100;
constexpr int kSleepMediumLatency = 1000;
constexpr int kSleepHighLatency = 2000;
constexpr int kSleepMaxLatency = 65535;
constexpr int kDcvsDisable = 0;
constexpr int kDcvsEnable = 1;

// default rpc control latency - 0 us
constexpr int kRpcControlLatency = 0;
// default rpc polling time for high power modes - 9999 us
constexpr int kRpcPollingTimeHighPower = 9999;

#endif
