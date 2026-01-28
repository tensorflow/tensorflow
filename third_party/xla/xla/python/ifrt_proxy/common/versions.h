/* Copyright 2024 The OpenXLA Authors.

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

#ifndef XLA_PYTHON_IFRT_PROXY_COMMON_VERSIONS_H_
#define XLA_PYTHON_IFRT_PROXY_COMMON_VERSIONS_H_

namespace xla {
namespace ifrt {
namespace proxy {

namespace protocol_version {

enum {
  // There should not be any references to kAncient and earlier versions in the
  // code base.
  kAncient = 15,

  // kExecutableDevices adds a devices() method to Executable.
  kExecutableDevices = 16,

  // Optimize large transfers with the proxy-server and client in the same
  // machine to by using the file system.
  kGrpcAllowLargeTransferOptimizationViaSharedDirectory = 17,

  // kLoadedExecutableGetCostAnalysis implements GetCostAnalysis in Executable.
  kLoadedExecutableGetCostAnalysis = 18,

  // kLoadedExecutableGetHumanReadableProgramText implements
  // GetHumanReadableProgramText in Executable.
  kLoadedExecutableGetHumanReadableProgramText = 19,

  // kMpmdLoadedExecutableMethods implements MpmdLoadedExecutable methods such
  // as GetMpmdAddressableDevices, GetMpmdCompiledMemoryStats, and
  // GetMpmdCostAnalysis.
  kMpmdLoadedExecutableMethods = 20,

  // kExecuteResult adds a separate request/response type for Execution
  // results to return extra information such as device time measurement.
  kExecuteResult = 21,

  // kDevicePlatformName adds a PlatformName() method to Device.
  kDevicePlatformName = 22,

  // MakeArrayFromHostBuffer supports a layout argument.
  kMakeArrayFromHostBufferWithLayout = 23,

  // kSentiel is used to derive kCurrent below. Keep this as the last value of
  // the enum.
  kSentiel,
};

// The maximum protocol_version that the current client and server code
// understand.
constexpr int kCurrent = kSentiel - 1;

// The minimum and maximum protocol_version that the current client code
// understands.
inline constexpr int kClientMin = kExecutableDevices;
inline constexpr int kClientMax = kCurrent;

// The minimum and maximum protocol_version that the current server code
// understands.
inline constexpr int kServerMin = kExecutableDevices;
inline constexpr int kServerMax = kCurrent;

}  // namespace protocol_version

}  // namespace proxy
}  // namespace ifrt
}  // namespace xla

#endif  // XLA_PYTHON_IFRT_PROXY_COMMON_VERSIONS_H_
