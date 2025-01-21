/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_PROFILER_UTILS_XPLANE_SCHEMA_H_
#define TENSORFLOW_CORE_PROFILER_UTILS_XPLANE_SCHEMA_H_

#include "xla/tsl/profiler/utils/xplane_schema.h"

namespace tensorflow {
namespace profiler {

using tsl::profiler::FindHostEventType;             // NOLINT
using tsl::profiler::FindStatType;                  // NOLINT
using tsl::profiler::FindTaskEnvStatType;           // NOLINT
using tsl::profiler::FindTfOpEventType;             // NOLINT
using tsl::profiler::GetHostEventTypeStr;           // NOLINT
using tsl::profiler::GetStatTypeStr;                // NOLINT
using tsl::profiler::GpuPlaneName;                  // NOLINT
using tsl::profiler::HostEventType;                 // NOLINT
using tsl::profiler::IsHostEventType;               // NOLINT
using tsl::profiler::IsInternalEvent;               // NOLINT
using tsl::profiler::IsInternalStat;                // NOLINT
using tsl::profiler::IsStatType;                    // NOLINT
using tsl::profiler::kCuptiDriverApiPlaneName;      // NOLINT
using tsl::profiler::kCustomPlanePrefix;            // NOLINT
using tsl::profiler::kDeviceVendorAMD;              // NOLINT
using tsl::profiler::kDeviceVendorNvidia;           // NOLINT
using tsl::profiler::kGpuPlanePrefix;               // NOLINT
using tsl::profiler::kHostOffloadOpLineName;        // NOLINT
using tsl::profiler::kHostThreadsPlaneName;         // NOLINT
using tsl::profiler::kKernelLaunchLineName;         // NOLINT
using tsl::profiler::kMegaScaleBarrier;             // NOLINT
using tsl::profiler::kMegaScaleD2HTransferFinished;  // NOLINT
using tsl::profiler::kMegaScaleD2HTransferStart;     // NOLINT
using tsl::profiler::kMegaScaleDcnReceive;           // NOLINT
using tsl::profiler::kMegaScaleDcnSend;              // NOLINT
using tsl::profiler::kMegaScaleDcnSendFinished;      // NOLINT
using tsl::profiler::kMegaScaleH2DTransferFinished;  // NOLINT
using tsl::profiler::kMegaScaleH2DTransferStart;     // NOLINT
using tsl::profiler::kMegaScaleHostCommand;          // NOLINT
using tsl::profiler::kMegaScaleTopologyDiscovery;    // NOLINT
using tsl::profiler::kMetadataPlaneName;             // NOLINT
using tsl::profiler::kPythonTracerPlaneName;         // NOLINT
using tsl::profiler::kRoctracerApiPlaneName;         // NOLINT
using tsl::profiler::kSourceLineName;                // NOLINT
using tsl::profiler::kSparseCoreModuleLineName;      // NOLINT
using tsl::profiler::kSparseCoreOpLineName;          // NOLINT
using tsl::profiler::kSparseCorePlaneRegex;          // NOLINT
using tsl::profiler::kSparseCoreStepLineName;        // NOLINT
using tsl::profiler::kStepLineName;                  // NOLINT
using tsl::profiler::kTaskEnvPlaneName;              // NOLINT
using tsl::profiler::kTensorFlowNameScopeLineName;   // NOLINT
using tsl::profiler::kTensorFlowOpLineName;          // NOLINT
using tsl::profiler::kTFStreamzPlaneName;            // NOLINT
using tsl::profiler::kTpuPlanePrefix;                // NOLINT
using tsl::profiler::kTpuPlaneRegex;                 // NOLINT
using tsl::profiler::kTpuRuntimePlaneName;           // NOLINT
using tsl::profiler::kXlaAsyncOpLineName;            // NOLINT
using tsl::profiler::kXlaModuleLineName;             // NOLINT
using tsl::profiler::kXlaOpLineName;                 // NOLINT
using tsl::profiler::kXProfMetadataBufferSize;       // NOLINT
using tsl::profiler::kXProfMetadataFlow;             // NOLINT
using tsl::profiler::kXProfMetadataKey;              // NOLINT
using tsl::profiler::kXProfMetadataTransfers;        // NOLINT
using tsl::profiler::StatType;                       // NOLINT
using tsl::profiler::TaskEnvStatType;                // NOLINT
using tsl::profiler::TpuPlaneName;                   // NOLINT
using tsl::profiler::XFlow;                          // NOLINT

}  // namespace profiler
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PROFILER_UTILS_XPLANE_SCHEMA_H_
