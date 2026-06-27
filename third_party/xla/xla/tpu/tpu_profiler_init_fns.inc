#ifndef TENSORFLOW_COMPILER_XLA_TPU_TPU_PROFILER_INIT_FNS_INC_
#define TENSORFLOW_COMPILER_XLA_TPU_TPU_PROFILER_INIT_FNS_INC_

#include "xla/tpu/tpu_api.h"
#include "xla/tpu/tpu_profiler_c_api.h"

namespace {

absl::Status DoSetTpuProfilerApiFns(void* library_handle) {
  static TfTpu_ProfilerApiFn profiler_api_fn;
  TFTPU_SET_FN(&profiler_api_fn, TpuProfiler_Create);
  TFTPU_SET_FN(&profiler_api_fn, TpuProfiler_Destroy);
  TFTPU_SET_FN(&profiler_api_fn, TpuProfiler_Start);
  TFTPU_SET_FN(&profiler_api_fn, TpuProfiler_Stop);
  TFTPU_SET_FN(&profiler_api_fn, TpuProfiler_CollectData);

  TFTPU_SET_FN(&profiler_api_fn, TpuStatus_New);
  TFTPU_SET_FN(&profiler_api_fn, TpuStatus_Free);
  TFTPU_SET_FN(&profiler_api_fn, TpuStatus_Message);
  TFTPU_SET_FN(&profiler_api_fn, TpuStatus_Code);

  stream_executor::tpu::SetProfilerApiFn(&profiler_api_fn);
  return absl::OkStatus();
}

absl::Status SetTpuProfilerApiFns(void* library_handle) {
  absl::call_once(*::stream_executor::tpu::GetTpuProfilerInitOnceFlag(),
                  [library_handle]() {
                    ::stream_executor::tpu::GetTpuProfilerInitStatus() =
                        DoSetTpuProfilerApiFns(library_handle);
                  });
  return ::stream_executor::tpu::GetTpuProfilerInitStatus();
}

}  // namespace

#endif  // TENSORFLOW_COMPILER_XLA_TPU_TPU_PROFILER_INIT_FNS_INC_
