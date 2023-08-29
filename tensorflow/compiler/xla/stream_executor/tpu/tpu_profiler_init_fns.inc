#include "tensorflow/compiler/xla/stream_executor/tpu/tpu_api.h"
#include "tensorflow/compiler/xla/stream_executor/tpu/tpu_profiler_c_api.h"

namespace {

tsl::Status SetTpuProfilerApiFns(void* library_handle) {  // TENSORFLOW_STATUS_OK
  // Constant cast so that we can initialize the functions. The functions are
  // mutable here because this is the only place where they are initialized.
  auto* profiler_api_fn =
      const_cast<TfTpu_ProfilerApiFn*>(stream_executor::tpu::ProfilerApiFn());
  TFTPU_SET_FN(profiler_api_fn, TpuProfiler_Create);
  TFTPU_SET_FN(profiler_api_fn, TpuProfiler_Destroy);
  TFTPU_SET_FN(profiler_api_fn, TpuProfiler_Start);
  TFTPU_SET_FN(profiler_api_fn, TpuProfiler_Stop);
  TFTPU_SET_FN(profiler_api_fn, TpuProfiler_CollectData);
  return tsl::OkStatus();
}

}  // namespace
