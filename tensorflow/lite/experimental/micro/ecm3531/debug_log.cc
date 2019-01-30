#include "tensorflow/lite/experimental/micro/debug_log.h"

#include "eta_csp_io.h"

extern "C" void DebugLog(const char* s) {
 EtaCspIoPrintf("%s", s);
}
