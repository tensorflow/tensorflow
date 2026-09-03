// Standalone stub implementation used for HOST-SIDE testing of the
// TF integration code paths. This file is NOT compiled into the
// production TF binary — it is only used by the harness in
// tensorflow/core/kernels/kdnn/kdnn_test_harness.cc (and the
// adversarial test scripts) to verify that:
//
//   * the C ABI matches between kdnn.h and the library,
//   * the dispatch path works (create / apply / destroy),
//   * graceful fallback (unsupported activation) returns KDNN_ERR_UNSUPPORTED.
//
// Real KDNN is shipped as libkdnn.so — see third_party/KDNN/README.md.

// Real production path uses the BUILD include_prefix:
//   #include "third_party/KDNN/kdnn.h"
// For the host-side test harness we use the relative path so we can
// compile with `-I third_party/KDNN`.
#include "kdnn.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

extern "C" {

struct kdnn_context {
  char last_error[256];
};

kdnn_status_t kdnn_create(kdnn_context_t** ctx) {
  if (ctx == nullptr) return KDNN_ERR_INVALID_ARG;
  *ctx = static_cast<kdnn_context_t*>(std::calloc(1, sizeof(kdnn_context_t)));
  return KDNN_OK;
}

void kdnn_destroy(kdnn_context_t* ctx) {
  if (ctx != nullptr) std::free(ctx);
}

const char* kdnn_get_last_error(const kdnn_context_t* ctx) {
  if (ctx == nullptr) return "null context";
  return ctx->last_error;
}

int kdnn_get_version(void) {
  return 0x00010000;  // 1.0
}

int kdnn_activation_supported(kdnn_activation_t activation,
                              kdnn_data_type_t dtype) {
  // Stub: only float32 sigmoid is supported in the test harness.
  if (activation == KDNN_ACT_SIGMOID && dtype == KDNN_DT_FLOAT) return 1;
  return 0;
}

kdnn_status_t kdnn_apply_activation(
    kdnn_context_t* ctx,
    kdnn_activation_t activation,
    kdnn_data_type_t dtype,
    kdnn_layout_t /*layout*/,
    void* x,
    void* y,
    size_t n) {
  if (ctx == nullptr || x == nullptr || y == nullptr) {
    // KDNN_ERR_INVALID_ARG == 1
    return static_cast<kdnn_status_t>(1);
  }
  if (!kdnn_activation_supported(activation, dtype)) {
    // snprintf is in <cstdio> as a non-std:: name; in <stdio.h> as
    // a global. Calling std::snprintf is invalid because no std::
    // overload is declared. Use the global name from <cstdio>.
    snprintf(ctx->last_error, sizeof(ctx->last_error),
             "activation %d / dtype %d not supported", activation, dtype);
    return KDNN_ERR_UNSUPPORTED;
  }
  // Only float32 sigmoid is implemented in the stub.
  if (dtype != KDNN_DT_FLOAT) return KDNN_ERR_UNSUPPORTED;
  if (activation != KDNN_ACT_SIGMOID) return KDNN_ERR_UNSUPPORTED;

  const float* xf = static_cast<const float*>(x);
  float* yf = static_cast<float*>(y);
  for (size_t i = 0; i < n; ++i) {
    // 1 / (1 + exp(-x)) — naive but sufficient for harness tests.
    yf[i] = 1.0f / (1.0f + std::exp(-xf[i]));
  }
  return KDNN_OK;
}

}  // extern "C"
