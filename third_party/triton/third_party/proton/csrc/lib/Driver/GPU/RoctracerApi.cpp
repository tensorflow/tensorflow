#include "Driver/GPU/RoctracerApi.h"
#include "Driver/Dispatch.h"

namespace proton {

namespace roctracer {

struct ExternLibRoctracer : public ExternLibBase {
  using RetType = roctracer_status_t;
  static constexpr const char *name = "libroctracer64.so";
  static constexpr const char *defaultDir = "";
  static constexpr RetType success = ROCTRACER_STATUS_SUCCESS;
  static void *lib;
};

void *ExternLibRoctracer::lib = nullptr;

DEFINE_DISPATCH(ExternLibRoctracer, setProperties, roctracer_set_properties,
                roctracer_domain_t, void *)

DEFINE_DISPATCH(ExternLibRoctracer, getTimestamp, roctracer_get_timestamp,
                roctracer_timestamp_t *)

void start() {
  typedef void (*roctracer_start_t)();
  static roctracer_start_t func = nullptr;
  Dispatch<ExternLibRoctracer>::init(ExternLibRoctracer::name,
                                     &ExternLibRoctracer::lib);
  if (func == nullptr)
    func = reinterpret_cast<roctracer_start_t>(
        dlsym(ExternLibRoctracer::lib, "roctracer_start"));
  if (func)
    func();
}

void stop() {
  typedef void (*roctracer_stop_t)();
  static roctracer_stop_t func = nullptr;
  Dispatch<ExternLibRoctracer>::init(ExternLibRoctracer::name,
                                     &ExternLibRoctracer::lib);
  if (func == nullptr)
    func = reinterpret_cast<roctracer_stop_t>(
        dlsym(ExternLibRoctracer::lib, "roctracer_stop"));
  if (func)
    func();
}

char *getOpString(uint32_t domain, uint32_t op, uint32_t kind) {
  typedef char *(*roctracer_op_string_t)(uint32_t, uint32_t, uint32_t);
  static roctracer_op_string_t func = nullptr;
  Dispatch<ExternLibRoctracer>::init(ExternLibRoctracer::name,
                                     &ExternLibRoctracer::lib);
  if (func == nullptr)
    func = reinterpret_cast<roctracer_op_string_t>(
        dlsym(ExternLibRoctracer::lib, "roctracer_op_string"));
  return (func ? func(domain, op, kind) : NULL);
}

DEFINE_DISPATCH(ExternLibRoctracer, enableDomainCallback,
                roctracer_enable_domain_callback, activity_domain_t,
                activity_rtapi_callback_t, void *)

DEFINE_DISPATCH(ExternLibRoctracer, disableDomainCallback,
                roctracer_disable_domain_callback, activity_domain_t)

DEFINE_DISPATCH(ExternLibRoctracer, enableOpCallback,
                roctracer_enable_op_callback, activity_domain_t, uint32_t,
                activity_rtapi_callback_t, void *)

DEFINE_DISPATCH(ExternLibRoctracer, disableOpCallback,
                roctracer_disable_op_callback, activity_domain_t, uint32_t)

DEFINE_DISPATCH(ExternLibRoctracer, openPool, roctracer_open_pool,
                const roctracer_properties_t *)

DEFINE_DISPATCH(ExternLibRoctracer, closePool, roctracer_close_pool)

DEFINE_DISPATCH(ExternLibRoctracer, enableOpActivity,
                roctracer_enable_op_activity, activity_domain_t, uint32_t)

DEFINE_DISPATCH(ExternLibRoctracer, enableDomainActivity,
                roctracer_enable_domain_activity, activity_domain_t)

DEFINE_DISPATCH(ExternLibRoctracer, disableOpActivity,
                roctracer_disable_op_activity, activity_domain_t, uint32_t)

DEFINE_DISPATCH(ExternLibRoctracer, disableDomainActivity,
                roctracer_disable_domain_activity, activity_domain_t)

DEFINE_DISPATCH(ExternLibRoctracer, flushActivity, roctracer_flush_activity)

DEFINE_DISPATCH(ExternLibRoctracer, activityPushExternalCorrelationId,
                roctracer_activity_push_external_correlation_id,
                activity_correlation_id_t)

DEFINE_DISPATCH(ExternLibRoctracer, activityPopExternalCorrelationId,
                roctracer_activity_pop_external_correlation_id,
                activity_correlation_id_t *)

DEFINE_DISPATCH(ExternLibRoctracer, getNextRecord, roctracer_next_record,
                const activity_record_t *, const activity_record_t **)

} // namespace roctracer

} // namespace proton
