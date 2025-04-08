#ifndef PROTON_DRIVER_GPU_ROCTRACER_H_
#define PROTON_DRIVER_GPU_ROCTRACER_H_

#include "roctracer/roctracer.h"

namespace proton {

namespace roctracer {

template <bool CheckSuccess>
roctracer_status_t setProperties(roctracer_domain_t domain, void *properties);

template <bool CheckSuccess>
roctracer_status_t getTimestamp(roctracer_timestamp_t *timestamp);

void start();

void stop();

//
// Callbacks
//

template <bool CheckSuccess>
roctracer_status_t enableDomainCallback(activity_domain_t domain,
                                        activity_rtapi_callback_t callback,
                                        void *arg);

template <bool CheckSuccess>
roctracer_status_t disableDomainCallback(activity_domain_t domain);

template <bool CheckSuccess>
roctracer_status_t enableOpCallback(activity_domain_t domain, uint32_t op,
                                    activity_rtapi_callback_t callback,
                                    void *arg);

template <bool CheckSuccess>
roctracer_status_t disableOpCallback(activity_domain_t domain, uint32_t op);

//
// Activity
//

template <bool CheckSuccess>
roctracer_status_t openPool(const roctracer_properties_t *properties);

template <bool CheckSuccess> roctracer_status_t closePool();

template <bool CheckSuccess>
roctracer_status_t enableOpActivity(activity_domain_t domain, uint32_t op);

template <bool CheckSuccess>
roctracer_status_t enableDomainActivity(activity_domain_t domain);

template <bool CheckSuccess>
roctracer_status_t disableOpActivity(activity_domain_t domain, uint32_t op);

template <bool CheckSuccess>
roctracer_status_t disableDomainActivity(activity_domain_t domain);

template <bool CheckSuccess> roctracer_status_t flushActivity();

template <bool CheckSuccess>
roctracer_status_t getNextRecord(const activity_record_t *record,
                                 const activity_record_t **next);

char *getOpString(uint32_t domain, uint32_t op, uint32_t kind);

//
// External correlation
//

template <bool CheckSuccess>
roctracer_status_t
activityPushExternalCorrelationId(activity_correlation_id_t id);

template <bool CheckSuccess>
roctracer_status_t
activityPopExternalCorrelationId(activity_correlation_id_t *last_id);

} // namespace roctracer

} // namespace proton

#endif // PROTON_EXTERN_DISPATCH_H_
