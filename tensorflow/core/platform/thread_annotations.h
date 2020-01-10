#ifndef TENSORFLOW_PLATFORM_THREAD_ANNOTATIONS_H_
#define TENSORFLOW_PLATFORM_THREAD_ANNOTATIONS_H_

#include "tensorflow/core/platform/port.h"

#if defined(PLATFORM_GOOGLE) || defined(PLATFORM_GOOGLE_ANDROID)
#include "base/thread_annotations.h"
#elif defined(PLATFORM_POSIX) || defined(PLATFORM_POSIX_ANDROID)
#include "tensorflow/core/platform/default/thread_annotations.h"
#else
#error Define the appropriate PLATFORM_<foo> macro for this platform
#endif

#endif  // TENSORFLOW_PLATFORM_THREAD_ANNOTATIONS_H_
