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

#ifndef THIRD_PARTY_HWLOC_STATIC_COMPONENTS_H_
#define THIRD_PARTY_HWLOC_STATIC_COMPONENTS_H_

#include <private/internal-components.h>
static const struct hwloc_component* hwloc_static_components[] = {
    &hwloc_noos_component,
    &hwloc_xml_component,
    &hwloc_synthetic_component,
    &hwloc_xml_nolibxml_component,
#ifdef __linux__
    &hwloc_linux_component,
    &hwloc_linuxio_component,
#endif
#ifdef __FreeBSD__
    &hwloc_freebsd_component,
#endif
#if defined(__x86_64__) || defined(__amd64__) || defined(_M_IX86) || \
    defined(_M_X64)
    &hwloc_x86_component,
#endif
    NULL};

#endif  // THIRD_PARTY_HWLOC_STATIC_COMPONENTS_H_
