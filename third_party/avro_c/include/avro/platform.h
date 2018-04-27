/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to you under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.  See the License for the specific language governing
 * permissions and limitations under the License.
 */

#ifndef AVRO_PLATFORM_H
#define AVRO_PLATFORM_H
#ifdef __cplusplus
extern "C" {
#define CLOSE_EXTERN }
#else
#define CLOSE_EXTERN
#endif

/* Use this header file to include platform specific definitions */

#ifdef _WIN32
  #include <avro/msinttypes.h>
#else
  #include <inttypes.h>
#endif

// Defines for printing size_t.
#if defined(_WIN64)
  #define PRIsz PRIu64
#elif defined(_WIN32)
  #define PRIsz PRIu32
#else // GCC
  #define PRIsz "zu"
#endif

CLOSE_EXTERN
#endif
