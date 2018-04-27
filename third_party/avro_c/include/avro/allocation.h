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

#ifndef AVRO_ALLOCATION_H
#define AVRO_ALLOCATION_H
#ifdef __cplusplus
extern "C" {
#define CLOSE_EXTERN }
#else
#define CLOSE_EXTERN
#endif

#include <stdlib.h>

/*
 * Allocation interface.  You can provide a custom allocator for the
 * library, should you wish.  The allocator is provided as a single
 * generic function, which can emulate the standard malloc, realloc, and
 * free functions.  The design of this allocation interface is inspired
 * by the implementation of the Lua interpreter.
 *
 * The ptr parameter will be the location of any existing memory
 * buffer.  The osize parameter will be the size of this existing
 * buffer.  If ptr is NULL, then osize will be 0.  The nsize parameter
 * will be the size of the new buffer, or 0 if the new buffer should be
 * freed.
 *
 * If nsize is 0, then the allocation function must return NULL.  If
 * nsize is not 0, then it should return NULL if the allocation fails.
 */

typedef void *
(*avro_allocator_t)(void *user_data, void *ptr, size_t osize, size_t nsize);

void avro_set_allocator(avro_allocator_t alloc, void *user_data);

struct avro_allocator_state {
	avro_allocator_t  alloc;
	void  *user_data;
};

extern struct avro_allocator_state  AVRO_CURRENT_ALLOCATOR;

#define avro_realloc(ptr, osz, nsz)          \
	(AVRO_CURRENT_ALLOCATOR.alloc        \
	 (AVRO_CURRENT_ALLOCATOR.user_data,  \
	  (ptr), (osz), (nsz)))

#define avro_malloc(sz) (avro_realloc(NULL, 0, (sz)))
#define avro_free(ptr, osz) (avro_realloc((ptr), (osz), 0))

#define avro_new(type) (avro_realloc(NULL, 0, sizeof(type)))
#define avro_freet(type, ptr) (avro_realloc((ptr), sizeof(type), 0))

void *avro_calloc(size_t count, size_t size);

/*
 * This is probably too clever for our own good, but when we duplicate a
 * string, we actually store its size in the same allocated memory
 * buffer.  That lets us free the string later, without having to call
 * strlen to get its size, and without the containing struct having to
 * manually store the strings length.
 *
 * This means that any string return by avro_strdup MUST be freed using
 * avro_str_free, and the only thing that can be passed into
 * avro_str_free is a string created via avro_strdup.
 */

char *avro_str_alloc(size_t str_size);
char *avro_strdup(const char *str);
void avro_str_free(char *str);

CLOSE_EXTERN
#endif
