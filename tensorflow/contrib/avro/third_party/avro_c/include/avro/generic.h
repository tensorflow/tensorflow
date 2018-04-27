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

#ifndef AVRO_GENERIC_H
#define AVRO_GENERIC_H
#ifdef __cplusplus
extern "C" {
#define CLOSE_EXTERN }
#else
#define CLOSE_EXTERN
#endif

#include <avro/platform.h>
#include <stdlib.h>

#include <avro/schema.h>
#include <avro/value.h>

/*
 * This file contains an avro_value_t implementation that can store
 * values of any Avro schema.  It replaces the old avro_datum_t class.
 */


/**
 * Return a generic avro_value_iface_t implementation for the given
 * schema, regardless of what type it is.
 */

avro_value_iface_t *
avro_generic_class_from_schema(avro_schema_t schema);

/**
 * Allocate a new instance of the given generic value class.  @a iface
 * must have been created by @ref avro_generic_class_from_schema.
 */

int
avro_generic_value_new(avro_value_iface_t *iface, avro_value_t *dest);


/*
 * These functions return an avro_value_iface_t implementation for each
 * primitive schema type.  (For enum, fixed, and the compound types, you
 * must use the @ref avro_generic_class_from_schema function.)
 */

avro_value_iface_t *avro_generic_boolean_class(void);
avro_value_iface_t *avro_generic_bytes_class(void);
avro_value_iface_t *avro_generic_double_class(void);
avro_value_iface_t *avro_generic_float_class(void);
avro_value_iface_t *avro_generic_int_class(void);
avro_value_iface_t *avro_generic_long_class(void);
avro_value_iface_t *avro_generic_null_class(void);
avro_value_iface_t *avro_generic_string_class(void);


/*
 * These functions instantiate a new generic primitive value.
 */

int avro_generic_boolean_new(avro_value_t *value, int val);
int avro_generic_bytes_new(avro_value_t *value, void *buf, size_t size);
int avro_generic_double_new(avro_value_t *value, double val);
int avro_generic_float_new(avro_value_t *value, float val);
int avro_generic_int_new(avro_value_t *value, int32_t val);
int avro_generic_long_new(avro_value_t *value, int64_t val);
int avro_generic_null_new(avro_value_t *value);
int avro_generic_string_new(avro_value_t *value, const char *val);
int avro_generic_string_new_length(avro_value_t *value, const char *val, size_t size);


CLOSE_EXTERN
#endif
