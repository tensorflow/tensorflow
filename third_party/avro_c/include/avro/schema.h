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

#ifndef AVRO_SCHEMA_H
#define AVRO_SCHEMA_H
#ifdef __cplusplus
extern "C" {
#define CLOSE_EXTERN }
#else
#define CLOSE_EXTERN
#endif

#include <avro/platform.h>
#include <stdlib.h>

#include <avro/basics.h>

typedef struct avro_obj_t *avro_schema_t;

avro_schema_t avro_schema_string(void);
avro_schema_t avro_schema_bytes(void);
avro_schema_t avro_schema_int(void);
avro_schema_t avro_schema_long(void);
avro_schema_t avro_schema_float(void);
avro_schema_t avro_schema_double(void);
avro_schema_t avro_schema_boolean(void);
avro_schema_t avro_schema_null(void);

avro_schema_t avro_schema_record(const char *name, const char *space);
avro_schema_t avro_schema_record_field_get(const avro_schema_t
					   record, const char *field_name);
const char *avro_schema_record_field_name(const avro_schema_t schema, int index);
int avro_schema_record_field_get_index(const avro_schema_t schema,
				       const char *field_name);
avro_schema_t avro_schema_record_field_get_by_index
(const avro_schema_t record, int index);
int avro_schema_record_field_append(const avro_schema_t record,
				    const char *field_name,
				    const avro_schema_t type);
size_t avro_schema_record_size(const avro_schema_t record);

avro_schema_t avro_schema_enum(const char *name);
const char *avro_schema_enum_get(const avro_schema_t enump,
				 int index);
int avro_schema_enum_get_by_name(const avro_schema_t enump,
				 const char *symbol_name);
int avro_schema_enum_symbol_append(const avro_schema_t
				   enump, const char *symbol);

avro_schema_t avro_schema_fixed(const char *name, const int64_t len);
int64_t avro_schema_fixed_size(const avro_schema_t fixed);

avro_schema_t avro_schema_map(const avro_schema_t values);
avro_schema_t avro_schema_map_values(avro_schema_t map);

avro_schema_t avro_schema_array(const avro_schema_t items);
avro_schema_t avro_schema_array_items(avro_schema_t array);

avro_schema_t avro_schema_union(void);
size_t avro_schema_union_size(const avro_schema_t union_schema);
int avro_schema_union_append(const avro_schema_t
			     union_schema, const avro_schema_t schema);
avro_schema_t avro_schema_union_branch(avro_schema_t union_schema,
				       int branch_index);
avro_schema_t avro_schema_union_branch_by_name
(avro_schema_t union_schema, int *branch_index, const char *name);

avro_schema_t avro_schema_link(avro_schema_t schema);
avro_schema_t avro_schema_link_target(avro_schema_t schema);

typedef struct avro_schema_error_t_ *avro_schema_error_t;

int avro_schema_from_json(const char *jsontext, int32_t unused1,
			  avro_schema_t *schema, avro_schema_error_t *unused2);

/* jsontext does not need to be NUL terminated.  length must *NOT*
 * include the NUL terminator, if one is present. */
int avro_schema_from_json_length(const char *jsontext, size_t length,
				 avro_schema_t *schema);

/* A helper macro for loading a schema from a string literal.  The
 * literal must be declared as a char[], not a char *, since we use the
 * sizeof operator to determine its length. */
#define avro_schema_from_json_literal(json, schema) \
    (avro_schema_from_json_length((json), sizeof((json))-1, (schema)))

int avro_schema_to_specific(avro_schema_t schema, const char *prefix);

avro_schema_t avro_schema_get_subschema(const avro_schema_t schema,
         const char *name);
const char *avro_schema_name(const avro_schema_t schema);
const char *avro_schema_type_name(const avro_schema_t schema);
avro_schema_t avro_schema_copy(avro_schema_t schema);
int avro_schema_equal(avro_schema_t a, avro_schema_t b);

avro_schema_t avro_schema_incref(avro_schema_t schema);
int avro_schema_decref(avro_schema_t schema);

int avro_schema_match(avro_schema_t writers_schema,
		      avro_schema_t readers_schema);

CLOSE_EXTERN
#endif
