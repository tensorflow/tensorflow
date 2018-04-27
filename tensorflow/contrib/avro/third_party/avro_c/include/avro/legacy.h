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

#ifndef AVRO_LEGACY_H
#define AVRO_LEGACY_H
#ifdef __cplusplus
extern "C" {
#define CLOSE_EXTERN }
#else
#define CLOSE_EXTERN
#endif

#include <avro/platform.h>
#include <stdio.h>

#include <avro/basics.h>
#include <avro/data.h>
#include <avro/schema.h>
#include <avro/value.h>

/*
 * This file defines the deprecated interface for handling Avro values.
 * It's here solely for backwards compatibility.  New code should use
 * the avro_value_t interface (defined in avro/value.h).  The
 * avro_datum_t type has been replaced by the “generic” implementation
 * of the value interface, which is defined in avro/generic.h.  You can
 * also use your own application-specific types as Avro values by
 * defining your own avro_value_t implementation for them.
 */

/**
 * A function used to free a bytes, string, or fixed buffer once it is
 * no longer needed by the datum that wraps it.
 */

typedef void
(*avro_free_func_t)(void *ptr, size_t sz);

/**
 * An avro_free_func_t that frees the buffer using the custom allocator
 * provided to avro_set_allocator.
 */

void
avro_alloc_free_func(void *ptr, size_t sz);

/*
 * Datum constructors.  Each datum stores a reference to the schema that
 * the datum is an instance of.  The primitive datum constructors don't
 * need to take in an explicit avro_schema_t parameter, since there's
 * only one schema that they could be an instance of.  The complex
 * constructors do need an explicit schema parameter.
 */

typedef struct avro_obj_t *avro_datum_t;
avro_datum_t avro_string(const char *str);
avro_datum_t avro_givestring(const char *str,
			     avro_free_func_t free);
avro_datum_t avro_bytes(const char *buf, int64_t len);
avro_datum_t avro_givebytes(const char *buf, int64_t len,
			    avro_free_func_t free);
avro_datum_t avro_int32(int32_t i);
avro_datum_t avro_int64(int64_t l);
avro_datum_t avro_float(float f);
avro_datum_t avro_double(double d);
avro_datum_t avro_boolean(int8_t i);
avro_datum_t avro_null(void);
avro_datum_t avro_record(avro_schema_t schema);
avro_datum_t avro_enum(avro_schema_t schema, int i);
avro_datum_t avro_fixed(avro_schema_t schema,
			const char *bytes, const int64_t size);
avro_datum_t avro_givefixed(avro_schema_t schema,
			    const char *bytes, const int64_t size,
			    avro_free_func_t free);
avro_datum_t avro_map(avro_schema_t schema);
avro_datum_t avro_array(avro_schema_t schema);
avro_datum_t avro_union(avro_schema_t schema,
			int64_t discriminant, const avro_datum_t datum);

/**
 * Returns the schema that the datum is an instance of.
 */

avro_schema_t avro_datum_get_schema(const avro_datum_t datum);

/*
 * Constructs a new avro_datum_t instance that's appropriate for holding
 * values of the given schema.
 */

avro_datum_t avro_datum_from_schema(const avro_schema_t schema);

/* getters */
int avro_string_get(avro_datum_t datum, char **p);
int avro_bytes_get(avro_datum_t datum, char **bytes, int64_t * size);
int avro_int32_get(avro_datum_t datum, int32_t * i);
int avro_int64_get(avro_datum_t datum, int64_t * l);
int avro_float_get(avro_datum_t datum, float *f);
int avro_double_get(avro_datum_t datum, double *d);
int avro_boolean_get(avro_datum_t datum, int8_t * i);

int avro_enum_get(const avro_datum_t datum);
const char *avro_enum_get_name(const avro_datum_t datum);
int avro_fixed_get(avro_datum_t datum, char **bytes, int64_t * size);
int avro_record_get(const avro_datum_t record, const char *field_name,
		    avro_datum_t * value);

/*
 * A helper macro that extracts the value of the given field of a
 * record.
 */

#define avro_record_get_field_value(rc, rec, typ, fname, ...)	\
	do {							\
		avro_datum_t  field = NULL;			\
		(rc) = avro_record_get((rec), (fname), &field);	\
		if (rc) break;					\
		(rc) = avro_##typ##_get(field, __VA_ARGS__);	\
	} while (0)


int avro_map_get(const avro_datum_t datum, const char *key,
		 avro_datum_t * value);
/*
 * For maps, the "index" for each entry is based on the order that they
 * were added to the map.
 */
int avro_map_get_key(const avro_datum_t datum, int index,
		     const char **key);
int avro_map_get_index(const avro_datum_t datum, const char *key,
		       int *index);
size_t avro_map_size(const avro_datum_t datum);
int avro_array_get(const avro_datum_t datum, int64_t index, avro_datum_t * value);
size_t avro_array_size(const avro_datum_t datum);

/*
 * These accessors allow you to query the current branch of a union
 * value, returning either the branch's discriminant value or the
 * avro_datum_t of the branch.  A union value can be uninitialized, in
 * which case the discriminant will be -1 and the datum NULL.
 */

int64_t avro_union_discriminant(const avro_datum_t datum);
avro_datum_t avro_union_current_branch(avro_datum_t datum);

/* setters */
int avro_string_set(avro_datum_t datum, const char *p);
int avro_givestring_set(avro_datum_t datum, const char *p,
			avro_free_func_t free);

int avro_bytes_set(avro_datum_t datum, const char *bytes, const int64_t size);
int avro_givebytes_set(avro_datum_t datum, const char *bytes,
		       const int64_t size,
		       avro_free_func_t free);

int avro_int32_set(avro_datum_t datum, const int32_t i);
int avro_int64_set(avro_datum_t datum, const int64_t l);
int avro_float_set(avro_datum_t datum, const float f);
int avro_double_set(avro_datum_t datum, const double d);
int avro_boolean_set(avro_datum_t datum, const int8_t i);

int avro_enum_set(avro_datum_t datum, const int symbol_value);
int avro_enum_set_name(avro_datum_t datum, const char *symbol_name);
int avro_fixed_set(avro_datum_t datum, const char *bytes, const int64_t size);
int avro_givefixed_set(avro_datum_t datum, const char *bytes,
		       const int64_t size,
		       avro_free_func_t free);

int avro_record_set(avro_datum_t record, const char *field_name,
		    avro_datum_t value);

/*
 * A helper macro that sets the value of the given field of a record.
 */

#define avro_record_set_field_value(rc, rec, typ, fname, ...)	\
	do {							\
		avro_datum_t  field = NULL;			\
		(rc) = avro_record_get((rec), (fname), &field);	\
		if (rc) break;					\
		(rc) = avro_##typ##_set(field, __VA_ARGS__);	\
	} while (0)

int avro_map_set(avro_datum_t map, const char *key,
		 avro_datum_t value);
int avro_array_append_datum(avro_datum_t array_datum,
			    avro_datum_t datum);

/*
 * This function selects the active branch of a union value, and can be
 * safely called on an existing union to change the current branch.  If
 * the branch changes, we'll automatically construct a new avro_datum_t
 * for the new branch's schema type.  If the desired branch is already
 * the active branch of the union, we'll leave the existing datum
 * instance as-is.  The branch datum will be placed into the "branch"
 * parameter, regardless of whether we have to create a new datum
 * instance or not.
 */

int avro_union_set_discriminant(avro_datum_t unionp,
				int discriminant,
				avro_datum_t *branch);

/**
 * Resets a datum instance.  For arrays and maps, this frees all
 * elements and clears the container.  For records and unions, this
 * recursively resets any child datum instances.
 */

int
avro_datum_reset(avro_datum_t value);

/* reference counting */
avro_datum_t avro_datum_incref(avro_datum_t value);
void avro_datum_decref(avro_datum_t value);

void avro_datum_print(avro_datum_t value, FILE * fp);

int avro_datum_equal(avro_datum_t a, avro_datum_t b);

/*
 * Returns a string containing the JSON encoding of an Avro value.  You
 * must free this string when you're done with it, using the standard
 * free() function.  (*Not* using the custom Avro allocator.)
 */

int avro_datum_to_json(const avro_datum_t datum,
		       int one_line, char **json_str);


int avro_schema_datum_validate(avro_schema_t
			       expected_schema, avro_datum_t datum);

/*
 * An avro_value_t implementation for avro_datum_t objects.
 */

avro_value_iface_t *
avro_datum_class(void);

/*
 * Creates a new avro_value_t instance for the given datum.
 */

int
avro_datum_as_value(avro_value_t *value, avro_datum_t src);


CLOSE_EXTERN
#endif
