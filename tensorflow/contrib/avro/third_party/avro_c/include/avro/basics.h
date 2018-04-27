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

#ifndef AVRO_BASICS_H
#define AVRO_BASICS_H
#ifdef __cplusplus
extern "C" {
#define CLOSE_EXTERN }
#else
#define CLOSE_EXTERN
#endif


enum avro_type_t {
	AVRO_STRING,
	AVRO_BYTES,
	AVRO_INT32,
	AVRO_INT64,
	AVRO_FLOAT,
	AVRO_DOUBLE,
	AVRO_BOOLEAN,
	AVRO_NULL,
	AVRO_RECORD,
	AVRO_ENUM,
	AVRO_FIXED,
	AVRO_MAP,
	AVRO_ARRAY,
	AVRO_UNION,
	AVRO_LINK
};
typedef enum avro_type_t avro_type_t;

enum avro_class_t {
	AVRO_SCHEMA,
	AVRO_DATUM
};
typedef enum avro_class_t avro_class_t;

struct avro_obj_t {
	avro_type_t type;
	avro_class_t class_type;
	volatile int  refcount;
};

#define avro_classof(obj)     ((obj)->class_type)
#define is_avro_schema(obj)   (obj && avro_classof(obj) == AVRO_SCHEMA)
#define is_avro_datum(obj)    (obj && avro_classof(obj) == AVRO_DATUM)

#define avro_typeof(obj)      ((obj)->type)
#define is_avro_string(obj)   (obj && avro_typeof(obj) == AVRO_STRING)
#define is_avro_bytes(obj)    (obj && avro_typeof(obj) == AVRO_BYTES)
#define is_avro_int32(obj)    (obj && avro_typeof(obj) == AVRO_INT32)
#define is_avro_int64(obj)    (obj && avro_typeof(obj) == AVRO_INT64)
#define is_avro_float(obj)    (obj && avro_typeof(obj) == AVRO_FLOAT)
#define is_avro_double(obj)   (obj && avro_typeof(obj) == AVRO_DOUBLE)
#define is_avro_boolean(obj)  (obj && avro_typeof(obj) == AVRO_BOOLEAN)
#define is_avro_null(obj)     (obj && avro_typeof(obj) == AVRO_NULL)
#define is_avro_primitive(obj)(is_avro_string(obj) \
                             ||is_avro_bytes(obj) \
                             ||is_avro_int32(obj) \
                             ||is_avro_int64(obj) \
                             ||is_avro_float(obj) \
                             ||is_avro_double(obj) \
                             ||is_avro_boolean(obj) \
                             ||is_avro_null(obj))
#define is_avro_record(obj)   (obj && avro_typeof(obj) == AVRO_RECORD)
#define is_avro_enum(obj)     (obj && avro_typeof(obj) == AVRO_ENUM)
#define is_avro_fixed(obj)    (obj && avro_typeof(obj) == AVRO_FIXED)
#define is_avro_named_type(obj)(is_avro_record(obj) \
                              ||is_avro_enum(obj) \
                              ||is_avro_fixed(obj))
#define is_avro_map(obj)      (obj && avro_typeof(obj) == AVRO_MAP)
#define is_avro_array(obj)    (obj && avro_typeof(obj) == AVRO_ARRAY)
#define is_avro_union(obj)    (obj && avro_typeof(obj) == AVRO_UNION)
#define is_avro_complex_type(obj) (!(is_avro_primitive(obj))
#define is_avro_link(obj)     (obj && avro_typeof(obj) == AVRO_LINK)



CLOSE_EXTERN
#endif
