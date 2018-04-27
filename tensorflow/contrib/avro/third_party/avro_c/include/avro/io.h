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

#ifndef AVRO_IO_H
#define AVRO_IO_H
#ifdef __cplusplus
extern "C" {
#define CLOSE_EXTERN }
#else
#define CLOSE_EXTERN
#endif

#include <avro/platform.h>
#include <stdio.h>

#include <avro/basics.h>
#include <avro/legacy.h>
#include <avro/schema.h>
#include <avro/value.h>

typedef struct avro_reader_t_ *avro_reader_t;
typedef struct avro_writer_t_ *avro_writer_t;

/*
 * io
 */

avro_reader_t avro_reader_file(FILE * fp);
avro_reader_t avro_reader_file_fp(FILE * fp, int should_close);
avro_writer_t avro_writer_file(FILE * fp);
avro_writer_t avro_writer_file_fp(FILE * fp, int should_close);
avro_reader_t avro_reader_memory(const char *buf, int64_t len);
avro_writer_t avro_writer_memory(const char *buf, int64_t len);

void
avro_reader_memory_set_source(avro_reader_t reader, const char *buf, int64_t len);

void
avro_writer_memory_set_dest(avro_writer_t writer, const char *buf, int64_t len);

int avro_read(avro_reader_t reader, void *buf, int64_t len);
int avro_skip(avro_reader_t reader, int64_t len);
int avro_write(avro_writer_t writer, void *buf, int64_t len);

void avro_reader_reset(avro_reader_t reader);

void avro_writer_reset(avro_writer_t writer);
int64_t avro_writer_tell(avro_writer_t writer);
void avro_writer_flush(avro_writer_t writer);

void avro_writer_dump(avro_writer_t writer, FILE * fp);
void avro_reader_dump(avro_reader_t reader, FILE * fp);

int avro_reader_is_eof(avro_reader_t reader);

void avro_reader_free(avro_reader_t reader);
void avro_writer_free(avro_writer_t writer);

int avro_schema_to_json(const avro_schema_t schema, avro_writer_t out);

/*
 * Reads a binary-encoded Avro value from the given reader object,
 * storing the result into dest.
 */

int
avro_value_read(avro_reader_t reader, avro_value_t *dest);

/*
 * Writes a binary-encoded Avro value to the given writer object.
 */

int
avro_value_write(avro_writer_t writer, avro_value_t *src);

/*
 * Returns the size of the binary encoding of the given Avro value.
 */

int
avro_value_sizeof(avro_value_t *src, size_t *size);


/* File object container */
typedef struct avro_file_reader_t_ *avro_file_reader_t;
typedef struct avro_file_writer_t_ *avro_file_writer_t;

int avro_file_writer_create(const char *path, avro_schema_t schema,
			    avro_file_writer_t * writer);
int avro_file_writer_create_fp(FILE *fp, const char *path, int should_close,
				avro_schema_t schema, avro_file_writer_t * writer);
int avro_file_writer_create_with_codec(const char *path,
				avro_schema_t schema, avro_file_writer_t * writer,
				const char *codec, size_t block_size);
int avro_file_writer_create_with_codec_fp(FILE *fp, const char *path, int should_close,
				avro_schema_t schema, avro_file_writer_t * writer,
				const char *codec, size_t block_size);
int avro_file_writer_open(const char *path, avro_file_writer_t * writer);
int avro_file_writer_open_bs(const char *path, avro_file_writer_t * writer, size_t block_size);
int avro_file_reader(const char *path, avro_file_reader_t * reader);
int avro_file_reader_fp(FILE *fp, const char *path, int should_close,
			avro_file_reader_t * reader);

avro_schema_t
avro_file_reader_get_writer_schema(avro_file_reader_t reader);

int avro_file_writer_sync(avro_file_writer_t writer);
int avro_file_writer_flush(avro_file_writer_t writer);
int avro_file_writer_close(avro_file_writer_t writer);

int avro_file_reader_close(avro_file_reader_t reader);

int
avro_file_reader_read_value(avro_file_reader_t reader, avro_value_t *dest);

int
avro_file_writer_append_value(avro_file_writer_t writer, avro_value_t *src);

int
avro_file_writer_append_encoded(avro_file_writer_t writer,
				const void *buf, int64_t len);

/*
 * Legacy avro_datum_t API
 */

int avro_read_data(avro_reader_t reader,
		   avro_schema_t writer_schema,
		   avro_schema_t reader_schema, avro_datum_t * datum);
int avro_skip_data(avro_reader_t reader, avro_schema_t writer_schema);
int avro_write_data(avro_writer_t writer,
		    avro_schema_t writer_schema, avro_datum_t datum);
int64_t avro_size_data(avro_writer_t writer,
		       avro_schema_t writer_schema, avro_datum_t datum);

int avro_file_writer_append(avro_file_writer_t writer, avro_datum_t datum);

int avro_file_reader_read(avro_file_reader_t reader,
			  avro_schema_t readers_schema, avro_datum_t * datum);

CLOSE_EXTERN
#endif
