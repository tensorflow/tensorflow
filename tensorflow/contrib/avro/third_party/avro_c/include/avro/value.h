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

#ifndef AVRO_VALUE_H
#define AVRO_VALUE_H
#ifdef __cplusplus
extern "C" {
#define CLOSE_EXTERN }
#else
#define CLOSE_EXTERN
#endif

#include <errno.h>
#include <avro/platform.h>
#include <stdlib.h>

#include <avro/data.h>
#include <avro/schema.h>

/*
 * This file defines an interface struct for Avro data.  Most of the
 * interesting parts of this library will work with Avro data values
 * that are expressed in whatever C type you want, as long as you can
 * provide an implementation of this interface for that type.
 */

typedef struct avro_value_iface  avro_value_iface_t;

typedef struct avro_value {
	avro_value_iface_t  *iface;
	void  *self;
} avro_value_t;

struct avro_value_iface {
	/*-------------------------------------------------------------
	 * "class" methods
	 */

	/**
	 * Increment the reference count of the interface struct.  This
	 * should be a no-op for static structs, since they don't need
	 * reference counts.
	 */
	avro_value_iface_t *
	(*incref_iface)(avro_value_iface_t *iface);

	/**
	 * Decrement the reference count of the interface struct.  If
	 * the count falls to 0, free the struct.  This should be a
	 * no-op for static structs, since they don't need reference
	 * counts.
	 */
	void
	(*decref_iface)(avro_value_iface_t *iface);

	/*-------------------------------------------------------------
	 * General "instance" methods
	 */

	/**
	 * Increments the reference count of a value.
	 */

	void
	(*incref)(avro_value_t *value);

	/**
	 * Decrements the reference count of a value, and frees the
	 * value if the reference count drops to 0.  After calling this
	 * method, your value instance is undefined, and cannot be used
	 * anymore.
	 */

	void
	(*decref)(avro_value_t *value);

	/**
	 * Reset the instance to its "empty", default value.  You don't
	 * have to free the underlying storage, if you want to keep it
	 * around for later values.
	 */
	int
	(*reset)(const avro_value_iface_t *iface, void *self);

	/**
	 * Return the general Avro type of a value instance.
	 */
	avro_type_t
	(*get_type)(const avro_value_iface_t *iface, const void *self);

	/**
	 * Return the Avro schema that a value is an instance of.
	 */
	avro_schema_t
	(*get_schema)(const avro_value_iface_t *iface, const void *self);

	/*-------------------------------------------------------------
	 * Primitive value getters
	 */
	int (*get_boolean)(const avro_value_iface_t *iface,
			   const void *self, int *out);
	int (*get_bytes)(const avro_value_iface_t *iface,
			 const void *self, const void **buf, size_t *size);
	int (*grab_bytes)(const avro_value_iface_t *iface,
			  const void *self, avro_wrapped_buffer_t *dest);
	int (*get_double)(const avro_value_iface_t *iface,
			  const void *self, double *out);
	int (*get_float)(const avro_value_iface_t *iface,
			 const void *self, float *out);
	int (*get_int)(const avro_value_iface_t *iface,
		       const void *self, int32_t *out);
	int (*get_long)(const avro_value_iface_t *iface,
			const void *self, int64_t *out);
	int (*get_null)(const avro_value_iface_t *iface,
			const void *self);
	/* The result will be NUL-terminated; the size will INCLUDE the
	 * NUL terminator.  str will never be NULL unless there's an
	 * error. */
	int (*get_string)(const avro_value_iface_t *iface,
			  const void *self, const char **str, size_t *size);
	int (*grab_string)(const avro_value_iface_t *iface,
			   const void *self, avro_wrapped_buffer_t *dest);

	int (*get_enum)(const avro_value_iface_t *iface,
			const void *self, int *out);
	int (*get_fixed)(const avro_value_iface_t *iface,
			 const void *self, const void **buf, size_t *size);
	int (*grab_fixed)(const avro_value_iface_t *iface,
			  const void *self, avro_wrapped_buffer_t *dest);

	/*-------------------------------------------------------------
	 * Primitive value setters
	 */

	/*
	 * The "give" setters can be used to give control of an existing
	 * buffer to a bytes, fixed, or string value.  The free function
	 * will be called when the buffer is no longer needed.  (It's
	 * okay for free to be NULL; that just means that nothing
	 * special needs to be done to free the buffer.  That's useful
	 * for a static string, for instance.)
	 *
	 * If your class can't take control of an existing buffer, then
	 * your give functions should pass the buffer into the
	 * corresponding "set" method and then immediately free the
	 * buffer.
	 *
	 * Note that for strings, the free function will be called with
	 * a size that *includes* the NUL terminator, even though you
	 * provide a size that does *not*.
	 */

	int (*set_boolean)(const avro_value_iface_t *iface,
			   void *self, int val);
	int (*set_bytes)(const avro_value_iface_t *iface,
			 void *self, void *buf, size_t size);
	int (*give_bytes)(const avro_value_iface_t *iface,
			  void *self, avro_wrapped_buffer_t *buf);
	int (*set_double)(const avro_value_iface_t *iface,
			  void *self, double val);
	int (*set_float)(const avro_value_iface_t *iface,
			 void *self, float val);
	int (*set_int)(const avro_value_iface_t *iface,
		       void *self, int32_t val);
	int (*set_long)(const avro_value_iface_t *iface,
			void *self, int64_t val);
	int (*set_null)(const avro_value_iface_t *iface, void *self);
	/* The input must be NUL-terminated */
	int (*set_string)(const avro_value_iface_t *iface,
			  void *self, const char *str);
	/* and size must INCLUDE the NUL terminator */
	int (*set_string_len)(const avro_value_iface_t *iface,
			      void *self, const char *str, size_t size);
	int (*give_string_len)(const avro_value_iface_t *iface,
			       void *self, avro_wrapped_buffer_t *buf);

	int (*set_enum)(const avro_value_iface_t *iface,
			void *self, int val);
	int (*set_fixed)(const avro_value_iface_t *iface,
			 void *self, void *buf, size_t size);
	int (*give_fixed)(const avro_value_iface_t *iface,
			  void *self, avro_wrapped_buffer_t *buf);

	/*-------------------------------------------------------------
	 * Compound value getters
	 */

	/* Number of elements in array/map, or the number of fields in a
	 * record. */
	int (*get_size)(const avro_value_iface_t *iface,
			const void *self, size_t *size);

	/*
	 * For arrays and maps, returns the element with the given
	 * index.  (For maps, the "index" is based on the order that the
	 * keys were added to the map.)  For records, returns the field
	 * with that index in the schema.
	 *
	 * For maps and records, the name parameter (if given) will be
	 * filled in with the key or field name of the returned value.
	 * For arrays, the name parameter will always be ignored.
	 */
	int (*get_by_index)(const avro_value_iface_t *iface,
			    const void *self, size_t index,
			    avro_value_t *child, const char **name);

	/*
	 * For maps, returns the element with the given key.  For
	 * records, returns the element with the given field name.  If
	 * index is given, it will be filled in with the numeric index
	 * of the returned value.
	 */
	int (*get_by_name)(const avro_value_iface_t *iface,
			   const void *self, const char *name,
			   avro_value_t *child, size_t *index);

	/* Discriminant of current union value */
	int (*get_discriminant)(const avro_value_iface_t *iface,
				const void *self, int *out);
	/* Current union value */
	int (*get_current_branch)(const avro_value_iface_t *iface,
				  const void *self, avro_value_t *branch);

	/*-------------------------------------------------------------
	 * Compound value setters
	 */

	/*
	 * For all of these, the value class should know which class to
	 * use for its children.
	 */

	/* Creates a new array element. */
	int (*append)(const avro_value_iface_t *iface,
		      void *self, avro_value_t *child_out, size_t *new_index);

	/* Creates a new map element, or returns an existing one. */
	int (*add)(const avro_value_iface_t *iface,
		   void *self, const char *key,
		   avro_value_t *child, size_t *index, int *is_new);

	/* Select a union branch. */
	int (*set_branch)(const avro_value_iface_t *iface,
			  void *self, int discriminant,
			  avro_value_t *branch);
};


/**
 * Increments the reference count of a value instance.  Normally you
 * don't need to call this directly; you'll have a reference whenever
 * you create the value, and @ref avro_value_copy and @ref
 * avro_value_move update the reference counts correctly for you.
 */

void
avro_value_incref(avro_value_t *value);

/**
 * Decremenets the reference count of a value instance, freeing it if
 * its reference count drops to 0.
 */

void
avro_value_decref(avro_value_t *value);

/**
 * Copies a reference to a value.  This does not copy any of the data
 * in the value; you get two avro_value_t references that point at the
 * same underlying value instance.
 */

void
avro_value_copy_ref(avro_value_t *dest, const avro_value_t *src);

/**
 * Moves a reference to a value.  This does not copy any of the data in
 * the value.  The @ref src value is invalidated by this function; its
 * equivalent to the following:
 *
 * <code>
 * avro_value_copy_ref(dest, src);
 * avro_value_decref(src);
 * </code>
 */

void
avro_value_move_ref(avro_value_t *dest, avro_value_t *src);

/**
 * Compares two values for equality.  The two values don't need to have
 * the same implementation of the value interface, but they do need to
 * represent Avro values of the same schema.  This function ensures that
 * the schemas match; if you want to skip this check, use
 * avro_value_equal_fast.
 */

int
avro_value_equal(avro_value_t *val1, avro_value_t *val2);

/**
 * Compares two values for equality.  The two values don't need to have
 * the same implementation of the value interface, but they do need to
 * represent Avro values of the same schema.  This function assumes that
 * the schemas match; if you can't guarantee this, you should use
 * avro_value_equal, which compares the schemas before comparing the
 * values.
 */

int
avro_value_equal_fast(avro_value_t *val1, avro_value_t *val2);

/**
 * Compares two values using the sort order defined in the Avro
 * specification.  The two values don't need to have the same
 * implementation of the value interface, but they do need to represent
 * Avro values of the same schema.  This function ensures that the
 * schemas match; if you want to skip this check, use
 * avro_value_cmp_fast.
 */

int
avro_value_cmp(avro_value_t *val1, avro_value_t *val2);

/**
 * Compares two values using the sort order defined in the Avro
 * specification.  The two values don't need to have the same
 * implementation of the value interface, but they do need to represent
 * Avro values of the same schema.  This function assumes that the
 * schemas match; if you can't guarantee this, you should use
 * avro_value_cmp, which compares the schemas before comparing the
 * values.
 */

int
avro_value_cmp_fast(avro_value_t *val1, avro_value_t *val2);



/**
 * Copies the contents of src into dest.  The two values don't need to
 * have the same implementation of the value interface, but they do need
 * to represent Avro values of the same schema.  This function ensures
 * that the schemas match; if you want to skip this check, use
 * avro_value_copy_fast.
 */

int
avro_value_copy(avro_value_t *dest, const avro_value_t *src);

/**
 * Copies the contents of src into dest.  The two values don't need to
 * have the same implementation of the value interface, but they do need
 * to represent Avro values of the same schema.  This function assumes
 * that the schemas match; if you can't guarantee this, you should use
 * avro_value_copy, which compares the schemas before comparing the
 * values.
 */

int
avro_value_copy_fast(avro_value_t *dest, const avro_value_t *src);

/**
 * Returns a hash value for a given Avro value.
 */

uint32_t
avro_value_hash(avro_value_t *value);

/*
 * Returns a string containing the JSON encoding of an Avro value.  You
 * must free this string when you're done with it, using the standard
 * free() function.  (*Not* using the custom Avro allocator.)
 */

int
avro_value_to_json(const avro_value_t *value,
		   int one_line, char **json_str);


/**
 * A helper macro for calling a given method in a value instance, if
 * it's present.  If the value's class doesn't implement the given
 * method, we return dflt.  You usually won't call this directly; it's
 * just here to implement the macros below.
 */

#define avro_value_call0(value, method, dflt) \
    ((value)->iface->method == NULL? (dflt): \
     (value)->iface->method((value)->iface, (value)->self))

#define avro_value_call(value, method, dflt, ...) \
    ((value)->iface->method == NULL? (dflt): \
     (value)->iface->method((value)->iface, (value)->self, __VA_ARGS__))


#define avro_value_iface_incref(cls) \
    ((cls)->incref_iface == NULL? (cls): (cls)->incref_iface((cls)))
#define avro_value_iface_decref(cls) \
    ((cls)->decref_iface == NULL? (void) 0: (cls)->decref_iface((cls)))

#define avro_value_reset(value) \
    avro_value_call0(value, reset, EINVAL)
#define avro_value_get_type(value) \
    avro_value_call0(value, get_type, (avro_type_t) -1)
#define avro_value_get_schema(value) \
    avro_value_call0(value, get_schema, NULL)

#define avro_value_get_boolean(value, out) \
    avro_value_call(value, get_boolean, EINVAL, out)
#define avro_value_get_bytes(value, buf, size) \
    avro_value_call(value, get_bytes, EINVAL, buf, size)
#define avro_value_grab_bytes(value, dest) \
    avro_value_call(value, grab_bytes, EINVAL, dest)
#define avro_value_get_double(value, out) \
    avro_value_call(value, get_double, EINVAL, out)
#define avro_value_get_float(value, out) \
    avro_value_call(value, get_float, EINVAL, out)
#define avro_value_get_int(value, out) \
    avro_value_call(value, get_int, EINVAL, out)
#define avro_value_get_long(value, out) \
    avro_value_call(value, get_long, EINVAL, out)
#define avro_value_get_null(value) \
    avro_value_call0(value, get_null, EINVAL)
#define avro_value_get_string(value, str, size) \
    avro_value_call(value, get_string, EINVAL, str, size)
#define avro_value_grab_string(value, dest) \
    avro_value_call(value, grab_string, EINVAL, dest)
#define avro_value_get_enum(value, out) \
    avro_value_call(value, get_enum, EINVAL, out)
#define avro_value_get_fixed(value, buf, size) \
    avro_value_call(value, get_fixed, EINVAL, buf, size)
#define avro_value_grab_fixed(value, dest) \
    avro_value_call(value, grab_fixed, EINVAL, dest)

#define avro_value_set_boolean(value, val) \
    avro_value_call(value, set_boolean, EINVAL, val)
#define avro_value_set_bytes(value, buf, size) \
    avro_value_call(value, set_bytes, EINVAL, buf, size)
#define avro_value_give_bytes(value, buf) \
    avro_value_call(value, give_bytes, EINVAL, buf)
#define avro_value_set_double(value, val) \
    avro_value_call(value, set_double, EINVAL, val)
#define avro_value_set_float(value, val) \
    avro_value_call(value, set_float, EINVAL, val)
#define avro_value_set_int(value, val) \
    avro_value_call(value, set_int, EINVAL, val)
#define avro_value_set_long(value, val) \
    avro_value_call(value, set_long, EINVAL, val)
#define avro_value_set_null(value) \
    avro_value_call0(value, set_null, EINVAL)
#define avro_value_set_string(value, str) \
    avro_value_call(value, set_string, EINVAL, str)
#define avro_value_set_string_len(value, str, size) \
    avro_value_call(value, set_string_len, EINVAL, str, size)
#define avro_value_give_string_len(value, buf) \
    avro_value_call(value, give_string_len, EINVAL, buf)
#define avro_value_set_enum(value, val) \
    avro_value_call(value, set_enum, EINVAL, val)
#define avro_value_set_fixed(value, buf, size) \
    avro_value_call(value, set_fixed, EINVAL, buf, size)
#define avro_value_give_fixed(value, buf) \
    avro_value_call(value, give_fixed, EINVAL, buf)

#define avro_value_get_size(value, size) \
    avro_value_call(value, get_size, EINVAL, size)
#define avro_value_get_by_index(value, idx, child, name) \
    avro_value_call(value, get_by_index, EINVAL, idx, child, name)
#define avro_value_get_by_name(value, name, child, index) \
    avro_value_call(value, get_by_name, EINVAL, name, child, index)
#define avro_value_get_discriminant(value, out) \
    avro_value_call(value, get_discriminant, EINVAL, out)
#define avro_value_get_current_branch(value, branch) \
    avro_value_call(value, get_current_branch, EINVAL, branch)

#define avro_value_append(value, child, new_index) \
    avro_value_call(value, append, EINVAL, child, new_index)
#define avro_value_add(value, key, child, index, is_new) \
    avro_value_call(value, add, EINVAL, key, child, index, is_new)
#define avro_value_set_branch(value, discriminant, branch) \
    avro_value_call(value, set_branch, EINVAL, discriminant, branch)

CLOSE_EXTERN
#endif
