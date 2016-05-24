### `tf.get_collection_ref(key)` {#get_collection_ref}

Wrapper for `Graph.get_collection_ref()` using the default graph.

See [`Graph.get_collection_ref()`](../../api_docs/python/framework.md#Graph.get_collection_ref)
for more details.

##### Args:


*  <b>`key`</b>: The key for the collection. For example, the `GraphKeys` class
    contains many standard names for collections.

##### Returns:

  The list of values in the collection with the given `name`, or an empty
  list if no value has been added to that collection.  Note that this returns
  the collection list itself, which can be modified in place to change the
  collection.

