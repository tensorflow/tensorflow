### `tf.get_default_session()` {#get_default_session}

Returns the default session for the current thread.

The returned `Session` will be the innermost session on which a
`Session` or `Session.as_default()` context has been entered.

NOTE: The default session is a property of the current thread. If you
create a new thread, and wish to use the default session in that
thread, you must explicitly add a `with sess.as_default():` in that
thread's function.

##### Returns:

  The default `Session` being used in the current thread.

