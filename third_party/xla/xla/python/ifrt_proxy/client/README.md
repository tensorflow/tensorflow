This directory implements the IFRT proxy client.

## Expected behavior when connection to the IFRT proxy server fails

If a connection to the proxy server fails abruptly, any in-progress or further
IFRT API calls and `Future`s are expected to either return valid values (if the
value was already fetched from the server and is being cached locally) or an
error from `rpc_helper.cc`'s `WrapAsConnectionError()`. They are expected to
neither "hang" beyond the brief period required to determine whether the
connection has failed nor crash the process internally within the proxy client
library.
