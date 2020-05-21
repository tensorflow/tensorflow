# Summary Changes
| Old Implementation  | New Implementation  |
|---|---|
| auth_provider | google-cloud-cpp |
| compute_engine_metadata_client | google-cloud-cpp |
| compute_engine_zone_provider | google-cloud-cpp | 
| curl_http_request | google-cloud-cpp |
| expiring_lru_cache | Mimic the old implementation |
| file_block_cache | Mimic the old implementation |
| gcs_dsn_cache | Mimic the old implementation |
| gcs_file_system | Mimic the old implementation |
| gcs_throttle | Mimic the old implementation |
| google_auth_provider | google-cloud-cpp |
| http_request | google-cloud-cpp |
| now_seconds_env | For testing purpose only |
| oauth_client | google-cloud-cpp |
| ram_file_block_cache | Mimic the old implementation |
| time_util | google-cloud-cpp |
| zone_provider | google-cloud-cpp |

# Detail 
## Replaced by google-cloud-cpp
### auth_provider & google_auth_provider & oauth_client
- Interface for a provider of authentication bearer tokens.
- We dont use it because google-cloud-cpp has already had an [auth_provider](https://github.com/googleapis/google-cloud-cpp/tree/ef8b727da2f8c5fc49bc4c7ccaaefe862b41d367/google/cloud/storage/oauth2).
### compute_engine_metadata_client & zone_provider
- A client that accesses to the metadata server running on GCE hosts.
- `google-cloud-cpp` has a private member [function](https://github.com/googleapis/google-cloud-cpp/blob/ef8b727da2f8c5fc49bc4c7ccaaefe862b41d367/google/cloud/storage/oauth2/compute_engine_credentials.h#L132) to get metadata from GCE server. We could add a patch file to make this function public or mimic this function.

### compute_engine_zone_provider
- Use `compute_engine_metadata_client` to get the metadata `instance/zone`

### curl_http_request & http_request
- A basic HTTP client based on the libcurl library.
- We dont use it because google-cloud-cpp has already had a [curl_request](https://github.com/googleapis/google-cloud-cpp/blob/ef8b727da2f8c5fc49bc4c7ccaaefe862b41d367/google/cloud/storage/internal/curl_request.h).

### time-util
- Parses the timestamp in RFC 3339 format
- google-cloud-cpp already has a [parser](https://github.com/googleapis/google-cloud-cpp/blob/6fcdc362757d7743e4edf34143e5ac4eaa6b5a85/google/cloud/internal/parse_rfc3339.h)

## Mimic Implementation
### expiring_lru_cache
An LRU cache of string keys and arbitrary values, with configurable max item age (in seconds) and max entries.
- Usage: It is used for caching matching paths, bucket locations and object metadata ( to check if the object on cloud server has been overwritten or not )
- Deps: env, mutex
- Solution: `env` is used only for getting time. We could pass a callable object that return time instead of `env`. With the mutex, we may have to use `std::mutex` and `std::shared_mutex`

### file_block_cache & ram_file_block_cache
An LRU block cache of file contents, keyed by {filename, offset}.
- Usage: 
  + It is used for caching content of objects fetched from GCS server. ( the cloud server in general )
  + It is used for `RandomAccessFile`. When we `Read` a range of bytes from an object on cloud server, it caches the content with a key `filename@offset`
  + When we call `Read` again on that object, first it sends a request to the server to check if the metadata has been changed or not ( by using `expiring_lru_cache<objectmetadata>` ). Next, it calculates how many bytes has been cached. Lastly, it fetches only the missing bytes from server.
  ```cpp
  // We want to read from 3 -> 5
  Read() 
  // fetches and save to a cached file named fname@3 with size 2

  // Next we read from 1 -> 4
  Read() 
  // fetches from 1 -> 3 and save to a cached file name fname@1 with size 2
  // Next, it copies the content of fname@1 and fname@3 ( from 3 -> 4 ) to the result
  ```
- Deps: env, mutex
- Solution: `env` is used only for getting time. We could pass a callable object that return time instead of `env`. With the mutex, we may have to use `std::mutex` and `std::shared_mutex`
- Note:
  + We may want to rewrite `ram_file_block_cache` only, `file_block_cache` is just an interface.
  + We may want to add `ram_file_block_cache` to a seperate folder `plugins/ram_cache` so any filesystem needs to use cache can use this ( `s3` )
  + `Status` will be converted to `TF_Status`. Some functions maybe return  `void` and set the `TF_Status*` instead of return `Status`

### gcs_dsn_cache
DnsCache is a userspace DNS cache specialized for the GCS filesystem

### gcs_throttle
GcsThrottle is used to ensure fair use of the available GCS capacity
- Usage: GcsThrottle operates around a concept of tokens. Tokens are consumed when making requests to the GCS service. Tokens are consumed both based on the number of requests made, as well as the bandwidth consumed (response sizes).
- Note: We don't have much controls about throttle when working with `google-cloud-cpp`. So there are 3 possible solutions:
  + one call to `gcs::Client` is count as one request
  + let `gcs::Client` controls the throttle
  + fork `google-cloud-cpp` and add some controls for throttle

### gcs_file_system
GcsStatsInterface
- An object to collect runtime statistics from the GcsFilesystem
- We may want to simply save those stats to `filesystem->plugin_filesystem`
- I can't find any implementation of this interface. Maybe we could drop this feature ?

TimeoutConfig
- Structure containing the information for timeouts related
-  We don't have much controls about times out so we have to fork `google-cloud-cpp` if we want to use this.

RandomAccessFile
- If caching is on, we use file_block_cache
- If caching is off, we read directly to buffer and return to Tensorflow

WritableFile
- Writing to a temporary file and upload that file to server ( already implemented in Modular plugins )

Directory related operation
- If the directory is a bucket, we will mimic the current implementation
- If the directory is an object, I think we could set the `status` to `TF_OK`, do nothing else and return. The current implementation append to the object's name a `/` and treated it like an empty object. It is useless.