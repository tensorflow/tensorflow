# IFRT SerDes Version Management

IFRT SerDes (serialization/deserialization) uses a versioning mechanism to
evolve serialization formats. This allows for adding or removing features, using
improved data representations, and removing obsolete formats that are no longer
supported.

IFRT maintains a single global version that is incremented whenever a new
serialization format is introduced for any IFRT data type. During serialization,
this version determines the format to be used and is recorded in the serialized
data. During deserialization, the format version is read to select the
corresponding parsing logic.

The most recent version is available via `SerDesVersion::current()` and should
be used by default for most serialization tasks.

In some cases, data serialized by a newer version of IFRT needs to be read by an
older version. For these scenarios, an older serialization version must be used.
For example, `SerDesWeek4OldVersionAccessor::Get()` provides a version that is
at least four weeks old, ensuring that it can be deserialized by an IFRT build
from that time.

For more complex scenarios, such as IFRT Proxy where serialization and
deserialization occur between two different builds of IFRT, a common and highest
version can be negotiated between them and then selected using
`SerDesAnyVersionAccessor::Get()`.

## Adding a New Format

When defining a new serialization format for an IFRT type, follow these steps:

* Increment the version number and add it to the `SerDesVersion` list.
This entry should include the introduction date, which helps old version
selectors (e.g., `SerDesVersion::week_4_old()`) choose the correct version
number.
* As needed, update `SerDesVersion::week_4_old()` to point to a
version number that is recent but not more recent than 4 weeks from now.
* In `ToProto()` or `SerDes::Serialize()`, write the new serialization logic.
This logic is executed if the requested version is greater than or equal to the
new version number. The new version number is recorded in the serialized data.
Nested serialization calls should use the same requested version.
* In `FromProto()` or `SerDes::Deserialize()`, write the new deserialization
logic. This logic is chosen if the version in the serialized data matches the
new version number. Nested deserialization calls will resolve their versions
independently.

## Removing an Old Format

Removing logic for old formats helps maintain code health. This process may be
deferred so that multiple old formats are removed in batches at a convenient
time. To support symmetric serialization and deserialization in IFRT Proxy, it
is recommended to remove both serialization and deserialization logic at the
same time after the backward compatibility window has passed.

* Remove the serialization and deserialization logic for the old format.
* `SerDesVersion::week_4_old()` must be updated to point to a
version number newer than the one for the removed format (but not more recent
than 4 weeks ago from now). This ensures that new serializations do not use the
old format.
* `SerDesVersion::minimum()` must be updated to point to a
version number newer than the one for the removed format. This prevents
deserialization of the old format.
