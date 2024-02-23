# Fuzzing

Fuzzing support using [go-fuzz](https://github.com/dvyukov/go-fuzz).

Basic operation:

```sh
$ go install github.com/dvyukov/go-fuzz/go-fuzz
$ go install github.com/mdempsky/go114-fuzz-build
$ cd internal/fuzz/{fuzzer}
$ go114-fuzz-build google.golang.org/protobuf/internal/fuzz/{fuzzer}
$ go-fuzz
```

## OSS-Fuzz

Fuzzers are automatically run by
[OSS-Fuzz](https://github.com/google/oss-fuzz).

The OSS-Fuzz
[configuration](https://github.com/google/oss-fuzz/blob/master/projects/golang-protobuf/build.sh)
currently builds fuzzers in every directory under internal/fuzz.
Only add fuzzers (not support packages) in this directory.

Fuzzing results are available at the [OSS-Fuzz console](https://oss-fuzz.com/),
under `golang-protobuf`.
