# Adding a Custom Filesystem Plugin

## Background

The TensorFlow framework is often used in multi-process and
multi-machine environments, such as Google data centers, Google Cloud
Machine Learning, Amazon Web Services (AWS), and on-site distributed clusters.
In order to both share and save certain types of state produced by TensorFlow,
the framework assumes the existence of a reliable, shared filesystem. This
shared filesystem has numerous uses, for example:

*   Checkpoints of state are often saved to a distributed filesystem for
    reliability and fault-tolerance.
*   Training processes communicate with TensorBoard by writing event files
    to a directory, which TensorBoard watches. A shared filesystem allows this
    communication to work even when TensorBoard runs in a different process or
    machine.

There are many different implementations of shared or distributed filesystems in
the real world, so TensorFlow provides an ability for users to implement a
custom FileSystem plugin that can be registered with the TensorFlow runtime.
When the TensorFlow runtime attempts to write to a file through the `FileSystem`
interface, it uses a portion of the pathname to dynamically select the
implementation that should be used for filesystem operations. Thus, adding
support for your custom filesystem requires implementing a `FileSystem`
interface, building a shared object containing that implementation, and loading
that object at runtime in whichever process needs to write to that filesystem.

Note that TensorFlow already includes many filesystem implementations, such as:

*   A standard POSIX filesystem

    Note: NFS filesystems often mount as a POSIX interface, and so standard
    TensorFlow can work on top of NFS-mounted remote filesystems.

*   HDFS - the Hadoop File System
*   GCS - Google Cloud Storage filesystem
*   S3 - Amazon Simple Storage Service filesystem
*   A "memory-mapped-file" filesystem

The rest of this guide describes how to implement a custom filesystem.

## Implementing a custom filesystem plugin

To implement a custom filesystem plugin, you must do the following:

*   Implement subclasses of `RandomAccessFile`, `WriteableFile`,
    `AppendableFile`, and `ReadOnlyMemoryRegion`.
*   Implement the `FileSystem` interface as a subclass.
*   Register the `FileSystem` implementation with an appropriate prefix pattern.
*   Load the filesystem plugin in a process that wants to write to that
    filesystem.

### The FileSystem interface

The `FileSystem` interface is an abstract C++ interface defined in
[file_system.h](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/platform/file_system.h).
An implementation of the `FileSystem` interface should implement all relevant
the methods defined by the interface. Implementing the interface requires
defining operations such as creating `RandomAccessFile`, `WritableFile`, and
implementing standard filesystem operations such as `FileExists`, `IsDirectory`,
`GetMatchingPaths`, `DeleteFile`, and so on. An implementation of these
interfaces will often involve translating the function's input arguments to
delegate to an already-existing library function implementing the equivalent
functionality in your custom filesystem.

For example, the `PosixFileSystem` implementation implements `DeleteFile` using
the POSIX `unlink()` function; `CreateDir` simply calls `mkdir()`; `GetFileSize`
involves calling `stat()` on the file and then returns the filesize as reported
by the return of the stat object. Similarly, for the `HDFSFileSystem`
implementation, these calls simply delegate to the `libHDFS` implementation of
similar functionality, such as `hdfsDelete` for
[DeleteFile](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/platform/hadoop/hadoop_file_system.cc#L386).

We suggest looking through these code examples to get an idea of how different
filesystem implementations call their existing libraries. Examples include:

*   [POSIX
    plugin](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/platform/posix/posix_file_system.h)
*   [HDFS
    plugin](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/platform/hadoop/hadoop_file_system.h)
*   [GCS
    plugin](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/platform/cloud/gcs_file_system.h)
*   [S3
    plugin](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/platform/s3/s3_file_system.h)

#### The File interfaces

Beyond operations that allow you to query and manipulate files and directories
in a filesystem, the `FileSystem` interface requires you to implement factories
that return implementations of abstract objects such as the
[RandomAccessFile](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/platform/file_system.h#L223),
the `WritableFile`, so that TensorFlow code and read and write to files in that
`FileSystem` implementation.

To implement a `RandomAccessFile`, you must implement a single interface called
`Read()`, in which the implementation must provide a way to read from an offset
within a named file.

For example, below is the implementation of RandomAccessFile for the POSIX
filesystem, which uses the `pread()` random-access POSIX function to implement
read. Notice that the particular implementation must know how to retry or
propagate errors from the underlying filesystem.

```C++
    class PosixRandomAccessFile : public RandomAccessFile {
     public:
      PosixRandomAccessFile(const string& fname, int fd)
          : filename_(fname), fd_(fd) {}
      ~PosixRandomAccessFile() override { close(fd_); }

      Status Read(uint64 offset, size_t n, StringPiece* result,
                  char* scratch) const override {
        Status s;
        char* dst = scratch;
        while (n > 0 && s.ok()) {
          ssize_t r = pread(fd_, dst, n, static_cast<off_t>(offset));
          if (r > 0) {
            dst += r;
            n -= r;
            offset += r;
          } else if (r == 0) {
            s = Status(error::OUT_OF_RANGE, "Read less bytes than requested");
          } else if (errno == EINTR || errno == EAGAIN) {
            // Retry
          } else {
            s = IOError(filename_, errno);
          }
        }
        *result = StringPiece(scratch, dst - scratch);
        return s;
      }

     private:
      string filename_;
      int fd_;
    };
```

To implement the WritableFile sequential-writing abstraction, one must implement
a few interfaces, such as `Append()`, `Flush()`, `Sync()`, and `Close()`.

For example, below is the implementation of WritableFile for the POSIX
filesystem, which takes a `FILE` object in its constructor and uses standard
posix functions on that object to implement the interface.

```C++
    class PosixWritableFile : public WritableFile {
     public:
      PosixWritableFile(const string& fname, FILE* f)
          : filename_(fname), file_(f) {}

      ~PosixWritableFile() override {
        if (file_ != NULL) {
          fclose(file_);
        }
      }

      Status Append(const StringPiece& data) override {
        size_t r = fwrite(data.data(), 1, data.size(), file_);
        if (r != data.size()) {
          return IOError(filename_, errno);
        }
        return Status::OK();
      }

      Status Close() override {
        Status result;
        if (fclose(file_) != 0) {
          result = IOError(filename_, errno);
        }
        file_ = NULL;
        return result;
      }

      Status Flush() override {
        if (fflush(file_) != 0) {
          return IOError(filename_, errno);
        }
        return Status::OK();
      }

      Status Sync() override {
        Status s;
        if (fflush(file_) != 0) {
          s = IOError(filename_, errno);
        }
        return s;
      }

     private:
      string filename_;
      FILE* file_;
    };

```

For more details, please see the documentations of those interfaces, and look at
example implementations for inspiration.

### Registering and loading the filesystem

Once you have implemented the `FileSystem` implementation for your custom
filesystem, you need to register it under a "scheme" so that paths prefixed with
that scheme are directed to your implementation. To do this, you call
`REGISTER_FILE_SYSTEM`::

```
    REGISTER_FILE_SYSTEM("foobar", FooBarFileSystem);
```

When TensorFlow tries to operate on a file whose path starts with `foobar://`,
it will use the `FooBarFileSystem` implementation.

```C++
    string filename = "foobar://path/to/file.txt";
    std::unique_ptr<WritableFile> file;

    // Calls FooBarFileSystem::NewWritableFile to return
    // a WritableFile class, which happens to be the FooBarFileSystem's
    // WritableFile implementation.
    TF_RETURN_IF_ERROR(env->NewWritableFile(filename, &file));
```

Next, you must build a shared object containing this implementation. An example
of doing so using bazel's `cc_binary` rule can be found
[here](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/BUILD#L244),
but you may use any build system to do so. See the section on @{$adding_an_op#build_the_op_library$building the op library} for similar
instructions.

The result of building this target is a `.so` shared object file.

Lastly, you must dynamically load this implementation in the process. In Python,
you can call the `tf.load_file_system_library(file_system_library)` function,
passing the path to the shared object. Calling this in your client program loads
the shared object in the process, thus registering your implementation as
available for any file operations going through the `FileSystem` interface. You
can see
[test_file_system.py](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/framework/file_system_test.py)
for an example.

## What goes through this interface?

Almost all core C++ file operations within TensorFlow use the `FileSystem`
interface, such as the `CheckpointWriter`, the `EventsWriter`, and many other
utilities. This means implementing a `FileSystem` implementation allows most of
your TensorFlow programs to write to your shared filesystem.

In Python, the `gfile` and `file_io` classes bind underneath to the `FileSystem
implementation via SWIG, which means that once you have loaded this filesystem
library, you can do:

```
with gfile.Open("foobar://path/to/file.txt") as w:

  w.write("hi")
```

When you do this, a file containing "hi" will appear in the "/path/to/file.txt"
of your shared filesystem.
