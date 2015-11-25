# Class `tensorflow::EnvWrapper`

An implementation of Env that forwards all calls to another Env .

May be useful to clients who wish to override just part of the functionality of another Env .

##Member Summary

* [`tensorflow::EnvWrapper::EnvWrapper(Env *t)`](#tensorflow_EnvWrapper_EnvWrapper)
  * Initializes an EnvWrapper that delegates all calls to *t.
* [`virtual tensorflow::EnvWrapper::~EnvWrapper()`](#virtual_tensorflow_EnvWrapper_EnvWrapper)
* [`Env* tensorflow::EnvWrapper::target() const`](#Env_tensorflow_EnvWrapper_target)
  * Returns the target to which this Env forwards all calls.
* [`Status tensorflow::EnvWrapper::NewRandomAccessFile(const string &f, RandomAccessFile **r) override`](#Status_tensorflow_EnvWrapper_NewRandomAccessFile)
  * Creates a brand new random access read-only file with the specified name.
* [`Status tensorflow::EnvWrapper::NewWritableFile(const string &f, WritableFile **r) override`](#Status_tensorflow_EnvWrapper_NewWritableFile)
  * Creates an object that writes to a new file with the specified name.
* [`Status tensorflow::EnvWrapper::NewAppendableFile(const string &f, WritableFile **r) override`](#Status_tensorflow_EnvWrapper_NewAppendableFile)
  * Creates an object that either appends to an existing file, or writes to a new file (if the file does not exist to begin with).
* [`bool tensorflow::EnvWrapper::FileExists(const string &f) override`](#bool_tensorflow_EnvWrapper_FileExists)
  * Returns true iff the named file exists.
* [`Status tensorflow::EnvWrapper::GetChildren(const string &dir, std::vector< string > *r) override`](#Status_tensorflow_EnvWrapper_GetChildren)
  * Stores in *result the names of the children of the specified directory. The names are relative to "dir".
* [`Status tensorflow::EnvWrapper::DeleteFile(const string &f) override`](#Status_tensorflow_EnvWrapper_DeleteFile)
  * Deletes the named file.
* [`Status tensorflow::EnvWrapper::CreateDir(const string &d) override`](#Status_tensorflow_EnvWrapper_CreateDir)
  * Creates the specified directory.
* [`Status tensorflow::EnvWrapper::DeleteDir(const string &d) override`](#Status_tensorflow_EnvWrapper_DeleteDir)
  * Deletes the specified directory.
* [`Status tensorflow::EnvWrapper::GetFileSize(const string &f, uint64 *s) override`](#Status_tensorflow_EnvWrapper_GetFileSize)
  * Stores the size of `fname` in `*file_size`.
* [`Status tensorflow::EnvWrapper::RenameFile(const string &s, const string &t) override`](#Status_tensorflow_EnvWrapper_RenameFile)
  * Renames file src to target. If target already exists, it will be replaced.
* [`uint64 tensorflow::EnvWrapper::NowMicros() override`](#uint64_tensorflow_EnvWrapper_NowMicros)
  * Returns the number of micro-seconds since some fixed point in time. Only useful for computing deltas of time.
* [`void tensorflow::EnvWrapper::SleepForMicroseconds(int micros) override`](#void_tensorflow_EnvWrapper_SleepForMicroseconds)
  * Sleeps/delays the thread for the prescribed number of micro-seconds.
* [`Thread* tensorflow::EnvWrapper::StartThread(const ThreadOptions &thread_options, const string &name, std::function< void()> fn) override`](#Thread_tensorflow_EnvWrapper_StartThread)
  * Returns a new thread that is running fn() and is identified (for debugging/performance-analysis) by "name".

##Member Details

#### `tensorflow::EnvWrapper::EnvWrapper(Env *t)` {#tensorflow_EnvWrapper_EnvWrapper}

Initializes an EnvWrapper that delegates all calls to *t.



#### `virtual tensorflow::EnvWrapper::~EnvWrapper()` {#virtual_tensorflow_EnvWrapper_EnvWrapper}





#### `Env* tensorflow::EnvWrapper::target() const` {#Env_tensorflow_EnvWrapper_target}

Returns the target to which this Env forwards all calls.



#### `Status tensorflow::EnvWrapper::NewRandomAccessFile(const string &f, RandomAccessFile **r) override` {#Status_tensorflow_EnvWrapper_NewRandomAccessFile}

Creates a brand new random access read-only file with the specified name.

On success, stores a pointer to the new file in *result and returns OK. On failure stores NULL in *result and returns non-OK. If the file does not exist, returns a non-OK status.

The returned file may be concurrently accessed by multiple threads.

#### `Status tensorflow::EnvWrapper::NewWritableFile(const string &f, WritableFile **r) override` {#Status_tensorflow_EnvWrapper_NewWritableFile}

Creates an object that writes to a new file with the specified name.

Deletes any existing file with the same name and creates a new file. On success, stores a pointer to the new file in *result and returns OK. On failure stores NULL in *result and returns non-OK.

The returned file will only be accessed by one thread at a time.

#### `Status tensorflow::EnvWrapper::NewAppendableFile(const string &f, WritableFile **r) override` {#Status_tensorflow_EnvWrapper_NewAppendableFile}

Creates an object that either appends to an existing file, or writes to a new file (if the file does not exist to begin with).

On success, stores a pointer to the new file in *result and returns OK. On failure stores NULL in *result and returns non-OK.

The returned file will only be accessed by one thread at a time.

#### `bool tensorflow::EnvWrapper::FileExists(const string &f) override` {#bool_tensorflow_EnvWrapper_FileExists}

Returns true iff the named file exists.



#### `Status tensorflow::EnvWrapper::GetChildren(const string &dir, std::vector< string > *r) override` {#Status_tensorflow_EnvWrapper_GetChildren}

Stores in *result the names of the children of the specified directory. The names are relative to "dir".

Original contents of *results are dropped.

#### `Status tensorflow::EnvWrapper::DeleteFile(const string &f) override` {#Status_tensorflow_EnvWrapper_DeleteFile}

Deletes the named file.



#### `Status tensorflow::EnvWrapper::CreateDir(const string &d) override` {#Status_tensorflow_EnvWrapper_CreateDir}

Creates the specified directory.



#### `Status tensorflow::EnvWrapper::DeleteDir(const string &d) override` {#Status_tensorflow_EnvWrapper_DeleteDir}

Deletes the specified directory.



#### `Status tensorflow::EnvWrapper::GetFileSize(const string &f, uint64 *s) override` {#Status_tensorflow_EnvWrapper_GetFileSize}

Stores the size of `fname` in `*file_size`.



#### `Status tensorflow::EnvWrapper::RenameFile(const string &s, const string &t) override` {#Status_tensorflow_EnvWrapper_RenameFile}

Renames file src to target. If target already exists, it will be replaced.



#### `uint64 tensorflow::EnvWrapper::NowMicros() override` {#uint64_tensorflow_EnvWrapper_NowMicros}

Returns the number of micro-seconds since some fixed point in time. Only useful for computing deltas of time.



#### `void tensorflow::EnvWrapper::SleepForMicroseconds(int micros) override` {#void_tensorflow_EnvWrapper_SleepForMicroseconds}

Sleeps/delays the thread for the prescribed number of micro-seconds.



#### `Thread* tensorflow::EnvWrapper::StartThread(const ThreadOptions &thread_options, const string &name, std::function< void()> fn) override` {#Thread_tensorflow_EnvWrapper_StartThread}

Returns a new thread that is running fn() and is identified (for debugging/performance-analysis) by "name".

Caller takes ownership of the result and must delete it eventually (the deletion will block until fn() stops running).
