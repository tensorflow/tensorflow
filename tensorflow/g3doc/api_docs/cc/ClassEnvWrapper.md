# `class tensorflow::EnvWrapper`

An implementation of Env that forwards all calls to another Env .

May be useful to clients who wish to override just part of the functionality of another Env .

###Member Details

#### `tensorflow::EnvWrapper::EnvWrapper(Env *t)` {#tensorflow_EnvWrapper_EnvWrapper}

Initializes an EnvWrapper that delegates all calls to *t.



#### `tensorflow::EnvWrapper::~EnvWrapper()` {#tensorflow_EnvWrapper_EnvWrapper}





#### `Env* tensorflow::EnvWrapper::target() const` {#Env_tensorflow_EnvWrapper_target}

Returns the target to which this Env forwards all calls.



#### `Status tensorflow::EnvWrapper::GetFileSystemForFile(const string &fname, FileSystem **result) override` {#Status_tensorflow_EnvWrapper_GetFileSystemForFile}

Returns the FileSystem object to handle operations on the file specified by &apos;fname&apos;. The FileSystem object is used as the implementation for the file system related (non-virtual) functions that follow. Returned FileSystem object is still owned by the Env object and will.



#### `Status tensorflow::EnvWrapper::GetRegisteredFileSystemSchemes(std::vector< string > *schemes) override` {#Status_tensorflow_EnvWrapper_GetRegisteredFileSystemSchemes}

Returns the file system schemes registered for this Env .



#### `Status tensorflow::EnvWrapper::RegisterFileSystem(const string &scheme, FileSystemRegistry::Factory factory) override` {#Status_tensorflow_EnvWrapper_RegisterFileSystem}





#### `bool tensorflow::EnvWrapper::MatchPath(const string &path, const string &pattern) override` {#bool_tensorflow_EnvWrapper_MatchPath}

Returns true if the path matches the given pattern. The wildcards allowed in pattern are described in FileSystem::GetMatchingPaths.



#### `uint64 tensorflow::EnvWrapper::NowMicros() override` {#uint64_tensorflow_EnvWrapper_NowMicros}

Returns the number of micro-seconds since some fixed point in time. Only useful for computing deltas of time.



#### `void tensorflow::EnvWrapper::SleepForMicroseconds(int64 micros) override` {#void_tensorflow_EnvWrapper_SleepForMicroseconds}

Sleeps/delays the thread for the prescribed number of micro-seconds.



#### `Thread* tensorflow::EnvWrapper::StartThread(const ThreadOptions &thread_options, const string &name, std::function< void()> fn) override` {#Thread_tensorflow_EnvWrapper_StartThread}

Returns a new thread that is running fn() and is identified (for debugging/performance-analysis) by "name".

Caller takes ownership of the result and must delete it eventually (the deletion will block until fn() stops running).

#### `void tensorflow::EnvWrapper::SchedClosure(std::function< void()> closure) override` {#void_tensorflow_EnvWrapper_SchedClosure}





#### `void tensorflow::EnvWrapper::SchedClosureAfter(int64 micros, std::function< void()> closure) override` {#void_tensorflow_EnvWrapper_SchedClosureAfter}





#### `Status tensorflow::EnvWrapper::LoadLibrary(const char *library_filename, void **handle) override` {#Status_tensorflow_EnvWrapper_LoadLibrary}





#### `Status tensorflow::EnvWrapper::GetSymbolFromLibrary(void *handle, const char *symbol_name, void **symbol) override` {#Status_tensorflow_EnvWrapper_GetSymbolFromLibrary}





#### `string tensorflow::EnvWrapper::FormatLibraryFileName(const string &name, const string &version) override` {#string_tensorflow_EnvWrapper_FormatLibraryFileName}




