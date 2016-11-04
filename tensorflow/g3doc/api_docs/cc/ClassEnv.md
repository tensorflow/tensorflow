# `class tensorflow::Env`

An interface used by the tensorflow implementation to access operating system functionality like the filesystem etc.

Callers may wish to provide a custom Env object to get fine grain control.

All Env implementations are safe for concurrent access from multiple threads without any external synchronization.

###Member Details

#### `tensorflow::Env::Env()` {#tensorflow_Env_Env}





#### `virtual tensorflow::Env::~Env()=default` {#virtual_tensorflow_Env_Env}





#### `Status tensorflow::Env::GetFileSystemForFile(const string &fname, FileSystem **result)` {#Status_tensorflow_Env_GetFileSystemForFile}

Returns the FileSystem object to handle operations on the file specified by &apos;fname&apos;. The FileSystem object is used as the implementation for the file system related (non-virtual) functions that follow. Returned FileSystem object is still owned by the Env object and will.



#### `Status tensorflow::Env::GetRegisteredFileSystemSchemes(std::vector< string > *schemes)` {#Status_tensorflow_Env_GetRegisteredFileSystemSchemes}

Returns the file system schemes registered for this Env .



#### `Status tensorflow::Env::RegisterFileSystem(const string &scheme, FileSystemRegistry::Factory factory)` {#Status_tensorflow_Env_RegisterFileSystem}





#### `Status tensorflow::Env::NewRandomAccessFile(const string &fname, std::unique_ptr< RandomAccessFile > *result)` {#Status_tensorflow_Env_NewRandomAccessFile}

Creates a brand new random access read-only file with the specified name.

On success, stores a pointer to the new file in *result and returns OK. On failure stores NULL in *result and returns non-OK. If the file does not exist, returns a non-OK status.

The returned file may be concurrently accessed by multiple threads.

The ownership of the returned RandomAccessFile is passed to the caller and the object should be deleted when is not used. The file object shouldn&apos;t live longer than the Env object.

#### `Status tensorflow::Env::NewWritableFile(const string &fname, std::unique_ptr< WritableFile > *result)` {#Status_tensorflow_Env_NewWritableFile}

Creates an object that writes to a new file with the specified name.

Deletes any existing file with the same name and creates a new file. On success, stores a pointer to the new file in *result and returns OK. On failure stores NULL in *result and returns non-OK.

The returned file will only be accessed by one thread at a time.

The ownership of the returned WritableFile is passed to the caller and the object should be deleted when is not used. The file object shouldn&apos;t live longer than the Env object.

#### `Status tensorflow::Env::NewAppendableFile(const string &fname, std::unique_ptr< WritableFile > *result)` {#Status_tensorflow_Env_NewAppendableFile}

Creates an object that either appends to an existing file, or writes to a new file (if the file does not exist to begin with).

On success, stores a pointer to the new file in *result and returns OK. On failure stores NULL in *result and returns non-OK.

The returned file will only be accessed by one thread at a time.

The ownership of the returned WritableFile is passed to the caller and the object should be deleted when is not used. The file object shouldn&apos;t live longer than the Env object.

#### `Status tensorflow::Env::NewReadOnlyMemoryRegionFromFile(const string &fname, std::unique_ptr< ReadOnlyMemoryRegion > *result)` {#Status_tensorflow_Env_NewReadOnlyMemoryRegionFromFile}

Creates a readonly region of memory with the file context.

On success, it returns a pointer to read-only memory region from the content of file fname. The ownership of the region is passed to the caller. On failure stores nullptr in *result and returns non-OK.

The returned memory region can be accessed from many threads in parallel.

The ownership of the returned ReadOnlyMemoryRegion is passed to the caller and the object should be deleted when is not used. The memory region object shouldn&apos;t live longer than the Env object.

#### `bool tensorflow::Env::FileExists(const string &fname)` {#bool_tensorflow_Env_FileExists}

Returns true iff the named file exists.



#### `Status tensorflow::Env::GetChildren(const string &dir, std::vector< string > *result)` {#Status_tensorflow_Env_GetChildren}

Stores in *result the names of the children of the specified directory. The names are relative to "dir".

Original contents of *results are dropped.

#### `virtual bool tensorflow::Env::MatchPath(const string &path, const string &pattern)=0` {#virtual_bool_tensorflow_Env_MatchPath}

Returns true if the path matches the given pattern. The wildcards allowed in pattern are described below (GetMatchingPaths).



#### `Status tensorflow::Env::GetMatchingPaths(const string &pattern, std::vector< string > *results)` {#Status_tensorflow_Env_GetMatchingPaths}

Given a pattern, stores in *results the set of paths that matches that pattern. *results is cleared.

pattern must match all of a name, not just a substring. pattern: { term } term: &apos;*&apos;: matches any sequence of non-&apos;/&apos; characters &apos;?&apos;: matches a single non-&apos;/&apos; character &apos;[&apos; [ &apos;^&apos; ] { match-list } &apos;]&apos;: matches any single character (not) on the list c: matches character c (c != &apos;*&apos;, &apos;?&apos;, &apos;\&apos;, &apos;[&apos;) &apos;\&apos; c: matches character c character-range: c: matches character c (c != &apos;\&apos;, &apos;-&apos;, &apos;]&apos;) &apos;\&apos; c: matches character c lo &apos;-&apos; hi: matches character c for lo <= c <= hi

Typical return codes

OK - no errors

UNIMPLEMENTED - Some underlying functions (like GetChildren) are not implemented The default implementation uses a combination of GetChildren, MatchPath and IsDirectory.

#### `Status tensorflow::Env::DeleteFile(const string &fname)` {#Status_tensorflow_Env_DeleteFile}

Deletes the named file.



#### `Status tensorflow::Env::DeleteRecursively(const string &dirname, int64 *undeleted_files, int64 *undeleted_dirs)` {#Status_tensorflow_Env_DeleteRecursively}

Deletes the specified directory and all subdirectories and files underneath it. undeleted_files and undeleted_dirs stores the number of files and directories that weren&apos;t deleted (unspecified if the return status is not OK). REQUIRES: undeleted_files, undeleted_dirs to be not null. Typical return codes.



OK - dirname exists and we were able to delete everything underneath.

NOT_FOUND - dirname doesn&apos;t exist

PERMISSION_DENIED - dirname or some descendant is not writable

UNIMPLEMENTED - Some underlying functions (like Delete) are not implemented

#### `Status tensorflow::Env::RecursivelyCreateDir(const string &dirname)` {#Status_tensorflow_Env_RecursivelyCreateDir}

Creates the specified directory and all the necessary subdirectories. Typical return codes.



OK - successfully created the directory and sub directories, even if they were already created.

PERMISSION_DENIED - dirname or some subdirectory is not writable.

#### `Status tensorflow::Env::CreateDir(const string &dirname)` {#Status_tensorflow_Env_CreateDir}

Creates the specified directory. Typical return codes.



OK - successfully created the directory.

ALREADY_EXISTS - directory already exists.

PERMISSION_DENIED - dirname is not writable.

#### `Status tensorflow::Env::DeleteDir(const string &dirname)` {#Status_tensorflow_Env_DeleteDir}

Deletes the specified directory.



#### `Status tensorflow::Env::Stat(const string &fname, FileStatistics *stat)` {#Status_tensorflow_Env_Stat}

Obtains statistics for the given path.



#### `Status tensorflow::Env::IsDirectory(const string &fname)` {#Status_tensorflow_Env_IsDirectory}

Returns whether the given path is a directory or not. Typical return codes (not guaranteed exhaustive):



OK - The path exists and is a directory.

FAILED_PRECONDITION - The path exists and is not a directory.

NOT_FOUND - The path entry does not exist.

PERMISSION_DENIED - Insufficient permissions.

UNIMPLEMENTED - The file factory doesn&apos;t support directories.

#### `Status tensorflow::Env::GetFileSize(const string &fname, uint64 *file_size)` {#Status_tensorflow_Env_GetFileSize}

Stores the size of `fname` in `*file_size`.



#### `Status tensorflow::Env::RenameFile(const string &src, const string &target)` {#Status_tensorflow_Env_RenameFile}

Renames file src to target. If target already exists, it will be replaced.



#### `virtual uint64 tensorflow::Env::NowMicros()=0` {#virtual_uint64_tensorflow_Env_NowMicros}

Returns the number of micro-seconds since some fixed point in time. Only useful for computing deltas of time.



#### `virtual uint64 tensorflow::Env::NowSeconds()` {#virtual_uint64_tensorflow_Env_NowSeconds}

Returns the number of seconds since some fixed point in time. Only useful for computing deltas of time.



#### `virtual void tensorflow::Env::SleepForMicroseconds(int64 micros)=0` {#virtual_void_tensorflow_Env_SleepForMicroseconds}

Sleeps/delays the thread for the prescribed number of micro-seconds.



#### `virtual Thread* tensorflow::Env::StartThread(const ThreadOptions &thread_options, const string &name, std::function< void()> fn) TF_MUST_USE_RESULT=0` {#virtual_Thread_tensorflow_Env_StartThread}

Returns a new thread that is running fn() and is identified (for debugging/performance-analysis) by "name".

Caller takes ownership of the result and must delete it eventually (the deletion will block until fn() stops running).

#### `virtual void tensorflow::Env::SchedClosure(std::function< void()> closure)=0` {#virtual_void_tensorflow_Env_SchedClosure}





#### `virtual void tensorflow::Env::SchedClosureAfter(int64 micros, std::function< void()> closure)=0` {#virtual_void_tensorflow_Env_SchedClosureAfter}





#### `virtual Status tensorflow::Env::LoadLibrary(const char *library_filename, void **handle)=0` {#virtual_Status_tensorflow_Env_LoadLibrary}





#### `virtual Status tensorflow::Env::GetSymbolFromLibrary(void *handle, const char *symbol_name, void **symbol)=0` {#virtual_Status_tensorflow_Env_GetSymbolFromLibrary}





#### `static Env* tensorflow::Env::Default()` {#static_Env_tensorflow_Env_Default}

Returns a default environment suitable for the current operating system.

Sophisticated users may wish to provide their own Env implementation instead of relying on this default environment.

The result of Default() belongs to this library and must never be deleted.
