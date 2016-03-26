# `class tensorflow::RandomAccessFile`

A file abstraction for randomly reading the contents of a file.



###Member Details

#### `tensorflow::RandomAccessFile::RandomAccessFile()` {#tensorflow_RandomAccessFile_RandomAccessFile}





#### `tensorflow::RandomAccessFile::~RandomAccessFile()` {#tensorflow_RandomAccessFile_RandomAccessFile}





#### `virtual Status tensorflow::RandomAccessFile::Read(uint64 offset, size_t n, StringPiece *result, char *scratch) const =0` {#virtual_Status_tensorflow_RandomAccessFile_Read}

Reads up to `n` bytes from the file starting at `offset`.

`scratch[0..n-1]` may be written by this routine. Sets `*result` to the data that was read (including if fewer than `n` bytes were successfully read). May set `*result` to point at data in `scratch[0..n-1]`, so `scratch[0..n-1]` must be live when `*result` is used.

On OK returned status: `n` bytes have been stored in `*result`. On non-OK returned status: `[0..n]` bytes have been stored in `*result`.

Returns `OUT_OF_RANGE` if fewer than n bytes were stored in `*result` because of EOF.

Safe for concurrent use by multiple threads.
