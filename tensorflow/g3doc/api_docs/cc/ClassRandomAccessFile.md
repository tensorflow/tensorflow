#Class tensorflow::RandomAccessFile

A file abstraction for randomly reading the contents of a file.



##Member Summary

* [tensorflow::RandomAccessFile::RandomAccessFile](#tensorflow_RandomAccessFile_RandomAccessFile)
* [virtual tensorflow::RandomAccessFile::~RandomAccessFile](#virtual_tensorflow_RandomAccessFile_RandomAccessFile)
* [virtual Status tensorflow::RandomAccessFile::Read](#virtual_Status_tensorflow_RandomAccessFile_Read)
  * Reads up to &quot;n&quot; bytes from the file starting at &quot;offset&quot;.

##Member Details

#### tensorflow::RandomAccessFile::RandomAccessFile() {#tensorflow_RandomAccessFile_RandomAccessFile}





#### virtual tensorflow::RandomAccessFile::~RandomAccessFile() {#virtual_tensorflow_RandomAccessFile_RandomAccessFile}





#### virtual Status tensorflow::RandomAccessFile::Read(uint64 offset, size_t n, StringPiece *result, char *scratch) const =0 {#virtual_Status_tensorflow_RandomAccessFile_Read}

Reads up to &quot;n&quot; bytes from the file starting at &quot;offset&quot;.

&quot;scratch[0..n-1]&quot; may be written by this routine. Sets &quot;*result&quot; to the data that was read (including if fewer than &quot;n&quot; bytes were successfully read). May set &quot;*result&quot; to point at data in &quot;scratch[0..n-1]&quot;, so &quot;scratch[0..n-1]&quot; must be live when &quot;*result&quot; is used.

On OK returned status: &quot;n&quot; bytes have been stored in &quot;*result&quot;. On non-OK returned status: [0..n] bytes have been stored in &quot;*result&quot;.

Returns OUT_OF_RANGE if fewer than n bytes were stored in &quot;*result&quot; because of EOF.

Safe for concurrent use by multiple threads.
