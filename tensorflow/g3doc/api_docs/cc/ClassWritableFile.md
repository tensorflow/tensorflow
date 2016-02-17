# `class tensorflow::WritableFile`

A file abstraction for sequential writing.

The implementation must provide buffering since callers may append small fragments at a time to the file.

###Member Details

#### `tensorflow::WritableFile::WritableFile()` {#tensorflow_WritableFile_WritableFile}





#### `tensorflow::WritableFile::~WritableFile()` {#tensorflow_WritableFile_WritableFile}





#### `virtual Status tensorflow::WritableFile::Append(const StringPiece &data)=0` {#virtual_Status_tensorflow_WritableFile_Append}





#### `virtual Status tensorflow::WritableFile::Close()=0` {#virtual_Status_tensorflow_WritableFile_Close}





#### `virtual Status tensorflow::WritableFile::Flush()=0` {#virtual_Status_tensorflow_WritableFile_Flush}





#### `virtual Status tensorflow::WritableFile::Sync()=0` {#virtual_Status_tensorflow_WritableFile_Sync}




