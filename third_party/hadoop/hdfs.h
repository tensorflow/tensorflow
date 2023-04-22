/**
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef TENSORFLOW_THIRD_PARTY_HADOOP_HDFS_H_
#define TENSORFLOW_THIRD_PARTY_HADOOP_HDFS_H_

#include <errno.h>  /* for EINTERNAL, etc. */
#include <fcntl.h>  /* for O_RDONLY, O_WRONLY */
#include <stdint.h> /* for uint64_t, etc. */
#include <time.h>   /* for time_t */

/*
 * Support export of DLL symbols during libhdfs build, and import of DLL symbols
 * during client application build.  A client application may optionally define
 * symbol LIBHDFS_DLL_IMPORT in its build.  This is not strictly required, but
 * the compiler can produce more efficient code with it.
 */
#ifdef WIN32
#ifdef LIBHDFS_DLL_EXPORT
#define LIBHDFS_EXTERNAL __declspec(dllexport)
#elif LIBHDFS_DLL_IMPORT
#define LIBHDFS_EXTERNAL __declspec(dllimport)
#else
#define LIBHDFS_EXTERNAL
#endif
#else
#ifdef LIBHDFS_DLL_EXPORT
#define LIBHDFS_EXTERNAL __attribute__((visibility("default")))
#elif LIBHDFS_DLL_IMPORT
#define LIBHDFS_EXTERNAL __attribute__((visibility("default")))
#else
#define LIBHDFS_EXTERNAL
#endif
#endif

#ifndef O_RDONLY
#define O_RDONLY 1
#endif

#ifndef O_WRONLY
#define O_WRONLY 2
#endif

#ifndef EINTERNAL
#define EINTERNAL 255
#endif

#define ELASTIC_BYTE_BUFFER_POOL_CLASS \
  "org/apache/hadoop/io/ElasticByteBufferPool"

/** All APIs set errno to meaningful values */

#ifdef __cplusplus
extern "C" {
#endif
/**
 * Some utility decls used in libhdfs.
 */
struct hdfsBuilder;
typedef int32_t tSize;    /// size of data for read/write io ops
typedef time_t tTime;     /// time type in seconds
typedef int64_t tOffset;  /// offset within the file
typedef uint16_t tPort;   /// port
typedef enum tObjectKind {
  kObjectKindFile = 'F',
  kObjectKindDirectory = 'D',
} tObjectKind;

/**
 * The C reflection of org.apache.org.hadoop.FileSystem .
 */
struct hdfs_internal;
typedef struct hdfs_internal *hdfsFS;

struct hdfsFile_internal;
typedef struct hdfsFile_internal *hdfsFile;

struct hadoopRzOptions;

struct hadoopRzBuffer;

/**
 * Determine if a file is open for read.
 *
 * @param file     The HDFS file
 * @return         1 if the file is open for read; 0 otherwise
 */
LIBHDFS_EXTERNAL
int hdfsFileIsOpenForRead(hdfsFile file);

/**
 * Determine if a file is open for write.
 *
 * @param file     The HDFS file
 * @return         1 if the file is open for write; 0 otherwise
 */
LIBHDFS_EXTERNAL
int hdfsFileIsOpenForWrite(hdfsFile file);

struct hdfsReadStatistics {
  uint64_t totalBytesRead;
  uint64_t totalLocalBytesRead;
  uint64_t totalShortCircuitBytesRead;
  uint64_t totalZeroCopyBytesRead;
};

/**
 * Get read statistics about a file.  This is only applicable to files
 * opened for reading.
 *
 * @param file     The HDFS file
 * @param stats    (out parameter) on a successful return, the read
 *                 statistics.  Unchanged otherwise.  You must free the
 *                 returned statistics with hdfsFileFreeReadStatistics.
 * @return         0 if the statistics were successfully returned,
 *                 -1 otherwise.  On a failure, please check errno against
 *                 ENOTSUP.  webhdfs, LocalFilesystem, and so forth may
 *                 not support read statistics.
 */
LIBHDFS_EXTERNAL
int hdfsFileGetReadStatistics(hdfsFile file, struct hdfsReadStatistics **stats);

/**
 * @param stats    HDFS read statistics for a file.
 *
 * @return the number of remote bytes read.
 */
LIBHDFS_EXTERNAL
int64_t hdfsReadStatisticsGetRemoteBytesRead(
    const struct hdfsReadStatistics *stats);

/**
 * Clear the read statistics for a file.
 *
 * @param file      The file to clear the read statistics of.
 *
 * @return          0 on success; the error code otherwise.
 *                  EINVAL: the file is not open for reading.
 *                  ENOTSUP: the file does not support clearing the read
 *                  statistics.
 *                  Errno will also be set to this code on failure.
 */
LIBHDFS_EXTERNAL
int hdfsFileClearReadStatistics(hdfsFile file);

/**
 * Free some HDFS read statistics.
 *
 * @param stats    The HDFS read statistics to free.
 */
LIBHDFS_EXTERNAL
void hdfsFileFreeReadStatistics(struct hdfsReadStatistics *stats);

/**
 * hdfsConnectAsUser - Connect to an hdfs file system as a specific user
 * Connect to the hdfs.
 * @param nn   The NameNode.  See hdfsBuilderSetNameNode for details.
 * @param port The port on which the server is listening.
 * @param user the user name (this is hadoop domain user). Or NULL is equivalent
 * to hhdfsConnect(host, port)
 * @return Returns a handle to the filesystem or NULL on error.
 * @deprecated Use hdfsBuilderConnect instead.
 */
LIBHDFS_EXTERNAL
hdfsFS hdfsConnectAsUser(const char *nn, tPort port, const char *user);

/**
 * hdfsConnect - Connect to an hdfs file system.
 * Connect to the hdfs.
 * @param nn   The NameNode.  See hdfsBuilderSetNameNode for details.
 * @param port The port on which the server is listening.
 * @return Returns a handle to the filesystem or NULL on error.
 * @deprecated Use hdfsBuilderConnect instead.
 */
LIBHDFS_EXTERNAL
hdfsFS hdfsConnect(const char *nn, tPort port);

/**
 * hdfsConnect - Connect to an hdfs file system.
 *
 * Forces a new instance to be created
 *
 * @param nn     The NameNode.  See hdfsBuilderSetNameNode for details.
 * @param port   The port on which the server is listening.
 * @param user   The user name to use when connecting
 * @return       Returns a handle to the filesystem or NULL on error.
 * @deprecated   Use hdfsBuilderConnect instead.
 */
LIBHDFS_EXTERNAL
hdfsFS hdfsConnectAsUserNewInstance(const char *nn, tPort port,
                                    const char *user);

/**
 * hdfsConnect - Connect to an hdfs file system.
 *
 * Forces a new instance to be created
 *
 * @param nn     The NameNode.  See hdfsBuilderSetNameNode for details.
 * @param port   The port on which the server is listening.
 * @return       Returns a handle to the filesystem or NULL on error.
 * @deprecated   Use hdfsBuilderConnect instead.
 */
LIBHDFS_EXTERNAL
hdfsFS hdfsConnectNewInstance(const char *nn, tPort port);

/**
 * Connect to HDFS using the parameters defined by the builder.
 *
 * The HDFS builder will be freed, whether or not the connection was
 * successful.
 *
 * Every successful call to hdfsBuilderConnect should be matched with a call
 * to hdfsDisconnect, when the hdfsFS is no longer needed.
 *
 * @param bld    The HDFS builder
 * @return       Returns a handle to the filesystem, or NULL on error.
 */
LIBHDFS_EXTERNAL
hdfsFS hdfsBuilderConnect(struct hdfsBuilder *bld);

/**
 * Create an HDFS builder.
 *
 * @return The HDFS builder, or NULL on error.
 */
LIBHDFS_EXTERNAL
struct hdfsBuilder *hdfsNewBuilder(void);

/**
 * Force the builder to always create a new instance of the FileSystem,
 * rather than possibly finding one in the cache.
 *
 * @param bld The HDFS builder
 */
LIBHDFS_EXTERNAL
void hdfsBuilderSetForceNewInstance(struct hdfsBuilder *bld);

/**
 * Set the HDFS NameNode to connect to.
 *
 * @param bld  The HDFS builder
 * @param nn   The NameNode to use.
 *
 *             If the string given is 'default', the default NameNode
 *             configuration will be used (from the XML configuration files)
 *
 *             If NULL is given, a LocalFileSystem will be created.
 *
 *             If the string starts with a protocol type such as file:// or
 *             hdfs://, this protocol type will be used.  If not, the
 *             hdfs:// protocol type will be used.
 *
 *             You may specify a NameNode port in the usual way by
 *             passing a string of the format hdfs://<hostname>:<port>.
 *             Alternately, you may set the port with
 *             hdfsBuilderSetNameNodePort.  However, you must not pass the
 *             port in two different ways.
 */
LIBHDFS_EXTERNAL
void hdfsBuilderSetNameNode(struct hdfsBuilder *bld, const char *nn);

/**
 * Set the port of the HDFS NameNode to connect to.
 *
 * @param bld The HDFS builder
 * @param port The port.
 */
LIBHDFS_EXTERNAL
void hdfsBuilderSetNameNodePort(struct hdfsBuilder *bld, tPort port);

/**
 * Set the username to use when connecting to the HDFS cluster.
 *
 * @param bld The HDFS builder
 * @param userName The user name.  The string will be shallow-copied.
 */
LIBHDFS_EXTERNAL
void hdfsBuilderSetUserName(struct hdfsBuilder *bld, const char *userName);

/**
 * Set the path to the Kerberos ticket cache to use when connecting to
 * the HDFS cluster.
 *
 * @param bld The HDFS builder
 * @param kerbTicketCachePath The Kerberos ticket cache path.  The string
 *                            will be shallow-copied.
 */
LIBHDFS_EXTERNAL
void hdfsBuilderSetKerbTicketCachePath(struct hdfsBuilder *bld,
                                       const char *kerbTicketCachePath);

/**
 * Free an HDFS builder.
 *
 * It is normally not necessary to call this function since
 * hdfsBuilderConnect frees the builder.
 *
 * @param bld The HDFS builder
 */
LIBHDFS_EXTERNAL
void hdfsFreeBuilder(struct hdfsBuilder *bld);

/**
 * Set a configuration string for an HdfsBuilder.
 *
 * @param key      The key to set.
 * @param val      The value, or NULL to set no value.
 *                 This will be shallow-copied.  You are responsible for
 *                 ensuring that it remains valid until the builder is
 *                 freed.
 *
 * @return         0 on success; nonzero error code otherwise.
 */
LIBHDFS_EXTERNAL
int hdfsBuilderConfSetStr(struct hdfsBuilder *bld, const char *key,
                          const char *val);

/**
 * Get a configuration string.
 *
 * @param key      The key to find
 * @param val      (out param) The value.  This will be set to NULL if the
 *                 key isn't found.  You must free this string with
 *                 hdfsConfStrFree.
 *
 * @return         0 on success; nonzero error code otherwise.
 *                 Failure to find the key is not an error.
 */
LIBHDFS_EXTERNAL
int hdfsConfGetStr(const char *key, char **val);

/**
 * Get a configuration integer.
 *
 * @param key      The key to find
 * @param val      (out param) The value.  This will NOT be changed if the
 *                 key isn't found.
 *
 * @return         0 on success; nonzero error code otherwise.
 *                 Failure to find the key is not an error.
 */
LIBHDFS_EXTERNAL
int hdfsConfGetInt(const char *key, int32_t *val);

/**
 * Free a configuration string found with hdfsConfGetStr.
 *
 * @param val      A configuration string obtained from hdfsConfGetStr
 */
LIBHDFS_EXTERNAL
void hdfsConfStrFree(char *val);

/**
 * hdfsDisconnect - Disconnect from the hdfs file system.
 * Disconnect from hdfs.
 * @param fs The configured filesystem handle.
 * @return Returns 0 on success, -1 on error.
 *         Even if there is an error, the resources associated with the
 *         hdfsFS will be freed.
 */
LIBHDFS_EXTERNAL
int hdfsDisconnect(hdfsFS fs);

/**
 * hdfsOpenFile - Open an hdfs file in given mode.
 * @param fs The configured filesystem handle.
 * @param path The full path to the file.
 * @param flags - an | of bits/fcntl.h file flags - supported flags are
 * O_RDONLY, O_WRONLY (meaning create or overwrite i.e., implies O_TRUNCAT),
 * O_WRONLY|O_APPEND. Other flags are generally ignored other than (O_RDWR ||
 * (O_EXCL & O_CREAT)) which return NULL and set errno equal ENOTSUP.
 * @param bufferSize Size of buffer for read/write - pass 0 if you want
 * to use the default configured values.
 * @param replication Block replication - pass 0 if you want to use
 * the default configured values.
 * @param blocksize Size of block - pass 0 if you want to use the
 * default configured values.
 * @return Returns the handle to the open file or NULL on error.
 */
LIBHDFS_EXTERNAL
hdfsFile hdfsOpenFile(hdfsFS fs, const char *path, int flags, int bufferSize,
                      short replication, tSize blocksize);

/**
 * hdfsTruncateFile - Truncate an hdfs file to given length.
 * @param fs The configured filesystem handle.
 * @param path The full path to the file.
 * @param newlength The size the file is to be truncated to
 * @return 1 if the file has been truncated to the desired newlength
 *         and is immediately available to be reused for write operations
 *         such as append.
 *         0 if a background process of adjusting the length of the last
 *         block has been started, and clients should wait for it to
 *         complete before proceeding with further file updates.
 *         -1 on error.
 */
int hdfsTruncateFile(hdfsFS fs, const char *path, tOffset newlength);

/**
 * hdfsUnbufferFile - Reduce the buffering done on a file.
 *
 * @param file  The file to unbuffer.
 * @return      0 on success
 *              ENOTSUP if the file does not support unbuffering
 *              Errno will also be set to this value.
 */
LIBHDFS_EXTERNAL
int hdfsUnbufferFile(hdfsFile file);

/**
 * hdfsCloseFile - Close an open file.
 * @param fs The configured filesystem handle.
 * @param file The file handle.
 * @return Returns 0 on success, -1 on error.
 *         On error, errno will be set appropriately.
 *         If the hdfs file was valid, the memory associated with it will
 *         be freed at the end of this call, even if there was an I/O
 *         error.
 */
LIBHDFS_EXTERNAL
int hdfsCloseFile(hdfsFS fs, hdfsFile file);

/**
 * hdfsExists - Checks if a given path exsits on the filesystem
 * @param fs The configured filesystem handle.
 * @param path The path to look for
 * @return Returns 0 on success, -1 on error.
 */
LIBHDFS_EXTERNAL
int hdfsExists(hdfsFS fs, const char *path);

/**
 * hdfsSeek - Seek to given offset in file.
 * This works only for files opened in read-only mode.
 * @param fs The configured filesystem handle.
 * @param file The file handle.
 * @param desiredPos Offset into the file to seek into.
 * @return Returns 0 on success, -1 on error.
 */
LIBHDFS_EXTERNAL
int hdfsSeek(hdfsFS fs, hdfsFile file, tOffset desiredPos);

/**
 * hdfsTell - Get the current offset in the file, in bytes.
 * @param fs The configured filesystem handle.
 * @param file The file handle.
 * @return Current offset, -1 on error.
 */
LIBHDFS_EXTERNAL
tOffset hdfsTell(hdfsFS fs, hdfsFile file);

/**
 * hdfsRead - Read data from an open file.
 * @param fs The configured filesystem handle.
 * @param file The file handle.
 * @param buffer The buffer to copy read bytes into.
 * @param length The length of the buffer.
 * @return      On success, a positive number indicating how many bytes
 *              were read.
 *              On end-of-file, 0.
 *              On error, -1.  Errno will be set to the error code.
 *              Just like the POSIX read function, hdfsRead will return -1
 *              and set errno to EINTR if data is temporarily unavailable,
 *              but we are not yet at the end of the file.
 */
LIBHDFS_EXTERNAL
tSize hdfsRead(hdfsFS fs, hdfsFile file, void *buffer, tSize length);

/**
 * hdfsPread - Positional read of data from an open file.
 * @param fs The configured filesystem handle.
 * @param file The file handle.
 * @param position Position from which to read
 * @param buffer The buffer to copy read bytes into.
 * @param length The length of the buffer.
 * @return      See hdfsRead
 */
LIBHDFS_EXTERNAL
tSize hdfsPread(hdfsFS fs, hdfsFile file, tOffset position, void *buffer,
                tSize length);

/**
 * hdfsWrite - Write data into an open file.
 * @param fs The configured filesystem handle.
 * @param file The file handle.
 * @param buffer The data.
 * @param length The no. of bytes to write.
 * @return Returns the number of bytes written, -1 on error.
 */
LIBHDFS_EXTERNAL
tSize hdfsWrite(hdfsFS fs, hdfsFile file, const void *buffer, tSize length);

/**
 * hdfsWrite - Flush the data.
 * @param fs The configured filesystem handle.
 * @param file The file handle.
 * @return Returns 0 on success, -1 on error.
 */
LIBHDFS_EXTERNAL
int hdfsFlush(hdfsFS fs, hdfsFile file);

/**
 * hdfsHFlush - Flush out the data in client's user buffer. After the
 * return of this call, new readers will see the data.
 * @param fs configured filesystem handle
 * @param file file handle
 * @return 0 on success, -1 on error and sets errno
 */
LIBHDFS_EXTERNAL
int hdfsHFlush(hdfsFS fs, hdfsFile file);

/**
 * hdfsHSync - Similar to posix fsync, Flush out the data in client's
 * user buffer. all the way to the disk device (but the disk may have
 * it in its cache).
 * @param fs configured filesystem handle
 * @param file file handle
 * @return 0 on success, -1 on error and sets errno
 */
LIBHDFS_EXTERNAL
int hdfsHSync(hdfsFS fs, hdfsFile file);

/**
 * hdfsAvailable - Number of bytes that can be read from this
 * input stream without blocking.
 * @param fs The configured filesystem handle.
 * @param file The file handle.
 * @return Returns available bytes; -1 on error.
 */
LIBHDFS_EXTERNAL
int hdfsAvailable(hdfsFS fs, hdfsFile file);

/**
 * hdfsCopy - Copy file from one filesystem to another.
 * @param srcFS The handle to source filesystem.
 * @param src The path of source file.
 * @param dstFS The handle to destination filesystem.
 * @param dst The path of destination file.
 * @return Returns 0 on success, -1 on error.
 */
LIBHDFS_EXTERNAL
int hdfsCopy(hdfsFS srcFS, const char *src, hdfsFS dstFS, const char *dst);

/**
 * hdfsMove - Move file from one filesystem to another.
 * @param srcFS The handle to source filesystem.
 * @param src The path of source file.
 * @param dstFS The handle to destination filesystem.
 * @param dst The path of destination file.
 * @return Returns 0 on success, -1 on error.
 */
LIBHDFS_EXTERNAL
int hdfsMove(hdfsFS srcFS, const char *src, hdfsFS dstFS, const char *dst);

/**
 * hdfsDelete - Delete file.
 * @param fs The configured filesystem handle.
 * @param path The path of the file.
 * @param recursive if path is a directory and set to
 * non-zero, the directory is deleted else throws an exception. In
 * case of a file the recursive argument is irrelevant.
 * @return Returns 0 on success, -1 on error.
 */
LIBHDFS_EXTERNAL
int hdfsDelete(hdfsFS fs, const char *path, int recursive);

/**
 * hdfsRename - Rename file.
 * @param fs The configured filesystem handle.
 * @param oldPath The path of the source file.
 * @param newPath The path of the destination file.
 * @return Returns 0 on success, -1 on error.
 */
LIBHDFS_EXTERNAL
int hdfsRename(hdfsFS fs, const char *oldPath, const char *newPath);

/**
 * hdfsGetWorkingDirectory - Get the current working directory for
 * the given filesystem.
 * @param fs The configured filesystem handle.
 * @param buffer The user-buffer to copy path of cwd into.
 * @param bufferSize The length of user-buffer.
 * @return Returns buffer, NULL on error.
 */
LIBHDFS_EXTERNAL
char *hdfsGetWorkingDirectory(hdfsFS fs, char *buffer, size_t bufferSize);

/**
 * hdfsSetWorkingDirectory - Set the working directory. All relative
 * paths will be resolved relative to it.
 * @param fs The configured filesystem handle.
 * @param path The path of the new 'cwd'.
 * @return Returns 0 on success, -1 on error.
 */
LIBHDFS_EXTERNAL
int hdfsSetWorkingDirectory(hdfsFS fs, const char *path);

/**
 * hdfsCreateDirectory - Make the given file and all non-existent
 * parents into directories.
 * @param fs The configured filesystem handle.
 * @param path The path of the directory.
 * @return Returns 0 on success, -1 on error.
 */
LIBHDFS_EXTERNAL
int hdfsCreateDirectory(hdfsFS fs, const char *path);

/**
 * hdfsSetReplication - Set the replication of the specified
 * file to the supplied value
 * @param fs The configured filesystem handle.
 * @param path The path of the file.
 * @return Returns 0 on success, -1 on error.
 */
LIBHDFS_EXTERNAL
int hdfsSetReplication(hdfsFS fs, const char *path, int16_t replication);

/**
 * hdfsFileInfo - Information about a file/directory.
 */
typedef struct {
  tObjectKind mKind;  /* file or directory */
  char *mName;        /* the name of the file */
  tTime mLastMod;     /* the last modification time for the file in seconds */
  tOffset mSize;      /* the size of the file in bytes */
  short mReplication; /* the count of replicas */
  tOffset mBlockSize; /* the block size for the file */
  char *mOwner;       /* the owner of the file */
  char *mGroup;       /* the group associated with the file */
  short mPermissions; /* the permissions associated with the file */
  tTime mLastAccess;  /* the last access time for the file in seconds */
} hdfsFileInfo;

/**
 * hdfsListDirectory - Get list of files/directories for a given
 * directory-path. hdfsFreeFileInfo should be called to deallocate memory.
 * @param fs The configured filesystem handle.
 * @param path The path of the directory.
 * @param numEntries Set to the number of files/directories in path.
 * @return Returns a dynamically-allocated array of hdfsFileInfo
 * objects; NULL on error.
 */
LIBHDFS_EXTERNAL
hdfsFileInfo *hdfsListDirectory(hdfsFS fs, const char *path, int *numEntries);

/**
 * hdfsGetPathInfo - Get information about a path as a (dynamically
 * allocated) single hdfsFileInfo struct. hdfsFreeFileInfo should be
 * called when the pointer is no longer needed.
 * @param fs The configured filesystem handle.
 * @param path The path of the file.
 * @return Returns a dynamically-allocated hdfsFileInfo object;
 * NULL on error.
 */
LIBHDFS_EXTERNAL
hdfsFileInfo *hdfsGetPathInfo(hdfsFS fs, const char *path);

/**
 * hdfsFreeFileInfo - Free up the hdfsFileInfo array (including fields)
 * @param hdfsFileInfo The array of dynamically-allocated hdfsFileInfo
 * objects.
 * @param numEntries The size of the array.
 */
LIBHDFS_EXTERNAL
void hdfsFreeFileInfo(hdfsFileInfo *hdfsFileInfo, int numEntries);

/**
 * hdfsFileIsEncrypted: determine if a file is encrypted based on its
 * hdfsFileInfo.
 * @return -1 if there was an error (errno will be set), 0 if the file is
 *         not encrypted, 1 if the file is encrypted.
 */
LIBHDFS_EXTERNAL
int hdfsFileIsEncrypted(hdfsFileInfo *hdfsFileInfo);

/**
 * hdfsGetHosts - Get hostnames where a particular block (determined by
 * pos & blocksize) of a file is stored. The last element in the array
 * is NULL. Due to replication, a single block could be present on
 * multiple hosts.
 * @param fs The configured filesystem handle.
 * @param path The path of the file.
 * @param start The start of the block.
 * @param length The length of the block.
 * @return Returns a dynamically-allocated 2-d array of blocks-hosts;
 * NULL on error.
 */
LIBHDFS_EXTERNAL
char ***hdfsGetHosts(hdfsFS fs, const char *path, tOffset start,
                     tOffset length);

/**
 * hdfsFreeHosts - Free up the structure returned by hdfsGetHosts
 * @param hdfsFileInfo The array of dynamically-allocated hdfsFileInfo
 * objects.
 * @param numEntries The size of the array.
 */
LIBHDFS_EXTERNAL
void hdfsFreeHosts(char ***blockHosts);

/**
 * hdfsGetDefaultBlockSize - Get the default blocksize.
 *
 * @param fs            The configured filesystem handle.
 * @deprecated          Use hdfsGetDefaultBlockSizeAtPath instead.
 *
 * @return              Returns the default blocksize, or -1 on error.
 */
LIBHDFS_EXTERNAL
tOffset hdfsGetDefaultBlockSize(hdfsFS fs);

/**
 * hdfsGetDefaultBlockSizeAtPath - Get the default blocksize at the
 * filesystem indicated by a given path.
 *
 * @param fs            The configured filesystem handle.
 * @param path          The given path will be used to locate the actual
 *                      filesystem.  The full path does not have to exist.
 *
 * @return              Returns the default blocksize, or -1 on error.
 */
LIBHDFS_EXTERNAL
tOffset hdfsGetDefaultBlockSizeAtPath(hdfsFS fs, const char *path);

/**
 * hdfsGetCapacity - Return the raw capacity of the filesystem.
 * @param fs The configured filesystem handle.
 * @return Returns the raw-capacity; -1 on error.
 */
LIBHDFS_EXTERNAL
tOffset hdfsGetCapacity(hdfsFS fs);

/**
 * hdfsGetUsed - Return the total raw size of all files in the filesystem.
 * @param fs The configured filesystem handle.
 * @return Returns the total-size; -1 on error.
 */
LIBHDFS_EXTERNAL
tOffset hdfsGetUsed(hdfsFS fs);

/**
 * Change the user and/or group of a file or directory.
 *
 * @param fs            The configured filesystem handle.
 * @param path          the path to the file or directory
 * @param owner         User string.  Set to NULL for 'no change'
 * @param group         Group string.  Set to NULL for 'no change'
 * @return              0 on success else -1
 */
LIBHDFS_EXTERNAL
int hdfsChown(hdfsFS fs, const char *path, const char *owner,
              const char *group);

/**
 * hdfsChmod
 * @param fs The configured filesystem handle.
 * @param path the path to the file or directory
 * @param mode the bitmask to set it to
 * @return 0 on success else -1
 */
LIBHDFS_EXTERNAL
int hdfsChmod(hdfsFS fs, const char *path, short mode);

/**
 * hdfsUtime
 * @param fs The configured filesystem handle.
 * @param path the path to the file or directory
 * @param mtime new modification time or -1 for no change
 * @param atime new access time or -1 for no change
 * @return 0 on success else -1
 */
LIBHDFS_EXTERNAL
int hdfsUtime(hdfsFS fs, const char *path, tTime mtime, tTime atime);

/**
 * Allocate a zero-copy options structure.
 *
 * You must free all options structures allocated with this function using
 * hadoopRzOptionsFree.
 *
 * @return            A zero-copy options structure, or NULL if one could
 *                    not be allocated.  If NULL is returned, errno will
 *                    contain the error number.
 */
LIBHDFS_EXTERNAL
struct hadoopRzOptions *hadoopRzOptionsAlloc(void);

/**
 * Determine whether we should skip checksums in read0.
 *
 * @param opts        The options structure.
 * @param skip        Nonzero to skip checksums sometimes; zero to always
 *                    check them.
 *
 * @return            0 on success; -1 plus errno on failure.
 */
LIBHDFS_EXTERNAL
int hadoopRzOptionsSetSkipChecksum(struct hadoopRzOptions *opts, int skip);

/**
 * Set the ByteBufferPool to use with read0.
 *
 * @param opts        The options structure.
 * @param className   If this is NULL, we will not use any
 *                    ByteBufferPool.  If this is non-NULL, it will be
 *                    treated as the name of the pool class to use.
 *                    For example, you can use
 *                    ELASTIC_BYTE_BUFFER_POOL_CLASS.
 *
 * @return            0 if the ByteBufferPool class was found and
 *                    instantiated;
 *                    -1 plus errno otherwise.
 */
LIBHDFS_EXTERNAL
int hadoopRzOptionsSetByteBufferPool(struct hadoopRzOptions *opts,
                                     const char *className);

/**
 * Free a hadoopRzOptionsFree structure.
 *
 * @param opts        The options structure to free.
 *                    Any associated ByteBufferPool will also be freed.
 */
LIBHDFS_EXTERNAL
void hadoopRzOptionsFree(struct hadoopRzOptions *opts);

/**
 * Perform a byte buffer read.
 * If possible, this will be a zero-copy (mmap) read.
 *
 * @param file       The file to read from.
 * @param opts       An options structure created by hadoopRzOptionsAlloc.
 * @param maxLength  The maximum length to read.  We may read fewer bytes
 *                   than this length.
 *
 * @return           On success, we will return a new hadoopRzBuffer.
 *                   This buffer will continue to be valid and readable
 *                   until it is released by readZeroBufferFree.  Failure to
 *                   release a buffer will lead to a memory leak.
 *                   You can access the data within the hadoopRzBuffer with
 *                   hadoopRzBufferGet.  If you have reached EOF, the data
 *                   within the hadoopRzBuffer will be NULL.  You must still
 *                   free hadoopRzBuffer instances containing NULL.
 *
 *                   On failure, we will return NULL plus an errno code.
 *                   errno = EOPNOTSUPP indicates that we could not do a
 *                   zero-copy read, and there was no ByteBufferPool
 *                   supplied.
 */
LIBHDFS_EXTERNAL
struct hadoopRzBuffer *hadoopReadZero(hdfsFile file,
                                      struct hadoopRzOptions *opts,
                                      int32_t maxLength);

/**
 * Determine the length of the buffer returned from readZero.
 *
 * @param buffer     a buffer returned from readZero.
 * @return           the length of the buffer.
 */
LIBHDFS_EXTERNAL
int32_t hadoopRzBufferLength(const struct hadoopRzBuffer *buffer);

/**
 * Get a pointer to the raw buffer returned from readZero.
 *
 * To find out how many bytes this buffer contains, call
 * hadoopRzBufferLength.
 *
 * @param buffer     a buffer returned from readZero.
 * @return           a pointer to the start of the buffer.  This will be
 *                   NULL when end-of-file has been reached.
 */
LIBHDFS_EXTERNAL
const void *hadoopRzBufferGet(const struct hadoopRzBuffer *buffer);

/**
 * Release a buffer obtained through readZero.
 *
 * @param file       The hdfs stream that created this buffer.  This must be
 *                   the same stream you called hadoopReadZero on.
 * @param buffer     The buffer to release.
 */
LIBHDFS_EXTERNAL
void hadoopRzBufferFree(hdfsFile file, struct hadoopRzBuffer *buffer);

#ifdef __cplusplus
}
#endif

#undef LIBHDFS_EXTERNAL
#endif  // TENSORFLOW_THIRD_PARTY_HADOOP_HDFS_H_

/**
 * vim: ts=4: sw=4: et
 */
