/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

package org.tensorflow.lite.support.metadata;

import static org.tensorflow.lite.support.metadata.Preconditions.checkArgument;
import static org.tensorflow.lite.support.metadata.Preconditions.checkNotNull;

import java.io.Closeable;
import java.io.EOFException;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.Charset;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.zip.ZipException;

/**
 * Reads uncompressed files from the TFLite model, a zip file.
 *
 * <p>TODO(b/150237111): add a link to the webpage of MetadataPopulator once it's available.
 *
 * <p>A TFLite model file becomes a zip file when it contains associated files. The associated files
 * can be packed to a TFLite model file using the MetadataPopulator. The associated files are not
 * compressed when being added to the model file.
 *
 * <p>{@link ZipFile} does not support Zip64 format, because TFLite models are much smaller than the
 * size limit for Zip64, which is 4GB.
 */
final class ZipFile implements Closeable {
  /** Maps String to list of ZipEntrys, name -> actual entries. */
  private final Map<String, List<ZipEntry>> nameMap;

  /** The actual data source. */
  private final ByteBufferChannel archive;

  /**
   * Opens the given {@link ByteBufferChannel} for reading, assuming "UTF8" for file names. {@link
   * ZipFile} does not synchronized over the buffer that is passed into it.
   *
   * @param channel the archive
   * @throws IOException if an error occurs while creating this {@link ZipFile}
   * @throws ZipException if the channel is not a zip archive
   * @throws NullPointerException if the archive is null
   */
  public static ZipFile createFrom(ByteBufferChannel channel) throws IOException {
    checkNotNull(channel);
    ZipParser zipParser = new ZipParser(channel);
    Map<String, List<ZipEntry>> nameMap = zipParser.parseEntries();
    return new ZipFile(channel, nameMap);
  }

  @Override
  public void close() {
    archive.close();
  }

  /**
   * Exposes the raw stream of the archive entry.
   *
   * <p>Since the associated files will not be compressed when being packed to the zip file, the raw
   * stream represents the non-compressed files.
   *
   * <p><b>WARNING:</b> The returned {@link InputStream}, is <b>not</b> thread-safe. If multiple
   * threads concurrently reading from the returned {@link InputStream}, it must be synchronized
   * externally.
   *
   * @param name name of the entry to get the stream for
   * @return the raw input stream containing data
   * @throws IllegalArgumentException if the specified file does not exist in the zip file
   */
  public InputStream getRawInputStream(String name) {
    checkArgument(
        nameMap.containsKey(name),
        String.format("The file, %s, does not exist in the zip file.", name));

    List<ZipEntry> entriesWithTheSameName = nameMap.get(name);
    ZipEntry entry = entriesWithTheSameName.get(0);
    long start = entry.getDataOffset();
    long remaining = entry.getSize();
    return new BoundedInputStream(archive, start, remaining);
  }

  private ZipFile(ByteBufferChannel channel, Map<String, List<ZipEntry>> nameMap) {
    archive = channel;
    this.nameMap = nameMap;
  }

  /* Parses a Zip archive and gets the information for each {@link ZipEntry}. */
  private static class ZipParser {
    private final ByteBufferChannel archive;

    // Cached buffers that will only be used locally in the class to reduce garbage collection.
    private final ByteBuffer longBuffer =
        ByteBuffer.allocate(ZipConstants.LONG_BYTE_SIZE).order(ByteOrder.LITTLE_ENDIAN);
    private final ByteBuffer intBuffer =
        ByteBuffer.allocate(ZipConstants.INT_BYTE_SIZE).order(ByteOrder.LITTLE_ENDIAN);
    private final ByteBuffer shortBuffer =
        ByteBuffer.allocate(ZipConstants.SHORT_BYTE_SIZE).order(ByteOrder.LITTLE_ENDIAN);

    private ZipParser(ByteBufferChannel archive) {
      this.archive = archive;
    }

    /**
     * Parses the underlying {@code archive} and returns the information as a list of {@link
     * ZipEntry}.
     */
    private Map<String, List<ZipEntry>> parseEntries() throws IOException {
      List<ZipEntry> entries = parseCentralDirectory();
      return parseLocalFileHeaderData(entries);
    }

    /**
     * Checks if the current position contains a central file header signature, {@link
     * ZipConstants#CENSIG}.
     */
    private boolean foundCentralFileheaderSignature() {
      long signature = (long) getInt();
      return signature == ZipConstants.CENSIG;
    }

    /**
     * Gets the value as a Java int from two bytes starting at the current position of the archive.
     */
    private int getShort() {
      shortBuffer.rewind();
      archive.read(shortBuffer);
      shortBuffer.flip();
      return (int) shortBuffer.getShort();
    }

    /**
     * Gets the value as a Java long from four bytes starting at the current position of the
     * archive.
     */
    private int getInt() {
      intBuffer.rewind();
      archive.read(intBuffer);
      intBuffer.flip();
      return intBuffer.getInt();
    }

    /**
     * Gets the value as a Java long from four bytes starting at the current position of the
     * archive.
     */
    private long getLong() {
      longBuffer.rewind();
      archive.read(longBuffer);
      longBuffer.flip();
      return longBuffer.getLong();
    }

    /**
     * Positions the archive at the start of the central directory.
     *
     * <p>First, it searches for the signature of the "end of central directory record", {@link
     * ZipConstants#ENDSIG}. Position the stream at the start of the "end of central directory
     * record". The zip file are created without archive comments, thus {@link ZipConstants#ENDSIG}
     * should appear exactly at {@link ZipConstants#ENDHDR} from the end of the zip file.
     *
     * <p>Then, parse the "end of central dir record" and position the archive at the start of the
     * central directory.
     */
    private void locateCentralDirectory() throws IOException {
      if (archive.size() < ZipConstants.ENDHDR) {
        throw new ZipException("The archive is not a ZIP archive.");
      }

      // Positions the archive at the start of the "end of central directory record".
      long offsetRecord = archive.size() - ZipConstants.ENDHDR;
      archive.position(offsetRecord);

      // Checks for the signature, {@link ZipConstants#ENDSIG}.
      long endSig = getLong();
      if (endSig != ZipConstants.ENDSIG) {
        throw new ZipException("The archive is not a ZIP archive.");
      }

      // Positions the archive at the “offset of central directory”.
      skipBytes(ZipConstants.ENDOFF - ZipConstants.ENDSUB);
      // Gets the offset to central directory
      long offsetDirectory = getInt();
      // Goes to the central directory.
      archive.position(offsetDirectory);
    }

    /**
     * Reads the central directory of the given archive and populates the internal tables with
     * {@link ZipEntry} instances.
     */
    private List<ZipEntry> parseCentralDirectory() throws IOException {
      /** List of entries in the order they appear inside the central directory. */
      List<ZipEntry> entries = new ArrayList<>();
      locateCentralDirectory();

      while (foundCentralFileheaderSignature()) {
        ZipEntry entry = parseCentralDirectoryEntry();
        entries.add(entry);
      }

      return entries;
    }

    /**
     * Reads an individual entry of the central directory, creats an ZipEntry from it and adds it to
     * the global maps.
     */
    private ZipEntry parseCentralDirectoryEntry() throws IOException {
      // Positions the archive at the "compressed size" and read the value.
      skipBytes(ZipConstants.CENSIZ - ZipConstants.CENVEM);
      long compressSize = getInt();

      // Positions the archive at the "filename length" and read the value.
      skipBytes(ZipConstants.CENNAM - ZipConstants.CENLEN);
      int fileNameLen = getShort();

      // Reads the extra field length and the comment length.
      int extraLen = getShort();
      int commentLen = getShort();

      // Positions the archive at the "local file header offset" and read the value.
      skipBytes(ZipConstants.CENOFF - ZipConstants.CENDSK);
      long localHeaderOffset = getInt();

      // Reads the file name.
      byte[] fileNameBuf = new byte[fileNameLen];
      archive.read(ByteBuffer.wrap(fileNameBuf));
      String fileName = new String(fileNameBuf, Charset.forName("UTF-8"));

      // Skips the extra field and the comment.
      skipBytes(extraLen + commentLen);

      ZipEntry entry = new ZipEntry();
      entry.setSize(compressSize);
      entry.setLocalHeaderOffset(localHeaderOffset);
      entry.setName(fileName);

      return entry;
    }

    /** Walks through all recorded entries and records the offsets for the entry data. */
    private Map<String, List<ZipEntry>> parseLocalFileHeaderData(List<ZipEntry> entries) {
      /** Maps String to list of ZipEntrys, name -> actual entries. */
      Map<String, List<ZipEntry>> nameMap = new LinkedHashMap<>();

      for (ZipEntry entry : entries) {
        long offset = entry.getLocalHeaderOffset();
        archive.position(offset + ZipConstants.LOCNAM);

        // Gets the data offset of this entry.
        int fileNameLen = getShort();
        int extraFieldLen = getShort();
        long dataOffset =
            offset
                + ZipConstants.LOCEXT
                + ZipConstants.SHORT_BYTE_SIZE
                + fileNameLen
                + extraFieldLen;
        entry.setDataOffset(dataOffset);

        // Puts the entry into the nameMap.
        String name = entry.getName();
        List<ZipEntry> entriesWithTheSameName;
        if (nameMap.containsKey(name)) {
          entriesWithTheSameName = nameMap.get(name);
        } else {
          entriesWithTheSameName = new ArrayList<>();
          nameMap.put(name, entriesWithTheSameName);
        }
        entriesWithTheSameName.add(entry);
      }

      return nameMap;
    }

    /** Skips the given number of bytes or throws an EOFException if skipping failed. */
    private void skipBytes(int count) throws IOException {
      long currentPosition = archive.position();
      long newPosition = currentPosition + count;
      if (newPosition > archive.size()) {
        throw new EOFException();
      }
      archive.position(newPosition);
    }
  }

  /** Stores the data offset and the size of an entry in the archive. */
  private static class ZipEntry {

    private String name;
    private long dataOffset = -1;
    private long size = -1;
    private long localHeaderOffset = -1;

    public long getSize() {
      return size;
    }

    public long getDataOffset() {
      return dataOffset;
    }

    public String getName() {
      return name;
    }

    public long getLocalHeaderOffset() {
      return localHeaderOffset;
    }

    public void setSize(long size) {
      this.size = size;
    }

    public void setDataOffset(long dataOffset) {
      this.dataOffset = dataOffset;
    }

    public void setName(String name) {
      this.name = name;
    }

    public void setLocalHeaderOffset(long localHeaderOffset) {
      this.localHeaderOffset = localHeaderOffset;
    }
  }

  /**
   * Various constants for this {@link ZipFile}.
   *
   * <p>Referenced from {@link java.util.zip.ZipConstants}.
   */
  private static class ZipConstants {
    /** length of Java short in bytes. */
    static final int SHORT_BYTE_SIZE = Short.SIZE / 8;

    /** length of Java int in bytes. */
    static final int INT_BYTE_SIZE = Integer.SIZE / 8;

    /** length of Java long in bytes. */
    static final int LONG_BYTE_SIZE = Long.SIZE / 8;

    /*
     * Header signatures
     */
    static final long LOCSIG = 0x04034b50L; // "PK\003\004"
    static final long EXTSIG = 0x08074b50L; // "PK\007\008"
    static final long CENSIG = 0x02014b50L; // "PK\001\002"
    static final long ENDSIG = 0x06054b50L; // "PK\005\006"

    /*
     * Header sizes in bytes (including signatures)
     */
    static final int LOCHDR = 30; // LOC header size
    static final int EXTHDR = 16; // EXT header size
    static final int CENHDR = 46; // CEN header size
    static final int ENDHDR = 22; // END header size

    /*
     * Local file (LOC) header field offsets
     */
    static final int LOCVER = 4; // version needed to extract
    static final int LOCFLG = 6; // general purpose bit flag
    static final int LOCHOW = 8; // compression method
    static final int LOCTIM = 10; // modification time
    static final int LOCCRC = 14; // uncompressed file crc-32 value
    static final int LOCSIZ = 18; // compressed size
    static final int LOCLEN = 22; // uncompressed size
    static final int LOCNAM = 26; // filename length
    static final int LOCEXT = 28; // extra field length

    /*
     * Extra local (EXT) header field offsets
     */
    static final int EXTCRC = 4; // uncompressed file crc-32 value
    static final int EXTSIZ = 8; // compressed size
    static final int EXTLEN = 12; // uncompressed size

    /*
     * Central directory (CEN) header field offsets
     */
    static final int CENVEM = 4; // version made by
    static final int CENVER = 6; // version needed to extract
    static final int CENFLG = 8; // encrypt, decrypt flags
    static final int CENHOW = 10; // compression method
    static final int CENTIM = 12; // modification time
    static final int CENCRC = 16; // uncompressed file crc-32 value
    static final int CENSIZ = 20; // compressed size
    static final int CENLEN = 24; // uncompressed size
    static final int CENNAM = 28; // filename length
    static final int CENEXT = 30; // extra field length
    static final int CENCOM = 32; // comment length
    static final int CENDSK = 34; // disk number start
    static final int CENATT = 36; // internal file attributes
    static final int CENATX = 38; // external file attributes
    static final int CENOFF = 42; // LOC header offset

    /*
     * End of central directory (END) header field offsets
     */
    static final int ENDSUB = 8; // number of entries on this disk
    static final int ENDTOT = 10; // total number of entries
    static final int ENDSIZ = 12; // central directory size in bytes
    static final int ENDOFF = 16; // offset of first CEN header
    static final int ENDCOM = 20; // zip file comment length

    private ZipConstants() {}
  }
}
