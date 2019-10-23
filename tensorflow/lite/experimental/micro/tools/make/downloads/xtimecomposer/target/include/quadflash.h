/*
 * Copyright (C) XMOS Limited 2010
 * 
 * The copyrights, all other intellectual and industrial property rights are
 * retained by XMOS and/or its licensors.
 *
 * The code is provided "AS IS" without a warranty of any kind. XMOS and its
 * licensors disclaim all other warranties, express or implied, including any
 * implied warranty of merchantability/satisfactory quality, fitness for a
 * particular purpose, or non-infringement except to the extent that these
 * disclaimers are held to be legally invalid under applicable law.
 *
 * Version: Community_15.0.0_eng
 */

#ifndef _quadflash_h_
#define _quadflash_h_
#include <xs1_clock.h>

#ifdef __cplusplus
extern "C" {
#endif

/** 
 * \file flash.h
 * \brief SPI Flash device access functions.
 *
 * This file contains functions to access SPI Flash devices.
 */

/* General Operations. */

/** Enum for the type of protection on a flash device. */
typedef enum {
  PROT_TYPE_NONE=0, /**< No protection. */
  PROT_TYPE_SR=1,   /**< Device can be protected by writing the status register. */
  PROT_TYPE_SECS=2,  /**< Device has commands for protecting individual sectors. */
  PROT_TYPE_SR_2X=3,   /**< Device can be protected by writing the status register (need double write). */
} fl_ProtectionType;

/** Enum for the type of sector layout on a flash device. */
typedef enum {
  SECTOR_LAYOUT_REGULAR=0, /**< All sectors the same size. */
  SECTOR_LAYOUT_IRREGULAR  /**< Sectors have different sizes. */
} fl_SectorLayout;

/** Struct containing properties of a flash device. */
typedef struct {
  /** The value to return from fl_getFlashType(). */
  unsigned flashId;
  /** The page size in bytes. */
  unsigned pageSize;
  /** The number of pages. */
  unsigned numPages;
  /** The number of address bytes to send. */
  unsigned char addrSize;
  /** The divider used to generate the SPI clock from the reference clock. */
  unsigned clockDiv;
  /** The command used to read the device ID. */
  unsigned char idCommand;
  /**
   * The number of dummy bytes returned by the device after a read ID command
   * before the ID.
   */
  unsigned char idDummyBytes;
  /** Size in bytes of the device ID. */
  unsigned char idBytes;
  /** Expected device ID. */
  unsigned idValue;
  /** The command use to erase all or part of a sector. */
  unsigned char sectorEraseCommand;
  /**
   * The number of bytes erased by the sector erase command. Set to 0 if the
   * sector erase command erases the entire sector.
   */
  unsigned sectorEraseSize;
   /** The command used to write enable the device. */
  unsigned char writeEnableCommand;
  /** The command used to write disable the device. */
  unsigned char writeDisableCommand;
  /** The type of protection used on the device. */
  fl_ProtectionType protectionType;
  struct {
    struct {
      /** The value to write to the status register to protect the device. */
      unsigned char setProtectedValue;
      /** The value to write to the status register to unprotect the device. */
      unsigned char setUnprotectedValue;
    } statusBits;
    struct {
      /** Command used to protect a sector. */
      unsigned char sectorProtectCommand;
      /** Command used to unprotect a sector. */
      unsigned char sectorUnprotectCommand;
    } commandValues;
  } protection;
  /** Command used to program a page. */
  unsigned int programPageCommand;
  /** Command used to read data. */
  unsigned char readCommand;
  /**
   * The number of dummy bytes returned by the device after a read
   * command before the first byte of data.
   */
  unsigned char readDummyBytes;
  /** The type of sector layout. */
  fl_SectorLayout sectorLayout;
  struct {
    /** The sector size in bytes when the sector layout is regular. */
    unsigned regularSectorSize;
    struct {
      /** The number of sectors when the sector layout is irregular. */
      unsigned char sectorCount; 
      /**
       * Log base 2 of the number in each sector when the sector layout is
       * irregular.
       */
      unsigned char sectorSizesLog2[32];
    } irregularSectorSizes;
  } sectorSizes;
  /** Command used to read the status register. */
  unsigned char readSRCommand;
  /** Command used to write the status register. */
  unsigned int writeSRCommand;
  /** Bit mask for the write in progress bit. */
  unsigned char wipBitMask;
} fl_QuadDeviceSpec;

/** Struct containing ports and clocks used to access a flash device. */
typedef struct {
#ifdef __XC__
  out port qspiCS;               /**< chip select (CS) port. */
  out port qspiSCLK;             /**< Serial clock (SCLK) port. */
  [[bidirectional]]buffered port:32 qspiSIO;      /**< Input/Output port. */
  clock qspiClkblk;              /**< Clock block for use with QSPI ports. */
#else
  unsigned qspiCS;
  unsigned qspiSCLK;
  unsigned qspiSIO;
  unsigned qspiClkblk;
#endif
} fl_QSPIPorts;

/**
 * Connect to a SPI flash device.
 * \param SPI The SPI ports to use.
 * \return 0 on success, non zero on failure.
 */
#ifdef __XC__
int fl_connect(fl_QSPIPorts& QSPI);
#else
int fl_connect(fl_QSPIPorts* QSPI);
#endif

/**
 * Connect to a SPI flash device. The flash type must match one of flash
 * specifications in the \a spec array. A pointer to the spec \a array is
 * captured and used during subsequent flash library calls. The \a spec
 * array should not be modified and must remain in scope until after the flash
 * device is disconnected.
 * \param SPI The SPI ports to use.
 * \param spec An array of flash device specifications to use.
 * \param n The length of the \a spec array
 * \return 0 on success, non zero on failure.
 */
#ifdef __XC__
int fl_connectToDevice(fl_QSPIPorts &QSPI, const fl_QuadDeviceSpec spec[], unsigned n);
#else
int fl_connectToDevice(fl_QSPIPorts *QSPI, const fl_QuadDeviceSpec spec[], unsigned n);
#endif

/**
 * Allows the SPI clock divider to be changed from that defined within the
 * fl_QuadDeviceSpec files for the connected device.
 * Must be called after fl_connect or fl_connectToDevice.
 * \param div New SPI clock divider value to use.
 * \return 0 on success, non zero on failure.
 */
int fl_dividerOverride(int div);

/**
 * Returns the type of the flash device. If fl_connectToDevice() was used to
 * connect to the device the value of the flashId member of the device specification is
 * returned. If fl_connect() was used to connect to the device the return value
 * is determined by possible values of the fl_FlashId type.
 * If not connected to a flash device the result is undefined.
 * \return The flash type.
 */
int fl_getFlashType();

/**
 * Returns the capacity in bytes of the connected SPI device.
 * If not connected to a flash device the result is undefined.
 * \return The flash size.
 */
unsigned fl_getFlashSize();

/**
 * Returns the actual manufacturer spi id 
 * as read from the connected flash device
 */
#ifdef __XC__
void fl_getSpiId(fl_QSPIPorts & QSPI, unsigned id_command, unsigned char destination[]);
#else
void fl_getSpiId(fl_QSPIPorts * QSPI, unsigned id_command, unsigned char destination[]);
#endif

/**
 * Returns the status register 
 * as read from the flash device
 */
#ifdef __XC__
unsigned fl_getSpiStatus(fl_QSPIPorts & QSPI, unsigned status_command);
#else
unsigned fl_getSpiStatus(fl_QSPIPorts * QSPI, unsigned status_command);
#endif

/**
 * Close the connection to the SPI device.
 * \return 0 on success, non zero on failure.
 */
int fl_disconnect();

/* Data partition functions. */

/**
 * Retrieve the application identifier number from the flash.
 * \return -1 on error
 */
int fl_getFlashIdNum();

/**
 * Retrieve the application identifier string from the flash.
 * The string is copied into the supplied buffer using no more than
 * the specified number of bytes. If the buffer is too small for
 * the full string it is truncated but not null-terminated.
 * \return -1 on error or the string length on success
 */
int fl_getFlashIdStr( char buf[], int maxlen );


/* Boot partition functions. */

/** Struct describing a bootable image. */
typedef struct {
  unsigned startAddress; /**< The address of the start of the image. */
  unsigned size; /**< The size in bytes of the image. */
  unsigned version; /**< The image version. */
  int factory; /**< 1 if the image is the factory image, 0 otherwise. */
} fl_BootImageInfo;

/**
 * Provides information about the factory boot image.
 * \param bootImageInfo Reference to structure that should be written.
 * \return 0 on success, non zero if no factory image cannot be found.
 * */
#ifndef __XC__
int fl_getFactoryImage(fl_BootImageInfo* bootImageInfo);
#else
int fl_getFactoryImage(fl_BootImageInfo& bootImageInfo);
#endif

/**
 * Provides information about the next upgrade image after the specified
 * image.
 * \param bootImageInfo Reference to information on the current image.
 *                      The structure is updated with information on the next
 *                      image
 * \return 0 on success, non zero if no subsequent image cannot be found.
 */
#ifndef __XC__
int fl_getNextBootImage(fl_BootImageInfo* bootImageInfo);
#else
int fl_getNextBootImage(fl_BootImageInfo& bootImageInfo);
#endif

/**
 * Returns the version number of the specified image.
 * \param bootImageInfo Reference to the image to query.
 * \return The version number of the image.
 */
#ifndef __XC__
unsigned fl_getImageVersion(fl_BootImageInfo* bootImageInfo);
#else
unsigned fl_getImageVersion(fl_BootImageInfo& bootImageInfo);
#endif

/**
 * Prepare the SPI device for adding an image after the specified image.
 * Attempting to write into the data partition or another upgrade image is
 * invalid. The return value indicates whether the operation completed.
 * A positive return value indicates the preparation is not complete and
 * the function must be called again.
 * \param bootImageInfo Reference to the previous image.
 * \param maxsize The maximum size in bytes of the upgrade image.
 * \param padding The amount of padding in bytes to allow after the previous
 *                image.
 * \return 0 on success, 1 if the operation is not complete and negative
 *         values on failure.
 */
#ifndef __XC__
int fl_startImageAdd(fl_BootImageInfo *bootImageInfo, unsigned maxsize,
                     unsigned padding);
#else
int fl_startImageAdd(fl_BootImageInfo &bootImageInfo, unsigned maxsize,
                     unsigned padding);
#endif

/**
 * Prepare the SPI device for adding an image into a space at the specified
 * address.  Attempting to write into the data partition or over an image is
 * invalid. The return value indicates whether the operation completed.
 * A positive return value indicates the preparation is not complete and
 * the function must be called again.
 * \param offset  Offset of this image write from the base of the first sector
 *                after the factory image.
 * \param maxsize The maximum size in bytes of the upgrade image.
 * \return 0 on success, 1 if the operation is not complete and negative
 *         values on failure.
 */
#ifndef __XC__
int fl_startImageAddAt( unsigned offset, unsigned maxsize);
#else
int fl_startImageAddAt( unsigned offset, unsigned maxsize);
#endif

/**
 * Prepare the SPI device for replacing the specified image. The image can no
 * longer be assumed to exist after this call.
 * Attempting to write into the data partition or another upgrade image is
 * invalid. The return value indicates whether the operation completed.
 * A positive return value indicates the preparation is not complete and
 * the function must be called again.
 * \param bootImageInfo Reference to the image to replace.
 * \param maxsize The maximum size in bytes of the upgrade image.
 * \return 0 on success, 1 if the operation is not complete and negative
 *         values on failure.
 */
#ifndef __XC__
int fl_startImageReplace(fl_BootImageInfo *bootImageInfo, unsigned maxsize);
#else
int fl_startImageReplace(fl_BootImageInfo &bootImageInfo, unsigned maxsize);
#endif

/**
 * Waits until the SPI device is ready to write the next page and outputs
 * the next page of data to the device. Attempting to write past the maximum
 * size passed to fl_startImageUpdate or fl_startImageAdd is invalid.
 * \param page Data to write.
 * \return 0 on success, non zero on failure.
 */
int fl_writeImagePage(const unsigned char page[]);

/**
 * Waits until the SPI device has written the last page of data to its memory.
 * \return 0 on success, non zero on failure.
 */
int fl_endWriteImage(void);

/**
 * Deletes the specified image by erasing the first sector of the image.
 * \param bootImageInfo The image to delete.
 * \return 0 on success, non zero on failure.
 */
#ifndef __XC__
int fl_deleteImage(fl_BootImageInfo *bootImageInfo);
#else
int fl_deleteImage(fl_BootImageInfo &bootImageInfo);
#endif

/**
 * Prepare the SPI device for erasing the first sector of the image.
 * \param bootImageInfo The image to delete.
 * \return 0 on success, 1 if the operation is not complete and negative
 *         values on failure.
 */
#ifndef __XC__
int fl_startDeleteImage(fl_BootImageInfo *bootImageInfo);
#else
int fl_startDeleteImage(fl_BootImageInfo &bootImageInfo);
#endif

/**
 * Prepare for reading the contents of the specified upgrade image.
 * \return 0 on success, non zero on failure.
 */
#ifndef __XC__
int fl_startImageRead(fl_BootImageInfo *bootImageInfo);
#else
int fl_startImageRead(fl_BootImageInfo &bootImageInfo);
#endif

/**
 * Read the next page of data from the SPI device.
 * \param page Array to fill with image data.
 * \return 0 on success, non zero on failure.
 */
int fl_readImagePage(unsigned char page[]);

/**
 * fl_readImageRead is deprecated - use fl_readImagePage instead.
 */
#ifdef __XC__
#define fl_readImageRead(page) _Pragma("warning \"fl_readImageRead is deprecated, use fl_readImagePage instead\"") fl_readImagePage(page)
#else
__attribute__((deprecated)) static inline int fl_readImageRead(unsigned char page[])
{
  return fl_readImagePage(page);
}
#endif


/* Data partition functions. */

/**
 * Returns the size of the data partition in bytes.
 * \return The size in bytes.
 */
unsigned fl_getDataPartitionSize(void);


/**
 * Reads data from the data partition.
 * \param offset The offset from the start of the data partition in bytes.
 * \param size The number of bytes to read.
 * \param dst Array to fill with data.
 * \return 0 on success, non zero on failure.
 */
int fl_readData(unsigned offset, unsigned size, unsigned char dst[]);

/**
 * Returns the buffer size required for fl_writeData() with the given
 * parameters.
 * \param offset The offset from the start of the data partition in bytes.
 * \param size The number of bytes to write.
 * \return Buffer size on success, 0 if writing is not possible.
 */
unsigned fl_getWriteScratchSize(unsigned offset, unsigned size);

/**
 * Write data to the data partition. If not writing a whole number of sectors
 * a scratch buffer is required. The minimum size of the required buffer can be
 * determined by calling fl_getWriteScratchSize().
 * \param offset The offset from the start of the data partition in bytes.
 * \param size The number of bytes to write.
 * \param src The data to write.
 * \param buffer A scratch buffer.
 * \return 0 on success, non zero on failure.
 */
int fl_writeData(unsigned offset, unsigned size, const unsigned char src[],
                 unsigned char buffer[]);

/* Page level functions. */

/**
 * Returns the page size in bytes of the connected flash device.
 * \return Page size.
 */
unsigned fl_getPageSize(void);

/**
 * Returns the number of pages in the data partition.
 * \return Number of pages.
 */
unsigned fl_getNumDataPages(void);

/**
 * Write to the nth page in the data partition. The sector containing the page
 * must have been erased previously.
 * \param n The page number to write.
 * \param src The data to write.
 * \return 0 on success, non zero on failure.
 */
int fl_writeDataPage(unsigned n, const unsigned char src[]);

/**
 * Read the nth page in the data partition.
 * \param n The page number to read.
 * \param dst The array to fill with data.
 * \return 0 on success, non zero on failure.
 */
int fl_readDataPage(unsigned n, unsigned char dst[]);

/* Sector level functions. */

/**
 * Returns the number of sectors in the data partition.
 * \return The number of sectors.
 */
unsigned fl_getNumDataSectors(void);

/**
 * Returns the size in bytes of the n-th sector in the data partition.
 * \param n The sector number.
 * \return The size in bytes.
 */
unsigned fl_getDataSectorSize(unsigned n);

/**
 * Erases the n-th sector in the data partition.
 * \param n The sector number.
 * \return 0 on success, non-zero on failure.
 */
int fl_eraseDataSector(unsigned n);

/**
 * Erases all sectors in the the data partition.
 * \return 0 on success, non-zero on failure.
 */
int fl_eraseAllDataSectors(void);

/**
 * Enum of built-in flash device types. These values are used for the return
 * value of fl_getFlashType() when fl_connect() is used to connect to a device.
 */
#include "QuadSpecEnum.h"

/**
 * The standard supported devices as user accessible macros.
 */
#include "QuadSpecMacros.h"

#ifdef __cplusplus
} //extern "C" 
#endif

#endif /* _quadflash_h_ */
