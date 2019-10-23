/*
 * Copyright (C) XMOS Limited 2009 - 2010
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

#ifndef HAVE_QUADFLASHLIB_H_
#define HAVE_QUADFLASHLIB_H_

#include <quadflash.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Backwards compatibility. */
typedef fl_QSPIPorts fl_PortHolderStruct;
//#define SEC_PROT_NONE PROT_TYPE_NONE
//#define SEC_PROT_SR PROT_TYPE_SR
//#define SEC_PROT_SECS PROT_TYPE_SECS

/* Get the current busy status (!0=busy). */
int fl_getBusyStatus();

/* Get the SR value. */
unsigned int fl_getFullStatus();


/* Device level operations.
*/

/* Enable QUAD SPI mode */
int fl_quadEnable();

/* Clear the whole thing. */
int fl_eraseAll();

/* Enable or disable writing to the device. */
int fl_setWritability(int enable);

/* Protect the device as much as possible. */
//int fl_setProtection( int protect );


/* Sector level operations.
*/

/* Get sector layout type. */
fl_SectorLayout fl_getSectorLayoutType();

/* Get the number of sectors. */
int fl_getNumSectors();

/* Get the size (in bytes) of a particular sector. */
int fl_getSectorSize(int sectorNum);

/* Get the address of a particular sector. */
int fl_getSectorAddress(int sectorNum);

/* Erase a sector. */
int fl_eraseSector(int sectorNum);

/* Protect/unprotect a sector. */
//int fl_setSectorProtection( int sectorNum, int protect );


/* Page level operations.
*/

/* Get the number of pages. */
unsigned fl_getNumPages();

/* Program a page at the given address. */
int fl_writePage(unsigned int address, const unsigned char data[]);

/* Read a page at the given address. */
int fl_readPage(unsigned int address, unsigned char data[]);


/* Boot/store level operations
*/

/* Basic information.
*/

/* Sets and returns the size of the boot partition. */
unsigned int fl_setBootPartitionSize( unsigned int s );
unsigned int fl_getBootPartitionSize();

/* Returns the base and the size of the persistant store partition. */
unsigned fl_getDataPartitionBase();
#define fl_getStorePartitionBase() fl_getDataPartitionBase()
#define fl_getStorePartitionSize() fl_getDataPartitionSize()

/* Query and modify the boot partition.
*/

/* Backwards compatibility. */
#define fl_getFirstBootImage(info) fl_getFactoryImage(info)

/* Erase a boot image. */
#ifndef __XC__
int fl_eraseNextBootImage( fl_BootImageInfo* bootImageInfo );
#else
int fl_eraseNextBootImage( fl_BootImageInfo& bootImageInfo );
#endif

/* Add a new boot image after the supplied one. */
#ifndef __XC__
int fl_addBootImage( fl_BootImageInfo* bootImageInfo, unsigned int imageSize, unsigned int (*getData)(void*,unsigned int,unsigned char*), void* userPtr );
#endif

/* Query and modify data in the store partition.
*  Addresses are offsets in the store, not flash addresses.
*/

#define fl_readStore(offset, size, dst) fl_readData(offset, size, dst)
#define fl_writeStore(offset, size, src, buffer) fl_writeData(offset, size, src, buffer)

#ifdef __cplusplus
} //extern "C" 
#endif

#endif /* HAVE_QUADFLASHLIB_H_ */
