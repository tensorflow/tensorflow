//------------------------------------------------------------------------------
// picojpeg.c v1.1 - Public domain, Rich Geldreich <richgel99@gmail.com>
// Nov. 27, 2010 - Initial release
// Feb. 9, 2013 - Added H1V2/H2V1 support, cleaned up macros, signed shift fixes 
// Also integrated and tested changes from Chris Phoenix <cphoenix@gmail.com>.
//------------------------------------------------------------------------------
#include "picojpeg.h"
//------------------------------------------------------------------------------
// Set to 1 if right shifts on signed ints are always unsigned (logical) shifts
// When 1, arithmetic right shifts will be emulated by using a logical shift
// with special case code to ensure the sign bit is replicated.
#define PJPG_RIGHT_SHIFT_IS_ALWAYS_UNSIGNED 1

// Define PJPG_INLINE to "inline" if your C compiler supports explicit inlining
#define PJPG_INLINE
//------------------------------------------------------------------------------
typedef unsigned char   uint8;
typedef unsigned short  uint16;
typedef signed char     int8;
typedef signed short    int16;
//------------------------------------------------------------------------------
#if PJPG_RIGHT_SHIFT_IS_ALWAYS_UNSIGNED
static int16 replicateSignBit16(int8 n)
{
   switch (n)
   {
      case 0:  return 0x0000;
      case 1:  return 0x8000;
      case 2:  return 0xC000;
      case 3:  return 0xE000;
      case 4:  return 0xF000;
      case 5:  return 0xF800;
      case 6:  return 0xFC00;
      case 7:  return 0xFE00;
      case 8:  return 0xFF00;
      case 9:  return 0xFF80;
      case 10: return 0xFFC0;
      case 11: return 0xFFE0;
      case 12: return 0xFFF0; 
      case 13: return 0xFFF8;
      case 14: return 0xFFFC;
      case 15: return 0xFFFE;
      default: return 0xFFFF;
   }
}
static PJPG_INLINE int16 arithmeticRightShiftN16(int16 x, int8 n) 
{
   int16 r = (uint16)x >> (uint8)n;
   if (x < 0)
      r |= replicateSignBit16(n);
   return r;
}
static PJPG_INLINE long arithmeticRightShift8L(long x) 
{
   long r = (unsigned long)x >> 8U;
   if (x < 0)
      r |= ~(~(unsigned long)0U >> 8U);
   return r;
}
#define PJPG_ARITH_SHIFT_RIGHT_N_16(x, n) arithmeticRightShiftN16(x, n)
#define PJPG_ARITH_SHIFT_RIGHT_8_L(x) arithmeticRightShift8L(x)
#else
#define PJPG_ARITH_SHIFT_RIGHT_N_16(x, n) ((x) >> (n))
#define PJPG_ARITH_SHIFT_RIGHT_8_L(x) ((x) >> 8)
#endif
//------------------------------------------------------------------------------
// Change as needed - the PJPG_MAX_WIDTH/PJPG_MAX_HEIGHT checks are only present
// to quickly detect bogus files.
#define PJPG_MAX_WIDTH 16384
#define PJPG_MAX_HEIGHT 16384
#define PJPG_MAXCOMPSINSCAN 3
//------------------------------------------------------------------------------
typedef enum
{
   M_SOF0  = 0xC0,
   M_SOF1  = 0xC1,
   M_SOF2  = 0xC2,
   M_SOF3  = 0xC3,

   M_SOF5  = 0xC5,
   M_SOF6  = 0xC6,
   M_SOF7  = 0xC7,

   M_JPG   = 0xC8,
   M_SOF9  = 0xC9,
   M_SOF10 = 0xCA,
   M_SOF11 = 0xCB,

   M_SOF13 = 0xCD,
   M_SOF14 = 0xCE,
   M_SOF15 = 0xCF,

   M_DHT   = 0xC4,

   M_DAC   = 0xCC,

   M_RST0  = 0xD0,
   M_RST1  = 0xD1,
   M_RST2  = 0xD2,
   M_RST3  = 0xD3,
   M_RST4  = 0xD4,
   M_RST5  = 0xD5,
   M_RST6  = 0xD6,
   M_RST7  = 0xD7,

   M_SOI   = 0xD8,
   M_EOI   = 0xD9,
   M_SOS   = 0xDA,
   M_DQT   = 0xDB,
   M_DNL   = 0xDC,
   M_DRI   = 0xDD,
   M_DHP   = 0xDE,
   M_EXP   = 0xDF,

   M_APP0  = 0xE0,
   M_APP15 = 0xEF,

   M_JPG0  = 0xF0,
   M_JPG13 = 0xFD,
   M_COM   = 0xFE,

   M_TEM   = 0x01,

   M_ERROR = 0x100,
   
   RST0    = 0xD0
} JPEG_MARKER;
//------------------------------------------------------------------------------
static const int8 ZAG[] = 
{
   0,  1,  8, 16,  9,  2,  3, 10,
   17, 24, 32, 25, 18, 11,  4,  5,
   12, 19, 26, 33, 40, 48, 41, 34,
   27, 20, 13,  6,  7, 14, 21, 28,
   35, 42, 49, 56, 57, 50, 43, 36,
   29, 22, 15, 23, 30, 37, 44, 51,
   58, 59, 52, 45, 38, 31, 39, 46,
   53, 60, 61, 54, 47, 55, 62, 63,
};
//------------------------------------------------------------------------------
// 128 bytes
static int16 gCoeffBuf[8*8];

// 8*8*4 bytes * 3 = 768
static uint8 gMCUBufR[256];
static uint8 gMCUBufG[256];
static uint8 gMCUBufB[256];

// 256 bytes
static int16 gQuant0[8*8];
static int16 gQuant1[8*8];

// 6 bytes
static int16 gLastDC[3];

typedef struct HuffTableT
{
   uint16 mMinCode[16];
   uint16 mMaxCode[16];
   uint8 mValPtr[16];
} HuffTable;

// DC - 192
static HuffTable gHuffTab0;

static uint8 gHuffVal0[16];

static HuffTable gHuffTab1;
static uint8 gHuffVal1[16];

// AC - 672
static HuffTable gHuffTab2;
static uint8 gHuffVal2[256];

static HuffTable gHuffTab3;
static uint8 gHuffVal3[256];

static uint8 gValidHuffTables;
static uint8 gValidQuantTables;

static uint8 gTemFlag;
#define PJPG_MAX_IN_BUF_SIZE 256
static uint8 gInBuf[PJPG_MAX_IN_BUF_SIZE];
static uint8 gInBufOfs;
static uint8 gInBufLeft;

static uint16 gBitBuf;
static uint8 gBitsLeft;
//------------------------------------------------------------------------------
static uint16 gImageXSize;
static uint16 gImageYSize;
static uint8 gCompsInFrame;
static uint8 gCompIdent[3];
static uint8 gCompHSamp[3];
static uint8 gCompVSamp[3];
static uint8 gCompQuant[3];

static uint16 gRestartInterval;
static uint16 gNextRestartNum;
static uint16 gRestartsLeft;

static uint8 gCompsInScan;
static uint8 gCompList[3];
static uint8 gCompDCTab[3]; // 0,1
static uint8 gCompACTab[3]; // 0,1

static pjpeg_scan_type_t gScanType;

static uint8 gMaxBlocksPerMCU;
static uint8 gMaxMCUXSize;
static uint8 gMaxMCUYSize;
static uint16 gMaxMCUSPerRow;
static uint16 gMaxMCUSPerCol;
static uint16 gNumMCUSRemaining;
static uint8 gMCUOrg[6];

static pjpeg_need_bytes_callback_t g_pNeedBytesCallback;
static void *g_pCallback_data;
static uint8 gCallbackStatus;
static uint8 gReduce;
//------------------------------------------------------------------------------
static void fillInBuf(void)
{
   unsigned char status;

   // Reserve a few bytes at the beginning of the buffer for putting back ("stuffing") chars.
   gInBufOfs = 4;
   gInBufLeft = 0;

   status = (*g_pNeedBytesCallback)(gInBuf + gInBufOfs, PJPG_MAX_IN_BUF_SIZE - gInBufOfs, &gInBufLeft, g_pCallback_data);
   if (status)
   {
      // The user provided need bytes callback has indicated an error, so record the error and continue trying to decode.
      // The highest level pjpeg entrypoints will catch the error and return the non-zero status.
      gCallbackStatus = status;
   }
}   
//------------------------------------------------------------------------------
static PJPG_INLINE uint8 getChar(void)
{
   if (!gInBufLeft)
   {
      fillInBuf();
      if (!gInBufLeft)
      {
         gTemFlag = ~gTemFlag;
         return gTemFlag ? 0xFF : 0xD9;
      } 
   }
   
   gInBufLeft--;
   return gInBuf[gInBufOfs++];
}
//------------------------------------------------------------------------------
static PJPG_INLINE void stuffChar(uint8 i)
{
   gInBufOfs--;
   gInBuf[gInBufOfs] = i;
   gInBufLeft++;
}
//------------------------------------------------------------------------------
static PJPG_INLINE uint8 getOctet(uint8 FFCheck)
{
   uint8 c = getChar();
      
   if ((FFCheck) && (c == 0xFF))
   {
      uint8 n = getChar();

      if (n)
      {
         stuffChar(n);
         stuffChar(0xFF);
      }
   }

   return c;
}
//------------------------------------------------------------------------------
static uint16 getBits(uint8 numBits, uint8 FFCheck)
{
   uint8 origBits = numBits;
   uint16 ret = gBitBuf;
   
   if (numBits > 8)
   {
      numBits -= 8;
      
      gBitBuf <<= gBitsLeft;
      
      gBitBuf |= getOctet(FFCheck);
      
      gBitBuf <<= (8 - gBitsLeft);
      
      ret = (ret & 0xFF00) | (gBitBuf >> 8);
   }
      
   if (gBitsLeft < numBits)
   {
      gBitBuf <<= gBitsLeft;
      
      gBitBuf |= getOctet(FFCheck);
      
      gBitBuf <<= (numBits - gBitsLeft);
                        
      gBitsLeft = 8 - (numBits - gBitsLeft);
   }
   else
   {
      gBitsLeft = (uint8)(gBitsLeft - numBits);
      gBitBuf <<= numBits;
   }
   
   return ret >> (16 - origBits);
}
//------------------------------------------------------------------------------
static PJPG_INLINE uint16 getBits1(uint8 numBits)
{
   return getBits(numBits, 0);
}
//------------------------------------------------------------------------------
static PJPG_INLINE uint16 getBits2(uint8 numBits)
{
   return getBits(numBits, 1);
}
//------------------------------------------------------------------------------
static PJPG_INLINE uint8 getBit(void)
{
   uint8 ret = 0;
   if (gBitBuf & 0x8000) 
      ret = 1;
   
   if (!gBitsLeft)
   {
      gBitBuf |= getOctet(1);

      gBitsLeft += 8;
   }
   
   gBitsLeft--;
   gBitBuf <<= 1;
   
   return ret;
}
//------------------------------------------------------------------------------
static uint16 getExtendTest(uint8 i)
{
   switch (i)
   {
      case 0: return 0;
      case 1: return 0x0001;
      case 2: return 0x0002;
      case 3: return 0x0004;
      case 4: return 0x0008;
      case 5: return 0x0010; 
      case 6: return 0x0020;
      case 7: return 0x0040;
      case 8:  return 0x0080;
      case 9:  return 0x0100;
      case 10: return 0x0200;
      case 11: return 0x0400;
      case 12: return 0x0800;
      case 13: return 0x1000;
      case 14: return 0x2000; 
      case 15: return 0x4000;
      default: return 0;
   }      
}
//------------------------------------------------------------------------------
static int16 getExtendOffset(uint8 i)
{ 
   switch (i)
   {
      case 0: return 0;
      case 1: return ((-1)<<1) + 1; 
      case 2: return ((-1)<<2) + 1; 
      case 3: return ((-1)<<3) + 1; 
      case 4: return ((-1)<<4) + 1; 
      case 5: return ((-1)<<5) + 1; 
      case 6: return ((-1)<<6) + 1; 
      case 7: return ((-1)<<7) + 1; 
      case 8: return ((-1)<<8) + 1; 
      case 9: return ((-1)<<9) + 1;
      case 10: return ((-1)<<10) + 1; 
      case 11: return ((-1)<<11) + 1; 
      case 12: return ((-1)<<12) + 1; 
      case 13: return ((-1)<<13) + 1; 
      case 14: return ((-1)<<14) + 1; 
      case 15: return ((-1)<<15) + 1;
      default: return 0;
   }
};
//------------------------------------------------------------------------------
static PJPG_INLINE int16 huffExtend(uint16 x, uint8 s)
{
   return ((x < getExtendTest(s)) ? ((int16)x + getExtendOffset(s)) : (int16)x);
}
//------------------------------------------------------------------------------
static PJPG_INLINE uint8 huffDecode(const HuffTable* pHuffTable, const uint8* pHuffVal)
{
   uint8 i = 0;
   uint8 j;
   uint16 code = getBit();

   // This func only reads a bit at a time, which on modern CPU's is not terribly efficient.
   // But on microcontrollers without strong integer shifting support this seems like a 
   // more reasonable approach.
   for ( ; ; )
   {
      uint16 maxCode;

      if (i == 16)
         return 0;

      maxCode = pHuffTable->mMaxCode[i];
      if ((code <= maxCode) && (maxCode != 0xFFFF))
         break;

      i++;
      code <<= 1;
      code |= getBit();
   }

   j = pHuffTable->mValPtr[i];
   j = (uint8)(j + (code - pHuffTable->mMinCode[i]));

   return pHuffVal[j];
}
//------------------------------------------------------------------------------
static void huffCreate(const uint8* pBits, HuffTable* pHuffTable)
{
   uint8 i = 0;
   uint8 j = 0;

   uint16 code = 0;
      
   for ( ; ; )
   {
      uint8 num = pBits[i];
      
      if (!num)
      {
         pHuffTable->mMinCode[i] = 0x0000;
         pHuffTable->mMaxCode[i] = 0xFFFF;
         pHuffTable->mValPtr[i] = 0;
      }
      else
      {
         pHuffTable->mMinCode[i] = code;
         pHuffTable->mMaxCode[i] = code + num - 1;
         pHuffTable->mValPtr[i] = j;
         
         j = (uint8)(j + num);
         
         code = (uint16)(code + num);
      }
      
      code <<= 1;
      
      i++;
      if (i > 15)
         break;
   }
}
//------------------------------------------------------------------------------
static HuffTable* getHuffTable(uint8 index)
{
   // 0-1 = DC
   // 2-3 = AC
   switch (index)
   {
      case 0: return &gHuffTab0;
      case 1: return &gHuffTab1;
      case 2: return &gHuffTab2;
      case 3: return &gHuffTab3;
      default: return 0;
   }
}
//------------------------------------------------------------------------------
static uint8* getHuffVal(uint8 index)
{
   // 0-1 = DC
   // 2-3 = AC
   switch (index)
   {
      case 0: return gHuffVal0;
      case 1: return gHuffVal1;
      case 2: return gHuffVal2;
      case 3: return gHuffVal3;
      default: return 0;
   }
}
//------------------------------------------------------------------------------
static uint16 getMaxHuffCodes(uint8 index)
{
   return (index < 2) ? 12 : 255;
}
//------------------------------------------------------------------------------
static uint8 readDHTMarker(void)
{
   uint8 bits[16];
   uint16 left = getBits1(16);

   if (left < 2)
      return PJPG_BAD_DHT_MARKER;

   left -= 2;

   while (left)
   {
      uint8 i, tableIndex, index;
      uint8* pHuffVal;
      HuffTable* pHuffTable;
      uint16 count, totalRead;
            
      index = (uint8)getBits1(8);
      
      if ( ((index & 0xF) > 1) || ((index & 0xF0) > 0x10) )
         return PJPG_BAD_DHT_INDEX;
      
      tableIndex = ((index >> 3) & 2) + (index & 1);
      
      pHuffTable = getHuffTable(tableIndex);
      pHuffVal = getHuffVal(tableIndex);
      
      gValidHuffTables |= (1 << tableIndex);
            
      count = 0;
      for (i = 0; i <= 15; i++)
      {
         uint8 n = (uint8)getBits1(8);
         bits[i] = n;
         count = (uint16)(count + n);
      }
      
      if (count > getMaxHuffCodes(tableIndex))
         return PJPG_BAD_DHT_COUNTS;

      for (i = 0; i < count; i++)
         pHuffVal[i] = (uint8)getBits1(8);

      totalRead = 1 + 16 + count;

      if (left < totalRead)
         return PJPG_BAD_DHT_MARKER;

      left = (uint16)(left - totalRead);

      huffCreate(bits, pHuffTable);
   }
      
   return 0;
}
//------------------------------------------------------------------------------
static void createWinogradQuant(int16* pQuant);

static uint8 readDQTMarker(void)
{
   uint16 left = getBits1(16);

   if (left < 2)
      return PJPG_BAD_DQT_MARKER;

   left -= 2;

   while (left)
   {
      uint8 i;
      uint8 n = (uint8)getBits1(8);
      uint8 prec = n >> 4;
      uint16 totalRead;

      n &= 0x0F;

      if (n > 1)
         return PJPG_BAD_DQT_TABLE;

      gValidQuantTables |= (n ? 2 : 1);         

      // read quantization entries, in zag order
      for (i = 0; i < 64; i++)
      {
         uint16 temp = getBits1(8);

         if (prec)
            temp = (temp << 8) + getBits1(8);

         if (n)
            gQuant1[i] = (int16)temp;            
         else
            gQuant0[i] = (int16)temp;            
      }
      
      createWinogradQuant(n ? gQuant1 : gQuant0);

      totalRead = 64 + 1;

      if (prec)
         totalRead += 64;

      if (left < totalRead)
         return PJPG_BAD_DQT_LENGTH;

      left = (uint16)(left - totalRead);
   }
   
   return 0;
}
//------------------------------------------------------------------------------
static uint8 readSOFMarker(void)
{
   uint8 i;
   uint16 left = getBits1(16);

   if (getBits1(8) != 8)   
      return PJPG_BAD_PRECISION;

   gImageYSize = getBits1(16);

   if ((!gImageYSize) || (gImageYSize > PJPG_MAX_HEIGHT))
      return PJPG_BAD_HEIGHT;

   gImageXSize = getBits1(16);

   if ((!gImageXSize) || (gImageXSize > PJPG_MAX_WIDTH))
      return PJPG_BAD_WIDTH;

   gCompsInFrame = (uint8)getBits1(8);

   if (gCompsInFrame > 3)
      return PJPG_TOO_MANY_COMPONENTS;

   if (left != (gCompsInFrame + gCompsInFrame + gCompsInFrame + 8))
      return PJPG_BAD_SOF_LENGTH;
   
   for (i = 0; i < gCompsInFrame; i++)
   {
      gCompIdent[i] = (uint8)getBits1(8);
      gCompHSamp[i] = (uint8)getBits1(4);
      gCompVSamp[i] = (uint8)getBits1(4);
      gCompQuant[i] = (uint8)getBits1(8);
      
      if (gCompQuant[i] > 1)
         return PJPG_UNSUPPORTED_QUANT_TABLE;
   }
   
   return 0;
}
//------------------------------------------------------------------------------
// Used to skip unrecognized markers.
static uint8 skipVariableMarker(void)
{
   uint16 left = getBits1(16);

   if (left < 2)
      return PJPG_BAD_VARIABLE_MARKER;

   left -= 2;

   while (left)
   {
      getBits1(8);
      left--;
   }
   
   return 0;
}
//------------------------------------------------------------------------------
// Read a define restart interval (DRI) marker.
static uint8 readDRIMarker(void)
{
   if (getBits1(16) != 4)
      return PJPG_BAD_DRI_LENGTH;

   gRestartInterval = getBits1(16);
   
   return 0;
}
//------------------------------------------------------------------------------
// Read a start of scan (SOS) marker.
static uint8 readSOSMarker(void)
{
   uint8 i;
   uint16 left = getBits1(16);
   uint8 spectral_start, spectral_end, successive_high, successive_low;

   gCompsInScan = (uint8)getBits1(8);

   left -= 3;

   if ( (left != (gCompsInScan + gCompsInScan + 3)) || (gCompsInScan < 1) || (gCompsInScan > PJPG_MAXCOMPSINSCAN) )
      return PJPG_BAD_SOS_LENGTH;
   
   for (i = 0; i < gCompsInScan; i++)
   {
      uint8 cc = (uint8)getBits1(8);
      uint8 c = (uint8)getBits1(8);
      uint8 ci;
      
      left -= 2;
     
      for (ci = 0; ci < gCompsInFrame; ci++)
         if (cc == gCompIdent[ci])
            break;

      if (ci >= gCompsInFrame)
         return PJPG_BAD_SOS_COMP_ID;

      gCompList[i]    = ci;
      gCompDCTab[ci] = (c >> 4) & 15;
      gCompACTab[ci] = (c & 15);
   }

   spectral_start  = (uint8)getBits1(8);
   spectral_end    = (uint8)getBits1(8);
   successive_high = (uint8)getBits1(4);
   successive_low  = (uint8)getBits1(4);

   left -= 3;

   while (left)                  
   {
      getBits1(8);
      left--;
   }
   
   return 0;
}
//------------------------------------------------------------------------------
static uint8 nextMarker(void)
{
   uint8 c;
   uint8 bytes = 0;

   do
   {
      do
      {
         bytes++;

         c = (uint8)getBits1(8);

      } while (c != 0xFF);

      do
      {
         c = (uint8)getBits1(8);

      } while (c == 0xFF);

   } while (c == 0);

   // If bytes > 0 here, there where extra bytes before the marker (not good).

   return c;
}
//------------------------------------------------------------------------------
// Process markers. Returns when an SOFx, SOI, EOI, or SOS marker is
// encountered.
static uint8 processMarkers(uint8* pMarker)
{
   for ( ; ; )
   {
      uint8 c = nextMarker();

      switch (c)
      {
         case M_SOF0:
         case M_SOF1:
         case M_SOF2:
         case M_SOF3:
         case M_SOF5:
         case M_SOF6:
         case M_SOF7:
         //      case M_JPG:
         case M_SOF9:
         case M_SOF10:
         case M_SOF11:
         case M_SOF13:
         case M_SOF14:
         case M_SOF15:
         case M_SOI:
         case M_EOI:
         case M_SOS:
         {
            *pMarker = c;
            return 0;
         }
         case M_DHT:
         {
            readDHTMarker();
            break;
         }
         // Sorry, no arithmetic support at this time. Dumb patents!
         case M_DAC:
         {
            return PJPG_NO_ARITHMITIC_SUPPORT;
         }
         case M_DQT:
         {
            readDQTMarker();
            break;
         }
         case M_DRI:
         {
            readDRIMarker();
            break;
         }
         //case M_APP0:  /* no need to read the JFIF marker */

         case M_JPG:
         case M_RST0:    /* no parameters */
         case M_RST1:
         case M_RST2:
         case M_RST3:
         case M_RST4:
         case M_RST5:
         case M_RST6:
         case M_RST7:
         case M_TEM:
         {
            return PJPG_UNEXPECTED_MARKER;
         }
         default:    /* must be DNL, DHP, EXP, APPn, JPGn, COM, or RESn or APP0 */
         {
            skipVariableMarker();
            break;
         }
      }
   }
//   return 0;
}
//------------------------------------------------------------------------------
// Finds the start of image (SOI) marker.
static uint8 locateSOIMarker(void)
{
   uint16 bytesleft;
   
   uint8 lastchar = (uint8)getBits1(8);

   uint8 thischar = (uint8)getBits1(8);

   /* ok if it's a normal JPEG file without a special header */

   if ((lastchar == 0xFF) && (thischar == M_SOI))
      return 0;

   bytesleft = 4096; //512;

   for ( ; ; )
   {
      if (--bytesleft == 0)
         return PJPG_NOT_JPEG;

      lastchar = thischar;

      thischar = (uint8)getBits1(8);

      if (lastchar == 0xFF) 
      {
         if (thischar == M_SOI)
            break;
         else if (thischar == M_EOI)	//getBits1 will keep returning M_EOI if we read past the end
            return PJPG_NOT_JPEG;
      }
   }

   /* Check the next character after marker: if it's not 0xFF, it can't
   be the start of the next marker, so the file is bad */

   thischar = (uint8)((gBitBuf >> 8) & 0xFF);

   if (thischar != 0xFF)
      return PJPG_NOT_JPEG;
      
   return 0;
}
//------------------------------------------------------------------------------
// Find a start of frame (SOF) marker.
static uint8 locateSOFMarker(void)
{
   uint8 c;

   uint8 status = locateSOIMarker();
   if (status)
      return status;
   
   status = processMarkers(&c);
   if (status)
      return status;

   switch (c)
   {
      case M_SOF2:
      {
         // Progressive JPEG - not supported by picojpeg (would require too
         // much memory, or too many IDCT's for embedded systems).
         return PJPG_UNSUPPORTED_MODE;
      }
      case M_SOF0:  /* baseline DCT */
      {
         status = readSOFMarker();
         if (status)
            return status;
            
         break;
      }
      case M_SOF9:  
      {
         return PJPG_NO_ARITHMITIC_SUPPORT;
      }
      case M_SOF1:  /* extended sequential DCT */
      default:
      {
         return PJPG_UNSUPPORTED_MARKER;
      }
   }
   
   return 0;
}
//------------------------------------------------------------------------------
// Find a start of scan (SOS) marker.
static uint8 locateSOSMarker(uint8* pFoundEOI)
{
   uint8 c;
   uint8 status;

   *pFoundEOI = 0;
      
   status = processMarkers(&c);
   if (status)
      return status;

   if (c == M_EOI)
   {
      *pFoundEOI = 1;
      return 0;
   }
   else if (c != M_SOS)
      return PJPG_UNEXPECTED_MARKER;

   return readSOSMarker();
}
//------------------------------------------------------------------------------
static uint8 init(void)
{
   gImageXSize = 0;
   gImageYSize = 0;
   gCompsInFrame = 0;
   gRestartInterval = 0;
   gCompsInScan = 0;
   gValidHuffTables = 0;
   gValidQuantTables = 0;
   gTemFlag = 0;
   gInBufOfs = 0;
   gInBufLeft = 0;
   gBitBuf = 0;
   gBitsLeft = 8;

   getBits1(8);
   getBits1(8);

   return 0;
}
//------------------------------------------------------------------------------
// This method throws back into the stream any bytes that where read
// into the bit buffer during initial marker scanning.
static void fixInBuffer(void)
{
   /* In case any 0xFF's where pulled into the buffer during marker scanning */

   if (gBitsLeft > 0)  
      stuffChar((uint8)gBitBuf);
   
   stuffChar((uint8)(gBitBuf >> 8));
   
   gBitsLeft = 8;
   getBits2(8);
   getBits2(8);
}
//------------------------------------------------------------------------------
// Restart interval processing.
static uint8 processRestart(void)
{
   // Let's scan a little bit to find the marker, but not _too_ far.
   // 1536 is a "fudge factor" that determines how much to scan.
   uint16 i;
   uint8 c = 0;

   for (i = 1536; i > 0; i--)
      if (getChar() == 0xFF)
         break;

   if (i == 0)
      return PJPG_BAD_RESTART_MARKER;
   
   for ( ; i > 0; i--)
      if ((c = getChar()) != 0xFF)
         break;

   if (i == 0)
      return PJPG_BAD_RESTART_MARKER;

   // Is it the expected marker? If not, something bad happened.
   if (c != (gNextRestartNum + M_RST0))
      return PJPG_BAD_RESTART_MARKER;

   // Reset each component's DC prediction values.
   gLastDC[0] = 0;
   gLastDC[1] = 0;
   gLastDC[2] = 0;

   gRestartsLeft = gRestartInterval;

   gNextRestartNum = (gNextRestartNum + 1) & 7;

   // Get the bit buffer going again

   gBitsLeft = 8;
   getBits2(8);
   getBits2(8);
   
   return 0;
}
//------------------------------------------------------------------------------
// FIXME: findEOI() is not actually called at the end of the image 
// (it's optional, and probably not needed on embedded devices)
static uint8 findEOI(void)
{
   uint8 c;
   uint8 status;

   // Prime the bit buffer
   gBitsLeft = 8;
   getBits1(8);
   getBits1(8);

   // The next marker _should_ be EOI
   status = processMarkers(&c);
   if (status)
      return status;
   else if (gCallbackStatus)
      return gCallbackStatus;
   
   //gTotalBytesRead -= in_buf_left;
   if (c != M_EOI)
      return PJPG_UNEXPECTED_MARKER;
   
   return 0;
}
//------------------------------------------------------------------------------
static uint8 checkHuffTables(void)
{
   uint8 i;

   for (i = 0; i < gCompsInScan; i++)
   {
      uint8 compDCTab = gCompDCTab[gCompList[i]];
      uint8 compACTab = gCompACTab[gCompList[i]] + 2;
      
      if ( ((gValidHuffTables & (1 << compDCTab)) == 0) ||
           ((gValidHuffTables & (1 << compACTab)) == 0) )
         return PJPG_UNDEFINED_HUFF_TABLE;           
   }
   
   return 0;
}
//------------------------------------------------------------------------------
static uint8 checkQuantTables(void)
{
   uint8 i;

   for (i = 0; i < gCompsInScan; i++)
   {
      uint8 compQuantMask = gCompQuant[gCompList[i]] ? 2 : 1;
      
      if ((gValidQuantTables & compQuantMask) == 0)
         return PJPG_UNDEFINED_QUANT_TABLE;
   }         

   return 0;         
}
//------------------------------------------------------------------------------
static uint8 initScan(void)
{
   uint8 foundEOI;
   uint8 status = locateSOSMarker(&foundEOI);
   if (status)
      return status;
   if (foundEOI)
      return PJPG_UNEXPECTED_MARKER;
   
   status = checkHuffTables();
   if (status)
      return status;

   status = checkQuantTables();
   if (status)
      return status;

   gLastDC[0] = 0;
   gLastDC[1] = 0;
   gLastDC[2] = 0;

   if (gRestartInterval)
   {
      gRestartsLeft = gRestartInterval;
      gNextRestartNum = 0;
   }

   fixInBuffer();

   return 0;
}
//------------------------------------------------------------------------------
static uint8 initFrame(void)
{
   if (gCompsInFrame == 1)
   {
      if ((gCompHSamp[0] != 1) || (gCompVSamp[0] != 1))
         return PJPG_UNSUPPORTED_SAMP_FACTORS;

      gScanType = PJPG_GRAYSCALE;

      gMaxBlocksPerMCU = 1;
      gMCUOrg[0] = 0;

      gMaxMCUXSize     = 8;
      gMaxMCUYSize     = 8;
   }
   else if (gCompsInFrame == 3)
   {
      if ( ((gCompHSamp[1] != 1) || (gCompVSamp[1] != 1)) ||
         ((gCompHSamp[2] != 1) || (gCompVSamp[2] != 1)) )
         return PJPG_UNSUPPORTED_SAMP_FACTORS;

      if ((gCompHSamp[0] == 1) && (gCompVSamp[0] == 1))
      {
         gScanType = PJPG_YH1V1;

         gMaxBlocksPerMCU = 3;
         gMCUOrg[0] = 0;
         gMCUOrg[1] = 1;
         gMCUOrg[2] = 2;
                  
         gMaxMCUXSize = 8;
         gMaxMCUYSize = 8;
      }
      else if ((gCompHSamp[0] == 1) && (gCompVSamp[0] == 2))
      {
         gScanType = PJPG_YH1V2;

         gMaxBlocksPerMCU = 4;
         gMCUOrg[0] = 0;
         gMCUOrg[1] = 0;
         gMCUOrg[2] = 1;
         gMCUOrg[3] = 2;

         gMaxMCUXSize = 8;
         gMaxMCUYSize = 16;
      }
      else if ((gCompHSamp[0] == 2) && (gCompVSamp[0] == 1))
      {
         gScanType = PJPG_YH2V1;

         gMaxBlocksPerMCU = 4;
         gMCUOrg[0] = 0;
         gMCUOrg[1] = 0;
         gMCUOrg[2] = 1;
         gMCUOrg[3] = 2;

         gMaxMCUXSize = 16;
         gMaxMCUYSize = 8;
      }
      else if ((gCompHSamp[0] == 2) && (gCompVSamp[0] == 2))
      {
         gScanType = PJPG_YH2V2;

         gMaxBlocksPerMCU = 6;
         gMCUOrg[0] = 0;
         gMCUOrg[1] = 0;
         gMCUOrg[2] = 0;
         gMCUOrg[3] = 0;
         gMCUOrg[4] = 1;
         gMCUOrg[5] = 2;

         gMaxMCUXSize = 16;
         gMaxMCUYSize = 16;
      }
      else
         return PJPG_UNSUPPORTED_SAMP_FACTORS;
   }
   else
      return PJPG_UNSUPPORTED_COLORSPACE;

   gMaxMCUSPerRow = (gImageXSize + (gMaxMCUXSize - 1)) >> ((gMaxMCUXSize == 8) ? 3 : 4);
   gMaxMCUSPerCol = (gImageYSize + (gMaxMCUYSize - 1)) >> ((gMaxMCUYSize == 8) ? 3 : 4);
   
   gNumMCUSRemaining = gMaxMCUSPerRow * gMaxMCUSPerCol;
   
   return 0;
}
//----------------------------------------------------------------------------
// Winograd IDCT: 5 multiplies per row/col, up to 80 muls for the 2D IDCT

#define PJPG_DCT_SCALE_BITS 7

#define PJPG_DCT_SCALE (1U << PJPG_DCT_SCALE_BITS)

#define PJPG_DESCALE(x) PJPG_ARITH_SHIFT_RIGHT_N_16(((x) + (1U << (PJPG_DCT_SCALE_BITS - 1))), PJPG_DCT_SCALE_BITS)

#define PJPG_WFIX(x) ((x) * PJPG_DCT_SCALE + 0.5f)

#define PJPG_WINOGRAD_QUANT_SCALE_BITS 10

const uint8 gWinogradQuant[] = 
{
   128,  178,  178,  167,  246,  167,  151,  232,
   232,  151,  128,  209,  219,  209,  128,  101,
   178,  197,  197,  178,  101,   69,  139,  167,
   177,  167,  139,   69,   35,   96,  131,  151,
   151,  131,   96,   35,   49,   91,  118,  128,
   118,   91,   49,   46,   81,  101,  101,   81,
   46,   42,   69,   79,   69,   42,   35,   54,
   54,   35,   28,   37,   28,   19,   19,   10,
};   

// Multiply quantization matrix by the Winograd IDCT scale factors
static void createWinogradQuant(int16* pQuant)
{
   uint8 i;
   
   for (i = 0; i < 64; i++) 
   {
      long x = pQuant[i];
      x *= gWinogradQuant[i];
      pQuant[i] = (int16)((x + (1 << (PJPG_WINOGRAD_QUANT_SCALE_BITS - PJPG_DCT_SCALE_BITS - 1))) >> (PJPG_WINOGRAD_QUANT_SCALE_BITS - PJPG_DCT_SCALE_BITS));
   }
}

// These multiply helper functions are the 4 types of signed multiplies needed by the Winograd IDCT.
// A smart C compiler will optimize them to use 16x8 = 24 bit muls, if not you may need to tweak
// these functions or drop to CPU specific inline assembly.

// 1/cos(4*pi/16)
// 362, 256+106
static PJPG_INLINE int16 imul_b1_b3(int16 w)
{
   long x = (w * 362L);
   x += 128L;
   return (int16)(PJPG_ARITH_SHIFT_RIGHT_8_L(x));
}

// 1/cos(6*pi/16)
// 669, 256+256+157
static PJPG_INLINE int16 imul_b2(int16 w)
{
   long x = (w * 669L);
   x += 128L;
   return (int16)(PJPG_ARITH_SHIFT_RIGHT_8_L(x));
}

// 1/cos(2*pi/16)
// 277, 256+21
static PJPG_INLINE int16 imul_b4(int16 w)
{
   long x = (w * 277L);
   x += 128L;
   return (int16)(PJPG_ARITH_SHIFT_RIGHT_8_L(x));
}

// 1/(cos(2*pi/16) + cos(6*pi/16))
// 196, 196
static PJPG_INLINE int16 imul_b5(int16 w)
{
   long x = (w * 196L);
   x += 128L;
   return (int16)(PJPG_ARITH_SHIFT_RIGHT_8_L(x));
}

static PJPG_INLINE uint8 clamp(int16 s)
{
   if ((uint16)s > 255U)
   {
      if (s < 0) 
         return 0; 
      else if (s > 255) 
         return 255;
   }
      
   return (uint8)s;
}

static void idctRows(void)
{
   uint8 i;
   int16* pSrc = gCoeffBuf;
            
   for (i = 0; i < 8; i++)
   {
      if ((pSrc[1] | pSrc[2] | pSrc[3] | pSrc[4] | pSrc[5] | pSrc[6] | pSrc[7]) == 0)
      {
         // Short circuit the 1D IDCT if only the DC component is non-zero
         int16 src0 = *pSrc;

         *(pSrc+1) = src0;
         *(pSrc+2) = src0;
         *(pSrc+3) = src0;
         *(pSrc+4) = src0;
         *(pSrc+5) = src0;
         *(pSrc+6) = src0;
         *(pSrc+7) = src0;
      }
      else
      {
         int16 src4 = *(pSrc+5);
         int16 src7 = *(pSrc+3);
         int16 x4  = src4 - src7;
         int16 x7  = src4 + src7;

         int16 src5 = *(pSrc+1);
         int16 src6 = *(pSrc+7);
         int16 x5  = src5 + src6;
         int16 x6  = src5 - src6;

         int16 tmp1 = imul_b5(x4 - x6);
         int16 stg26 = imul_b4(x6) - tmp1;

         int16 x24 = tmp1 - imul_b2(x4);

         int16 x15 = x5 - x7;
         int16 x17 = x5 + x7;

         int16 tmp2 = stg26 - x17;
         int16 tmp3 = imul_b1_b3(x15) - tmp2;
         int16 x44 = tmp3 + x24;

         int16 src0 = *(pSrc+0);
         int16 src1 = *(pSrc+4);
         int16 x30 = src0 + src1;
         int16 x31 = src0 - src1;

         int16 src2 = *(pSrc+2);
         int16 src3 = *(pSrc+6);
         int16 x12 = src2 - src3;
         int16 x13 = src2 + src3;

         int16 x32 = imul_b1_b3(x12) - x13;

         int16 x40 = x30 + x13;
         int16 x43 = x30 - x13;
         int16 x41 = x31 + x32;
         int16 x42 = x31 - x32;

         *(pSrc+0) = x40 + x17;
         *(pSrc+1) = x41 + tmp2;
         *(pSrc+2) = x42 + tmp3;
         *(pSrc+3) = x43 - x44;
         *(pSrc+4) = x43 + x44;
         *(pSrc+5) = x42 - tmp3;
         *(pSrc+6) = x41 - tmp2;
         *(pSrc+7) = x40 - x17;
      }
                  
      pSrc += 8;
   }      
}

static void idctCols(void)
{
   uint8 i;
      
   int16* pSrc = gCoeffBuf;
   
   for (i = 0; i < 8; i++)
   {
      if ((pSrc[1*8] | pSrc[2*8] | pSrc[3*8] | pSrc[4*8] | pSrc[5*8] | pSrc[6*8] | pSrc[7*8]) == 0)
      {
         // Short circuit the 1D IDCT if only the DC component is non-zero
         uint8 c = clamp(PJPG_DESCALE(*pSrc) + 128);
         *(pSrc+0*8) = c;
         *(pSrc+1*8) = c;
         *(pSrc+2*8) = c;
         *(pSrc+3*8) = c;
         *(pSrc+4*8) = c;
         *(pSrc+5*8) = c;
         *(pSrc+6*8) = c;
         *(pSrc+7*8) = c;
      }
      else
      {
         int16 src4 = *(pSrc+5*8);
         int16 src7 = *(pSrc+3*8);
         int16 x4  = src4 - src7;
         int16 x7  = src4 + src7;

         int16 src5 = *(pSrc+1*8);
         int16 src6 = *(pSrc+7*8);
         int16 x5  = src5 + src6;
         int16 x6  = src5 - src6;

         int16 tmp1 = imul_b5(x4 - x6);
         int16 stg26 = imul_b4(x6) - tmp1;

         int16 x24 = tmp1 - imul_b2(x4);

         int16 x15 = x5 - x7;
         int16 x17 = x5 + x7;

         int16 tmp2 = stg26 - x17;
         int16 tmp3 = imul_b1_b3(x15) - tmp2;
         int16 x44 = tmp3 + x24;

         int16 src0 = *(pSrc+0*8);
         int16 src1 = *(pSrc+4*8);
         int16 x30 = src0 + src1;
         int16 x31 = src0 - src1;

         int16 src2 = *(pSrc+2*8);
         int16 src3 = *(pSrc+6*8);
         int16 x12 = src2 - src3;
         int16 x13 = src2 + src3;

         int16 x32 = imul_b1_b3(x12) - x13;

         int16 x40 = x30 + x13;
         int16 x43 = x30 - x13;
         int16 x41 = x31 + x32;
         int16 x42 = x31 - x32;

         // descale, convert to unsigned and clamp to 8-bit
         *(pSrc+0*8) = clamp(PJPG_DESCALE(x40 + x17)  + 128);
         *(pSrc+1*8) = clamp(PJPG_DESCALE(x41 + tmp2) + 128);
         *(pSrc+2*8) = clamp(PJPG_DESCALE(x42 + tmp3) + 128);
         *(pSrc+3*8) = clamp(PJPG_DESCALE(x43 - x44)  + 128);
         *(pSrc+4*8) = clamp(PJPG_DESCALE(x43 + x44)  + 128);
         *(pSrc+5*8) = clamp(PJPG_DESCALE(x42 - tmp3) + 128);
         *(pSrc+6*8) = clamp(PJPG_DESCALE(x41 - tmp2) + 128);
         *(pSrc+7*8) = clamp(PJPG_DESCALE(x40 - x17)  + 128);
      }

      pSrc++;      
   }      
}

/*----------------------------------------------------------------------------*/
static PJPG_INLINE uint8 addAndClamp(uint8 a, int16 b)
{
   b = a + b;
   
   if ((uint16)b > 255U)
   {
      if (b < 0)
         return 0;
      else if (b > 255)
         return 255;
   }
      
   return (uint8)b;
}
/*----------------------------------------------------------------------------*/
static PJPG_INLINE uint8 subAndClamp(uint8 a, int16 b)
{
   b = a - b;

   if ((uint16)b > 255U)
   {
      if (b < 0)
         return 0;
      else if (b > 255)
         return 255;
   }

   return (uint8)b;
}
/*----------------------------------------------------------------------------*/
// 103/256
//R = Y + 1.402 (Cr-128)

// 88/256, 183/256
//G = Y - 0.34414 (Cb-128) - 0.71414 (Cr-128)

// 198/256
//B = Y + 1.772 (Cb-128)
/*----------------------------------------------------------------------------*/
// Cb upsample and accumulate, 4x4 to 8x8
static void upsampleCb(uint8 srcOfs, uint8 dstOfs)
{
   // Cb - affects G and B
   uint8 x, y;
   int16* pSrc = gCoeffBuf + srcOfs;
   uint8* pDstG = gMCUBufG + dstOfs;
   uint8* pDstB = gMCUBufB + dstOfs;
   for (y = 0; y < 4; y++)
   {
      for (x = 0; x < 4; x++)
      {
         uint8 cb = (uint8)*pSrc++;
         int16 cbG, cbB;

         cbG = ((cb * 88U) >> 8U) - 44U;
         pDstG[0] = subAndClamp(pDstG[0], cbG);
         pDstG[1] = subAndClamp(pDstG[1], cbG);
         pDstG[8] = subAndClamp(pDstG[8], cbG);
         pDstG[9] = subAndClamp(pDstG[9], cbG);

         cbB = (cb + ((cb * 198U) >> 8U)) - 227U;
         pDstB[0] = addAndClamp(pDstB[0], cbB);
         pDstB[1] = addAndClamp(pDstB[1], cbB);
         pDstB[8] = addAndClamp(pDstB[8], cbB);
         pDstB[9] = addAndClamp(pDstB[9], cbB);

         pDstG += 2;
         pDstB += 2;
      }

      pSrc = pSrc - 4 + 8;
      pDstG = pDstG - 8 + 16;
      pDstB = pDstB - 8 + 16;
   }
}   
/*----------------------------------------------------------------------------*/
// Cb upsample and accumulate, 4x8 to 8x8
static void upsampleCbH(uint8 srcOfs, uint8 dstOfs)
{
   // Cb - affects G and B
   uint8 x, y;
   int16* pSrc = gCoeffBuf + srcOfs;
   uint8* pDstG = gMCUBufG + dstOfs;
   uint8* pDstB = gMCUBufB + dstOfs;
   for (y = 0; y < 8; y++)
   {
      for (x = 0; x < 4; x++)
      {
         uint8 cb = (uint8)*pSrc++;
         int16 cbG, cbB;

         cbG = ((cb * 88U) >> 8U) - 44U;
         pDstG[0] = subAndClamp(pDstG[0], cbG);
         pDstG[1] = subAndClamp(pDstG[1], cbG);

         cbB = (cb + ((cb * 198U) >> 8U)) - 227U;
         pDstB[0] = addAndClamp(pDstB[0], cbB);
         pDstB[1] = addAndClamp(pDstB[1], cbB);

         pDstG += 2;
         pDstB += 2;
      }

      pSrc = pSrc - 4 + 8;
   }
}   
/*----------------------------------------------------------------------------*/
// Cb upsample and accumulate, 8x4 to 8x8
static void upsampleCbV(uint8 srcOfs, uint8 dstOfs)
{
   // Cb - affects G and B
   uint8 x, y;
   int16* pSrc = gCoeffBuf + srcOfs;
   uint8* pDstG = gMCUBufG + dstOfs;
   uint8* pDstB = gMCUBufB + dstOfs;
   for (y = 0; y < 4; y++)
   {
      for (x = 0; x < 8; x++)
      {
         uint8 cb = (uint8)*pSrc++;
         int16 cbG, cbB;

         cbG = ((cb * 88U) >> 8U) - 44U;
         pDstG[0] = subAndClamp(pDstG[0], cbG);
         pDstG[8] = subAndClamp(pDstG[8], cbG);

         cbB = (cb + ((cb * 198U) >> 8U)) - 227U;
         pDstB[0] = addAndClamp(pDstB[0], cbB);
         pDstB[8] = addAndClamp(pDstB[8], cbB);

         ++pDstG;
         ++pDstB;
      }

      pDstG = pDstG - 8 + 16;
      pDstB = pDstB - 8 + 16;
   }
}   
/*----------------------------------------------------------------------------*/
// 103/256
//R = Y + 1.402 (Cr-128)

// 88/256, 183/256
//G = Y - 0.34414 (Cb-128) - 0.71414 (Cr-128)

// 198/256
//B = Y + 1.772 (Cb-128)
/*----------------------------------------------------------------------------*/
// Cr upsample and accumulate, 4x4 to 8x8
static void upsampleCr(uint8 srcOfs, uint8 dstOfs)
{
   // Cr - affects R and G
   uint8 x, y;
   int16* pSrc = gCoeffBuf + srcOfs;
   uint8* pDstR = gMCUBufR + dstOfs;
   uint8* pDstG = gMCUBufG + dstOfs;
   for (y = 0; y < 4; y++)
   {
      for (x = 0; x < 4; x++)
      {
         uint8 cr = (uint8)*pSrc++;
         int16 crR, crG;

         crR = (cr + ((cr * 103U) >> 8U)) - 179;
         pDstR[0] = addAndClamp(pDstR[0], crR);
         pDstR[1] = addAndClamp(pDstR[1], crR);
         pDstR[8] = addAndClamp(pDstR[8], crR);
         pDstR[9] = addAndClamp(pDstR[9], crR);
         
         crG = ((cr * 183U) >> 8U) - 91;
         pDstG[0] = subAndClamp(pDstG[0], crG);
         pDstG[1] = subAndClamp(pDstG[1], crG);
         pDstG[8] = subAndClamp(pDstG[8], crG);
         pDstG[9] = subAndClamp(pDstG[9], crG);
         
         pDstR += 2;
         pDstG += 2;
      }

      pSrc = pSrc - 4 + 8;
      pDstR = pDstR - 8 + 16;
      pDstG = pDstG - 8 + 16;
   }
}   
/*----------------------------------------------------------------------------*/
// Cr upsample and accumulate, 4x8 to 8x8
static void upsampleCrH(uint8 srcOfs, uint8 dstOfs)
{
   // Cr - affects R and G
   uint8 x, y;
   int16* pSrc = gCoeffBuf + srcOfs;
   uint8* pDstR = gMCUBufR + dstOfs;
   uint8* pDstG = gMCUBufG + dstOfs;
   for (y = 0; y < 8; y++)
   {
      for (x = 0; x < 4; x++)
      {
         uint8 cr = (uint8)*pSrc++;
         int16 crR, crG;

         crR = (cr + ((cr * 103U) >> 8U)) - 179;
         pDstR[0] = addAndClamp(pDstR[0], crR);
         pDstR[1] = addAndClamp(pDstR[1], crR);
         
         crG = ((cr * 183U) >> 8U) - 91;
         pDstG[0] = subAndClamp(pDstG[0], crG);
         pDstG[1] = subAndClamp(pDstG[1], crG);
         
         pDstR += 2;
         pDstG += 2;
      }

      pSrc = pSrc - 4 + 8;
   }
}   
/*----------------------------------------------------------------------------*/
// Cr upsample and accumulate, 8x4 to 8x8
static void upsampleCrV(uint8 srcOfs, uint8 dstOfs)
{
   // Cr - affects R and G
   uint8 x, y;
   int16* pSrc = gCoeffBuf + srcOfs;
   uint8* pDstR = gMCUBufR + dstOfs;
   uint8* pDstG = gMCUBufG + dstOfs;
   for (y = 0; y < 4; y++)
   {
      for (x = 0; x < 8; x++)
      {
         uint8 cr = (uint8)*pSrc++;
         int16 crR, crG;

         crR = (cr + ((cr * 103U) >> 8U)) - 179;
         pDstR[0] = addAndClamp(pDstR[0], crR);
         pDstR[8] = addAndClamp(pDstR[8], crR);

         crG = ((cr * 183U) >> 8U) - 91;
         pDstG[0] = subAndClamp(pDstG[0], crG);
         pDstG[8] = subAndClamp(pDstG[8], crG);

         ++pDstR;
         ++pDstG;
      }

      pDstR = pDstR - 8 + 16;
      pDstG = pDstG - 8 + 16;
   }
} 
/*----------------------------------------------------------------------------*/
// Convert Y to RGB
static void copyY(uint8 dstOfs)
{
   uint8 i;
   uint8* pRDst = gMCUBufR + dstOfs;
   uint8* pGDst = gMCUBufG + dstOfs;
   uint8* pBDst = gMCUBufB + dstOfs;
   int16* pSrc = gCoeffBuf;
   
   for (i = 64; i > 0; i--)
   {
      uint8 c = (uint8)*pSrc++;
      
      *pRDst++ = c;
      *pGDst++ = c;
      *pBDst++ = c;
   }
}
/*----------------------------------------------------------------------------*/
// Cb convert to RGB and accumulate
static void convertCb(uint8 dstOfs)
{
   uint8 i;
   uint8* pDstG = gMCUBufG + dstOfs;
   uint8* pDstB = gMCUBufB + dstOfs;
   int16* pSrc = gCoeffBuf;

   for (i = 64; i > 0; i--)
   {
      uint8 cb = (uint8)*pSrc++;
      int16 cbG, cbB;

      cbG = ((cb * 88U) >> 8U) - 44U;
      pDstG[0] = subAndClamp(pDstG[0], cbG);

      cbB = (cb + ((cb * 198U) >> 8U)) - 227U;
      pDstB[0] = addAndClamp(pDstB[0], cbB);
      
      ++pDstG;
      ++pDstB;
   }
}
/*----------------------------------------------------------------------------*/
// Cr convert to RGB and accumulate
static void convertCr(uint8 dstOfs)
{
   uint8 i;
   uint8* pDstR = gMCUBufR + dstOfs;
   uint8* pDstG = gMCUBufG + dstOfs;
   int16* pSrc = gCoeffBuf;

   for (i = 64; i > 0; i--)
   {
      uint8 cr = (uint8)*pSrc++;
      int16 crR, crG;

      crR = (cr + ((cr * 103U) >> 8U)) - 179;
      pDstR[0] = addAndClamp(pDstR[0], crR);

      crG = ((cr * 183U) >> 8U) - 91;
      pDstG[0] = subAndClamp(pDstG[0], crG);
      
      ++pDstR;
      ++pDstG;
      }
}
/*----------------------------------------------------------------------------*/
static void transformBlock(uint8 mcuBlock)
{
   idctRows();
   idctCols();
   
   switch (gScanType)
   {
      case PJPG_GRAYSCALE:
      {
         // MCU size: 1, 1 block per MCU
         copyY(0);
         break;
      }
      case PJPG_YH1V1:
      {
         // MCU size: 8x8, 3 blocks per MCU
         switch (mcuBlock)
         {
            case 0:
            {
               copyY(0);
               break;
            }
            case 1:
            {
               convertCb(0);
               break;
            }
            case 2:
            {
               convertCr(0);
               break;
            }
         }

         break;
      }
      case PJPG_YH1V2:
      {
         // MCU size: 8x16, 4 blocks per MCU
         switch (mcuBlock)
         {
            case 0:
            {
               copyY(0);
               break;
            }
            case 1:
            {
               copyY(128);
               break;
            }
            case 2:
            {
               upsampleCbV(0, 0);
               upsampleCbV(4*8, 128);
               break;
            }
            case 3:
            {
               upsampleCrV(0, 0);
               upsampleCrV(4*8, 128);
               break;
            }
         }

         break;
      }        
      case PJPG_YH2V1:
      {
         // MCU size: 16x8, 4 blocks per MCU
         switch (mcuBlock)
         {
            case 0:
            {
               copyY(0);
               break;
            }
            case 1:
            {
               copyY(64);
               break;
            }
            case 2:
            {
               upsampleCbH(0, 0);
               upsampleCbH(4, 64);
               break;
            }
            case 3:
            {
               upsampleCrH(0, 0);
               upsampleCrH(4, 64);
               break;
            }
         }
         
         break;
      }        
      case PJPG_YH2V2:
      {
         // MCU size: 16x16, 6 blocks per MCU
         switch (mcuBlock)
         {
            case 0:
            {
               copyY(0);
               break;
            }
            case 1:
            {
               copyY(64);
               break;
            }
            case 2:
            {
               copyY(128);
               break;
            }
            case 3:
            {
               copyY(192);
               break;
            }
            case 4:
            {
               upsampleCb(0, 0);
               upsampleCb(4, 64);
               upsampleCb(4*8, 128);
               upsampleCb(4+4*8, 192);
               break;
            }
            case 5:
            {
               upsampleCr(0, 0);
               upsampleCr(4, 64);
               upsampleCr(4*8, 128);
               upsampleCr(4+4*8, 192);
               break;
            }
         }

         break;
      }         
   }      
}
//------------------------------------------------------------------------------
static void transformBlockReduce(uint8 mcuBlock)
{
   uint8 c = clamp(PJPG_DESCALE(gCoeffBuf[0]) + 128);
   int16 cbG, cbB, crR, crG;

   switch (gScanType)
   {
      case PJPG_GRAYSCALE:
      {
         // MCU size: 1, 1 block per MCU
         gMCUBufR[0] = c;
         break;
      }
      case PJPG_YH1V1:
      {
         // MCU size: 8x8, 3 blocks per MCU
         switch (mcuBlock)
         {
            case 0:
            {
               gMCUBufR[0] = c;
               gMCUBufG[0] = c;
               gMCUBufB[0] = c;
               break;
            }
            case 1:
            {
               cbG = ((c * 88U) >> 8U) - 44U;
               gMCUBufG[0] = subAndClamp(gMCUBufG[0], cbG);

               cbB = (c + ((c * 198U) >> 8U)) - 227U;
               gMCUBufB[0] = addAndClamp(gMCUBufB[0], cbB);
               break;
            }
            case 2:
            {
               crR = (c + ((c * 103U) >> 8U)) - 179;
               gMCUBufR[0] = addAndClamp(gMCUBufR[0], crR);

               crG = ((c * 183U) >> 8U) - 91;
               gMCUBufG[0] = subAndClamp(gMCUBufG[0], crG);
               break;
            }
         }

         break;
      }
      case PJPG_YH1V2:
      {
         // MCU size: 8x16, 4 blocks per MCU
         switch (mcuBlock)
         {
            case 0:
            {
               gMCUBufR[0] = c;
               gMCUBufG[0] = c;
               gMCUBufB[0] = c;
               break;
            }
            case 1:
            {
               gMCUBufR[128] = c;
               gMCUBufG[128] = c;
               gMCUBufB[128] = c;
               break;
            }
            case 2:
            {
               cbG = ((c * 88U) >> 8U) - 44U;
               gMCUBufG[0] = subAndClamp(gMCUBufG[0], cbG);
               gMCUBufG[128] = subAndClamp(gMCUBufG[128], cbG);

               cbB = (c + ((c * 198U) >> 8U)) - 227U;
               gMCUBufB[0] = addAndClamp(gMCUBufB[0], cbB);
               gMCUBufB[128] = addAndClamp(gMCUBufB[128], cbB);

               break;
            }
            case 3:
            {
               crR = (c + ((c * 103U) >> 8U)) - 179;
               gMCUBufR[0] = addAndClamp(gMCUBufR[0], crR);
               gMCUBufR[128] = addAndClamp(gMCUBufR[128], crR);

               crG = ((c * 183U) >> 8U) - 91;
               gMCUBufG[0] = subAndClamp(gMCUBufG[0], crG);
               gMCUBufG[128] = subAndClamp(gMCUBufG[128], crG);

               break;
            }
         }
         break;
      }
      case PJPG_YH2V1:
      {
         // MCU size: 16x8, 4 blocks per MCU
         switch (mcuBlock)
         {
            case 0:
            {
               gMCUBufR[0] = c;
               gMCUBufG[0] = c;
               gMCUBufB[0] = c;
               break;
            }
            case 1:
            {
               gMCUBufR[64] = c;
               gMCUBufG[64] = c;
               gMCUBufB[64] = c;
               break;
            }
            case 2:
            {
               cbG = ((c * 88U) >> 8U) - 44U;
               gMCUBufG[0] = subAndClamp(gMCUBufG[0], cbG);
               gMCUBufG[64] = subAndClamp(gMCUBufG[64], cbG);

               cbB = (c + ((c * 198U) >> 8U)) - 227U;
               gMCUBufB[0] = addAndClamp(gMCUBufB[0], cbB);
               gMCUBufB[64] = addAndClamp(gMCUBufB[64], cbB);

               break;
            }
            case 3:
            {
               crR = (c + ((c * 103U) >> 8U)) - 179;
               gMCUBufR[0] = addAndClamp(gMCUBufR[0], crR);
               gMCUBufR[64] = addAndClamp(gMCUBufR[64], crR);

               crG = ((c * 183U) >> 8U) - 91;
               gMCUBufG[0] = subAndClamp(gMCUBufG[0], crG);
               gMCUBufG[64] = subAndClamp(gMCUBufG[64], crG);

               break;
            }
         }
         break;
      }
      case PJPG_YH2V2:
      {
         // MCU size: 16x16, 6 blocks per MCU
         switch (mcuBlock)
         {
            case 0:
            {
               gMCUBufR[0] = c;
               gMCUBufG[0] = c;
               gMCUBufB[0] = c;
               break;
            }
            case 1:
            {
               gMCUBufR[64] = c;
               gMCUBufG[64] = c;
               gMCUBufB[64] = c;
               break;
            }
            case 2:
            {
               gMCUBufR[128] = c;
               gMCUBufG[128] = c;
               gMCUBufB[128] = c;
               break;
            }
            case 3:
            {
               gMCUBufR[192] = c;
               gMCUBufG[192] = c;
               gMCUBufB[192] = c;
               break;
            }
            case 4:
            {
               cbG = ((c * 88U) >> 8U) - 44U;
               gMCUBufG[0] = subAndClamp(gMCUBufG[0], cbG);
               gMCUBufG[64] = subAndClamp(gMCUBufG[64], cbG);
               gMCUBufG[128] = subAndClamp(gMCUBufG[128], cbG);
               gMCUBufG[192] = subAndClamp(gMCUBufG[192], cbG);

               cbB = (c + ((c * 198U) >> 8U)) - 227U;
               gMCUBufB[0] = addAndClamp(gMCUBufB[0], cbB);
               gMCUBufB[64] = addAndClamp(gMCUBufB[64], cbB);
               gMCUBufB[128] = addAndClamp(gMCUBufB[128], cbB);
               gMCUBufB[192] = addAndClamp(gMCUBufB[192], cbB);

               break;
            }
            case 5:
            {
               crR = (c + ((c * 103U) >> 8U)) - 179;
               gMCUBufR[0] = addAndClamp(gMCUBufR[0], crR);
               gMCUBufR[64] = addAndClamp(gMCUBufR[64], crR);
               gMCUBufR[128] = addAndClamp(gMCUBufR[128], crR);
               gMCUBufR[192] = addAndClamp(gMCUBufR[192], crR);

               crG = ((c * 183U) >> 8U) - 91;
               gMCUBufG[0] = subAndClamp(gMCUBufG[0], crG);
               gMCUBufG[64] = subAndClamp(gMCUBufG[64], crG);
               gMCUBufG[128] = subAndClamp(gMCUBufG[128], crG);
               gMCUBufG[192] = subAndClamp(gMCUBufG[192], crG);

               break;
            }
         }
         break;
      }
   }
}
//------------------------------------------------------------------------------
static uint8 decodeNextMCU(void)
{
   uint8 status;
   uint8 mcuBlock;   

   if (gRestartInterval) 
   {
      if (gRestartsLeft == 0)
      {
         status = processRestart();
         if (status)
            return status;
      }
      gRestartsLeft--;
   }      
   
   for (mcuBlock = 0; mcuBlock < gMaxBlocksPerMCU; mcuBlock++)
   {
      uint8 componentID = gMCUOrg[mcuBlock];
      uint8 compQuant = gCompQuant[componentID];	
      uint8 compDCTab = gCompDCTab[componentID];
      uint8 numExtraBits, compACTab, k;
      const int16* pQ = compQuant ? gQuant1 : gQuant0;
      uint16 r, dc;

      uint8 s = huffDecode(compDCTab ? &gHuffTab1 : &gHuffTab0, compDCTab ? gHuffVal1 : gHuffVal0);
      
      r = 0;
      numExtraBits = s & 0xF;
      if (numExtraBits)
         r = getBits2(numExtraBits);
      dc = huffExtend(r, s);
            
      dc = dc + gLastDC[componentID];
      gLastDC[componentID] = dc;
            
      gCoeffBuf[0] = dc * pQ[0];

      compACTab = gCompACTab[componentID];

      if (gReduce)
      {
         // Decode, but throw out the AC coefficients in reduce mode.
         for (k = 1; k < 64; k++)
         {
            s = huffDecode(compACTab ? &gHuffTab3 : &gHuffTab2, compACTab ? gHuffVal3 : gHuffVal2);

            numExtraBits = s & 0xF;
            if (numExtraBits)
               getBits2(numExtraBits);

            r = s >> 4;
            s &= 15;

            if (s)
            {
               if (r)
               {
                  if ((k + r) > 63)
                     return PJPG_DECODE_ERROR;

                  k = (uint8)(k + r);
               }
            }
            else
            {
               if (r == 15)
               {
                  if ((k + 16) > 64)
                     return PJPG_DECODE_ERROR;

                  k += (16 - 1); // - 1 because the loop counter is k
               }
               else
                  break;
            }
         }

         transformBlockReduce(mcuBlock); 
      }
      else
      {
         // Decode and dequantize AC coefficients
         for (k = 1; k < 64; k++)
         {
            uint16 extraBits;

            s = huffDecode(compACTab ? &gHuffTab3 : &gHuffTab2, compACTab ? gHuffVal3 : gHuffVal2);

            extraBits = 0;
            numExtraBits = s & 0xF;
            if (numExtraBits)
               extraBits = getBits2(numExtraBits);

            r = s >> 4;
            s &= 15;

            if (s)
            {
               int16 ac;

               if (r)
               {
                  if ((k + r) > 63)
                     return PJPG_DECODE_ERROR;

                  while (r)
                  {
                     gCoeffBuf[ZAG[k++]] = 0;
                     r--;
                  }
               }

               ac = huffExtend(extraBits, s);
               
               gCoeffBuf[ZAG[k]] = ac * pQ[k]; 
            }
            else
            {
               if (r == 15)
               {
                  if ((k + 16) > 64)
                     return PJPG_DECODE_ERROR;
                  
                  for (r = 16; r > 0; r--)
                     gCoeffBuf[ZAG[k++]] = 0;
                  
                  k--; // - 1 because the loop counter is k
               }
               else
                  break;
            }
         }
         
         while (k < 64)
            gCoeffBuf[ZAG[k++]] = 0;

         transformBlock(mcuBlock); 
      }
   }
         
   return 0;
}
//------------------------------------------------------------------------------
unsigned char pjpeg_decode_mcu(void)
{
   uint8 status;
   
   if (gCallbackStatus)
      return gCallbackStatus;
   
   if (!gNumMCUSRemaining)
      return PJPG_NO_MORE_BLOCKS;
      
   status = decodeNextMCU();
   if ((status) || (gCallbackStatus))
      return gCallbackStatus ? gCallbackStatus : status;
      
   gNumMCUSRemaining--;
   
   return 0;
}
//------------------------------------------------------------------------------
unsigned char pjpeg_decode_init(pjpeg_image_info_t *pInfo, pjpeg_need_bytes_callback_t pNeed_bytes_callback, void *pCallback_data, unsigned char reduce)
{
   uint8 status;
   
   pInfo->m_width = 0; pInfo->m_height = 0; pInfo->m_comps = 0;
   pInfo->m_MCUSPerRow = 0; pInfo->m_MCUSPerCol = 0;
   pInfo->m_scanType = PJPG_GRAYSCALE;
   pInfo->m_MCUWidth = 0; pInfo->m_MCUHeight = 0;
   pInfo->m_pMCUBufR = (unsigned char*)0; pInfo->m_pMCUBufG = (unsigned char*)0; pInfo->m_pMCUBufB = (unsigned char*)0;

   g_pNeedBytesCallback = pNeed_bytes_callback;
   g_pCallback_data = pCallback_data;
   gCallbackStatus = 0;
   gReduce = reduce;
    
   status = init();
   if ((status) || (gCallbackStatus))
      return gCallbackStatus ? gCallbackStatus : status;
   
   status = locateSOFMarker();
   if ((status) || (gCallbackStatus))
      return gCallbackStatus ? gCallbackStatus : status;

   status = initFrame();
   if ((status) || (gCallbackStatus))
      return gCallbackStatus ? gCallbackStatus : status;

   status = initScan();
   if ((status) || (gCallbackStatus))
      return gCallbackStatus ? gCallbackStatus : status;

   pInfo->m_width = gImageXSize; pInfo->m_height = gImageYSize; pInfo->m_comps = gCompsInFrame;
   pInfo->m_scanType = gScanType;
   pInfo->m_MCUSPerRow = gMaxMCUSPerRow; pInfo->m_MCUSPerCol = gMaxMCUSPerCol;
   pInfo->m_MCUWidth = gMaxMCUXSize; pInfo->m_MCUHeight = gMaxMCUYSize;
   pInfo->m_pMCUBufR = gMCUBufR; pInfo->m_pMCUBufG = gMCUBufG; pInfo->m_pMCUBufB = gMCUBufB;
      
   return 0;
}
