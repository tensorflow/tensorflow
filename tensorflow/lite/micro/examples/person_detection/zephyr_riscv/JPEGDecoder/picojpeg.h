//------------------------------------------------------------------------------
// picojpeg - Public domain, Rich Geldreich <richgel99@gmail.com>
//------------------------------------------------------------------------------
#ifndef PICOJPEG_H
#define PICOJPEG_H

#ifdef __cplusplus
extern "C" {
#endif

// Error codes
enum
{
   PJPG_NO_MORE_BLOCKS = 1,
   PJPG_BAD_DHT_COUNTS,
   PJPG_BAD_DHT_INDEX,
   PJPG_BAD_DHT_MARKER,
   PJPG_BAD_DQT_MARKER,
   PJPG_BAD_DQT_TABLE,
   PJPG_BAD_PRECISION,
   PJPG_BAD_HEIGHT,
   PJPG_BAD_WIDTH,
   PJPG_TOO_MANY_COMPONENTS,
   PJPG_BAD_SOF_LENGTH,
   PJPG_BAD_VARIABLE_MARKER,
   PJPG_BAD_DRI_LENGTH,
   PJPG_BAD_SOS_LENGTH,
   PJPG_BAD_SOS_COMP_ID,
   PJPG_W_EXTRA_BYTES_BEFORE_MARKER,
   PJPG_NO_ARITHMITIC_SUPPORT,
   PJPG_UNEXPECTED_MARKER,
   PJPG_NOT_JPEG,
   PJPG_UNSUPPORTED_MARKER,
   PJPG_BAD_DQT_LENGTH,
   PJPG_TOO_MANY_BLOCKS,
   PJPG_UNDEFINED_QUANT_TABLE,
   PJPG_UNDEFINED_HUFF_TABLE,
   PJPG_NOT_SINGLE_SCAN,
   PJPG_UNSUPPORTED_COLORSPACE,
   PJPG_UNSUPPORTED_SAMP_FACTORS,
   PJPG_DECODE_ERROR,
   PJPG_BAD_RESTART_MARKER,
   PJPG_ASSERTION_ERROR,
   PJPG_BAD_SOS_SPECTRAL,
   PJPG_BAD_SOS_SUCCESSIVE,
   PJPG_STREAM_READ_ERROR,
   PJPG_NOTENOUGHMEM,
   PJPG_UNSUPPORTED_COMP_IDENT,
   PJPG_UNSUPPORTED_QUANT_TABLE,
   PJPG_UNSUPPORTED_MODE,        // picojpeg doesn't support progressive JPEG's
};  

// Scan types
typedef enum
{
   PJPG_GRAYSCALE,
   PJPG_YH1V1,
   PJPG_YH2V1,
   PJPG_YH1V2,
   PJPG_YH2V2
} pjpeg_scan_type_t;

typedef struct
{
   // Image resolution
   int m_width;
   int m_height;
   
   // Number of components (1 or 3)
   int m_comps;
   
   // Total number of minimum coded units (MCU's) per row/col.
   int m_MCUSPerRow;
   int m_MCUSPerCol;
   
   // Scan type
   pjpeg_scan_type_t m_scanType;
   
   // MCU width/height in pixels (each is either 8 or 16 depending on the scan type)
   int m_MCUWidth;
   int m_MCUHeight;

   // m_pMCUBufR, m_pMCUBufG, and m_pMCUBufB are pointers to internal MCU Y or RGB pixel component buffers.
   // Each time pjpegDecodeMCU() is called successfully these buffers will be filled with 8x8 pixel blocks of Y or RGB pixels.
   // Each MCU consists of (m_MCUWidth/8)*(m_MCUHeight/8) Y/RGB blocks: 1 for greyscale/no subsampling, 2 for H1V2/H2V1, or 4 blocks for H2V2 sampling factors. 
   // Each block is a contiguous array of 64 (8x8) bytes of a single component: either Y for grayscale images, or R, G or B components for color images.
   //
   // The 8x8 pixel blocks are organized in these byte arrays like this:
   //
   // PJPG_GRAYSCALE: Each MCU is decoded to a single block of 8x8 grayscale pixels. 
   // Only the values in m_pMCUBufR are valid. Each 8 bytes is a row of pixels (raster order: left to right, top to bottom) from the 8x8 block.
   //
   // PJPG_H1V1: Each MCU contains is decoded to a single block of 8x8 RGB pixels.
   //
   // PJPG_YH2V1: Each MCU is decoded to 2 blocks, or 16x8 pixels.
   // The 2 RGB blocks are at byte offsets: 0, 64
   //
   // PJPG_YH1V2: Each MCU is decoded to 2 blocks, or 8x16 pixels. 
   // The 2 RGB blocks are at byte offsets: 0, 
   //                                       128
   //
   // PJPG_YH2V2: Each MCU is decoded to 4 blocks, or 16x16 pixels.
   // The 2x2 block array is organized at byte offsets:   0,  64, 
   //                                                   128, 192
   //
   // It is up to the caller to copy or blit these pixels from these buffers into the destination bitmap.
   unsigned char *m_pMCUBufR;
   unsigned char *m_pMCUBufG;
   unsigned char *m_pMCUBufB;
} pjpeg_image_info_t;

typedef unsigned char (*pjpeg_need_bytes_callback_t)(unsigned char* pBuf, unsigned char buf_size, unsigned char *pBytes_actually_read, void *pCallback_data);

// Initializes the decompressor. Returns 0 on success, or one of the above error codes on failure.
// pNeed_bytes_callback will be called to fill the decompressor's internal input buffer.
// If reduce is 1, only the first pixel of each block will be decoded. This mode is much faster because it skips the AC dequantization, IDCT and chroma upsampling of every image pixel.
// Not thread safe.
unsigned char pjpeg_decode_init(pjpeg_image_info_t *pInfo, pjpeg_need_bytes_callback_t pNeed_bytes_callback, void *pCallback_data, unsigned char reduce);

// Decompresses the file's next MCU. Returns 0 on success, PJPG_NO_MORE_BLOCKS if no more blocks are available, or an error code.
// Must be called a total of m_MCUSPerRow*m_MCUSPerCol times to completely decompress the image.
// Not thread safe.
unsigned char pjpeg_decode_mcu(void);

#ifdef __cplusplus
}
#endif

#endif // PICOJPEG_H
