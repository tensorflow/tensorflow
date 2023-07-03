
#ifndef ACC_CONFIG_H2
#define ACC_CONFIG_H2




// #define INP_BUF_LEN 2048
// #define WGT_BUF_LEN 2048 * 4
// #define UF 16

// // Number of PEs
// #define PE_COUNT 3

// // needs to support ks * ks * depth / UF
// #define PE_WGTCOLBUF_SIZE 128

// // wgt_col_sum needs to support ks * ks
// #define PE_WGTCOLSUMBUF_SIZE 128

// // inp_row_buf needs to support depth / UF
// #define PE_INPROWBUF_SIZE 128

// // support ir * ks * ks gemm outputs where ir is the number
// // of input rows
// #define PE_OUTBUF_SIZE 128

// // max value is ks * ks
// #define PE_POUTDEXBUF_SIZE 128

// // Max number of MM2IM outputs storable per PE, should allow OH * OW
// #define PE_ACC_BUF_SIZE 2048


#define INP_BUF_LEN 2048
#define WGT_BUF_LEN 2048 * 4
#define UF 16

// Number of PEs
#define PE_COUNT 8

// needs to support ks * ks * depth / UF
#define PE_WGTCOLBUF_SIZE 512

// wgt_col_sum needs to support ks * ks
#define PE_WGTCOLSUMBUF_SIZE 64

// inp_row_buf needs to support depth / UF
#define PE_INPROWBUF_SIZE 16

// support ir * ks * ks gemm outputs where ir is the number
// of input rows
#define PE_OUTBUF_SIZE 1024

// max value is ks * ks
#define PE_POUTDEXBUF_SIZE 64

// Max number of MM2IM outputs storable per PE, should allow OH * OW
#define PE_ACC_BUF_SIZE 256

#define MAX 2147483647
#define MIN -2147483648
#define POS 1073741824
#define NEG -1073741823
#define DIVMAX 2147483648
#define MAX8 127
#define MIN8 -128

#define acc_address 0x43C00000
#define dma_addr0 0x40400000
#define dma_addr1 0x40410000
#define dma_addr2 0x40420000
#define dma_addr3 0x40430000
#define dma_in0 0x16000000
#define dma_in1 0x18000000
#define dma_in2 0x1a000000
#define dma_in3 0x1c000000
#define dma_out0 0x16800000
#define dma_out1 0x18800000
#define dma_out2 0x1a800000
#define dma_out3 0x1c800000
#define DMA_BL 4194304

#endif // ACC_CONFIG_H2