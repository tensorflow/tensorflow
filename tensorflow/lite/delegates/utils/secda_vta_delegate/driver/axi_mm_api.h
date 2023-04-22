#ifndef AXI_MM_APIv1
#define AXI_MM_APIv1

#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <unistd.h>
#include <fstream>

#define PAGE_SIZE getpagesize()
#define MM_BL 4194304

#define acc_address 0x43C00000



// #define in_addr 0x17000000
// #define wgt_addr 0x17400000
// #define bias_addr 0x17800000
// #define crf_addr 0x17c00000
// #define crx_addr 0x18000000
// #define out_addr 0x18400000

// #define wgt_addr 0x18400000
// #define bias_addr 0x18800000
// #define out_addr 0x19400000

// #define opc_addr 0x16000000
// #define uop_addr 0x16400000
// #define wgt_addr 0x17000000
// #define in_addr 0x18000000
// #define crf_addr 0x18c00000
// #define crx_addr 0x19000000
// #define out_addr 0x1a000000
// #define bias_addr 0x1b000000



#define opc_addr 0x16000000
#define uop_addr 0x16400000
#define in_addr 0x17000000
#define crf_addr 0x17400000
#define crx_addr 0x17800000
#define wgt_addr 0x18000000
#define out_addr 0x19000000
#define bias_addr 0x1a000000



void writeMappedReg(int* acc, uint32_t offset, uint32_t val) {
    void* base_addr = (void*) acc;
    *((volatile uint32_t *) (reinterpret_cast<char *>(base_addr) + offset)) = val;
}

uint32_t readMappedReg(int* acc, uint32_t offset) {
    void* base_addr = (void*) acc;
    return *((volatile uint32_t *) (reinterpret_cast<char *>(base_addr) + offset));
}

template <typename T>
T* mm_init_writable(unsigned int _dma_input_address,
                   unsigned int _dma_input_buffer_size) {

                     
  int dh = open("/dev/mem", O_RDWR | O_SYNC);
  void *dma_in_mm =
      mmap(NULL, _dma_input_buffer_size, PROT_READ | PROT_WRITE, MAP_SHARED, dh,
           _dma_input_address); // Memory map source address
//   close(dh);
  return reinterpret_cast<T*>(dma_in_mm);
}

template <typename T>
T* mm_init(unsigned int _dma_input_address,
                   unsigned int _dma_input_buffer_size) {
  int dh = open("/dev/mem", O_RDWR | O_SYNC);
  void *dma_in_mm =
      mmap(NULL, _dma_input_buffer_size, PROT_READ, MAP_SHARED, dh,
           _dma_input_address); // Memory map source address
//   close(dh);
  return reinterpret_cast<T*>(dma_in_mm);
}

int* getArray(size_t base_addr, size_t length) {
  std::fstream myfile;
  size_t virt_base = base_addr & ~(getpagesize() - 1);
  size_t virt_offset = base_addr - virt_base;
  int fd = open("/dev/mem", O_RDWR | O_SYNC);
  void* addr = mmap(NULL, length + virt_offset, PROT_READ | PROT_WRITE,
                    MAP_SHARED, fd, virt_base);
  close(fd);
  if (addr == (void*)-1) exit(EXIT_FAILURE);
  int* array = reinterpret_cast<int*>(addr);
  return array;
}

template <typename T>
void copytoMM(T* dst_buf, T* src_buf, int size) {
    // cout << size << endl;
    for(int i=0; i<size; i++){
        dst_buf[i] = src_buf[i];
        // std::cout << dst_buf[i]  << std::endl;
    }
}

// template <typename T>
// void copytoMM(T* dst_buf, T* src_buf, int size, bool print) {
//     if(print)cout << "-------------"  << endl;
//     for(int i=0; i<size; i++){
//         dst_buf[i] = src_buf[i];
//         if(print)cout << i << ": " << dst_buf[i]  << endl;
//     }
//     if(print)cout << "-------------"  << endl;
// }

#endif 