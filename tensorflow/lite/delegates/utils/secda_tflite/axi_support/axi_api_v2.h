#ifndef AXI_API_V2_H
#define AXI_API_V2_H

#ifdef SYSC
#include "../sysc_integrator/axi4s_engine.sc.h"
#endif

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <chrono>
#include <fstream>
#include <iostream>
#include <string>
#include <typeinfo>
#include <vector>

// TODO: Remove hardcode addresses, make it cleaner
using namespace std;
#define MM2S_CONTROL_REGISTER 0x00
#define MM2S_STATUS_REGISTER 0x04
#define MM2S_START_ADDRESS 0x18
#define MM2S_LENGTH 0x28

#define S2MM_CONTROL_REGISTER 0x30
#define S2MM_STATUS_REGISTER 0x34
#define S2MM_DESTINATION_ADDRESS 0x48
#define S2MM_LENGTH 0x58
#define PAGE_SIZE getpagesize()

// TODO: Clean up code and seperate to AXI4Lite, AXI4S, AXI4MM

// ================================================================================
// AXI4Lite API
// ================================================================================
template <typename T>
T* getAccBaseAddress(size_t base_addr, size_t length) {
  int fd = open("/dev/mem", O_RDWR | O_SYNC);
  size_t virt_base = base_addr & ~(getpagesize() - 1);
  size_t virt_offset = base_addr - virt_base;
  void* addr = mmap(NULL, length + virt_offset, PROT_READ | PROT_WRITE,
                    MAP_SHARED, fd, virt_base);
  close(fd);
  if (addr == (void*)-1) exit(EXIT_FAILURE);
  T* acc = reinterpret_cast<T*>(addr);
  return acc;
}

template <typename T>
void writeMappedReg(int* acc, uint32_t offset, T val) {
  void* base_addr = (void*)acc;
  *((volatile T*)(reinterpret_cast<char*>(base_addr) + offset)) = val;
}

template <typename T>
T readMappedReg(int* acc, uint32_t offset) {
  void* base_addr = (void*)acc;
  return *((volatile T*)(reinterpret_cast<char*>(base_addr) + offset));
}

struct acc_regmap {
  int* acc_addr;

  uint32_t* control_registers_offset;
  uint32_t* status_registers_offset;

  string* control_registers;
  string* status_registers;

  acc_regmap(size_t base_addr, size_t length);

  void writeAccReg(uint32_t offset, unsigned int val);

  unsigned int readAccReg(uint32_t offset);

  // TODO: parse JSON file to load offset map for control and status registers
  void parseOffsetJSON();

  // TODO: checks control and status register arrays to find the offsets for the
  // register
  uint32_t findRegOffset(string reg_name);

  void writeToControlReg(string reg_name, unsigned int val);

  unsigned int readToControlReg(string reg_name);
};
// ================================================================================
// Memory Map API
// ================================================================================
template <typename T>
T* mm_alloc_rw(unsigned int address, unsigned int buffer_size) {
  int fd = open("/dev/mem", O_RDWR | O_SYNC);
  size_t virt_base = address & ~(getpagesize() - 1);
  size_t virt_offset = address - virt_base;
  void* addr = mmap(NULL, buffer_size + virt_offset, PROT_READ | PROT_WRITE,
                    MAP_SHARED, fd, virt_base);
  close(fd);
  if (addr == (void*)-1) exit(EXIT_FAILURE);
  T* acc = reinterpret_cast<T*>(addr);
  return acc;

  // int dh = open("/dev/mem", O_RDWR | O_SYNC);
  // void* mm =
  //     mmap(NULL, buffer_size, PROT_READ | PROT_WRITE, MAP_SHARED, dh,
  //     address);
  // close(dh);
  // if (mm == (void*)-1) exit(EXIT_FAILURE);
  // return reinterpret_cast<T*>(mm);
}

template <typename T>
T* mm_alloc_r(unsigned int address, unsigned int buffer_size) {
  int fd = open("/dev/mem", O_RDWR | O_SYNC);
  size_t virt_base = address & ~(getpagesize() - 1);
  size_t virt_offset = address - virt_base;
  void* addr = mmap(NULL, buffer_size + virt_offset, PROT_READ, MAP_SHARED, fd,
                    virt_base);
  close(fd);
  if (addr == (void*)-1) exit(EXIT_FAILURE);
  T* acc = reinterpret_cast<T*>(addr);
  return acc;
  // int dh = open("/dev/mem", O_RDWR | O_SYNC);
  // void* mm = mmap(NULL, buffer_size, PROT_READ, MAP_SHARED | MAP_NORESERVE,
  // dh,
  //                 address);
  // close(dh);
  // if (mm == (void*)-1) exit(EXIT_FAILURE);
  // return reinterpret_cast<T*>(mm);
}

// struct mm_dma {
// #ifdef SYSC
//   AXI4MM_ENGINE* mmdma;
// #endif
// };

// ================================================================================
// Stream DMA API
// ================================================================================
struct stream_dma {
  unsigned int* dma_addr;
  int* input;
  int* output;
  unsigned int input_size;
  unsigned int output_size;

  unsigned int input_addr;
  unsigned int output_addr;

  static int s_id;
  const int id;

#ifdef SYSC
  // AXIS_ENGINE rdmad;
  AXIS_ENGINE* dmad;
#endif

  stream_dma(unsigned int _dma_addr, unsigned int _input,
             unsigned int _input_size, unsigned int _output,
             unsigned int _output_size);

  stream_dma();

  void dma_init(unsigned int _dma_addr, unsigned int _input,
                unsigned int _input_size, unsigned int _output,
                unsigned int _output_size);

  void initDMA(unsigned int src, unsigned int dst);

  void dma_free();

  void dma_change_start(int offset);

  void dma_change_end(int offset);

  int* dma_get_inbuffer();

  int* dma_get_outbuffer();

  void dma_start_send(int length);

  void dma_wait_send();

  int dma_check_send();

  void dma_start_recv(int length);

  void dma_wait_recv();

  int dma_check_recv();

  //********************************** Unexposed Functions
  //**********************************

  void writeMappedReg(uint32_t offset, unsigned int val);
  unsigned int readMappedReg(uint32_t offset);
  void dma_mm2s_sync();
  void dma_s2mm_sync();
};

struct multi_dma {
  struct stream_dma* dmas;
  unsigned int* dma_addrs;
  unsigned int* dma_addrs_in;
  unsigned int* dma_addrs_out;
  unsigned int buffer_size;
  int dma_count;

  multi_dma(int _dma_count, unsigned int* _dma_addrs,
            unsigned int* _dma_addrs_in, unsigned int* _dma_addrs_out,
            unsigned int buffer_size);

  void multi_free_dmas();

  void multi_dma_change_start(int offset);

  void multi_dma_change_start_4(int offset);

  void multi_dma_change_end(int offset);

  void multi_dma_start_send(int length);

  void multi_dma_wait_send();

  int multi_dma_check_send();

  void multi_dma_start_recv(int length);

  void multi_dma_start_recv();

  void multi_dma_wait_recv();

  void multi_dma_wait_recv_4();

  int multi_dma_check_recv();
};

#endif