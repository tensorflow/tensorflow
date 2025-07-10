#include "fully_connected_bram_driver.h"
#include <iostream>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <cstdint>
#include <cstring>
#include <stdexcept>

#define TEST_MAIN

FpgaBramDriver::FpgaBramDriver() : bram_dev_mem_fd(-1), 
    bram_mapped_input_block(nullptr), 
    bram_mapped_weight_block(nullptr), 
    bram_mapped_bias_block(nullptr), 
    bram_mapped_output_block(nullptr), 
    bram_size_other_than_weight(256), // Size for input, output, and bias BRAMs
    bram_size_weight(1024) { // Size for weight BRAM

        std::cout << "Size of int32_t: " << sizeof(int32_t) << std::endl;

        std::cout << "Size of bram_size_other_than_weight: " << sizeof(bram_size_other_than_weight) << std::endl;
        std::cout << "Size of bram_size_weight: " << sizeof(bram_size_weight) << std::endl;

        bram_input_base_address = 0x80010000;
        bram_weight_base_address = 0x80100000;
        bram_bias_base_address = 0x80011000;
        bram_output_base_address = 0x80012000;

}

FpgaBramDriver::~FpgaBramDriver() {
    if (bram_mapped_input_block != MAP_FAILED && bram_mapped_input_block != nullptr) {
        munmap(bram_mapped_input_block, bram_size_other_than_weight);
    }
    if (bram_mapped_weight_block != MAP_FAILED && bram_mapped_weight_block != nullptr) {
        munmap(bram_mapped_weight_block, bram_size_weight);
    }
    if (bram_mapped_bias_block != MAP_FAILED && bram_mapped_bias_block != nullptr) {
        munmap(bram_mapped_bias_block, bram_size_other_than_weight);
    }
    if (bram_mapped_output_block != MAP_FAILED && bram_mapped_output_block != nullptr) {
        munmap(bram_mapped_output_block, bram_size_other_than_weight);
    }
    if (bram_dev_mem_fd >= 0) {
        close(bram_dev_mem_fd);
    }
}

void FpgaBramDriver::initialize_bram(bool clear_bram = false) {
    std::cout << "Initializing BRAM..." << std::endl;

    bram_dev_mem_fd = open("/dev/mem", O_RDWR | O_SYNC);
    if (bram_dev_mem_fd < 0) {
        throw std::runtime_error("Failed to open /dev/mem");
    }

    bram_mapped_input_block = mmap(nullptr, bram_size_other_than_weight, PROT_READ | PROT_WRITE, MAP_SHARED, bram_dev_mem_fd, bram_input_base_address);
    if (bram_mapped_input_block == MAP_FAILED) {
        close(bram_dev_mem_fd);
        throw std::runtime_error("Failed to memory map input BRAM");
    }

    bram_mapped_weight_block = mmap(nullptr, bram_size_weight, PROT_READ | PROT_WRITE, MAP_SHARED, bram_dev_mem_fd, bram_weight_base_address);
    if (bram_mapped_weight_block == MAP_FAILED) {
        munmap(bram_mapped_input_block, bram_size_other_than_weight);
        close(bram_dev_mem_fd);
        throw std::runtime_error("Failed to memory map weight BRAM");
    }

    bram_mapped_bias_block = mmap(nullptr, bram_size_other_than_weight, PROT_READ | PROT_WRITE, MAP_SHARED, bram_dev_mem_fd, bram_bias_base_address);
    if (bram_mapped_bias_block == MAP_FAILED) {
        munmap(bram_mapped_input_block, bram_size_other_than_weight);
        munmap(bram_mapped_weight_block, bram_size_weight);
        close(bram_dev_mem_fd);
        throw std::runtime_error("Failed to memory map bias BRAM");
    }

    bram_mapped_output_block = mmap(nullptr, bram_size_other_than_weight, PROT_READ | PROT_WRITE, MAP_SHARED, bram_dev_mem_fd, bram_output_base_address);
    if (bram_mapped_output_block == MAP_FAILED) {
        munmap(bram_mapped_input_block, bram_size_other_than_weight);
        munmap(bram_mapped_weight_block, bram_size_weight);
        munmap(bram_mapped_bias_block, bram_size_other_than_weight);
        close(bram_dev_mem_fd);
        throw std::runtime_error("Failed to memory map output BRAM");
    }

    // Initialize BRAM address map
    bram_address["input_bram"] = bram_mapped_input_block;
    bram_address["weight_bram"] = bram_mapped_weight_block;
    bram_address["bias_bram"] = bram_mapped_bias_block;
    bram_address["output_bram"] = bram_mapped_output_block;

    if (clear_bram){
        std::cout << "Clearing BRAM..." << std::endl;
        // Clear initial values in BRAMs
        std::memset(bram_mapped_input_block, 0, bram_size_other_than_weight);
        std::memset(bram_mapped_weight_block, 0, bram_size_weight);
        std::memset(bram_mapped_bias_block, 0, bram_size_other_than_weight);
        std::memset(bram_mapped_output_block, 0, bram_size_other_than_weight);
    }
    std::cout << "BRAM initialization complete." << std::endl;
}

void FpgaBramDriver::write_to_bram(const std::string& bram_name, int32_t* ptr) {
    std::cout << "Writing to BRAM: " << bram_name << std::endl;

    if (bram_address.find(bram_name) == bram_address.end()) {
        std::cerr << "Invalid BRAM name: " << bram_name << std::endl;
        return;
    }

    void* bram_ptr = bram_address[bram_name];
    if (bram_ptr == nullptr) {
        std::cerr << "BRAM pointer is null for: " << bram_name << std::endl;
        return;
    }

    // Write data to BRAM
    if (bram_name == "weight_bram") {
        // For weight BRAM, we write the full size
        std::memcpy(bram_ptr, ptr, bram_size_weight);
    } else {
        // For other BRAMs, we write only the size defined for them
        std::cout << "Writing to non-weight BRAM: " << bram_name << std::endl;
        std::memcpy(bram_ptr, ptr, bram_size_other_than_weight);
    }
    
}
int32_t* FpgaBramDriver::read_from_bram(const std::string& bram_name) {
    std::cout << "Reading from BRAM: " << bram_name << std::endl;

    if (bram_address.find(bram_name) == bram_address.end()) {
        std::cerr << "Invalid BRAM name: " << bram_name << std::endl;
        return nullptr;
    }

    void* bram_ptr = bram_address[bram_name];
    if (bram_ptr == nullptr) {
        std::cerr << "BRAM pointer is null for: " << bram_name << std::endl;
        return nullptr;
    }

    return static_cast<int32_t*>(bram_ptr);
}

int FpgaBramDriver::write_weights_to_bram(const int32_t* weights, const int size) {
    if (size > bram_size_weight / sizeof(int32_t)) {
        std::cerr << "Error: Size exceeds weight BRAM capacity." << std::endl;
        return -1;
    }
    write_to_bram("weight_bram", const_cast<int32_t*>(weights));
    return 0;
}
int FpgaBramDriver::write_bias_to_bram(const int32_t* bias, const int size) {
    if (size > bram_size_other_than_weight / sizeof(int32_t)) {
        std::cerr << "Error: Size exceeds bias BRAM capacity." << std::endl;
        return -1;
    }
    write_to_bram("bias_bram", const_cast<int32_t*>(bias));
    return 0;
}
int FpgaBramDriver::clear_output_bram() {
    std::cout << "Clearing output BRAM..." << std::endl;
    if (bram_address.find("output_bram") == bram_address.end()) {
        std::cerr << "Invalid BRAM name: output_bram" << std::endl;
        return -1;
    }
    void* bram_ptr = bram_address["output_bram"];
    if (bram_ptr == nullptr) {
        std::cerr << "BRAM pointer is null for: output_bram" << std::endl;
        return -1;
    }
    std::memset(bram_ptr, 0, bram_size_other_than_weight);
    std::cout << "Output BRAM cleared." << std::endl;
    return 0;
}

int FpgaBramDriver::write_input_to_bram(const int32_t* input, const int size) {
    if (size > bram_size_other_than_weight / sizeof(int32_t)) {
        std::cerr << "Error: Size exceeds input BRAM capacity." << std::endl;
        return -1;
    }
    write_to_bram("input_bram", const_cast<int32_t*>(input));
    return 0;
}

int FpgaBramDriver::read_output_from_bram(int32_t* output, const int size) {
    if (size > bram_size_other_than_weight / sizeof(int32_t)) {
        std::cerr << "Error: Size exceeds output BRAM capacity." << std::endl;
        return -1;
    }
    int32_t* bram_output = read_from_bram("output_bram");
    if (bram_output) {
        std::memcpy(output, bram_output, size * sizeof(int32_t));
        return 0;
    }
    return -1;
}

