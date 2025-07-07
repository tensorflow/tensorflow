#include "fully_connected_bram_driver.h"
#include <iostream>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <cstdint>
#include <cstring>
#include <stdexcept>

#define TEST_MAIN

FullyConnectedBRAMDriver::FullyConnectedBRAMDriver() : bram_dev_mem_fd(-1), 
    bram_mapped_input_block(nullptr), 
    bram_mapped_weight_block(nullptr), 
    bram_mapped_bias_block(nullptr), 
    bram_mapped_output_block(nullptr), 
    bram_size(4096) {

        bram_input_base_address = 0x80020000;
        bram_weight_base_address = 0x80022000;
        bram_bias_base_address = 0x80024000;
        bram_output_base_address = 0x80026000;

        initialize_bram();
}

FullyConnectedBRAMDriver::~FullyConnectedBRAMDriver() {
    if (bram_mapped_input_block != MAP_FAILED && bram_mapped_input_block != nullptr) {
        munmap(bram_mapped_input_block, bram_size);
    }
    if (bram_mapped_weight_block != MAP_FAILED && bram_mapped_weight_block != nullptr) {
        munmap(bram_mapped_weight_block, bram_size);
    }
    if (bram_mapped_bias_block != MAP_FAILED && bram_mapped_bias_block != nullptr) {
        munmap(bram_mapped_bias_block, bram_size);
    }
    if (bram_mapped_output_block != MAP_FAILED && bram_mapped_output_block != nullptr) {
        munmap(bram_mapped_output_block, bram_size);
    }
    if (bram_dev_mem_fd >= 0) {
        close(bram_dev_mem_fd);
    }
}

void FullyConnectedBRAMDriver::initialize_bram() {
    std::cout << "Initializing BRAM..." << std::endl;

    bram_dev_mem_fd = open("/dev/mem", O_RDWR | O_SYNC);
    if (bram_dev_mem_fd < 0) {
        throw std::runtime_error("Failed to open /dev/mem");
    }

    bram_mapped_input_block = mmap(nullptr, bram_size, PROT_READ | PROT_WRITE, MAP_SHARED, bram_dev_mem_fd, bram_input_base_address);
    if (bram_mapped_input_block == MAP_FAILED) {
        close(bram_dev_mem_fd);
        throw std::runtime_error("Failed to memory map input BRAM");
    }

    bram_mapped_weight_block = mmap(nullptr, bram_size, PROT_READ | PROT_WRITE, MAP_SHARED, bram_dev_mem_fd, bram_weight_base_address);
    if (bram_mapped_weight_block == MAP_FAILED) {
        munmap(bram_mapped_input_block, bram_size);
        close(bram_dev_mem_fd);
        throw std::runtime_error("Failed to memory map weight BRAM");
    }

    bram_mapped_bias_block = mmap(nullptr, bram_size, PROT_READ | PROT_WRITE, MAP_SHARED, bram_dev_mem_fd, bram_bias_base_address);
    if (bram_mapped_bias_block == MAP_FAILED) {
        munmap(bram_mapped_input_block, bram_size);
        munmap(bram_mapped_weight_block, bram_size);
        close(bram_dev_mem_fd);
        throw std::runtime_error("Failed to memory map bias BRAM");
    }

    bram_mapped_output_block = mmap(nullptr, bram_size, PROT_READ | PROT_WRITE, MAP_SHARED, bram_dev_mem_fd, bram_output_base_address);
    if (bram_mapped_output_block == MAP_FAILED) {
        munmap(bram_mapped_input_block, bram_size);
        munmap(bram_mapped_weight_block, bram_size);
        munmap(bram_mapped_bias_block, bram_size);
        close(bram_dev_mem_fd);
        throw std::runtime_error("Failed to memory map output BRAM");
    }

    // Initialize BRAM address map
    bram_address["input_bram"] = bram_mapped_input_block;
    bram_address["weight_bram"] = bram_mapped_weight_block;
    bram_address["bias_bram"] = bram_mapped_bias_block;
    bram_address["output_bram"] = bram_mapped_output_block;

    // //clear initial values in BRAM
    // std::memset(bram_mapped_input_block, 0, bram_size);
    // std::memset(bram_mapped_weight_block, 0, bram_size);
    // std::memset(bram_mapped_bias_block, 0, bram_size);
    // std::memset(bram_mapped_output_block, 0, bram_size);

    std::cout << "BRAM initialization complete." << std::endl;
}

void FullyConnectedBRAMDriver::write_to_bram(const std::string& bram_name, uint32_t* ptr) {
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
    std::memcpy(bram_ptr, ptr, bram_size);
}
uint32_t* FullyConnectedBRAMDriver::read_from_bram(const std::string& bram_name) {
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

    return static_cast<uint32_t*>(bram_ptr);
}

#ifdef TEST_MAIN
int main() {
    FullyConnectedBRAMDriver bram_driver;

    // Test BRAM write and read
    uint32_t test_data[32] = {0};
    uint32_t test_data_weight[1024] = {0};
    for (int i = 0; i < 32; ++i) {
        test_data[i] = i;
    }
    for (int i = 0; i < 1024; ++i) {
        test_data_weight[i] = i + 100; // Different data for weights
    }
    bram_driver.write_to_bram("input_bram", test_data);
    uint32_t* read_data = bram_driver.read_from_bram("input_bram");
    if (read_data) {
        for (int i = 0; i < 32; ++i) {
            std::cout << "Read data[" << i << "] = " << read_data[i] << std::endl;
        }
    }
    bram_driver.write_to_bram("weight_bram", test_data_weight);
    read_data = bram_driver.read_from_bram("weight_bram");
    if (read_data) {
        for (int i = 0; i < 1024; ++i) {
            std::cout << "Read data[" << i << "] = " << read_data[i] << std::endl;
        }
    }
    bram_driver.write_to_bram("bias_bram", test_data);
    read_data = bram_driver.read_from_bram("bias_bram");
    if (read_data) {
        for (int i = 0; i < 32; ++i) {
            std::cout << "Read data[" << i << "] = " << read_data[i] << std::endl;
        }
    }
    bram_driver.write_to_bram("output_bram", test_data);
    read_data = bram_driver.read_from_bram("output_bram");
    if (read_data) {
        for (int i = 0; i < 32; ++i) {
            std::cout << "Read data[" << i << "] = " << read_data[i] << std::endl;
        }
    }

    return 0;
}
#endif