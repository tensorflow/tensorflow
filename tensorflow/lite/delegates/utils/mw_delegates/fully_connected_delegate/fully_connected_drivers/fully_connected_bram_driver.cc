#include "fully_connected_bram_driver.h"
#include <iostream>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <cstdint>
#include <cstring>
#include <stdexcept>

#define BRAM_SIZE_INPUT 4096 // Size for input BRAM 4K
#define BRAM_SIZE_WEIGHT 262144 // Size for weight BRAM 256K
#define BRAM_SIZE_BIAS 4096 // Size for bias BRAM 4K
#define BRAM_SIZE_OUTPUT 4096 // Size for output BRAM 4K

FpgaBramDriver::FpgaBramDriver() : bram_dev_mem_fd(-1), 
    bram_mapped_input_block(nullptr), 
    bram_mapped_weight_block(nullptr), 
    bram_mapped_bias_block(nullptr), 
    bram_mapped_output_block(nullptr) { // Size for weight BRAM

        std::cout << "Size of float: " << sizeof(float) <<" bytes" << std::endl;

        bram_input_base_address = 0x80010000;
        bram_weight_base_address = 0x80100000;
        bram_bias_base_address = 0x80011000;
        bram_output_base_address = 0x80012000;

        max_supported_non_weight_dimension = 32; // Size for input, output, and bias BRAMs // can support more but capped for now
        max_supported_weight_dimension = 32 * 32; // Size for weight BRAM // can support more but capped for now

}

FpgaBramDriver::~FpgaBramDriver() {
    if (bram_mapped_input_block != MAP_FAILED && bram_mapped_input_block != nullptr) {
        munmap(bram_mapped_input_block, BRAM_SIZE_INPUT);
    }
    if (bram_mapped_weight_block != MAP_FAILED && bram_mapped_weight_block != nullptr) {
        munmap(bram_mapped_weight_block, BRAM_SIZE_WEIGHT);
    }
    if (bram_mapped_bias_block != MAP_FAILED && bram_mapped_bias_block != nullptr) {
        munmap(bram_mapped_bias_block, BRAM_SIZE_BIAS);
    }
    if (bram_mapped_output_block != MAP_FAILED && bram_mapped_output_block != nullptr) {
        munmap(bram_mapped_output_block, BRAM_SIZE_OUTPUT);
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

    bram_mapped_input_block = mmap(nullptr, BRAM_SIZE_INPUT, PROT_READ | PROT_WRITE, MAP_SHARED, bram_dev_mem_fd, bram_input_base_address);
    if (bram_mapped_input_block == MAP_FAILED) {
        close(bram_dev_mem_fd);
        throw std::runtime_error("Failed to memory map input BRAM");
    }

    bram_mapped_weight_block = mmap(nullptr, BRAM_SIZE_WEIGHT, PROT_READ | PROT_WRITE, MAP_SHARED, bram_dev_mem_fd, bram_weight_base_address);
    if (bram_mapped_weight_block == MAP_FAILED) {
        munmap(bram_mapped_input_block, BRAM_SIZE_INPUT);
        close(bram_dev_mem_fd);
        throw std::runtime_error("Failed to memory map weight BRAM");
    }

    bram_mapped_bias_block = mmap(nullptr, BRAM_SIZE_BIAS, PROT_READ | PROT_WRITE, MAP_SHARED, bram_dev_mem_fd, bram_bias_base_address);
    if (bram_mapped_bias_block == MAP_FAILED) {
        munmap(bram_mapped_input_block, BRAM_SIZE_INPUT);
        munmap(bram_mapped_weight_block, BRAM_SIZE_WEIGHT);
        close(bram_dev_mem_fd);
        throw std::runtime_error("Failed to memory map bias BRAM");
    }

    bram_mapped_output_block = mmap(nullptr, BRAM_SIZE_OUTPUT, PROT_READ | PROT_WRITE, MAP_SHARED, bram_dev_mem_fd, bram_output_base_address);
    if (bram_mapped_output_block == MAP_FAILED) {
        munmap(bram_mapped_input_block, BRAM_SIZE_INPUT);
        munmap(bram_mapped_weight_block, BRAM_SIZE_WEIGHT);
        munmap(bram_mapped_bias_block, BRAM_SIZE_BIAS);
        close(bram_dev_mem_fd);
        throw std::runtime_error("Failed to memory map output BRAM");
    }

    // Initialize BRAM address map
    bram_address["input_bram"] = bram_mapped_input_block;
    bram_address["weight_bram"] = bram_mapped_weight_block;
    bram_address["bias_bram"] = bram_mapped_bias_block;
    bram_address["output_bram"] = bram_mapped_output_block;

    //TODO: BUS error occurs here when clearing BRAM
    // // if (clear_bram){
    //     std::cout << "Clearing BRAM..." << std::endl;
    //     // Clear initial values in BRAMs
    //     std::memset(bram_mapped_input_block, 0, max_supported_non_weight_dimension * sizeof(float)); // Clear input BRAM
    //     std::memset(bram_mapped_weight_block, 0, max_supported_weight_dimension * sizeof(float)); // Clear weight BRAM
    //     std::memset(bram_mapped_bias_block, 0, max_supported_non_weight_dimension * sizeof(float)); // Clear bias BRAM
    //     std::memset(bram_mapped_output_block, 0, max_supported_non_weight_dimension * sizeof(float)); // Clear output BRAM
    // // }
    std::cout << "BRAM initialization complete." << std::endl;
}

void FpgaBramDriver::write_to_bram(const std::string& bram_name, float* ptr, size_t num_elements) {
    std::cout << "Writing to BRAM: " << bram_name << " (elements: " << num_elements << ")" << std::endl;

    if (bram_address.find(bram_name) == bram_address.end()) {
        std::cerr << "Invalid BRAM name: " << bram_name << std::endl;
        return;
    }

    void* bram_ptr = bram_address[bram_name];
    if (bram_ptr == nullptr) {
        std::cerr << "BRAM pointer is null for: " << bram_name << std::endl;
        return;
    }

    // Calculate the size in bytes
    size_t bytes_to_write = num_elements * sizeof(float);
    
    // Check capacity
    size_t bram_capacity_bytes = (bram_name == "weight_bram") ? max_supported_weight_dimension * sizeof(float) : max_supported_non_weight_dimension * sizeof(float);

    if (bytes_to_write > bram_capacity_bytes) {
        std::cerr << "Error: Data size (" << bytes_to_write << " bytes) exceeds BRAM capacity (" << bram_capacity_bytes << " bytes)" << std::endl;
        return;
    }

    if (ptr != nullptr) {
        std::cout << "bram_ptr address: 0x" << std::hex << bram_ptr << " ptr address: 0x" << std::hex << ptr << std::endl;
        std::memcpy(bram_ptr, ptr, bytes_to_write);
    }
    std::cout << "Data written to BRAM: " << bram_name << " successfully." << std::endl;
}
float* FpgaBramDriver::read_from_bram(const std::string& bram_name) {
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

    return static_cast<float*>(bram_ptr);
}

int FpgaBramDriver::write_weights_to_bram(const float* weights, const int size) {
    std::cout << "size of weights: " << size << std::endl;
    if (size > max_supported_weight_dimension) {
        std::cerr << "Error: Size exceeds weight BRAM capacity." << std::endl;
        return -1;
    }
    write_to_bram("weight_bram", const_cast<float*>(weights), size);
    return 0;
}
int FpgaBramDriver::write_bias_to_bram(const float* bias, const int size) {

    std::cout << "size of bias: " << size << std::endl;
    if (size > max_supported_non_weight_dimension) {
        std::cerr << "Error: Size exceeds bias BRAM capacity." << std::endl;
        return -1;
    }
    write_to_bram("bias_bram", const_cast<float*>(bias), size);
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
    std::memset(bram_ptr, 0, max_supported_non_weight_dimension * sizeof(float)); // Clear the output BRAM
    std::cout << "Output BRAM cleared." << std::endl;
    return 0;
}

int FpgaBramDriver::write_input_to_bram(const float* input, const int size) {
    std::cout << "size of input: " << size << std::endl;

    if (size > max_supported_non_weight_dimension) {
        std::cerr << "Error: Size exceeds input BRAM capacity." << std::endl;
        return -1;
    }
    write_to_bram("input_bram", const_cast<float*>(input), size);
    return 0;
}

int FpgaBramDriver::read_output_from_bram(float* output, const int size) {
    if (size > max_supported_non_weight_dimension) {
        std::cerr << "Error: Size exceeds output BRAM capacity." << std::endl;
        return -1;
    }
    //TODO: REMOVE THIS LINE

    output = reinterpret_cast<float*>(bram_mapped_input_block);
    ////////////////////////////////////
    float* bram_output = read_from_bram("output_bram");
    std::cout << "Entering If : going to memcpy" << std::endl;
    if (bram_output) {
        std::cout << "output address: 0x"<<std::hex << output << " bram_output address: 0x"<<std::hex << bram_output << std::endl;
        std::memcpy(output, bram_output, size * sizeof(float));
        std::cout << "memcpy successful" << std::endl;
        return 0;
    }
    return -1;
}

int FpgaBramDriver::test_read_input_bram(float* output, const int size) {
    if (size > max_supported_non_weight_dimension) {
        std::cerr << "Error: Size exceeds input BRAM capacity." << std::endl;
        return -1;
    }
    float* bram_input = read_from_bram("input_bram");
    if (bram_input) {
        std::memcpy(output, bram_input, size * sizeof(float));
        return 0;
    }
    return -1;
}

int FpgaBramDriver::test_read_weights_bram(float* output, const int size) {
    if (size > max_supported_weight_dimension) {
        std::cerr << "Error: Size exceeds weight BRAM capacity." << std::endl;
        return -1;
    }
    float* bram_weights = read_from_bram("weight_bram");
    if (bram_weights) {
        std::memcpy(output, bram_weights, size * sizeof(float));
        return 0;
    }
    return -1;
}

int FpgaBramDriver::test_read_bias_bram(float* output, const int size) {
    std::cout << "Reading from BRAM: bias_bram (size = " << size << ")" << std::endl;

    if (size > max_supported_non_weight_dimension) {
        std::cerr << "Error: Size exceeds bias BRAM capacity." << std::endl;
        return -1;
    }

    float* bram_bias = read_from_bram("bias_bram");
    if (bram_bias == nullptr) {
        std::cerr << "BRAM pointer is null for bias_bram" << std::endl;
        return -1;
    }

    std::memcpy(output, bram_bias, size * sizeof(float));
    return 0;
}