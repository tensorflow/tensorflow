#include "fully_connected_ip_driver.h"
#include <iostream>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <cstdint>
#include <cstring>
#include <stdexcept>

#define FULLY_CONNECTED_IP_DRIVER_TEST_MAIN

FullyConnectedIpDriver::FullyConnectedIpDriver() : dev_mem_fd(-1), mapped_fpga_ip(nullptr), size(4096) {//one page size
    // Initialize member variables
    ip_base_address = 0x80000000;
    ip_input_size_offset = 0x10;
    ip_output_size_offset = 0x18;
    ip_control_register_offset = 0x00;
    
    initialize_fpga(); // Initialize FPGA on construction
}

FullyConnectedIpDriver::~FullyConnectedIpDriver() {
    if (mapped_fpga_ip != MAP_FAILED && mapped_fpga_ip != nullptr) {
        munmap(mapped_fpga_ip, size);
    }
    if (dev_mem_fd >= 0) {
        close(dev_mem_fd);
    }
}

void FullyConnectedIpDriver::initialize_fpga() {
    std::cout << "Initializing FPGA..." << std::endl;

    dev_mem_fd = open("/dev/mem", O_RDWR | O_SYNC);
    if (dev_mem_fd < 0) {
        std::cerr << "Failed to open /dev/mem" << std::endl;
        return;
    }

    mapped_fpga_ip = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, dev_mem_fd, ip_base_address);
    if (mapped_fpga_ip == MAP_FAILED) {
        std::cerr << "Failed to memory map FPGA IP" << std::endl;
        close(dev_mem_fd);
        return;
    }

    // Initialize offset map (just storing offsets, not absolute addresses)
    ip_address["input_size"] = ip_input_size_offset;
    ip_address["output_size"] = ip_output_size_offset;
    ip_address["control_register"] = ip_control_register_offset;

    std::cout << "FPGA initialization complete." << std::endl;
}

void FullyConnectedIpDriver::write_to_fpga(const std::string& reg_name, uint32_t value) {
    if (!mapped_fpga_ip || mapped_fpga_ip == MAP_FAILED) {
        std::cerr << "Memory not mapped" << std::endl;
        return;
    }
    if (ip_address.find(reg_name) == ip_address.end()) {
        std::cerr << "Invalid register name: " << reg_name << std::endl;
        return;
    }   
    uint32_t offset = ip_address[reg_name];
    volatile uint32_t* ptr = reinterpret_cast<volatile uint32_t*>(
        reinterpret_cast<uint8_t*>(mapped_fpga_ip) + offset
    );
    *ptr = value;

    std::cout << "Wrote 0x" << std::hex << value
              << " to FPGA register [" << reg_name << "] at offset 0x"
              << offset << std::endl;
}

int32_t FullyConnectedIpDriver::read_from_fpga(const std::string& reg_name) {
    if (!mapped_fpga_ip || ip_address.find(reg_name) == ip_address.end()) return 0;

    uint32_t offset = ip_address[reg_name];
    volatile uint32_t* ptr = reinterpret_cast<volatile uint32_t*>(
        reinterpret_cast<uint8_t*>(mapped_fpga_ip) + offset);
    return *ptr;
}

int FullyConnectedIpDriver::fpga_compute(int32_t input_size, int32_t output_size) {
    write_to_fpga("input_size", input_size);
    write_to_fpga("output_size", output_size);
    
    // Trigger computation by writing to control register
    write_to_fpga("control_register", CONTROL_START); // Start computation

    // Polling for done (bit 1)
    int count = 0;
    uint32_t ctrl_val = 0;
    do {
        ctrl_val = read_from_fpga("control_register");
        if (++count > 10000) {
            throw std::runtime_error("Timeout waiting for FPGA computation to finish");
        }
    } while ((ctrl_val & (1 << 1)) == 0); // check for done bit.

    write_to_fpga("control_register", CONTROL_CONTINUE); // Clear done bit by writing CONTROL_CONTINUE

    return 0; // Success
}


#ifdef FULLY_CONNECTED_IP_DRIVER_TEST_MAIN
int main() {
    FullyConnectedIpDriver driver;


    std::cout << "Enter input size: ";
    int32_t input_size;
    std::cin >> input_size;

    std::cout << "Enter output size: ";
    int32_t output_size;
    std::cin >> output_size;

    // Test FPGA write and read
    driver.fpga_compute(input_size, output_size);

    return 0;
}
#endif