#include <iostream>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <cstdint>
#include <cstring>
#include <map>

class FpgaIpDriver {
private:
    int dev_mem_fd;
    void* mapped_fpga_ip;  // Store mapped pointer here
    size_t size = 4096;    // Size of mapping (should be at least one page)

    uint32_t ip_base_address = 0x80000000;

    uint32_t ip_input_1_offset = 0x18;
    uint32_t ip_input_2_offset = 0x20;
    uint32_t ip_output_offset = 0x10;
    uint32_t ip_add_flag_offset = 0x28;
    uint32_t ip_control_register_offset = 0x00;

    static constexpr uint32_t CONTROL_START    = 0x01;
    static constexpr uint32_t CONTROL_CONTINUE = 0x10; 

    std::map<std::string, uint32_t> ip_address;

public:
    FpgaIpDriver() : dev_mem_fd(-1), mapped_fpga_ip(nullptr) {
        initialize_fpga(); // Initialize FPGA on construction
    } //Member initializer list

    ~FpgaIpDriver() {
        if (mapped_fpga_ip != MAP_FAILED && mapped_fpga_ip != nullptr) {
            munmap(mapped_fpga_ip, size);
        }
        if (dev_mem_fd >= 0) {
            close(dev_mem_fd);
        }
    }

    // Initialize the FPGA IP driver
    void initialize_fpga() {
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
        ip_address["input_1"] = ip_input_1_offset;
        ip_address["input_2"] = ip_input_2_offset;
        ip_address["output"] = ip_output_offset;
        ip_address["add_flag"] = ip_add_flag_offset;
        ip_address["control_register"] = ip_control_register_offset;

        // Clear initial values
        write_to_fpga("input_1", 0);
        write_to_fpga("input_2", 0);
        write_to_fpga("add_flag", 0);
    }

    void write_to_fpga(const std::string& reg_name, uint32_t value) {
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

    int32_t read_from_fpga(const std::string& reg_name) {
        if (!mapped_fpga_ip || ip_address.find(reg_name) == ip_address.end()) return 0;

        uint32_t offset = ip_address[reg_name];
        volatile uint32_t* ptr = reinterpret_cast<volatile uint32_t*>(
            reinterpret_cast<uint8_t*>(mapped_fpga_ip) + offset);
        return *ptr;
    }
    
    

    int32_t fpga_compute(int32_t input_1, int32_t input_2, bool add_flag) {
        write_to_fpga("input_1", input_1);
        write_to_fpga("input_2", input_2);
        write_to_fpga("add_flag", add_flag ? 1 : 0);

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


        write_to_fpga("control_register", CONTROL_CONTINUE); // Continue computation

        return read_from_fpga("output");
    }

};

//test code
// int main() {
//     try {
//         FpgaIpDriver driver;
//         int32_t input_1, input_2;
//         bool add_flag;
//         // take user inputs for computation
//         std::cout << "Enter two integers to add/subtract: " << std::endl;
//         std::cin >> input_1 >> input_2; // Get user input for input_1 and input_2
//         std::cout << "Enter 1 for addition or 0 for subtraction: ";
//         std::cin >> add_flag; // Get user input for add_flag


//         if (std::cin.fail()) {
//             std::cerr << "Invalid input. Please enter integers." << std::endl;
//             return 1;
//         }

//         int32_t result = driver.fpga_compute(input_1, input_2, add_flag);

//         std::cout << "Result from FPGA: " << result << std::endl;
//     } catch (const std::exception& ex) {
//         std::cerr << "Exception: " << ex.what() << std::endl;
//         return 1;
//     }

//     return 0;
// }   
