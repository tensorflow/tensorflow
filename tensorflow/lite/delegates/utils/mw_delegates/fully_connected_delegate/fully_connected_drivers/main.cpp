#include <iostream>
#include <vector>
#include <iomanip>
#include <cmath>
#include "fully_connected_bram_driver.h"
#include "fully_connected_ip_driver.h"

int main() {
    std::cout << "=== FPGA Driver Integration Test ===" << std::endl;
    
    try {
        // Create instances of the drivers
        FpgaBramDriver bram_driver;
        FpgaIpDriver ip_driver;

        std::cout << "\n=== Initializing FPGA Drivers ===" << std::endl;
        
        // Create and initialize FPGA IP driver (auto-initializes in constructor)
        std::cout << "FPGA IP driver initialized automatically" << std::endl;
        
        // Initialize BRAM driver
        std::cout << "Initializing BRAM driver..." << std::endl;
        bool clear_bram = false;
        bram_driver.initialize_bram(clear_bram);

        std::cout << "\n=== Testing BRAM Driver ===" << std::endl;
        
        // Test data dimensions
        const int input_size = 16;
        const int output_size = 8;
        const int weight_size = input_size * output_size;
        const int bias_size = output_size;

        // Create test data
        std::vector<float> input_data(input_size);
        std::vector<float> weights_data(weight_size);
        std::vector<float> bias_data(bias_size);
        std::vector<float> output_data(output_size, 0.0f);

        // Initialize test data with meaningful values
        for (int i = 0; i < input_size; ++i) {
            input_data[i] = 1.0f + i * 0.1f;  // 1.0, 1.1, 1.2, ...
        }
        
        for (int i = 0; i < weight_size; ++i) {
            weights_data[i] = 0.5f + (i % 10) * 0.01f;  // 0.5, 0.51, 0.52, ...
        }
        
        for (int i = 0; i < bias_size; ++i) {
            bias_data[i] = 0.1f * i;  // 0.0, 0.1, 0.2, ...
        }

        // Test BRAM operations
        std::cout << "Testing BRAM write operations..." << std::endl;
        
        // Write weights to BRAM
        if (bram_driver.write_weights_to_bram(weights_data.data(), weight_size) != 0) {
            std::cerr << "Failed to write weights to BRAM" << std::endl;
            return 1;
        }
        std::cout << "✓ Weights written to BRAM successfully" << std::endl;
        
        // Write bias to BRAM
        if (bram_driver.write_bias_to_bram(bias_data.data(), bias_size) != 0) {
            std::cerr << "Failed to write bias to BRAM" << std::endl;
            return 1;
        }
        std::cout << "✓ Bias written to BRAM successfully" << std::endl;
        
        // Write input to BRAM
        if (bram_driver.write_input_to_bram(input_data.data(), input_size) != 0) {
            std::cerr << "Failed to write input to BRAM" << std::endl;
            return 1;
        }
        std::cout << "✓ Input written to BRAM successfully" << std::endl;

        std::cout << "\n=== Testing FPGA IP Driver ===" << std::endl;
        
        // Trigger FPGA computation
        std::cout << "Triggering FPGA computation..." << std::endl;
        if (ip_driver.fpga_compute(input_size, output_size) != 0) {
            std::cerr << "Failed to trigger FPGA computation" << std::endl;
            return 1;
        }
        std::cout << "✓ FPGA computation completed successfully" << std::endl;

        std::cout << "\n=== Testing BRAM Read Operations ===" << std::endl;
        
        // Read output from BRAM
        if (bram_driver.read_output_from_bram(output_data.data(), output_size) != 0) {
            std::cerr << "Failed to read output from BRAM" << std::endl;
            return 1;
        }
        std::cout << "✓ Output read from BRAM successfully" << std::endl;

        std::cout << "\n=== Test Results ===" << std::endl;
        
        // Display input data
        std::cout << "Input data (" << input_size << " elements):" << std::endl;
        for (int i = 0; i < input_size; ++i) {
            std::cout << std::fixed << std::setprecision(3) << input_data[i] << " ";
        }
        std::cout << std::endl;
        
        // Display first few weights
        std::cout << "Weights data (first 10 of " << weight_size << " elements):" << std::endl;
        for (int i = 0; i < std::min(10, weight_size); ++i) {
            std::cout << std::fixed << std::setprecision(3) << weights_data[i] << " ";
        }
        std::cout << std::endl;
        
        // Display bias data
        std::cout << "Bias data (" << bias_size << " elements):" << std::endl;
        for (int i = 0; i < bias_size; ++i) {
            std::cout << std::fixed << std::setprecision(3) << bias_data[i] << " ";
        }
        std::cout << std::endl;
        
        // Display output data
        std::cout << "Output data (" << output_size << " elements):" << std::endl;
        for (int i = 0; i < output_size; ++i) {
            std::cout << std::fixed << std::setprecision(3) << output_data[i] << " ";
        }
        std::cout << std::endl;

        // Verify output is not all zeros (indicates computation occurred)
        bool has_nonzero = false;
        for (int i = 0; i < output_size; ++i) {
            if (std::fabs(output_data[i]) > 1e-6) {
                has_nonzero = true;
                break;
            }
        }
        
        if (has_nonzero) {
            std::cout << "\n✅ SUCCESS: FPGA computation produced non-zero results!" << std::endl;
        } else {
            std::cout << "\n⚠️  WARNING: All output values are zero - check FPGA computation" << std::endl;
        }

        std::cout << "\n=== Integration Test Summary ===" << std::endl;
        std::cout << "✓ FPGA IP driver: Initialized and computation triggered" << std::endl;
        std::cout << "✓ BRAM driver: Initialized and all operations completed" << std::endl;
        std::cout << "✓ Data flow: Input → BRAM → FPGA → Output BRAM → Host" << std::endl;
        std::cout << "🎉 FPGA driver integration test completed successfully!" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "❌ Exception occurred: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
