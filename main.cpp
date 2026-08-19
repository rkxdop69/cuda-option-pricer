#include "CSVReader.h"
#include "CPUOptionEngine.h"
#include "GPUOptionEngine.h"
#include <iostream>
#include <chrono>
#include <iomanip>
#include <vector>

void run_benchmark_for_ticker(const std::string& ticker) {
    std::string data_file = "data/" + ticker + ".csv";
    auto options = CSVReader::read(data_file);

    if (options.empty()) {
        std::cerr << "No options data found in " << data_file << std::endl;
        return;
    }

    // Pick an At-The-Money (ATM) option for stable Monte Carlo variance comparison
    OptionData opt = options[0];
    real_t min_diff = std::abs(opt.strike - opt.underlying_price);
    for (const auto& o : options) {
        real_t diff = std::abs(o.strike - o.underlying_price);
        if (diff < min_diff) {
            min_diff = diff;
            opt = o;
        }
    }

    int num_paths = 400000;
    int num_steps = 252; // For American

    std::cout << "======================================================================\n";
    std::cout << "Pricing Option for Ticker: " << ticker << "\n";
    std::cout << "Type: " << (opt.type == OptionType::Call ? "Call" : "Put") << "\n";
    std::cout << "Strike: " << opt.strike << "\n";
    std::cout << "Underlying: " << opt.underlying_price << "\n";
    std::cout << "Expiry: " << opt.expiry_years << " years\n";
    std::cout << "Market Mid-Price: " << (opt.bid + opt.ask) / 2.0f << "\n";
    std::cout << "Implied Vol: " << opt.implied_volatility << "\n\n";

    // ---------------------------------------------------------
    // European Pricing
    // ---------------------------------------------------------
    std::cout << "--- EUROPEAN OPTION ---\n";

    // CPU Warmup
    CPUOptionEngine::priceEuropean(opt, 1000);
    
    auto start_cpu_eur = std::chrono::high_resolution_clock::now();
    PricingResult cpu_eur = CPUOptionEngine::priceEuropean(opt, num_paths);
    auto end_cpu_eur = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff_cpu_eur = end_cpu_eur - start_cpu_eur;

    // GPU Warmup
    GPUOptionEngine::priceEuropean(opt, 1000);

    auto start_gpu_eur = std::chrono::high_resolution_clock::now();
    PricingResult gpu_eur = GPUOptionEngine::priceEuropean(opt, num_paths);
    auto end_gpu_eur = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff_gpu_eur = end_gpu_eur - start_gpu_eur;

    std::cout << std::left << std::setw(15) << "Method" 
              << std::setw(15) << "Price" 
              << std::setw(15) << "Delta" 
              << std::setw(15) << "Gamma" 
              << "Time (s)" << "\n";
    std::cout << std::string(70, '-') << "\n";
    std::cout << std::left << std::setw(15) << "CPU (OpenMP)" 
              << std::setw(15) << cpu_eur.price 
              << std::setw(15) << cpu_eur.delta 
              << std::setw(15) << cpu_eur.gamma 
              << diff_cpu_eur.count() << "\n";
    std::cout << std::left << std::setw(15) << "GPU (CUDA)" 
              << std::setw(15) << gpu_eur.price 
              << std::setw(15) << gpu_eur.delta 
              << std::setw(15) << gpu_eur.gamma 
              << diff_gpu_eur.count() << "\n\n";


    // ---------------------------------------------------------
    // American Pricing
    // ---------------------------------------------------------
    std::cout << "--- AMERICAN OPTION (LSM) ---\n";
    std::cout << "Steps: " << num_steps << "\n";
    
    auto start_cpu_am = std::chrono::high_resolution_clock::now();
    PricingResult cpu_am = CPUOptionEngine::priceAmerican(opt, num_paths, num_steps);
    auto end_cpu_am = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff_cpu_am = end_cpu_am - start_cpu_am;

    auto start_gpu_am = std::chrono::high_resolution_clock::now();
    PricingResult gpu_am = GPUOptionEngine::priceAmerican(opt, num_paths, num_steps);
    auto end_gpu_am = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff_gpu_am = end_gpu_am - start_gpu_am;

    std::cout << std::left << std::setw(15) << "Method" 
              << std::setw(15) << "Price" 
              << std::setw(15) << "Delta" 
              << std::setw(15) << "Gamma" 
              << "Time (s)" << "\n";
    std::cout << std::string(70, '-') << "\n";
    std::cout << std::left << std::setw(15) << "CPU (OpenMP)" 
              << std::setw(15) << cpu_am.price 
              << std::setw(15) << cpu_am.delta 
              << std::setw(15) << cpu_am.gamma 
              << diff_cpu_am.count() << "\n";
    std::cout << std::left << std::setw(15) << "GPU (CUDA)" 
              << std::setw(15) << gpu_am.price 
              << std::setw(15) << gpu_am.delta 
              << std::setw(15) << gpu_am.gamma 
              << diff_gpu_am.count() << "\n\n";
}

int main() {
    std::vector<std::string> tickers = {"SPY", "AAPL", "QQQ", "MSFT"};
    for (const auto& ticker : tickers) {
        run_benchmark_for_ticker(ticker);
    }
    return 0;
}
