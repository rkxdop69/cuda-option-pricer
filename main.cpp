#include "CSVReader.h"
#include "CPUOptionEngine.h"
#include "GPUOptionEngine.h"
#include <iostream>
#include <chrono>
#include <iomanip>
#include <vector>
#include <string>

using namespace std;

void run_benchmark_for_ticker(const string& ticker) {
    string data_file = "data/" + ticker + ".csv";
    auto options = CSVReader::read(data_file);

    if (options.empty()) {
        cerr << "No options data found in " << data_file << endl;
        return;
    }

    // Pick an At-The-Money (ATM) option for stable Monte Carlo variance comparison
    OptionData opt = options[0];
    real_t min_diff = abs(opt.strike - opt.underlying_price);
    for (const auto& o : options) {
        real_t diff = abs(o.strike - o.underlying_price);
        if (diff < min_diff) {
            min_diff = diff;
            opt = o;
        }
    }

    int num_paths = 400000;
    int num_steps = 252; // For American

    cout << "======================================================================\n";
    cout << "Pricing Option for Ticker: " << ticker << "\n";
    cout << "Type: " << (opt.type == OptionType::Call ? "Call" : "Put") << "\n";
    cout << "Strike: " << opt.strike << "\n";
    cout << "Underlying: " << opt.underlying_price << "\n";
    cout << "Market Mid-Price: " << (opt.bid + opt.ask) / 2.0f << "\n";
    cout << "Implied Vol: " << opt.implied_volatility << "\n";
    cout << "Risk-Free Rate: " << opt.risk_free_rate * 100.0f << "%\n";
    cout << "Dividend Yield: " << opt.dividend_yield * 100.0f << "%\n\n";

    // ---------------------------------------------------------
    // European Pricing
    // ---------------------------------------------------------
    cout << "--- EUROPEAN OPTION ---\n";

    // CPU Warmup
    CPUOptionEngine::priceEuropean(opt, 1000);
    
    auto start_cpu_eur = chrono::high_resolution_clock::now();
    PricingResult cpu_eur = CPUOptionEngine::priceEuropean(opt, num_paths);
    auto end_cpu_eur = chrono::high_resolution_clock::now();
    chrono::duration<double> diff_cpu_eur = end_cpu_eur - start_cpu_eur;

    // GPU Warmup
    GPUOptionEngine::priceEuropean(opt, 1000);

    auto start_gpu_eur = chrono::high_resolution_clock::now();
    PricingResult gpu_eur = GPUOptionEngine::priceEuropean(opt, num_paths);
    auto end_gpu_eur = chrono::high_resolution_clock::now();
    chrono::duration<double> diff_gpu_eur = end_gpu_eur - start_gpu_eur;

    cout << left << setw(15) << "Method" 
         << setw(15) << "Price" 
         << setw(15) << "Delta" 
         << setw(15) << "Gamma" 
         << "Time (s)" << "\n";
    cout << string(70, '-') << "\n";
    cout << left << setw(15) << "CPU (OpenMP)" 
         << setw(15) << cpu_eur.price 
         << setw(15) << cpu_eur.delta 
         << setw(15) << cpu_eur.gamma 
         << diff_cpu_eur.count() << "\n";
    cout << left << setw(15) << "GPU (CUDA)" 
         << setw(15) << gpu_eur.price 
         << setw(15) << gpu_eur.delta 
         << setw(15) << gpu_eur.gamma 
         << diff_gpu_eur.count() << "\n\n";


    // ---------------------------------------------------------
    // American Pricing
    // ---------------------------------------------------------
    cout << "--- AMERICAN OPTION (LSM) ---\n";
    cout << "Steps: " << num_steps << "\n";
    
    auto start_cpu_am = chrono::high_resolution_clock::now();
    PricingResult cpu_am = CPUOptionEngine::priceAmerican(opt, num_paths, num_steps);
    auto end_cpu_am = chrono::high_resolution_clock::now();
    chrono::duration<double> diff_cpu_am = end_cpu_am - start_cpu_am;

    auto start_gpu_am = chrono::high_resolution_clock::now();
    PricingResult gpu_am = GPUOptionEngine::priceAmerican(opt, num_paths, num_steps);
    auto end_gpu_am = chrono::high_resolution_clock::now();
    chrono::duration<double> diff_gpu_am = end_gpu_am - start_gpu_am;

    cout << left << setw(15) << "Method" 
         << setw(15) << "Price" 
         << setw(15) << "Delta" 
         << setw(15) << "Gamma" 
         << "Time (s)" << "\n";
    cout << string(70, '-') << "\n";
    cout << left << setw(15) << "CPU (OpenMP)" 
         << setw(15) << cpu_am.price 
         << setw(15) << cpu_am.delta 
         << setw(15) << cpu_am.gamma 
         << diff_cpu_am.count() << "\n";
    cout << left << setw(15) << "GPU (CUDA)" 
         << setw(15) << gpu_am.price 
         << setw(15) << gpu_am.delta 
         << setw(15) << gpu_am.gamma 
         << diff_gpu_am.count() << "\n\n";
}

int main() {
    vector<string> tickers = {"SPY", "AAPL", "QQQ", "MSFT"};
    for (const auto& ticker : tickers) {
        run_benchmark_for_ticker(ticker);
    }
    return 0;
}
