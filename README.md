# GPU-Accelerated Monte Carlo Options Pricer

This project is a high-performance, GPU-accelerated Monte Carlo options pricer built in C++ and CUDA. It fetches real-time options data using Python, and uses both CPU (OpenMP) and GPU (CUDA) to price European and American options, as well as calculate Greeks (Delta and Gamma) using Common Random Numbers (CRN) and Finite Differences.

## Architecture & Data Organization

The project follows a clean, lightweight Object-Oriented design:
- **`fetch_data.py`**: A Python script to download the latest options chain for multiple highly liquid tickers (SPY, AAPL, QQQ, MSFT) using `yfinance`. It extracts metrics like Bid/Ask, Implied Volatility, and Underlying Price, saving them to `data/market_data.csv`.
- **`include/`**: Contains the core headers:
  - **`CSVReader.h`**: A C++ CSV reader leveraging **`rapidcsv`** (a header-only CSV library providing column-based lookup similar to Python's `pandas`) to parse `data/{TICKER}.csv` into typed `OptionData` structures.
  - **`rapidcsv.h`**: Industry-standard header-only C++ CSV library.
  - **`common.h`**: Common types and precision toggles (`typedef float real_t;`).
  - **`CPUOptionEngine.h`** & **`GPUOptionEngine.h`**: Headers for the CPU and GPU engines.
- **`CPUOptionEngine.cpp`**: Uses standard `<random>` and OpenMP for parallel path generation on the CPU.
- **`GPUOptionEngine.cu`**: Uses `cuRAND` for on-device random number generation and highly optimized CUDA kernels to accelerate the Monte Carlo simulations.

## Usage of Real Market Data
Real market data is downloaded via Yahoo Finance to provide realistic parameters for our Monte Carlo simulations. Specifically:
- **Underlying Price ($S_0$)**: The last closing price of the underlying asset.
- **Strike Price ($K$)** & **Time to Expiry ($T$)**: The actual quoted option characteristics.
- **Implied Volatility ($\sigma$)**: We use the market's implied volatility to simulate future price paths realistically.
- **Risk-Free Rate ($r$)**: Approximated using the 10-year treasury yield (`^TNX`).

The application simulates theoretical prices for these options using geometric Brownian motion and compares our model's output to the real **Market Mid-Price** (average of Bid and Ask) dynamically extracted from the live order book data.

## Mathematical Concepts

### 1. Geometric Brownian Motion (GBM)
We simulate the future paths of the underlying asset $S_t$ using the risk-neutral GBM process:

$$
S_t = S_0 \exp\left(\left(r - \frac{\sigma^2}{2}\right)t + \sigma \sqrt{t} Z\right)
$$

Where $Z \sim \mathcal{N}(0,1)$ is drawn from a standard normal distribution via OpenMP local RNGs or `cuRAND`.

### 2. Finite Difference for Greeks (CRN)
We compute the Greeks (Delta $\Delta$ and Gamma $\Gamma$) using Central Finite Differences paired with Common Random Numbers (CRN). CRN reduces variance by using the same standard normal $Z$ paths for the bumped prices:

$$
\Delta = \frac{V(S_0 + dS) - V(S_0 - dS)}{2\,dS}
$$

$$
\Gamma = \frac{V(S_0 + dS) - 2V(S_0) + V(S_0 - dS)}{(dS)^2}
$$

### 3. Least Squares Monte Carlo (LSM)
For **American Options**, early exercise is permitted at any time. We determine the optimal exercise strategy using Longstaff-Schwartz LSM:
- Simulate all price paths up to expiry $T$.
- Moving backward in time (for each step $t$), we isolate paths that are "In-The-Money" (ITM).
- We regress the discounted future payoffs $Y$ against the current stock price using quadratic basis functions $\{1, S, S^2\}$:

 ($1, S, S^2$)

- If the immediate exercise payoff exceeds the estimated continuation value from the regression, we update the path's payoff to the exercise value.

## GPU Acceleration Mechanics
For **European Options**, the pricing uses an optimized single-step kernel leveraging:
- **`Philox4_32_10` Vector RNG**: Generates 4 normal random numbers per thread in a single vector instruction with $O(1)$ near-zero initialization overhead.
- **Hardware Warp Shuffle Reductions (`__shfl_down_sync`)**: Eliminates shared memory bank conflicts by reducing payoffs at the register level within 32-thread warps.
- **Grid-Stride Loop**: Distributes paths across thread blocks efficiently to keep all Streaming Multiprocessors (SMs) fully saturated.
- **Single Consolidated Device Buffer**: Replaces multiple individual allocations with a single 3-element buffer for the reduction sums, minimizing driver API overhead.

For **American Options**, we implement a **100% On-Device Least Squares Monte Carlo (LSM)** pipeline:
- **Parallel Path Generation**: Simulates 400,000 multi-step paths directly in GPU VRAM using `Philox4`.
- **Parallel Regression Sums**: Computes the 8 polynomial regression normal equations ($\sum 1, \sum X, \sum X^2, \dots$) across in-the-money paths via warp-shuffle reductions.
- **On-Device Matrix Inversion**: Solves the $3 \times 3$ polynomial system in parallel and updates continuation payoffs without copying paths back to the CPU.
- **Zero PCIe Round-trips**: All 252 backward induction steps execute asynchronously on the GPU.

## Performance Benchmark

We simulate $N = 400{,}000$ paths (4x scale) for an American option with 252 steps, and for a single-step European option across four major tickers. We benchmark **At-The-Money (ATM)** options to verify high precision and statistical convergence.

### 1. SPY (Call Option, Strike: 770, Underlying: 769.60)
**Market Mid-Price**: 13.015  |  **Implied Vol**: 13.37%

| Engine | European Price | Euro Time | American (LSM) Price | LSM Time |
|---|---|---|---|---|
| **CPU (OpenMP)** | 14.4847 | ~2.21 ms | 14.2352 | ~2.29 s |
| **GPU (CUDA)** | 14.4885 | **~0.49 ms (4.5x faster)** | 14.1611 | **~0.34 s (6.7x faster, Sub-Second!)** |

### 2. AAPL (Call Option, Strike: 315, Underlying: 315.15)
**Market Mid-Price**: 10.850  |  **Implied Vol**: 26.91%

| Engine | European Price | Euro Time | American (LSM) Price | LSM Time |
|---|---|---|---|---|
| **CPU (OpenMP)** | 11.4037 | ~2.78 ms | 11.2422 | ~3.32 s |
| **GPU (CUDA)** | 11.4054 | **~0.52 ms (5.3x faster)** | 11.1883 | **~0.33 s (10.0x faster, Sub-Second!)** |

### 3. QQQ (Call Option, Strike: 717, Underlying: 716.68)
**Market Mid-Price**: 18.555  |  **Implied Vol**: 20.35%

| Engine | European Price | Euro Time | American (LSM) Price | LSM Time |
|---|---|---|---|---|
| **CPU (OpenMP)** | 19.7442 | ~1.41 ms | 19.4403 | ~3.19 s |
| **GPU (CUDA)** | 19.7476 | **~0.77 ms (1.8x faster)** | 19.3365 | **~0.34 s (9.4x faster, Sub-Second!)** |

### 4. MSFT (Call Option, Strike: 485, Underlying: 483.80)
**Market Mid-Price**: 15.725  |  **Implied Vol**: 27.33%

| Engine | European Price | Euro Time | American (LSM) Price | LSM Time |
|---|---|---|---|---|
| **CPU (OpenMP)** | 17.0531 | ~2.71 ms | 16.8078 | ~3.29 s |
| **GPU (CUDA)** | 17.0548 | **~0.62 ms (4.4x faster)** | 16.7248 | **~0.49 s (6.7x faster, Sub-Second!)** |

*Note: With 100% on-device LSM backward induction, American option pricing (with 400,000 paths x 252 steps x 3 Greeks bumps) dropped from ~15 seconds down to **~0.33–0.49 seconds**, achieving true sub-second performance!*

## How to Run

### 1. Fetch Market Data
First, install the required Python libraries and fetch real-time market data.
```bash
pip install yfinance pandas numpy
python fetch_data.py
```
This will produce a `data/market_data.csv` file.

### 2. Build the C++ Project
Ensure you have CMake, Visual Studio (MSVC), and the CUDA Toolkit installed. 

```bash
mkdir build
cd build
cmake ..
cmake --build . --config Release
```

### 3. Run the Pricer
Once built, execute the compiled binary to see the benchmarks and pricing results:
```bash
.\Release\OptionPricer.exe
```
