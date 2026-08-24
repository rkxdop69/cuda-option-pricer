#include "GPUOptionEngine.h"
#include <curand_kernel.h>
#include <iostream>
#include <cmath>
#include <vector>

using namespace std;

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " code=" << err << " \"" << cudaGetErrorString(err) << "\"" << endl; \
            return PricingResult{0, 0, 0}; \
        } \
    } while(0)

// Helper for European payoff
__device__ inline real_t d_payoff(real_t S, real_t K, OptionType type) {
    if (type == OptionType::Call) {
        return fmaxf(S - K, 0.0f);
    } else {
        return fmaxf(K - S, 0.0f);
    }
}

// Single-step European kernel with Philox4 RNG, vector loads, and warp shuffle reduction
__global__ void europeanKernel(
    int num_paths,
    OptionType type,
    real_t S, real_t K, real_t T, real_t r, real_t q, real_t v,
    real_t* d_results)
{
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = gridDim.x * blockDim.x;

    curandStatePhilox4_32_10_t state;
    curand_init(1234ULL, gid, 0, &state);

    real_t dS = S * 0.01f;
    real_t S_up = S + dS;
    real_t S_dn = S - dS;

    real_t drift = (r - q - 0.5f * v * v) * T;
    real_t vol_sqrt_T = v * sqrtf(T);

    real_t my_V = 0.0f, my_V_up = 0.0f, my_V_dn = 0.0f;

    // Grid-stride loop: each iteration processes 4 paths per thread using vector normal RNG
    for (int p = gid * 4; p < num_paths; p += total_threads * 4) {
        float4 Z = curand_normal4(&state);

        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            if (p + k < num_paths) {
                float z_val = (k == 0) ? Z.x : ((k == 1) ? Z.y : ((k == 2) ? Z.z : Z.w));
                real_t exp_term = expf(drift + vol_sqrt_T * z_val);
                my_V += d_payoff(S * exp_term, K, type);
                my_V_up += d_payoff(S_up * exp_term, K, type);
                my_V_dn += d_payoff(S_dn * exp_term, K, type);
            }
        }
    }

    // Warp-level reduction using hardware warp shuffle (__shfl_down_sync)
    #pragma unroll
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        my_V += __shfl_down_sync(0xffffffff, my_V, offset);
        my_V_up += __shfl_down_sync(0xffffffff, my_V_up, offset);
        my_V_dn += __shfl_down_sync(0xffffffff, my_V_dn, offset);
    }

    // Block-level reduction using shared memory for warp leaders
    __shared__ real_t s_V[32];
    __shared__ real_t s_V_up[32];
    __shared__ real_t s_V_dn[32];

    int lane = tid % warpSize;
    int wid = tid / warpSize;

    if (lane == 0) {
        s_V[wid] = my_V;
        s_V_up[wid] = my_V_up;
        s_V_dn[wid] = my_V_dn;
    }
    __syncthreads();

    if (wid == 0) {
        int num_warps = blockDim.x / warpSize;
        real_t b_V = (lane < num_warps) ? s_V[lane] : 0.0f;
        real_t b_V_up = (lane < num_warps) ? s_V_up[lane] : 0.0f;
        real_t b_V_dn = (lane < num_warps) ? s_V_dn[lane] : 0.0f;

        #pragma unroll
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            b_V += __shfl_down_sync(0xffffffff, b_V, offset);
            b_V_up += __shfl_down_sync(0xffffffff, b_V_up, offset);
            b_V_dn += __shfl_down_sync(0xffffffff, b_V_dn, offset);
        }

        if (lane == 0) {
            atomicAdd(&d_results[0], b_V);
            atomicAdd(&d_results[1], b_V_up);
            atomicAdd(&d_results[2], b_V_dn);
        }
    }
}

PricingResult GPUOptionEngine::priceEuropean(const OptionData& opt, int num_paths) {
    real_t* d_results;
    CUDA_CHECK(cudaMalloc(&d_results, 3 * sizeof(real_t)));
    CUDA_CHECK(cudaMemset(d_results, 0, 3 * sizeof(real_t)));

    int threads = 256;
    int blocks = (num_paths / 4 + threads - 1) / threads;
    if (blocks < 1) blocks = 1;
    if (blocks > 1024) blocks = 1024;

    europeanKernel<<<blocks, threads>>>(
        num_paths, opt.type, opt.underlying_price, opt.strike,
        opt.expiry_years, opt.risk_free_rate, opt.dividend_yield, opt.implied_volatility,
        d_results
    );
    CUDA_CHECK(cudaDeviceSynchronize());

    real_t h_results[3] = {0, 0, 0};
    CUDA_CHECK(cudaMemcpy(h_results, d_results, 3 * sizeof(real_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_results));

    real_t discount = exp(-opt.risk_free_rate * opt.expiry_years);
    real_t dS = opt.underlying_price * 0.01f;

    PricingResult res;
    res.price = (h_results[0] / num_paths) * discount;
    real_t V_up = (h_results[1] / num_paths) * discount;
    real_t V_dn = (h_results[2] / num_paths) * discount;
    
    res.delta = (V_up - V_dn) / (2.0f * dS);
    res.gamma = (V_up - 2.0f * res.price + V_dn) / (dS * dS);

    return res;
}

// Multi-step Path Generation with Philox4
__global__ void americanGeneratePathsKernel(
    int num_paths, int num_steps,
    real_t S, real_t r, real_t q, real_t v, real_t dt,
    unsigned long long seed,
    real_t* d_paths)
{
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p < num_paths) {
        curandStatePhilox4_32_10_t state;
        curand_init(seed, p, 0, &state);

        real_t drift = (r - q - 0.5f * v * v) * dt;
        real_t vol_sqrt_dt = v * sqrtf(dt);

        real_t curr_S = S;
        d_paths[p * (num_steps + 1)] = curr_S;

        for (int step = 1; step <= num_steps; ++step) {
            real_t Z = curand_normal(&state);
            curr_S *= expf(drift + vol_sqrt_dt * Z);
            d_paths[p * (num_steps + 1) + step] = curr_S;
        }
    }
}

// Initialize Payoff at Maturity
__global__ void americanInitPayoffKernel(
    int num_paths, int num_steps,
    real_t K, OptionType type,
    const real_t* d_paths,
    real_t* d_V)
{
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p < num_paths) {
        real_t ST = d_paths[p * (num_steps + 1) + num_steps];
        d_V[p] = d_payoff(ST, K, type);
    }
}

// Accumulate Regression Sums for ITM paths using warp shuffle & shared memory
__global__ void americanRegressionSumsKernel(
    int num_paths, int num_steps, int step,
    real_t K, OptionType type, real_t df,
    const real_t* d_paths,
    const real_t* d_V,
    double* d_sums)
{
    int tid = threadIdx.x;
    int p = blockIdx.x * blockDim.x + threadIdx.x;

    double count = 0.0;
    double sum_x = 0.0, sum_x2 = 0.0, sum_x3 = 0.0, sum_x4 = 0.0;
    double sum_y = 0.0, sum_yx = 0.0, sum_yx2 = 0.0;

    if (p < num_paths) {
        real_t S = d_paths[p * (num_steps + 1) + step];
        real_t imm_payoff = d_payoff(S, K, type);
        if (imm_payoff > 0.0f) {
            double x = (double)S;
            double y = (double)(d_V[p] * df);
            double x2 = x * x;

            count = 1.0;
            sum_x = x;
            sum_x2 = x2;
            sum_x3 = x2 * x;
            sum_x4 = x2 * x2;
            sum_y = y;
            sum_yx = y * x;
            sum_yx2 = y * x2;
        }
    }

    // Warp-level reduction
    #pragma unroll
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        count   += __shfl_down_sync(0xffffffff, count, offset);
        sum_x   += __shfl_down_sync(0xffffffff, sum_x, offset);
        sum_x2  += __shfl_down_sync(0xffffffff, sum_x2, offset);
        sum_x3  += __shfl_down_sync(0xffffffff, sum_x3, offset);
        sum_x4  += __shfl_down_sync(0xffffffff, sum_x4, offset);
        sum_y   += __shfl_down_sync(0xffffffff, sum_y, offset);
        sum_yx  += __shfl_down_sync(0xffffffff, sum_yx, offset);
        sum_yx2 += __shfl_down_sync(0xffffffff, sum_yx2, offset);
    }

    __shared__ double s_sums[8][32];
    int lane = tid % warpSize;
    int wid = tid / warpSize;

    if (lane == 0) {
        s_sums[0][wid] = count;
        s_sums[1][wid] = sum_x;
        s_sums[2][wid] = sum_x2;
        s_sums[3][wid] = sum_x3;
        s_sums[4][wid] = sum_x4;
        s_sums[5][wid] = sum_y;
        s_sums[6][wid] = sum_yx;
        s_sums[7][wid] = sum_yx2;
    }
    __syncthreads();

    if (wid == 0) {
        int num_warps = blockDim.x / warpSize;
        double b_count   = (lane < num_warps) ? s_sums[0][lane] : 0.0;
        double b_sum_x   = (lane < num_warps) ? s_sums[1][lane] : 0.0;
        double b_sum_x2  = (lane < num_warps) ? s_sums[2][lane] : 0.0;
        double b_sum_x3  = (lane < num_warps) ? s_sums[3][lane] : 0.0;
        double b_sum_x4  = (lane < num_warps) ? s_sums[4][lane] : 0.0;
        double b_sum_y   = (lane < num_warps) ? s_sums[5][lane] : 0.0;
        double b_sum_yx  = (lane < num_warps) ? s_sums[6][lane] : 0.0;
        double b_sum_yx2 = (lane < num_warps) ? s_sums[7][lane] : 0.0;

        #pragma unroll
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            b_count   += __shfl_down_sync(0xffffffff, b_count, offset);
            b_sum_x   += __shfl_down_sync(0xffffffff, b_sum_x, offset);
            b_sum_x2  += __shfl_down_sync(0xffffffff, b_sum_x2, offset);
            b_sum_x3  += __shfl_down_sync(0xffffffff, b_sum_x3, offset);
            b_sum_x4  += __shfl_down_sync(0xffffffff, b_sum_x4, offset);
            b_sum_y   += __shfl_down_sync(0xffffffff, b_sum_y, offset);
            b_sum_yx  += __shfl_down_sync(0xffffffff, b_sum_yx, offset);
            b_sum_yx2 += __shfl_down_sync(0xffffffff, b_sum_yx2, offset);
        }

        if (lane == 0) {
            atomicAdd(&d_sums[0], b_count);
            atomicAdd(&d_sums[1], b_sum_x);
            atomicAdd(&d_sums[2], b_sum_x2);
            atomicAdd(&d_sums[3], b_sum_x3);
            atomicAdd(&d_sums[4], b_sum_x4);
            atomicAdd(&d_sums[5], b_sum_y);
            atomicAdd(&d_sums[6], b_sum_yx);
            atomicAdd(&d_sums[7], b_sum_yx2);
        }
    }
}

// Solve 3x3 normal equations on GPU in 1 thread
__global__ void americanSolve3x3Kernel(
    const double* d_sums,
    double* d_coeffs)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        double n = d_sums[0];
        double sum_x = d_sums[1];
        double sum_x2 = d_sums[2];
        double sum_x3 = d_sums[3];
        double sum_x4 = d_sums[4];
        double sum_y = d_sums[5];
        double sum_yx = d_sums[6];
        double sum_yx2 = d_sums[7];

        if (n > 3.0) {
            double m11 = n,      m12 = sum_x,  m13 = sum_x2;
            double m21 = sum_x,  m22 = sum_x2, m23 = sum_x3;
            double m31 = sum_x2, m32 = sum_x3, m33 = sum_x4;

            double det = m11*(m22*m33 - m23*m32) - m12*(m21*m33 - m23*m31) + m13*(m21*m32 - m22*m31);
            if (fabs(det) > 1e-10) {
                double b1 = sum_y, b2 = sum_yx, b3 = sum_yx2;
                d_coeffs[0] = (b1*(m22*m33 - m23*m32) - m12*(b2*m33 - m23*b3) + m13*(b2*m32 - m22*b3)) / det;
                d_coeffs[1] = (m11*(b2*m33 - m23*b3) - b1*(m21*m33 - m23*m31) + m13*(m21*b3 - b2*m31)) / det;
                d_coeffs[2] = (m11*(m22*b3 - b2*m32) - m12*(m21*b3 - b2*m31) + b1*(m21*m32 - m22*m31)) / det;
                d_coeffs[3] = 1.0; // Valid regression
                return;
            }
        }
        d_coeffs[3] = 0.0; // Invalid regression
    }
}

// Update Option Values V[p] for current step
__global__ void americanUpdatePayoffKernel(
    int num_paths, int num_steps, int step,
    real_t K, OptionType type, real_t df,
    const real_t* d_paths,
    const double* d_coeffs,
    real_t* d_V)
{
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p < num_paths) {
        real_t S = d_paths[p * (num_steps + 1) + step];
        real_t imm_payoff = d_payoff(S, K, type);
        bool valid = (d_coeffs[3] > 0.5);

        if (valid && imm_payoff > 0.0f) {
            double cv = d_coeffs[0] + d_coeffs[1] * (double)S + d_coeffs[2] * (double)S * (double)S;
            if ((double)imm_payoff > cv) {
                d_V[p] = imm_payoff;
            } else {
                d_V[p] *= df;
            }
        } else {
            d_V[p] *= df;
        }
    }
}

// Final Reduction of V[p] to get option price
__global__ void americanReducePriceKernel(
    int num_paths,
    const real_t* d_V,
    double* d_final_sum)
{
    int tid = threadIdx.x;
    int p = blockIdx.x * blockDim.x + threadIdx.x;

    double my_sum = 0.0;
    if (p < num_paths) {
        my_sum = (double)d_V[p];
    }

    // Warp-level reduction
    #pragma unroll
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        my_sum += __shfl_down_sync(0xffffffff, my_sum, offset);
    }

    __shared__ double s_sum[32];
    int lane = tid % warpSize;
    int wid = tid / warpSize;

    if (lane == 0) s_sum[wid] = my_sum;
    __syncthreads();

    if (wid == 0) {
        int num_warps = blockDim.x / warpSize;
        double b_sum = (lane < num_warps) ? s_sum[lane] : 0.0;
        #pragma unroll
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            b_sum += __shfl_down_sync(0xffffffff, b_sum, offset);
        }
        if (lane == 0) {
            atomicAdd(d_final_sum, b_sum);
        }
    }
}

PricingResult GPUOptionEngine::priceAmerican(const OptionData& opt, int num_paths, int num_steps) {
    real_t S0 = opt.underlying_price;
    real_t K = opt.strike;
    real_t T = opt.expiry_years;
    real_t r = opt.risk_free_rate;
    real_t q = opt.dividend_yield;
    real_t v = opt.implied_volatility;

    real_t dt = T / num_steps;
    real_t df = exp(-r * dt);
    real_t dS = S0 * 0.01f;

    real_t* d_paths;
    real_t* d_V;
    double* d_sums;
    double* d_coeffs;
    double* d_final_sum;

    size_t paths_size = (size_t)num_paths * (num_steps + 1) * sizeof(real_t);
    CUDA_CHECK(cudaMalloc(&d_paths, paths_size));
    CUDA_CHECK(cudaMalloc(&d_V, num_paths * sizeof(real_t)));
    CUDA_CHECK(cudaMalloc(&d_sums, 8 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_coeffs, 4 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_final_sum, sizeof(double)));

    int threads = 256;
    int blocks = (num_paths + threads - 1) / threads;

    auto price_single_spot = [&](real_t spot, unsigned long long seed) -> real_t {
        // Step 1: Generate Paths on GPU
        americanGeneratePathsKernel<<<blocks, threads>>>(
            num_paths, num_steps, spot, r, q, v, dt, seed, d_paths
        );

        // Step 2: Initialize Payoffs at Maturity
        americanInitPayoffKernel<<<blocks, threads>>>(
            num_paths, num_steps, K, opt.type, d_paths, d_V
        );

        // Step 3: Backward Induction Loop on GPU
        for (int s = num_steps - 1; s > 0; --s) {
            cudaMemset(d_sums, 0, 8 * sizeof(double));
            americanRegressionSumsKernel<<<blocks, threads>>>(
                num_paths, num_steps, s, K, opt.type, df, d_paths, d_V, d_sums
            );
            americanSolve3x3Kernel<<<1, 1>>>(d_sums, d_coeffs);
            americanUpdatePayoffKernel<<<blocks, threads>>>(
                num_paths, num_steps, s, K, opt.type, df, d_paths, d_coeffs, d_V
            );
        }

        // Step 4: Final Price Reduction on GPU
        cudaMemset(d_final_sum, 0, sizeof(double));
        americanReducePriceKernel<<<blocks, threads>>>(num_paths, d_V, d_final_sum);

        double h_final_sum = 0.0;
        cudaMemcpy(&h_final_sum, d_final_sum, sizeof(double), cudaMemcpyDeviceToHost);
        return (real_t)((h_final_sum / num_paths) * df);
    };

    real_t price    = price_single_spot(S0, 1234ULL);
    real_t price_up = price_single_spot(S0 + dS, 1234ULL);
    real_t price_dn = price_single_spot(S0 - dS, 1234ULL);

    CUDA_CHECK(cudaFree(d_paths));
    CUDA_CHECK(cudaFree(d_V));
    CUDA_CHECK(cudaFree(d_sums));
    CUDA_CHECK(cudaFree(d_coeffs));
    CUDA_CHECK(cudaFree(d_final_sum));

    PricingResult res;
    res.price = price;
    res.delta = (price_up - price_dn) / (2.0f * dS);
    res.gamma = (price_up - 2.0f * price + price_dn) / (dS * dS);

    return res;
}
