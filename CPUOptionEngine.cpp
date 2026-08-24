#include "CPUOptionEngine.h"
#include <cmath>
#include <random>
#include <omp.h>
#include <vector>
#include <algorithm>

using namespace std;

namespace {
    // Generate standard normal
    inline real_t generate_normal(mt19937& gen) {
        normal_distribution<real_t> dist(0.0, 1.0);
        return dist(gen);
    }

    // Payoff function
    inline real_t payoff(real_t S, real_t K, OptionType type) {
        if (type == OptionType::Call) {
            return max<real_t>(S - K, 0.0);
        } else {
            return max<real_t>(K - S, 0.0);
        }
    }
}

PricingResult CPUOptionEngine::priceEuropean(const OptionData& opt, int num_paths) {
    real_t S = opt.underlying_price;
    real_t K = opt.strike;
    real_t T = opt.expiry_years;
    real_t r = opt.risk_free_rate;
    real_t q = opt.dividend_yield;
    real_t v = opt.implied_volatility;

    real_t dS = S * 0.01f; // 1% bump for Greeks
    real_t S_up = S + dS;
    real_t S_dn = S - dS;

    real_t drift = (r - q - 0.5f * v * v) * T;
    real_t vol_sqrt_T = v * sqrt(T);
    real_t discount = exp(-r * T);

    real_t sum_V = 0.0, sum_V_up = 0.0, sum_V_dn = 0.0;

    #pragma omp parallel
    {
        // Thread-local random generator
        mt19937 gen(42 + omp_get_thread_num());
        real_t local_V = 0, local_V_up = 0, local_V_dn = 0;

        #pragma omp for
        for (int i = 0; i < num_paths; ++i) {
            real_t Z = generate_normal(gen);
            real_t exp_term = exp(drift + vol_sqrt_T * Z);
            
            real_t ST = S * exp_term;
            real_t ST_up = S_up * exp_term;
            real_t ST_dn = S_dn * exp_term;

            local_V += payoff(ST, K, opt.type);
            local_V_up += payoff(ST_up, K, opt.type);
            local_V_dn += payoff(ST_dn, K, opt.type);
        }

        #pragma omp atomic
        sum_V += local_V;
        #pragma omp atomic
        sum_V_up += local_V_up;
        #pragma omp atomic
        sum_V_dn += local_V_dn;
    }

    PricingResult res;
    res.price = (sum_V / num_paths) * discount;
    real_t V_up = (sum_V_up / num_paths) * discount;
    real_t V_dn = (sum_V_dn / num_paths) * discount;

    res.delta = (V_up - V_dn) / (2.0f * dS);
    res.gamma = (V_up - 2.0f * res.price + V_dn) / (dS * dS);

    return res;
}

PricingResult CPUOptionEngine::priceAmerican(const OptionData& opt, int num_paths, int num_steps) {
    real_t S0 = opt.underlying_price;
    real_t K = opt.strike;
    real_t T = opt.expiry_years;
    real_t r = opt.risk_free_rate;
    real_t q = opt.dividend_yield;
    real_t v = opt.implied_volatility;

    real_t dt = T / num_steps;
    real_t df = exp(-r * dt);
    real_t drift = (r - q - 0.5f * v * v) * dt;
    real_t vol_sqrt_dt = v * sqrt(dt);

    real_t dS = S0 * 0.01f;

    auto simulate_paths = [&](real_t init_S, vector<real_t>& V) -> real_t {
        vector<real_t> paths((size_t)num_paths * (num_steps + 1));
        
        #pragma omp parallel
        {
            mt19937 gen(1234 + omp_get_thread_num());
            #pragma omp for
            for (int p = 0; p < num_paths; ++p) {
                real_t curr_S = init_S;
                paths[p * (num_steps + 1)] = curr_S;
                for (int s = 1; s <= num_steps; ++s) {
                    real_t Z = generate_normal(gen);
                    curr_S *= exp(drift + vol_sqrt_dt * Z);
                    paths[p * (num_steps + 1) + s] = curr_S;
                }
            }
        }

        V.resize(num_paths);
        for (int p = 0; p < num_paths; ++p) {
            V[p] = payoff(paths[p * (num_steps + 1) + num_steps], K, opt.type);
        }

        for (int s = num_steps - 1; s > 0; --s) {
            double sum_x = 0, sum_x2 = 0, sum_x3 = 0, sum_x4 = 0;
            double sum_y = 0, sum_yx = 0, sum_yx2 = 0;
            int n = 0;

            #pragma omp parallel for reduction(+:n, sum_x, sum_x2, sum_x3, sum_x4, sum_y, sum_yx, sum_yx2)
            for (int p = 0; p < num_paths; ++p) {
                real_t current_S = paths[p * (num_steps + 1) + s];
                real_t current_payoff = payoff(current_S, K, opt.type);
                if (current_payoff > 0) {
                    double x = (double)current_S;
                    double y = (double)(V[p] * df);
                    double x2 = x * x;
                    n += 1;
                    sum_x += x;
                    sum_x2 += x2;
                    sum_x3 += x2 * x;
                    sum_x4 += x2 * x2;
                    sum_y += y;
                    sum_yx += y * x;
                    sum_yx2 += y * x2;
                }
            }

            double a = 0, b = 0, c = 0;
            bool valid = false;
            if (n > 3) {
                double m11 = n,      m12 = sum_x,  m13 = sum_x2;
                double m21 = sum_x,  m22 = sum_x2, m23 = sum_x3;
                double m31 = sum_x2, m32 = sum_x3, m33 = sum_x4;

                double det = m11*(m22*m33 - m23*m32) - m12*(m21*m33 - m23*m31) + m13*(m21*m32 - m22*m31);
                if (abs(det) > 1e-10) {
                    double b1 = sum_y, b2 = sum_yx, b3 = sum_yx2;
                    a = (b1*(m22*m33 - m23*m32) - m12*(b2*m33 - m23*b3) + m13*(b2*m32 - m22*b3)) / det;
                    b = (m11*(b2*m33 - m23*b3) - b1*(m21*m33 - m23*m31) + m13*(m21*b3 - b2*m31)) / det;
                    c = (m11*(m22*b3 - b2*m32) - m12*(m21*b3 - b2*m31) + b1*(m21*m32 - m22*m31)) / det;
                    valid = true;
                }
            }

            #pragma omp parallel for
            for (int p = 0; p < num_paths; ++p) {
                real_t current_S = paths[p * (num_steps + 1) + s];
                real_t current_payoff = payoff(current_S, K, opt.type);
                if (valid && current_payoff > 0) {
                    double x = (double)current_S;
                    double cv = a + b * x + c * x * x;
                    if ((double)current_payoff > cv) {
                        V[p] = current_payoff;
                    } else {
                        V[p] *= df;
                    }
                } else {
                    V[p] *= df;
                }
            }
        }
        
        double sum = 0;
        #pragma omp parallel for reduction(+:sum)
        for (int p = 0; p < num_paths; ++p) sum += (double)V[p];
        return (real_t)((sum / num_paths) * df);
    };

    vector<real_t> V_center, V_up, V_dn;
    real_t price = simulate_paths(S0, V_center);
    real_t price_up = simulate_paths(S0 + dS, V_up);
    real_t price_dn = simulate_paths(S0 - dS, V_dn);

    PricingResult res;
    res.price = price;
    res.delta = (price_up - price_dn) / (2.0f * dS);
    res.gamma = (price_up - 2.0f * price + price_dn) / (dS * dS);
    return res;
}
