#pragma once

// Use a simple typedef for precision to easily toggle between float and double
typedef float real_t;

enum class OptionType {
    Call,
    Put
};

struct OptionData {
    OptionType type;
    real_t strike;
    real_t expiry_years;
    real_t underlying_price;
    real_t bid;
    real_t ask;
    real_t implied_volatility;
    real_t risk_free_rate;
    real_t dividend_yield;
};
