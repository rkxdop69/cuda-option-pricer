#pragma once
#include "common.h"

struct PricingResult {
    real_t price;
    real_t delta;
    real_t gamma;
};

class CPUOptionEngine {
public:
    static PricingResult priceEuropean(const OptionData& opt, int num_paths);
    static PricingResult priceAmerican(const OptionData& opt, int num_paths, int num_steps);
};
