#pragma once
#include "common.h"
#include "CPUOptionEngine.h"

class GPUOptionEngine {
public:
    static PricingResult priceEuropean(const OptionData& opt, int num_paths);
    static PricingResult priceAmerican(const OptionData& opt, int num_paths, int num_steps);
};
