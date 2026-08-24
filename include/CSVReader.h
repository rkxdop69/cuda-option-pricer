#pragma once
#include <string>
#include <vector>
#include <iostream>
#include "common.h"
#include "rapidcsv.h"

class CSVReader {
public:
    static std::vector<OptionData> read(const std::string& filename) {
        std::vector<OptionData> data;
        try {
            // Read CSV with row 0 as header labels (like pandas DataFrame)
            rapidcsv::Document doc(filename, rapidcsv::LabelParams(0, -1));
            
            std::vector<std::string> types = doc.GetColumn<std::string>("Type");
            std::vector<real_t> strikes = doc.GetColumn<real_t>("Strike");
            std::vector<real_t> expiries = doc.GetColumn<real_t>("Expiry_Years");
            std::vector<real_t> underlyings = doc.GetColumn<real_t>("Underlying_Price");
            std::vector<real_t> bids = doc.GetColumn<real_t>("Bid");
            std::vector<real_t> asks = doc.GetColumn<real_t>("Ask");
            std::vector<real_t> ivs = doc.GetColumn<real_t>("Implied_Volatility");
            std::vector<real_t> rfrs = doc.GetColumn<real_t>("Risk_Free_Rate");
            size_t num_rows = doc.GetRowCount();
            data.reserve(num_rows);

            std::vector<real_t> divs;
            try {
                divs = doc.GetColumn<real_t>("Dividend_Yield");
            } catch (...) {
                divs.assign(num_rows, 0.0f);
            }

            for (size_t i = 0; i < num_rows; ++i) {
                OptionData opt;
                opt.type = (types[i] == "Call" || types[i] == "call") ? OptionType::Call : OptionType::Put;
                opt.strike = strikes[i];
                opt.expiry_years = expiries[i];
                opt.underlying_price = underlyings[i];
                opt.bid = bids[i];
                opt.ask = asks[i];
                opt.implied_volatility = ivs[i];
                opt.risk_free_rate = rfrs[i];
                opt.dividend_yield = (i < divs.size()) ? divs[i] : 0.0f;
                data.push_back(opt);
            }
        } catch (const std::exception& e) {
            std::cerr << "Error reading CSV file " << filename << ": " << e.what() << std::endl;
        }

        return data;
    }
};
