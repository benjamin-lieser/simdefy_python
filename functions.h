//
// Created by benjamin on 22.02.22.
//

#ifndef SIMDEFY_PYTHON_FUNCTIONS_H
#define SIMDEFY_PYTHON_FUNCTIONS_H

#include "simd.h"

template<typename F>
void log1exp_scalar(F *input, F *output, size_t L) {
    for(size_t i = 0; i < L; i++) {
        output[i] = std::log((F)1.0 + std::exp(input[i]));
    }
}

template<typename F>
typename SIMD<F>::type log1exp_simd_register(typename SIMD<F>::type data) {
    auto ones = SIMD<F>::set(1.0);
    auto log2 = SIMD<F>::set(std::log((F)2.0));
    auto log2e = SIMD<F>::set(1.44269504089);
    auto exp_inner = SIMD<F>::mul(data, log2e);
    auto exp = SIMD<F>::exp2(exp_inner);
    auto exp_plus_1 = SIMD<F>::add(exp, ones);
    auto log_prior = SIMD<F>::log2(exp_plus_1);
    return SIMD<F>::mul(log_prior, log2);
}

template<typename F>
void log1exp_simd(F *input, F *output, size_t L) {
    size_t simd_blocks = L / SIMD<F>::count;

    for(size_t index = 0; index < simd_blocks; index++) {
        auto data = SIMD<F>::loadU(input + index * SIMD<F>::count);
        auto result = log1exp_simd_register<F>(data);
        SIMD<F>::storeU(output + index * SIMD<F>::count, result);
    }
}

#endif //SIMDEFY_PYTHON_FUNCTIONS_H
