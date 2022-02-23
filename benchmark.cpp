//
// Created by benjamin on 23.02.22.
//

#include "functions.h"
#include <chrono>


template<class timepoint>
double duration(timepoint t1, timepoint t2) {
    return std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count() / 1e6;
}

template<typename F>
void run() {
    unsigned int N = 10000;
    unsigned int L = 100;
    F *input = new F[N*L]();
    auto t1 = std::chrono::high_resolution_clock::now();
    for(unsigned int i = 0; i < N; i++) {
        F *output = allocAligned<F>(L);
        log1exp_simd(input + i * L, output, L);
        freeAligned(output);
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    std::cout << duration(t1, t2) << std::endl;
    std::cout << (duration(t1, t2) / (N*L)) * 1e9 << "ns per evalution" << std::endl;
}

int main() {
    run<float>();
}
