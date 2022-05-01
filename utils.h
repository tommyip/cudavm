#pragma once

#define DEBUG true

#define debug_printf(fmt, ...) do { if (DEBUG) printf(fmt, __VA_ARGS__); } while (0)

#include <iostream>
#include <vector>
#include <chrono>

template<typename T>
void print_vector(std::vector<T> v) {
    for (auto x : v) {
        std::cout << x << " ";
    }
    std::cout << std::endl;
}

template<typename T>
T ceil_div(T x, T y) {
    return (x + y - 1) / y;
}

inline std::chrono::time_point<std::chrono::high_resolution_clock> clock_start() {
    return std::chrono::high_resolution_clock::now();
}

inline double clock_end(std::chrono::time_point<std::chrono::high_resolution_clock> start) {
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    return diff.count();
}
