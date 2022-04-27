#pragma once

#define DEBUG false

#define debug_printf(fmt, ...) do { if (DEBUG) printf(fmt, __VA_ARGS__); } while (0)

#include <iostream>
#include <vector>

template<typename T>
void print_vector(std::vector<T> v) {
    for (auto x : v) {
        std::cout << x << " ";
    }
    std::cout << std::endl;
}
