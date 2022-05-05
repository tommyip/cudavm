#pragma once

#include "cudavm.h"

void generate_transactions(
    CudaVM& vm,
    size_t n_pools,
    size_t n_swappers,
    size_t n_swap_txns,
    size_t n_payment_txns
);
