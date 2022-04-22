#pragma once

#include <vector>

#include "vm.h"

class CudaVM {
public:
    void schedule_invocation(
        std::vector<long long int>& params,
        std::vector<int>& account_indices
    );
    void execute_serial();
    void execute_parallel();

private:
    std::vector<Program> programs;
    std::vector<Account> accounts;

    int n_calls = 0;
    std::vector<long long int> params;
    std::vector<int> account_indices;
};
