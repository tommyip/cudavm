#pragma once

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

#include <vector>

#include "vm.h"

const size_t BLOCK_SIZE = 256;

struct ScheduledInvocation {
    unsigned int program_id;
    std::vector<long long int> args;
    std::vector<unsigned int> account_indices;
};

class CudaVM {
public:
    unsigned int register_program(Program program);
    unsigned int register_account(Account account);
    void schedule_invocation(
        unsigned int program_id,
        std::vector<long long int>& args,
        std::vector<unsigned int>& account_indices
    );
    void execute_serial();
    void execute_parallel();

    std::vector<Account> accounts;

    // Feature flags
    bool group_independent_txns = false;
    // bool group_programs = false; // TODO: Write more programs!

private:
    std::vector<std::vector<int>> optimize_invocation_order();
    void execute_serial(std::vector<int>& invocation_indices);

    std::vector<Program> programs;
    std::vector<ScheduledInvocation> invocations;
};

template<typename T>
void push_with_padding(
    std::vector<T>& input,
    std::vector<T>& output,
    size_t chunk_length,
    T pad
) {
    std::copy(input.begin(), input.end(), std::back_inserter(output));
    size_t padding = chunk_length - input.size();
    if (padding > 0) {
        output.resize(output.size() + padding, pad);
    }
}

__host__ __device__
long long int int_pow(long long int base, long long int exp);
