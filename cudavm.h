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

const size_t BLOCK_SIZE = 128;
const size_t WARP_SIZE = 32;

struct ScheduledInvocation {
    int program_id;
    std::vector<int> args;
    std::vector<int> account_indices;
};

class Chunks {
    const int max_programs;
    const int max_chunks;
    const int max_accounts;
    std::vector<bool> allocated_accounts;

public:
    // chunks -> programs -> indices
    std::vector<std::vector<std::vector<int>>> gpu_indices;
    std::vector<int> cpu_indices;

    Chunks(int max_programs, int max_chunks, int max_accounts);
    void add_invocation(int i, ScheduledInvocation& invocation);
    int max_chunk_size();
};

class CudaVM {
public:
    int register_program(Program program);
    int register_account(Account account);
    void schedule_invocation(
        int program_id,
        std::vector<int> args,
        std::vector<int> account_indices
    );
    void execute_serial();
    void execute_parallel();

    std::vector<Account> accounts;
    std::vector<ScheduledInvocation> invocations;

private:
    Chunks* optimize_invocation_order();
    void execute_serial(std::vector<int>& invocation_indices);

    std::vector<Program> programs;
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
int int_pow(int base, int exp);
