#include <iostream>
#include <vector>
#include <chrono>

#include "cudavm.h"
#include "vm.h"
#include "simulator.h"

const size_t N_POOLS = 10000;
const size_t N_SWAPPERS = 10000;
const size_t N_TXNS = 1000000;

int main() {
    CudaVM vm;
    generate_transactions(vm, N_POOLS, N_SWAPPERS, N_TXNS);

    const std::vector<Account> accounts_backup(vm.accounts);

    auto cpu_start = clock_start();
    vm.execute_serial();
    auto cpu_secs = clock_end(cpu_start);
    size_t cpu_tps = (double)N_TXNS / cpu_secs;
    printf("CPU | Time elapsed: %fs | TPS: %d\n", cpu_secs, cpu_tps);

    const std::vector<Account> cpu_accounts_snapshot(vm.accounts);
    vm.accounts = accounts_backup;

    auto gpu_start = clock_start();
    vm.execute_parallel();
    double gpu_secs = clock_end(gpu_start);
    size_t gpu_tps = (double)N_TXNS / gpu_secs;
    printf("GPU | Time elapsed: %fs | TPS: %d\n", gpu_secs, gpu_tps);

    double speedup = cpu_secs / gpu_secs;
    printf("Speedup: %fx\n", speedup);

    if (cpu_accounts_snapshot == vm.accounts) {
        printf("CPU & GPU accounts state equal");
    } else {
        printf("CPU & GPU accounts state NOT equal");
    }

    // for (size_t i = 0; i < vm.accounts.size(); ++i) {
    //     std::cout << "Account " << i << ": ";
    //     vm.accounts[i].display();
    //     std::cout << std::endl;
    // }
}
