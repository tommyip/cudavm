#include <iostream>
#include <vector>

#include "cudavm.h"
#include "vm.h"
#include "simulator.h"

int main() {
    CudaVM vm;
    generate_transactions(vm, 10, 1000, 100000);

    // vm.execute_serial();
    vm.execute_parallel();

    for (size_t i = 0; i < vm.accounts.size(); ++i) {
        std::cout << "Account " << i << ": ";
        vm.accounts[i].display();
        std::cout << std::endl;
    }
}
