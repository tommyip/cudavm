cudavm: main.o vm.o cudavm.o utils.h
	nvcc -arch sm_53 main.o vm.o cudavm.o -o cudavm

main.o: main.cpp
	g++ -Wall -std=c++11 -c main.cpp -o main.o

vm.o: vm.cpp vm.h
	g++ -Wall -std=c++11 -c vm.cpp -o vm.o

cudavm.o: cudavm.cu cudavm.h
	nvcc -std=c++11 -arch sm_53 -c cudavm.cu -o cudavm.o

run: cudavm
	./cudavm

clean:
	rm -f main.o vm.o cudavm
