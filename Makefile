cudavm: main.o vm.o cudavm.o simulator.o
	nvcc -arch sm_53 main.o vm.o cudavm.o simulator.o -o cudavm

main.o: main.cpp utils.h
	nvcc -std=c++11 -arch sm_52 -x cu -c main.cpp -o main.o

vm.o: vm.cpp vm.h utils.h
	nvcc -std=c++11 -arch sm_52 -x cu -c vm.cpp -o vm.o

cudavm.o: cudavm.cu cudavm.h vm.h utils.h
	nvcc -std=c++11 -arch sm_52 -c cudavm.cu -o cudavm.o

simulator.o: simulator.cpp simulator.h contracts.h cudavm.h utils.h
	nvcc -std=c++11 -arch sm_52 -x cu -c simulator.cpp -o simulator.o

run: cudavm
	./cudavm

clean:
	rm -f main.o vm.o cudavm.o simulator.o cudavm
