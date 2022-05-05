cudavm: main.o vm.o cudavm.o simulator.o contracts.o
	nvcc -std=c++11 -arch sm_52 main.o vm.o cudavm.o simulator.o contracts.o -o cudavm

main.o: main.cpp utils.h
	nvcc -std=c++11 -arch sm_52 -x cu -c main.cpp -o main.o

vm.o: vm.cpp vm.h utils.h
	nvcc -std=c++11 -arch sm_52 -x cu -c vm.cpp -o vm.o

cudavm.o: cudavm.cu cudavm.h vm.h utils.h
	nvcc -std=c++11 -arch sm_52 -Xcompiler -fopenmp -c cudavm.cu -o cudavm.o

simulator.o: simulator.cpp simulator.h contracts.h cudavm.h utils.h
	nvcc -std=c++11 -arch sm_52 -x cu -c simulator.cpp -o simulator.o

contracts.o: contracts.cpp contracts.h
	nvcc -std=c++11 -arch sm_52 -x cu -c contracts.cpp -o contracts.o

run: cudavm
	./cudavm

clean:
	rm -f main.o vm.o cudavm.o simulator.o contracts.o cudavm
