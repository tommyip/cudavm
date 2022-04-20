cudavm: main.o vm.o
	g++ -Wall -std=c++11 -o cudavm main.o vm.o

main.o: main.cpp
	g++ -Wall -std=c++11 -c main.cpp -o main.o

vm.o: vm.cpp
	g++ -Wall -std=c++11 -c vm.cpp -o vm.o

run: cudavm
	./cudavm

clean:
	rm -f main.o vm.o cudavm
