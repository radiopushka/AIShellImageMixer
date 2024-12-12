
all:
	cc network.c -march=native -Wall -lm -O3 -ffast-math -c -fPIC -o network.o
	cc nnnet.c -march=native -Wall -lm -O3 -ffast-math -c -fPIC -o nnnet.o
	ar rcs n_networks.a nnnet.o network.o
	cc test_net_nn.c n_networks.a -lm -o test.out
	rm ./*.o
