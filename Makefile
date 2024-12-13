all:
	make -C ./CAI
	cc main.c CAI/n_networks.a config/CF_file.c image_reader/reader.c image_writer/writer.c -march=native -lm -lpng -Wall -o nnNet

