fuzzer: fuzzer.cc
	g++ -std=c++11 -g -Wall -Wextra -O3 $< -o $@ -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lnvinfer -lcudart