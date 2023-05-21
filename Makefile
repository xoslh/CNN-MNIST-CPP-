default all:
	$(CXX) src/main.cpp -I . -o main -O3 -mavx -mfma -ffast-math
