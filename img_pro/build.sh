g++ -O3 -Wall -shared -std=c++11 -fPIC `python3 -m pybind11 --includes` -L/home/joy/anaconda3/lib -lopencv_core -lopencv_imgproc adadoc.cpp adath.cpp -o adadoc`python3-config --extension-suffix`
