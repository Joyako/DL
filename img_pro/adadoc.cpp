#include "adath.h"
#include "ppht.h"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <cstdint>
#include <iostream>

/*
 * Wrapper function for adath
 * */
pybind11::array_t<unsigned char>
py_adath(pybind11::array_t<unsigned char, pybind11::array::c_style | pybind11::array::forcecast> src,
        unsigned int method, int xblock, int yblock, float k, float dR, int C)
{
  namespace py = pybind11;
  using namespace std;
  
  py::buffer_info info = src.request();
  string format = info.format;

  // sentinel
  if (info.ndim != 2 | info.itemsize != 1)
    throw std::runtime_error("error: src must be 2-d arrray");

  py::ssize_t nrows = info.shape[0];
  py::ssize_t ncols = info.shape[1];
  int stride = info.strides[0];                                   
  auto ptr = static_cast<unsigned char *>(info.ptr);
  cv::Mat src_(nrows, ncols, CV_8UC1, ptr, stride);
  cv::Mat dst_ = cv::Mat::zeros(nrows, ncols, CV_8UC1);

  // wrapped C++ function
  adath(src_, dst_, method, xblock, yblock, k, dR, C);

  py::ssize_t              ndim    = 2;
  std::vector<py::ssize_t> shape   = { nrows , ncols };
  std::vector<py::ssize_t> strides = { ssize_t(sizeof(unsigned char)*ncols) , sizeof(unsigned char) };
  
  // return 2-D NumPy array
  return
    py::array(py::buffer_info(
                dst_.data,                                      /* data as contiguous array  */
                sizeof(unsigned char),                          /* size of one scalar        */
                py::format_descriptor<unsigned char>::format(), /* data type                 */
                ndim,                                           /* number of dimensions      */
                shape,                                          /* shape of the matrix       */
                strides                                         /* strides for each axis     */
                ));      
}


/*
 * Wrapper function for pphough
 * */
pybind11::array_t<int>
py_pphough(pybind11::array_t<unsigned char, pybind11::array::c_style|pybind11::array::forcecast> src,
        float rho, float theta, int threshold,
        int min_line_len, int max_line_gap,
        int max_lines,
        float min_theta, float max_theta)
{
  namespace py = pybind11;
  using namespace std;
  
  py::buffer_info info = src.request();
  string format = info.format;

  // sentinel
  if (info.ndim != 2 | info.itemsize != 1)
    throw std::runtime_error("error: src must be 2-d arrray");

  // prepare for calling C++ function
  py::ssize_t nrows = info.shape[0];
  py::ssize_t ncols = info.shape[1];
  int stride = info.strides[0];
  auto ptr = static_cast<unsigned char *>(info.ptr);
  cv::Mat src_(nrows, ncols, CV_8UC1, ptr, stride);
 
  // wrapped C++ function
  vector<cv::Vec4i> lines;
  vector<cv::Vec2i> stats;
  pphough(src_, rho, theta, threshold,  min_line_len, max_line_gap, lines,
          max_lines, min_theta, max_theta, stats);

  // prepare output
  vector<int> lines_(lines.size()*6, 0);
  for (size_t i=0; i < lines.size(); ++i) {
    cv::Vec4i v = lines[i];

    int ii = 6*i;
    lines_[ii] = v[0];
    lines_[ii + 1] = v[1];
    lines_[ii + 2] = v[2];
    lines_[ii + 3] = v[3];

    cv::Vec2i u = stats[i];
    lines_[ii + 4] = u[0];
    lines_[ii + 5] = u[1];
  }
  
  py::ssize_t              ndim    = 2;
  std::vector<py::ssize_t> shape   = { py::ssize_t(lines.size()) , 6 };
  std::vector<py::ssize_t> strides = { sizeof(int)*6, sizeof(int) };
  // return 2-D NumPy array
  return
    py::array(py::buffer_info(
                lines_.data(),                        /* data as contiguous array  */
                sizeof(int),                          /* size of one scalar        */
                py::format_descriptor<int>::format(), /* data type                 */
                ndim,                                 /* number of dimensions      */
                shape,                                /* shape of the matrix       */
                strides                               /* strides for each axis     */
                ));
}


pybind11::array_t<unsigned char>
py_contrst_enhance(pybind11::array_t<unsigned char, pybind11::array::c_style|pybind11::array::forcecast> src,
                   const std::string& method, float param1, float param2)
{
  namespace py = pybind11;
  using namespace std;
  
  py::buffer_info info = src.request();
  string format = info.format;

  // sentinel
  if (info.ndim != 2 | info.itemsize != 1)
    throw std::runtime_error("error: src must be a 8-bit 2-d arrray");
  // if (method != "gamma")A
  //   throw std::runtime_error("error: only gamma enhancement is supported now");

  py::ssize_t nrows = info.shape[0];
  py::ssize_t ncols = info.shape[1];
  int stride = info.strides[0];                                   
  auto ptr = static_cast<unsigned char *>(info.ptr);
  cv::Mat src_(nrows, ncols, CV_8UC1, ptr, stride);
  cv::Mat dst_ = cv::Mat::zeros(nrows, ncols, CV_8UC1);

  // call wrapped function
  //gamma_lut(src_, dst_, param);
  contrst_enhance(src_, dst_, method, param1, param2);

  py::ssize_t              ndim    = 2;
  std::vector<py::ssize_t> shape   = { nrows , ncols };
  std::vector<py::ssize_t> strides = { ssize_t(sizeof(unsigned char)*ncols) , sizeof(unsigned char) };
  
  // return 2-D NumPy array
  return
    py::array(py::buffer_info(
                dst_.data,                                      /* data as contiguous array  */
                sizeof(unsigned char),                          /* size of one scalar        */
                py::format_descriptor<unsigned char>::format(), /* data type                 */
                ndim,                                           /* number of dimensions      */
                shape,                                          /* shape of the matrix       */
                strides                                         /* strides for each axis     */
                ));      
}

/*
 * Wrapper function for greycoprops
 * */
pybind11::array_t<float>
py_greycoprops(pybind11::array_t<int, pybind11::array::c_style | pybind11::array::forcecast> src)
{
  namespace py = pybind11;
  using namespace std;

  py::buffer_info info = src.request();
  string format = info.format;

  // sentinel
  if (info.ndim != 2 | info.itemsize != sizeof(int))
    throw std::runtime_error("error: src must be 2-d arrray");

  py::ssize_t nrows = info.shape[0];
  py::ssize_t ncols = info.shape[1];

  int stride = info.strides[0];
  auto ptr = static_cast<unsigned int *>(info.ptr);

  cv::Mat glcm_(nrows, ncols, CV_32S, ptr, stride);
  cv::Mat feature_ = cv::Mat::zeros(1, 3, CV_32FC1);
 
  // wrapped C++ function
  greycoprops(glcm_, feature_);

  py::ssize_t              ndim    = 2;
  std::vector<py::ssize_t> shape   = { 1 , 3 };
  std::vector<py::ssize_t> strides = { ssize_t(sizeof(float)*3) , sizeof(float) };

  // return 2-D NumPy array
  return
          py::array(py::buffer_info(
                  feature_.data,                                  /* data as contiguous array  */
                  sizeof(float),                                  /* size of one scalar        */
                  py::format_descriptor<float>::format(),         /* data type                 */
                  ndim,                                           /* number of dimensions      */
                  shape,                                          /* shape of the matrix       */
                  strides                                         /* strides for each axis     */
          ));
}

                                       
PYBIND11_MODULE(adadoc, m) {
  m.doc() = "AdaDoc"; // optional module docstring
  
  using namespace pybind11::literals;
  m.def("adath", &py_adath, "Adaptive Thresholding for Doc Image", "src"_a,
        "method"_a, "xblock"_a, "yblock"_a, "k"_a, "dR"_a, "C"_a);

  m.def("greycoprops", &py_greycoprops, "Calculate texture properties of a GLCM.", "glcm"_a);
  
  m.attr("ADATH_MEAN")       = uint32_t(ADATH_MEAN);
  m.attr("ADATH_NIBLACK")    = uint32_t(ADATH_NIBLACK);
  m.attr("ADATH_SAUVOLA")    = uint32_t(ADATH_SAUVOLA);
  m.attr("ADATH_WOLFJOLION") = uint32_t(ADATH_WOLFJOLION);
  m.attr("ADATH_KASAR")      = uint32_t(ADATH_KASAR);
  m.attr("ADATH_INVTHRESH")  = uint32_t(ADATH_INVTHRESH);

  m.def("ppht", &py_pphough, "Progressive Probabilistc Hough Transform",
        "src"_a, "rho"_a, "theta"_a, "threshold"_a, "min_line_len"_a,
        "max_line_gap"_a, "max_lines"_a, "min_theta"_a, "max_theta"_a);

  m.def("contrst_enhance", &py_contrst_enhance, "Image Contrast Enhancement",
        "src"_a, "method"_a, "param1"_a, "param2"_a);
}

