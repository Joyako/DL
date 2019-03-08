#include "adath.h"

#include <iostream>
#include <string>
#include <algorithm> // for fill, transform
#include <cctype> // for std::tolower


/*!
 * Mean value of low outliers as pseudo-minimum.
 * */
static unsigned char
hist_pmin(const cv::Mat& src, float p)
{
  using namespace std;
  using namespace cv;

  int nrows = src.rows;
  int ncols = src.cols;

  // calculate histogram
  int hist[256];
  fill(hist, hist + 256, 0);
  for (int i = 0; i < nrows; ++i) {
    const unsigned char *p = src.ptr<unsigned char>(i);
    for (int j = 0; j < ncols; ++j) {
      ++hist[p[j]];
    }
  }

  // locate p% value
  int nr_px = 0;
  int idx_p = 0;
  {
    int idx = 0;
    for (; idx < ncols; ++idx) {
      nr_px += hist[idx];
      if (nr_px > (nrows * ncols) * p)
        break;
    }

    if (idx == 0)
      idx_p = idx;
    else if (nr_px - hist[idx] > 0) {
      idx_p = idx - 1;
      nr_px = nr_px - hist[idx];
    } else
      idx_p = idx;
  }

  // calculate the mean of p-% outlier as pseudo-minimum value;
  int sum_outlier = 0;
  for (int i=0; i<=idx_p; ++i) {
    sum_outlier += i*hist[i];
  }
  unsigned char pminval = (unsigned char)(float(sum_outlier) / nr_px);

  // TODO: test using idx_p as the pseudo-minimum value;

  return pminval;
}


static void
moving_max(cv::Mat &src, cv::Mat &dst, int xblock, int yblock)
{
  using namespace std;
  using namespace cv;

  CV_Assert(src.type() == CV_8UC1 && dst.type() == CV_8UC1);
  CV_Assert(src.size() == dst.size());

  const int nrows = src.rows;
  const int ncols = src.cols;

  int x_hlf = floor(xblock / 2.);
  int y_hlf = floor(yblock / 2.);

  for (int j = x_hlf; j < ncols - x_hlf; ++j) {
    int hist[256];
    fill(hist, hist + 256, 0);

    for (int i = y_hlf; i < nrows - y_hlf; ++i) {
      // initialize histogram inside first block;
      if (i == y_hlf) {
        for (int ii = 0; ii < yblock; ++ii) {
          const uchar *p = src.ptr<uchar>(i - y_hlf + ii);
          for (int jj = 0; jj < xblock; ++jj) {
            int val = p[j - x_hlf + jj];
            ++hist[val];
          }
        }
        // locate max
        int k = 255;
        for (; k >= 0; --k) {
          if (hist[k])
            break;
        }
        dst.ptr<uchar>(i)[j] = k;
      } else {
        // update histogram in following blocks;
        const uchar *p1 = src.ptr<uchar>(i - y_hlf - 1);
        const uchar *p2 = src.ptr<uchar>(i + y_hlf);
        for (int jj = 0; jj < xblock; ++jj) {
          --hist[p1[jj]];
          ++hist[p2[jj]];
        }
        // locate max
        int k = 255;
        for (; k >= 0; --k) {
          if (hist[k])
            break;
        }
        dst.ptr<uchar>(i)[j] = k;
      }
    }
  }

}



/*! \brief Adaptive Thresholding for Document Binarization
 *
 * For input image, src, the foreground is considered as low-value grayscale pixel,
 * while background as high-value grayscale. That is "white paper and black
 * character". This assumption is important, since it will influence the local
 * statistics in the moving block; the bright pixels (background) are dominant
 * and statistics are highly influenced by them. If the assumption for input is
 * reverted, please change parameters.
 *
 * For method ADATH_KASAR, the input image should be processed first by
 * gaussian-smoothing and grayscale morphological black-hat operation. NOTE that
 * usually black-hat presents results with inverted brightness: dark pixels to
 * bright and bright to dark.
 *
 * Typical parameter values suggested from original papers.
 * ADATH_NIBLACK: k=-0.2; xblock and yblock are set to covering at least of 1-2 characters.
 * ADATH_SAUVOLA: k=0.5; dR is the dynamics of the standard deviation fixed to 128;
 * ADATH_WOLFJOLION:
 *
 * REF:
 * 1. C. Wolf, J. Jolion and et al, "Text Localization, Enhancement and Binarization
 * in Multimedia Documents", ICPR 2002.
 * 2. C. Wolf and David Doermann, "Binarization of Low Quality Text using a Markov
 * Random Field Model", ICPR 2002.
 *
 * \param
 *
 */
void
adath(cv::Mat& src, cv::Mat& dst,
      unsigned int method, int xblock, int yblock,
      float k, float dR, int C/*=0*/)
{
  using namespace std;
  using namespace cv;

  // sentinel
  CV_Assert(src.type() == CV_8UC1 && src.dims == 2);
  CV_Assert(xblock % 2 == 1 && yblock % 2 == 1);
  CV_Assert(src.size() == dst.size());
  CV_Assert(dst.type() == CV_8UC1);

  int x_hlf = cvFloor(xblock / 2.);
  int y_hlf = cvFloor(yblock / 2.);
  int nr_el = xblock * yblock;

  unsigned char hival, loval;
  if ((method & 0x00f0) == ADATH_INVTHRESH) {
    hival = 0;
    loval = 255;
  } else {
    hival = 255;
    loval = 0;
  }

  //extend pixels by copying edge pixel
  //cv::Mat extend;
  cv::copyMakeBorder(src, src, y_hlf, y_hlf, x_hlf, x_hlf, cv::BORDER_REPLICATE);
  
  int ncols = src.cols;
  int nrows = src.rows;

  //  todo: find minimum 1% outlier as pseudo minimum value?
  unsigned char pminval = hist_pmin(src, 0.001);

  //calculate min and max value
  double minval_px, maxval_px;
  minMaxLoc(src, &minval_px, &maxval_px);

  // compute integral map of pixel value and squared value
  Mat map_intg = cv::Mat::zeros(src.size(), CV_32S);
  Mat map_sqintg = cv::Mat::zeros(src.size(), CV_64F);
  integral(src, map_intg, map_sqintg, CV_32S, CV_64F);

  // store mean value and standard deviance
  Mat map_mu = cv::Mat::zeros(src.size(), CV_32F);
  Mat map_stdd = cv::Mat::zeros(src.size(), CV_32F);

  // iterate all positions to get statistics;
  float max_stdd = 0.0;
  for (int r = y_hlf; r < nrows - y_hlf; ++r) {
    //
    // calc mean and stdd in block
    //
    const int *top1 = map_intg.ptr<int>(r - y_hlf);
    const int *bottom1 = map_intg.ptr<int>(r + y_hlf);

    const double *top2 = map_sqintg.ptr<double>(r - y_hlf);
    const double *bottom2 = map_sqintg.ptr<double>(r + y_hlf);

    float *p1 = map_mu.ptr<float>(r);
    float *p2 = map_stdd.ptr<float>(r);
    for (int c = x_hlf; c < ncols - x_hlf; ++c) {
      // calculate statistics in block
      double mu;
      double stdd;
      {
        // mean value
        mu = top1[c-x_hlf] + bottom1[c+x_hlf] - top1[c+x_hlf] - bottom1[c-x_hlf];
        mu = mu / nr_el;
        // standard deviance
        // sum of squared
        double ssq = top2[c - x_hlf] + bottom2[c + x_hlf] - top2[c + x_hlf]
                - bottom2[c - x_hlf];
        stdd = (ssq - mu * mu * nr_el) / (nr_el - 1);
        stdd = sqrt(stdd);
        // keep maximum value of stdd
        if (stdd > max_stdd)
          max_stdd = (float)stdd;
      }
      // store statistics
      p1[c] = mu;
      p2[c] = stdd;
    }
  }
  
  // todo: debug
  Mat map_max;
  if ( (method & 0x000f) == ADATH_KASAR) {
    map_max = Mat::zeros(src.size(), CV_8UC1);
    moving_max(src, map_max, xblock, yblock);
  }

 
  // calculate threshold value
  for (int r = y_hlf; r < nrows - y_hlf; ++r) {
    const float *p1 = map_mu.ptr<float>(r);
    const float *p2 = map_stdd.ptr<float>(r);
    const uchar* p3 = map_max.ptr<uchar>(r);

    uchar *p = src.ptr<uchar>(r);
    uchar *q = dst.ptr<uchar>(r - y_hlf);
    for (int c = x_hlf; c < ncols - x_hlf; ++c) {
      float mu = p1[c];
      float stdd = p2[c];

      float th = 0.0;
      // calculate threshold
      switch (method & 0x000f) {
        case ADATH_MEAN:
          th = mu;
          break;
        case ADATH_NIBLACK:
          th = mu + k * stdd;
          break;
        case ADATH_SAUVOLA:
          th = mu * (1 - k * (1 - stdd / dR));
          break;
        case ADATH_WOLFJOLION: {
          // todo:  ?????
          minval_px = pminval;
          float alpha = 1 - stdd / max_stdd;
          th = mu - k * alpha * (mu - minval_px);
        }
          break;
        case ADATH_KASAR: {
          // todo: add maxval
          const int maxval = p3[c];
          th = k * maxval;
        }
          break;
        default:
          cerr << "Error: Unknown threshold type.\n";
          exit(1);
      }
      // adjust threshold by C
      th += C;
      // do thresholding
      q[c - x_hlf] = (int) th < p[c] ? hival : loval;
    }
  }

  // TODO: handle boundary problem
}


/*!
 * @brief
 *
 * REF:
 * 1. http://crblpocr.blogspot.com/2007/06/run-length-smoothing-algorithm-rlsa.html
 * 2. http://crblpocr.blogspot.com/2007/06/determination-of-run-length-smoothing.html
 *
 * @param src
 * @param dst
 * @param th
 * @param method
 */

void
runlen_smear(const cv::Mat& src, cv::Mat& dst, int th, unsigned int method)
{
  using namespace std;
  using namespace cv;

  // sentinel
  CV_Assert( src.type() == CV_8UC1 && src.dims == 2);
  CV_Assert( dst.type() == CV_8UC1 && dst.dims == 2);
  CV_Assert( src.size() == dst.size() );

  int nrows = src.rows;
  int ncols = src.cols;

  if (method == RLS_HORIZONTAL) {
    for (int i = 0; i < nrows; ++i) {
      const unsigned char *p = src.ptr<unsigned char>(i);
      unsigned char *q = dst.ptr<unsigned char>(i);
      for (int j = 0; j < ncols; ++j) {
        // 1's is still 1's in sequence y;
        if (p[j] != 0)
          q[j] = 0xff;
        else {
          // count continuous zeros
          int idx_beg = j;
          int idx_end = j + 1;
          for (; idx_end < ncols && p[idx_end] == 0; ++idx_end) {};

          int val;
          int gap = idx_end - idx_beg - 1;
          if (gap < th)
            val = 0xff;
          else
            val = 0x00;
          for (int ii = idx_beg; ii < idx_end; ++ii)
            q[ii] = val;
          j = idx_end - 1;
        }
      }
    }
  } else if (method == RLS_VERTICAL) {
    for (int j = 0; j < ncols; ++j) {
      for (int i = 0; i < nrows; ++i) {
        unsigned char p = src.ptr<unsigned char>(i)[j];
        // 1's is still 1's in sequence y;
        if (p)
          dst.ptr<unsigned char>(i)[j] = 0xff;
        else {
          // count continuous zeros
          int idx_beg = i;
          int idx_end = i + 1;
          for (; idx_end < nrows && src.ptr<unsigned char>(idx_end)[j] == 0; ++idx_end)
          {};

          int val;
          int gap = idx_end - idx_beg - 1;
          if (gap < th)
            val = 0xff;
          else
            val = 0x00;
          for (int ii = idx_beg; ii < idx_end; ++ii)
             dst.ptr<unsigned char>(ii)[j] = val;
          i = idx_end - 1;
        }
      }
    }
  }

}


/*! \brief Contrast Enhancement Function
 *
 *  \param src     8-bit single-channel image;
 *  \param dst     returned 8-bit single-channel image; in-place return is supported;
 *  \param method  "gamma" and "linear_gamma" are supported;
 *  \param param1  for "gamma" and "linear_gamma", it is parameter of exponential;
 *  \param param2  for "linear_gamma", it is the conjunction point of exponential segment and linear segment.
  *
 * */
void
contrst_enhance(const cv::Mat& src, cv::Mat& dst, const std::string& method,
                float param1, float param2)
{
  using namespace std;
  using namespace cv;

  string str = method;
  transform(str.begin(), str.end(), str.begin(),
            [](unsigned char c) { return tolower(c); }
            );

  if (method == "gamma")
    gamma_lut(src, dst, param1);
  else if (method == "linear_gamma")
    linear_gamma(src, dst, param1, param2);
  else
    throw runtime_error("error: method is not supported.");

}


/*! \brief Contrast Enhancement Function: gamma transform
 *
 *  \param src     8-bit single-channel image;
 *  \param dst     returned 8-bit single-channel image; in-place return is supported;
 *  \param gamma   a typical value is 2.2;
 *
 * */
void
gamma_lut(const cv::Mat& src, cv::Mat& dst, float gamma)
{
  using namespace std;
  using namespace cv;

  CV_Assert( src.type() == CV_8UC1 && dst.type() == CV_8UC1 );
  CV_Assert( src.size() == dst.size() );

  //  build lookup-table
  Mat lut(1, 256, CV_8UC1);
  unsigned char* buf = lut.ptr<unsigned char>(0);
  for (int i=0; i<256; ++i) {
    float v = 255*pow(i/255., gamma);
    v = round(v);
    if (v > 255) v = 255;
    buf[i] = v;
  }

  // apply gamma transform
  int nrows = src.rows;
  int ncols = src.cols;
  for (int i=0; i<nrows; ++i) {
    const unsigned char* p = src.ptr<unsigned char>(i);
    unsigned char* q = dst.ptr<unsigned char>(i);
    for (int j=0; j<ncols; ++j) {
      q[j] = buf[p[j]];
    }
  }

}


/*! \brief Contrast Enhancement Function: linear gamma transform
 *
 *  \param src     8-bit single-channel image;
 *  \param dst     returned 8-bit single-channel image; in-place return is supported;
 *  \param gamma   a typical value is 2.2;
 *  \param delta   connection point between gamma curve and linear curve;
 *
 * */
void
linear_gamma(const cv::Mat& src, cv::Mat& dst, float gamma, float delta)
{
  using namespace std;
  using namespace cv;

  CV_Assert( src.type() == CV_8UC1 && dst.type() == CV_8UC1 );
  CV_Assert( src.size() == dst.size() );
  CV_Assert( delta < 1 && delta > 0 );

  // filling lut
  vector<unsigned char> lut(256, 0);
  {
    // in [0, delta], exponential function is calculated
    int idx_delta = floor(delta * 255);
    for (int i = 0; i <= idx_delta; ++i) {
      float v = 255 * pow(i / 255., gamma);
      v = round(v);
      if (v > 255) v = 255;
      lut[i] = v;
    }

    // in (delta, 1], linear equation is calculated
    // calculate slope and intercept of the line
    float k = (1. - lut[idx_delta] / 255.) / (1. - idx_delta / 255.);
    float b = 1. + k * (-1.);
    for (int i = idx_delta + 1; i < 256; ++i) {
      float v = k * (i / 255.) + b;
      v = round(v*255);
      if (v > 255) v = 255;
      lut[i] = v;
    }
  }

  const int nrows = src.rows;
  const int ncols = src.cols;
  for (int i=0; i < nrows; ++i) {
    const unsigned char* p = src.ptr<unsigned char>(i);
    unsigned char* q = dst.ptr<unsigned char>(i);
    for (int j=0; j < ncols; ++j)
      q[j] = lut[p[j]];
  }
}

/*Calculate texture properties of a GLCM.
  Compute a feature of a grey level co-occurrence matrix to serve as
  a compact summary of the matrix. The properties are computed as follows:
  
  - 'contrast': :math:`\\sum_{i,j=0}^{levels-1} P_{i,j}(i-j)^2`
  - 'dissimilarity': :math:`\\sum_{i,j=0}^{levels-1}P_{i,j}|i-j|`
  - 'homogeneity': :math:`\\sum_{i,j=0}^{levels-1}\\frac{P_{i,j}}{1+(i-j)^2}`
  - 'ASM': :math:`\\sum_{i,j=0}^{levels-1} P_{i,j}^2`
  - 'energy': :math:`\\sqrt{ASM}`
  - 'correlation':
       .. math:: \\sum_{i,j=0}^{levels-1} P_{i,j}\\left[\\frac{(i-\\mu_i) \\
                  (j-\\mu_j)}{\\sqrt{(\\sigma_i^2)(\\sigma_j^2)}}\\right]

  Parameters
  ----------
  glcm: 32-bit signal channel matrix
        Input matrix: 'glcm' is the grey-level co-occurrence histogram
        for which to compute the specified property. 
*/
void greycoprops(cv::Mat &glcm, cv::Mat &feature) 
{
	//std::cout << glcm << std::endl;
    int num_level  = glcm.rows;
    int num_level2 = glcm.cols;
    CV_Assert(num_level == num_level2);
	CV_Assert(glcm.type() == CV_32SC1);

    // create weight for specified property
    // define restore matrix
    cv::Mat M = cv::Mat(num_level, num_level, CV_32SC1);
    cv::Mat weights0 = cv::Mat(num_level, num_level2, CV_32SC1);
    cv::Mat weights1 = cv::Mat(num_level, num_level2, CV_32SC1);
    cv::Mat weights2 = cv::Mat(num_level, num_level2, CV_32FC1);
	
    cv::Mat ret0 = cv::Mat(num_level, num_level2, CV_32FC1);
	cv::Mat ret1 = cv::Mat(num_level, num_level2, CV_32FC1);
	cv::Mat ret2 = cv::Mat(num_level, num_level2, CV_32FC1);
    
    for (size_t i = 0; i < num_level; ++i) {
		int *m = M.ptr<int>(i);
        for (size_t j = 0; j < num_level; ++j) {
            m[j] = i - j;
        }
    }
	
    // calculate weight
	// contrast weights0 = M ** 2
    weights0 = M.mul(M);
	// dissimilarity
    weights1 = cv::abs(M);
	// homogeneity
    // weights2 = 1. / (1. + M * M)
	weights0.convertTo(weights0, CV_32FC1);
    cv::divide(1., (1. + weights0), weights2);

	// matrix multiply between weight and glcm 
	glcm.convertTo(glcm, CV_32FC1);
    ret0 = glcm.mul(weights0);
	weights1.convertTo(weights1, CV_32FC1);  
    ret1 = glcm.mul(weights1);
	ret2 = glcm.mul(weights2);
	//std::cout << ret0 << std::endl;

	float s0 = 0., s1 = 0., s2 = 0.; 
	//s0 = cv::sum(ret0)[0];
	//s1 = cv::sum(ret1)[0];
	//s2 = cv::sum(ret2)[0];
    // compute property for each GLCM
    for (int i = 0; i < num_level; ++i) {
        float *r0 = ret0.ptr<float>(i);
		float *r1 = ret1.ptr<float>(i);
		float *r2 = ret2.ptr<float>(i);	
        for (int j = 0; j < num_level; ++j) {
            s0 += r0[j];
			s1 += r1[j];
			s2 += r2[j];
        }
    }
	feature.at<float>(0, 0) = s0;
	feature.at<float>(0, 1) = s1;
	feature.at<float>(0, 2) = s2;
	//std::cout << s0 << '\n' << s1 << '\n' << s2 << std::endl;
}


#if 0
#include <boost/filesystem.hpp> // for basename

int main()//(int argc, char** argv)
{
  using namespace std;
  using namespace cv;
  namespace fs=boost::filesystem;

  string ifnames[] = {
          "mm.png",
          "0411.jpg",
          "0412.jpg",
          "0805.jpg",
          "0877.jpg",
          "0904.jpg",
          "0907.jpg",
          "0910.jpg",
          "0911.jpg",
          "0916.jpg",
          "0944.jpg",
          "IMG_1878.JPG",
          "IMG_1884.JPG"
  };

  for (auto f : ifnames) {
    f = "../../../data/" + f;
    // read image and scale to width of 1000
    Mat img;
    {
      cv::Mat I = cv::imread(f.c_str(), 0);
        if (I.empty()) {
          cerr << "open file failed." << endl;
          return -1;
        }

      cv::resize(I, img, cv::Size(0, 0), 2000. / I.cols, 2000. / I.cols,
                 cv::INTER_LINEAR);
    }

    cv::Mat dst = cv::Mat::zeros(img.size(), CV_8U);

    // seems Gaussian smoothing does it well
    cv::GaussianBlur(img, img, Size(0,0), 1., 1.);
    /*
     * k=0.1 works well for these camera images.
     *
     * xsize = 51, ysize = 51;
     *
     * ADATH_WOLFJOLION: k=0.1
     * ADATH_NIBLACK: k=-0.2
     * ADATH_SAUVOLA: k=0.1, dR = 64
     * C = 0;
     * */
    unsigned int method = ADATH_KASAR;ADATH_WOLFJOLION;ADATH_NIBLACK; ADATH_SAUVOLA;
    method |= ADATH_INVTHRESH;
    int xsiz = 51;
    int ysiz = 51;
    float k = 0.9;0.1;-0.2; 0.1;
    float dR = 64;
    int C = 0;
    adath(img, dst, method, xsiz, ysiz, k, dR, C);
    //dst = adaptiveThreshold(img);

    string ofname = string("../../../tmp/") + "bw_" + fs::basename(f) + ".png";
    cv::imwrite(ofname.c_str(), dst);
  }

  return 0;
}
#endif

#if 0
int main()
{
  using namespace cv;
  using namespace std;

  unsigned char buf[] = {0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0};
  Mat src(1, sizeof(buf), CV_8UC1, buf);
  Mat dst(1, sizeof(buf), CV_8UC1);

  runlen_smear(src, dst, 4, RLS_HORIZONTAL);

  for (int i=0; i < sizeof(buf); ++i) {
    cout << (int) dst.ptr<unsigned char>(0)[i] << ", ";
    if ((i+1) % 4 == 0 )
      cout << endl;
  }

  return 0;
}

#endif

#if 0
int main()
{
  using namespace std;
  using namespace cv;

  string fname = "../0307_bw.png";
  Mat bw = cv::imread(fname.c_str(), 0);
  if (bw.empty()) {
    cerr << "file open failed" << endl;
    return -1;
  }

  Mat neg_bw;
  //bitwise_not(bw, neg_bw);
  neg_bw = bw;
  imwrite("bw.png", neg_bw);

  Mat H(neg_bw.size(), neg_bw.type());
  runlen_smear(neg_bw, H, 30, RLS_HORIZONTAL);
  Mat V(neg_bw.size(), neg_bw.type(), Scalar(0xff));
  runlen_smear(neg_bw, V, 40, RLS_VERTICAL);

  imwrite("ret_H.png", H);
  imwrite("ret_V.png", V);

  Mat L;
  bitwise_and(H, V, L);

  imwrite("ret_L.png", L);

  return 0;
}
#endif


#if 0
int main()
{
  using namespace std;
  using namespace cv;

  string fname = "../0028_gray.png";
  Mat gray = cv::imread(fname.c_str(), 0);
  if (gray.empty()) {
    cerr << "file open failed" << endl;
    return -1;
  }

  Mat M = gray.clone();
  gamma_lut(gray, gray, 2.2);

  imwrite("../ret_gamma.png", gray);

  Mat ret = M.clone();
  contrst_enhance(M, ret, "linear_gamma", 2.2, 0.5);

  imwrite("../ret_lingamma.png", ret);

  return 0;
}
#endif


