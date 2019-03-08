#include "ppht.h"

#include <algorithm>
#include <iterator>
#include <iostream>
#include <random> // for random_device; C11
#include <cmath> // for round; C11


/*! \brief Progressive Probabilistic Hough Transform for Lines.
 *
 *  This function is modified from "opencv/modules/imgproc/src/hough.cpp", and
 *  a bug is fixed.
 *
 *  \param image         8-bit single-channel image;
 *  \param rho           distance resolution for line equation;
 *  \param theta         angle resolution for line equation;
 *  \param threshold     threshold of line existence for hough voting;
 *  \param min_line_len  acceptable minimum length of detected line;
 *  \param max_line_gap  acceptable continuous zero's along the detected line;
 *  \param line_ends     end points for detected lines;
 *  \param max_lines     maximum limited of returned lines of detection;
 *  \param min_theta     only lines in a range [min_theta, max_theta] of theta will be detected;
 *  \param max_theta     see above. Note (max_theta - min_theta) must be less than pi;
 *  \param stats_line    each row corresponds to a line's statistics; for each row,
 *                       the first element is the number of points, and the second
 *                       element is the total number of  discontinuous points.
 *
 * */
void
pphough(cv::Mat &image,
        float rho, float theta, int threshold,
        int min_line_len, int max_line_gap,
        std::vector<cv::Vec4i> &line_ends, int max_lines,
        float min_theta, float max_theta,
        std::vector<cv::Vec2i> &stats_line)
{
  using namespace std;
  using namespace cv;

  // input paramter sentinel
  CV_Assert(image.type() == CV_8UC1);
  CV_Assert(min_theta < max_theta && max_theta-min_theta <= CV_PI+FLT_EPSILON);

  // todo: random seed
  //random_device rd;
  //RNG rng(rd());

  // fixed RNG for debugging
  RNG rng((uint64) -1);

  int width = image.cols;
  int height = image.rows;

  /*
   *  Line equation is expressed in polar form (rho$, theta). When theta
   *  is limited to (-pi/2, pi/2], rho may take negative value: in
   *  the case of actual "theta" is out of range of (-pi/2, pi/2], rho is
   *  negative. The negative sign of rho compensates the other half plane to
   *  (-pi/2, pi/2].
   *
   *  ?? The maximum of rho is max_rho = sqrt(width^2 + height^2);
   *  The size of rhos should be ceil(max_rho)*2 considering negative sign.
   * */
  int nr_angle = (int)round( (max_theta - min_theta) / theta );
  //int nr_rho = (int)round( ((width + height) * 2 + 1) / rho );
  int len_diag = (int)round( sqrt(width*width + height*height +0.0f) );
  int nr_rho = (int)ceil( (len_diag + 1) * 2. / rho );

  Mat accum = Mat::zeros(nr_angle, nr_rho, CV_32SC1);
  Mat cand(height, width, CV_8UC1);
  // if voted, marked as non-zero
  Mat voted = Mat::zeros(height, width, CV_8UC1);

  //  initialize the lookup table for cos, sin
  vector<float> lut_trig(nr_angle * 2);
  float irho = 1. / rho;
  for (int n=0; n < nr_angle; ++n) {
    double angle = min_theta + n*theta;
    lut_trig[n * 2] = (float) (cos(angle) * irho);
    lut_trig[n * 2 + 1] = (float) (sin(angle) * irho);
  }
  const float *p_lut = &lut_trig[0];

  //
  // stage 1. collect non-zero image points
  //
  Point pt;
  std::vector<Point> nzloc;
  for (pt.y = 0; pt.y < height; pt.y++) {
    const uchar *p = image.ptr(pt.y);
    uchar *q = cand.ptr(pt.y);
    for (pt.x = 0; pt.x < width; pt.x++) {
      if (p[pt.x]) {
        q[pt.x] = (uchar) 255;
        nzloc.push_back(pt);
      } else
        q[pt.x] = 0;
    }
  }

  //
  // stage 2. process all the points in random order
  //
  int count = (int) nzloc.size();
  for (; count > 0; count--) {
    // choose random point out of the remaining ones
    int idx = rng.uniform(0, count);
    Point point = nzloc[idx];
    Point line_end[2];

    const int shift = 16;

    // "remove" it by overriding it with the last element
    nzloc[idx] = nzloc[count - 1];

    // check if it has been excluded already (i.e. belongs to some other line)
    int i = point.y, j = point.x;
    if ( !(cand.ptr<uchar>(i)[j]) )
      continue;

    // marked as voted
    voted.ptr<uchar>(i)[j] = 0xff;

    // update accumulator, find the most probable line
    int max_val = threshold - 1;
    int max_n = 0;
    for (int n = 0; n < nr_angle; n++) {
      int r = cvRound(j * p_lut[n * 2] + i * p_lut[n * 2 + 1]);
      // add offset to rho to make sure index positive
      r += (nr_rho - 1) / 2;
      int *p_accu = accum.ptr<int>(n);
      int val = ++p_accu[r];
      if (max_val < val) {
        max_val = val;
        max_n = n;
      }
    }

    // if it is too "weak", continue with another point
    if (max_val < threshold)
      continue;

    // from the current point walk in each direction
    // along the found line and extract the line segment
    float a = -p_lut[max_n * 2 + 1];
    float b = p_lut[max_n * 2];
    int xflag;
    int x0 = j, y0 = i;
    int dx0, dy0;
    if (fabs(a) > fabs(b)) {
      xflag = 1;
      dx0 = a > 0 ? 1 : -1;
      dy0 = cvRound(b * (1 << shift) / fabs(a));
      y0 = (y0 << shift) + (1 << (shift - 1));
    } else {
      xflag = 0;
      dy0 = b > 0 ? 1 : -1;
      dx0 = cvRound(a * (1 << shift) / fabs(b));
      x0 = (x0 << shift) + (1 << (shift - 1));
    }

    // actual number of points on the line.
    int nr_pts = 0;

    /* there are two directions for a point to search for the line;
     * */
    int gap_total = 0;
    for (int k = 0; k < 2; k++) {
      int dx = dx0, dy = dy0;
      if (k > 0)
        dx = -dx, dy = -dy;

      // walk along the line using fixed-point arithmetics,
      // stop at the image border or in case of too big gap
      int gap = 0;
      int x = x0, y = y0;
      for (;; x += dx, y += dy) {
        int i1, j1;
        if (xflag) {
          j1 = x;
          i1 = y >> shift;
        } else {
          j1 = x >> shift;
          i1 = y;
        }

        // border sentinel
        if (j1 < 0 || j1 >= width || i1 < 0 || i1 >= height)
          break;

        // for each non-zero point:
        //    update line end,
        //    clear the cand element
        //    reset the gap
        const uchar is_valid = cand.ptr<uchar>(i1)[j1];
        if (is_valid) {
          gap_total += gap;
          gap = 0;
          line_end[k].y = i1;
          line_end[k].x = j1;
          ++nr_pts;
        } else if (++gap > max_line_gap)
          break;
      }
    }

    // todo: is it a fast approximation for line length??
    int good_line = std::abs(line_end[1].x - line_end[0].x) >= min_line_len ||
                std::abs(line_end[1].y - line_end[0].y) >= min_line_len;

    // revisit the points in 2 directions of the line, and
    // restore accumulation array;
    for (int k = 0; k < 2; k++) {
      int x = x0, y = y0, dx = dx0, dy = dy0;

      if (k > 0)
        dx = -dx, dy = -dy;

      // walk along the line using fixed-point arithmetics,
      // stop at the image border or in case of too big gap
      for (;; x += dx, y += dy) {
        int i1, j1;
        if (xflag) {
          j1 = x;
          i1 = y >> shift;
        } else {
          j1 = x >> shift;
          i1 = y;
        }

        // for each non-zero point:
        //    clear the cand element
        //    reset the gap
        uchar &valid_pt = cand.ptr<uchar>(i1)[j1];
        if (valid_pt) {
          if (good_line) {
            /* FIX: Here is a bug for OpenCV version.
             *
             * Before undoing contribution of (i1,j1) to accum, we need to check
             * if the point made a vote.
             * */
            uchar& is_voted_pt = voted.ptr<uchar>(i1)[j1];
            if ( is_voted_pt ) {
              // undo the voting;
              for (int n = 0; n < nr_angle; n++) {
                int r = cvRound(j1 * p_lut[n * 2] + i1 * p_lut[n * 2 + 1]);
                r += (nr_rho - 1) / 2;
                int *p_accu = accum.ptr<int>(n);
                p_accu[r]--;
               }
            }
            // todo: reset vote?
            is_voted_pt = 0;
            //voted.ptr<uchar>(i1)[j1] = 0;
          }
          // remove the point from candidates
          valid_pt = 0;
        }

        if (i1 == line_end[k].y && j1 == line_end[k].x)
          break;
      }
    }

    // store the detected line
    if (good_line) {
      Vec4i lr(line_end[0].x, line_end[0].y, line_end[1].x, line_end[1].y);
      line_ends.push_back(lr);
      Vec2i v(nr_pts, gap_total);
      stats_line.push_back(v);
      if ((int) line_ends.size() >= max_lines)
        return;
    }
  }

}


/*! \brief Progressive Probabilistic Hough Transform for Lines.
 *
 *  It is a simple wrapper for pphough, and the difference is that output is stored
 *  in OpenCV array instead of vector.
 *
 * */
void ppht(cv::InputArray image_, cv::OutputArray line_ends_,
          double rho, double theta, int threshold,
          int min_line_len, int max_line_gap,
          double min_theta, double max_theta, cv::OutputArray stats_line)
{
  using namespace cv;
  using namespace std;

  // preprare for input
  Mat image = image_.getMat();
  vector<Vec4i> lines;
  vector<Vec2i> stats;

  pphough(image,
          (float) rho, (float) theta, threshold,
          min_line_len, max_line_gap, lines, INT_MAX,
          min_theta, max_theta, stats);

  // prepare output
  Mat(lines).copyTo(line_ends_);
  Mat(stats).copyTo(stats_line);
}


#if 0
#include <boost/filesystem.hpp> // for basename
#include <string>
int main()
{
  using namespace cv;
  using namespace std;

  namespace fs=boost::filesystem;

  string ifnames[] = {
          "ret_seg.png",
          "0411_bw.png",
          "0412_bw.png",
          "0805_bw.png",
          "0877_bw.png",
          "0904_bw.png",
          "0907_bw.png",
          "0910_bw.png",
          "0911_bw.png",
          "0916_bw.png",
          "0944_bw.png",
          "IMG_1878_bw.png",
          "IMG_1884_bw.png"
  };

  for (auto f : ifnames) {
    string infname = "../../../tmp/" + f;

    string ofname1 = string("../../../tmp/") + fs::basename(f) + "_ppht.png";
    string ofname2 = string("../../../tmp/") + fs::basename(f) + "_ocv.png";

    cout << "processing " << infname << endl;

    // read image as grayscale
    Mat src = imread(infname, 0);
    if (src.empty()) {
      cout << "can not open " << infname << endl;
      return -1;
    }

    Mat dst = src;
    //Canny(src, dst, 50, 200, 3);

    vector<Vec4i> lines_ppht;
    vector<Vec2i> stats;
    {
      vector<Vec4i> vlines;

      double rho = 1;
      double theta = CV_PI / 180;
      int threshold = 50;
      int min_line_len = 50;
      int max_line_gap = 10;
      double min_theta = -CV_PI / 12;
      double max_theta = CV_PI / 12;
      ppht(dst.clone(), vlines, rho, theta, threshold, min_line_len,
           max_line_gap, min_theta, max_theta, stats);

      vector<Vec4i> hlines;
      min_theta = CV_PI * 5 / 12;
      max_theta = CV_PI * 7 / 12;
      ppht(dst.clone(), hlines, rho, theta, threshold, min_line_len,
           max_line_gap, min_theta, max_theta, stats);

      lines_ppht.assign(vlines.begin(), vlines.end());
      lines_ppht.insert(lines_ppht.end(), hlines.begin(), hlines.end());
    }

    // draw lines
    Mat plane;
    cvtColor(dst, plane, CV_GRAY2BGR);
    for (auto l: lines_ppht)
      line(plane, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0, 0, 255), 2,
           CV_AA);
    imwrite(ofname1.c_str(), plane);


    vector<Vec4i> lines_ocv;
    HoughLinesP(dst.clone(), lines_ocv, 1, CV_PI / 180, 50, 50, 10);

    // draw lines
    cvtColor(dst, plane, CV_GRAY2BGR);
    for (auto l : lines_ocv)
      line(plane, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0, 0, 255), 2,
           CV_AA);
    imwrite(ofname2.c_str(), plane);
  }

  return 0;
}
#endif
