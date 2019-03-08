#pragma once

#include <string>
#include <opencv/cv.hpp>

enum {
	ADATH_MEAN = 0x0000,
	ADATH_NIBLACK = 0x0001,
	ADATH_SAUVOLA = 0x0002,
	ADATH_WOLFJOLION = 0x0003,

  ADATH_KASAR = 0X0004,

	ADATH_INVTHRESH = 0x0010
};

enum{
  RLS_VERTICAL = 0,
  RLS_HORIZONTAL = 1,
};


void
adath(cv::Mat &src, cv::Mat &dst,
      unsigned int method, int xblock, int yblock,
      float k, float dR, int C = 0);

void
runlen_smear(const cv::Mat& src, cv::Mat& dst, int th, unsigned int method);


void
contrst_enhance(const cv::Mat& src, cv::Mat& dst, const std::string& method,
                float param1, float param2);

void
gamma_lut(const cv::Mat& src, cv::Mat& dst, float gamma);
void
linear_gamma(const cv::Mat& src, cv::Mat& dst, float delta, float gamma);

void 
greycoprops(cv::Mat &glcm, cv::Mat &feature);

