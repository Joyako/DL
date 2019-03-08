#pragma once

#include <opencv/cv.hpp>


void
pphough(cv::Mat &image,
        float rho, float theta, int threshold,
        int min_line_len, int max_line_gap,
        std::vector<cv::Vec4i> &line_ends, int max_lines,
        float min_theta, float max_theta,
        std::vector<cv::Vec2i> &stats_line);

void
ppht(cv::InputArray image_, cv::OutputArray line_ends_,
     double rho, double theta, int threshold,
     int min_line_len, int max_line_gap,
     double min_theta, double max_theta,
     cv::OutputArray stats_line);
