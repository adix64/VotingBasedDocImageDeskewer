#pragma once

#include <iostream>
#include <utility>
//#include <opencv2/core/core.hpp>
//#include <opencv2/highgui/highgui.hpp>
//#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
//#define IPLIMAGEIPL
void showImg(cv::Mat &image, char *name, float scale = 1.f, bool createWindow = true);

#ifdef IPLIMAGEIPL
IplImage* drawImageInSquare(IplImage* src);
IplImage* rotateImage(const IplImage *src, float angleDegrees);
#endif
cv::Mat drawImageInSquare(cv::Mat &src);

// Rotate the image clockwise (or counter-clockwise if negative).
// Remember to free the returned image.

cv::Mat rotateImage(cv::Mat &src, double angle);


std::string type2str(int type);


void RotatePoints(std::vector<cv::Vec2f> &points, std::vector<cv::Vec2f> &dst,
	float angle, cv::Vec2f rotPivot);

void DrawPoints(cv::Mat &frame, std::vector<cv::Vec2f> &points, cv::Vec3b &color, int radius);