#pragma once
#include "SAAPI_utils.h"
#include "DeskewAlgorithm.hpp"
class HoughSkewDetector : public DeskewAlgorithm
{
public:
	char filename[128];
	int Hthresh = 300, HmaxLinLen = 100, HmaxLineGap = 20;

	cv::Mat inputImageSQ;

	DeskewAlgorithmRetType Run(cv::Mat &input, const char *filename, bool p_computeVisuals) override;
	void DrawInfLineDocCenter(cv::Mat &dst, cv::Vec4f segment, cv::Scalar color, int thickness);
};
