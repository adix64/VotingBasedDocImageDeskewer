#pragma once
#include "SAAPI_utils.h"
#include "DeskewAlgorithm.hpp"
class RowVariance_SkewDetector : public DeskewAlgorithm
{
public:
	int histoSize = 4096;

	cv::Mat inputImage;
	cv::Mat squareFrame;
	char filename2[128];

	int SKEWangle_slider2 = 45, SKEWangle_slider_max2 = 90;
	float downscaleFact2;
	int displayHeight;
	int displayWidth;

	std::vector<cv::Vec2f> topV, bottomV, leftV, rightV;
	std::vector<cv::Vec2f> rTopV, rBottomV, rLeftV, rRightV;
	unsigned int *vScanlineAccumTop;
	unsigned int *vScanlineAccumBottom;
	unsigned int *vScanlineAccumLeft;
	unsigned int *vScanlineAccumRight;

	float topVariance, topMean;
	float bottomVariance, bottomMean;
	float leftVariance;
	float rightVariance;
	float maxBottomAccum = -FLT_MAX;
	float maxTopAccum = -FLT_MAX;


	~RowVariance_SkewDetector()
	{
		delete vScanlineAccumBottom;
		delete vScanlineAccumTop;
		delete vScanlineAccumLeft;
		delete vScanlineAccumRight;
	}
	void ComputeHistoForDisplay(cv::Mat &, cv::Mat &);
	void ComputeConfidence();
	DeskewAlgorithmRetType Run(cv::Mat &src, const char *fname, bool _pComputeVisuals) override;

	float ScanlineAccumulateGetVariance(unsigned int *dst, std::vector<cv::Vec2f> &points, int n, float &maxVal, float &meanVal);

	float RowVariance_Update2(float skewAngle, bool computeVisuals = true);

	void RowVariance_on_trackbar2(int);
};

