#pragma once
#include "SAAPI_utils.h"
#include "DeskewAlgorithm.hpp"

class FFT_SkewDetector : public DeskewAlgorithm
{
public:
	bool useProjectionProfiling;
	char filename[128];
	int Hthresh = 100, HmaxLinLen = 512, HmaxLineGap = 10;
	//int blockSz = 43, C = 0;
	int rotAngleSliderVal = 45;
	cv::Mat inputImageSQ;
	std::vector<cv::Vec2f> pointsV, rPointsV;

	DeskewAlgorithmRetType Run(cv::Mat &input, const char *filename, bool _pComputeVisuals) override;
	FFT_SkewDetector(bool _pp) : useProjectionProfiling(_pp) {}
	~FFT_SkewDetector()
	{
		delete vScanlineAccumPoints;
		delete hScanlineAccumPoints;
	}
	DeskewAlgorithmRetType VarianceSolution(cv::Mat &src, const char *fname);
	float RowVarianceUpdate(float skewAngle, bool computeVisuals = true);
	void ComputeHistoForDisplay(cv::Mat &histogramV, cv::Mat &histogramH);
	float ScanlineAccumulateGetVariance(unsigned int *dst, std::vector<cv::Vec2f> &points,
										int n, float &maxVal, float &meanVal,int horizontal = 1);

	void ComputeVarianceConfidence();
	void ComputeVisuals(std::vector<cv::Vec2f> &pPts, char* winName, cv::Vec2i pos, cv::Vec2i size);

	unsigned int *vScanlineAccumPoints, *hScanlineAccumPoints;
	int histoSize = 1024;
	const float lowerLimit = cos(3.1415926f * 0.25f) * histoSize / 4;
	const float upperLimit = histoSize - 1 - cos(3.1415926f * 0.25f) * histoSize / 4;

	float maxPointsAccumV = -FLT_MAX, maxPointsAccumH = -FLT_MAX;
	float pointsMeanV, pointsMeanH, pointsVarianceV, pointsVarianceH;
	float downscaleFact2;
	int displayHeight;
	int displayWidth;
	int N;
};
