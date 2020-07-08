#pragma once
#include <opencv2/core.hpp>

struct DeskewAlgorithmRetType
{
	char name[256];
	float skewAngle = 0.f;
	float confidence = 0.f;
	bool valid = true;
	float duration = 0.f;
};

class DeskewAlgorithm
{
protected:
	bool m_pComputeVisuals;
	DeskewAlgorithmRetType mResult;
public:
	DeskewAlgorithm()
	{
		memset(mResult.name, 0, 256);
	}
	virtual DeskewAlgorithmRetType Run(cv::Mat &src, const char *fname, bool _pPomputeVisuals) = 0;
	void PrintStats()
	{
		printf("    %s >>\n\tdetected skew angle..... %.2f\370\n\tconfidence.............. %.0f\%%\n", mResult.name, mResult.skewAngle, mResult.confidence * 100.f);;
	}
};
