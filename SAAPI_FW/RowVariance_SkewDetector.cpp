#include "RowVariance_SkewDetector.h"
#include <chrono>
#define WINDOW_HEIGHT 630
using namespace cv;


float RowVariance_SkewDetector::ScanlineAccumulateGetVariance(unsigned int *dst, std::vector<Vec2f> &points, int n, float &maxVal, float &meanVal)
{
	memset(dst, 0, histoSize * sizeof(unsigned int));
	meanVal = 0.f;
	float hpIdx;
	for (size_t i = 0; i < points.size(); i++)
	{
		hpIdx = (float)(points[i][1] * histoSize) / (float)n;
		dst[max(min(histoSize - 1,(int)hpIdx), 0)]++;
		meanVal += 1.f;
	}
	meanVal /= histoSize;
	
	float variance = 0.f;
	float err;
	maxVal = 0.f;
	for (size_t i = 0; i < histoSize; i++)
	{
		if (dst[i] <= 0)
			continue;
		err = dst[i] - meanVal;
		err *= err;
		variance += err;
		if (dst[i] > maxVal)
			maxVal = dst[i];
	}
	return variance / histoSize;
}

float RowVariance_SkewDetector::RowVariance_Update2(float skewAngle, bool computeVisuals)
{
	std::chrono::high_resolution_clock::time_point tstart = std::chrono::high_resolution_clock::now();

	int n = squareFrame.cols, hf = squareFrame.cols / 2;
	RotatePoints(topV, rTopV, skewAngle, Vec2f(hf, hf));
	RotatePoints(bottomV, rBottomV, skewAngle, Vec2f(hf, hf));
	RotatePoints(leftV, rLeftV, skewAngle, Vec2f(hf, hf));
	RotatePoints(rightV, rRightV, skewAngle, Vec2f(hf, hf));
	
	topVariance = ScanlineAccumulateGetVariance(vScanlineAccumTop, rTopV, n, maxTopAccum, topMean);
	bottomVariance = ScanlineAccumulateGetVariance(vScanlineAccumBottom, rBottomV, n, maxBottomAccum, bottomMean);
	//leftVariance = ScanlineAccumulateGetVariance(vScanlineAccumLeft, rLeftV, n);
	//rightVariance = ScanlineAccumulateGetVariance(vScanlineAccumRight, rRightV, n);

	if (computeVisuals)
	{
		Mat frame(squareFrame.rows, squareFrame.cols, CV_8UC3, Scalar(255, 255, 255));
		DrawPoints(frame, rTopV,    Vec3b(128, 0, 0),4);
		DrawPoints(frame, rBottomV, Vec3b(0, 0, 128),4);
		//DrawPoints(frame, rLeftV,   Vec3b(0, 0, 255),5);
		//DrawPoints(frame, rRightV,  Vec3b(0, 255, 0),5);

		char winName[128];
		if (m_pComputeVisuals)
		{
			memset(winName, 0, 128);
			sprintf_s(winName, "Corrected Skew %s", filename2);
			showImg(frame, winName);
			cv::resizeWindow(winName, 1024, 1024);
			moveWindow(winName, 0, 0);
		}
		//cvMoveWindow("Rotated Img - Connected Components", displayWidth, 0);
	}
	std::chrono::high_resolution_clock::time_point tend = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(tend - tstart).count();

	//std:: cout << "DURATION :" <<duration << "\n";
	return bottomVariance + topVariance;// +leftVariance + rightVariance;	
}

//Callback for any slider touch
void RowVariance_SkewDetector::RowVariance_on_trackbar2(int)
{
	RowVariance_Update2((SKEWangle_slider2 - 45.f) * ((float)CV_PI / 180.0f));
}
void RowVariance_SkewDetector::ComputeHistoForDisplay(Mat &histogramB, Mat &histogramT)
{
	float bottomCount, topCount;
	for (int xx = 0; xx < histoSize; xx++)
	{
		bottomCount = (vScanlineAccumBottom[xx]) / maxBottomAccum * 99;
		topCount = (vScanlineAccumTop[xx]) / maxTopAccum * 99;

		//bottomCount = min(bottomCount, 99.f);
		for (int j = 0; j < bottomCount; j++)
		{
			histogramB.at<Vec3b>(xx, j) = Vec3b(0, 0, 255);
		}
		for (int j = 0; j < topCount; j++)
		{
			histogramT.at<Vec3b>(xx, j) = Vec3b(255, 0, 0);
		}
	}
}
//The "Run(..)" routine
//void RowVariance_SkewDetector(const char *filename)
			
DeskewAlgorithmRetType RowVariance_SkewDetector::Run(Mat &src, const char *fname, bool _pComputeVisuals)
{
	m_pComputeVisuals = _pComputeVisuals;
	sprintf(mResult.name, "RowVariance_SkewDetector");
	char winName[128];

	strcpy_s(filename2, fname);
	//inputImage = cvLoadImage(filename, 0);
	inputImage = src;
	//resize(inputImage, inputImage, Size(inputImage.cols, inputImage.rows));
	cv::threshold(inputImage, inputImage, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
	
	squareFrame = drawImageInSquare(inputImage);


	vScanlineAccumTop = new unsigned int[histoSize];
	vScanlineAccumBottom = new unsigned int[histoSize];
	vScanlineAccumLeft = new unsigned int[histoSize];
	vScanlineAccumRight = new unsigned int[histoSize];
	for (int i = 0; i < histoSize; i++)
	{
		vScanlineAccumTop[i] = vScanlineAccumBottom[i] =
		vScanlineAccumLeft[i] = vScanlineAccumRight[i] = 0.f;
	}
	// Showing the color image
	downscaleFact2 = (float)WINDOW_HEIGHT / (float)squareFrame.rows;
	displayHeight = (float)squareFrame.rows* downscaleFact2;
	displayWidth = (float)squareFrame.cols* downscaleFact2;

	Mat cln = squareFrame.clone();

	//CvMemStorage *memStorage = cvCreateMemStorage();
	//CvSeq *seq = NULL;
	std::vector<std::vector<Point> > contours;
	std::vector<Vec4i> hierarchy;
	findContours(cln, contours, hierarchy, cv::RETR_LIST, cv::CHAIN_APPROX_NONE, Point(0, 0));
	Rect boundbox;
	for (int i = 0; i< contours.size(); i++)
	{
		boundbox = boundingRect(Mat(contours[i]));

		topV.push_back(Vec2f(boundbox.x + boundbox.width / 2, boundbox.y));
		bottomV.push_back(Vec2f(boundbox.x + boundbox.width / 2, boundbox.y + boundbox.height));
		leftV.push_back(Vec2f(boundbox.x, boundbox.y + boundbox.height / 2));
		rightV.push_back(Vec2f(boundbox.x + boundbox.width, boundbox.y + boundbox.height / 2));
	}

	rTopV = topV;
	rBottomV = bottomV;
	rLeftV = leftV;
	rRightV = rightV;
	
	Mat frr(squareFrame.rows, squareFrame.cols, CV_8UC3, Scalar(255, 255, 255));
	//cvtColor(squareFrame, frr, cv	::COLOR_GRAY2BGR);//
	DrawPoints(frr, rTopV, Vec3b(128, 0, 0), 4);
	DrawPoints(frr, rBottomV, Vec3b(0, 0, 128), 4);
	//initial variance display
#ifdef SHOW_ALL_ROW_VAR
	memset(winName, 0, 128);
	sprintf_s(winName, "Input: %s", filename2);
	showImg(squareFrame, winName);
	resizeWindow(winName, 1024, 1024);
	moveWindow(winName, 0, 0);
	memset(winName, 0, 128);

	memset(winName, 0, 128);
	sprintf_s(winName, "Input Points: %s", filename2);
	showImg(frr, winName);
	resizeWindow(winName, 1024, 1024);
	moveWindow(winName, 0, 0);
	memset(winName, 0, 128);
#endif

	{
		RowVariance_Update2(0);
		Mat histogramB(histoSize, 100, CV_8UC3, Scalar(255, 255, 255));
		Mat histogramT(histoSize, 100, CV_8UC3, Scalar(255, 255, 255));
		ComputeHistoForDisplay(histogramB, histogramT);
		resize(histogramB, histogramB, Size(100, 768));
		resize(histogramT, histogramT, Size(100, 768));
		if (m_pComputeVisuals)
		{
			memset(winName, 0, 128);
			sprintf_s(winName, "HistogramOriginalT: %s", filename2);
			showImg(histogramT, winName);
			resizeWindow(winName, 100, 1024);
			moveWindow(winName, 1024, 0);
			memset(winName, 0, 128);
			sprintf_s(winName, "HistogramOriginalB: %s", filename2);
			showImg(histogramB, winName);
			resizeWindow(winName, 100, 1024);
			moveWindow(winName, 1145, 0);
		}
	}

	float bestT = -15, maxVariance = -FLT_MAX, variance;
	float avgVariance = 0.f;
	int iterations = 0;
	float metaHisto[300];

	for (float t = -15; t <= 15.f; t += .1f)
	{
		variance = RowVariance_Update2(t * ((float)CV_PI / 180.0f), false);
		avgVariance += variance;
		metaHisto[iterations] = variance;
		if (variance > maxVariance)
		{
			bestT = t;
			maxVariance = variance;
		}
		iterations++;
	}
	avgVariance /= (float)iterations;

	float metaVariance = 0.f, dev;
	for (int i = 0; i < 300; i++)
	{
		dev = metaHisto[i] - avgVariance;
		dev *= dev;
		metaVariance += dev;
	}
	metaVariance /= 300.f;
	float metaStandardDeviation = sqrt(metaVariance);
	//printf("Best variance :%f    Best T: %f\n", maxVariance, bestT);
	variance = RowVariance_Update2(bestT * ((float)CV_PI / 180.0f));
	mResult.confidence = pow(1.f - metaStandardDeviation/(maxVariance - avgVariance), 1.05f);
	//mResult.confidence = pow(mResult.confidence, 0.1f);
	{
		Mat histogramB(histoSize, 100, CV_8UC3, Scalar(255, 255, 255));
		Mat histogramT(histoSize, 100, CV_8UC3, Scalar(255, 255, 255));
		ComputeHistoForDisplay(histogramB, histogramT);
		resize(histogramB, histogramB, Size(100, 768));
		resize(histogramT, histogramT, Size(100, 768));
		if (m_pComputeVisuals)
		{
			memset(winName, 0, 128);
			sprintf_s(winName, "HistogramDeskewedT: %s", filename2);
			showImg(histogramT, winName);
			resizeWindow(winName, 100, 1024);
			moveWindow(winName, 1024, 0);
			memset(winName, 0, 128);
			sprintf_s(winName, "HistogramDeskewedB: %s", filename2);
			showImg(histogramB, winName);
			resizeWindow(winName, 100, 1024);
			moveWindow(winName, 1145, 0);
		}

	}	
	ComputeConfidence();
	//sprintf_s(winName, "Rotated Img - Connected Components %s", filename2);
	//createTrackbar("SKEWangle", winName, &SKEWangle_slider2, SKEWangle_slider_max2, RowVariance_on_trackbar2);
	mResult.skewAngle = bestT;
	PrintStats();
	return mResult;
}

#define clamp(x, a, b) max(a,min(b,x)) 
void RowVariance_SkewDetector::ComputeConfidence()
{
	//vScanlineAccumBotto
	float bottomStandardDeviation = sqrt(bottomVariance);
	float dev, biggestDev = -FLT_MAX;
	for (int i = 0; i < histoSize; i++)
	{
		if (vScanlineAccumBottom[i] > 0)
		{
			dev = vScanlineAccumBottom[i] - bottomMean;
			dev *= dev;
			if(dev > biggestDev)
			{
				biggestDev = dev;
			}
		}
	}
	//float confidence = bottomStandardDeviation / biggestDev * 10.f;
	//mResult.confidence = bottomStandardDeviation / sqrt(biggestDev);
	//confidence = clamp(confidence, 0.f, 1.f);
	//printf("THE VARIANCE is %f\n", bottomVariance);
	//printf("THE Confidence is %f\n",  confidence);
}
