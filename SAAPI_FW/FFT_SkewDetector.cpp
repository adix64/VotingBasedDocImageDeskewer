#include "FFT_SkewDetector.h"
#include <chrono>
#define WINDOW_HEIGHT 630
#define FFT_DESKEW_ANGLE_STEP (CV_PI / 720.f)

using namespace cv;
cv::Mat GetFFT(cv::Mat &img)
{
	cv::Mat padded;                            //expand input image to optimal size
	int m = getOptimalDFTSize(img.rows);
	int n = getOptimalDFTSize(img.cols); // on the border add zero values
	copyMakeBorder(img, padded, 0, m - img.rows, 0, n - img.cols, BORDER_CONSTANT, Scalar::all(0));
	cv::Mat planes[] = { Mat_<float>(padded), cv::Mat::zeros(padded.size(), CV_32F) };
	cv::Mat complexI;
	merge(planes, 2, complexI);         // Add to the expanded another plane with zeros
	dft(complexI, complexI);            // this way the result may fit in the source matrix
										// compute the magnitude and switch to logarithmic scale
										// => log(1 + sqrt(Re(DFT(I))^2 + Im(DFT(I))^2))
	split(complexI, planes);                   // planes[0] = Re(DFT(I), planes[1] = Im(DFT(I))
	magnitude(planes[0], planes[1], planes[0]);// planes[0] = magnitude
	cv::Mat magI = planes[0];
	magI += Scalar::all(1);                    // switch to logarithmic scale
	log(magI, magI);
	// crop the spectrum, if it has an odd number of rows or columns
	magI = magI(Rect(0, 0, magI.cols & -2, magI.rows & -2));
	// rearrange the quadrants of Fourier image  so that the origin is at the image center
	int cx = magI.cols / 2;
	int cy = magI.rows / 2;
	cv::Mat q0(magI, Rect(0, 0, cx, cy));   // Top-Left - Create a ROI per quadrant
	cv::Mat q1(magI, Rect(cx, 0, cx, cy));  // Top-Right
	cv::Mat q2(magI, Rect(0, cy, cx, cy));  // Bottom-Left
	cv::Mat q3(magI, Rect(cx, cy, cx, cy)); // Bottom-Right
	cv::Mat tmp;                           // swap quadrants (Top-Left with Bottom-Right)
	q0.copyTo(tmp);
	q3.copyTo(q0);
	tmp.copyTo(q3);
	q1.copyTo(tmp);                    // swap quadrant (Top-Right with Bottom-Left)
	q2.copyTo(q1);
	tmp.copyTo(q2);
	normalize(magI, magI, 0, 1, NORM_MINMAX); // Transform the matrix with float values into a
											  // viewable image form (float between values 0 and 1).

											  //cv::medianBlur(magI, magI, 5); cv::medianBlur(magI, magI, 5);
											  //cv::GaussianBlur(magI, magI, cvSize(5, 5), 10);
	magI.convertTo(magI, CV_8UC1, 255);
	return magI;
}
DeskewAlgorithmRetType FFT_SkewDetector::Run(cv::Mat &src, const char *fname, bool _pComputeVisuals)
{
	m_pComputeVisuals = _pComputeVisuals;

	char winName[128];

	strcpy_s(filename, fname);
	cv::Mat inputImage = src;
	cv::threshold(inputImage, inputImage, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

	inputImageSQ = drawImageInSquare(inputImage);
	cv::resize(inputImageSQ, inputImageSQ, cv::Size(1024, 1024));
	
	cv::Mat fft1 = GetFFT(inputImageSQ);
 	
	cv::Mat thresholdedfft1;
	cv::threshold(fft1, thresholdedfft1, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

	cv::Mat fftAmplitude = fft1.clone();

	for (int y = 0; y < fftAmplitude.rows; y++)
	{
		for (int x = 0; x < fftAmplitude.cols; x++)
		{
			float stretchedVal = fftAmplitude.at<uchar>(y, x);
			stretchedVal = 2 * (float)(stretchedVal - 128) + 32;
			fftAmplitude.at<uchar>(y, x) = saturate_cast<uchar>(stretchedVal);
		}
	}
	equalizeHist(fftAmplitude, fftAmplitude);

	cv::Mat threshImg;
	cv::threshold(fftAmplitude, threshImg, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
	
	if (!useProjectionProfiling)
	{
		std::vector<Vec4i> lines;
		cv::HoughLinesP(threshImg, lines, 1, FFT_DESKEW_ANGLE_STEP, Hthresh, HmaxLinLen, HmaxLineGap);

		cv::Mat HoughFFTimg;
		cv::cvtColor(threshImg, HoughFFTimg, cv::COLOR_GRAY2BGR);//
		for (size_t i = 0; i < lines.size(); i++)
			line(HoughFFTimg, Point(lines[i][0], lines[i][1]), 
				Point(lines[i][2], lines[i][3]), Scalar(255, 200, 200), 2, cv::LineTypes::LINE_AA);

		if (m_pComputeVisuals)
		{
			memset(winName, 0, 128);
			sprintf_s(winName, "Original %s", filename);
			showImg(inputImageSQ, winName);
			cv::resizeWindow(winName, 384, 384);
			cv::moveWindow(winName, 0, 0);

			memset(winName, 0, 128);
			sprintf_s(winName, "FFT original %s", filename);
			showImg(fft1, winName);
			cv::resizeWindow(winName, 384, 384);
			cv::moveWindow(winName, 384, 0);

			memset(winName, 0, 128);
			sprintf_s(winName, "FFT orig thresh%s", filename);
			showImg(thresholdedfft1, winName);
			cv::resizeWindow(winName, 384, 384);
			cv::moveWindow(winName, 768, 0);

			memset(winName, 0, 128);
			sprintf_s(winName, "FFT amplitude %s", filename);
			showImg(fftAmplitude, winName);
			cv::resizeWindow(winName, 384, 384);
			cv::moveWindow(winName, 0, 384);

			memset(winName, 0, 128);
			sprintf_s(winName, "FFT OTSU %s", filename);
			showImg(threshImg, winName);
			cv::resizeWindow(winName, 384, 384);
			cv::moveWindow(winName, 384, 384);

		}

		float biggestDotY = 0.5f;
		Vec2f v, bestV;
		bool foundSomeAngle = false;
		Vec4i l;
		int bestIdx;
		for (size_t i = 0; i < lines.size(); i++)
		{
			l = lines[i];
			v = Vec2f(l[2], l[3]) - Vec2f(l[0], l[1]);

			if (l[1] > l[3])
				v = -v;//v points upwards
			float pmag = sqrt(v[0] * v[0] + v[1] * v[1]);
			v /= pmag;//normalize v

			if (v[1] > biggestDotY && v[1] > .85f && v[1] < 1.f)
			{
				foundSomeAngle = true;
				biggestDotY = v[1];
				bestV = v;
				bestIdx = i;
			}
		}
#ifdef DO_WITH_HOUGH
		if (!foundSomeAngle)
			return mResult;
#endif
		if (foundSomeAngle)
		{
			l = lines[bestIdx];
			line(HoughFFTimg, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0, 0, 255), 5, cv::LineTypes::LINE_AA);

			if (m_pComputeVisuals)
			{
				memset(winName, 0, 128);
				sprintf_s(winName, "FFT Hough Lines %s", filename);
				showImg(HoughFFTimg, winName);
				cv::resizeWindow(winName, 384, 384);
				cv::moveWindow(winName, 768, 384);
			}

			float signCorrect = bestV[0] != 0.f ? bestV[0] / abs(bestV[0]) : 1.f;

			mResult.skewAngle = signCorrect * acos(biggestDotY) * 180 / CV_PI;
			cv::LineIterator lIt(threshImg, Point(l[0], l[1]), Point(l[2], l[3]));
			float lineAvgCol = 0.f;
			for (int i = 0; i < lIt.count; i++)
			{
				lineAvgCol += (float)((*lIt)[0]);
			}
			lineAvgCol /= (float)lIt.count;
			Scalar fftImageAvgCol = cv::mean(threshImg);
			mResult.confidence = fabs(lineAvgCol - fftImageAvgCol[0]) / 255.f;
			mResult.confidence = pow(mResult.confidence, 0.5f);
		}
	}
	else
	{
		N = threshImg.cols;
		float cx2, cy2;
		cv::Mat varImg = threshImg.clone();
		for (int i = 0; i < N; i++)
		{
			cx2 = i - 512.f;
			cx2 *= cx2;
			for (int j = 0; j < N; j++)
			{
				cy2 = j - 512.f;
				cy2 *= cy2;
				if (sqrt(cx2 + cy2) > 512.f)
					varImg.at<uchar>(i, j) = 0;
			}
		}
		mResult = VarianceSolution(varImg, filename);
	}		
	sprintf(mResult.name, "FFT Skew Detector (%s)", useProjectionProfiling ? "Proj.Profiling" : "HoughLines");
	PrintStats();
	return mResult;
}
void FFT_SkewDetector::ComputeVisuals(std::vector<Vec2f> &pPts, char *winName, Vec2i winpos, Vec2i winsize)
{
	cv::Mat frame(N, N, CV_8UC3, Scalar(0, 0, 0));
	std::vector<Vec2f> pts = pPts;
	std::vector<Vec2f> blue, orange, grey;
	float tmp;

	for (int i = 0; i < pts.size(); i++)
	{
		pts[i] = Vec2f(pts[i][1], pts[i][0]);
		if (pts[i][0] > lowerLimit && pts[i][0] < upperLimit)
			if (pts[i][1] > lowerLimit && pts[i][1] < upperLimit)
				grey.push_back(pts[i]);
			else
				blue.push_back(pts[i]);
		else
			if (pts[i][1] > lowerLimit && pts[i][1] < upperLimit)
				orange.push_back(pts[i]);
	}


	DrawPoints(frame, orange, Vec3b(0, 128, 255), 0);
	DrawPoints(frame, blue, Vec3b(255, 64, 0), 0);
	DrawPoints(frame, grey, Vec3b(0, 128, 192) * 2 / 3 + Vec3b(192, 64, 0) * 2 / 3, 0);

	if (m_pComputeVisuals)
	{
		showImg(frame, winName);
		cv::resizeWindow(winName, winsize[0], winsize[1]);
		cv::moveWindow(winName, winpos[0], winpos[1]);
	}
}
//...............................................................................................................

DeskewAlgorithmRetType FFT_SkewDetector::VarianceSolution(cv::Mat &src, const char *fname)
{
	
	char winName[128];


	vScanlineAccumPoints = new unsigned int[histoSize];
	hScanlineAccumPoints = new unsigned int[histoSize];

	for (int i = 0; i < histoSize; i++)
	{
		vScanlineAccumPoints[i] = hScanlineAccumPoints[i]= 0.f;
	}
	// Showing the color image
	downscaleFact2 = (float)WINDOW_HEIGHT / (float)src.rows;
	displayHeight = (float)src.rows * downscaleFact2;
	displayWidth = (float)src.cols * downscaleFact2;

	cv::Mat cln = src.clone();

	std::vector<std::vector<Point> > contours;
	std::vector<Vec4i> hierarchy;

	for (int i = 0; i < src.rows; i++)
		for (int j = 0; j < src.cols; j++)
			if(src.at<uchar>(i,j))
				pointsV.push_back(Vec2f(i, j));

	rPointsV = pointsV;

	cv::Mat frr(src.rows, src.cols, CV_8UC3, Scalar(255, 255, 255));
	//cvtColor(squareFrame, frr, cv	::COLOR_GRAY2BGR);//
	DrawPoints(frr, rPointsV, Vec3b(128, 0, 0), 4);
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
		RowVarianceUpdate(0);
		cv::Mat histogramV(histoSize, 50, CV_8UC3, Scalar(255, 255, 255));
		cv::Mat histogramH(50, histoSize, CV_8UC3, Scalar(255, 255, 255));
		ComputeHistoForDisplay(histogramV, histogramH);

		if (m_pComputeVisuals)
		{
			sprintf_s(winName, "V FFTHistogramOriginalT: %s", filename);
			showImg(histogramV, winName);
			resizeWindow(winName, 100, 512);
			cv::moveWindow(winName, 512, 0);

			sprintf_s(winName, "H FFTHistogramOriginalT: %s", filename);
			showImg(histogramH, winName);
			resizeWindow(winName, 512, 100);
			cv::moveWindow(winName, 0, 512);
		}
	}

	float bestT = -15, maxVariance = -FLT_MAX, variance;
	float avgVariance = 0.f;
	int iterations = 0;
	float metaHisto[300];

	for (float t = -15; t <= 15.f; t += .1f)
	{
		variance = RowVarianceUpdate(t * ((float)CV_PI / 180.0f), false);
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
	variance = RowVarianceUpdate(bestT * ((float)CV_PI / 180.0f));
	DeskewAlgorithmRetType varianceResult;
	sprintf(varianceResult.name, "Frequency Domain PrjProfile");
	varianceResult.confidence = pow(1.f - metaStandardDeviation / (maxVariance - avgVariance), 0.25f);

	{
		cv::Mat histogramV(histoSize, 50, CV_8UC3, Scalar(255, 255, 255));
		cv::Mat histogramH(50, histoSize, CV_8UC3, Scalar(255, 255, 255));
		ComputeHistoForDisplay(histogramV, histogramH);

		if (m_pComputeVisuals)
		{
			memset(winName, 0, 128);
			sprintf_s(winName, "V FFTHistogram Deskewed: %s", filename);
			showImg(histogramV, winName);
			resizeWindow(winName, 100, 512);
			cv::moveWindow(winName, 1224, 0);

			memset(winName, 0, 128);
			sprintf_s(winName, "H FFTHistogram Deskewed: %s", filename);
			showImg(histogramH, winName);
			resizeWindow(winName, 512, 100);
			cv::moveWindow(winName, 712, 512);
		}
	}
	//sprintf_s(winName, "Rotated Img - Connected Components %s", filename2);
	//createTrackbar("SKEWangle", winName, &SKEWangle_slider2, SKEWangle_slider_max2, RowVariance_on_trackbar2);
	varianceResult.skewAngle = -bestT;
	//PrintStats();

	//char winName[128];

	{
	
		char winName[128];
		memset(winName, 0, 128);
		sprintf_s(winName, "SkeweD FFT %s", filename);
		ComputeVisuals(pointsV, winName, {0, 0}, {512, 512});
	}

	return varianceResult;
}

#define WINDOW_HEIGHT 630
using namespace cv;


float FFT_SkewDetector::ScanlineAccumulateGetVariance(unsigned int *histo, std::vector<Vec2f> &points, int n, float &maxVal, float &meanVal, int horizontal)
{
	memset(histo, 0, histoSize * sizeof(unsigned int));
	meanVal = 0.f;
	float hpIdx;
	float halfN = n * 0.5f;
	float targethpIdx;
	
	for (size_t i = 0; i < points.size(); i++)
	{
		hpIdx = (float)(points[i][horizontal] * histoSize) / (float)n;
		
		targethpIdx = max(min(upperLimit, hpIdx), lowerLimit);
		if (hpIdx == targethpIdx)
			histo[(int)targethpIdx]++;
		meanVal += 1.f;
	}

	meanVal /= 1024.f * (upperLimit - lowerLimit);

	float variance = 0.f;
	float err;
	maxVal = 0.f;
	float fdst;
	for (size_t i = lowerLimit; i <= upperLimit; i++)
	{
		if (histo[i] <= 0)
			continue;

		fdst = (float)histo[i] / 1024.f;
		err = fdst - meanVal;
		err *= err;
		variance += err;
		if (fdst > maxVal)
			maxVal = fdst;
	}
	return variance / (upperLimit - lowerLimit);
}

float FFT_SkewDetector::RowVarianceUpdate(float skewAngle, bool computeVisuals)
{
	std::chrono::high_resolution_clock::time_point tstart = std::chrono::high_resolution_clock::now();

	int hf = N / 2;
	RotatePoints(pointsV, rPointsV, skewAngle, Vec2f(hf, hf));
	pointsVarianceV = ScanlineAccumulateGetVariance(vScanlineAccumPoints, rPointsV, N, maxPointsAccumV, pointsMeanV, 0);
	pointsVarianceH = ScanlineAccumulateGetVariance(hScanlineAccumPoints, rPointsV, N, maxPointsAccumH, pointsMeanH, 1);

	if (computeVisuals && m_pComputeVisuals)
	{
		char winName[128];
		memset(winName, 0, 128);
		sprintf_s(winName, "FFT Corrected Skew %s", filename);
		ComputeVisuals(rPointsV, winName, { 712, 0 }, { 512, 512 });
		//cvMoveWindow("Rotated Img - Connected Components", displayWidth, 0);
	}
	std::chrono::high_resolution_clock::time_point tend = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(tend - tstart).count();

	//std:: cout << "DURATION :" <<duration << "\n";
	return pointsVarianceV + pointsVarianceH;// +leftVariance + rightVariance;	
}

void FFT_SkewDetector::ComputeHistoForDisplay(cv::Mat &histogramV, cv::Mat &histogramH)
{
	float pointsCount;
	/*DrawPoints(frame, orange, Vec3b(0, 128, 255), 0);
	DrawPoints(frame, blue, Vec3b(255, 64, 0), 0);*/
	for (int xx = 0; xx < histoSize; xx++)
	{
		pointsCount = ((float)vScanlineAccumPoints[xx] / 1024.f) / maxPointsAccumV * 49;

		for (int j = 0; j < pointsCount; j++)
			histogramV.at<Vec3b>(xx, j) = Vec3b(0, 128, 255);

		pointsCount = ((float)hScanlineAccumPoints[xx] / 1024.f) / maxPointsAccumH * 49;


		for (int j = 0; j < pointsCount; j++)
			histogramH.at<Vec3b>(j, xx) = Vec3b(255, 64, 0);
	}
}

#define clamp(x, a, b) max(a,min(b,x)) 
