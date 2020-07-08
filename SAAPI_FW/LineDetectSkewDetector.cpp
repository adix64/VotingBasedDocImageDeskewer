#include "LineDetectSkewDetector.h"
#include "DisjointSets.hpp"
#include <unordered_map>
#define WINDOW_HEIGHT 630
#define HOUGH_DESKEW_ANGLE_STEP (CV_PI / 720.f)
using namespace cv;
Scalar getRandomColor()
{
	float r = (float)rand() / RAND_MAX * 255;
	float g = (float)rand() / RAND_MAX * 255;
	float b = (float)rand() / RAND_MAX * 255;
	return Scalar(b, g, r);
}
DeskewAlgorithmRetType HoughSkewDetector::Run(Mat &src, const char *fname, bool p_computeVisuals)
{
	m_pComputeVisuals = p_computeVisuals;
	sprintf(mResult.name, "Hough Lines in Spatial Domain");

	strcpy_s(filename, fname);
	cv::Mat inputImage = src;
	
	char winName[128];
	if(m_pComputeVisuals)
	{
		memset(winName, 0, 128);
		sprintf_s(winName, "Original %s", filename);
		showImg(inputImage, winName);
		cv::resizeWindow(winName, 800 * inputImage.cols / inputImage.rows, 800);
		cv::moveWindow(winName, 0, 0);
	}

	std::vector<Vec4i> lines;
	

	//cv::resize(inputImage, inputImage, cvSize(inputImage.cols * 1024 / (float)inputImage.rows, 1024));
	cv::resize(inputImage, inputImage, Size(inputImage.cols * 2048 / (float)inputImage.rows, 2048));
	for (int i = 0; i < inputImage.rows; i++)
		for (int j = 0; j < inputImage.cols; j++)
			inputImage.at<uchar>(i, j) = 255 - inputImage.at<uchar>(i, j);

	cv::threshold(inputImage, inputImage, 0, 255, THRESH_BINARY | THRESH_OTSU);

	Sobel(inputImage, inputImage, CV_8U, 1, 0);

	int houghMinLineLen = inputImage.cols / 7;
	int houghMaxLineGap = inputImage.cols / 100	;
	int houghThreshold = houghMinLineLen;

	HoughLinesP(inputImage, lines, 1, HOUGH_DESKEW_ANGLE_STEP, houghThreshold, houghMinLineLen, houghMaxLineGap);
	//printf("\nNum. lines HoughLinesP returned:%d\t", lines.size());

///////////////////////////////////////
	std::vector<Vec4i> originalLines(lines.begin(), lines.end());
	std::vector<bool>relevantLines(lines.size(), false);
	for (int i = 0; i < lines.size(); i++)
	{
		Vec4f l = lines[i];
		Vec2f v = Vec2f(l[2], l[3]) - Vec2f(l[0], l[1]);

		if (l[1] > l[3])
			v = -v;//v points upwards
		float pmag = sqrt(v[0] * v[0] + v[1] * v[1]);
		v /= pmag;//normalize v

		if (fabs(v[0]) > cos(CV_PI /12.f) || fabs(v[0]) < cos(CV_PI /2.f - CV_PI /12.f))
		{
			relevantLines[i] = true;
		}
	}
	lines.clear();
	for (int i = 0; i < originalLines.size(); i++)
		if (relevantLines[i])
			lines.emplace_back(originalLines[i]);
	//printf("Number of lines of interest %d\n", lines.size());
	////////////////////////////////////////////////////////////////////////
	Mat ccdst = inputImage.clone();

	ccdst.convertTo(ccdst, CV_8UC3);
	cv::cvtColor(ccdst, ccdst, cv::COLOR_GRAY2BGR);//
	

	

	Vec2f v,v1,v2;
	Vec4f l, l1,l2;

	DisjointSet ds(lines.size());
	std::unordered_map<int, std::vector<int>> hsh;
	
	bool **parallelLines = new bool*[lines.size()];
	float **linePairsDots = new float*[lines.size()];
	for (int i = 0; i < lines.size(); i++)
	{
		parallelLines[i] = new bool[lines.size()];
		linePairsDots[i] = new float[lines.size()];
	}
	float dotProd;
	for (size_t i = 0; i < lines.size(); i++)
	{
		l1 = lines[i];
		v1 = Vec2f(l1[2], l1[3]) - Vec2f(l1[0], l1[1]);

		v1 /= sqrt(v1[0] * v1[0] + v1[1] * v1[1]);//normalize v
		for (size_t j = 0; j < lines.size(); j++)
		{
			if (i == j)
				continue;
			l2 = lines[j];
			v2 = Vec2f(l2[2], l2[3]) - Vec2f(l2[0], l2[1]);
			v2 /= sqrt(v2[0] * v2[0] + v2[1] * v2[1]);//normalize v

			dotProd = (v1[0] * v2[0] + v1[1] * v2[1]);

#define ANGLETHRSH 0.999f
			if (dotProd < -ANGLETHRSH || dotProd > ANGLETHRSH)
			{
				parallelLines[i][j] = parallelLines[j][i] = true;
			}
			else
			{
				parallelLines[i][j] = parallelLines[j][i] = false;
			}
			linePairsDots[i][j] = linePairsDots[j][i] = fabs(dotProd);//????
		}
	}

	std::vector<int> iSet, jSet;
	iSet.reserve(lines.size());
	jSet.reserve(lines.size());

	bool canMerge;
	int setofK;
	int setofI, setofJ;
	for (size_t i = 0; i < lines.size(); i++)
	{
		for (size_t j = 0; j < lines.size(); j++)
		{
			if (i == j)
				continue;

			setofI = ds.Find(i + 1);
			setofJ = ds.Find(j + 1);
			if (setofI == setofJ || !parallelLines[i][j])
				continue;
			iSet.clear();
			jSet.clear();
			
			for (int k = 0; k < lines.size(); k++)
			{
				setofK = ds.Find(k + 1);
				if (setofK == setofI)
					iSet.emplace_back(k);
				else if (setofK == setofJ)
					jSet.emplace_back(k);
			}

			//canMerge = true;
			for (int ki = 0; ki < iSet.size(); ki++)
				for (int kj = 0; kj < jSet.size(); kj++)
					if (!parallelLines[iSet[ki]][jSet[kj]])
					{
						//canMerge = false;
						goto _cannotMerge;
					}
				
			//if(canMerge)
			ds.Union(i+1, j+1);
			_cannotMerge:;
		}	
	}
	for (int i = 0; i < lines.size(); i++)
		delete parallelLines[i];
	delete parallelLines;

	int setOfI;
	for (int i = 0; i < lines.size(); i++)
	{
		setOfI = ds.Find(i+1);
		auto found = hsh.find(setOfI);
		if (found != hsh.end())
		{
			hsh[setOfI].push_back(i);
		}
		else
		{
			hsh[setOfI] = std::vector<int>();
			hsh[setOfI].push_back(i);
		}
	}

	std::vector < float > setScores(hsh.size(), 0);
	Vec4f crtLine;
	float setScoresSum = 0.f;
	int kx = 0;
	for (auto it = hsh.begin(); it != hsh.end(); it++)
	{
		std::vector<int> &lineIdxs = it->second;
		for (int i = 0; i < lineIdxs.size(); i++)
		{
			crtLine = lines[lineIdxs[i]];
			setScores[kx] += norm(Point2f(crtLine[0], crtLine[1]) - Point2f(crtLine[2], crtLine[3]));
		}
		setScores[kx] *= lineIdxs.size();
		setScoresSum += setScores[kx];
		kx++;
	}

	unsigned int mostParallels = 0;
	int winnerSetID, winnerSetID2;
	for (auto it = hsh.begin(); it != hsh.end(); it++)
	{
		if (mostParallels < it->second.size())
		{
			mostParallels = it->second.size();
			winnerSetID = it->first;
		}
	}
	mostParallels = 0;
	for (auto it = hsh.begin(); it != hsh.end(); it++)
	{
		if (it->first == winnerSetID)//take second winner
			continue;
		if (mostParallels < it->second.size())
		{
			mostParallels = it->second.size();
			winnerSetID2 = it->first;
		}
	}

	

	std::vector<Vec4f> classRepresentatives;
	std::vector<int> classReprHashes;
	Vec4f helperLine;
	for (auto it = hsh.begin(); it != hsh.end(); it++)
	{
		Vec4f repr = Vec4f(0, 0, 0, 0);
		for (int i = 0; i < it->second.size(); i++)
		{
			helperLine = lines[it->second[i]];
			if(helperLine[0] > helperLine[2])
				repr += Vec4f(helperLine[2], helperLine[3], helperLine[0], helperLine[1]);
			else repr += helperLine;
		}
		repr = repr / (float)it->second.size();
		float r = (float)rand() / RAND_MAX * 255;
		float g = (float)rand() / RAND_MAX * 255;
		float b = (float)rand() / RAND_MAX * 255;

		classReprHashes.push_back(it->first);
		classRepresentatives.push_back(repr);
	}
	
	for (auto it = hsh.begin(); it != hsh.end(); it++)
	{
		std::vector<int> &winnerSet = hsh[it->first];
		float r, g, b;
		if (it->first == winnerSetID2)
		{
			r = 0;
			g = 64;
			b = 255;
		}
		else if(it->first == winnerSetID)
		{
			r = b = 0;
			g = 255;
		}
		else
		{
			r = (float)rand() / RAND_MAX * 128 + 64;
			g = (float)rand() / RAND_MAX * 128 + 64;
			b = (float)rand() / RAND_MAX * 128 + 32;
		}
		for (int i = 0; i < winnerSet.size(); i++)
		{
			Vec4f l = lines[winnerSet[i]];
			cv::line(ccdst, Point2f(l[0], l[1]), Point2f(l[2], l[3]), Scalar(b,g,r), 3, cv::LineTypes::LINE_AA);
		}
	}
	//{
	//	std::vector<int> &winnerSet = hsh[winnerSetID];
	//	for (int i = 0; i < winnerSet.size(); i++)
	//	{
	//		Vec4i l = lines[winnerSet[i]];
	//		cv::line(ccdst, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(64, 0, 255), 3, cv::LineTypes::LINE_AA);
	//	}
	//}
	//{
	//	std::vector<int> &winnerSet = hsh[winnerSetID2];
	//	for (int i = 0; i < winnerSet.size(); i++)
	//	{
	//		Vec4i l = lines[winnerSet[i]];
	//		cv::line(ccdst, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(255, 0, 64), 3, cv::LineTypes::LINE_AA);
	//	}
	//}
	if (classRepresentatives.size() > 1)
	{//the case of at least two perpendicular lines
		//printf("Num class representatives: %d\n", classRepresentatives.size());

		int bestV, bestH;
		float smallestDot = FLT_MAX;
		//find the two most perpendicular lines
		for (int i = 0; i < classRepresentatives.size(); i++)
		{
			for (int j = 0; j < classRepresentatives.size(); j++)
			{
				if (i == j) continue;
				l1 = classRepresentatives[i];
				v1 = Vec2f(l1[2], l1[3]) - Vec2f(l1[0], l1[1]);

				if (l1[1] > l1[3])
					v1 = -v1;//v points upwards

				v1 /= sqrt(v1[0] * v1[0] + v1[1] * v1[1]);;//normalize v

				l2 = classRepresentatives[j];
				v2 = Vec2f(l2[2], l2[3]) - Vec2f(l2[0], l2[1]);

				if (l2[1] > l2[3])
					v2 = -v2;//v points upwards

				v2 /= sqrt(v2[0] * v2[0] + v2[1] * v2[1]);//normalize v
				float absDot = fabs(v1[0] * v2[0] + v1[1] * v2[1]);
				if (absDot < smallestDot)
				{
					smallestDot = absDot;
					if (fabs(v1[0]) > fabs(v1[1]))
					{
						bestH = i;
						bestV = j;
					}
					else
					{
						bestV = i;
						bestH = j;
					}
				}
			}
		}
		
		Vec4f reprV = classRepresentatives[bestV];
		Vec4f reprH = classRepresentatives[bestH];
		
		Vec2f vertWinner, horWinner;
		
		vertWinner = Vec2f(reprV[2], reprV[3]) - Vec2f(reprV[0], reprV[1]);
		if (reprV[1] > reprV[3])
			vertWinner = -vertWinner;//v points upwards
		float pmag = sqrt(vertWinner[0] * vertWinner[0] + vertWinner[1] * vertWinner[1]);
		vertWinner /= pmag;//normalize v
		float signCorrect = vertWinner[0] != 0.f ? vertWinner[0] / abs(vertWinner[0]) : 1.f;
		float vSkewAngle  = signCorrect * acos(vertWinner[1]) * 180 / CV_PI;

		horWinner = Vec2f(reprH[2], reprH[3]) - Vec2f(reprH[0], reprH[1]);
		if (reprH[0] > reprH[2])
			horWinner = -horWinner;//v points right
		pmag = sqrt(horWinner[0] * horWinner[0] + horWinner[1] * horWinner[1]);
		horWinner /= pmag;//normalize v
		signCorrect = horWinner[1] != 0.f ? -horWinner[1] / abs(horWinner[1]) : 1.f;
		float hSkewAngle = signCorrect * acos(horWinner[0]) * 180 / CV_PI;
		
		mResult.skewAngle = (vSkewAngle * setScores[bestV] + hSkewAngle * setScores[bestH]) / (setScores[bestV] + setScores[bestH]);
		mResult.confidence = 1.f - abs(horWinner[0] * vertWinner[0] + horWinner[1] * vertWinner[1]);
		mResult.confidence = asin(mResult.confidence) / (CV_PI * 0.5f);
		mResult.confidence *= (setScores[bestV] + setScores[bestH]) / setScoresSum;
		
		std::vector<int> &vWinnerSet = hsh[classReprHashes[bestV]];
		
		if (vWinnerSet.size() > 1)
		{
			int count = 0;
			float dotSum = 0.f;
			for (int i = 0; i < vWinnerSet.size(); i++)
			{
				for (int j = 0; j < i; j++)
				{
					dotSum += linePairsDots[vWinnerSet[i]][vWinnerSet[j]];
					count++;
				}
			}
			dotSum /= (float)count;

			mResult.confidence *= dotSum;
		}
		DrawInfLineDocCenter(ccdst, reprV, Scalar(0, 0, 255), ccdst.cols / 35);
		DrawInfLineDocCenter(ccdst, reprH, Scalar(0, 0, 255), ccdst.cols / 35);
		DrawInfLineDocCenter(ccdst, reprV, Scalar(255, 255, 255), ccdst.cols / 100);
		DrawInfLineDocCenter(ccdst, reprH, Scalar(255, 255, 255), ccdst.cols / 100);
		
	}
	else if (classRepresentatives.size())
	{
		//return mResult;
		//printf("Num class representatives: %d\n", classRepresentatives.size());

		float signCorrect;
		Vec4f winner = classRepresentatives[0];
		Vec2f winnerDir = Vec2f(winner[2], winner[3]) - Vec2f(winner[0], winner[1]);
		if (abs(winnerDir[0]) > abs(winnerDir[1]))
		{//horizontal line
			if (winner[0] > winner[2])
				winnerDir = -winnerDir;//v points right
			winnerDir /= sqrt(winnerDir[0] * winnerDir[0] + winnerDir[1] * winnerDir[1]);//normalize v
			signCorrect = winnerDir[1] != 0.f ? -winnerDir[1] / abs(winnerDir[1]) : 1.f;
			mResult.skewAngle = signCorrect * acos(winnerDir[0]) * 180 / CV_PI;
			mResult.confidence = setScores[0] / setScoresSum;
		}
		else
		{//vertical line
			if (winner[1] > winner[3])
				winnerDir = -winnerDir;//v points upwards
			winnerDir /= sqrt(winnerDir[0] * winnerDir[0] + winnerDir[1] * winnerDir[1]);
			float signCorrect = winnerDir[0] != 0.f ? winnerDir[0] / abs(winnerDir[0]) : 1.f;
			mResult.skewAngle = signCorrect * acos(winnerDir[1]) * 180 / CV_PI;
			mResult.confidence = setScores[0] / setScoresSum;
		}

		std::vector<int> &vWinnerSet = hsh[classReprHashes[0]];

		if (vWinnerSet.size() > 1)
		{
			int count = 0;
			float dotSum = 0.f;
			for (int i = 0; i < vWinnerSet.size(); i++)
			{ 
				for (int j = 0; j < i; j++)
				{
					dotSum += linePairsDots[vWinnerSet[i]][vWinnerSet[j]];
					count++;
				}
			}
			dotSum /= (float)count;

			mResult.confidence *= dotSum;
		}
		DrawInfLineDocCenter(ccdst, winner, Scalar(0, 0, 255), ccdst.cols / 35);
		DrawInfLineDocCenter(ccdst, winner, Scalar(255, 255, 255), ccdst.cols / 100);
	}
	else
	{
		printf("\n___________ERROR! No class representatives!___________\n");
	}
	
	if (m_pComputeVisuals)
	{
		memset(winName, 0, 128);
		sprintf_s(winName, "Line Hough Lines %s", filename);
		showImg(ccdst, winName);
		cv::resizeWindow(winName, 800 * inputImage.cols / inputImage.rows, 800);
		cv::moveWindow(winName, 800 * inputImage.cols / inputImage.rows, 0);
	}
	mResult.confidence = pow(mResult.confidence, 1.21f);
	PrintStats();
	for (int i = 0; i < lines.size(); i++)
		delete linePairsDots[i];
	delete linePairsDots;
	return mResult;
}
void HoughSkewDetector::DrawInfLineDocCenter(Mat &dst, Vec4f segment, cv::Scalar color, int thickness)
{
	Point2f start, end, midPoint;
	Point2f pageCenter = Point2f(dst.cols * 0.5f, dst.rows * 0.5f);
	//COMPUTE AND DRAW ORIENTATION AXES CENTERED ON PAGE
	start = Point2f(segment[0], segment[1]);
	end = Point2f(segment[2], segment[3]);
	midPoint = (start + end) / 2.f;
	start -= midPoint; end -= midPoint;
	start *= 1000.f; end *= 1000.f;
	start += pageCenter; end += pageCenter;
	cv::line(dst, start, end,
		color, thickness, cv::LineTypes::LINE_AA);
}