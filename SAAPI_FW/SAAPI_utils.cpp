
#include "SAAPI_utils.h"

using namespace cv;

void showImg(cv::Mat &image, char *name, float scale, bool createWindow)
{
	if(createWindow)
		cv::namedWindow(name, cv::WINDOW_FREERATIO);// Create a window for display.
	cv::imshow(name, image);                   // Show our image inside it.
	int width = image.cols, height = image.rows;
	int resWidth = (int)((float)width * scale);
	int resHeight = (int)((float)height* scale);
	cv::resizeWindow(name, resWidth, resHeight);
	                                       // Wait for a keystroke in the window
}
#ifdef IPLIMAGEIPL

IplImage* drawImageInSquare(IplImage* src)
{
	// Make a spare image for the result
	CvSize sizeRotated;
	int w = src->width;
	int h = src->height;

	int diagonal = sqrt(w * w + h * h) + 8;
	sizeRotated.width = sizeRotated.height = diagonal;

	// Rotate
	IplImage *imgsq = cvCreateImage(sizeRotated,
		src->depth, src->nChannels);

	memset(imgsq->imageData, 255, diagonal * diagonal);

	int woffset = (diagonal - w) / 2;
	int hoffset = (diagonal - h) / 2;
	for (int i = 0; i < h; i++)
	{
		for (int j = 0; j < w; j++)
		{
			imgsq->imageData[(i + hoffset) * imgsq->widthStep + (j + woffset)] = src->imageData[i * src->widthStep + j];
		}
	}
	return imgsq;
}
#endif


cv::Mat drawImageInSquare(cv::Mat &src)
{
	// Make a spare image for the result
	Size sizeRotated;
	int w = src.cols;
	int h = src.rows;

	int diagonal = sqrt(w * w + h * h);
	sizeRotated.width = sizeRotated.height = diagonal;

	cv::Mat imgsq(cv::Size(diagonal, diagonal), CV_8U, Scalar(255));

	int woffset = (diagonal - w) / 2;
	int hoffset = (diagonal - h) / 2;
	for (int i = 0; i < h; i++)
	{
		for (int j = 0; j < w; j++)
		{
			imgsq.at<uchar>((i + hoffset), (j + woffset)) = src.at<uchar>(i, j);
		}
	}
	return imgsq;
}

#ifdef IPLIMAGEIPL

// Rotate the image clockwise (or counter-clockwise if negative).
// Remember to free the returned image.
IplImage *rotateImage(const IplImage *src, float angleDegrees)
{
	// Create a map_matrix, where the left 2x2 matrix
	// is the transform and the right 2x1 is the dimensions.
	float m[6];
	CvMat M = cvMat(2, 3, CV_32F, m);
	int w = src->width;
	int h = src->height;
	float angleRadians = angleDegrees * ((float)CV_PI / 180.0f);
	m[0] = (float)(cos(angleRadians));
	m[1] = (float)(sin(angleRadians));
	m[3] = -m[1];
	m[4] = m[0];
	m[2] = w * 0.5f;
	m[5] = h * 0.5f;

	// Make a spare image for the result
	CvSize sizeRotated;
	sizeRotated.width = cvRound(w);
	sizeRotated.height = cvRound(h);

	// Rotate
	IplImage *imageRotated = cvCreateImage(sizeRotated,
		src->depth, src->nChannels);

	// Transform the image
	cvGetQuadrangleSubPix(src, imageRotated, &M);

	return imageRotated;
}
#endif
cv::Mat rotateImage(cv::Mat &source, double angle)
{
	//cv::Mat source = cv::cvarrToMat(iplsrc);

	Point2f src_center(source.cols / 2.0F, source.rows / 2.0F);
	Mat rot_mat = getRotationMatrix2D(src_center, angle, 1.0);
	Mat dst;

	warpAffine(source, dst, rot_mat, source.size(),
		cv::INTER_LINEAR,
		cv::BORDER_CONSTANT,
		cv::Scalar(255, 255, 255));
	return dst;
	//IplImage* ipldst = cvCloneImage(&(IplImage)dst);

	//return ipldst;
}


std::string type2str(int type) {
	std::string r;

	uchar depth = type & CV_MAT_DEPTH_MASK;
	uchar chans = 1 + (type >> CV_CN_SHIFT);

	switch (depth) {
	case CV_8U:  r = "8U"; break;
	case CV_8S:  r = "8S"; break;
	case CV_16U: r = "16U"; break;
	case CV_16S: r = "16S"; break;
	case CV_32S: r = "32S"; break;
	case CV_32F: r = "32F"; break;
	case CV_64F: r = "64F"; break;
	default:     r = "User"; break;
	}

	r += "C";
	r += (chans + '0');

	return r;
}



void RotatePoints(std::vector<Vec2f> &points, std::vector<Vec2f> &dst,
	float angle, Vec2f rotPivot)
{
	float c = cos(angle), s = sin(angle);
	float px, py;

	for (int i = 0; i < points.size(); i++)
	{	//move point to origin
		px = points[i][0] - rotPivot[0];
		py = points[i][1] - rotPivot[1];
		//rotate and translate back in place
		dst[i][0] = px * c - py * s + rotPivot[0];
		dst[i][1] = px * s + py * c + rotPivot[1];
	}
}

void DrawPoints(Mat &frame, std::vector<Vec2f> &points, Vec3b &color, int radius)
{

	int x, y;
	for (int k = 0; k < points.size(); k++)
	{
		x = points[k][0];
		y = points[k][1];
		x = min(x, frame.cols - 4);
		y = min(y, frame.rows - 4);
		x = max(x, 4);
		y = max(y, 4);
		//circle(frame, Point(x, y), radius, color,-1);
		rectangle(frame, Rect(x - radius, y - radius, 2 * radius + 1, 2 * radius + 1), color, -1);
	}
}