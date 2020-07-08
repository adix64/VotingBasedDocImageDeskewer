#include "SAAPI_utils.h"
#include "RowVariance_SkewDetector.h"
#include "FFT_SkewDetector.h"
#include "LineDetectSkewDetector.h"
#include "dirent.h"
#include <chrono>
#include <QtWidgets/qapplication.h>
#include <QtWidgets/qpushbutton.h>
#include <QtWidgets/qradiobutton.h>
#include <QtWidgets/qcheckbox.h>
#include <QtWidgets/qaction.h>
#include <QtWidgets/qfiledialog.h>
#include <QtCore/qcoreapplication.h>
#include <QtCore/qobject.h>
#include <QtWidgets/qlabel.h>
//#define TEST_MODE

using namespace std;
using namespace cv;
#define FFT_SKEW_DETECTION 1
#define ROW_ALIGN_VARIANCE_SKEW_DETECTION 2
#define HOUGH_LINES 3
#define BEST_FIRST_VOTING 4
#define WEIGHTED_VOTING 5
#define UNANIMOUS_VOTING 6


float nFiles = 0.f;
#ifdef TEST_MODE
float fftAvgErr = 0.f, fftNsuccessful = 0.f;
float fftAvgConf = 0.f;
float fftAvgTime = 0.f;

float varianceAvgErr = 0.f, varianceNsuccessful = 0.f;
float varianceAvgConf = 0.f;
float varianceAvgTime = 0.f;

float lineDetectAvgErr = 0.f, lineDetectNsuccessful = 0.f;
float lineDetectAvgConf = 0.f;
float lineDetectAvgTime = 0.f;

float bestFirstAvgErr = 0.f;
float bestFirstAvgConf = 0.f;
float bestFirstAvgTime = 0.f;

float weightVotingAvgErr = 0.f;
float weightVotingAvgConf = 0.f;
float weightVotingAvgTime = 0.f;

float unanimVotingAvgErr = 0.f;
float unanimVotingAvgConf = 0.f;
float unanimVotingAvgTime = 0.f;
#endif

bool gComputeVisuals[3] = { true,true,true };
bool gFFTprojProfiling = true;
DeskewAlgorithmRetType Test_SkewDetectorFILE(const char* filename, float angle, int method = FFT_SKEW_DETECTION, const char *outputFilePath = NULL, const char *bareFname = NULL)
{
	cv::Mat inputImage = cv::imread(filename, cv::IMREAD_GRAYSCALE);
	cv::Mat rotatedImage = angle != 0.f ? rotateImage(inputImage, angle) : inputImage;
	std::cout << "____________________________________________________________________________________\n";
	printf("FILE: %s, %d x %d\n", filename, inputImage.cols, inputImage.rows);
#ifdef TEST_MODE
	printf("  >introduced skew angle: %f\n", angle);
#endif
	DeskewAlgorithmRetType deskewResult;
	DeskewAlgorithm *deskewAlgorithm = NULL;

	bool votingSolution = false;
	switch(method)
	{
	case FFT_SKEW_DETECTION:
		deskewAlgorithm = new FFT_SkewDetector(gFFTprojProfiling);
		break;
	case ROW_ALIGN_VARIANCE_SKEW_DETECTION:
		deskewAlgorithm = new RowVariance_SkewDetector();
		break;
	case HOUGH_LINES:
		deskewAlgorithm = new HoughSkewDetector();
		break;
	case WEIGHTED_VOTING: case UNANIMOUS_VOTING: case BEST_FIRST_VOTING:
		votingSolution = true;
		break;
	default:
		break;
	}
	std::chrono::high_resolution_clock::time_point t1, t2;

	if(votingSolution)
	{
		DeskewAlgorithm *algorithms[3];
		std::vector<DeskewAlgorithmRetType> candidates(3);
		algorithms[0] = new FFT_SkewDetector(gFFTprojProfiling);
		algorithms[1] = new RowVariance_SkewDetector();
		algorithms[2] = new HoughSkewDetector();
		float err;
#ifdef TEST_MODE
		float *avgConfs[3] = { &fftAvgConf, &varianceAvgConf, &lineDetectAvgConf};
		float *avgErrs[3] = { &fftAvgErr, &varianceAvgErr, &lineDetectAvgErr };
		float *avgTime[3] = { &fftAvgTime, &varianceAvgTime, &lineDetectAvgTime };
		float *nSuccesses[3] = { &fftNsuccessful, &varianceNsuccessful, &lineDetectNsuccessful };
#endif
		for (int i = 0; i < 3; i++)
		{
			t1 = std::chrono::high_resolution_clock::now();
			candidates[i] = algorithms[i]->Run(rotatedImage.clone(), bareFname ? bareFname : filename, gComputeVisuals[i]);//????
			t2 = std::chrono::high_resolution_clock::now();
			auto duration = std::chrono::duration_cast<std::chrono::milliseconds > (t2 - t1).count();
			
			if (candidates[i].confidence < 0.5f) {
				candidates[i].valid = false;
				printf("candidate %s was discarded for low confidence (%.0f\%%).\n", candidates[i].name, candidates[i].confidence * 100.f);
			}
#ifdef TEST_MODE
			else
			{
				err = fabs(candidates[i].skewAngle - angle);
				printf("\terror................... %.2f\370\n", err);
				*(avgErrs[i]) += err;
				*(nSuccesses[i]) += 1.f;
			}
			*(avgConfs[i]) += candidates[i].confidence;
			*(avgTime[i]) += duration;
#endif
			std::cout << "\ttime.................. " << duration << "ms\n";
		}
		
		std::sort(candidates.begin(), candidates.end(), 
			[](DeskewAlgorithmRetType &c1, DeskewAlgorithmRetType &c2) 
		{return c1.confidence > c2.confidence; });

		//BEST FIRST
		if (candidates[0].valid)
			deskewResult = candidates[0];
		else
			deskewResult = DeskewAlgorithmRetType();

		printf("BEST FIRST POLICY:\tangle: %.2f\370\tconfidence: %.0f\%%\n", deskewResult.skewAngle, deskewResult.confidence * 100.f);
#ifdef TEST_MODE
		err = fabs(deskewResult.skewAngle - angle);
		printf("\terror................... %.2f\370\n", err);
		bestFirstAvgConf += deskewResult.confidence;
		bestFirstAvgErr += err;
#endif
		float totalConfidence, weightedConfidence = 0.f;
		float resultAngle;
		int validCount = 0;
		//WEIGHTED VOTING
		totalConfidence = resultAngle = 0;
		for (int i = 0; i < 3; i++)
		{
			if (!candidates[i].valid)
				continue;
			totalConfidence += candidates[i].confidence;
			weightedConfidence += candidates[i].confidence * candidates[i].confidence;
			resultAngle += candidates[i].skewAngle * candidates[i].confidence;
			validCount++;
		}
		resultAngle /= totalConfidence;
		weightedConfidence /= totalConfidence;
		printf("WEIGHTED VOTING POLICY:\tangle: %.2f\370\tconfidence: %.0f\%%\n", resultAngle, weightedConfidence * 100.f);
		err = fabs(resultAngle - angle);
		printf("\terror................... %.2f\370\n", err);
#ifdef TEST_MODE
		weightVotingAvgConf += weightedConfidence;
		weightVotingAvgErr += err;
#endif
		//UNANIMITY VOTING
		totalConfidence = resultAngle = 0;
		for (int i = 0; i < 3; i++)
		{
			if (!candidates[i].valid)
				continue;
			totalConfidence += candidates[i].confidence;
			resultAngle += candidates[i].skewAngle;
		}
		resultAngle /= (float)validCount;
		totalConfidence /= 3.f;

		printf("UNANIMITY VOTING POLICY:\tangle: %.2f\370\tconfidence: %.0f\%%\n", resultAngle, totalConfidence * 100.f);
		err = fabs(resultAngle - angle);
#ifdef TEST_MODE
		unanimVotingAvgConf += totalConfidence;
		unanimVotingAvgErr += err;
		printf("\terror................... %.2f\370\n", err);
#endif

		for (int i = 0; i < 3; i++)delete algorithms[i];
	}
	else {
		deskewResult = deskewAlgorithm->Run(rotatedImage, bareFname ? bareFname : filename, gComputeVisuals[method - 1]);//????
#ifdef TEST_MODE
		printf("\terror................... %.2f\370\n", fabs(deskewResult.skewAngle - angle));
#endif
	}

	std::cout << "\n____________________________________________________________________________________\n";
	delete deskewAlgorithm;
	if (outputFilePath)
	{
		cv::Mat colorInputImage = cv::imread(filename, cv::IMREAD_COLOR);
		Mat deskewed = rotateImage(colorInputImage, -deskewResult.skewAngle);
		imwrite(outputFilePath, deskewed);

	}
	return deskewResult;
}

void Test_SkewDetectorFOLDER(const char *foldername, int method = FFT_SKEW_DETECTION, bool writeOutput = false)
{
	char num[8];
	DIR *dir;
	struct dirent *ent;

	char slashType = '\\';

	char *path = _strdup(foldername);
	int slashIdx = strlen(path) - 1;
	if (path[slashIdx] == '\\' || path[slashIdx] == '/')
	{
		path[slashIdx] = '\0';
		slashType = path[slashIdx];
		slashIdx--;
	}
	while (path[slashIdx] != '\\' && path[slashIdx] != '/' && slashIdx > 0)
		slashIdx--;
	if (slashIdx > 0)
	{
		slashType = path[slashIdx];
	}
	char *dirName = _strdup(&(path[slashIdx + 1]));
	path[slashIdx] = '\0';
	
	for (int i = 0; i < slashIdx; i++)
		if (path[i] == '/')
			path[i] = '\\';

	std::string outPath = std::string(path) + "\\deskewed_" + dirName;
	if (outPath[0] == '\\')
		outPath = std::string(outPath.begin() + 1, outPath.end());

	//check outDir exists, if not, create it
	if ((dir = opendir(outPath.c_str())) != NULL)
		closedir(dir);
	else {
		std::string command = std::string("mkdir ") + outPath;
		std::cout << "\n" << command << "\n";
		system(command.c_str());
	}

	if ((dir = opendir(foldername)) != NULL)
	{
		while ((ent = readdir(dir)) != NULL) {
			if (!strcmp(ent->d_name, "."))
				continue;
			if (!strcmp(ent->d_name, ".."))
				continue;
			if (!strcmp(ent->d_name, "ignore"))
				continue;
			float angle = (float)rand() / RAND_MAX * 2.f - 1.f;
			angle *= 15.f;
		/*	float angle = rand() % 30 - 15.f;
			angle += rand() % 3 * 0.25f;*/
			Test_SkewDetectorFILE((std::string(foldername) + slashType + ent->d_name).c_str(), 0, method, (outPath + slashType + ent->d_name).c_str());
			nFiles += 1.f;
		}
		closedir(dir);
	}
	
}
char *usageMessage = "\nCLI Usage: DocumentDeskewer.exe -f/-d <fileName/directoryName> "\
"[-method 1|2|3|4|5|6]\n\t1 : Dominant Lines in Frequency Domain\n"\
"\t2 : Projection Profiling Row Align\n\t3 : Lines in Spatial Domain\n"\
"\t4 : Best First Vote of 1,2,3\n\t5 : Weighted Vote of 1,2,3\n"\
"\t6 : Unanimity Vote of 1,2,3\n\n";

void badUsageExit()
{
	printf(usageMessage);
	exit(0);
}
#define DESKEWER_WINDOW_MODE


class DeskewerGUI : public QWidget
{
public:
	QWidget *mainWidget;
	QRadioButton *radioButtons[6];
	QRadioButton *fftRadioButtons[2];
	QCheckBox *visualsChckBoxs[3];
	

	DeskewerGUI(QWidget *parent = NULL)
	: QWidget(parent){
		
	}
	void openWithDialog()
	{

		QStringList files = QFileDialog::getOpenFileNames(mainWidget, "Open File for Deskew");
		for (auto it = files.begin(); it != files.end(); it++)
		{
			std::string __filename = (*it).toUtf8().constData();
			char *filename = _strdup(__filename.c_str());

			char *path = _strdup(filename);
			int slashIdx = strlen(path) - 1;

			while (path[slashIdx] != '\\' && path[slashIdx] != '/' && slashIdx > 0)
				slashIdx--;
			char *fname = filename;
			if (slashIdx > 0)
			{
				path[slashIdx + 1] = '\0';
				fname = filename + slashIdx + 1;
			}
			else path[0] = '\0';
			int method = 0;
			for (int i = 0; i < 6; i++)
				if (radioButtons[i]->isChecked())
					method = i;


			for (int i = 0; i < 3; i++)
				gComputeVisuals[i] = visualsChckBoxs[i]->isChecked();

			gFFTprojProfiling = fftRadioButtons[1]->isChecked();
			Test_SkewDetectorFILE(filename, 0.f, method + 1, (std::string(path) + "deskewed_" + fname).c_str(), fname);
		}
	}
	void Exec()
	{
		Qt::WindowFlags wf;
		
		mainWidget = new QWidget(NULL, wf);
		mainWidget->setObjectName(QString("Deskewer"));
		mainWidget->show();
		mainWidget->setFixedSize(288, 450);
		QPushButton *hello = new QPushButton("Deskew Files...", mainWidget);
		char *methodNames[6] = { "Frequency Domain", "Character Projection Profiling",
						"Lines in Spatial Domain", "Best First Vote", "Weighted Vote", "Unanimity Vote" };
		for (int i = 0; i < 3; i++)
		{
			radioButtons[i] = new QRadioButton(QString(methodNames[i]), mainWidget);
			radioButtons[i]->show();
			radioButtons[i]->move(QPoint(64, 64 * i + 30));
		}
		for (int i = 3; i < 6; i++)
		{
			radioButtons[i] = new QRadioButton(QString(methodNames[i]), mainWidget);
			radioButtons[i]->show();
			radioButtons[i]->move(QPoint(64, 32 * i + 180));
		}
		radioButtons[3]->setChecked(true);

		wchar_t eyeSymbol[] = { 0xD83D,0xDC41};
		for (int i = 0; i < 3; i++)
		{
			visualsChckBoxs[i] = new QCheckBox(QString(eyeSymbol[0])+ QString(eyeSymbol[1]), mainWidget);
			visualsChckBoxs[i]->show();
			visualsChckBoxs[i]->setChecked(true);
			visualsChckBoxs[i]->move(QPoint(16, 64 * i + 30));
		}

		Qt::WindowFlags wf2;
		wf2.setFlag(Qt::WindowType::Widget, true);
		QWidget *fftWidget = new QWidget(mainWidget,wf2);
		fftWidget->show();
		fftWidget->setFixedSize(200, 32);
		fftWidget->move(96, 56);

		fftRadioButtons[0] = new QRadioButton(QString("HoughLines"), fftWidget);
		fftRadioButtons[0]->show();
		fftRadioButtons[0]->move(QPoint(0, 0));

		fftRadioButtons[1] = new QRadioButton(QString("PrjProfiling"), fftWidget);
		fftRadioButtons[1]->show();
		fftRadioButtons[1]->move(QPoint(100, 0));

		fftRadioButtons[1]->setChecked(true); // PRJ PROFILING

		hello->resize(128, 30);
		hello->move(QPoint(80, 400));
		QObject::connect(hello, &QPushButton::released, this, &DeskewerGUI::openWithDialog);

		QLabel *ql = new QLabel(mainWidget);
		ql->setText("\tDeskew by Algorithm\n"
			"_____________________________________________________"
			"\n\t\t    Voting-Based Deskew");
		ql->move(0, 200);
		ql->show();
		ql->setDisabled(true);
		hello->show();
		
	}
};

int main(int argc, char *argv[])
{
	printf(usageMessage);
#ifndef TEST_MODE
	srand(time(NULL));
	if (argc > 1)
	{

		if (argc != 3 && argc != 5)
			badUsageExit();

		bool deskewFileMode = !strcmp(argv[1], "-f");
		bool deskewDirMode = !strcmp(argv[1], "-d");

		if (!deskewFileMode && !deskewDirMode)
			badUsageExit();

		int mode = BEST_FIRST_VOTING;
		if (argc == 5) {
			if (strcmp(argv[3], "-method"))
				badUsageExit();
			else
				mode = atoi(argv[4]);
		}

		if (mode < 1 || mode > 6)
			badUsageExit();


		if (deskewFileMode)
		{
			char *path = _strdup(argv[2]);
			int slashIdx = strlen(path) - 1;

			while (path[slashIdx] != '\\' && path[slashIdx] != '/' && slashIdx > 0)
				slashIdx--;
			char *fname = argv[2];
			if (slashIdx > 0)
			{
				path[slashIdx + 1] = '\0';
				fname = argv[2] + slashIdx + 1;
			}
			else path[0] = '\0';

			Test_SkewDetectorFILE(argv[2], 0.f, mode, (std::string(path) + "deskewed_" + fname).c_str());
		}
		else
			Test_SkewDetectorFOLDER(argv[2], mode, true);
	}
	else {
		QApplication *app = new QApplication(argc, argv);
		DeskewerGUI gui;
		gui.Exec();
		return app->exec();
	}
#else
	//TessTest("../../../4.png");//5 .. / skew_images / book2 / 00015.tif");
	Test_SkewDetectorFOLDER("../skew_images/FFTest/", WEIGHTED_VOTING);
	//Test_SkewDetectorFOLDER("../skew_images/arabic/", HOUGH_LINES);
	//Test_SkewDetectorFOLDER("../skew_images/LW/", HOUGH_LINES);


	fftAvgConf /= nFiles;
	fftAvgErr /= fftNsuccessful;
	fftAvgTime /= nFiles;
	lineDetectAvgConf /= nFiles;
	lineDetectAvgErr /= lineDetectNsuccessful;
	lineDetectAvgTime /= nFiles;
	varianceAvgConf /= nFiles;
	varianceAvgErr /= varianceNsuccessful;
	varianceAvgTime /= nFiles;
	bestFirstAvgConf /= nFiles;
	bestFirstAvgErr /= nFiles;
	bestFirstAvgTime /= nFiles;
	weightVotingAvgConf /= nFiles;
	weightVotingAvgErr /= nFiles;
	weightVotingAvgTime /= nFiles;
	unanimVotingAvgConf /= nFiles;
	unanimVotingAvgErr /= nFiles;
	unanimVotingAvgTime /= nFiles;
	std::cout << "\n________________________________________________________________________\n";
	printf("\n   FREQUENCY DOMAIN   Accuracy %.2f \%%...Avg Conf %.0f \%%...Avg Err %.3f\370...Avg Time %.0fms...SuccessRate %.0f\n", 100*(1.f - fftAvgErr / 30.f), fftAvgConf * 100.f, fftAvgErr, fftAvgTime, fftNsuccessful / nFiles * 100.f);
	printf("\nPROJECTION PROFILING  Accuracy %.2f \%%...Avg Conf %.0f \%%...Avg Err %.3f\370...Avg Time %.0fms...SuccessRate %.0f\n", 100 * (1.f - varianceAvgErr / 30.f), varianceAvgConf * 100.f, varianceAvgErr, varianceAvgTime, varianceNsuccessful / nFiles * 100.f);
	printf("\nSPATIAL DOMAIN LINES  Accuracy %.2f \%%...Avg Conf %.0f \%%...Avg Err %.3f\370...Avg Time %.0fms...SuccessRate %.0f\n\n", 100 * (1.f - lineDetectAvgErr / 30.f), lineDetectAvgConf * 100.f, lineDetectAvgErr, lineDetectAvgTime, lineDetectNsuccessful / nFiles*100.f);
	float totalT = fftAvgTime + varianceAvgTime + lineDetectAvgTime;
	printf("\n BEST FIRST VOTING    Accuracy %.2f \%%...Avg Conf %.0f \%%...Avg Err %.3f\370...Avg Time %.0fms...SuccessRate %.0f\n", 100 * (1.f - bestFirstAvgErr / 30.f), bestFirstAvgConf * 100.f, bestFirstAvgErr, totalT);
	printf("\n   WEIGHTED VOTING    Accuracy %.2f \%%...Avg Conf %.0f \%%...Avg Err %.3f\370...Avg Time %.0fms...SuccessRate %.0f\n", 100 * (1.f - weightVotingAvgErr / 30.f), weightVotingAvgConf * 100.f, weightVotingAvgErr, totalT);
	printf("\n  UNANIMITY VOTING    Accuracy %.2f \%%...Avg Conf %.0f \%%...Avg Err %.3f\370...Avg Time %.0fms...SuccessRate %.0f\n", 100 * (1.f - unanimVotingAvgErr / 30.f), unanimVotingAvgConf * 100.f, unanimVotingAvgErr, totalT);
#endif
	for(int i = 0; i < 3; i++)
		cv::waitKey(0);
	return 0;

}