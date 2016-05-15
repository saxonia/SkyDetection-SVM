#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml/ml.hpp>
#include <opencv2\imgproc\imgproc.hpp>

using namespace std;
using namespace cv;

#define dataWidth 320
#define dataHeight 240
#define trainMode 0
#define testMode 1

const string trackbarWindowName = "Trackbars";
const string imagePath = "ImagePath.xml";
const string dataSetPath = "SVMDataset/";
const string dataTestPath = "SVMDataset/opencountry/fie/";
const string testImagePath = "(10).jpg";

int mode = 1; 
const int stIdx = 0;
const int enIdx = 204;
const int numFiles = enIdx-stIdx+1;
const int channel = 9; 
const Vec3b green(0,255,0), blue (255,0,0);

int H_MIN = 0;
int H_MAX = 255;
int L_MIN = 0;
int L_MAX = 255;
int S_MIN = 0;
int S_MAX = 255;
int V_MIN = 0;
int V_MAX = 255;


// Mouse Callback function
void CallBackFunc(int event,int x,int y,int flags,void *userdata)
{
	if(event == EVENT_LBUTTONDOWN)
	{
		cout << "x = " << x << ", y = " << y << endl;
		Mat m = *(Mat*)userdata;
		//cout << "Intensity = " << (int) m.at<uchar>(y,x) << endl;
		cout << "Hue = " << (int) m.at<Vec3b>(y,x)[0] << endl;
		cout << "Saturate = " << (int) m.at<Vec3b>(y,x)[1] << endl;
		cout << "Value = " << (int) m.at<Vec3b>(y,x)[2] << endl;
	}
}

void on_trackbar( int, void* )
{
	//This function gets called whenever a trackbar position is changed
}

// Create Trackbars for color Segmentation
void createTrackbars()
{
	namedWindow(trackbarWindowName,CV_WINDOW_AUTOSIZE);
	//create memory to store trackbar name on window
	char TrackbarName[50];
	sprintf( TrackbarName, "H_MIN", H_MIN);
	sprintf( TrackbarName, "H_MAX", H_MAX);
	sprintf( TrackbarName, "L_MIN", L_MIN);
	sprintf( TrackbarName, "L_MAX", L_MAX);
	sprintf( TrackbarName, "S_MIN", S_MIN);
	sprintf( TrackbarName, "S_MAX", S_MAX);
	sprintf( TrackbarName, "V_MIN", V_MIN);
	sprintf( TrackbarName, "V_MAX", V_MAX);

	createTrackbar( "H_MIN", trackbarWindowName, &H_MIN, H_MAX, on_trackbar );
	createTrackbar( "H_MAX", trackbarWindowName, &H_MAX, H_MAX, on_trackbar );
	createTrackbar( "S_MIN", trackbarWindowName, &S_MIN, S_MAX, on_trackbar );
	createTrackbar( "S_MAX", trackbarWindowName, &S_MAX, S_MAX, on_trackbar );
	createTrackbar( "V_MIN", trackbarWindowName, &V_MIN, V_MAX, on_trackbar );
	createTrackbar( "V_MAX", trackbarWindowName, &V_MAX, V_MAX, on_trackbar );
}

bool readImageList(const string& filename,vector<string>& imageList)
{
	FileStorage fs(filename,FileStorage::READ);
	if(!fs.isOpened())
	{
		return false;
	}
	FileNode n = fs.getFirstTopLevelNode();
	/*if(n.type() != FileNode::SEQ)
	{
	return false;
	}*/
	FileNodeIterator it = n.begin();
	for(;it != n.end();it++)
	{
		imageList.push_back((string)*it);
	}
	return true;
}

int main(int argc, char* argv[])
{
	createTrackbars();
	namedWindow("Trackbars");
	moveWindow("Trackbars",100,100);
	Mat res;
	if(argc > 1)
	{
		cout << argv[1] << endl;
		mode = stoi(argv[1]);

	}
	if(mode == trainMode)
	{
		int imgArea = dataWidth*dataHeight;
		Mat trainingMat(imgArea*numFiles,channel,CV_32FC1);
		Mat labelsMat(imgArea*numFiles,1,CV_32SC1);


		vector<string> imageList;
		bool checkInput = readImageList(imagePath,imageList);

		if(!checkInput || stIdx > imageList.size() || enIdx > imageList.size() )
			return -1;

		int ii = 0; // Current row in trainingMat
		int jj = 0; // Current row in labelsMat
		for(int k = stIdx;k <= enIdx;k++)
		{
			Mat inputImg = imread(imageList[k]);
			cout << imageList[k] << endl;
			//String imgName = imageList[k-1];
			//Mat inputImg = imread(imgName); // I used 0 for greyscale

			if(inputImg.rows*inputImg.cols < 1) return -2;
			resize(inputImg,inputImg,Size(dataWidth,dataHeight));
			medianBlur(inputImg,inputImg,3);
			Mat inputHSV;
			Mat inputYCbCr;
			Mat inputLuv;
			cvtColor(inputImg,inputHSV,CV_BGR2HSV);
			cvtColor(inputImg,inputYCbCr,CV_BGR2YCrCb);
			cvtColor(inputImg,inputLuv,CV_BGR2Luv);

			/*imshow("one",inputHSV);
			setMouseCallback("one",CallBackFunc,&inputHSV);*/

			for (int i = 0; i<inputImg.rows; i++) {
				for (int j = 0; j < inputImg.cols; j++) {
					trainingMat.at<float>(ii,0) = inputHSV.at<Vec3b>(i,j)[0];
					trainingMat.at<float>(ii,1) = inputHSV.at<Vec3b>(i,j)[1];
					trainingMat.at<float>(ii,2) = inputHSV.at<Vec3b>(i,j)[2];
					trainingMat.at<float>(ii,3) = inputYCbCr.at<Vec3b>(i,j)[0];
					trainingMat.at<float>(ii,4) = inputYCbCr.at<Vec3b>(i,j)[1];
					trainingMat.at<float>(ii,5) = inputYCbCr.at<Vec3b>(i,j)[2];
					trainingMat.at<float>(ii,6) = inputLuv.at<Vec3b>(i,j)[0];
					trainingMat.at<float>(ii,7) = inputLuv.at<Vec3b>(i,j)[1];
					trainingMat.at<float>(ii,8) = inputLuv.at<Vec3b>(i,j)[2];
					ii++;
				}
			}

			int lowerBH = 255, upperBH = 0;
				int lowerBS = 255, upperBS = 0;
				int lowerBV = 255, upperBV = 0;
				int lowerBY = 255, upperBY = 0;
				int lowerBCb = 255, upperBCb = 0;
				int lowerBCr = 255, upperBCr = 0;
				if(imageList[k].find("highway") != string::npos)
				{
					lowerBH = 92;
					upperBH = 177;
					lowerBS = 59;
					upperBS = 255;
					lowerBV = 105;
					upperBV = 255;
					lowerBY = 131;
					upperBY = 255;
					lowerBCb = 0;
					upperBCb = 255;
					lowerBCr = 128;
					upperBCr = 255;
				}
				else if(imageList[k].find("mountain") != string::npos && imageList[k].find("nat") != string::npos)
				{
					lowerBH = 90;
					upperBH = 179;
					lowerBS = 94;
					upperBS = 255;
					lowerBV = 115;
					upperBV = 255;
					lowerBY = 0;
					upperBY = 255;
					lowerBCb = 0;
					upperBCb = 120;
					lowerBCr = 0;
					upperBCr = 255;
				}
				else if(imageList[k].find("opencountry") != string::npos && imageList[k].find("fie") != string::npos)
				{
					//cout << "Yes" << endl;
					lowerBH = 0;
					upperBH = 179;
					lowerBS = 0;
					upperBS = 90;
					lowerBV = 0;
					upperBV = 255;
					lowerBY = 133;
					upperBY = 255;
					lowerBCb = 0;
					upperBCb = 255;
					lowerBCr = 0;
					upperBCr = 255;
					/*lowerBH = 0;
					upperBH = 179;
					lowerBS = 0;
					upperBS = 100;
					lowerBV = 147;
					upperBV = 255;
					lowerBY = 90;
					upperBY = 255;
					lowerBCb = 80;
					upperBCb = 255;
					lowerBCr = 114;
					upperBCr = 255;*/
				}
				else if(imageList[k].find("moun") != string::npos)
				{
					lowerBH = 90;
					upperBH = 150;
					lowerBS = 120;
					upperBS = 255;
					lowerBV = 135;
					upperBV = 255;
					lowerBY = 190;
					upperBY = 255;
					lowerBCb = 110;
					upperBCb = 255;
					lowerBCr = 130;
					upperBCr = 255;
				}
				else if(imageList[k].find("n") != string::npos)
				{
					lowerBH = 80;
					upperBH = 130;
					lowerBS = 120;
					upperBS = 255;
					lowerBV = 135;
					upperBV = 255;
					lowerBY = 10;
					upperBY = 50;
					lowerBCb = 90;
					upperBCb = 130;
					lowerBCr = 150;
					upperBCr = 200;
				}
				else
				{
					lowerBH = 75;
					upperBH = 179;
					lowerBS = 120;
					upperBS = 255;
					lowerBV = 135;
					upperBV = 255;
					lowerBY = 90;
					upperBY = 255;
					lowerBCb = 80;
					upperBCb = 255;
					lowerBCr = 160;
					upperBCr = 255;
				}

			for(int i = 0;i < imgArea;i++)
			{
				int poX = i%inputImg.cols, poY = i/inputImg.cols;
				bool clause1 = inputHSV.at<Vec3b>(poY,poX)[0] >= lowerBH;
				bool clause2 = inputHSV.at<Vec3b>(poY,poX)[0] <= upperBH;
				bool clause3 = inputHSV.at<Vec3b>(poY,poX)[1] >= lowerBS;
				bool clause4 = inputHSV.at<Vec3b>(poY,poX)[1] <= upperBS;
				bool clause5 = inputHSV.at<Vec3b>(poY,poX)[2] >= lowerBV;
				bool clause6 = inputHSV.at<Vec3b>(poY,poX)[2] <= upperBV;

				//bool clause3 = inputHSV.at<Vec3b>(poY,poX)[1] >= 0;
				//bool clause4 = inputHSV.at<Vec3b>(poY,poX)[1] <= 20;
				bool clauseR1 = clause1 && clause2 && clause3 && clause4 && clause5 && clause6;

				clause1 = inputYCbCr.at<Vec3b>(poY,poX)[0] >= lowerBY;
				clause2 = inputYCbCr.at<Vec3b>(poY,poX)[0] <= upperBY;
				clause3 = inputYCbCr.at<Vec3b>(poY,poX)[1] >= lowerBCb;
				clause4 = inputYCbCr.at<Vec3b>(poY,poX)[1] <= upperBCb;
				clause5 = inputYCbCr.at<Vec3b>(poY,poX)[2] >= lowerBCr;
				clause6 = inputYCbCr.at<Vec3b>(poY,poX)[2] <= upperBCr;

				bool clauseR2 = clause1 && clause2 && clause3 && clause4 && clause5 && clause6;

				if(clauseR1 && clauseR2)
				{
					labelsMat.at<int>(jj++,0) = 1;
				}
				else
				{
					labelsMat.at<int>(jj++,0) = -1;	
				}
				/*if((clause1 && clause2) || (clause3 && clause4))
				labelsMat.at<int>(jj++,0) = 1.0;
				else
				labelsMat.at<int>(jj++,0) = -1.0;*/

			}
		}
		//cout << string::npos << endl;
		int blue = 0;
		int green = 0;
		for (int i = 0; i < labelsMat.rows*labelsMat.cols; i++)
		{
			if(labelsMat.at<int>(i,0) == 1) blue++;
			else green++;
		}
		cout << "blue" << blue << endl;
		cout << "green" << green << endl;
		cout << "total" << blue+green << endl;
		cout << trainingMat.size() << endl;
		cout << labelsMat.size() << endl;

		Ptr<ml::SVM> svm = ml::SVM::create();
		svm->setType(ml::SVM::C_SVC);
		svm->setKernel(ml::SVM::CHI2);
		svm->setTermCriteria(cv::TermCriteria(TermCriteria::MAX_ITER, 100, 1e-6));
		//svm->setTermCriteria(cv::TermCriteria(TermCriteria::MAX_ITER, 100, FLT_EPSILON));

		svm->train(trainingMat,ml::ROW_SAMPLE, labelsMat);
		//svm->train(trainingMat);

		svm->save("fin");
	}
	else if(mode == testMode)
	{
		Ptr<ml::SVM> svm = ml::SVM::create();
		svm = cv::Algorithm::load<ml::SVM>("fin");

		Mat testImage = imread(dataTestPath +  testImagePath);
		//Mat testImage = imread("SVMDataTest/(1).jpg");
		if(testImage.rows*testImage.cols < 1) return -6;
		resize(testImage,testImage,Size(dataWidth,dataHeight));

		medianBlur(testImage,testImage,3);

		Mat testHSV, testYCbCr, testLuv;
		cvtColor(testImage,testHSV,CV_BGR2HSV);
		cvtColor(testImage,testYCbCr,CV_BGR2YCrCb);
		cvtColor(testImage,testLuv,CV_BGR2Luv);


		res = Mat::zeros(Size(testImage.cols,testImage.rows),CV_8UC3);

		for (int i = 0; i < testImage.rows; ++i)
		{
			for (int j = 0; j < testImage.cols; ++j)
			{
				//Mat sampleMat = (Mat_<float>(1,1) << 1);
				//Mat sampleMat = (Mat_<float>(1,1) << testImage.at<Vec3b>(i,j)[0]);
				Mat sampleMat = (Mat_<float>(1,channel) << testHSV.at<Vec3b>(i,j)[0], testHSV.at<Vec3b>(i,j)[1], testHSV.at<Vec3b>(i,j)[2]
				,testYCbCr.at<Vec3b>(i,j)[0], testYCbCr.at<Vec3b>(i,j)[1], testYCbCr.at<Vec3b>(i,j)[2]
				,testLuv.at<Vec3b>(i,j)[0], testLuv.at<Vec3b>(i,j)[1], testLuv.at<Vec3b>(i,j)[2]
				);

				float response = svm->predict(sampleMat);

				if (response == 1)
					res.at<Vec3b>(i,j)  = blue;
				else if (response == -1)
					res.at<Vec3b>(i,j)  = green;

			}
		}

		Mat spv = svm->getUncompressedSupportVectors();
		//cout << spv.size() ;

		/*for (int i = 0; i < spv.rows; ++i)
		{
		const float* v = spv.ptr<float>(i);
		circle( res,  Point( (int) v[0], (int) v[1]),   6,  Scalar(128, 128, 128), 2, 8);
		}*/
		namedWindow("result");
		moveWindow("result",100,420);
		imshow("result",res);
		setMouseCallback("result",CallBackFunc,&res);
		//imshow("spv",spv);
	}

	Mat inputImg = imread(dataTestPath + testImagePath);
	if(inputImg.dims < 1) return -10;
	resize(inputImg,inputImg,Size(dataWidth,dataHeight));
	namedWindow("test");
	moveWindow("test",500,420);

	namedWindow("testHSV");
	moveWindow("testHSV",1000,420);
	namedWindow("testYCbCr");
	moveWindow("testYCbCr",1000,100);
	
	while (true)
	{
		Mat inputHSV;
		Mat inputYCbCr;
		cvtColor(inputImg,inputHSV,CV_BGR2HSV);
		cvtColor(inputImg,inputYCbCr,CV_BGR2YCrCb);
		Mat inputHSVInRange;
		Mat inputYCbCrInRange;
		inRange(inputHSV,Scalar(H_MIN,S_MIN,V_MIN),Scalar(H_MAX,S_MAX,V_MAX),inputHSVInRange);
		inRange(inputYCbCr,Scalar(H_MIN,S_MIN,V_MIN),Scalar(H_MAX,S_MAX,V_MAX),inputYCbCrInRange);
		imshow("test",inputImg);
		imshow("testHSV",inputHSVInRange);
		imshow("testYCbCr",inputYCbCrInRange);
		setMouseCallback("test",CallBackFunc,&inputHSV);
		if(waitKey(30) == 27) break;
	}


	waitKey(0);

}
