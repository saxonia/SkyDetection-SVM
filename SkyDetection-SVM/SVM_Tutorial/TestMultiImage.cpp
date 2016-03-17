#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml/ml.hpp>
#include <opencv2\imgproc\imgproc.hpp>

using namespace std;
using namespace cv;

#define dataWidth 600
#define dataHeight 350

int numFiles = 9;
const int stIdx = 0;
const int enIdx = 8;


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

int main()
{
	//New 
	int imgArea = dataWidth*dataHeight;
	Mat trainingMat(imgArea*numFiles,2,CV_32FC1);
	Mat labelsMat(imgArea*numFiles,1,CV_32FC1);

	int ii = 0; // Current row in trainingMat
	int jj = 0; // Current row in labelsMat
	for(int k = stIdx;k < enIdx;k++)
	{
		stringstream ss;
		ss << k;
		String str = ss.str();
		String imgName = "SVMDataset/";
		imgName = imgName + str + ".png";
		//cout << imgName << endl;
		Mat inputImg = imread(imgName); // I used 0 for greyscale
		resize(inputImg,inputImg,Size(dataWidth,dataHeight));

		Mat inputHSV;
		cvtColor(inputImg,inputHSV,CV_BGR2HSV);

		for (int i = 0; i<inputImg.rows; i++) {
			for (int j = 0; j < inputImg.cols; j++) {
				trainingMat.at<float>(ii,0) = inputHSV.at<Vec3b>(i,j)[0];
				trainingMat.at<float>(ii++,1) = inputHSV.at<Vec3b>(i,j)[1];
				//trainingMat.at<float>(ii++,k) = inputHSV.at<Vec3b>(i,j)[2];
			}
		}

		for(int i = 0;i < imgArea;i++)
		{
			bool clause1 = inputHSV.at<Vec3b>(i/inputImg.cols,i%inputImg.cols)[0] >= 75;
			bool clause2 = inputHSV.at<Vec3b>(i/inputImg.cols,i%inputImg.cols)[0] <= 120;
			bool clause3 = inputHSV.at<Vec3b>(i/inputImg.cols,i%inputImg.cols)[1] >= 20;
			bool clause4 = inputHSV.at<Vec3b>(i/inputImg.cols,i%inputImg.cols)[1] <= 252;
			if(clause1 && clause2 )
				labelsMat.at<float>(jj++,0) = 1.0;
			else
				labelsMat.at<float>(jj++,0) = -1.0;
		}
	}





	cout << trainingMat.size() << endl;
	cout << labelsMat.size() << endl;

	// Set up SVM's parameters
	CvSVMParams params;
	params.svm_type    = CvSVM::C_SVC;
	params.kernel_type = CvSVM::LINEAR;
	params.term_crit   = cvTermCriteria(CV_TERMCRIT_ITER, 100, 1e-6);


	CvSVM svm;
	svm.train(trainingMat, labelsMat, Mat(), Mat(), params);

	//svm.save("fin");
	svm.load("fin");

	Mat testImage = imread("SVMDataset/3.png");
	resize(testImage,testImage,Size(dataWidth,dataHeight));
	Mat testImageHSV;
	cvtColor(testImage,testImageHSV,CV_BGR2HSV);
	Mat res = Mat::zeros(Size(testImage.cols,testImage.rows),CV_8UC3);
	Vec3b green(0,255,0), blue (255,0,0);

	for (int i = 0; i < testImage.rows; ++i)
		for (int j = 0; j < testImage.cols; ++j)
		{
			//Mat sampleMat = (Mat_<float>(1,1) << 1);

			Mat sampleMat = (Mat_<float>(1,2) << testImageHSV.at<Vec3b>(i,j)[0],testImageHSV.at<Vec3b>(i,j)[1]);
			float response = svm.predict(sampleMat);

			if (response == 1)
				res.at<Vec3b>(i,j)  = blue;
			else if (response == -1)
				res.at<Vec3b>(i,j)  = green;
		}

	imshow("testImage",testImage);
	setMouseCallback("testImage",CallBackFunc,&testImageHSV);
	imshow("res",res);
	/*Mat grey = imread("SVMDataset/5.png");
	resize(grey,grey,Size(dataWidth,dataHeight));
	cvtColor(grey,grey,CV_BGR2HSV);
	imshow("grey",grey);
	setMouseCallback("grey",CallBackFunc,&grey);*/

	
	//imshow("testImage",testImage);
	waitKey(0);

}
