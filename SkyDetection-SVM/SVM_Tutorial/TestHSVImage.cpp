#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml/ml.hpp>
#include <opencv2\imgproc\imgproc.hpp>

using namespace std;
using namespace cv;

// Mouse Callback function
void CallBackFunc(int event,int x,int y,int flags,void *userdata)
{
	if(event == EVENT_LBUTTONDOWN)
	{
		cout << "x = " << x << ", y = " << y << endl;
		Mat m = *(Mat*)userdata;
		//cout << "Intensity = " << (int) m.at<uchar>(y,x) << endl;
		cout << "Hue = " << (int) m.at<Vec3b>(y,x)[0] << endl;
	}
}

int main()
{
	//New 
	int numFiles = 1;
	int imgArea = 600*350;
	Mat training_mat(imgArea,numFiles,CV_32FC1);
	Mat labelsMat(imgArea,1,CV_32FC1);

	for(int k = 0;k < numFiles;k++)
	{
		stringstream ss;
		ss << k;
		String str = ss.str();
		String imgName = "SVMDataset/";
		imgName = imgName + str + ".png";
		cout << imgName << endl;
		Mat inputImg = imread(imgName); // I used 0 for greyscale
		resize(inputImg,inputImg,Size(600,350));

		Mat inputHSV;
		cvtColor(inputImg,inputHSV,CV_BGR2HSV);

		int ii = 0; // Current column in training_mat
		for (int i = 0; i<inputImg.rows; i++) {
			for (int j = 0; j < inputImg.cols; j++) {
				training_mat.at<float>(ii++,k) = inputHSV.at<Vec3b>(i,j)[0];
				//training_mat.at<float>(ii++,k) = inputHSV.at<Vec3b>(i,j)[1];
				//training_mat.at<float>(ii++,k) = inputHSV.at<Vec3b>(i,j)[2];
			}
		}

		for(int i = 0;i < imgArea;i++)
		{
			bool clause1 = inputHSV.at<Vec3b>(i/inputImg.cols,i%inputImg.cols)[0] > 75;
			bool clause2 = inputHSV.at<Vec3b>(i/inputImg.cols,i%inputImg.cols)[0] < 120;
			if(clause1 && clause2)
				labelsMat.at<float>(i,k) = 1.0;
			else
				labelsMat.at<float>(i,k) = -1.0;
		}
	}





	cout << training_mat.size() << endl;
	cout << labelsMat.size() << endl;

	// Set up SVM's parameters
	CvSVMParams params;
	params.svm_type    = CvSVM::C_SVC;
	params.kernel_type = CvSVM::LINEAR;
	params.term_crit   = cvTermCriteria(CV_TERMCRIT_ITER, 100, 1e-6);


	CvSVM svm;
	svm.train(training_mat, labelsMat, Mat(), Mat(), params);

	svm.save("fin");
	svm.load("fin");

	Mat image = imread("1.png",0);
	resize(image,image,Size(600,350));
	Mat res = Mat::zeros(Size(image.cols,image.rows),CV_8UC3);
	Vec3b green(0,255,0), blue (255,0,0);

	for (int i = 0; i < image.rows; ++i)
		for (int j = 0; j < image.cols; ++j)
		{
			//Mat sampleMat = (Mat_<float>(1,1) << 1);

			Mat sampleMat = (Mat_<float>(1,1) << image.at<uchar>(i,j));
			float response = svm.predict(sampleMat);

			if (response == 1)
				res.at<Vec3b>(i,j)  = blue;
			else if (response == -1)
				res.at<Vec3b>(i,j)  = green;
		}


		imshow("res",res);
		Mat grey = imread("SVMDataset/1.png");
		cvtColor(grey,grey,CV_BGR2HSV);
		imshow("grey",grey);
		setMouseCallback("grey",CallBackFunc,&grey);

		Mat grey1 = imread("1.png");
		cvtColor(grey1,grey1,CV_BGR2HSV);
		imshow("grey1",grey1);
		setMouseCallback("grey1",CallBackFunc,&grey1);

		waitKey(0);

}
