#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
//#include "opencv2/imgcodecs.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml/ml.hpp>

using namespace std;
using namespace cv;

// Mouse Callback function
void CallBackFunc(int event,int x,int y,int flags,void *userdata)
{
	if(event == EVENT_LBUTTONDOWN)
	{
		cout << "x = " << x << ", y = " << y << endl;
		Mat m = *(Mat*)userdata;
		cout << "Intensity = " << (int) m.at<uchar>(y,x) << endl;
	}
}

int main()
{
	//New 
	int num_files = 1;
	int img_area = 600*350;
	Mat training_mat(img_area,num_files,CV_32FC1);

	

	for(int k = 0;k < num_files;k++)
	{
		stringstream ss;
		ss << k;
		String str = ss.str();
		String imgname = "SVMDataset/";
		imgname = imgname + str + ".png";
		cout << imgname << endl;
		Mat img_mat = imread(imgname,0); // I used 0 for greyscale
		resize(img_mat,img_mat,Size(600,350));

		int ii = 0; // Current column in training_mat
		for (int i = 0; i<img_mat.rows; i++) {
			for (int j = 0; j < img_mat.cols; j++) {
				training_mat.at<float>(ii++,k) = img_mat.at<uchar>(i,j);
			}
		}
	}

	//float labels[1] = {1.0};
	Mat labelsMat(img_area,1,CV_32SC1);
	for(int i = 0;i < img_area;i++)
	{
		if(i < img_area/2)
			labelsMat.at<int>(i,0) = 1.0;
		else
			labelsMat.at<int>(i,0) = -1.0;
	}

	cout << training_mat.size() << endl;
	cout << labelsMat.size() << endl;

	//Set up SVM
	Ptr<ml::SVM> svm = ml::SVM::create();
	svm->setType(ml::SVM::C_SVC);
	svm->setKernel(ml::SVM::LINEAR);
	svm->setTermCriteria(cv::TermCriteria(CV_TERMCRIT_ITER,100,1e-6));

	svm->train(training_mat,ml::ROW_SAMPLE,labelsMat);
	
	svm->save("fin");
	
	svm = Algorithm::load<ml::SVM>("fin");

	Mat image = imread("SVMDataTest/1.png",0);
	resize(image,image,Size(600,350));
	Mat res = Mat::zeros(Size(image.cols,image.rows),CV_8UC3);
	Vec3b green(0,255,0), blue (255,0,0);

	for (int i = 0; i < image.rows; ++i)
	{
        for (int j = 0; j < image.cols; ++j)
        {
            //Mat sampleMat = (Mat_<float>(1,1) << 1);
			
			Mat sampleMat = (Mat_<float>(1,1) << image.at<uchar>(i,j));
			float response = svm->predict(sampleMat);

            if (response == 1)
                res.at<Vec3b>(i,j)  = green;
            else if (response == -1)
                 res.at<Vec3b>(i,j)  = blue;
        }
	}

	imshow("image",image);
	imshow("res",res);
	//Mat grey = imread("SVMDataset/1.png",0);
	//imshow("grey",grey);
	//setMouseCallback("grey",CallBackFunc,&grey);
	waitKey(0);

}
