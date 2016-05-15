//#include <opencv2/core/core.hpp>
//#include <opencv2/highgui/highgui.hpp>
//#include <opencv2/ml/ml.hpp>
//#include <opencv2\imgproc\imgproc.hpp>
//
//using namespace std;
//using namespace cv;
//
//// Mouse Callback function
//void CallBackFunc(int event,int x,int y,int flags,void *userdata)
//{
//	if(event == EVENT_LBUTTONDOWN)
//	{
//		cout << "x = " << x << ", y = " << y << endl;
//		Mat m = *(Mat*)userdata;
//		//cout << "Intensity = " << (int) m.at<uchar>(y,x) << endl;
//		cout << "Hue = " << (int) m.at<Vec3b>(y,x)[0] << endl;
//	}
//}
//
//int mode = 1;
//
//int main()
//{
//	if(mode == 0)
//	{
//		//New 
//		int numFiles = 77;
//		int imgArea = 600*350;
//		Mat training_mat(imgArea*numFiles,1,CV_32FC1);
//		Mat labelsMat(imgArea*numFiles,1,CV_32SC1);
//
//		for(int k = 0;k < numFiles;k++)
//		{
//			stringstream ss;
//			ss << (k+1);
//			String str = ss.str();
//			String imgName = "SVMDataset/";
//			imgName = imgName + "("+ str + ")" + ".jpg";
//			cout << imgName << endl;
//			Mat inputImg = imread(imgName); // I used 0 for greyscale
//			cout << inputImg.size() << endl;
//			resize(inputImg,inputImg,Size(600,350));
//
//			Mat inputHSV;
//			cvtColor(inputImg,inputHSV,CV_BGR2HSV);
//
//			int ii = 0; // Current column in training_mat
//			for (int i = 0; i<inputImg.rows; i++) {
//				for (int j = 0; j < inputImg.cols; j++) {
//					training_mat.at<float>(ii++,k) = inputHSV.at<Vec3b>(i,j)[0];
//					//training_mat.at<float>(ii++,k) = inputHSV.at<Vec3b>(i,j)[1];
//					//training_mat.at<float>(ii++,k) = inputHSV.at<Vec3b>(i,j)[2];
//				}
//			}
//
//			for(int i = 0;i < imgArea;i++)
//			{
//				bool clause1 = inputHSV.at<Vec3b>(i/inputImg.cols,i%inputImg.cols)[0] > 70;
//				bool clause2 = inputHSV.at<Vec3b>(i/inputImg.cols,i%inputImg.cols)[0] < 255;
//				if(clause1 && clause2)
//					labelsMat.at<int>(i,k) = 1;
//				else
//					labelsMat.at<int>(i,k) = -1;
//			}
//		}
//
//
//
//
//
//		cout << training_mat.size() << endl;
//		cout << labelsMat.size() << endl;
//
//		Ptr<ml::SVM> svm = ml::SVM::create();
//		svm->setType(ml::SVM::C_SVC);
//		svm->setKernel(ml::SVM::LINEAR);
//		svm->setTermCriteria(cv::TermCriteria(CV_TERMCRIT_ITER, 100, 1e-6));
//		svm->train(training_mat,ml::ROW_SAMPLE, labelsMat);
//
//
//		svm->save("fin");
//	}
//	else if(mode == 1)
//	{
//		Ptr<ml::SVM> svm = ml::SVM::create();
//		svm = cv::Algorithm::load<ml::SVM>("fin");
//
//
//		Mat image = imread("SVMDataTest/1.png");
//		//Mat image = imread("SVMDataTest/1.png");
//		//resize(image,image,Size(600,350));
//		//cvtColor(image,image,CV_BGR2HSV);
//		Mat res = Mat::zeros(Size(image.cols,image.rows),CV_8UC3);
//		Vec3b green(0,255,0), blue (255,0,0);
//
//		for (int i = 0; i < image.rows; ++i)
//			for (int j = 0; j < image.cols; ++j)
//			{
//				//Mat sampleMat = (Mat_<float>(1,1) << 1);
//
//				Mat sampleMat = (Mat_<float>(1,1) << image.at<Vec3b>(i,j)[0]);
//				float response = svm->predict(sampleMat);
//
//				if (response == 1)
//					res.at<Vec3b>(i,j)  = blue;
//				else if (response == -1)
//					res.at<Vec3b>(i,j)  = green;
//			}
//
//
//			imshow("res",res);
//			/*Mat grey = imread("SVMDataset/0.png");
//			resize(grey,grey,Size(600,350));
//			cvtColor(grey,grey,CV_BGR2HSV);
//			imshow("grey",grey);
//			setMouseCallback("grey",CallBackFunc,&grey);*/
//
//			Mat grey1 = imread("SVMDataTest/1.png");
//			cvtColor(grey1,grey1,CV_BGR2HSV);
//			imshow("grey1",grey1);
//			setMouseCallback("grey1",CallBackFunc,&grey1);
//	}
//	waitKey(0);
//
//}
//
