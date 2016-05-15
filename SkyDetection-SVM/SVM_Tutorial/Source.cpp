//#include <opencv2/core/core.hpp>
//#include <opencv2/highgui/highgui.hpp>
//#include <opencv2/ml/ml.hpp>
//#include <opencv2\imgproc\imgproc.hpp>
//
//using namespace std;
//using namespace cv;
//
//#define dataWidth 600
//#define dataHeight 350
//#define trainMode 0
//#define testMode 1
//
//const string imageCategory[] = {"opencountry/"};
//const string openCountryCat[] = {"fie/", "moun/", "n/"};
//const string dataSetPath = "SVMDataset/";
//const string dataTestPath = "SVMDataTest/";
//const string testImagePath = "(1).jpg";
//
//const int mode = 0; 
//const int stIdx = 1;
//const int enIdx = 23;
//const int numFiles = enIdx-stIdx+1;
//const int channel = 1; 
//const Vec3b green(0,255,0), blue (255,0,0);
//
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
//		cout << "Saturate = " << (int) m.at<Vec3b>(y,x)[1] << endl;
//		cout << "Value = " << (int) m.at<Vec3b>(y,x)[2] << endl;
//	}
//}
//
//int main()
//{
//	if(mode == trainMode)
//	{
//		int imgArea = dataWidth*dataHeight;
//		Mat trainingMat(imgArea*numFiles,channel,CV_32FC1);
//		Mat labelsMat(imgArea*numFiles,1,CV_32SC1);
//		
//		
//		for (int i = 0; i < sizeof(imageCategory)/sizeof(string); i++)
//		{
//			if(imageCategory[i].compare("opencountry/"))
//			for (int j = 0; j < sizeof(openCountryCat)/sizeof(string); j++)
//			{
//				for (int k = 0; k < ; k++)
//				{
//
//				}
//			}
//		}
//
//		//int jj = 0; // Current row in labelsMat
//		//for(int k = stIdx;k <= enIdx;k++)
//		//{
//		//	stringstream ss;
//		//	ss << k;
//		//	String str = ss.str();
//		//	String imgName = dataSetPath + imageCategory[] +"("+ str + ")" + ".jpg";
//		//	cout << imgName << endl;
//		//	Mat inputImg = imread(imgName); // I used 0 for greyscale
//		//	if(inputImg.rows*inputImg.cols < 1) return -5;
//		//	resize(inputImg,inputImg,Size(dataWidth,dataHeight));
//
//		//	Mat inputHSV;
//		//	cvtColor(inputImg,inputHSV,CV_BGR2HSV);
//
//		//	imshow("one",inputHSV);
//		//	setMouseCallback("one",CallBackFunc,&inputHSV);
//
//		//	int ii = 0; // Current row in trainingMat
//		//	for (int i = 0; i<inputImg.rows; i++) {
//		//		for (int j = 0; j < inputImg.cols; j++) {
//		//			trainingMat.at<float>(ii,k) = inputHSV.at<Vec3b>(i,j)[0];
//		//			//trainingMat.at<float>(ii,1) = (int) inputHSV.at<Vec3b>(i,j)[1];
//		//			//trainingMat.at<float>(ii++,k) = inputHSV.at<Vec3b>(i,j)[2];
//		//			ii++;
//		//		}
//		//	}
//
//		//	for(int i = 0;i < imgArea;i++)
//		//	{
//		//		bool clause1 = inputHSV.at<Vec3b>(i/inputImg.cols,i%inputImg.cols)[0] > 70;
//		//		bool clause2 = inputHSV.at<Vec3b>(i/inputImg.cols,i%inputImg.cols)[0] < 255;
//		//		//bool clause3 = inputHSV.at<Vec3b>(i/inputImg.cols,i%inputImg.cols)[1] >= 20;
//		//		//bool clause4 = inputHSV.at<Vec3b>(i/inputImg.cols,i%inputImg.cols)[1] <= 252;
//		//		if(clause1 && clause2)
//		//			labelsMat.at<int>(i,k) = 1;
//		//		else
//		//			labelsMat.at<int>(i,k) = -1;
//		//		//if(clause1 && clause2 && clause3 && clause4)
//		//		//	labelsMat.at<int>(jj++,0) = 1.0;
//		//		//else
//		//		//	labelsMat.at<int>(jj++,0) = -1.0;
//		//	}
//		//}
//
//		//cout << trainingMat.size() << endl;
//		//cout << labelsMat.size() << endl;
//
//		//Ptr<ml::SVM> svm = ml::SVM::create();
//		//svm->setType(ml::SVM::C_SVC);
//		//svm->setKernel(ml::SVM::LINEAR);
//		//svm->setTermCriteria(cv::TermCriteria(CV_TERMCRIT_ITER, 100, 1e-6));
//		//svm->train(trainingMat,ml::ROW_SAMPLE, labelsMat);
//
//		//svm->save("fin");
//	}
//	else if(mode == testMode)
//	{
//		Ptr<ml::SVM> svm = ml::SVM::create();
//		svm = cv::Algorithm::load<ml::SVM>("fin");
//
//		Mat testImage = imread(dataTestPath +  testImagePath);
//		//Mat testImage = imread("SVMDataTest/(1).jpg");
//		if(testImage.rows*testImage.cols < 1) return -6;
//		resize(testImage,testImage,Size(dataWidth,dataHeight));
//		cvtColor(testImage,testImage,CV_BGR2HSV);
//
//		Mat res = Mat::zeros(Size(testImage.cols,testImage.rows),CV_8UC3);
//
//		for (int i = 0; i < testImage.rows; ++i)
//		{
//			for (int j = 0; j < testImage.cols; ++j)
//			{
//				//Mat sampleMat = (Mat_<float>(1,1) << 1);
//				Mat sampleMat = (Mat_<float>(1,1) << testImage.at<Vec3b>(i,j)[0]);
//
//				float response = svm->predict(sampleMat);
//
//				if (response == 1)
//					res.at<Vec3b>(i,j)  = blue;
//				else if (response == -1)
//					res.at<Vec3b>(i,j)  = green;
//			}
//		}
//		imshow("res",res);
//	}
//
//	Mat inputImg = imread(dataTestPath + "("+ "1" + ")" + ".jpg");
//	resize(inputImg,inputImg,Size(dataWidth,dataHeight));
//	Mat inputHSV;
//	cvtColor(inputImg,inputHSV,CV_BGR2HSV);
//	imshow("one",inputHSV);
//	setMouseCallback("one",CallBackFunc,&inputHSV);
//
//	waitKey(0);
//
//}
