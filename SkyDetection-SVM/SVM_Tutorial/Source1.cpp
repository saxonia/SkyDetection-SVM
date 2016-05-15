#include <opencv2\opencv.hpp>

int main()
{
	/*cv::VideoCapture cap(0);
	cv::Mat frame;
	while (true)
	{
		cap >> frame;
		cv::imshow("1",frame);
		cv::waitKey(1);
	}*/

	cv::Mat img = cv::imread("SVMDataset/opencountry/n/(1).jpg");
	cv::imshow("1",img);
	cv::waitKey();

	return 0;
}
