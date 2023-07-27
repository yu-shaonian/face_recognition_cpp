#include <iostream>
#include <opencv2/opencv.hpp>
#include "cv_dnn_centerface.h"
#include "torch/torch.h"
#include "torch/script.h"

int main(int argc, char** argv) {
	if (argc !=3)
	{
		std::cout << " .exe mode_path image_file" << std::endl;
		return -1;
	}

	std::string model_path = argv[1];
	std::string image_file = argv[2];

	Centerface centerface(model_path,640,480);

	cv::Mat image = cv::imread(image_file);
	std::vector<FaceInfo> face_info;
	
	centerface.detect(image, face_info);

	for (int i = 0; i < face_info.size(); i++) {
		cv::rectangle(image, cv::Point(face_info[i].x1, face_info[i].y1), cv::Point(face_info[i].x2, face_info[i].y2), cv::Scalar(0, 255, 0), 2);

	}

	cv::imwrite("test.jpg", image);
	torch::Tensor output = torch::randn({ 3,2 });
    std::cout << output;


	return 0;
}
