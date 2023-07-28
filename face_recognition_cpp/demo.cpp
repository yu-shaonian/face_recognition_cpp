#include <iostream>
#include <opencv2/opencv.hpp>
#include "cv_dnn_centerface.h"
#include "torch/torch.h"
#include "torch/script.h"

using namespace torch::indexing;

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

	at::Tensor img_tensor = torch::from_blob(image.data, { image.rows, image.cols, 3 }, torch::kByte).permute({ 2, 0, 1 }); // Channels x Height x Width

	auto img_test = img_tensor.index({ "...", Slice(2, 50), Slice(2, 50) });  
	std::cout<<"测试尺寸："<<img_test.sizes()<<std::endl;

	centerface.detect(image, face_info);

	for (int i = 0; i < face_info.size(); i++) {
		cv::rectangle(image, cv::Point(face_info[i].x1, face_info[i].y1), cv::Point(face_info[i].x2, face_info[i].y2), cv::Scalar(0, 255, 0), 2);

	}

	cv::imwrite("test.jpg", image);
	std::cout << "识别到的人数为："<< face_info.size()<<"\n";
	torch::Tensor output = torch::randn({ 3,2 });
    std::cout << output;
	std::cout << "\n人脸识别，libtorch测试成功";


	return 0;
}
