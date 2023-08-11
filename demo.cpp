#include <iostream>
#include <opencv2/opencv.hpp>
#include "ncnn_centerface.h"
#include "torch/torch.h"
#include "torch/script.h"

using namespace torch::indexing;
using namespace cv;

int main(int argc, char** argv) {

	using torch::jit::script::Module;
	std::string model_path = "../models";
	Module module = torch::jit::load("../face_model_ir_se50.pt");
	module.eval();


	// std::string model_path = argv[1];
	std::string image_file = "../1686205515075.mp4";

	Centerface centerface;
	centerface.init(model_path);


	cv::VideoCapture cap(image_file);
	if(!cap.isOpened()) {
		std::cout << "Failed to open video file." << std::endl;
		return -1;
	}
	cv::Mat frame;
	at::Tensor output;
	while(cap.isOpened()) {
		cap >> frame;
		// cv::Mat image = cv::imread(image_file);
		cv::Mat image = frame;
		std::vector<FaceInfo> face_info;
		ncnn::Mat inmat = ncnn::Mat::from_pixels(image.data, ncnn::Mat::PIXEL_BGR2RGB, image.cols, image.rows);
		centerface.detect(inmat, face_info, image.cols, image.rows);

		for (int i = 0; i < face_info.size(); i++) {
			try{
				cv::rectangle(image, cv::Point(face_info[i].x1, face_info[i].y1), cv::Point(face_info[i].x2, face_info[i].y2), cv::Scalar(0, 255, 0), 2);
				std::cout<<(int)face_info[i].x1<<"y1" << (int)face_info[i].y1<<"x2"<<(int)face_info[i].x2<<"y2"<< (int)face_info[i].y2<<std::endl;
				cv::Rect m_select = Rect((int)face_info[i].x1,(int)face_info[i].y1,(int)face_info[i].x2-(int)face_info[i].x1,(int)face_info[i].y2-(int)face_info[i].y1);
				Mat image_crop = image(m_select);
				resize(image_crop, image_crop, Size(112, 112), 0, 0, INTER_CUBIC);

				std::vector<int64_t> sizes = { 1, image_crop.rows, image_crop.cols,3 };
				at::TensorOptions options(at::ScalarType::Byte);

				at::Tensor img_input = torch::from_blob(image_crop.data, at::IntList(sizes), options); // Ch
				img_input = img_input.toType(at::kFloat);//转为浮点型张量数据
				img_input = img_input.permute({ 0, 3, 1, 2 }).div(255);
				output = module.forward({ img_input }).toTensor();
				std::cout << "output:" << output.sizes() << std::endl;
			}
			catch (int myNum) {
			std::cout << "数据输入有问题！";
			}
			// auto img_test = img_tensor.index({ "...", Slice(face_info[i].x1, face_info[i].x2), Slice(face_info[i].y1, face_info[i].y2) });  
			// std::cout<<"测试尺寸："<<img_test.sizes()<<std::endl;

		}

		cv::imwrite("test.jpg", image);
		std::cout << "识别到的人数为："<< face_info.size()<<"\n";
		cv::waitKey(10);
	}

	cap.release();
	// torch::Tensor input = torch::randn({ 1,3,112,112 });
	// torch::Tensor embedding = torch::randn({ 8,512 });
	// torch::Tensor diff = output - embedding;
	// diff = torch::pow(diff, 1);
	// std::cout<<diff.sizes()<<std::endl;;
	// diff = diff.sum({1});
	// std::cout<<diff.sizes()<<std::endl;
	// std::tuple<torch::Tensor, torch::Tensor> b_tensor= torch::max(diff, 0, false);
	// torch::Tensor max_vaule = std::get<0>(b_tensor);
	// torch::Tensor max_index = std::get<1>(b_tensor);
	// std::cout<<max_vaule<<std::endl;
	// std::cout<<max_index<<std::endl;


	// at::Tensor test = module.forward({ input }).toTensor();
	// std::cout << "test:" << test.sizes() << std::endl;
    // std::cout << output.sizes();
	std::cout << "\n人脸识别，ncnn,libtorch测试成功";


	return 0;
}
