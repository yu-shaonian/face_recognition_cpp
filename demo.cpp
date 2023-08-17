#include <iostream>
#include <opencv2/opencv.hpp>
#include "ncnn_centerface.h"
#include "torch/torch.h"
#include "torch/script.h"
#include "utils.h"

using namespace torch::indexing;
using namespace cv;

int main(int argc, char** argv) {

	using torch::jit::script::Module;
	std::string model_path = "../models";
	std::string face_data_path = "../face_user";
	std::vector<std::string> names;
	std::filesystem::path folderPath(face_data_path);
	std::ifstream namesFile(folderPath / "names.bin", std::ios::binary);
	if (namesFile.is_open()) {
		while (true) {
			size_t length;
			namesFile.read(reinterpret_cast<char*>(&length), sizeof(size_t));
			if (namesFile.eof())
				break;
			std::string name(length, '\0');
			namesFile.read(reinterpret_cast<char*>(name.data()), length);
			names.emplace_back(name);
		}
		namesFile.close();
	}
	std::vector<char> f = get_the_bytes("embedding.pt");
	torch::IValue x = torch::pickle_load(f);
	torch::Tensor embedding = x.toTensor();

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
		// cv::Mat image = cv::imread("/lgj/research/face_recognition_cpp.bak/face_user/company/company.jpg");
		cv::Mat image = frame;
		std::vector<FaceInfo> face_info;
		ncnn::Mat inmat = ncnn::Mat::from_pixels(image.data, ncnn::Mat::PIXEL_RGB, image.cols, image.rows);
		centerface.detect(inmat, face_info, image.cols, image.rows);

		for (int i = 0; i < face_info.size(); i++) {
				try{
					cv::Mat image_raw = image.clone();
					cv::rectangle(image, cv::Point(face_info[i].x1, face_info[i].y1), cv::Point(face_info[i].x2, face_info[i].y2), cv::Scalar(0, 255, 0), 2);
					cv::Mat image_crop = align(image_raw, face_info[i]);

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
		}

		cv::imwrite("test.jpg", image);
		std::cout << "识别到的人数为："<< face_info.size()<<"\n";
		if(face_info.size() > 0){

			torch::Tensor diff = output - embedding;
			diff = torch::pow(diff, 1);
			// std::cout<<"结果是："<<diff.sizes()<<diff<<std::endl;
			diff = diff.sum({1});
			// std::cout<<"结果是："<<diff.sizes()<<diff<<std::endl;
			std::tuple<torch::Tensor, torch::Tensor> b_tensor= torch::min(diff, -1, false);
			torch::Tensor min_vaule = std::get<0>(b_tensor);
			torch::Tensor min_index = std::get<1>(b_tensor);
			std::cout<<"最小值是："<<min_vaule<<std::endl;
			std::cout<<"最小值是对应的序号是："<<min_index<<std::endl;
			int face_num = min_index.item<int>();
			std::string face_name = names[face_num];
			std::cout << "识别到的人是："<<face_name<<std::endl;

		}

		cv::waitKey(10);
	}

	cap.release();
	std::cout << "\n人脸识别，ncnn,libtorch测试成功";


	return 0;
}
