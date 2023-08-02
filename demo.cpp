#include <iostream>
#include <opencv2/opencv.hpp>
#include "cv_dnn_centerface.h"
#include "torch/torch.h"
#include "torch/script.h"

using namespace torch::indexing;
using namespace cv;





std::vector<char> get_the_bytes(std::string filename) {
    std::ifstream input(filename, std::ios::binary);
    std::vector<char> bytes(
        (std::istreambuf_iterator<char>(input)),
        (std::istreambuf_iterator<char>()));

    input.close();
    return bytes;
}



int main(int argc, char** argv) {
	if (argc !=3)
	{
		std::cout << " .exe mode_path image_file" << std::endl;
		return -1;
	}

	using torch::jit::script::Module;
	Module module = torch::jit::load("../face_model_ir_se50.pt");
	module.eval();


	std::string model_path = argv[1];
	std::string image_file = argv[2];

	Centerface centerface(model_path,640,480);

	cv::Mat image = cv::imread(image_file);
	std::vector<FaceInfo> face_info;

	at::Tensor img_tensor = torch::from_blob(image.data, { image.rows, image.cols, 3 }, torch::kByte).permute({ 2, 0, 1 }); // Channels x Height x Width



	centerface.detect(image, face_info);
	at::Tensor output;

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



	std::vector<char> f = get_the_bytes("embedding.pt");
    torch::IValue x = torch::pickle_load(f);
    torch::Tensor embedding = x.toTensor();


	torch::Tensor diff = output - embedding;
	diff = torch::pow(diff, 1);
	std::cout<<diff.sizes()<<std::endl;;
	diff = diff.sum({1});
	std::cout<<diff.sizes()<<std::endl;
	std::tuple<torch::Tensor, torch::Tensor> b_tensor= torch::max(diff, 0, false);
	torch::Tensor max_vaule = std::get<0>(b_tensor);
	torch::Tensor max_index = std::get<1>(b_tensor);
	std::cout<<max_vaule<<std::endl;
	std::cout<<max_index<<std::endl;


	std::cout << "\n人脸识别，libtorch测试成功";


	return 0;
}
