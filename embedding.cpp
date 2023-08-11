#include <iostream>
#include <opencv2/opencv.hpp>
#include "cv_dnn_centerface.h"
#include "torch/torch.h"
#include "torch/script.h"
#include "utils.h"

using namespace torch::indexing;
using namespace cv;




void prepareFaceData(std::string& face_data_path, std::vector<lite::types::FaceContent>& faces_data, std::vector<std::string>& names) {
	//names.emplace_back("Unknown");
	std::filesystem::path folderPath(face_data_path);
	if (std::filesystem::exists(folderPath) && std::filesystem::is_directory(folderPath)) {
		for (const auto& entry : std::filesystem::directory_iterator(folderPath)) {
			if (entry.is_directory()) {
				std::cout << "directory: " << entry.path().string() << std::endl;
				names.emplace_back(entry.path().filename());

				for (const auto& imageEntry : std::filesystem::directory_iterator(entry.path())) {
					if (imageEntry.is_regular_file()) {
						if (imageEntry.path().extension() == ".jpg" || imageEntry.path().extension() == ".png") {
							cv::Mat image = cv::imread(imageEntry.path().string());

							if (image.empty()) {
								std::cout << "Unable to read face_ Data image file: " << imageEntry.path().string() << std::endl;
							}
							else {
								lite::types::FaceContent face_content_temp;
								glint_arcface->detect(image, face_content_temp);
								if (face_content_temp.flag) {
									faces_data.emplace_back(face_content_temp);
								}
								else {
									std::cout << "face_content.flag == false: " << imageEntry.path().string() << std::endl;
								}
								//cv::imshow("Image", image);
								//cv::waitKey(0);
							}
						}
					}
				}

			}
		}
	}
	else {
		std::cout << "The folder does not exist or the path is incorrect!" << std::endl;
	}

	std::ofstream facesFile(folderPath / "faces_data.bin", std::ios::binary);
	if (facesFile.is_open()) {
		for (const auto& face : faces_data) {
			facesFile.write(reinterpret_cast<const char*>(&face.dim), sizeof(unsigned int));
			facesFile.write(reinterpret_cast<const char*>(face.embedding.data()), face.dim * sizeof(float));
			facesFile.write(reinterpret_cast<const char*>(&face.flag), sizeof(bool));
		}
		facesFile.close();
	}

	std::ofstream namesFile(folderPath / "names.bin", std::ios::binary);
	if (namesFile.is_open()) {
		for (const auto& name : names) {
			size_t length = name.size();
			namesFile.write(reinterpret_cast<const char*>(&length), sizeof(size_t));
			namesFile.write(reinterpret_cast<const char*>(name.data()), length);
		}
		namesFile.close();
	}
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
	at::Tensor embedding;

	for (int i = 0; i < face_info.size(); i++) {
		try{
			cv::Mat image_raw = image.clone();
			cv::rectangle(image, cv::Point(face_info[i].x1, face_info[i].y1), cv::Point(face_info[i].x2, face_info[i].y2), cv::Scalar(0, 255, 0), 2);
			// std::cout<<(int)face_info[i].x1<<"y1" << (int)face_info[i].y1<<"x2"<<(int)face_info[i].x2<<"y2"<< (int)face_info[i].y2<<std::endl;

			cv::Mat image_crop = align(image_raw, face_info[i]);
			// cv::Rect m_select = Rect((int)face_info[i].x1,(int)face_info[i].y1,(int)face_info[i].x2-(int)face_info[i].x1,(int)face_info[i].y2-(int)face_info[i].y1);
			// Mat image_crop = image(m_select);
			// resize(image_crop, image_crop, Size(112, 112), 0, 0, INTER_CUBIC);

			std::vector<int64_t> sizes = { 1, image_crop.rows, image_crop.cols,3 };
			at::TensorOptions options(at::ScalarType::Byte);
			at::Tensor img_input = torch::from_blob(image_crop.data, at::IntList(sizes), options); // Ch
			img_input = img_input.toType(at::kFloat);//转为浮点型张量数据
			img_input = img_input.permute({ 0, 3, 1, 2 }).div(255);
			output = module.forward({ img_input }).toTensor();
			
			if( i == 0){
				embedding = output;

			}
			else{
				embedding = torch::cat({embedding, output});
			}
			std::cout << "output:" << output.sizes() << std::endl;
		}
		catch (int myNum) {
		std::cout << "数据输入有问题！";
		}
		// auto img_test = img_tensor.index({ "...", Slice(face_info[i].x1, face_info[i].x2), Slice(face_info[i].y1, face_info[i].y2) });  
		// std::cout<<"测试尺寸："<<img_test.sizes()<<std::endl;

	}

	std::cout << "embedding:" << embedding.sizes() << std::endl;



	std::cout << "\n人脸识别，embedding生成成功";
	auto bytes = torch::jit::pickle_save(embedding);
	std::ofstream fout("embedding.pt", std::ios::out | std::ios::binary);
	fout.write(bytes.data(), bytes.size());
	fout.close();



	return 0;
}





