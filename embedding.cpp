#include <iostream>
#include <opencv2/opencv.hpp>
#include "ncnn_centerface.h"
#include "torch/torch.h"
#include "torch/script.h"
#include "utils.h"



using namespace torch::indexing;
using namespace cv;



int main(int argc, char** argv) {
	// if (argc !=3)
	// {
	// 	std::cout << " .exe mode_path image_file" << std::endl;
	// 	return -1;
	// }

	std::string face_data_path = "../face_user";
	std::string model_path = "../models";
	std::vector<std::string> names;

	using torch::jit::script::Module;
	Module module = torch::jit::load("../face_model_ir_se50.pt");
	module.eval();
	std::filesystem::path folderPath(face_data_path);
	Centerface centerface;
	centerface.init(model_path);
	at::Tensor output;
	at::Tensor embedding;
	at::Tensor embedding_all;


	if (std::filesystem::exists(folderPath) && std::filesystem::is_directory(folderPath)) {
		int num_person = 0;
		for (const auto& entry : std::filesystem::directory_iterator(folderPath)) {
			if (entry.is_directory()) {
				std::cout << "directory: " << entry.path().string() << std::endl;
				names.emplace_back(entry.path().filename());
				int person_num_face = 0;
				for (const auto& imageEntry : std::filesystem::directory_iterator(entry.path())) {
					
					if (imageEntry.is_regular_file()) {
						if (imageEntry.path().extension() == ".jpg" || imageEntry.path().extension() == ".png") {
							cv::Mat image = cv::imread(imageEntry.path().string());
							// std::cout<<imageEntry.path().string()<<std::endl;

							if (image.empty()) {
								std::cout << "Unable to read face_ Data image file: " << imageEntry.path().string() << std::endl;
							}
							else {

								std::vector<FaceInfo> face_info;
								ncnn::Mat inmat = ncnn::Mat::from_pixels(image.data, ncnn::Mat::PIXEL_RGB, image.cols, image.rows);
								centerface.detect(inmat, face_info, image.cols, image.rows);
								for (int i = 0; i < face_info.size(); i++) {
										try{
											cv::Mat image_raw = image.clone();
											cv::rectangle(image, cv::Point(face_info[i].x1, face_info[i].y1), cv::Point(face_info[i].x2, face_info[i].y2), cv::Scalar(0, 255, 0), 2);
											cv::Mat image_crop = align(image_raw, face_info[i]);


											at::Tensor img_input = torch::from_blob(image_crop.data, { image_crop.rows, image_crop.cols, 3 }, torch::kByte);	//{ 256,256,3 }
											img_input = img_input.toType(torch::kFloat);					// 为了下一步归一化除255，将无符号整型转为float型
											img_input = img_input.div(255.0);								// 归一化
											img_input = img_input.permute({ 2,0,1 }).unsqueeze(0);




											// std::vector<int64_t> sizes = { 1, image_crop.rows, image_crop.cols,3 };
											// at::TensorOptions options(at::ScalarType::Byte);
											// at::Tensor img_input = torch::from_blob(image_crop.data, at::IntList(sizes), options); // Ch
											// img_input = img_input.toType(at::kFloat);//转为浮点型张量数据
											// std::cout << "img_input:" << img_input.sizes()<<img_input<<img_input.sum({-1}).sum({-1}).sum({-1}) << std::endl;


											// img_input = img_input.permute({ 0, 3, 1, 2 });
											std::cout << "随机的img_input:" << img_input.sizes()<<img_input.sum({-1}).sum({-1}).sum({-1}) << std::endl;
											output = module.forward({ img_input }).toTensor();											
											std::cout << "output:" << output.sizes()<<output.sum({1}) << std::endl;
										}
										catch (int myNum) {
										std::cout << "数据输入有问题！";
										}
								}
							}
						}
					}
					if( person_num_face == 0){
						embedding = output;
					}
					else{
						embedding += output;
					}
					person_num_face ++;		
				}
				embedding = embedding / person_num_face;
				std::cout << "每个人的mebedding:" << embedding.sum({1}) << std::endl;
				//对同一张人脸的embedding求均值
				if( num_person == 0){
				embedding_all = embedding;
				std::cout << "embedding_all_0:" << embedding_all.sizes() << std::endl;
				}
				else{
				embedding_all = torch::cat({embedding_all, embedding});
				std::cout << "embedding_all:" << embedding_all.sizes()<<embedding_all.sum({1})<< std::endl;
				}
				num_person ++;				
			}
	
		}
	}
	std::cout << "embedding_all:" << embedding_all.sizes() << std::endl;
	embedding = embedding_all.sum({1});
	std::cout<<"结果是："<<embedding.sizes()<<embedding<<std::endl;
	std::cout << "\n人脸识别，embedding生成成功";
	auto bytes = torch::jit::pickle_save(embedding_all);
	std::ofstream fout("embedding.pt", std::ios::out | std::ios::binary);
	fout.write(bytes.data(), bytes.size());
	fout.close();

	std::ofstream namesFile(folderPath / "names.bin", std::ios::binary);
	if (namesFile.is_open()) {
		for (const auto& name : names) {
			size_t length = name.size();
			namesFile.write(reinterpret_cast<const char*>(&length), sizeof(size_t));
			namesFile.write(reinterpret_cast<const char*>(name.data()), length);
		}
		namesFile.close();
	}
	return 0;
}





