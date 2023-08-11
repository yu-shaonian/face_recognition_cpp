#include "utils.h"


cv::Mat align(cv::Mat& image, FaceInfo& face) {

	float scale = std::min(112.0f / (face.x2-face.x1), 112.0f / (face.y2-face.y1));
	std::vector<cv::Point2f> points_from;

	cv::Size2f resizeSizeTemp = {image.size().width * scale, image.size().height * scale};
	cv::Mat image_clone = image.clone();
	cv::resize(image_clone, image_clone, resizeSizeTemp);

	for (int i = 0; i < 5; i++) {
		points_from.emplace_back(face.landmarks[2 * i] * scale, face.landmarks[2 * i + 1] * scale);
		// std::cout<<"landmarks: "<<face.landmarks[2 * i] * scale<<", "<<face.landmarks[2 * i + 1] * scale<<std::endl;

		cv::circle(image_clone, points_from[i], 2, cv::Scalar(255, 255, 0), 2);
		
	}

	// cv::InputArray inputArray_from(points_from);
	std::vector<cv::Point2f> points_to = { {37.5f, 51.5f},
		{74.5f, 51.5f},
		{56.0f, 71.74f},
		{40.5f, 92.25f},
		{71.5f, 92.25f} };
	// cv::InputArray inputArray_to(points_to);

	for (int j = 0; j < 5; j++) {
		cv::circle(image_clone, points_to[j], 2, cv::Scalar(0, 0, 255), 2);
	}

	cv::Mat M = cv::estimateAffinePartial2D(points_from, points_to);

	// std::cout<<"Matrix: "<<M<<std::endl;

	cv::Mat resizedImage;
	cv::Mat warpedImage;
	cv::Size2f resizeSize = {image.size().width * scale, image.size().height * scale};
	// size of the output image
	cv::Size size = { 112, 112 };
	cv::resize(image, resizedImage, resizeSize);
	cv::warpAffine(resizedImage, warpedImage, M, size);

	return warpedImage;
}






