#define _CRT_SECURE_NO_WARNINGS
#include <iostream>
#include <fstream>
#include <string>
#include <math.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
//#include <cuda_provider_factory.h>
#include <onnxruntime_cxx_api.h>

using namespace cv;
using namespace std;
using namespace Ort;


class U2Net
{
public:
	U2Net();
	Mat detect(Mat& cv_image);
private:
	vector<float> input_image_;
	int inpWidth;
	int inpHeight;
	int outWidth;
	int outHeight;
	const float mean[3] = { 0.485, 0.456, 0.406 };
	const float stds[3] = { 0.229, 0.224, 0.225 };

	Env env = Env(ORT_LOGGING_LEVEL_ERROR, "u2net");
	Ort::Session *ort_session = nullptr;
	SessionOptions sessionOptions = SessionOptions();
	vector<char*> input_names;
	vector<char*> output_names;
	vector<vector<int64_t>> input_node_dims; // >=1 outputs
	vector<vector<int64_t>> output_node_dims; // >=1 outputs
};

U2Net::U2Net()
{
	string model_path = "u2net_portrait.onnx";
	std::wstring widestr = std::wstring(model_path.begin(), model_path.end());
	//OrtStatus* status = OrtSessionOptionsAppendExecutionProvider_CUDA(sessionOptions, 0);
	sessionOptions.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);
	ort_session = new Session(env, widestr.c_str(), sessionOptions);
	size_t numInputNodes = ort_session->GetInputCount();
	size_t numOutputNodes = ort_session->GetOutputCount();
	AllocatorWithDefaultOptions allocator;
	for (int i = 0; i < numInputNodes; i++)
	{
		input_names.push_back(ort_session->GetInputName(i, allocator));
		Ort::TypeInfo input_type_info = ort_session->GetInputTypeInfo(i);
		auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
		auto input_dims = input_tensor_info.GetShape();
		input_node_dims.push_back(input_dims);
	}
	for (int i = 0; i < numOutputNodes; i++)
	{
		output_names.push_back(ort_session->GetOutputName(i, allocator));
		Ort::TypeInfo output_type_info = ort_session->GetOutputTypeInfo(i);
		auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
		auto output_dims = output_tensor_info.GetShape();
		output_node_dims.push_back(output_dims);
	}
	this->inpHeight = input_node_dims[0][2];
	this->inpWidth = input_node_dims[0][3];
	this->outHeight = output_node_dims[0][2];
	this->outWidth = output_node_dims[0][3];
}

Mat U2Net::detect(Mat& srcimg)
{
	Mat dstimg;
	resize(srcimg, dstimg, Size(this->inpWidth, this->inpHeight));
	this->input_image_.resize(this->inpWidth * this->inpHeight * dstimg.channels());
	for (int c = 0; c < 3; c++)
	{
		for (int i = 0; i < this->inpHeight; i++)
		{
			for (int j = 0; j < this->inpWidth; j++)
			{
				float pix = dstimg.ptr<uchar>(i)[j * 3 + 2 - c];
				this->input_image_[c * this->inpHeight * this->inpWidth + i * this->inpWidth + j] = (pix /255.0 - mean[c]) / stds[c];
			}
		}
	}
	array<int64_t, 4> input_shape_{ 1, 3, this->inpHeight, this->inpWidth };

	auto allocator_info = MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
	Value input_tensor_ = Value::CreateTensor<float>(allocator_info, input_image_.data(), input_image_.size(), input_shape_.data(), input_shape_.size());

	// 开始推理
	vector<Value> ort_outputs = ort_session->Run(RunOptions{ nullptr }, &input_names[0], &input_tensor_, 1, output_names.data(), output_names.size());   // 开始推理
	float* pred = ort_outputs[0].GetTensorMutableData<float>();
	Mat result(outHeight, outWidth, CV_32FC1, pred);
	result = 1 - result;
	double min_value, max_value;
	minMaxLoc(result, &min_value, &max_value, 0, 0);
	result = (result - min_value) / (max_value - min_value);
	result *= 255;
	result.convertTo(result, CV_8UC1);
	return result;
}

int main()
{
	U2Net mynet;
	string imgpath = "sample.jpg";
	Mat srcimg = imread(imgpath);
	Mat result = mynet.detect(srcimg);
	resize(result, result, Size(srcimg.cols, srcimg.rows));

	namedWindow("srcimg", WINDOW_NORMAL);
	imshow("srcimg", srcimg);
	static const string kWinName = "Deep learning object detection in ONNXRuntime";
	namedWindow(kWinName, WINDOW_NORMAL);
	imshow(kWinName, result);
	waitKey(0);
	destroyAllWindows();
}