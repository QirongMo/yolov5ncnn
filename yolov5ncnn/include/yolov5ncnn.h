#pragma once


#include<iostream>
#include "utils.h"


class YoloV5Ncnn {
public:
    YoloV5Ncnn(const char* paramPath, const char* modelPath, int targetSize, float confThreshold,
        float nmsThreshold, const std::vector<std::vector<float>> anchors);

    int32_t loadModel();

    std::vector<Object> detectImage(const cv::Mat& bgr);

    void releaseModel();

    ~YoloV5Ncnn();

private:
    bool isLoad;
    ncnn::Net net;
    std::vector<const char*>input_names;
    std::vector<const char*>output_names;

private:
    const char* paramPath;
    const char* modelPath;
    const std::vector<std::vector<float>> anchors;
    int targetSize;
    float confThreshold;
    float nmsThreshold;

};