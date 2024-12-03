

#pragma once
#ifndef UTILH
#define UTILH
#include "layer.h"
#include "net.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <float.h>
#include <stdio.h>
#include <vector>


struct Object
{
    cv::Rect_<float> rect;
    int label;
    float prob;
};

struct ImgData 
{
    cv::Mat orgImg;
    int imgW;
    int imgH;

    ncnn::Mat inputImg;
    float scaleX;
    float scaleY;
    int padW;
    int padH;
};

float intersection_area(const Object& a, const Object& b);

void qsort_descent_inplace(std::vector<Object>& faceobjects, int left, int right);

void qsort_descent_inplace(std::vector<Object>& faceobjects);

void nms_sorted_bboxes(const std::vector<Object>& faceobjects, std::vector<int>& picked, float nms_threshold, bool agnostic = false);

inline float sigmoid(float x);

void generate_proposals(const std::vector<float> anchors, int stride, const ncnn::Mat& feat_blob, 
    float prob_threshold, std::vector<Object>& objects);

void draw_objects(const cv::Mat& bgr, const std::vector<Object>& objects);

void preprocess(const cv::Mat& bgr, ncnn::Mat& in_pad, int& wpad, int& hpad, float& scale);

void preprocess(ImgData& img, int target_size);

#endif

