
#include "yolov5ncnn.h"

int main() {
    std::string paramPath = "./sources/yolov5n.ncnn.param";
    std::string modelPath = "./sources/yolov5n.ncnn.bin";

    // char parampath[256];
    // char modelpath[256];
    // sprintf_s(parampath, "./sources/yolov5n.ncnn.param");
    //  sprintf_s(modelpath, "./sources/yolov5n.ncnn.bin");

    std::vector<std::vector<float>> anchors = {
     { 10.f, 13.f, 16.f, 30.f, 33.f, 23.f },
     { 30.f, 61.f, 62.f, 45.f, 59.f, 119.f },
     { 116.f, 90.f, 156.f, 198.f, 373.f, 326.f }
    };

    YoloV5Ncnn model(paramPath.c_str(), modelPath.c_str(), 640, 0.15, 0.45, anchors);
    int32_t ret;
    ret = model.loadModel();
    if (ret != 0)
    {
        fprintf(stderr, "load model failed\n");
        return -1;
    }
    const char* imagepath = "test.jpg";

    cv::Mat img = cv::imread(imagepath, 1);
    if (img.empty())
    {
        fprintf(stderr, "cv::imread %s failed\n", imagepath);
        return -1;
    }

    std::vector<Object> results = model.detectImage(img);

    draw_objects(img, results);
}
