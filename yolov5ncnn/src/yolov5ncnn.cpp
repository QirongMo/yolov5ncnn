

#include "yolov5ncnn.h"


YoloV5Ncnn::YoloV5Ncnn(const char* paramPath, const char* modelPath, int targetSize, float confThreshold,
    float nmsThreshold, const std::vector<std::vector<float>> anchors):paramPath(paramPath), modelPath(modelPath), targetSize(targetSize),
    confThreshold(confThreshold), nmsThreshold(nmsThreshold), anchors(anchors)
{
    net.opt.use_vulkan_compute = true;

    // net.opt.use_bf16_storage = true;
    // net.opt.use_fp16_arithmetic = true;
    // net.opt.use_fp16_packed = true;

    isLoad = false;
}

int32_t YoloV5Ncnn::loadModel()
{
    if (isLoad) return 0;
    if (net.load_param(paramPath))
        return 1;
    if (net.load_model(modelPath))
        return 2;
    isLoad = true;
    input_names = net.input_names();
    output_names = net.output_names();  // 输出节点名称
    return 0;
}

std::vector<Object> YoloV5Ncnn::detectImage(const cv::Mat& bgr)
{
        
    std::vector<Object> results;
    if (!isLoad) return results;

    ImgData img;
    img.orgImg = bgr;

    int img_w = bgr.cols;
    int img_h = bgr.rows;
    img.imgW = img_w;
    img.imgH = img_h;

    // 预处理
    preprocess(img, targetSize);

    // 创建extractor
    ncnn::Extractor ex = net.create_extractor();
       
    // 输入数据
    ex.input(input_names[0], img.inputImg);
    // std::cout << "input shape: (" << img.inputImg.w << ", " << img.inputImg.h << ", " << img.inputImg.c << ")\n";
    // anchor setting from yolov5/models/yolov5s.yaml
    // 输出，生成proposals
    std::vector<Object> proposals;
    for (int i = 0; i < output_names.size(); i++) {
        ncnn::Mat out;
        ex.extract(output_names[i], out);
        if(out.d > 1)
            out = out.reshape(out.w, out.h, 1, out.c * out.d);
        // std::cout << "out shape: (" << out.w << ", " << out.h << ", " << out.c << ", " << out.d << ")\n";
        const int num_grid_x = out.w;
        int stride = targetSize / num_grid_x;
        // std::cout << "stride: " << stride << std::endl;
        generate_proposals(anchors[i], stride, out, confThreshold, proposals);
    }
    
    // sort all proposals by score from highest to lowest
    qsort_descent_inplace(proposals);

    // apply nms with nms_threshold
    std::vector<int> picked;
    nms_sorted_bboxes(proposals, picked, nmsThreshold);

    int count = picked.size();

    results.resize(count);
    for (int i = 0; i < count; i++)
    {
        results[i] = proposals[picked[i]];

        // adjust offset to original unpadded
        float x0 = (results[i].rect.x - (img.padW / 2.0)) / img.scaleX;
        float y0 = (results[i].rect.y - (img.padH / 2.0)) / img.scaleY;
        float x1 = (results[i].rect.x + results[i].rect.width - (img.padW / 2.0)) / img.scaleX;
        float y1 = (results[i].rect.y + results[i].rect.height - (img.padH / 2.0)) / img.scaleY;

        // clip
        x0 = std::max(std::min(x0, (float)(img_w - 1)), 0.f);
        y0 = std::max(std::min(y0, (float)(img_h - 1)), 0.f);
        x1 = std::max(std::min(x1, (float)(img_w - 1)), 0.f);
        y1 = std::max(std::min(y1, (float)(img_h - 1)), 0.f);

        results[i].rect.x = x0;
        results[i].rect.y = y0;
        results[i].rect.width = x1 - x0;
        results[i].rect.height = y1 - y0;
    }
    
    return results;

}

void YoloV5Ncnn::releaseModel()
{
    if (isLoad)
    {
        net.clear();
        isLoad = false;
    }  
}

YoloV5Ncnn::~YoloV5Ncnn()
{
    releaseModel();
}

