#include "ai_model.h"

#include <string>
#include <unistd.h>
#include <mutex>

using namespace facethink;

static ImageSharpClassify *g_image_clarity{nullptr};
static std::mutex g_model_lock;

bool InitModel() {
    char curr_path[1024*2];
    if (!getcwd(curr_path, sizeof(curr_path)-1)) {
        return false;
    }
    std::string model_trt{curr_path};
    model_trt += "/../ai_model/model/cls_image_sharp_v1.0.0.trt";
    std::string model_config{curr_path};
    model_config += "/../ai_model/model/config.ini";
    g_image_clarity = ImageSharpClassify::create(model_trt, model_config);
    return g_image_clarity;
}

int DetectImageClarity(const cv::Mat &img, 
                       int &clarity, 
                       float &confidence) {
    std::unique_lock<std::mutex> lock_guard{g_model_lock};
    int ret = g_image_clarity->classify(img, clarity, confidence);
    return ret;
}
