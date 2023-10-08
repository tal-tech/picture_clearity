#include "ai_model.h"
#include <iostream>
using std::cout;
using std::endl;

int main()
{
    InitModel();

    // 视频文件路径
    std::string videoFilePath = "../det_face_id/video/824770142-1-16.mp4";
    // 输出图片目录
    std::string outputImageDir = "../det_face_id/images/testing/";

    cv::VideoCapture cap(videoFilePath); 

    if(!cap.isOpened())
    {
        std::cout << "Error opening video file" << std::endl;
        return -1;
    }

    int frameNum = 0; 

    cv::Mat frame;

    while(1)
    {
        cap >> frame; 
        if(frame.empty())
            break;
        
        int clarity = 0;
        float confidence = 0.0f;
        int ret = DetectImageClarity(frame,clarity,confidence);
        if (ret != 0) {
            cout << "Error" << endl;
        }
        cout << "clarity:" << clarity << endl;
        cout << "confidence:" << confidence << endl;
    }
    cap.release(); 

    
    return 0;
}