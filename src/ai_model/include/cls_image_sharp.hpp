///////////////////////////////////////////////////////////////////////////////////////
///  Copyright (C) 2017, TAL AILab Corporation, all rights reserved.
///
///  @file: cls_iamge_sharp.hpp
///  @brief 图片清晰度
///  @details 最初版本
//
//
///  @version 1.0.0.0
///  @author Yu Qiao
///  @date 2020-03-17
///
///  @see 使用参考：demo.cpp
///
///////////////////////////////////////////////////////////////////////////////////////
#ifndef __FACETHINK_API_CLS_IMAGE_SHARP_HPP__
#define __FACETHINK_API_CLS_IMAGE_SHARP_HPP__
#include <string>
#include <math.h>
#include <opencv2/opencv.hpp>
#ifdef WIN32
#ifdef DLL_EXPORTS
#define EXPORT_CLASS   __declspec(dllexport)
#define EXPORT_API  extern "C" __declspec(dllexport)
#define EXPORT_CLASS_API
#else
#define EXPORT_CLASS   __declspec(dllimport )
#define EXPORT_API  extern "C" __declspec(dllimport )
#endif
#else
#define EXPORT_CLASS
#define EXPORT_API  extern "C" __attribute__((visibility("default")))
#define EXPORT_CLASS_API __attribute__((visibility("default")))
#endif
namespace facethink {
	class EXPORT_CLASS ImageSharpClassify {
	public:
		EXPORT_CLASS_API explicit ImageSharpClassify(void);
		EXPORT_CLASS_API virtual ~ImageSharpClassify(void);
		
		/// \brief SDK初始化函数，必须先于任何其他SDK函数之前调用，create的重载函数。
		/// @param [in] det_model_file 指定SDK对应的模型文件路径。
		/// @param [in] config_file 指定SDK对应的参数配置文件路径,详情见config.ini文件。
		/// @return
		/// @remarks 初始化函数需要读取模型等文件，需要一定时间等待。
		EXPORT_CLASS_API static ImageSharpClassify* create(
			const std::string& det_model_file,
			const std::string& config_file);

		/// \brief 检测图片清晰度。
		/// @param [in] img 输入的图像数据，支持如下两种种种格式:
		/// - 1.BGR图：img为一维数组，每个元素（字节）表示一个像素点的单通道取值，三个连续元素表示一个像素点的三通道取值，顺序为BGR。
		/// - 2.RGB图：此时is_rgb_format应设置为true。
		/// @param [in] is_rgb_format，默认值为false时，表示图片格式为BGR。
		/// @param [out] results，0代表模糊，1代表清晰
		/// @params [out] confidence, 代表类别概率
		/// @return
		/// @retval 0 检测成功。
		/// @retval -1 图片为空或图片通道不为3或输入人体框为空
		/// @retval -2 模型运行出错。
		/// @retval -5 SDK本地鉴权失败
		/// @retval -6 数据上传模块参数配置错误
		EXPORT_CLASS_API virtual int classify(const cv::Mat& img, int& distinct, float& confidence, bool is_rgb_format = false) = 0;

	};
}
#endif