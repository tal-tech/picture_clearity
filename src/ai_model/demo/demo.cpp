#include "cls_image_sharp.hpp"
#include <iostream>
#include <fstream>
#include <chrono>
#include <string>
#include <iomanip>

#include <boost/log/core.hpp>
#include <boost/log/trivial.hpp>
#include <boost/log/expressions.hpp>
#include <boost/log/sinks/text_file_backend.hpp>
#include <boost/log/utility/setup/file.hpp>
#include <boost/log/utility/setup/common_attributes.hpp>
#include <boost/log/sources/severity_logger.hpp>
#include <boost/log/sources/record_ostream.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/ini_parser.hpp>
#include <boost/filesystem.hpp>
#include <boost/io/ios_state.hpp>
#include <boost/progress.hpp>
#include <boost/algorithm/string/classification.hpp>
#include <boost/algorithm/string/split.hpp>
#include <boost/lexical_cast.hpp>

using namespace facethink;
using namespace std;
using namespace cv;

void setupLog(std::string filename) {
	typedef boost::log::sinks::synchronous_sink< boost::log::sinks::text_file_backend > sink_t;
	boost::property_tree::ptree pt;
	boost::shared_ptr< sink_t > file_sink = boost::log::add_file_log
	(
		boost::log::keywords::auto_flush = true,
		boost::log::keywords::file_name = filename,
		boost::log::keywords::format = "[%TimeStamp%]: %Message%"
	);
	boost::log::add_common_attributes();
	int log_level = pt.get<int>("log_level", 2);
	boost::log::core::get()->set_filter(boost::log::trivial::severity >= log_level);
}

std::vector<std::string>getFilePath(std::string folder_path) {
	namespace fs = boost::filesystem;
	fs::directory_iterator end;
	int file_count = 0;
	std::vector<std::string>filePaths;
	for (fs::directory_iterator dir(folder_path); dir != end; dir++)
	{
		std::string fn = dir->path().string();
		//std::cout << fn << std::endl;
		filePaths.push_back(fn);
	}
	return filePaths;
}

int main(int argc, char *argv[]) {

	if (argc < 4) {
		std::cerr << "Usage: " << argv[0]
			<< " det_model"
			<< " image folder"
			<< " config file" << std::endl;
		return 1;
	}

	setupLog("cls_image_sharp_performance.log"); 
	const std::string det1_model = argv[1];
	const std::string images_folder = argv[2];
	const std::string config_file = argv[3];
	int repeat_count = argc > 4 ? std::stoi(argv[4]) : 1;
    std::vector<std::string> imgs_path = getFilePath(images_folder);


	std::cout << "running ... ,please wait" << std::endl;
    ImageSharpClassify *image_sharp_detector = ImageSharpClassify::create(
		det1_model,
		config_file);
	std::cout << "load over" << std::endl;
	for (int i = 0; repeat_count <= 0 || i < repeat_count; i++) {
		int count = -2;
		double cost_time_all = 0.0;
		for(int j = 0; j < imgs_path.size(); j++){
			std::string img_path = imgs_path[j];

			cv::Mat img = cv::imread(img_path);
			if (img.data == 0) {
				BOOST_LOG_TRIVIAL(error) << "read image failed:" << img_path;
				return -1;
		    }
			auto time_start = std::chrono::steady_clock::now();

			int distinct;
			float confidence;
			int ret = image_sharp_detector->classify(img, distinct, confidence,true);
			std::cout << distinct<<std::endl;
			std::cout << confidence<<std::endl;

			if (ret < 0) {
				BOOST_LOG_TRIVIAL(info) << "ImageSharpDetection Error: " << ret;
			}

			auto time_end = std::chrono::steady_clock::now();
			double cost_time = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count();
			count++;
			if (count > 0) { cost_time_all += cost_time; }
			std::stringstream info;
			info << cost_time << " ms" <<" "<< img_path <<" " << distinct <<" "<< confidence <<" ";

			BOOST_LOG_TRIVIAL(info) << info.str();

		}		
		BOOST_LOG_TRIVIAL(info) << " average time: " << cost_time_all / count << " ms";
	}
    return 0;
}
