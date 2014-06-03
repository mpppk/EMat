#include "RVec.h"

using namespace std;

namespace mc{
	bool RVec::isValid() const{
		if(mat_.rows != 1)	return false;
		return true;
	}

	RVec::RVec(const cv::Mat& mat) : mat_(mat){
		if(!isValid())	throw invalid_argument("mat is not vector!");
		mat_ = mat;
	}

	RVec::RVec(const vector<string>& content) : mat_(toMat(content)){
		if(!isValid())	throw invalid_argument("mat is not vector!");
	}

	RVec& RVec::operator=(const cv::Mat& mat){
		if(!isValid())	throw invalid_argument("mat is not vector!");
		mat_ = mat;
		return *this;
	}

	double& RVec::operator[](unsigned int index){
		return mat_.at<double>(0, index);
	}

	cv::Mat& RVec::m(){return mat_;}

	cv::Mat RVec::toMat(const vector<string>& content){
		cv::Mat mat(1, content.size(), CV_64F);
		for(int i = 0; i < content.size(); i++){
			mat.at<double>(0, i) = boost::lexical_cast<double>(content[i]);
		}
		return mat;
	}

}