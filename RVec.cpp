#include "RVec.h"

using namespace std;

namespace mc{
	bool RVec::isValid() const{
		if(this->rows != 1)	return false;
		return true;
	}

	bool RVec::isValid(cv::Mat mat) const{
		if(this->rows != 1)	return false;
		return true;
	}

	RVec& RVec::operator=(cv::Mat mat){
		if(mat.rows != 1)	throw invalid_argument("argument can not pass to RVec");
		this->create(1, mat.cols, CV_64F);
		*this = mat.clone();
		// for(int i = 0; i < mat.cols; i++){
		// 	// this->at<double>(0, i) =  mat.at<double>(0, i);
		// }
		return *this;
	}

	double& RVec::operator[](unsigned int index){
		return this->at<double>(0, index);
	}

	RVec::RVec(){}

	RVec::RVec(int _rows, int _cols, int _type) : EMat(_rows, _cols, _type){
		if(!isValid())	throw invalid_argument("argument can not pass to RVec");
	}

	RVec::RVec(const cv::Mat mat) : EMat(mat){
		if(!isValid())	throw invalid_argument("argument can not pass to RVec");
	}

	RVec::RVec(const vector< vector<string> > contents) : EMat(contents){
		if(!isValid())	throw invalid_argument("argument can not pass to RVec");
	}

	RVec RVec::cast(const cv::Mat mat){
		RVec rvec(mat);
		return rvec;
	}
}