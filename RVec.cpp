#include <RVec.h>

using namespace std;

namespace mc{
	bool RVec::isValid() const{
		if(mat_.rows != 1)	return false;
		return true;
	}

	RVec::RVec(const unsigned int size) : mat_(cv::Mat(1, size, CV_64F)){
		if(!isValid())	throw invalid_argument("mat is not vector!");
		// mat_ = cv::Mat(0, size, CV_64F);
	}

	RVec::RVec(const cv::Mat& mat) : mat_(mat){
		if(!isValid())	throw invalid_argument("mat is not vector!");
	}

	RVec::RVec(const cv::MatExpr& mate) : mat_(mate){
		if(!isValid())	throw invalid_argument("mat is not vector!");
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
	
	double RVec::operator[](unsigned int index) const{
		return mat_.at<double>(0, index);
	}

	RVec& RVec::operator+(double n){
		mat_ += n;
		return *this;
	}

	RVec& RVec::operator+=(double n){
		mat_ += n;
		return *this;
	}

	RVec& RVec::operator-(double n){
		mat_ -= n;
		return *this;
	}

	RVec& RVec::operator-=(double n){
		mat_ -= n;
		return *this;
	}

	RVec& RVec::operator/(double n){
		mat_ /= n;
		return *this;
	}

	RVec& RVec::operator/=(double n){
		mat_ /= n;
		return *this;
	}

	RVec& RVec::operator*(double n){
		mat_ *= n;
		return *this;
	}

	RVec& RVec::operator*=(double n){
		mat_ *= n;
		return *this;
	}

	int RVec::size() const{ return mat_.cols; }

	cv::Mat& RVec::m(){return mat_;}
	const cv::Mat& RVec::m() const{return mat_;}

	vector<RVec> RVec::cast(vector<cv::Mat> mats){
		vector<RVec> vecs;
		for(cv::Mat mat : mats){
			vecs.push_back( RVec(mat) );
		}
		return vecs;
	}

	cv::Mat RVec::toMat(const vector<string>& content){
		cv::Mat mat(1, content.size(), CV_64F);
		for(int i = 0; i < content.size(); i++){
			mat.at<double>(0, i) = boost::lexical_cast<double>(content[i]);
		}
		return mat;
	}
}

