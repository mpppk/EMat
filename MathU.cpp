// ユーザー定義
#include <MathU.h>

using namespace std;

namespace mc{

	// L1normを計算する
	double MathU::calcL1norm(const cv::Mat &vec){
		return calcL1norm(RVec(vec));
	}

	double MathU::calcL1norm(const RVec &vec){
		double L1norm = 0;
		for(int i = 0; i < vec.size(); i++){ L1norm += fabs(vec[i]); }
		return L1norm;
	}

	// L1normは絶対値をとるが、この関数はとらない
	double MathU::calcTotalSum(const cv::Mat &vec){
		return calcTotalSum(RVec(vec));
	}

	double MathU::calcTotalSum(const RVec &vec){
		double sum = 0;
		for(int i = 0; i < vec.size(); i++){ sum += vec[i]; }
		return sum;
	}

	double MathU::calcL2norm(const cv::Mat &vec){
		return calcL2norm(RVec(vec));
	}

	double MathU::calcL2norm(const RVec &vec){
		double L2norm = 0;
		for(int i = 0; i < vec.size(); i++){ L2norm += pow(vec[i], 2); }
		return sqrt(L2norm);
	}

	// 重み付の共分散を計算する
	double MathU::calcWCov(const cv::Mat &vec1, const cv::Mat &vec2,
		double mean1, double mean2, const cv::Mat &vec_weight){
		return calcWCov(RVec(vec1), RVec(vec2), mean1, mean2, RVec(vec_weight));
	}

	double MathU::calcWCov(const RVec &vec1, const RVec &vec2,
		double mean1, double mean2, const RVec &vec_weight){
		if(vec1.size() != vec2.size()){
			ostringstream os;
			os << "[in calcWCov] vec1's col num is " << vec1.size() << "," << endl
			<< "but vec2's col num is " << vec2.size() << "." << endl;
			throw runtime_error(os.str());
		}

		if(vec1.size() != vec_weight.size()){
			ostringstream os;
			os << "[in calcWCov] vec1's col num is " << vec1.size() << "," << endl
			<< "but weight's col num is " << vec2.size() << "." << endl;
			throw runtime_error(os.str());
		}

		RVec vec1SubedMean = vec1.m().clone();
		RVec vec2SubedMean = vec2.m().clone();

		// データから平均を引いた行列を計算
		vec1SubedMean -= mean1;
		vec2SubedMean -= mean2;

		return vec1SubedMean.m().dot( vec2SubedMean.m().mul(vec_weight.m()) );
	}

	// 重み付の共分散行列を計算する
	// dataMat 1行に一つの事例を格納
	cv::Mat MathU::calcWCovMat(const cv::Mat &dataMat, const cv::Mat &vec_mean, const cv::Mat &vec_weight){
		return calcWCovMat(dataMat, RVec(vec_mean), RVec(vec_weight));
	}

	cv::Mat MathU::calcWCovMat(const cv::Mat &dataMat, const RVec &vec_mean, const RVec &vec_weight){
		if(dataMat.cols != vec_mean.size()){
			ostringstream os;
			os << "[in calcWCovMat] dataMat's col num is " << dataMat.cols << "," << endl
			<< "but mean's col num is " << vec_mean.size() << "." << endl;
			throw runtime_error(os.str());
		}

		if(dataMat.rows != vec_weight.size()){
			ostringstream os;
			os << "[in calcWCovMat] dataMat's row num is " << dataMat.rows << "," << endl
			<< "but weight's col num is " << vec_weight.size() << "." << endl;
			throw runtime_error(os.str());
		}
		// ----------エラーチェックここまで----------

		cv::Mat covMat(dataMat.cols, dataMat.cols, CV_64F);
		cv::Mat dataMatSubMean = dataMat.clone();

		// 重みなしの共分散行列が正しく推定できているかの確認用
		RVec vec_noWeight = cv::Mat::ones(1, dataMat.rows, CV_64F);

		//共分散行列の各要素を計算
		for(int covMatRow_i = 0; covMatRow_i < covMat.rows; covMatRow_i++){
			for(int covMatCol_i = 0; covMatCol_i < covMat.cols; covMatCol_i++){
				covMat.at<double>(covMatRow_i, covMatCol_i) = 
				calcWCov(RVec( dataMat.col(covMatRow_i).t() ) ,
					RVec( dataMat.col(covMatCol_i).t() ),
					vec_mean[covMatRow_i],
					vec_mean[covMatCol_i],
					vec_weight);
			}
		}
		return covMat;
	}

	cv::Mat MathU::calcWCovMat(const cv::Mat &dataMat, const cv::Mat &vec_weight){
		return MathU::calcWCovMat(dataMat, RVec(vec_weight));
	}

	cv::Mat MathU::calcWCovMat(const cv::Mat &dataMat, const RVec &vec_weight){
		RVec vec_dataMean = cv::Mat::Mat::zeros(1, dataMat.cols, CV_64F);
		// 重みを考慮した平均を計算する
		for (int col_i = 0; col_i < dataMat.cols; ++col_i){
			for (int row_i = 0; row_i < dataMat.rows; ++row_i){
				vec_dataMean[col_i] += dataMat.at<double>(row_i, col_i) * vec_weight[row_i];
			}
		}
		vec_dataMean /= calcTotalSum(vec_weight);
		// cout << "dbg in calcWCovMat mean(正しいことを確認済み): " << vec_dataMean << endl;
		return calcWCovMat(dataMat, vec_dataMean, vec_weight);
	}

	// オプションを指定できるcalcWCovMat
	cv::Mat MathU::calcWCovMat(const cv::Mat &dataMat, const cv::Mat &vec_weight, const bitset<CovMatOptionsNum> flags){
		return calcWCovMat(dataMat, vec_weight, flags);
	}

	// オプションを指定できるcalcWCovMat
	cv::Mat MathU::calcWCovMat(const cv::Mat &dataMat, const RVec &vec_weight, const bitset<CovMatOptionsNum> flags){
		cv::Mat result = calcWCovMat(dataMat, vec_weight);
		if(flags[SCALE])	result /= calcTotalSum(vec_weight);
		return result;
	}

	// opencv2.0用
	double MathU::rbfKernel(const cv::Mat &vec1, const cv::Mat &vec2, const double rbfSigma){
		return rbfKernel(RVec(vec1), RVec(vec2), rbfSigma);
	}
	
	double MathU::rbfKernel(const RVec &vec1, const RVec &vec2, const double rbfSigma){
		RVec tempMat = vec1.m() - vec2.m();
		return exp(-1 * tempMat.m().dot(tempMat.m()) / rbfSigma);
	}

	// 座標(x,y), 長辺と短辺が(a,b),rot分回転した楕円を描画のための1度ごとの座標を返す
	cv::Mat MathU::plotEllipse(double x,double y,double a,double b,int rot){
		FILE *fp;
		double theta;
		unsigned int rotNum = 360;
		cv::Mat retMat = cv::Mat::ones(rotNum, 2, CV_64F) * -999;
		// cv::Mat mat1 = cv::Mat::ones(5, 5, CV_8U)*3;
		cv::Mat tempMat(1, 2, CV_64F);
		for(int i = 0; i < 360; i++){
			theta = 2*M_PI/360;

			retMat.at<double>(i,0) = a * cos((double)theta*i) * cos((double)theta*rot) - b * sin((double)theta*i) * sin((double)theta*rot) + x;
			retMat.at<double>(i,1)  = a * cos((double)theta*i) * sin((double)theta*rot) + b * sin((double)theta*i) * cos((double)theta*rot) + y;
			// retMat.at<double>(i,0) = a * cos((double)theta*i) - b * sin((double)theta*i) + x;
			// retMat.at<double>(i,1)  = a * cos((double)theta*i) + b * sin((double)theta*i) + y;
			// fprintf(fp,"%.3lf",a * cos((double)theta*i) * cos((double)theta*rot) - b * sin((double)theta*i) * sin((double)theta*rot) + x);
			// fprintf(fp,",%.3lf\n",a * cos((double)theta*i) * sin((double)theta*rot) + b * sin((double)theta*i) * cos((double)theta*rot) + y);
		}
		return retMat;
	}

	// 共分散行列から、確立楕円描画のための1度ごとの座標を返す
	// meanは2値でなければいけない
	cv::Mat MathU::plotConfidenceEllipse(const cv::Mat &mean, const cv::Mat &covMat, const int sigma){
		return plotConfidenceEllipse(RVec(mean), covMat, sigma);
	}

	cv::Mat MathU::plotConfidenceEllipse(const RVec &mean, const cv::Mat &covMat, const int sigma){
		cv::Mat eigenValues, eigenVectors;
		cv::eigen(covMat, eigenValues, eigenVectors);
		double degree = atan2(eigenVectors.at<double>(1,0), eigenVectors.at<double>(0,0)) * 180.0 / M_PI;
		// int degree = 45;
		return plotEllipse(
			mean[0],
			mean[1],
			sigma * eigenValues.at<double>(0,0),
			sigma * eigenValues.at<double>(1,0),
			(int)fabs(degree));
		// TODO: degreeがこれでいいのか考える
	}

	map<string, cv::Mat> MathU::normalize(const cv::Mat &data, const cv::Mat &mean, const cv::Mat &sd){
		return normalize(data, RVec(mean), RVec(sd));
	}

	map<string, cv::Mat> MathU::normalize(const cv::Mat &data, const RVec &mean, const RVec &sd){
		// 行列を正規化
		cv::Mat normalizedMat = data.clone();

		for (int row_i = 0; row_i < data.rows; ++row_i){
			for (int col_i = 0; col_i < data.cols; ++col_i){
				normalizedMat.at<double>(row_i, col_i) -= mean[col_i];
				normalizedMat.at<double>(row_i, col_i) /= sd[col_i];
			}
		}

		// 返り値
		map<string, cv::Mat> retMats;
		retMats.insert( map<string, cv::Mat>::value_type( "mean", mean.m() ) );
		retMats.insert( map<string, cv::Mat>::value_type( "sd", sd.m() ) );
		retMats.insert( map<string, cv::Mat>::value_type( "normalizedMat", normalizedMat ) );
		return retMats;
	}

	map<string, cv::Mat> MathU::normalize(const cv::Mat &data){
		RVec mean(data.cols);
		RVec squareMean(data.cols);
		RVec sum = cv::Mat::zeros(1, data.cols, CV_64F);
		RVec squareSum = cv::Mat::zeros(1, data.cols, CV_64F);
		RVec variance(data.cols);
		RVec sd(data.cols);
		cv::Mat normalizedMat = data.clone();

		// 各要素の合計とその二乗を計算
		for (int row_i = 0; row_i < data.rows; ++row_i){
			for (int col_i = 0; col_i < data.cols; ++col_i){
				sum[col_i] += data.at<double>(row_i, col_i);
				squareSum[col_i] += pow(data.at<double>(row_i, col_i), 2);
			}
		}

		// 平均と標準偏差の計算
		// 分散 = ２乗の平均　－　平均の２乗
		for (int col_i = 0; col_i < data.cols; ++col_i){
			mean[col_i] = sum[col_i] / data.rows;
			squareMean[col_i] = squareSum[col_i] / data.rows;
			variance[col_i] = squareMean[col_i] - pow(mean[col_i], 2);
			sd[col_i] = sqrt(variance[col_i]);
		}

		// 行列を正規化
		for (int row_i = 0; row_i < data.rows; ++row_i){
			for (int col_i = 0; col_i < data.cols; ++col_i){
				normalizedMat.at<double>(row_i, col_i) -= mean[col_i];
				normalizedMat.at<double>(row_i, col_i) /= sd[col_i];				
			}
		}

		// 返り値
		map<string, cv::Mat> retMats;
		retMats.insert( map<string, cv::Mat>::value_type( "mean", mean.m() ) );
		retMats.insert( map<string, cv::Mat>::value_type( "variance", variance.m() ) );
		retMats.insert( map<string, cv::Mat>::value_type( "sd", sd.m() ) );
		retMats.insert( map<string, cv::Mat>::value_type( "normalizedMat", normalizedMat ) );
		return retMats;
	}

	map<string, cv::Mat> MathU::unnormalize(const cv::Mat &data, const cv::Mat &mean, const cv::Mat &sd){
		return unnormalize(data, RVec(mean), RVec(sd));
	}

	map<string, cv::Mat> MathU::unnormalize(const cv::Mat &data, const RVec &mean, const RVec &sd){
		cv::Mat unnormalizedMat = data.clone();

		for (int row_i = 0; row_i < data.rows; ++row_i){
			for (int col_i = 0; col_i < data.cols; ++col_i){
				unnormalizedMat.at<double>(row_i, col_i) *= sd[col_i];
				unnormalizedMat.at<double>(row_i, col_i) += mean[col_i];				
			}
		}

		// 返り値
		map<string, cv::Mat> retMats;
		retMats.insert( map<string, cv::Mat>::value_type( "mean", mean.m() ) );
		retMats.insert( map<string, cv::Mat>::value_type( "sd", sd.m() ) );
		retMats.insert( map<string, cv::Mat>::value_type( "unnormalizedMat", unnormalizedMat ) );
		return retMats;
	}

	cv::Mat MathU::normalizeHistogram(const cv::Mat &src){
		cv::Mat dst;
		// minとmaxを決定する
		double minVal, maxVal;
		cv::Point minLoc, maxLoc;
		cv::minMaxLoc(src, &minVal, &maxVal, &minLoc, &maxLoc);

		cv::convertScaleAbs(src, dst, 255.0/(maxVal-minVal), (255.0/(maxVal-minVal))*(-minVal));
		return dst;
	}

	double MathU::calcEuclideanDist(const cv::Mat &point1, const cv::Mat &point2){
		return calcEuclideanDist(RVec(point1), RVec(point2));
	}

	double MathU::calcEuclideanDist(const RVec &point1, const RVec &point2){
		// ベクトルの行数が違う場合は例外
		if(point1.size() != point2.size()){
			throw invalid_argument("in calcEuclideanDist invalid value is passed. vector nums are different.");
		}

		double dist = 0;
		for (int i = 0; i < point1.size(); ++i){
			dist += pow( point1[i] - point2[i], 2 );
		}

		return sqrt(dist);
	}
}


