#include <MatU.h>

using namespace std;

namespace mc{

	vector< vector<string> > MatU::toVec(const cv::Mat& mat){// static
		vector< vector<string> > vecs;
		for (int row_i = 0; row_i < mat.rows; ++row_i){
			vector<string> v;
			for(int col_i = 0; col_i < mat.cols; ++col_i){
				string str = boost::lexical_cast<string>(mat.at<double>(row_i, col_i)); 
				v.push_back(str);
			}
			vecs.push_back(v);
		}
		return vecs;
	}

	vector< vector< vector<string> > > MatU::toVec(const vector<cv::Mat>& mats){// static
		if(mats.size() == 0)	throw invalid_argument("mats num is zero");

		vector< vector< vector<string> > > retVec;
		for (int i = 0; i < mats.size(); ++i)	retVec.push_back(toVec(mats[i]));
		return retVec;
	}

	cv::Mat MatU::toMat(const vector<string>& content){
		return toRVec(content).m();
	}
	RVec MatU::toRVec(const vector<string>& content){
		RVec vec(content.size());
		for(int i = 0; i < content.size(); i++){
			vec[i] = boost::lexical_cast<double>(content[i]);
		}
		return vec;

	}

	cv::Mat MatU::toMat(const vector< vector<string> >& contents){// static
		cv::Mat mat(contents.size(), contents[0].size(), CV_64F);
		// ----------ここからcontentsの要素を代入する処理----------
		for (int i = 0; i < contents.size(); ++i){// 行のループ
			vector<string> row = contents[i];// 現在の行をrowに代入
			for (int j = 0; j < contents[0].size(); ++j){// 列のループ
				mat.at<double>(i, j) = boost::lexical_cast<double>(row[j]);
			}
		}// ----------ここまでcontentsの要素を代入する処理----------
		return mat;
	}

	//srcの指定した行からdstの指定した行へをコピーする
	void MatU::copyRow(const cv::Mat &src, cv::Mat &dst, const unsigned int srcRowIndex, const unsigned int dstRowIndex){
		// srcとcolの列数が同じかどうか
		if (src.cols != dst.cols){
			ostringstream os;
			os << "Mat's colmun is different [in mc::copyRow]" << endl
				<< "src.cols is (" << src.cols << ")" << endl
				<< "dst.cols is (" << dst.cols << ")" << endl;
			throw invalid_argument(os.str());
		}

		// 指定されたsrcの行は存在するか
		if(srcRowIndex >= src.rows){
			ostringstream os;
			os << "srcRowIndex is invalid value.(" << srcRowIndex << ") [in mc::copyRow]" << endl
				<< "src.rows is (" << src.rows << ")" << endl;
			throw invalid_argument(os.str());
		}

		// 指定されたdstの行は存在するか
		if(dstRowIndex >= dst.rows){
			ostringstream os;
			os << "dstRowIndex is invalid value.(" << dstRowIndex << ") [in mc::copyRow]" << endl
				<< "dst.rows is (" << dst.rows << ")" << endl;
			throw invalid_argument(os.str());
		}

		for(int i = 0; i < src.cols; i++)
			dst.at<double>(dstRowIndex, i) = src.at<double>(srcRowIndex, i);
	}

	//指定した行をsrcからdstへコピーする
	void MatU::copyRow(const cv::Mat &src, cv::Mat &dst, const unsigned int row){
		copyRow(src, dst, row, row);
	}

	void MatU::copyCol(const cv::Mat &src, cv::Mat &dst, const unsigned int srcColIndex, const unsigned int dstColIndex){
		// srcとcolの列数が同じかどうか
		if (src.rows != dst.rows){
			ostringstream os;
			os << "Mat's rows is different [in mc::copyCol]" << endl
				<< "src.rows is (" << src.rows << ")" << endl
				<< "dst.rows is (" << dst.rows << ")" << endl;
			throw invalid_argument(os.str());
		}

		// 指定されたsrcの行は存在するか
		if(srcColIndex >= src.cols){
			ostringstream os;
			os << "srcColIndex is invalid value.(" << srcColIndex << ") [in mc::copyCol]" << endl
				<< "src.cols is (" << src.cols << ")" << endl;
			throw invalid_argument(os.str());
		}

		// 指定されたdstの行は存在するか
		if(dstColIndex >= dst.cols){
			ostringstream os;
			os << "dstColIndex is invalid value.(" << dstColIndex << ") [in mc::copyCol]" << endl
				<< "dst.cols is (" << dst.cols << ")" << endl;
			throw invalid_argument(os.str());
		}

		for(int i = 0; i < src.rows; i++)
			dst.at<double>(i, dstColIndex) = src.at<double>(i, srcColIndex);
	}

	void MatU::copyCol(const cv::Mat &src, cv::Mat &dst, const unsigned int col){
		copyCol(src, dst, col, col);
	}

	// vec1の後ろにvec2を連結する(vec1,vec2はそれぞれ横ベクトル)
	cv::Mat MatU::mergeRVec(const cv::Mat &vec1, const cv::Mat &vec2){
		return mergeRVec(RVec(vec1), RVec(vec2)).m();
	}
	mc::RVec MatU::mergeRVec(const RVec &vec1, const RVec &vec2){
		RVec retVec(vec1.size() + vec2.size());
		for(int i = 0; i < vec1.size(); i++)	retVec[i] = vec1[i];
		for(int i = 0; i < vec2.size(); i++)	retVec[ vec1.size() + i ] = vec2[i];
		return retVec;
	}

	// vector内の横ベクトルをすべて連結する
	cv::Mat MatU::mergeRVec(const vector<cv::Mat> &vecs){
		vector<RVec> rvecs = RVec::cast(vecs);
		return mergeRVec(rvecs).m();
	}

	RVec MatU::mergeRVec(const vector<RVec> &vecs){
		int totalSize = 0;
		for(RVec vec : vecs){ totalSize += vec.size(); }
		RVec retVec(totalSize);
		int index = 0;
		for (RVec vec : vecs){
			for (int i = 0; i < vec.size(); ++i){
				retVec[index] = vec[i];
				index++;
			}
		}
		return retVec;
	}

	cv::Mat MatU::mergeRVec(const cv::Mat &tempVecs){
		cv::Mat retMat(1, tempVecs.rows * tempVecs.cols, CV_64F);

		for (int row_i = 0; row_i < tempVecs.rows; ++row_i){
			cv::Mat tempVec = tempVecs.row(row_i).clone();
			for (int col_i = 0; col_i < tempVec.cols; ++col_i){
				retMat.at<double>(0, row_i * tempVec.cols + col_i) = tempVec.at<double>(0, col_i);
			}
		}

		return retMat;
	}

	// mat1の右側にmat2を連結する
	// マージの結果、埋まらない要素がある場合はfillBlankNumの値で埋める
	cv::Mat MatU::mergeMatToSide(const cv::Mat &mat1, const cv::Mat &mat2, int fillBlankNum){
		int biggerRows = mat1.rows > mat2.rows ? mat1.rows : mat2.rows;
		cv::Mat mergedMat = cv::Mat::ones(biggerRows, mat1.cols + mat2.cols, CV_64F) * fillBlankNum;
		cv::Mat r1 = mergedMat(cv::Rect(0, 0, mat1.cols, mat1.rows));
		cv::Mat r2 = mergedMat(cv::Rect(mat1.cols, 0, mat2.cols, mat2.rows));

		for (int row_i = 0; row_i < mat1.rows; ++row_i)	copyRow(mat1, r1, row_i);
		for (int row_i = 0; row_i < mat2.rows; ++row_i)	copyRow(mat2, r2, row_i);
		return mergedMat;
	}

	// mat1の下側にmat2を連結する
	cv::Mat MatU::mergeMatToBottom(const cv::Mat &mat1, const cv::Mat &mat2, int fillBlankNum){
		int biggerCols = mat1.cols > mat2.cols ? mat1.cols : mat2.cols;
		cv::Mat mergedMat = cv::Mat::ones(mat1.rows + mat2.rows, biggerCols, CV_64F) * fillBlankNum;
		cv::Mat r1 = mergedMat(cv::Rect(0, 0, mat1.cols, mat1.rows));
		cv::Mat r2 = mergedMat(cv::Rect(0, mat1.rows, mat2.cols, mat2.rows));

		for (int row_i = 0; row_i < mat1.rows; ++row_i)	copyRow(mat1, r1, row_i);
		for (int row_i = 0; row_i < mat2.rows; ++row_i)	copyRow(mat2, r2, row_i);
		return mergedMat;
	}

	cv::Mat MatU::mergeMatToBottom(const vector<cv::Mat> mats, int fillBlankNum){
		cv::Mat mergedMat;
		if(mats.size() == 1){
			cout << "warning:" << endl << "mats's element number is only one! in mc::mergeMatToBottom" << endl;
			return mats[0];
		}

		if(mats.size() <= 0){
			throw invalid_argument("invalid value is passed. (in mc::mergedMatToBottom)");
		}

		mergedMat = MatU::mergeMatToBottom(mats[0], mats[1], fillBlankNum);
		for (int i = 2; i < mats.size(); ++i){
			mergedMat = MatU::mergeMatToBottom(mergedMat, mats[i], fillBlankNum);
		}

		return mergedMat;
	}
}

