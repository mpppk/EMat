#include <gtest/gtest.h>
#include <EMat.h>

using namespace std;

inline int toInt(std::string s) {int v; std::istringstream sin(s);sin>>v;return v;}
template<class T> inline std::string toString(T x) {std::ostringstream sout;sout<<x;return sout.str();}

// startValueから連番の値を振った行列を返す
vector< vector<string> > getTempVec(int startValue = 1, int rowsNum = 3, int colsNum = 3){
	int value = startValue;
	vector< vector<string> > contents;

	for (int row_i = 0; row_i < rowsNum; ++row_i){
		vector<string> content;
		for (int i = 0; i < colsNum; ++i){
			content.push_back(toString<int>(value));
			value++;
		}
		contents.push_back(content);
	}

	return contents;
}

cv::Mat getTempMat(int cnt = 1, const int row = 3, const int col = 3){
	cv::Mat mat(row, col, CV_64F);
	for (int row_i = 0; row_i < mat.rows; ++row_i){
		for (int col_i = 0; col_i < mat.cols; ++col_i){
			mat.at<double>(row_i, col_i) = cnt;
			cnt++;
		}
	}
	return mat;
}

class EMatTest : public ::testing::Test{
protected:
	string fileDirPass;

	virtual void SetUp(){
		// ファイルの書きだし先
		fileDirPass = "../test/TestEMat/";
	}
};

TEST_F(EMatTest, constructorTest){
	EXPECT_EQ( mc::EMat( getTempMat() )(0, 0), 1 );
	EXPECT_EQ( mc::EMat( getTempVec() )(0, 0), 1 );
}

TEST_F(EMatTest, toVecTest){
	// toVec(), toVec(cv::Mat)
	mc::EMat emat(getTempVec());
	vector< vector<string> > vec = emat.toVec();
	
	EXPECT_EQ(1, toInt(vec[0][0]));
	EXPECT_EQ(9, toInt(vec[2][2]));

	// toVec(vector<cv::Mat>)
	vector<mc::EMat> emats;
	mc::EMat emat2(getTempVec(10));
	mc::EMat emat3(getTempVec(20));
	emats.push_back(emat);
	emats.push_back(emat2);
	emats.push_back(emat3);

	vector< vector< vector<string> > > ematsVec = mc::EMat::toVec(mc::EMat::cast(emats));
	EXPECT_EQ(1, toInt(ematsVec[0][0][0]));
	EXPECT_EQ(10, toInt(ematsVec[1][0][0]));
	EXPECT_EQ(28, toInt(ematsVec[2][2][2]));
}

TEST_F(EMatTest, castTest){
	vector<mc::EMat> emats;
	emats.push_back(mc::EMat(getTempVec()));
	emats.push_back(mc::EMat(getTempVec(10)));
	emats.push_back(mc::EMat(getTempVec(20)));
	vector<cv::Mat> mats = mc::EMat::cast(emats);
	EXPECT_EQ(20, mats[2].at<double>(0, 0));
}

TEST_F(EMatTest, RVecTest){
	ASSERT_THROW( mc::RVec rvec(getTempMat() ), invalid_argument );
	cv::Mat mat = ( cv::Mat_<double>(1, 3) << 1,2,3 );
	mc::RVec rvec = mat;
	EXPECT_EQ(3, rvec[2]);
	rvec[0] = 4;
	EXPECT_EQ(4, rvec[0]);
}

int main( int argc, char* argv[] ){
    ::testing::InitGoogleTest( &argc, argv );
    return RUN_ALL_TESTS();
}

