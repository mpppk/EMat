#include <gtest/gtest.h>
// #include <GP.h>
#include <EMat.h>

using namespace std;

inline int toInt(std::string s) {int v; std::istringstream sin(s);sin>>v;return v;}
template<class T> inline std::string toString(T x) {std::ostringstream sout;sout<<x;return sout.str();}

// startValueから連番の値を振った行列を返す
vector< vector<string> > getTempMat(int startValue = 1, int rowsNum = 3, int colsNum = 3){
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

class EMatTest : public ::testing::Test{
protected:
	string fileDirPass;

	virtual void SetUp(){
		// ファイルの書きだし先
		fileDirPass = "../test/TestEMat/";
	}
};

TEST_F(EMatTest, constructorTest){
	mc::EMat emat(getTempMat());
	cv::Mat mat = ( cv::Mat_<double>(3, 3) << 1,2,3,4,5,6,7,8,9 );
	mc::EMat emat2(mat);
	mc::EMat emat3;
	emat3 = mat;
	mc::EMat emat4 = mat;

	EXPECT_EQ(5, emat(1, 1));
	EXPECT_EQ(5, emat2(1, 1));
	EXPECT_EQ(5, emat3(1, 1));
	EXPECT_EQ(5, emat4(1, 1));
}

TEST_F(EMatTest, toVecTest){
	// toVec(), toVec(cv::Mat)
	mc::EMat emat(getTempMat());
	vector< vector<string> > vec = emat.toVec();
	
	EXPECT_EQ(1, toInt(vec[0][0]));
	EXPECT_EQ(9, toInt(vec[2][2]));

	// toVec(vector<cv::Mat>)
	vector<mc::EMat> emats;
	mc::EMat emat2(getTempMat(10));
	mc::EMat emat3(getTempMat(20));
	emats.push_back(emat);
	emats.push_back(emat2);
	emats.push_back(emat3);

	vector< vector< vector<string> > > ematsVec = mc::EMat::toVec(mc::EMat::cast(emats));
	EXPECT_EQ(1, toInt(ematsVec[0][0][0]));
	EXPECT_EQ(10, toInt(ematsVec[1][0][0]));
	EXPECT_EQ(28, toInt(ematsVec[2][2][2]));
}

TEST_F(EMatTest, normalizeTest){

}

TEST_F(EMatTest, castTest){
	vector<cv::Mat> mats;
	mats.push_back(mc::EMat(getTempMat()));
	mats.push_back(mc::EMat(getTempMat(10)));
	mats.push_back(mc::EMat(getTempMat(20)));
	EXPECT_EQ(20, mats[2].at<double>(0, 0));
}

int main( int argc, char* argv[] ){
    ::testing::InitGoogleTest( &argc, argv );
    return RUN_ALL_TESTS();
}

