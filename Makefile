# EMatのためのMakefile
CXX  = g++
SRCS  = MathU.cpp MatU.cpp RVec.cpp EMat.cpp
# OBJS = EMat.o
OBJS = $(SRCS:%.cpp=%.o)
DEPS = $(SRCS:%.cpp=%.d)

# FLAGS = -Wc++11-extensions
FLAGS = -std=c++11

INCLUDE_PASS = -I./

# boostへのパス
# 環境に合わせて変更してください
BOOST_INCPASS = -I/opt/local/include

# OpenCV のライブラリを指定
OPENCVINC    = `pkg-config --cflags opencv`
OPENCVLIB    = `pkg-config --libs opencv`

# ---- テスト用の設定 ----

# google test関連
# 環境に合わせて変更してください
GTEST_INCPASS    = -I/usr/local/include/
GTEST_LIBPASS    = -L/usr/local/lib/gtest/
GTEST_LIBS       = -lgtest -lpthread

TEST_DIR = test
TEST_SRCS = $(TEST_DIR)/testEMat.cpp
TEST_OBJS    = $(TEST_SRCS:%.cpp=%.o)
TEST_PROGRAM = $(TEST_DIR)/testEMatIO.test

all:            $(OBJS)

clean:;         rm -f *.o *~ test/*.o test/*.test

.cpp.o:
	$(CXX) -c -MMD $(FLAGS) $< $(OPENCVINC) $(INCLUDE_PASS)

$(TEST_OBJS):	$(TEST_SRCS)
	$(CXX)  $(FLAGS) -c -o $@ $^ $(GTEST_INCPASS) $(INCLUDE_PASS)

test:		$(TEST_OBJS) $(OBJS)
	$(CXX) $(FLAGS) $(TEST_OBJS) $(OBJS) -o $(TEST_PROGRAM) $(GTEST_LIBPASS) $(GTEST_LIBS) $(OPENCVLIB)

-include $(DEPS)
