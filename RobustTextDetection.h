//
//  RobustTextDetection.h
//  RobustTextDetection
//
//  Created by Saburo Okita on 08/06/14.
//  Copyright (c) 2014 Saburo Okita. All rights reserved.
//

#ifndef __RobustTextDetection__RobustTextDetection__
#define __RobustTextDetection__RobustTextDetection__

#include <iostream>
#include <opencv2/opencv.hpp>
#include <bitset>

using namespace std;
using namespace cv;


/**
 * Parameters for robust text detection, quite a handful
 */
struct RobustTextParam {
    int minMSERArea;
    int maxMSERArea;
    int cannyThresh1;
    int cannyThresh2;
    
    int maxConnCompCount;
    int minConnCompArea;
    int maxConnCompArea;
    
    float minEccentricity;
    float maxEccentricity;
    float minSolidity;
	float maxSolidity;
    float maxStdDevMeanRatio;
};


/**
 * Implementation of Chen, Huizhong, et al. "Robust Text Detection in Natural Images with Edge-Enhanced Maximally Stable Extremal
 * Regions." Image Processing (ICIP), 2011 18th IEEE International Conference on. IEEE, 2011.
 * 
 * http://www.stanford.edu/~hchen2/papers/ICIP2011_RobustTextDetection.pdf
 * http://www.mathworks.de/de/help/vision/examples/automatically-detect-and-recognize-text-in-natural-images.html#zmw57dd0e829
 */
class RobustTextDetection {
public:
    RobustTextDetection( string temp_img_directory = "" );
    RobustTextDetection( RobustTextParam& param, string temp_img_directory = "" );
    
    pair<Mat, Rect> apply( Mat& image );
    
protected:
    Mat preprocessImage( Mat& image );
    Mat computeStrokeWidth( Mat& dist ) ;
    Mat createMSERMask( Mat& grey );
    
    static int toBin( const float angle, const int neighbors = 8 );
    Mat growEdges(Mat& image, Mat& edges );
    
    vector<Point> convertToCoords( int x, int y, bitset<8> neighbors ) ;
    vector<Point> convertToCoords( Point& coord, bitset<8> neighbors ) ;
    vector<Point> convertToCoords( Point& coord, uchar neighbors ) ;
    bitset<8> getNeighborsLessThan( int * curr_ptr, int x, int * prev_ptr, int * next_ptr ) ;
    
    Rect clamp( Rect& rect, Size size );
    
private:
    string tempImageDirectory;
    RobustTextParam param;
};

#endif /* defined(__RobustTextDetection__RobustTextDetection__) */
