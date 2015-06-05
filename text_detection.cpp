//
//  main.cpp
//  RobustTextDetection
//
//  Created by Saburo Okita on 05/06/14.
//  Copyright (c) 2014 Saburo Okita. All rights reserved.
//

#include <iostream>
#include <string>
#include <iterator>
#include <opencv2/opencv.hpp>
#include <tesseract/baseapi.h>

#include "RobustTextDetection.h"
#include "ConnectedComponent.h"

using namespace std;
using namespace cv;


int main(int argc, const char * argv[])
{

	namedWindow( "" );
	moveWindow("", 0, 0);

	Mat image = imread( "image7.jpg" );

	/* Quite a handful or params */
	RobustTextParam param;
	param.minMSERArea        = 10;
	param.maxMSERArea        = 2000;
	param.cannyThresh1       = 20;
	param.cannyThresh2       = 100;

	param.maxConnCompCount   = 10000;
	param.minConnCompArea    =100;
	param.maxConnCompArea    = 1800;

	param.minEccentricity    = 0.01;
	param.maxEccentricity    = 0.9;
	param.minSolidity        = 0.5;
	param.maxSolidity        = 0.8;
	param.maxStdDevMeanRatio = 0.25;

	/* Apply Robust Text Detection */
	/* ... remove this temp output path if you don't want it to write temp image files */
	string temp_output_path = "./image";
	RobustTextDetection detector(param, temp_output_path );
	pair<Mat, Rect> result = detector.apply( image );

	/* Get the region where the candidate text is */
	Mat stroke_width( result.second.height, result.second.width, CV_8UC1, Scalar(0) );
	Mat(result.first, result.second).copyTo( stroke_width);


	/* Use Tesseract to try to decipher our image */
	tesseract::TessBaseAPI tesseract_api;
	tesseract_api.Init(NULL, "eng"  );
	tesseract_api.SetImage((uchar*) stroke_width.data, stroke_width.cols, stroke_width.rows, 1, stroke_width.cols);

	string out = string(tesseract_api.GetUTF8Text());

	/* Split the string by whitespace */
	vector<string> splitted;
	istringstream iss( out );
	copy( istream_iterator<string>(iss), istream_iterator<string>(), back_inserter( splitted ) );

	/* And draw them on screen */
	/*CvFont font = cvFont( 24.0, 2 );
	Point coord = Point( result.second.br().x + 10, result.second.tl().y );
	for (vector<string>::const_iterator str_iter=splitted.begin();str_iter!=splitted.end();++str_iter)
	{
		string line=*str_iter;
		putText( image, line, Point(100,100), CV_FONT_HERSHEY_COMPLEX,1,Scalar(255,0,0),2 );
		coord.x += 25;
	}
*/
	rectangle( image, result.second, Scalar(0, 0, 255), 2);

	/* Append the original and stroke width images together */
	cvtColor( stroke_width, stroke_width, CV_GRAY2BGR );
	Mat appended( image.rows, image.cols + stroke_width.cols, CV_8UC3 );
	image.copyTo( Mat(appended, Rect(0, 0, image.cols, image.rows)) );
	stroke_width.copyTo( Mat(appended, Rect(image.cols, 0, stroke_width.cols, stroke_width.rows)) );

	imshow("", appended );
	waitKey();

	return 0;
}