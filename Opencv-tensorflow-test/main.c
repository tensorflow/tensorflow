#include <stdio.h>
#include "opencv2/highgui/highgui_c.h"
#include "opencv2/imgproc/imgproc_c.h"
#include "opencv2/videoio/videoio_c.h"
#include "opencv2/imgcodecs/imgcodecs_c.h"

#include "tensorflow_interface.h"

void main(int argc, char** argv)
{
	init_main_interface();
	char* filename = "C:\\Users\\chemf\\source\\repos\\Opencv-tensorflow-test\\Test-15.jpg";
	int flag = 1;
	IplImage* src = cvLoadImage(filename,1);
	IplImage* dst = cvCreateImage(cvSize(src->width, src->height), 32, 3);
	cvConvertScale(src, dst, 1. / 255, 0.0);
	IplImage* resizedImage = cvCreateImage(cvSize(224, 224), 32, 3);
	cvResize(dst, resizedImage,1);
	cvNamedWindow("Input",CV_WINDOW_AUTOSIZE);
	cvShowImage("Input", resizedImage);
	char* output_labels = main_interface((float*)resizedImage->imageData);
	printf("She is in the %s\n", output_labels);
	free(output_labels);
	cvWaitKey(0);
	cvReleaseImage(&dst);
	cvReleaseImage(&resizedImage);
	cvReleaseImage(&src);
	//src = cvLoadImage(argv[2], 1);
	//dst = cvCreateImage(cvSize(src->width, src->height), IPL_DEPTH_32F, 3);
	//cvConvertScale(src, dst, 1. / 255, 0.0);
	//resizedImage = cvCreateImage(cvSize(224, 224), IPL_DEPTH_32F, 3);
	//cvResize(dst, resizedImage, 1);
	////cvNamedWindow("Input", CV_WINDOW_AUTOSIZE);
	//cvShowImage("Input", resizedImage);
	//output_labels = main_interface((float*)resizedImage->imageData);
	//printf("Now, She is in the %s\n", output_labels);
	//free(output_labels);
	//cvWaitKey(0);
	//cvReleaseImage(&dst);
	//cvReleaseImage(&resizedImage);
	//cvReleaseImage(&src);
}