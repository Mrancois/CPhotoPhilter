#include "opencv2/opencv.hpp"
#include <iostream>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/video/video.hpp"
#include "opencv2/opencv.hpp"

using namespace cv;

void overlayImage(const cv::Mat &background, const cv::Mat &foreground,
                  cv::Mat &output, cv::Point2i location);
void detectAndDisplay(Mat frame);

// Global variables
// Copy this file from opencv/data/haarscascades to target folder
string face_cascade_name = "/Users/francoisdevove/ClionProjects/FaceDetect/haarcascade_frontalface_alt.xml";
CascadeClassifier face_cascade;
string window_name = "Capture - Face detection";
int filenumber; // Number of file to be saved
string filename;
Mat img = cv::imread("/Users/francoisdevove/ClionProjects/FaceDetect/overlay.png", CV_LOAD_IMAGE_UNCHANGED);
Mat result;

int main(int, char**)
{
    VideoCapture cap(0); // open the default camera
    if(!cap.isOpened())  // check if we succeeded
        return -1;

    namedWindow("Video",1);

    if (!face_cascade.load(face_cascade_name)) {
        printf("--(!)Error loading\n");
        return (-1);
    }



    while(1) {
        // add image to the frame works !!

        Mat frame;
        cap >> frame;         // get a new frame from camera


        // detect face
        detectAndDisplay(frame);


        /*
        cv::cvtColor(img, img, CV_BGRA2BGR);
        cv::Rect roi(cv::Point(0, 0), cv::Size(422, 666));
        cv::Mat destinationROI = frame(roi);

        img.copyTo(destinationROI(cv::Rect(0, 0, img.cols, img.rows)));
        */
        imshow("Video", frame);
        /*
        Mat newFrame=frame.clone();

        int cx = (newFrame.cols - 70) / 2;

        cv::resize(img,img,Size(70,70));
        Mat gray_image = Mat(img.size(), CV_16UC3); // = Mat(img.size(), CV_16UC3);
        cvtColor(img, gray_image, COLOR_BGR2GRAY);
        Rect dstRC = Rect(cx, newFrame.rows/2, 70, 70);
        Mat dstROI = newFrame(dstRC);
        gray_image.copyTo(dstROI);

        imshow("Video", newFrame);
        */

        // Press 'c' to escape
        if(waitKey(30) == 'c') break;
    }
    return 0;
}

void overlayImage(const cv::Mat &background, const cv::Mat &foreground,
                  cv::Mat &output, cv::Point2i location)
{
    background.copyTo(output);


    // start at the row indicated by location, or at row 0 if location.y is negative.
    for(int y = std::max(location.y , 0); y < background.rows; ++y)
    {
        int fY = y - location.y; // because of the translation

        // we are done of we have processed all rows of the foreground image.
        if(fY >= foreground.rows)
            break;

        // start at the column indicated by location,

        // or at column 0 if location.x is negative.
        for(int x = std::max(location.x, 0); x < background.cols; ++x)
        {
            int fX = x - location.x; // because of the translation.

            // we are done with this row if the column is outside of the foreground image.
            if(fX >= foreground.cols)
                break;

            // determine the opacity of the foregrond pixel, using its fourth (alpha) channel.
            double opacity =
                    ((double)foreground.data[fY * foreground.step + fX * foreground.channels() + 3])

                    / 255.;


            // and now combine the background and foreground pixel, using the opacity,

            // but only if opacity > 0.
            for(int c = 0; opacity > 0 && c < output.channels(); ++c)
            {
                unsigned char foregroundPx =
                        foreground.data[fY * foreground.step + fX * foreground.channels() + c];
                unsigned char backgroundPx =
                        background.data[y * background.step + x * background.channels() + c];
                output.data[y*output.step + output.channels()*x + c] =
                        backgroundPx * (1.-opacity) + foregroundPx * opacity;
            }
        }
    }
}


void detectAndDisplay(Mat frame)
{
    std::vector<Rect> faces;
    Mat frame_gray;
    Mat crop;
    Mat res;
    Mat gray;
    string text;
    std::stringstream sstm;

    cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
    equalizeHist(frame_gray, frame_gray);

// Detect faces
    face_cascade.detectMultiScale(frame_gray, faces, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));


// Set Region of Interest
    cv::Rect roi_b;
    cv::Rect roi_c;

    size_t ic = 0; // ic is index of current element
    int ac = 0; // ac is area of current element

    size_t ib = 0; // ib is index of biggest element
    int ab = 0; // ab is area of biggest element

    for (ic = 0; ic < faces.size(); ic++) // Iterate through all current elements (detected faces)

    {
        roi_c.x = faces[ic].x;
        roi_c.y = faces[ic].y;
        roi_c.width = (faces[ic].width);
        roi_c.height = (faces[ic].height);

        ac = roi_c.width * roi_c.height; // Get the area of current element (detected face)

        roi_b.x = faces[ib].x;
        roi_b.y = faces[ib].y;
        roi_b.width = (faces[ib].width);
        roi_b.height = (faces[ib].height);

        ab = roi_b.width * roi_b.height; // Get the area of biggest element, at beginning it is same as "current" element

        if (ac > ab)
        {
            ib = ic;
            roi_b.x = faces[ib].x;
            roi_b.y = faces[ib].y;
            roi_b.width = (faces[ib].width);
            roi_b.height = (faces[ib].height);
        }


        crop = frame(roi_b);
        resize(crop, res, Size(128, 128), 0, 0, INTER_LINEAR); // This will be needed later while saving images
        cvtColor(crop, gray, CV_BGR2GRAY); // Convert cropped image to Grayscale

        // Form a filename
        filename = "";
        std::stringstream ssfn;
        ssfn << filenumber << ".png";
        filename = ssfn.str();
        filenumber++;

        imwrite(filename, gray);


        Point pt1(faces[ic].x, faces[ic].y); // Display detected faces on main window - live stream from camera
        Point pt2((faces[ic].x + faces[ic].height), (faces[ic].y + faces[ic].width));
        rectangle(frame, pt1, pt2, Scalar(255, 255, 0), 2, 8, 0);
    }

    Mat imgResized;
    // normal size img 587*422

    int heightImg = (int)(roi_b.height * 1.391);
    int widthImg = roi_b.height;

    if (roi_b.width) {
        Size size(widthImg, heightImg);
        resize(img, imgResized, size);

        overlayImage(frame, imgResized, result, cv::Point(roi_b.x,roi_b.y - 40));
    }



// Show image
    sstm << "X: " << roi_b.x << " Y: " << roi_b.y << " width: " << roi_b.width << " height: " << roi_b.height << " Filename: " << filename;
    text = sstm.str();

    putText(result, text, cvPoint(30, 30), FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(0, 0, 255), 1, CV_AA);


    imshow("frame", result);


    // GOOD: imshow("original", frame);
    /*
    if (!crop.empty())
    {
        imshow("detected", crop);
    }
    else
        destroyWindow("detected");
    */
}
