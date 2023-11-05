#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
#include <ctime>

using namespace std;
using namespace cv;

int tempo = 0;

void detectAndDraw(Mat& img, CascadeClassifier& cascade, double scale, bool tryflip);
void moveTowardsFace(Mat& img, int& xPos, int& yPos, int faceX, int faceY, double speed);

string cascadeName;

int main(int argc, const char** argv)
{
    VideoCapture capture;
    Mat frame;
    bool tryflip;
    CascadeClassifier cascade;
    double scale;

    cascadeName = "haarcascade_frontalface_default.xml";
    scale = 2; // Usar 1, 2, 4.
    if (scale < 1)
        scale = 1;
    tryflip = true;

    if (!cascade.load(cascadeName)) {
        cerr << "ERROR: Could not load classifier cascade: " << cascadeName << endl;
        return -1;
    }

    if (!capture.open(0)) // Usar a webcam padrão
    {
        cout << "Capture from camera #0 didn't work" << endl;
        return 1;
    }

    if (capture.isOpened()) {
        cout << "Video capturing has been started ..." << endl;

        while (1)
        {
            capture >> frame;
            if (frame.empty())
                break;

            detectAndDraw(frame, cascade, scale, tryflip);

            char c = (char)waitKey(10);
            if (c == 27 || c == 'q' || c == 'Q')
                break;
        }
    }

    return 0;
}

void detectAndDraw(Mat& img, CascadeClassifier& cascade, double scale, bool tryflip)
{
    double t = 0;
    vector<Rect> faces;
    Mat smallImg, gray;
    Scalar color = Scalar(255, 0, 0);

    double fx = 1 / scale;
    resize(img, smallImg, Size(), fx, fx, INTER_LINEAR_EXACT);
    if (tryflip)
        flip(smallImg, smallImg, 1);
    cvtColor(smallImg, gray, COLOR_BGR2GRAY);
    equalizeHist(gray, gray);

    t = (double)getTickCount();

    cascade.detectMultiScale(gray, faces, 1.3, 2, 0 | CASCADE_SCALE_IMAGE, Size(40, 40));
    t = (double)getTickCount() - t;
    printf("detection time = %g ms\n", t * 1000 / getTickFrequency());

    Mat protag = cv::imread("protag.png", IMREAD_UNCHANGED);

    // PERCORRE AS FACES ENCONTRADAS
    for (size_t i = 0; i < faces.size(); i++)
    {
        Rect r = faces[i];
        rectangle(smallImg, Point(cvRound(r.x), cvRound(r.y)),
            Point(cvRound((r.x + r.width - 1)), cvRound((r.y + r.height - 1))),
            color, 3);

        // Calcula o movimento da figura em direção ao rosto
        int protagX = r.x + (r.width) / 2 - (protag.cols) / 2;
        int protagY = r.y + (r.height) / 2 - (protag.rows) / 2;

        // Chama a função para mover a figura em direção ao rosto
        moveTowardsFace(smallImg, protagX, protagY, r.x + r.width / 2, r.y + r.height / 2, 2.0);

        drawTransparency(smallImg, protag, protagX, protagY);
    }

    printf("protag::width: %d, height=%d\n", protag.cols, protag.rows);

    tempo++;

    imshow("result", smallImg);
}

// Função para mover a figura em direção ao rosto
void moveTowardsFace(Mat& img, int& xPos, int& yPos, int faceX, int faceY, double speed)
{
    int dx = faceX - xPos;
    int dy = faceY - yPos;

    // Calcula a direção do movimento
    double distance = sqrt(dx * dx + dy * dy);
    if (distance > 0)
    {
        double directionX = dx / distance;
        double directionY = dy / distance;

        // Move a figura na direção do rosto
        xPos += speed * directionX;
        yPos += speed * directionY;
    }
}

void drawTransparency(Mat frame, Mat transp, int xPos, int yPos) {
    Mat mask;
    vector<Mat> layers;

    split(transp, layers); // Separa os canais
    Mat rgb[3] = { layers[0], layers[1], layers[2] };
    mask = layers[3]; // Canal alfa (transparência)
    merge(rgb, 3
