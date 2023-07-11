#include "yolov8.h"
#include <string>
#include <string.h>
#include <iostream>

int main(int argc, char *argv[])
{
    try
    {
        if (argc != 3)
        {
            std::cout << "Usage:" << argv[0] << " <path_to_model> <path_to_image>" << std::endl;
            return EXIT_FAILURE;
        }
        const std::string input_model_path{argv[1]};
        const std::string input_video_path{argv[2]};
        Config config = {0.2, 0.4, 0.4, 640, 640, input_model_path};

        YOLOV8 yolomodel(config);
        cv::VideoCapture cap;

        cap.open(input_video_path);

        int frame_number = cap.get(cv::CAP_PROP_FRAME_COUNT);

        std::cout << "Total number of frames are: " << frame_number << std::endl;
        auto resW = cap.get(cv::CAP_PROP_FRAME_WIDTH);
        auto resH = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
        std::cout << "Original video resolution: (" << resW << "x" << resH << ")" << std::endl;
        cap.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, 720);

        resW = cap.get(cv::CAP_PROP_FRAME_WIDTH);
        resH = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
        std::cout << "New video resolution: (" << resW << "x" << resH << ")" << std::endl;

        if (!cap.isOpened()) {
            throw std::runtime_error("Unable to open video capture!");
        }
        int num_frames = 0;
        auto start_time = clock();
        while (true)
        {
            // Grab frame
            cv::Mat img;
            cap >> img;
            if (img.empty())
            {
                throw std::runtime_error("Unable to decode image from video stream.");
            }

            num_frames++;
            
            // Run inference
            yolomodel.detect(img);

            auto elapsed = clock() - start_time;
            auto elapsed_seconds = double(elapsed - start_time) / CLOCKS_PER_SEC;
            auto fps = num_frames / elapsed_seconds;

            cv::putText(img, "FPS: " + std::to_string(fps), cv::Point(50, 50), cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(0, 0, 0), 2, 0);

            // Display the results
            cv::imshow("Object Detection", img);

            if (cv::waitKey(1) >= 0)
            {
                break;
            }
        }
        cap.release();
        cv::destroyAllWindows();
        exit(0);
        return 0;
    }
    catch (const std::exception &ex)
    {
        std::cerr << ex.what() << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}