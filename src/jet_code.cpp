#include <iostream>
#include <opencv2/core.hpp>

int main() {
    std::cout << "Hello World!";
    return 0;
}

// #include <opencv2/core.hpp>
// #include <opencv2/imgproc.hpp>

// cv::Scalar jet(double value, double minVal, double maxVal) {
//     // map the value to a range from 0.0 to 1.0
//     double v = (value - minVal) / (maxVal - minVal);

//     // map the value to a color using the "jet" colormap
//     int r, g, b;
//     if (v < 0.125) {
//         r = 0;
//         g = 0;
//         b = 255 * (4 * v + 0.5);
//     } else if (v < 0.375) {
//         r = 0;
//         g = 255 * (4 * v - 0.5);
//         b = 255;
//     } else if (v < 0.625) {
//         r = 255 * (4 * v - 1.5);
//         g = 255;
//         b = 255 * (-4 * v + 2.5);
//     } else if (v < 0.875) {
//         r = 255;
//         g = 255 * (-4 * v + 3.5);
//         b = 0;
//     } else {
//         r = 255 * (-4 * v + 4.5);
//         g = 0;
//         b = 0;
//     }

//     return cv::Scalar(b, g, r);
// }

// cv::Scalar color = jet(2.0, 1.0, 8.0);
//cout << color;
