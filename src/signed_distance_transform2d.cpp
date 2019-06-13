/**
 *   #, #,         CCCCCC  VV    VV MM      MM RRRRRRR
 *  %  %(  #%%#   CC    CC VV    VV MMM    MMM RR    RR
 *  %    %## #    CC        V    V  MM M  M MM RR    RR
 *   ,%      %    CC        VV  VV  MM  MM  MM RRRRRR
 *   (%      %,   CC    CC   VVVV   MM      MM RR   RR
 *     #%    %*    CCCCCC     VV    MM      MM RR    RR
 *    .%    %/
 *       (%.      Computer Vision & Mixed Reality Group
 *                For more information see <http://cvmr.info>
 *
 * This file is part of RBOT.
 *
 *  @copyright:   RheinMain University of Applied Sciences
 *                Wiesbaden Rüsselsheim
 *                Germany
 *     @author:   Henning Tjaden
 *                <henning dot tjaden at gmail dot com>
 *    @version:   1.0
 *       @date:   30.08.2018
 *
 * RBOT is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * RBOT is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with RBOT. If not, see <http://www.gnu.org/licenses/>.
 */

#include "signed_distance_transform2d.h"

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;
using namespace std;


SignedDistanceTransform2D::SignedDistanceTransform2D(float maxDist)
{
    this->maxDist = maxDist;
}

SignedDistanceTransform2D::~SignedDistanceTransform2D()
{
    
}

void SignedDistanceTransform2D::computeTransform(const Mat &src, Mat &sdt, Mat &xyPos, int threads, uchar key)
{
    sdt.create(src.size(), CV_32FC1);
    Mat dd(src.size(), CV_32SC1);
    Mat xPos(src.size(), CV_32SC1);
    xyPos.create(src.size(), CV_32SC2);
    
    sdt.setTo(0);
    xyPos.setTo(-1);
    
    int n = (src.cols > src.rows) ? src.cols : src.rows;
    
    int* v = (int *)malloc(threads*n*sizeof(int));
    int* z = (int *)malloc(threads*(n+1)*sizeof(int));
    int* f = (int *)malloc(threads*n*sizeof(int));
    
    int type = src.type();
    uchar depth = type & CV_MAT_DEPTH_MASK;
    
    if(depth == CV_8U)
    {
        if(key > 0)
        {
            parallel_for_(cv::Range(0, threads), Parallel_For_distanceTransformRowsWithKey(src, key, dd, xPos, v, z, threads));
        }
        else
        {
            parallel_for_(cv::Range(0, threads), Parallel_For_distanceTransformRows<uchar>(src, dd, xPos, v, z, threads));
        }
    }
    else if(depth == CV_32F)
    {
        parallel_for_(cv::Range(0, threads), Parallel_For_distanceTransformRows<float>(src, dd, xPos, v, z, threads));
    }
    else
    {
        cout << "WRONG IMAGE TYPE FOR SIGNED DISTANCE TRANSFORMATION! NOTE: USE FLOAT OR UCHAR." << endl;
    }
    
    parallel_for_(cv::Range(0, threads), Parallel_For_distanceTransformCols(dd, sdt, xPos, xyPos, maxDist, v, z, f, threads));
    
    
    free(z);
    free(v);
    free(f);
}


void SignedDistanceTransform2D::computeDerivatives(const cv::Mat &sdt, cv::Mat &dX, cv::Mat &dY, int threads)
{
    dX.create(sdt.size(), CV_32FC1);
    dY.create(sdt.size(), CV_32FC1);
    
    dX.col(0).setTo(0);
    dX.col(dX.cols-1).setTo(0);
    
    dY.row(0).setTo(0);
    dY.row(dY.rows-1).setTo(0);
    
    parallel_for_(cv::Range(0, threads), Parallel_For_distanceTransformDX<float>(sdt, dX, threads));
    parallel_for_(cv::Range(0, threads), Parallel_For_distanceTransformDY<float>(sdt, dY, threads));
}
