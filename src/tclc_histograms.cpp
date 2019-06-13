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

#include "tclc_histograms.h"
#include "model.h"

using namespace std;
using namespace cv;

TCLCHistograms::TCLCHistograms(Model *model, int numBins, int radius, float offset)
{
    this->_model = model;
    
    this->numBins = numBins;
    
    this->radius = radius;
    
    this->_offset = offset;
    
    this->_numHistograms = _model->getNumVertices();
    
    normalizedFG = Mat::zeros(this->_numHistograms, numBins*numBins*numBins, CV_32FC1);
    normalizedBG = Mat::zeros(this->_numHistograms, numBins*numBins*numBins, CV_32FC1);
    
    notNormalizedFG = Mat::zeros(this->_numHistograms, numBins*numBins*numBins, CV_32SC1);
    notNormalizedBG = Mat::zeros(this->_numHistograms, numBins*numBins*numBins, CV_32SC1);
    
    initialized = Mat::zeros(1, this->_numHistograms, CV_8UC1);
}

TCLCHistograms::~TCLCHistograms()
{
    
}

void TCLCHistograms::update(const Mat &frame, const Mat &mask, const Mat &depth, Matx33f &K, float zNear, float zFar)
{
    _centersIDs = parallelComputeLocalHistogramCenters(mask, depth, K, zNear, zFar, 0);
    
    filterHistogramCenters(100, 10.0f);
    
    int threads = (int)_centersIDs.size();
    
    memset(notNormalizedFG.ptr<int>(), 0, _centersIDs.size()*numBins*numBins*numBins*sizeof(int));
    memset(notNormalizedBG.ptr<int>(), 0, _centersIDs.size()*numBins*numBins*numBins*sizeof(int));
    
    memset(normalizedFG.ptr<float>(), 0, _centersIDs.size()*numBins*numBins*numBins*sizeof(float));
    memset(normalizedBG.ptr<float>(), 0, _centersIDs.size()*numBins*numBins*numBins*sizeof(float));
    
    Mat sumsFB = Mat::zeros((int)_centersIDs.size(), 1, CV_32SC2);
    
    parallel_for_(cv::Range(0, threads), Parallel_For_buildLocalHistograms(frame, mask, _centersIDs, radius, numBins, notNormalizedFG, notNormalizedBG, sumsFB, _model->getModelID(), threads));
    
    parallel_for_(cv::Range(0, threads), Parallel_For_mergeLocalHistograms(notNormalizedFG, notNormalizedBG, normalizedFG, normalizedBG, initialized, _centersIDs, sumsFB, 0.1f, 0.2f, threads));
}

void TCLCHistograms::updateCentersAndIds(const cv::Mat &mask, const cv::Mat &depth, const cv::Matx33f &K, float zNear, float zFar, int level)
{
    _centersIDs = parallelComputeLocalHistogramCenters(mask, depth, K, zNear, zFar, level);
    
    filterHistogramCenters(100, 10.0f);
}


vector<Point3i> TCLCHistograms::computeLocalHistogramCenters(const Mat &mask)
{
    uchar *maskData = mask.data;
    
    std::vector<Point3i> centers;
    
    int m_id = _model->getModelID();
    
    for(int i = 2; i < mask.rows - 2; i+=2)
    {
        for(int j = 2; j < mask.cols - 2; j+=2)
        {
            int idx = i*mask.cols + j;
            uchar val = maskData[idx];
            if(val == m_id)
            {
                if(maskData[idx + 2] != m_id || maskData[idx - 2] != m_id
                   || maskData[idx + 2*mask.cols] != m_id || maskData[idx - 2*mask.cols] != m_id
                   )
                {
                    centers.push_back(Point3i(j, i, (int)centers.size()));
                }
            }
        }
    }
    
    return centers;
}


vector<Point3i> TCLCHistograms::parallelComputeLocalHistogramCenters(const Mat &mask, const Mat &depth, const Matx33f &K, float zNear, float zFar, int level)
{
    vector<Point3i> res;
    
    vector<Vec3f> verticies = _model->getVertices();
    Matx44f T_cm = _model->getPose();
    Matx44f T_n = _model->getNormalization();
    
    vector<vector<Point3i> > centersIdsCollection;
    centersIdsCollection.resize(8);
    
    Matx44f T_cm_n = T_cm * T_n;
    
    int m_id = _model->getModelID();
    
    parallel_for_(cv::Range(0, 8), Parallel_For_computeHistogramCenters(mask, depth, verticies, T_cm_n, K, zNear, zFar, m_id, level, centersIdsCollection.data(), 8));
    
    for(int i = 0; i < centersIdsCollection.size(); i++)
    {
        vector<Point3i> tmp = centersIdsCollection[i];
        for(int j = 0; j < tmp.size(); j++)
        {
            res.push_back(tmp[j]);
        }
    }
    
    return res;
    
}


void TCLCHistograms::filterHistogramCenters(int numHistograms, float offset)
{
    int offset2 = (offset)*(offset);
    
    vector<Point3i> res;
    
    do
    {
        res.clear();
        
        while(_centersIDs.size() > 0)
        {
            Point3i center = _centersIDs[0];
            vector<Point3i> tmp;
            res.push_back(center);
            for(int c2 = 1; c2 < _centersIDs.size(); c2++)
            {
                Point3i center2 = _centersIDs[c2];
                int dx = center.x - center2.x;
                int dy = center.y - center2.y;
                int d = dx*dx + dy*dy;
                
                if(d >= offset2)
                {
                    tmp.push_back(center2);
                }
            }
            _centersIDs = tmp;
        }
        _centersIDs = res;
        
        offset += 1.0f;
        offset2 = offset*offset;
    }
    while(res.size() > numHistograms);
    
    _offset = offset;
}


Mat TCLCHistograms::getLocalForegroundHistograms()
{
    return normalizedFG;
}


Mat TCLCHistograms::getLocalBackgroundHistograms()
{
    return normalizedBG;
}


vector<Point3i> TCLCHistograms::getCentersAndIDs()
{
    return _centersIDs;
}


Mat TCLCHistograms::getInitialized()
{
    return initialized;
}


int TCLCHistograms::getNumBins()
{
    return numBins;
}

int TCLCHistograms::getNumHistograms()
{
    return _numHistograms;
}

int TCLCHistograms::getRadius()
{
    return radius;
}


float TCLCHistograms::getOffset()
{
    return _offset;
}


void TCLCHistograms::clear()
{
    normalizedFG = Mat::zeros(this->_numHistograms, numBins*numBins*numBins, CV_32FC1);
    normalizedBG = Mat::zeros(this->_numHistograms, numBins*numBins*numBins, CV_32FC1);
    
    notNormalizedFG = Mat::zeros(this->_numHistograms, numBins*numBins*numBins, CV_32SC1);
    notNormalizedBG = Mat::zeros(this->_numHistograms, numBins*numBins*numBins, CV_32SC1);
    
    initialized = Mat::zeros(1, this->_numHistograms, CV_8UC1);
}
