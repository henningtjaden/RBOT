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

#ifndef TRANSFORMATIONS_H
#define TRANSFORMATIONS_H

#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>

/**
 *  A collection of geometric transformations implemented using the OpenCV matrix types.
 */
class Transformations
{
    
public:
    /**
     *  Constructs a uniform 3D scale transform in 4x4 homogeneous matrix
     *  representation.
     *
     *  @param s The uniform scale factor in all three dimensions.
     *  @return A 4x4 homogenbeous 3D scale matrix.
     */
    static cv::Matx44f scaleMatrix(float s);
    
    /**
     *  Constructs a general 3D scale transform in 4x4 homogeneous matrix
     *  representation.
     *
     *  @param sx The scale factor in x-direction.
     *  @param sy The scale factor in y-direction.
     *  @param sz The scale factor in z-direction.
     *  @return A 4x4 homogenbeous 3D scale matrix.
     */
    static cv::Matx44f scaleMatrix(float sx, float sy, float sz);
    
    /**
     *  Constructs a 3D translation transform in 4x4 homogeneous matrix
     *  representation.
     *
     *  @param tvec The 3D vector (tx, ty, tz) to be used as translation.
     *  @return A 4x4 homogenbeous 3D translation matrix.
     */
    static cv::Matx44f translationMatrix(const cv::Vec3f tvec);
    
    /**
     *  Constructs a 3D translation transform in 4x4 homogeneous matrix
     *  representation.
     *
     *  @param tx Translation in x-direction.
     *  @param tx Translation in y-direction.
     *  @param tx Translation in z-direction.
     *  @return A 4x4 homogenbeous 3D translation matrix.
     */
    static cv::Matx44f translationMatrix(float tx, float ty, float tz);
    
    /**
     *  Constructs a 3D rotation transform in 4x4 homogeneous matrix
     *  representation from a given axis and angle of rotation.
     *
     *  @param angle The angle of rotation.
     *  @param axis  The axis of rotation.
     *  @return A 4x4 homogenbeous 3D rotation matrix.
     */
    static cv::Matx44f rotationMatrix(float angle, cv::Vec3f axis);
    
    /**
     *  Constructs an OpenGL look-at transform in 4x4 homogeneous matrix
     *  representation.
     *
     *  @param ex The x-coordinate of the eye/camera origin.
     *  @param ey The y-coordinate of the eye/camera origin.
     *  @param ez The z-coordinate of the eye/camera origin.
     *  @param cx The x-coordinate of the center point where the camera is looking at.
     *  @param cy The y-coordinate of the center point where the camera is looking at.
     *  @param cz The z-coordinate of the center point where the camera is looking at.
     *  @param ux The x-direction of the up-vector.
     *  @param uy The y-direction of the up-vector.
     *  @param uz The z-direction of the up-vector.
     *  @return A 4x4 homogenbeous look-at matrix.
     */
    static cv::Matx44f lookAtMatrix(float ex, float ey, float ez, float cx, float cy, float cz, float ux, float uy, float uz);
    
    /**
     *  Constructs a 4x4 OpenGL perspective projection matrix from a
     *  given field of view, aspect ration and a near and far plane.
     *
     *  @param fovy   The vertical field of view of the camera.
     *  @param aspect The aspect ratio of the image.
     *  @param near   The distance of the near clipping plane.
     *  @param far    The distance of the far clipping plane.
     *  @return A 4x4 perspective projection matrix.
     */
    static cv::Matx44f perspectiveMatrix(float fovy, float aspect, float zNear, float zFar);
    
    /**
     *  Constructs a 4x4 OpenGL perspective projection matrix from a
     *  given intrinsic matrix corresponding to a real camera.
     *
     *  @param K      A 3x3 intrinsic camera matrix.
     *  @param width  The width of the camer image in pixels.
     *  @param height The height of the camer image in pixels.
     *  @param near   The distance of the near clipping plane.
     *  @param far    The distance of the far clipping plane.
     *  @param flipY  A flag telling whether the direction of the y-axis of the camera is to flipped or not.
     *  @return A 4x4 perspective projection matrix matching the camera's intrinsics.
     */
    static cv::Matx44f perspectiveMatrix(const cv::Matx33f& K, int width, int height, float zNear, float zFar, bool flipY = false);
    
    /**
     *  Constructs 3x3 skew-symmetric matrix (also called axiator) from
     *  a given 3D vector.
     *
     *  @param a A 3D vector.
     *  @return The 3x3 skew-symmetric matrix corresponding to the 3D vector.
     */
    static cv::Matx33f axiator(cv::Vec3f a);
    
    /**
     *  Computes the exponential map from a given 6D vector of twist
     *  coordinates to the corresponding rigid body transform in 4x4
     *  homogeneous matrix representation.
     *
     *  @param xi A 6D vector of tiwst coordinates.
     *  @return A 4x4 homogenbeous rigid body transformation matrix corresponding to the twist coordinates.
     */
    static cv::Matx44f exp(cv::Matx61f xi);
};

#endif //TRANSFORMATIONS_H
