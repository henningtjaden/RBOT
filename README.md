# RBOT: Region-based Object Tracking

RBOT is a novel approach to real-time 6DOF pose pose estimation of rigid 3D objects using a monocular RGB camera. The key idea is to derive a region-based cost function using temporally consistent local color histograms and optimize it for pose with a Gauss-Newton scheme. The approach outperforms previous methods in cases of cluttered backgrounds, heterogenous objects, and occlusions. The proposed histograms are also used as statistical object descriptors within a template matching strategy for pose recovery after temporary tracking loss e.g. caused by massive occlusion or if the object leaves the camera's field of view. These descriptors can be trained online within a couple of seconds moving a handheld object in front of a camera.

### Preview Video

[![ICCV'17 supplementary video.](https://img.youtube.com/vi/gVX_gLIjQpI/0.jpg)](https://www.youtube.com/watch?v=gVX_gLIjQpI)


### Related Papers

* **A Region-based Gauss-Newton Approach to Real-Time Monocular Multiple Object Tracking**
*H. Tjaden, U. Schwanecke, E. Schömer, D. Cremers*, TPAMI '18

* **Real-Time Monocular Pose Estimation of 3D Objects using Temporally Consistent Local Color Histograms**
*H. Tjaden, U. Schwanecke, E. Schömer*, ICCV '17

* **Real-Time Monocular Segmentation and Pose Tracking of Multiple Objects**
*H. Tjaden, U. Schwanecke, E. Schömer*, ECCV '16


# Dependencies

RBOT depends on (recent versions of) the following software libraries:

* Assimp
* OpenCV
* OpenGL
* Qt

The code was developed and tested under macOS. It should, however, also run on Windows and Linux systems with (probably) a few minor changes required. Nothing is plattform specific by design.


# How To Use

The general usage of the algorithm is demonstrated in a small example command line application provided in `main.cpp`.  **It must be run from the root directory (that contains the *src* folder) otherwise the relative paths to the model and the shaders will be wrong.** Here the pose of a single 3D model is refined with respect to a given example image. The extension to actual pose tracking and using multiple objects should be straight foward based on this example. Simply replace the example image with the live feed from a camera or a video and add your own 3D models instead.

For the best performance when using your own 3D models, please **ensure that each 3D model consists of a maximum of around 4000 - 7000 vertices and is equally sampled across the visible surface**. This can be enforced by using a 3D mesh manipulation software such as MeshLab (http://www.meshlab.net/) or OpenFlipper (https://www.openflipper.org/).


# Dataset

To test the algorithm you can for example use the corresponding dataset available for download at: http://cvmr.mi.hs-rm.de/research/RBOT/


# License

RBOT is licensed under the GNU General Public License Version 3 (GPLv3), see http://www.gnu.org/licenses/gpl.html.
