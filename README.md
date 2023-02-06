# stereocamera
This is an image processing project, the realization of stereo matching.

a. Realize stereo matching to calculate parallax

b. Calculate depth from parallax

c. Calculate 3D point clouds

d. Achieve acceleration

e. Simple realization of binocular calibration and correction

The results are shown in the figures below

Disparity Map

![image](https://user-images.githubusercontent.com/80202433/216873502-fea973b4-6561-4ec9-8c36-ebacd99f89f9.png)

3D point cloud

Front view 

![image](https://user-images.githubusercontent.com/80202433/216873453-79258e23-9194-40d9-9763-e7b3c0ed223e.png)

End view 

![image](https://user-images.githubusercontent.com/80202433/216873946-d54bba3c-1987-4f62-b6f8-4ec45dcf8248.png)

Document description
3rdparty: Contains some of the databases necessary for opencv

Data: Data set required for the experiment

match_SGM folder:

auxiliary_fun cpp and h store some auxiliary functions

SGM_lab cpp and h store the main code of this experiment, such as the realization of SGM

main_lab houses the main framework of the experiment

pangolin: Contains the pangolin library

Unzip the library package required for configuration and copy the OpenCV folder to the 3rdparty folder. The Pangolin library can be configured in the following csdn method, where release x64 has been configured but the path of the containing directory and library directory needs to be changed.

Reproduction method

To run in a release x64 environment, you need to configure the opencv associated path, which you need to do in the properties because the pangolin library is called when implementing the 3d point cloud
Add the appropriate path in VC++ directory → Include directory and library directory, linker → input → Additional dependency configuration of the appropriate lib file

And then you can run it

In the debug x64 environment, you also need to perform related configurations
