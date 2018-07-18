Do inference for semantic segmentation with trained caffe models. 

1. Put the caffemodel and deploy file in the **models** folder. Then the deploy file is named **deploy.prototxt**, and the caffemodel file is named **segment.caffemodel**. Alternatively, you can also change the relevant parameters in the src/caffe\_cpp\_demo.cpp to the corresponding path:

	```string modelFile = the path of your deploy file;```
	```string trainedFile = the path of your caffemodel file; ```

2. Put the images in the **data** folder. Or, you can change the parameter in the src/caffe\_cpp\_demo.cpp to the corresponding path:
	
	```string image_path = the path of your images;```

3. Change INCLUDE\_DIRECTORIES and LINK\_DIRECTORIES for cuda, opencv, caffe, boost, gflags, and glog.
4. ```cd build```
5. ```cmake ..```
6. ```make```
7. ```sudo ./segDemoCPP```