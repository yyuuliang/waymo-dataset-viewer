# waymo-dataset-viewer

## Instruction
Today Waymo open its dataset with a script to view the data. I followed the tutorials on their github webpage and found that I can't build the bazel project on my machine. However, I need some python scripts in order to view the dataset, so I use their [colab project](https://colab.research.google.com/github/waymo-research/waymo-open-dataset/blob/v1.0.1/tutorial/tutorial.ipynb) to build the python script. If you have the same issue like "ImportError: cannot import name 'dataset_pb2'" or "ImportError: cannot import name 'label_pb2'", download the two scripts here.      
Code in test.py are mainly from their tutorial notebook, I add some salt to make it work with the dataset you get from waymo website.  

## How to use
    1. Download Waymo's github code [here](https://github.com/waymo-research/waymo-open-dataset)  
    2. Put dataset_pb2.py and label_pb2.py in waymo_open_dataset folder
    3. Change python path and waymo dataset path in test.py  
    4. Run test.py

## More ways to go
Here are some projects I found useful while processing the Waymo dataset. Please check.

    1. Waymo Open Dataset Viewer
        [https://github.com/erksch/waymo-open-dataset-viewer](https://github.com/erksch/waymo-open-dataset-viewer)
        A WebGL viewer for pointclouds of the waymo open dataset that runs seamlessly in the browser with an integrated python server that processes and serves the dataset segments.
    2. Waymo_Kitti_Adapter
        [https://github.com/Yao-Shao/Waymo_Kitti_Adapter](https://github.com/Yao-Shao/Waymo_Kitti_Adapter)
        This is a tool converting Waymo open dataset format to Kitti dataset format.