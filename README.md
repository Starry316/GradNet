# GradNet
A tensorflow 2.0 implementation of SIGGRAPH Asia 2019 paper: GradNet: Unsupervised Deep Screened Poisson Reconstruction for Gradient-Domain Rendering

The origin paper can be found [GradNet: Unsupervised Deep Screened Poisson Reconstruction for Gradient-Domain Rendering](http://sites.cs.ucsb.edu/~lingqi/publications/paper_gradnet.pdf)

Thanks **Mengtian Li** and **Jie Guo** for their help.

Note that, this is no the official implementation. 

## Environment
- Python 3.7
- Tensorboard 2.0.0
- Tensorflow 2.0
- Openexr 1.3.2 


## How to train

### Data prepartaion
- Generate images and the corresponding gradients by gradient domain render.You need to record normal, depth and albedo during the rendering procedure.
- Organize the directory structure like the description blow:
```
your_data_dir
  |--sence1
  |   |--grad
  |   |    |--sence1_1.gx
  |   |    |--sence1_1.gy
  |   |      ...
  |   |--sence1_1.exr
  |   |--sence1_1_feature.txt
  |   |  ...
  |
  |--sence2
   ...
```

- Modify the`data_dir` to your train data directory , `scene_list` to your scenes in `create_tfrecord_random.py`.
- Run `python create_tfrecord_random.py` to generate tfrecords. It will randomly extract 15 patches of size 256x256 from each image and preprocess them according to the paper.


### Train
- Copy some tfrecords to `data/val/` as validation set.
- Modify the `steps_per_epoch` and `val_steps` in `train.py`.
- Run `python train.py`.
- You can run `tensorboard --logdir log` to watch the training process.

### Test
- prepare test image, gradients and features.Put image and features into `data/test/`, put gradient image into `data/test/grad/`.
- Modify `filepath_weights` in `test.py`.
- Run `python test.py`.
- You can see the results in `data/test/res/`



