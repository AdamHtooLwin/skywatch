# Demo System for running FAS systems on local machine

Every file below needs a config file. As an example, run `main_lgsc.py -c configs_lgsc.yml`. 
Sample configs are provided in the configs folder.

Configs structure:

```yaml
device: 'cuda:0'
weights: "path/to/weights/epoch=10_v2.ckpt"  # path to weights file
frames_folder: "path/to/frames/output/" # path to saved cropped frames for testing

# mean and std dev to normalize with
mean:
  r: 0
  g: 0
  b: 0
std:
  r: 1
  g: 1
  b: 1
```

1. Run `main_lgsc.py` for writing output from video input to video.
2. Run `main_lgsc_webcam.py` for dynamically showing webcam input and overlaying bounding box and spoof score.
3. Run `main_meta_lgsc_webcam.py` for same as 2. but with meta-learned models.
