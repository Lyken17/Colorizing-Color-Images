# Deep-White-Balancer

  
## Setup  
```bash
luarocks install torch
luarocks install nn
luarocks install image
luarocks install lua-cjson

#GPU acceleration
luarocks install cutorch
luarocks install cunn
luarocks install cudnn
```

# Model

Pretrained model can be found in [Google Drive](https://drive.google.com/drive/folders/0B3lu5NBSC7pVdGs1Mzl0SEJ0TUk?usp=sharing)

## Colorize images  (deprecated, not for this project)
Assume you want to colorize image `input.jpg` and store result image as `output.png`  
  
```bash
#Colorize an image
th colorize.lua -model model.t7 -input_image input.jpg -output_image output.png -gpu 0
  
#If you want to colorize all images in a folder
mkdir -p output
th colorize.lua -model model.t7 -input_dir input -output_dir output -gpu 0
```

## Train your own model  
Assume you all your training data are in `train` and validation data are in `validation`.   
The python script recursively checks all image files (including images in sub-directory) and throw all gray ones.  

```bash
python make_dataset.py --train_dir train --val_dir validation --output_file dataset.h5
th train.lua -h5_file dataset.h5 -checkpoint_name model -gpu 0
```
  
To compute the prediction error of your model in validation dataset, use `validation.lua`.  
```bash
th validation.lua -h5_file dataset.h5 -model model.t7 -gpu 0
```
  
## Reference  
[Perceptual Losses for Real-Time Style Transfer and Super-Resolution](https://github.com/jcjohnson/fast-neural-style)  
  
## License  
GNU GPL 3.0 for personal or research use. COMMERCIAL USE PROHIBITED.
