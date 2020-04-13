# tfserving-image-inference

# Getting Started server side
Install Anaconda, or otherwise ensure you have python 3.7 since tensorflow doesn't yet support python 3.8.
```
conda create -n model_converter python=3.7 anaconda

conda activate model_converter

pip install -r requirements.txt
```

Download the model checkpoints saved to github: [Covid-net keras](https://github.com/lindawangg/COVID-Net/tree/covidnet-keras)

Specifically download the `hdf5` type of pretrained model. Then unzip and save the hdf5 file in a directory labeled `model/` in the root directory.

This file should contain both the graph of the model, and the pretrained weights. 

# Prepare model for serving
To convert this model to saved_model format, make sure you are in the model_converter conda environment, then run the following command:

```python model_converter.py```

# After model has been converted into saved model format. Run using docker. Change the source path to reflect the absolute path to saved_model.

`docker run -p 8501:8501 -e MODEL_NAME='covid-test' --mount type=bind,source=/Users/myusername/Desktop/repos/tfserving-image-inference/saved_model/,target=/models/covid-test tensorflow/serving`
