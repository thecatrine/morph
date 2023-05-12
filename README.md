# Morph

# Intro
I got tired of rewriting my simple training loop for each simple ML project, so I've tried to modularize it a bit and reduce the parts of it that are implementation dependent. I intend to clone and edit this as a starting point for my next projects.

Hopefully everything is clear and easy to understand. The multiprocessing stuff gets a little difficult to follow, but it should work for data parallelism across multiple GPUs if you set that option.

Currently everything is set up for the model I was working on at the time, which is a Convnet to do score based image generation.

## Usage

python morph.py -h

python morph.py train
python morph.py train --train-config=alternate_config.yaml