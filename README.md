# Scalable Recognition
A pythonic implementation of [Scalable Recognition with a Vocabulary Tree (2006)](https://people.eecs.berkeley.edu/~yang/courses/cs294-6/papers/nister_stewenius_cvpr2006.pdf).

Searching for an image match in a large set of images is a difficult task for a computer. 
In [Scalable Recognition with a Vocabulary Tree (2006)](https://people.eecs.berkeley.edu/~yang/courses/cs294-6/papers/nister_stewenius_cvpr2006.pdf) 
a method is proposed how this goal could be achieved. Despite the age of this paper, few python implementations are available which is one of the motivations for this repository. 
The goal of this implementation is that you can easily try out your own database matching without needing to program anything else. 
Parameters are used to customise your database matching to maximise your precision.

Almost all information is based on the before mentioned [paper](https://people.eecs.berkeley.edu/~yang/courses/cs294-6/papers/nister_stewenius_cvpr2006.pdf) and a course text on Image Analysis and Understanding of prof. Tuytelaars at the KU Leuven who was also my mentor for this student project.
## Installation
Using conda

    conda install -c conda-forge opencv numpy-indexed

Using pip

    pip install opencv-python numpy-indexed

## How to use
### 1) Setup

Put all your database images in the `data/` folder. Every image should have the format `id.jpg` where `id` is an integer. To be complete, the ids aren't required to be sequential. The images in the `data/` folder are often refered to as the db (database) images.

Put all your query images in the `test/` folder. Every image must have a `.jpg` extension.   
**_Optional:_** _if you'd like to test the accuracy, then name every query image `id.jpg` where `id` is the id of the correct db image match._

_**Optional:** For a cleaner output, you can add a `names.p` file. This file is a pickle file containing a dictionary<integer, string>.
The number of keys must match the number of images in the `data/` folder. Each image must have one corresponding key with its id.
The corresponding value represents the name of the db image._

**Important:** Check `parameters.py` to make sure everything is set up correctly for your purposes. If you don't fully understand a parameter, most of the time the defaults are fine, but please read the 'How it works' section to understand the parameters. 
### 2) Offline

In this fase the data structure for retrieval is built.
If you did everything in the previous steps, you only need to run the `offline.py` script and everything should be setup. 
If you'd like to understand what this script exactly does, first read through this readme file and afterwards read the detailed documentation in `offline.py`.

### 3) Online

//TODO

## How it works

In this section we'll explain the principles behind the code.
It's written in a way so no prior knowledge except some programming experience is required. 
First we define the problem, then we'll explain which basic principles we use and afterwards we show how these basic principles can be used to tackle the problem.

### The problem

We have a lot of objects as db images. 
A query image (possibly) contains the same object as a db but in a different context. 
The goal is to match the query image to the correct db image where speed and accuracy define the performance of the retrieval system

An example:  
Our dataset exists of common landmarks. The query image is a family picture in front of a landmark that they forgot the name of. 
They input the query image, and they have a match with a reference image of the Eiffel tower (the name comes from the `names.p` file).

How will we achieve this?  
First, we need to be able to tell if we have the same object in two images.  
When we know how we can do that, we'll use those principles with an efficient database structure to scale it.

### The basics

In this section we'll tackle the problem of telling if there's an identical object in two images. Traditionally this is done by using local invariant features. These work in three steps.

1) **_Feature extraction:_** Find points of interest in an image, often referred to as keypoints (kp). 
Keypoints aren't only locations but also contain information about its size and sometimes shape.
In short keypoints define "regions" of interest in an image.
Ideally these regions of interest should be the same on an object no mather at which angle or distance it's in the image.
Therefore, keypoint detectors are often designed to be rotation and scale invariant.
Why this is important will be clear after step 2) and 3).

2) **_Feature description:_** 
3) **_Feature matching:_** 




### Scaling up

## Further improvement



