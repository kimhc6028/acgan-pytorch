Pytorch implementation of [Conditional Image Synthesis with Auxiliary Classifier GANs](https://arxiv.org/pdf/1610.09585.pdf).

## Requirements

- Python 2.7
- [numpy](http://www.numpy.org/)
- [pytorch](http://pytorch.org/)

## Usage

	$ ./run.sh

or

	$ python main.py --dataset mnist --dataroot data --outf output --cuda                                                                           

will start training gpu with GPU. Datasets will be downloaded to /data, and generated images are stored in output directory.


![Alt text](./output/fake_samples_epoch_015.png?raw=true "Title")

Generated MNIST samples after 15 epoches.
