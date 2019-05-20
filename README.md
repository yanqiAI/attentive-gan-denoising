###
Using attentive gan to denoise dirty document images. The generator and discriminator are added attention map, the whole methods come from the paper 《Attentive Generative Adversarial Network for Raindrop Removal from A Single Image》

### data prepare
paired images are needed: clean image | dirty image

### pre-processing
1. run /data_provider/crop_images.py  get train dataset default size 256 * 256
2. run /data_provider/data_feed_pipline.py get train and validation tfrecords files

### model
attentive-gan-denoising
![network](https://github.com/yanqiAI/attentive-gan-denoising/blob/master/img/network.png)
attentive_gan_net.py generator network 
![generator](https://github.com/yanqiAI/attentive-gan-denoising/blob/master/img/generator.png)
attentive_gan_sr_net.py generator network combined super resolution
discriminative_net.py discriminator network
discriminative_sr_net.py discriminator network combined super resolution

 
### train model

run /tools/train_model.py

### test model
run /tools/test_model.py 

run /tools/test_denoising.py

# result
noise image | denoise image  |  denoise and post process 
![image](https://github.com/yanqiAI/attentive-gan-denoising/blob/master/img/0150915_out.jpg)
