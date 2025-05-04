# Notes

<img width="630" alt="Screenshot 2025-05-01 at 15 05 37" src="https://github.com/user-attachments/assets/a1eb2246-21c6-45d4-90e0-ec68be6fad53" />

## Guide to Generative Adversarial Networks (GANs)

Sources: https://jonathan-hui.medium.com/gan-gan-series-2d279f906e7b

GAN composes of two deep networks, the generator, and the discriminator. First, we sample some noise z using a normal or uniform distribution. With z as an input, we use a generator G to create an image x (x=G(z)). So what is this magic generator G? It performs multiple transposed convolutions to upsample z to generate the image x. But a generator alone will just create random noise. Conceptually, the discriminator in GAN provides guidance to the generator on what images to create.

By training with real images and generated images, GAN builds a discriminator to learn what features make images real. Then the same discriminator will provide feedback to the generator to create real images. So how is it done technically? The discriminator looks at real images (training samples) and generated images separately. It distinguishes whether the input image to the discriminator is real or generated. The output D(X) is the probability that the input x is real, i.e. P(class of input = real image).

We train the discriminator just like a deep network classifier. If the input is real, we want D(x)=1. If it is generated, it should be zero. Through this process, the discriminator identifies features that contribute to real images. On the other hand, we want the generator to create images with D(x) = 1 (matching the real image). So we can train the generator by backpropagation this target value all the way back to the generator, i.e. we train the generator to create images that towards what the discriminator thinks it is real.

We train both networks in alternating steps and lock them into a fierce competition to improve themselves. Eventually, the discriminator identifies the tiny difference between the real and the generated, and the generator creates images that the discriminator cannot tell the difference. The GAN model eventually converges and produces natural look images.

Now, we will go through some simple equations. The discriminator outputs a value D(x) indicating the chance that x is a real image. Our objective is to maximize the chance to recognize real images as real and generated images as fake. i.e. the maximum likelihood of the observed data. To measure the loss, we use cross-entropy as in most Deep Learning: p log(q). For real image, p (the true label for real images) equals to 1. For generated images, we reverse the label (i.e. one minus label). So the objective becomes:

![image](https://github.com/user-attachments/assets/422ff551-cb17-4182-a589-0fce22c9ece6)

On the generator side, its objective function wants the model to generate images with the highest possible value of D(x) to fool the discriminator.

![image](https://github.com/user-attachments/assets/14d881be-ce84-44c7-8f97-a9c501df6a63)

We often define GAN as a minimax game which G wants to minimize V while D wants to maximize it.

![image](https://github.com/user-attachments/assets/87f17112-2fc4-4623-8139-95d16f05abbd)

Once both objective functions are defined, they are learned jointly by the alternating gradient descent algorithm. We fix the generator model’s parameters and perform a single iteration of gradient ascend on the discriminator using the real and the generated images. Then we switch sides. Fix the discriminator and train the generator for another single iteration. We train both networks in alternating steps until the generator produces good-quality images. The following summarizes the data flow and the gradients used for the backpropagation.

![image](https://github.com/user-attachments/assets/2fc2d654-2932-4c7e-9aef-20c2d6405e44)

![image](https://github.com/user-attachments/assets/9a236306-cffb-4c37-8c24-f102566b5ff1)

Why is it so hard to train GANs? 

## Guide to Variational Autoencoders (VAE)

Sources: https://medium.com/data-science/intuitively-understanding-variational-autoencoders-1bfe67eb5daf

An autoencoder network is actually a pair of two connected networks, an encoder and a decoder. An encoder network takes in an input, and converts it into a smaller, dense representation, which the decoder network can use to convert it back to the original input. The encoder is simply is a network that takes in an input and produces a much smaller representation (the encoding), that contains enough information for the next part of the network to process it into the desired output format. Autoencoders take this idea, and slightly flip it on its head, by making the encoder generate encodings specifically useful for reconstructing its own input.

The entire network is usually trained as a whole. The loss function is usually either the mean-squared error or cross-entropy between the output and the input, known as the reconstruction loss, which penalizes the network for creating outputs different from the input. As the encoding (which is simply the output of the hidden layer in the middle) has far less units than the input, the encoder must choose to discard information. The encoder learns to preserve as much of the relevant information as possible in the limited encoding, and intelligently discard irrelevant parts. The decoder learns to take the encoding and properly reconstruct it into a full image. Together, they form an autoencoder.

The problem with standard autoencoders? The fundamental problem with autoencoders, for generation, is that the latent space they convert their inputs to and where their encoded vectors lie, may not be continuous, or allow easy interpolation. This is fine if you’re just replicating the same images. But when you’re building a generative model, you don’t want to prepare to replicate the same image you put in. You want to randomly sample from the latent space, or generate variations on an input image, from a continuous latent space. If the space has discontinuities (eg. gaps between clusters) and you sample/generate a variation from there, the decoder will simply generate an unrealistic output, because the decoder has no idea how to deal with that region of the latent space. During training, it never saw encoded vectors coming from that region of latent space.

Variational Autoencoders (VAEs) have one fundamentally unique property that separates them from vanilla autoencoders, and it is this property that makes them so useful for generative modeling: their latent spaces are, by design, continuous, allowing easy random sampling and interpolation. 

It achieves this by doing something that seems rather surprising at first: making its encoder not output an encoding vector of size n, rather, outputting two vectors of size n: a vector of means, μ, and another vector of standard deviations, σ. They form the parameters of a vector of random variables of length n, with the i th element of μ and σ being the mean and standard deviation of the i th random variable, X i, from which we sample, to obtain the sampled encoding which we pass onward to the decoder:

![image](https://github.com/user-attachments/assets/d9ea5b22-dece-441e-8e34-61f3e08c6150)

This stochastic generation means, that even for the same input, while the mean and standard deviations remain the same, the actual encoding will somewhat vary on every single pass simply due to sampling. As encodings are generated at random from anywhere inside the “circle” (the distribution), the decoder learns that not only is a single point in latent space referring to a sample of that class, but all nearby points refer to the same as well. This allows the decoder to not just decode single, specific encodings in the latent space (leaving the decodable latent space discontinuous), but ones that slightly vary too, as the decoder is exposed to a range of variations of the encoding of the same input during training.

The model is now exposed to a certain degree of local variation by varying the encoding of one sample, resulting in smooth latent spaces on a local scale, that is, for similar samples. Ideally, we want overlap between samples that are not very similar too, in order to interpolate between classes. However, since there are no limits on what values vectors μ and σ can take on, the encoder can learn to generate very different μ for different classes, clustering them apart, and minimize σ, making sure the encodings themselves don’t vary much for the same sample (that is, less uncertainty for the decoder). What we ideally want are encodings, all of which are as close as possible to each other while still being distinct, allowing smooth interpolation, and enabling the construction of new samples. In order to force this, we introduce the Kullback–Leibler divergence (KL divergence[2]) into the loss function.

Intuitively, this loss encourages the encoder to distribute all encodings (for all types of inputs, eg. all MNIST numbers), evenly around the center of the latent space. If it tries to “cheat” by clustering them apart into specific regions, away from the origin, it will be penalized.

So how do we actually produce these smooth interpolations we speak of? From here on out, it’s simple vector arithmetic in the latent space. For example, if you wish to generate a new sample halfway between two samples, just find the difference between their mean (μ) vectors, and add half the difference to the original, and then simply decode it.

![image](https://github.com/user-attachments/assets/a29d1a29-3711-43f4-a470-64fa8cf315ed)


## Guide to Diffusion Models


Sources:
https://www.youtube.com/watch?v=HoKDTa5jHvg
https://www.youtube.com/watch?v=a4Yfz2FxXiY

(1) https://arxiv.org/pdf/1503.03585.pdf
(2) https://arxiv.org/pdf/2006.11239
(3) https://arxiv.org/pdf/2102.09672.pdf
(4) https://arxiv.org/pdf/2105.05233.pdf

Another awesome source: https://github.com/diff-usion/Awesome-Diffusion-Models


(1) The essential idea, inspired by non-equilibrium statistical physics, is to systematically and slowly destroy structure in a data distribution through an iterative forward diffusion process. We then learn a reverse diffusion process that restores structure in data, yielding a highly flexible and tractable generative model of the data. 

Let's break this statement down: 1. During the forward diffusion process: Apply noise (sampled from normal distribution) to image in an iterative way until pure noise (that follows the noise distribution aka normal distribution); 2. During the reverse diffusion process: Train a neural network for removing noise in an iterative way. Then, we can give the model an image that consists of random noise (from a normal distribution) and let the model gradually remove the noise until an image is generated (that could be part of the training dataset). 

Why iterative/step-by-step? Learning in this framework involves estimating small per-turbations to a diffusion process. Estimating small perturbations is more tractable than explicitly describing the full distribution with a single, non-analytically-normalizable, potential function. Basically, modeling the distribution directly wouldn't be tractable and lead to worse outcomes.

(2) What are the in-/outputs of the neural network? The input of the neural network is a noised image. The output of the neural network is the mean of the noise. The mean of the noice represents the noice of the image, which can then be substracted from the noised image to get a less noised image. The standard deviation is fixed. So, this doesn't need to be predicted. In (3), the standard deviation of the noise is learned too. 

Important note about the iterative approach: in (2), they apply a linear schedule: add same amount of noise to image in every time step. This approach is sub-optimal as the last couple of time steps already seem like complete noise (redundant) and the information might be destroyed too fast. Hence, in (3), they apply a cosine schedule: this approach solves both problems from the linear schedule. 

What does the architecture of the neural network look like? In (2), they applied U-net-like architecture: 1. The image is downsampled using Resnet- and Downsample-blocks; 2. Bottleneck-block; 3. Upsample to original size using Resnet- and Upsample-blocks. Also, attention blocks are added at certain resolutions (before and after bottleneck-block?) and skip connections between layers of the same spatial resolutions. The model looks the same at each time step. However, different amount of noise is added at different time steps (remember the note about the schedules). A sinusoidal embedding is added to each residual block to inform the model about the time step. The model can then remove different amount of noise at different time steps. In (3), many improvements are added: 1. Increase depth and decrease width; 2. Add more attention blocks and increase attention heads; 3. Use BigGAN residual blocks; 4. Add addaptive group normalization. In (4), classifier guidance is added. 

This should give a basic intuition of how diffusion models work. Now, let's get into the math. 

Some basic notation: x_t = image at timestep t (x_O is the original image and x_T is the final image (that follows isotropic gaussian) where T depends on the paper); q(x_t|x_(t-1)) = the forward diffusion process: takes an image x_(t-1) and returns an image with noise added x_t; p(x_(t-1)|x_t) = the reverse diffusion process: takes an image x_t and returns a sample/image with noise removed x_(t-1). 

The formula for the forward diffusion process for a single time step: The beta-term refers to the schedule. In case of linear schedule: beta grows linearly from beta_start (f.e. 0.0001) to beta_end (f.e. 0.02). To apply X time steps, repeat the formule X times. However, there's a way to define this process in a single step. 

<img width="1265" alt="Screenshot 2025-05-01 at 12 06 47" src="https://github.com/user-attachments/assets/5e0249dd-42b3-4a72-b068-ea265d5947b9" />

Some more notation: alpha_t = 1 - beta_t; alpha_t = the cumulative product of all alphas from 0 until t (for example, if t=3, then alpha_t = alpha_1 * alpha_2 * alpha_3). 

Then, we can re-write the formula for the entire forward diffusion process as follows (important: reparameterization trick is applied in step 1):

<img width="814" alt="Screenshot 2025-05-01 at 12 17 39" src="https://github.com/user-attachments/assets/abe05e98-2a70-41d2-8136-d90cc1880789" />

The formula for the reverse diffusion process for a single time step: the neural network only needs to predict the mean. The variance is fixed according to a certain schedule and thus doesn't need to be predicted. 

<img width="1160" alt="Screenshot 2025-05-01 at 12 21 47" src="https://github.com/user-attachments/assets/1f04f8bb-50c4-43d3-b4ff-86c52d5e338f" />

The formula for the loss function: the negative log likelihood of the probability of x_0. There's a problem, this is difficult to calculate in practice as the probability of x_0 depends on all time steps before x_0. The solution is calcultating the variational lower bound of the objective. 

This lower bound is obtained by subtracting the KL divergence between the approximate (forward) and true (reverse) distributions from the original objective. Since KL divergence is always non-negative, subtracting it guarantees the result is less than or equal to the true log-likelihood—hence a "lower bound."

Because we're minimizing the negative log-likelihood, subtracting KL (i.e., maximizing the ELBO) is equivalent to adding KL divergence to the loss, helping guide the model toward a better approximation of the data distribution. 

<img width="803" alt="Screenshot 2025-05-01 at 13 26 41" src="https://github.com/user-attachments/assets/700defe9-7315-45c6-9cc8-6a14ab8708f3" />

However, the lower bound is still not computable since the uncomputable quantity is still in the formula. So, we need some reformulations to arrive at a better looking quantity. HERE I COULDNT FOLLOW THE VIDEO ANYMORE

Let's take a look at the algorithms for training: sample image from dataset, time step from uniform distribution, and noise from normal distribution. Then, the objective is optimized using gradient descent until convergence.  

<img width="658" alt="Screenshot 2025-05-01 at 13 42 08" src="https://github.com/user-attachments/assets/17417ea6-79a7-4439-9828-8d20753a2595" />

Let's take a look at the algorithms for sampling: sample x_T from normal distribution. Then, iteratively use formula (see above?) to sample x_(t-1). if t=1, then we don't want to add noise (it would make the quality of the final image worse). Finally, return x_O which should be an image that could come from the dataset.  

<img width="651" alt="Screenshot 2025-05-01 at 13 45 45" src="https://github.com/user-attachments/assets/0a9c96c1-f0f9-464d-9d2e-716b7805d89f" />


To implement a diffusion model, we need a noise scheduler, a neural network and timestep encoding. 

During the forward process, noise is added sequentially to the images. This markov is denoted with q and can be formulated as follows:

<img width="1270" alt="Screenshot 2025-05-01 at 15 23 27" src="https://github.com/user-attachments/assets/41ef9aa0-43c1-4e37-9b8d-069ce2277b97" />

The noise that is added to an image x_t depends only on the previous image x_(t-1). The noise for an image x_t is sampled with a conditional gaussian distribution where the mean depends on the previous image x_(t-1) and a fixed variance. The sequence of betas is the so-called variance schedule that described how much noise needs to be added for each of the time steps. So, the mean of the distribution for image x_t is the previous image x_(t-1) multiplied by the beta-term. The variance of the distribution for image x_t is the beta multiplied by the identity. 

Some more intuition about beta: In case of large beta, the pixel distribution is wider and more shifted. When sampling from this distribution, more noise is added. Eventually, beta controls how fast we converge to a standard gaussian distribution (zero mean). The right amount of noise needs to be added such that we arrive at an isotropic gaussian distribution (zero mean and fixed variance in all directions). Noise shouldn't be added too fast/slow, there are different schedules to get this. 

There's an even better way to do this, the closed-form forward process: instead of doing the forward pass sequentially, it's possible to sample the noisy version for any timestamp t. For this, the closed-form of the mean and variance need to be pre-computed based on the cumulative variance schedules. Imagine T=200 and beta (the variance schedule) is defined according to linear scheduler, introduce a new term alpha = 1-beta. By calculating the cumulative product of the alpha-terms, we get another new term alpha_overlined. This term can be used to re-write the forward process formula:

<img width="1382" alt="Screenshot 2025-05-01 at 15 43 38" src="https://github.com/user-attachments/assets/f39606c4-e1d2-4654-b5b3-59218aacf5d9" />

During the backward process, the U-net architecture is used. The input passes a series of convolutional and downsampling layers until a bottleneck is reached. Then, the tensors pass convolutional and upsamling layers until they have the same shape as the input. Also, residual connections, batch normalization and attention modules are added. The model takes in a noisy image and predicts the noise in the image. The output is one value per pixel which represents the mean of the gaussian distribution of the images (denoising score matching). The neural network is the same for every time step. So, the models gets timestep information using timestep embedding. 

The backward process can be formulated as follows:

<img width="1374" alt="Screenshot 2025-05-01 at 16 14 27" src="https://github.com/user-attachments/assets/34d9b3eb-c813-4d4a-a418-d28f1d947967" />

Start in x_T with gaussian noise with a zero mean and unit variance. In a seuence, the transition from one latence to the next is predicted. The model learns the probability density of an earlier time step given the current time step. During training, randomly sample time steps (don't go through entire seqeunce). During inference, iterate through entire sequence (from pure noise x_T to final image x_O). The density p is defined by the predicted gaussian noise distribution in the image. To get the actual image of time step t-1 x_(t-1), substract the predicted noise from the image in time step t x_t. 

How to consider the time step in the model? Positional embeddings are used. This is a clever way to encode discrete positional information, such as sequence steps. A positional embedding is a vector that looks different for every time step. The positional embeddings are added as additional input, besided the noisy image. 

Diffusion models are optimized with the variational lower bound. The final loss can be formulated as follows:

<img width="1004" alt="Screenshot 2025-05-01 at 16 21 56" src="https://github.com/user-attachments/assets/89c71387-32f6-4da3-9d2f-c1e87b84d806" />

The loss is the L2-distance of the predicted noise and the actual noise in the image (see previous video for better understanding of derivations). 

## Guide to Stable Diffusion Models

Some notes: 
- Likelihood-based models are generative models that explicitly learn a probability distribution over data by maximizing the likelihood of observed samples, enabling stable training and better coverage of data modes.
- Diffusion models try so hard to generate every tiny detail of images—even ones humans can’t see—that they use a lot of compute unnecessarily. A method called the reweighted variational objective helps avoid this by paying less attention to the early (most noisy) stages of image generation, where fine details don't matter much.
- The authors propose replacing the inefficient, two-stage learning process of pixel-space diffusion models with a more efficient setup: they use an autoencoder to handle perceptual compression upfront, then train the diffusion model in the resulting latent space to focus on learning high-level semantic structure, enabling faster and more scalable high-resolution image synthesis.
- Unlike earlier methods that shrunk images a lot to save compute, LDMs use a smarter, learned compression (latent space) that keeps images meaningful but compact, allowing fast and high-quality image generation without sacrificing resolution.
- The power of diffusion models comes from using UNet architectures, which naturally suit image data. By training with a reweighted objective that focuses more on noise levels critical to human perception, these models produce higher-quality images. In doing so, they behave like lossy compressors—preserving the key semantic and visual information while discarding imperceptible details—enabling a tunable trade-off between image quality and compression efficiency.
- In conditional diffusion models, cross-attention allows the model to guide image generation using inputs like text. The image features act as queries, asking “what should go here?”, while the text embeddings serve as keys and values, supplying relevant semantic content. The model uses similarity between queries and keys to decide which words influence which parts of the image, allowing it to inject text-based guidance into the denoising process and produce images that match the given prompt.



# Latent Diffusion Models
[arXiv](https://arxiv.org/abs/2112.10752) | [BibTeX](#bibtex)

<p align="center">
<img src=assets/results.gif />
</p>



[**High-Resolution Image Synthesis with Latent Diffusion Models**](https://arxiv.org/abs/2112.10752)<br/>
[Robin Rombach](https://github.com/rromb)\*,
[Andreas Blattmann](https://github.com/ablattmann)\*,
[Dominik Lorenz](https://github.com/qp-qp)\,
[Patrick Esser](https://github.com/pesser),
[Björn Ommer](https://hci.iwr.uni-heidelberg.de/Staff/bommer)<br/>
\* equal contribution

<p align="center">
<img src=assets/modelfigure.png />
</p>

## News

### July 2022
- Inference code and model weights to run our [retrieval-augmented diffusion models](https://arxiv.org/abs/2204.11824) are now available. See [this section](#retrieval-augmented-diffusion-models).
### April 2022
- Thanks to [Katherine Crowson](https://github.com/crowsonkb), classifier-free guidance received a ~2x speedup and the [PLMS sampler](https://arxiv.org/abs/2202.09778) is available. See also [this PR](https://github.com/CompVis/latent-diffusion/pull/51).

- Our 1.45B [latent diffusion LAION model](#text-to-image) was integrated into [Huggingface Spaces 🤗](https://huggingface.co/spaces) using [Gradio](https://github.com/gradio-app/gradio). Try out the Web Demo: [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/multimodalart/latentdiffusion)

- More pre-trained LDMs are available: 
  - A 1.45B [model](#text-to-image) trained on the [LAION-400M](https://arxiv.org/abs/2111.02114) database.
  - A class-conditional model on ImageNet, achieving a FID of 3.6 when using [classifier-free guidance](https://openreview.net/pdf?id=qw8AKxfYbI) Available via a [colab notebook](https://colab.research.google.com/github/CompVis/latent-diffusion/blob/main/scripts/latent_imagenet_diffusion.ipynb) [![][colab]][colab-cin].
  
## Requirements
A suitable [conda](https://conda.io/) environment named `ldm` can be created
and activated with:

```
conda env create -f environment.yaml
conda activate ldm
```

# Pretrained Models
A general list of all available checkpoints is available in via our [model zoo](#model-zoo).
If you use any of these models in your work, we are always happy to receive a [citation](#bibtex).

## Retrieval Augmented Diffusion Models
![rdm-figure](assets/rdm-preview.jpg)
We include inference code to run our retrieval-augmented diffusion models (RDMs) as described in [https://arxiv.org/abs/2204.11824](https://arxiv.org/abs/2204.11824).


To get started, install the additionally required python packages into your `ldm` environment
```shell script
pip install transformers==4.19.2 scann kornia==0.6.4 torchmetrics==0.6.0
pip install git+https://github.com/arogozhnikov/einops.git
```
and download the trained weights (preliminary ceckpoints):

```bash
mkdir -p models/rdm/rdm768x768/
wget -O models/rdm/rdm768x768/model.ckpt https://ommer-lab.com/files/rdm/model.ckpt
```
As these models are conditioned on a set of CLIP image embeddings, our RDMs support different inference modes, 
which are described in the following.
#### RDM with text-prompt only (no explicit retrieval needed)
Since CLIP offers a shared image/text feature space, and RDMs learn to cover a neighborhood of a given
example during training, we can directly take a CLIP text embedding of a given prompt and condition on it.
Run this mode via
```
python scripts/knn2img.py  --prompt "a happy bear reading a newspaper, oil on canvas"
```

#### RDM with text-to-image retrieval

To be able to run a RDM conditioned on a text-prompt and additionally images retrieved from this prompt, you will also need to download the corresponding retrieval database. 
We provide two distinct databases extracted from the [Openimages-](https://storage.googleapis.com/openimages/web/index.html) and [ArtBench-](https://github.com/liaopeiyuan/artbench) datasets. 
Interchanging the databases results in different capabilities of the model as visualized below, although the learned weights are the same in both cases. 

Download the retrieval-databases which contain the retrieval-datasets ([Openimages](https://storage.googleapis.com/openimages/web/index.html) (~11GB) and [ArtBench](https://github.com/liaopeiyuan/artbench) (~82MB)) compressed into CLIP image embeddings:
```bash
mkdir -p data/rdm/retrieval_databases
wget -O data/rdm/retrieval_databases/artbench.zip https://ommer-lab.com/files/rdm/artbench_databases.zip
wget -O data/rdm/retrieval_databases/openimages.zip https://ommer-lab.com/files/rdm/openimages_database.zip
unzip data/rdm/retrieval_databases/artbench.zip -d data/rdm/retrieval_databases/
unzip data/rdm/retrieval_databases/openimages.zip -d data/rdm/retrieval_databases/
```
We also provide trained [ScaNN](https://github.com/google-research/google-research/tree/master/scann) search indices for ArtBench. Download and extract via
```bash
mkdir -p data/rdm/searchers
wget -O data/rdm/searchers/artbench.zip https://ommer-lab.com/files/rdm/artbench_searchers.zip
unzip data/rdm/searchers/artbench.zip -d data/rdm/searchers
```

Since the index for OpenImages is large (~21 GB), we provide a script to create and save it for usage during sampling. Note however,
that sampling with the OpenImages database will not be possible without this index. Run the script via
```bash
python scripts/train_searcher.py
```

Retrieval based text-guided sampling with visual nearest neighbors can be started via 
```
python scripts/knn2img.py  --prompt "a happy pineapple" --use_neighbors --knn <number_of_neighbors> 
```
Note that the maximum supported number of neighbors is 20. 
The database can be changed via the cmd parameter ``--database`` which can be `[openimages, artbench-art_nouveau, artbench-baroque, artbench-expressionism, artbench-impressionism, artbench-post_impressionism, artbench-realism, artbench-renaissance, artbench-romanticism, artbench-surrealism, artbench-ukiyo_e]`.
For using `--database openimages`, the above script (`scripts/train_searcher.py`) must be executed before.
Due to their relatively small size, the artbench datasetbases are best suited for creating more abstract concepts and do not work well for detailed text control. 


#### Coming Soon
- better models
- more resolutions
- image-to-image retrieval

## Text-to-Image
![text2img-figure](assets/txt2img-preview.png) 


Download the pre-trained weights (5.7GB)
```
mkdir -p models/ldm/text2img-large/
wget -O models/ldm/text2img-large/model.ckpt https://ommer-lab.com/files/latent-diffusion/nitro/txt2img-f8-large/model.ckpt
```
and sample with
```
python scripts/txt2img.py --prompt "a virus monster is playing guitar, oil on canvas" --ddim_eta 0.0 --n_samples 4 --n_iter 4 --scale 5.0  --ddim_steps 50
```
This will save each sample individually as well as a grid of size `n_iter` x `n_samples` at the specified output location (default: `outputs/txt2img-samples`).
Quality, sampling speed and diversity are best controlled via the `scale`, `ddim_steps` and `ddim_eta` arguments.
As a rule of thumb, higher values of `scale` produce better samples at the cost of a reduced output diversity.   
Furthermore, increasing `ddim_steps` generally also gives higher quality samples, but returns are diminishing for values > 250.
Fast sampling (i.e. low values of `ddim_steps`) while retaining good quality can be achieved by using `--ddim_eta 0.0`.  
Faster sampling (i.e. even lower values of `ddim_steps`) while retaining good quality can be achieved by using `--ddim_eta 0.0` and `--plms` (see [Pseudo Numerical Methods for Diffusion Models on Manifolds](https://arxiv.org/abs/2202.09778)).

#### Beyond 256²

For certain inputs, simply running the model in a convolutional fashion on larger features than it was trained on
can sometimes result in interesting results. To try it out, tune the `H` and `W` arguments (which will be integer-divided
by 8 in order to calculate the corresponding latent size), e.g. run

```
python scripts/txt2img.py --prompt "a sunset behind a mountain range, vector image" --ddim_eta 1.0 --n_samples 1 --n_iter 1 --H 384 --W 1024 --scale 5.0  
```
to create a sample of size 384x1024. Note, however, that controllability is reduced compared to the 256x256 setting. 

The example below was generated using the above command. 
![text2img-figure-conv](assets/txt2img-convsample.png)



## Inpainting
![inpainting](assets/inpainting.png)

Download the pre-trained weights
```
wget -O models/ldm/inpainting_big/last.ckpt https://heibox.uni-heidelberg.de/f/4d9ac7ea40c64582b7c9/?dl=1
```

and sample with
```
python scripts/inpaint.py --indir data/inpainting_examples/ --outdir outputs/inpainting_results
```
`indir` should contain images `*.png` and masks `<image_fname>_mask.png` like
the examples provided in `data/inpainting_examples`.

## Class-Conditional ImageNet

Available via a [notebook](scripts/latent_imagenet_diffusion.ipynb) [![][colab]][colab-cin].
![class-conditional](assets/birdhouse.png)

[colab]: <https://colab.research.google.com/assets/colab-badge.svg>
[colab-cin]: <https://colab.research.google.com/github/CompVis/latent-diffusion/blob/main/scripts/latent_imagenet_diffusion.ipynb>


## Unconditional Models

We also provide a script for sampling from unconditional LDMs (e.g. LSUN, FFHQ, ...). Start it via

```shell script
CUDA_VISIBLE_DEVICES=<GPU_ID> python scripts/sample_diffusion.py -r models/ldm/<model_spec>/model.ckpt -l <logdir> -n <\#samples> --batch_size <batch_size> -c <\#ddim steps> -e <\#eta> 
```

# Train your own LDMs

## Data preparation

### Faces 
For downloading the CelebA-HQ and FFHQ datasets, proceed as described in the [taming-transformers](https://github.com/CompVis/taming-transformers#celeba-hq) 
repository.

### LSUN 

The LSUN datasets can be conveniently downloaded via the script available [here](https://github.com/fyu/lsun).
We performed a custom split into training and validation images, and provide the corresponding filenames
at [https://ommer-lab.com/files/lsun.zip](https://ommer-lab.com/files/lsun.zip). 
After downloading, extract them to `./data/lsun`. The beds/cats/churches subsets should
also be placed/symlinked at `./data/lsun/bedrooms`/`./data/lsun/cats`/`./data/lsun/churches`, respectively.

### ImageNet
The code will try to download (through [Academic
Torrents](http://academictorrents.com/)) and prepare ImageNet the first time it
is used. However, since ImageNet is quite large, this requires a lot of disk
space and time. If you already have ImageNet on your disk, you can speed things
up by putting the data into
`${XDG_CACHE}/autoencoders/data/ILSVRC2012_{split}/data/` (which defaults to
`~/.cache/autoencoders/data/ILSVRC2012_{split}/data/`), where `{split}` is one
of `train`/`validation`. It should have the following structure:

```
${XDG_CACHE}/autoencoders/data/ILSVRC2012_{split}/data/
├── n01440764
│   ├── n01440764_10026.JPEG
│   ├── n01440764_10027.JPEG
│   ├── ...
├── n01443537
│   ├── n01443537_10007.JPEG
│   ├── n01443537_10014.JPEG
│   ├── ...
├── ...
```

If you haven't extracted the data, you can also place
`ILSVRC2012_img_train.tar`/`ILSVRC2012_img_val.tar` (or symlinks to them) into
`${XDG_CACHE}/autoencoders/data/ILSVRC2012_train/` /
`${XDG_CACHE}/autoencoders/data/ILSVRC2012_validation/`, which will then be
extracted into above structure without downloading it again.  Note that this
will only happen if neither a folder
`${XDG_CACHE}/autoencoders/data/ILSVRC2012_{split}/data/` nor a file
`${XDG_CACHE}/autoencoders/data/ILSVRC2012_{split}/.ready` exist. Remove them
if you want to force running the dataset preparation again.


## Model Training

Logs and checkpoints for trained models are saved to `logs/<START_DATE_AND_TIME>_<config_spec>`.

### Training autoencoder models

Configs for training a KL-regularized autoencoder on ImageNet are provided at `configs/autoencoder`.
Training can be started by running
```
CUDA_VISIBLE_DEVICES=<GPU_ID> python main.py --base configs/autoencoder/<config_spec>.yaml -t --gpus 0,    
```
where `config_spec` is one of {`autoencoder_kl_8x8x64`(f=32, d=64), `autoencoder_kl_16x16x16`(f=16, d=16), 
`autoencoder_kl_32x32x4`(f=8, d=4), `autoencoder_kl_64x64x3`(f=4, d=3)}.

For training VQ-regularized models, see the [taming-transformers](https://github.com/CompVis/taming-transformers) 
repository.

### Training LDMs 

In ``configs/latent-diffusion/`` we provide configs for training LDMs on the LSUN-, CelebA-HQ, FFHQ and ImageNet datasets. 
Training can be started by running

```shell script
CUDA_VISIBLE_DEVICES=<GPU_ID> python main.py --base configs/latent-diffusion/<config_spec>.yaml -t --gpus 0,
``` 

where ``<config_spec>`` is one of {`celebahq-ldm-vq-4`(f=4, VQ-reg. autoencoder, spatial size 64x64x3),`ffhq-ldm-vq-4`(f=4, VQ-reg. autoencoder, spatial size 64x64x3),
`lsun_bedrooms-ldm-vq-4`(f=4, VQ-reg. autoencoder, spatial size 64x64x3),
`lsun_churches-ldm-vq-4`(f=8, KL-reg. autoencoder, spatial size 32x32x4),`cin-ldm-vq-8`(f=8, VQ-reg. autoencoder, spatial size 32x32x4)}.

# Model Zoo 

## Pretrained Autoencoding Models
![rec2](assets/reconstruction2.png)

All models were trained until convergence (no further substantial improvement in rFID).

| Model                   | rFID vs val | train steps           |PSNR           | PSIM          | Link                                                                                                                                                  | Comments              
|-------------------------|------------|----------------|----------------|---------------|-------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------|
| f=4, VQ (Z=8192, d=3)   | 0.58       | 533066 | 27.43  +/- 4.26 | 0.53 +/- 0.21 |     https://ommer-lab.com/files/latent-diffusion/vq-f4.zip                   |  |
| f=4, VQ (Z=8192, d=3)   | 1.06       | 658131 | 25.21 +/-  4.17 | 0.72 +/- 0.26 | https://heibox.uni-heidelberg.de/f/9c6681f64bb94338a069/?dl=1  | no attention          |
| f=8, VQ (Z=16384, d=4)  | 1.14       | 971043 | 23.07 +/- 3.99 | 1.17 +/- 0.36 |       https://ommer-lab.com/files/latent-diffusion/vq-f8.zip                     |                       |
| f=8, VQ (Z=256, d=4)    | 1.49       | 1608649 | 22.35 +/- 3.81 | 1.26 +/- 0.37 |   https://ommer-lab.com/files/latent-diffusion/vq-f8-n256.zip |  
| f=16, VQ (Z=16384, d=8) | 5.15       | 1101166 | 20.83 +/- 3.61 | 1.73 +/- 0.43 |             https://heibox.uni-heidelberg.de/f/0e42b04e2e904890a9b6/?dl=1                        |                       |
|                         |            |  |                |               |                                                                                                                                                    |                       |
| f=4, KL                 | 0.27       | 176991 | 27.53 +/- 4.54 | 0.55 +/- 0.24 |     https://ommer-lab.com/files/latent-diffusion/kl-f4.zip                                   |                       |
| f=8, KL                 | 0.90       | 246803 | 24.19 +/- 4.19 | 1.02 +/- 0.35 |             https://ommer-lab.com/files/latent-diffusion/kl-f8.zip                            |                       |
| f=16, KL     (d=16)     | 0.87       | 442998 | 24.08 +/- 4.22 | 1.07 +/- 0.36 |      https://ommer-lab.com/files/latent-diffusion/kl-f16.zip                                  |                       |
 | f=32, KL     (d=64)     | 2.04       | 406763 | 22.27 +/- 3.93 | 1.41 +/- 0.40 |             https://ommer-lab.com/files/latent-diffusion/kl-f32.zip                            |                       |

### Get the models

Running the following script downloads und extracts all available pretrained autoencoding models.   
```shell script
bash scripts/download_first_stages.sh
```

The first stage models can then be found in `models/first_stage_models/<model_spec>`



## Pretrained LDMs
| Datset                          |   Task    | Model        | FID           | IS              | Prec | Recall | Link                                                                                                                                                                                   | Comments                                        
|---------------------------------|------|--------------|---------------|-----------------|------|------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------|
| CelebA-HQ                       | Unconditional Image Synthesis    |  LDM-VQ-4 (200 DDIM steps, eta=0)| 5.11 (5.11)          | 3.29            | 0.72    | 0.49 |    https://ommer-lab.com/files/latent-diffusion/celeba.zip     |                                                 |  
| FFHQ                            | Unconditional Image Synthesis    |  LDM-VQ-4 (200 DDIM steps, eta=1)| 4.98 (4.98)  | 4.50 (4.50)   | 0.73 | 0.50 |              https://ommer-lab.com/files/latent-diffusion/ffhq.zip                                              |                                                 |
| LSUN-Churches                   | Unconditional Image Synthesis   |  LDM-KL-8 (400 DDIM steps, eta=0)| 4.02 (4.02) | 2.72 | 0.64 | 0.52 |         https://ommer-lab.com/files/latent-diffusion/lsun_churches.zip        |                                                 |  
| LSUN-Bedrooms                   | Unconditional Image Synthesis   |  LDM-VQ-4 (200 DDIM steps, eta=1)| 2.95 (3.0)          | 2.22 (2.23)| 0.66 | 0.48 | https://ommer-lab.com/files/latent-diffusion/lsun_bedrooms.zip |                                                 |  
| ImageNet                        | Class-conditional Image Synthesis | LDM-VQ-8 (200 DDIM steps, eta=1) | 7.77(7.76)* /15.82** | 201.56(209.52)* /78.82** | 0.84* / 0.65** | 0.35* / 0.63** |   https://ommer-lab.com/files/latent-diffusion/cin.zip                                                                   | *: w/ guiding, classifier_scale 10  **: w/o guiding, scores in bracket calculated with script provided by [ADM](https://github.com/openai/guided-diffusion) |   
| Conceptual Captions             |  Text-conditional Image Synthesis | LDM-VQ-f4 (100 DDIM steps, eta=0) | 16.79         | 13.89           | N/A | N/A |              https://ommer-lab.com/files/latent-diffusion/text2img.zip                                | finetuned from LAION                            |   
| OpenImages                      | Super-resolution   | LDM-VQ-4     | N/A            | N/A               | N/A    | N/A    |                                    https://ommer-lab.com/files/latent-diffusion/sr_bsr.zip                                    | BSR image degradation                           |
| OpenImages                      | Layout-to-Image Synthesis    | LDM-VQ-4 (200 DDIM steps, eta=0) | 32.02         | 15.92           | N/A    | N/A    |                  https://ommer-lab.com/files/latent-diffusion/layout2img_model.zip                                           |                                                 | 
| Landscapes      |  Semantic Image Synthesis   | LDM-VQ-4  | N/A             | N/A               | N/A    | N/A    |           https://ommer-lab.com/files/latent-diffusion/semantic_synthesis256.zip                                    |                                                 |
| Landscapes       |  Semantic Image Synthesis   | LDM-VQ-4  | N/A             | N/A               | N/A    | N/A    |           https://ommer-lab.com/files/latent-diffusion/semantic_synthesis.zip                                    |             finetuned on resolution 512x512                                     |


### Get the models

The LDMs listed above can jointly be downloaded and extracted via

```shell script
bash scripts/download_models.sh
```

The models can then be found in `models/ldm/<model_spec>`.



## Coming Soon...

* More inference scripts for conditional LDMs.
* In the meantime, you can play with our colab notebook https://colab.research.google.com/drive/1xqzUi2iXQXDqXBHQGP9Mqt2YrYW6cx-J?usp=sharing

## Comments 

- Our codebase for the diffusion models builds heavily on [OpenAI's ADM codebase](https://github.com/openai/guided-diffusion)
and [https://github.com/lucidrains/denoising-diffusion-pytorch](https://github.com/lucidrains/denoising-diffusion-pytorch). 
Thanks for open-sourcing!

- The implementation of the transformer encoder is from [x-transformers](https://github.com/lucidrains/x-transformers) by [lucidrains](https://github.com/lucidrains?tab=repositories). 


## BibTeX

```
@misc{rombach2021highresolution,
      title={High-Resolution Image Synthesis with Latent Diffusion Models}, 
      author={Robin Rombach and Andreas Blattmann and Dominik Lorenz and Patrick Esser and Björn Ommer},
      year={2021},
      eprint={2112.10752},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}

@misc{https://doi.org/10.48550/arxiv.2204.11824,
  doi = {10.48550/ARXIV.2204.11824},
  url = {https://arxiv.org/abs/2204.11824},
  author = {Blattmann, Andreas and Rombach, Robin and Oktay, Kaan and Ommer, Björn},
  keywords = {Computer Vision and Pattern Recognition (cs.CV), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {Retrieval-Augmented Diffusion Models},
  publisher = {arXiv},
  year = {2022},  
  copyright = {arXiv.org perpetual, non-exclusive license}
}


```


