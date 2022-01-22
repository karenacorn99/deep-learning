## Deep Learning Reading Log (cpsc 552)

Reading List:
***
Variations on SGD:
[The Loss Surfaces of Multilayer Networks](https://arxiv.org/abs/1412.0233)
[Adam](https://towardsdatascience.com/adam-latest-trends-in-deep-learning-optimization-6be9a291375c)
***
Unsupervised Learning and Autoencoders:
Hinton and Salakhutdinov Science 2006
Coifman and Lafon Applied and Computational Harmonic Analysis 2006
Belkin & Niyogi, Neural Computation 2003
Alain & Bengio, JMLR 2014
***
Generative Models and VAEs:
Kingma & Welling ICLR 2014
Dziugaite et al.  UAI 2015
Bengio et al. NeurIPS 2013
Lopez et al. Nature Methods 2018
Goodfellow Chapter 20
***
GANs:
[From GAN to WGAN](https://lilianweng.github.io/lil-log/2017/08/20/from-GAN-to-WGAN.html)
[Maximizing likelihood is equivalent to minimizing KL-Divergence](https://agustinus.kristia.de/techblog/2017/01/26/kl-mle/)
Goodfellow et al 2014 paper
Arjovsky et al 2017 paper
[StyleGAN](https://arxiv.org/pdf/1812.04948.pdf)
[InfoGAN](https://arxiv.org/pdf/1606.03657.pdf)
[DiscoGAN](https://arxiv.org/abs/1703.05192)
[CycleGAN](https://arxiv.org/abs/1703.10593)
[MAGAN](http://proceedings.mlr.press/v80/amodio18a.html)
[TraVeLGAN](https://arxiv.org/abs/1902.09631)
[Conditional GAN](https://arxiv.org/pdf/1411.1784.pdf)
***
CNNs:
[Convolutional Neural Networks for Visual Recognition](https://cs231n.github.io/convolutional-networks/)
Szegedy et al. Going deeper with convolutions 2014
He et al. Deep Residual Learning for Image Recognition 2015
Ronneberger et al.  U-Net: Convolutional Networks for Biomedical Image Segmentation 2014
[An Intuitive Explanation of Convolutional Neural Networks](https://ujjwalkarn.me/2016/08/11/intuitive-explanation-convnets/)
[A Simple Guide to the Versions of the Inception Network](https://towardsdatascience.com/a-simple-guide-to-the-versions-of-the-inception-network-7fc52b863202)
***
RNNs & LSTMs:
[word2vec](https://arxiv.org/abs/1301.3781) 
[Luong et al. 2015](https://arxiv.org/abs/1508.04025)
[Hochreiter and Schmidhuber. 1997](https://www.bioinf.jku.at/publications/older/2604.pdf)
[Attention in RNNs blog](https://medium.datadriveninvestor.com/attention-in-rnns-321fbcd64f05)
[Understanding LSTMs blog](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)
***
Transformers:
[Attention and its Different Forms](https://towardsdatascience.com/attention-and-its-different-forms-7fc3674d14dc)
[Illustrated Transformers](http://jalammar.github.io/illustrated-transformer/)
[Attention is All You Need](https://arxiv.org/abs/1706.03762)
Radford et al. (GPT2 paper)
Brown et al. (GPT3 paper)
Child et al. (Sparse Transformer)
[Illustrateed GPT2](https://jalammar.github.io/illustrated-gpt2 (Links to an external site.)/)
[The Journey of OpenAI GPT Models](https://medium.com/walmartglobaltech/the-journey-of-open-ai-gpt-models-32d95b7b7fb2)
***
Graph Neural Networks
Defferrard et al. Convolutional Neural Networks on Graphs
Bruna et al. Spectral Networks and Locally Connected Networks on Graphs
Hamilton et al. [Inductive Representation Learning on Large Graphs](https://cs.stanford.edu/people/jure/pubs/graphsage-nips17.pdf)
Perozzi et al. Deep Walk
Kipf & Welling, Semisupervised Graph Classifiation
[Spectral Graph Convolution Explained and Implemented Step by Step](https://towardsdatascience.com/spectral-graph-convolution-explained-and-implemented-step-by-step-2e495b57f801)
[Anisotropic, Dynamic, Spectral and Multiscale Filters Defined on Graphs](https://towardsdatascience.com/tutorial-on-graph-neural-networks-for-computer-vision-and-beyond-part-2-be6d71d70f49)
[A Gentle Introduction to Graph Neural Networks (Basics, DeepWalk, and GraphSage)](https://towardsdatascience.com/a-gentle-introduction-to-graph-neural-network-basics-deepwalk-and-graphsage-db5d540d50b3)
Gao et al., ICML 2019 [Geometric Scattering for Graph Data Analysis](http://proceedings.mlr.press/v97/gao19e/gao19e.pdf)
Min et al. NeurIPS 2020 [Scattering GCN: Overcoming Oversmoothness in Graph Convolutional Networks](https://arxiv.org/pdf/2003.08414.pdf)
Min et al. [Geometric Scattering Attention Networks](https://arxiv.org/abs/2010.15010)
***
Neural ODEs:
Chen et al. NeurIPS 2019 Neural Ordinary Differential Equations
Tong et al. ICML 2020 TrajectoryNet
Errico 1997 Adjoint Model
[The Story of Adjoint Sensitivity Method from Meteorology](https://towardsdatascience.com/the-story-of-adjoint-sensitivity-method-from-meteorology-906ab2796c73)
[Neural ODEs: breakdown of another deep learning breakthrough](https://towardsdatascience.com/neural-odes-breakdown-of-another-deep-learning-breakthrough-3e78c7213795)
***
Universality of Neural Networks:
DasGupta & Gupta 2002 Johnson 
[An Elementary Proof of a Theorem of Johnson and Lindenstrauss](https://cseweb.ucsd.edu/~dasgupta/papers/jl.pdf)
Frankle & Carbin 2019 [THE LOTTERY TICKET HYPOTHESIS: FINDING SPARSE, TRAINABLE NEURAL NETWORKS](https://arxiv.org/pdf/1803.03635.pdf)
Keskar et al. 2017 ON LARGE-BATCH TRAINING FOR DEEP LEARNING: GENERALIZATION GAP AND SHARP MINIMA
LeCun et al. 1990 Optimal Brain Damage
Draxler et al. 2019 Essentially No Barriers in Neural Network Energy Landscape
Li et al. 2018 Visualizing the Loss Landscape of Neural Nets
Choromanska et al. 2015 Loss Surfaces of Multilayer Neural Networks
Tishby. et al. Information Bottleneck Paper 
Gigante et al. M-PHATE Paper
Goldfeld et al. Information Flow
***
Generalization and Memorization:
Belkin, et al. 2019 PNAS Reconciling Modern Machine-Learning Practice and The Classical Bias-Variance Tradeoff
[Understanding Deep Learning requires rethinking generalization](https://arxiv.org/pdf/1611.03530.pdf), Zhang et al ICLR 2017
[A closer look at memorization in deep networks](https://arxiv.org/pdf/1706.05394.pdf) Arpit et al 2017
[In search of the real inductive bias: on the role of implicit regularization in deep learning](https://arxiv.org/pdf/1412.6614.pdf) Neyshabur et al, 2015
[The role of over-parametrization in generalization of neural networks](https://arxiv.org/pdf/1805.12076.pdf) Neyshabur et al, 2019
[Train faster, generalize better: stability of stochastic gradient descent](https://arxiv.org/pdf/1509.01240.pdf) Hardt et al, 2016
[Stability and generalization](https://www.academia.edu/13743279/Stability_and_generalization) Bousquet and A. Elisseeff 2002
[Rademacher complexity](http://www.cs.cmu.edu/~ (Links to an external site.)ninamf/ML11/lect1117.pdf)
Chatterjee S. ICLR 2020 Coherent Gradients: An approach to understanding generalization in gradient descent-based optimization
***
Neural Tanget Kernels:
Jacot et al. Neural Tangent Kernel: Convergence and Generalization in Neural Networks, NeurIPS 2018
Chizat et al. On Lazy Training in Differentiable Programming NeurIPS 2019
Arora, Sanjeev, et al. On exact computation with an infinitely wide neural net NeurIPS 2019
Li, Zhiyuan, et al. Enhanced Convolutional Neural Tangent Kernels NeurIPS 2019
[Kernel Functions](https://towardsdatascience.com/kernel-function-6f1d2be6091)
[Understanding the Neural Tangent Kernel](https://rajatvd.github.io/NTK/)



## Deep Learning with D2L Materials

Learning Log:

_12_24_2021_ - preliminaries: torch basics (2) <br/>
_12_25_2021_ - CNN: convolutions, padding, stride, multiple channels, pooling, LeNet (6) <br/>
_12_26_2021_ - RNN, language modeling concepts (8.1-8.4) <br/>
_12_27_2021_ - RNN from scratch, PyTorch RNN (8.5-8.7) <br/>
_12_28_2021_ - modern RNNS: GRU, LSTM, bidirectional (9) <br/>
_12_29_2021_ - Attention (10) <br/>
_12_30_2021_ - Linear NN: linear regression, softmax regression (3) <br/>
_12_31_2021_ - Multilayer perceptron regularization (4) <br/>
_01_01_2022_ - Optimization algorithms, learning rate scheduling (11) <br/>
_01_02_2022_ - deep learning computation, layers, blocks, I/O, GPU (5) <br/>
_01_03_2022_ - recommender system (16) <br/>
_01_04_2022_ - NLP pretraining, word2vec, GloVe, fastText, BERT (14) <br/>
_01_07_2022_ - modern CNN, AlexNet, VGG, NiN, GoogLeNet, batch normalization, ResNet, DenseNet (7) <br/>
_01_10_2022_ - NLP applications, sentiment analysis, NLI (15) <br/>
_01_12_2022_ - GANs (17) <br/>
_01_13_2022_ - computer vision overview (13) <br/>
_01_14_2022_ - computational performance (12) <br/>
_01_16_2022_ - Practical Machine Learning: Data I, Data II <br/>
_01_17_2022_ - Practical Machine Learning: Data III <br/>
_01_18_2022_ - Practical Machine Learning: ML Model, Model Validation <br/>
_01_19_2022_ - Practical Machine Learning: Complete 🚩 <br/>

| Chapter |   | Date |
|---------|---|------|
| 1       |   |      | 
| 2       | ✅ |  _12_24_2021_  |
| 3       | ✅ |  _12_30_2021_  |
| 4       | ✅ |  _12_31_2021_  |
| 5       | ✅ |  _01_02_2022_  |
| 6       | ✅ |  _12_25_2021_  |
| 7       | ✅ |  _01_07_2022_  |
| 8       | ✅ |  _12_27_2021_  |
| 9       | ✅ |  _12_28_2021_  |
| 10      | ✅ |  _12_29_2021_  |
| 11      | ✅ |  _01_01_2022_  |
| 12      | ✅ |  _01_14_2022_  |
| 13      | ✅ |  _01_13_2022_  |
| 14      | ✅ |  _01_04_2022_  |
| 15      | ✅ |  _01_10_2022_  |
| 16      | ✅ |  _01_03_2022_  |
| 17      | ✅ |  _01_12_2022_  |
| 18      |   |      |
| 19      |   |      |
