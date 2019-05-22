# Survey on End-To-End Machine Learning Automation

<div style="text-align: center">
<img src="Figures/MLPipe-1.png" width="900px" atl="Machine Learning Pipeline"/>
</div>
In this repository, we present the references mentioned in a comprehensive survey for the state-of-the-art efforts in tackling the automation of Machine Learning  AutoML, wether through fully automation to the role of data scientist or using some aiding tools that minimize the role of human in the loop. First, we focus on the Combined Algorithm Selection, and Hyperparameter Tuning (CASH) problem. In addition, we highlight the research work of automating the other steps of the full complex machine learning pipeline from data understanding till model deployment. Furthermore, we provide a comprehensive coverage for the various tools and frameworks that have been introduced in this domain.

<hr>

## Table of Contents & Organization:
This repository will be organized into X separate sections:
+ [Meta-Learning Techniques for AutoML search problem](#meta-learning-techniques-for-automl-search-problem)
  - [Learning From Model Evaluation](#learning-from-model-evaluation)
    - [Surrogate Models](#surrogate-models)
    - [Warm-Started Multi-task Learning](#warm-started-multi-task-learning)
    - [Relative Landmarks](#relative-landmarks)
  - [Learning From Task Properties](#learning-from-task-properties)
    - [Using Meta-Features](#using-meta-features)
    - [Using Meta-Models](#using-meta-models)
  - [Learning From Prior Models](#learning-from-prior-models)
    - [Transfer Learning](#transfer-learning)
    - [Few-Shot Learning](#few-shot-learning)
+ [Various approaches for tackling neural architecture search (NAS) problem.] (#Neural Architecture Search)
  - 
+ [Different approaches for automated hyper-parameter optimization (HPO).] (#HPO)
  - 
+ [Various tools and frameworks that have been implemented to tackle the CASH problem.](#Tools-Frameworks)
  - 
+ [Pre-modeling and Post-Modeling of the complex machine learning pipeline.](#Complex-Pipeline)
  - 

<hr>

## Meta-Learning Techniques for AutoML search problem:
Meta-learning can be described as the process of leaning from previous experience gained during applying various learning algorithms on different kinds of data, and hence reducing the needed time to learn new tasks.
  - 2018 | Meta-Learning: A Survey.  | Vanschoren | CoRR | [`PDF`](https://arxiv.org/abs/1810.03548)
  - 2008 | Metalearning: Applications to data mining | Brazdil et al. | Springer Science & Business Media | [`PDF`](https://www.springer.com/gp/book/9783540732624)

<div style="text-align: center">
<img src="Figures/MetaLearning-1.png" width="700px" atl="Machine Learning Pipeline"/>
</div>

### Learning From Model Evaluation
  + ### Surrogate Models
    - 2018 | Scalable Gaussian process-based transfer surrogates for hyperparameter optimization.  | Wistuba et al.  | Journal of ML | [`PDF`](https://link.springer.com/article/10.1007/s10994-017-5684-y)
  + ### Warm-Started Multi-task Learning
    - 2017 | Multiple adaptive Bayesian linear regression for scalable Bayesian optimization with warm start.  | Perrone et al. | [`PDF`](https://arxiv.org/pdf/1712.02902)
  + ### Relative Landmarks
    - 2001 | An evaluation of landmarking variants.  | Furnkranz and Petrak | ECML/PKDD | [`PDF`](http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.21.3221)
  
### Learning From Task Properties
  + ### Using Meta-Features
    - 2019 | SmartML: A Meta Learning-Based Framework for Automated
Selection and Hyperparameter Tuning for Machine Learning Algorithms.  | Maher and Sakr | EDBT | [`PDF`](https://openproceedings.org/2019/conf/edbt/EDBT19_paper_235.pdf)
    - 2017 | On the predictive power of meta-features in OpenML.  | Bilalli et al. | IJAMC | [`PDF`](https://dl.acm.org/citation.cfm?id=3214049)
    - 2013 | Collaborative hyperparameter tuning.  | Bardenet et al. | ICML | [`PDF`](http://proceedings.mlr.press/v28/bardenet13.pdf)
  + ### Using Meta-Models
    - 2018 | Predicting hyperparameters from meta-features in binary classification problems.  | Nisioti et al. | ICML | [`PDF`](http://assets.ctfassets.net/c5lel8y1n83c/5uAPDjSvcseoko2cCcQcEi/8bd1d8e3630e246946feac86271fe03b/PPC17-automl2018.pdf)
    - 2014 | Automatic classifier selection for non-experts. Pattern Analysis and Applications.  | Reif et al. | [`PDF`](https://dl.acm.org/citation.cfm?id=2737365)
    - 2012 | Imagenet classification with deep convolutional neural networks. | Krizhevsky et al. | NIPS | [`PDF`](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)
    - 2008 | Predicting the performance of learning algorithms using support vector machines as meta-regressors.  | Guerra et al. | ICANN | [`PDF`](http://cin.ufpe.br/~rbcp/papers/ICANN08.pdf)
    - 2008 | Metalearning-a tutorial. | Giraud-Carrier | ICMLA | [`PDF`](https://pdfs.semanticscholar.org/54ac/a33d66ba256ff96ebd12b7016dd2d6d137c1.pdf)
    - 2004 | Metalearning: Applications to data mining. | Soares et al. | Springer Science & Business Media | [`PDF`](https://www.springer.com/gp/book/9783540732624)
    - 2004 | Selection of time series forecasting models based on performance information.  | dos Santos et al. | HIS | [`PDF`](http://kt.ijs.si/MarkoBohanec/iddm2002/Koepf.pdf)
    - 2003 | Ranking learning algorithms: Using IBL and meta-learning on accuracy and time results. | Brazdil et al. | Journal of ML | [`PDF`](https://link.springer.com/article/10.1023/A:1021713901879)
    - 2002 | Combination of task description strategies and case base properties for meta-learning.  | Kopf and Iglezakis | [`PDF`](http://kt.ijs.si/MarkoBohanec/iddm2002/Koepf.pdf)

### Learning From Prior Models
  + ### Transfer Learning
    - 2014 | How transferable are features in deep neural networks? | Yosinski et al. | NIPS | [`PDF`](https://papers.nips.cc/paper/5347-how-transferable-are-features-in-deep-neural-networks)
    - 2014 | CNN features offthe-shelf: an astounding baseline for recognition. | Sharif Razavian et al. | IEEE CVPR | [`PDF`](http://openaccess.thecvf.com/content_cvpr_workshops_2014/W15/papers/Razavian_CNN_Features_Off-the-Shelf_2014_CVPR_paper.pdf)
    - 2014 | Decaf: A deep convolutional activation feature for generic visual recognition.  | Donahue et al. | ICML | [`PDF`](https://arxiv.org/abs/1310.1531)
    - 2012 |  Imagenet classification with deep convolutional neural networks. | Krizhevsky et al. | NIPS | [`PDF`](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)
    - 2012 | Deep learning of representations for unsupervised and transfer learning. | Bengio | ICML | [`PDF`](http://proceedings.mlr.press/v27/bengio12a/bengio12a.pdf)
    - 2010 | A survey on transfer learning.  | Pan and Yang | IEEE TKDE | [`PDF`](https://ieeexplore.ieee.org/document/5288526)
    - 1995 | Learning many related tasks at the same time with backpropagation. | Caruana | NIPS | [`PDF`](https://papers.nips.cc/paper/959-learning-many-related-tasks-at-the-same-time-with-backpropagation.pdf)
    - 1995 | Learning internal representations. | Baxter | [`PDF`](https://dl.acm.org/citation.cfm?id=225336)
  + ### Few-Shot Learning
    - 2017 | Prototypical networks for few-shot learning. | Snell et al. | NIPS | [`PDF`](https://arxiv.org/abs/1703.05175)
    - 2017 | Meta-Learning: A Survey.  | Vanschoren | CoRR | [`PDF`](https://arxiv.org/abs/1810.03548)
    - 2016 | Optimization as a model for few-shot learning. | Ravi and Larochelle | [`PDF`](https://openreview.net/pdf?id=rJY0-Kcll)
    
<hr>


