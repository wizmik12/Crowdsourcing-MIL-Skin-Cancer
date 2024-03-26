### Code for replicating the results of the paper [Annotation Protocol and Crowdsourcing Multiple Instance Learning Classification of Skin Histological Images: the CR-AI4SkIN Dataset]()

<img src="https://github.com/wizmik12/Crowdsourcing-MIL-Skin-Cancer/blob/main/overview.png" width="520">



#### Citation
~~~
@article{del2023annotation,
  title={Annotation protocol and crowdsourcing multiple instance learning classification of skin histological images: The CR-AI4SkIN dataset},
  author={Del Amor, Roc{\'\i}o and P{\'e}rez-Cano, Jose and L{\'o}pez-P{\'e}rez, Miguel and Terradez, Liria and Aneiros-Fernandez, Jose and Morales, Sandra and Mateos, Javier and Molina, Rafael and Naranjo, Valery},
  journal={Artificial Intelligence in Medicine},
  volume={145},
  pages={102686},
  year={2023},
  publisher={Elsevier}
}
~~~

## Abstract
Digital Pathology (DP) has experienced a significant growth in recent years and has become an essential tool for diagnosing and prognosis of tumors. 
The availability of Whole Slide Images (WSIs) and the implementation of Deep Learning (DL) algorithms have paved the way for the appearance of Artificial Intelligence (AI) systems that support the diagnosis process. These systems require extensive and varied data for their training to be successful. However, creating labeled datasets in histopathology is laborious and time-consuming. We have developed a crowdsourcing-multiple instance labeling/learning protocol that is applied to the creation and use of the CR-AI4SkIN dataset. CR-AI4SkIN contains 271 WSIs of 7 Cutaneous Spindle Cell (CSC) neoplasms with expert and non-expert labels at region and WSI levels. It is the first dataset of these types of neoplasms made available. The regions selected by the experts are used to learn an automatic extractor of  Regions of Interest (ROIs) from WSIs. To produce the embedding of each WSI, the representations of patches within the ROIs are obtained using a contrastive learning method, and then combined. Finally, they are fed to a Gaussian process-based crowdsourcing classifier, which utilizes the noisy non-expert WSI labels. We validate our crowdsourcing-multiple instance learning method in the CR-AI4SkIN dataset, addressing a binary classification problem (malign vs. benign). 
The proposed method obtains an F1 score of 0.7911 on the test set, outperforming three widely used aggregation methods for crowdsourcing tasks. Furthermore, our crowdsourcing method also outperforms the supervised model with expert labels on the test set (F1-score = 0.6035). The promising results support the proposed crowdsourcing multiple instance learning annotation protocol.  It also validates the automatic extraction of interest regions and the use of contrastive embedding and Gaussian process classification to perform crowdsourcing classification tasks.

## Data

**Link to download the dataset:** [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10880652.svg)](https://doi.org/10.5281/zenodo.10880652)

## Code
### Installation

Create a python environment (conda / pip) and run `pip install -r requirements.txt`. Code has been tested on CentOS Linux 7 and on macOS Ventura 15.3 (Apple Silicon) for python versions 3.7 up to 3.10. We have no plans on updating the code to be compatible with pandas 2.0 or with newer versions of GPFlow / GPFlux.


### Experiments

To run the experiments from the article run the command `python code/run_all.py`. You may use a CUDA capable device but it is not mandatory, code is fast enough to be run on CPU and it will take a few hours. Results will be saved under the newly created `results` folder.

If you want to test the method with your own data keep in mind the following format of the data. The features are numpy arrays of shape (N, D) where N is the number of samples and D is the number of dimensions of the features, which in our case is 512. The labels have different formats. Those with prefix `y_` represent the crowdsourced labels and are a dictionary with keys: 'Y', 'mask', 'A', 'K'. The 'Y' key corresponds to a numpy array of shape (N, A, K) that has the annotations of every annotator and the 'mask' key has another numpy array with shape (N, A) which contains 1s and 0s denoting whether each annotator gave a label or not for that instance. A is the number of annotators and K is the number of classes. The labels names with prefix `z_` are the expert annotations as a numpy array of shape (N,) with the class label as an integer. Finally, the labels with prefix `mv_` have the same format as those with prefix `z_` but represent the majority voting of the non-expert annotators.


## Feature extraction

The feature extractor was pretrained following the SimCLR procedure using the code from [this repository](https://github.com/binli123/dsmil-wsi). Just follow the instruction there for pretraining the feature extractor and extracting the features from the Whole Slide Images. Already precomputed features on our database are in the `Features` folder. The `Features_pretrained` folder uses the same model weights as those already provided in the repository and the other two folder with prefix `Features_` are the features obtained with the models pretrained on our own database, with batch size 256 and 512.
