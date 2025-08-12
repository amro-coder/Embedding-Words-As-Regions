# Embedding-Words-As-Regions

This repositery contains the code of my Thesis at the University of Cambridge (google drive link https://drive.google.com/file/d/1mloLsGVmMXoPhBCRZs0LAiwkUBsyrVYa/view?usp=sharing).

Abstract:

 While Large Language Models (LLMs) have achieved state-of-the-art performance on a
 wide range of language understanding and generation benchmarks, their reliance on mas
sive datasets and computational resources highlights a need for more data-efficient and
 representationally powerful models. A fundamental limitation, shared by both LLMs and
 traditional models like Word2Vec, is the use of point-based vectors, which struggle to
 capture complex, asymmetric linguistic relationships like hyponymy. Region-based repre
sentations offer a theoretically powerful alternative, modeling word meanings as regions
 where relationships like hyponymy can be represented by geometric containment.
 This thesis introduces Word2Ellipsoid, a novel region-based model that represents words
 as fuzzy hyperellipsoids derived from a Gaussian function. A key advantage of this ap
proach is that, unlike existing box-based models, it provides a closed-form, approximation
free framework for calculating the volume of a hyperellipsoid. To evaluate its effectiveness,
 Word2Ellipsoid is benchmarked against a strong vector baseline (Word2Vec) and a state
of-the-art region baseline (Word2Box). All models were assessed on a comprehensive suite
 of word similarity and hyponymy detection tasks.
 The empirical findings from our experiments yield three principal conclusions. First,
 Word2Ellipsoid outperformed Word2Box, indicating that our approximation-free mathe
matical framework is more precise and robust. Second, the Word2Vec baseline consistently
 outperformed both regional methods, which suggests it is a strong baseline whose capa
bilities may have been underestimated in prior research, potentially due to insufficient
 hyperparameter optimization. Finally, the evaluated regional models struggled to effec
tively capture hyponymy relations. This highlights the inherent difficulty of learning such
 complex relationships from purely distributional data using simple training objectives.


 Note:

 The config.yaml contains paths to repostries that some python code access, edit it with the correct local pathes for the code to work.

 Regarding the training dataset, the name of the dataset is WaCkypedia EN dataset,  here is a link to get the dataset from 
 https://wacky.sslmit.unibo.it/doku.php?id=start
 The dataset is in xml files and the full code to preproces it is provided in "prepare_dataset.py".
 
 

 
