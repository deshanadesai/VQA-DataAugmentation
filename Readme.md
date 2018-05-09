### VQA Data Augmentation

Baseline: Show, Ask, Attend, and Answer: A Strong Baseline For Visual Question Answering 
(paper: https://arxiv.org/pdf/1704.03162.pdf) 
(code: https://github.com/Cyanogenoid/pytorch-vqa)

Datasets: VQA2.0 http://www.visualqa.org/download.html

Evaluation: https://github.com/GT-Vision-Lab/VQA

Experiments To Do:

To run today:

* Add converse substitution to Language Only augmentation 
* Multiple word substitutions.
* Do some paraphrasing for known question types.

* How many <OBJ> / Color of <OBJ> - question substitution with hypernym doubt.
* Language augment other methods.
* Change all augmentation methods to fit the same vocab.
* Filter conceptnet based on question repetition.
* Add all working methods together for data augmentation.
* Make custom test set for places 365? Places 365 has adjectives as well as scene understanding.
* Add augmentation on image based on wrong answer or image type?
  
