# NER-Collocations Project Overview

The aim of this project was to develop a simple program that facilitates quick and easy text preprocessing, focusing not only on identifying named entities but also on extracting relevant word combinations. The primary emphasis was on evaluating the relevance of these word combinations in relation to the specified entity. The following approach was used:

- The text underwent preprocessing.
- It was structured into word combinations (for example, 2-word phrases were used).
- The final corpus was compiled in the form of n-grams.
- An optimal model was selected to obtain embeddings (vector representations of words).
- The relevance of each n-gram to the target entity was evaluated.

As a result, a brief example of this approach is provided in the file `example.ipynb`. Data for fine-tuning models, as well as the fine-tuned BERT and FastText models, are not published due to confidentiality. However, you can find the pipeline for model fine-tuning in the `notebooks folder`.

Please feel free to contact me if you have any questions.