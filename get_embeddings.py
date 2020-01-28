import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import sys
import argparse

#Store embedding size/dimension
parser = argparse.ArgumentParser()
parser.add_argument('--emb_size', action="store");
args = parser.parse_args();
embedding_size = args.emb_size


models_dict = {
'embed_20_model_url' : "https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim-with-oov/1",
'embed_50_model_url' : "https://tfhub.dev/google/tf2-preview/nnlm-en-dim50-with-normalization/1",
'embed_128_model_url' : "https://tfhub.dev/google/tf2-preview/nnlm-en-dim128-with-normalization/1",
'embed_250_model_url' : "https://tfhub.dev/google/Wiki-words-250/2",
'embed_512_model_url' : "https://tfhub.dev/google/universal-sentence-encoder/4", #trained with Deep Averaging Network (DAN)
'embed_512t_model_url' : "https://tfhub.dev/google/universal-sentence-encoder-large/5" #trained with transformer
}

model_url = 'embed_{}_model_url'.format(embedding_size)

embed = hub.load(models_dict[model_url])

""" This function takes as input a list of sentences/one sentence at a time and returns an n-dimensional embedding from
    pretrained sentence embedding module(s) from Tensorflow-Hub.

    Input parameter: Un-processed string sentences in a list/one at a time.
    
    For example inputs can be one of the following 2 types:
        Type 1: List of string type sentences ->  sentences_ = ["I am a sentence for which I would like to get its embedding.",
                        "Universal Sentence Encoder embeddings also support short paragraphs. ",
                        "There is no hard limit on how long the paragraph is. Roughly, the longer ",
                        "the more 'diluted' the embedding will be."]
        Type 2: one string sentence at a time -> sentences_ = "This is a single sentence input."

    Returns : Numpy array of dimension (m,n) where m is the number of sentences in input, and n is the embedding dimension
"""
def get_sentence_embedding(sentences):
    if not isinstance(sentences,(list,)):
        sentences = [sentences]
    
    sentence_embedding = embed(sentences)
    return np.array(sentence_embedding)

sentences_ = ["I am a sentence for which I would like to get its embedding.",
            "Universal Sentence Encoder embeddings also support short paragraphs. ",
            "There is no hard limit on how long the paragraph is. Roughly, the longer ",
            "the more 'diluted' the embedding will be." ]

# Single sentence as input ->
#sentences_ = "I am a sentence for which I would like to get its embedding."

# function call
embeddings = get_sentence_embedding(sentences_)

print("Shape of output/embeddings array", embeddings.shape)
# for sent_emb in embeddings:
#     print(sent_emb)