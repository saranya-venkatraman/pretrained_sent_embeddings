import tensorflow as tf
import tensorflow_hub as hub
import numpy as np

#set embedding size(20/50/128/250/512/512t) 
embedding_size = 128

#get input sentences/read from file into list
sentences_= ["Universal Sentence Encoder embeddings also support short paragraphs.",
             "There is no hard limit on how long the paragraph is.", 
             "Roughly, the longer the more 'diluted' the embedding will be."]

#dictionary to store URLs of pre-trained modules
models_dict = {
'embed_20_model_url' : "https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim-with-oov/1",
'embed_50_model_url' : "https://tfhub.dev/google/tf2-preview/nnlm-en-dim50-with-normalization/1",
'embed_128_model_url' : "https://tfhub.dev/google/tf2-preview/nnlm-en-dim128-with-normalization/1",
'embed_250_model_url' : "https://tfhub.dev/google/Wiki-words-250/2",
'embed_512_model_url' : "https://tfhub.dev/google/universal-sentence-encoder/4", #trained with Deep Averaging Network (DAN)
'embed_512t_model_url' : "https://tfhub.dev/google/universal-sentence-encoder-large/5" #trained with transformer
}

#get URL according to embedding size
model_url = 'embed_{}_model_url'.format(embedding_size)

#load pre-trained embeddings
embed = hub.load(models_dict[model_url])

def get_sentence_embedding(sentences):
    """ Fetches n-dimensional embedding per input sentence from pretrained sentence embedding module.
    
    Args:
        sentences: Python List of String/sequence of Strings.
        
    Returns:
        Numpy array of dimension (m,n) where m is the number of sentences in input, and n is the embedding dimension.
 
    """
    sentence_embedding = embed(sentences)
    return np.array(sentence_embedding)

#get embeddings from URL
embeddings = get_sentence_embedding(sentences_)

print("Shape of output/embeddings array", embeddings.shape)

# for sent_emb in embeddings:
#     print(sent_emb)