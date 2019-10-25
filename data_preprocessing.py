import nltk
import glob
from nltk import tokenize
import gensim
import re

TaggedDocument = gensim.models.doc2vec.TaggedDocument

def read_data(files_loc):
    content_list = []
    tag_list = []
    tokenizer = tokenize.RegexpTokenizer(r'\w+')
    
    files_loc = files_loc + "*.txt"
    for f in glob.glob(files_loc):
        tag_list.append(f)

        with open(f, 'r') as text:
            raw = text.read()
            content = re.sub(r'\d+', '', raw)
            content = tokenizer.tokenize(content)
            content = list(map(lambda x: x.lower(), content))

            ## Lemmatize? stop words? punctuation?

            content_list.append(content)

    return content_list, tag_list

if __name__ == "__main__":
    train_pos = "task1/train/positive/"
    train_neg = "task1/train/negative/"
    test = "task1/test/"

    pos_content_list, pos_tag_list = read_data(train_pos)
    neg_content_list, neg_tag_list = read_data(train_neg)
    content_list = pos_content_list + neg_content_list
    tag_list = pos_tag_list + neg_tag_list
    
    documents = [TaggedDocument(content_list[i], tag_list[i]) for i in range(len(content_list))]
    model = gensim.models.Doc2Vec(documents=documents, vector_size=200, window=10, min_count=3, workers=4)
    model.save('model_v200_w10.model')
    
    