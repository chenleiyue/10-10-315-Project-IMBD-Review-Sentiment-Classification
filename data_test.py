import gensim
import data_preprocessing

def test(model_loc, test_loc):
    model = gensim.models.Doc2Vec.load(model_loc)
    test_content_list, test_tag_list = data_preprocessing.read_data(test_loc)
    
    words = test_content_list[0]
    words_docvec = model.infer_vector(words)
    

model_loc = 'model_v200_w10.model'
test_loc = 'task1/test/'
test(model_loc, test_loc)