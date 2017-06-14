import json
import numpy as np
import os
from PIL import Image
import tarfile
import re
from keras.layers import Embedding
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.data_utils import get_file


EMBEDDING_DIM = 50
tokenizer = Tokenizer()


def load_data(path):
    f = open(path, 'r')
    data = []
    for l in f:
        jn = json.loads(l)
        s = jn['sentence']
        idn = jn['identifier']
        la = int(jn['label'] == 'true')
        data.append([idn, s, la])
    return data


def tokenize_data(sdata, mxlen):
    texts = [t[1] for t in sdata]
    tokenizer.fit_on_texts(texts)
    seqs = tokenizer.texts_to_sequences(texts)
    seqs = pad_sequences(seqs, mxlen)
    data = {}
    for k in range(len(sdata)):
        data[sdata[k][0]] = [seqs[k], sdata[k][2]]
    return data


def load_images(path, sdata, debug=False):
    data = {}
    cnt = 0
    N = 1000
    for lists in os.listdir(path):
        p = os.path.join(path, lists)
        for f in os.listdir(p):
            cnt += 1
            if debug and cnt > N:
                break
            im_path = os.path.join(p, f)
            im = Image.open(im_path)
            im = im.convert('RGB')
            im = im.resize((200, 50))
            im = np.array(im)
            idf = f[f.find('-') + 1:f.rfind('-')]
            data[f] = [im] + sdata[idf]
    ims, ws, labels = [], [], []
    for key in data:
        ims.append(data[key][0])
        ws.append(data[key][1])
        labels.append(data[key][2])
    data.clear()
    idx = np.arange(0, len(ims), 1)
    np.random.shuffle(idx)
    ims = [ims[t] for t in idx]
    ws = [ws[t] for t in idx]
    labels = [labels[t] for t in idx]
    ims = np.array(ims, dtype=np.float32)
    ws = np.array(ws, dtype=np.float32)
    labels = np.array(labels, dtype=np.float32)
    return ims, ws, labels


def get_embeddings_index():
    embeddings_index = {}
    path = r'C:\local\word2vec\glove.6B.50d.txt'
    f = open(path, 'r', errors='ignore')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    return embeddings_index


def get_embedding_matrix(word_index, embeddings_index):
    embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    return embedding_matrix


def embedding_layer(word_index, embedding_index, sequence_len):
    embedding_matrix = get_embedding_matrix(word_index, embedding_index)
    return Embedding(len(word_index) + 1,
                     EMBEDDING_DIM,
                     weights=[embedding_matrix],
                     input_length=sequence_len,
                     trainable=False)

def tokenize(sent):
    '''Return the tokens of a sentence including punctuation.

    >>> tokenize('Bob dropped the apple. Where is the apple?')
    ['Bob', 'dropped', 'the', 'apple', '.', 'Where', 'is', 'the', 'apple', '?']
    '''
    return [x.strip() for x in re.split('(\W+)?', sent) if x.strip()]


def parse_stories(lines, only_supporting=False):
    '''Parse stories provided in the bAbi tasks format

    If only_supporting is true, only the sentences
    that support the answer are kept.
    '''
    data = []
    story = []
    for idx, line in enumerate(lines):
        if line != '' and len(line.strip()) > 4:
            print("line:", idx)
            line = line.strip()
            nid, line = line.split(' ', 1)
            # if isinstance(nid, str):
            #     continue
            nid = int(nid)
            if nid == 1:
                story = []
            if '\t' in line:
                q, a, supporting = line.split('\t')
                q = tokenize(q)
                substory = None
                if only_supporting:
                    # Only select the related substory
                    supporting = map(int, supporting.split())
                    substory = [story[i - 1] for i in supporting]
                else:
                    # Provide all the substories
                    substory = [x for x in story if x]
                data.append((substory, q, a))
                story.append('')
            else:
                sent = tokenize(line)
                story.append(sent)

    # with open('parsed_stories.json', 'w') as fstories:
    #     fstories.write(json.dumps(data))
    return data


def get_stories(f, only_supporting=False, max_length=None, data_type=None, dataset=""):
    '''Given a file name, read the file,
    retrieve the stories,
    and then convert the sentences into a single story.

    If max_length is supplied,
    any stories longer than max_length tokens will be discarded.
    '''
    print("Parse stories start for " + data_type +  "...")
    try:
        fparsed = open("parsed_stories_" + dataset + "_" + data_type+".pickle", 'rb')
        data = pickle.load(fparsed)
        fparsed.close()
    except:
        data = parse_stories(f.readlines(), only_supporting=only_supporting)
        f.close()
        print("Save data as file")
        print(len(data))
        fparsed = open("parsed_stories_"+ dataset + "_" + data_type+".pickle", 'wb')
        pickle.dump(data, fparsed, protocol=pickle.HIGHEST_PROTOCOL)
        fparsed.close()
    print("Parse stories done")

    print("Flatten data start for " + data_type +  "...")
    try:
        fflat = open("parsed_flattened_stories_" + dataset + "_" +  data_type + ".pickle", 'rb')
        data = pickle.load(fflat)
        fflat.close()
    except:
        flatten = lambda data: reduce(lambda x, y: x + y, data)
        data = [(flatten(story), q, answer) for story, q, answer in data if not max_length or len(flatten(story)) < max_length]
        print("Flatten data done")
        print("Save flattened data")
        fflat = open("parsed_flattened_stories_"+data_type+".pickle", 'wb')
        pickle.dump(data, fflat, protocol=pickle.HIGHEST_PROTOCOL)
        fflat.close()
    print("Save flattened data done")
    return data


def vectorize_stories(data, word_idx, story_maxlen, query_maxlen):
    print("vectorize stories")
    X = []
    Xq = []
    Y = []
    for story, query, answer in data:
        x = [word_idx[w] for w in story]
        xq = [word_idx[w] for w in query]
        # let's not forget that index 0 is reserved
        y = np.zeros(len(word_idx) + 1)
        y[word_idx[answer]] = 1
        X.append(x)
        Xq.append(xq)
        Y.append(y)
    return (pad_sequences(X, maxlen=story_maxlen),
            pad_sequences(Xq, maxlen=query_maxlen), np.array(Y))


def default_babl_dataset():
    '''
    Downloads and loads the default Facebook BAbl dataset
    '''
    try:
        path = get_file('babi-tasks-v1-2.tar.gz', origin='https://s3.amazonaws.com/text-datasets/babi_tasks_1-20_v1-2.tar.gz')
    except:
        print('Error downloading dataset, please download it manually:\n'
              '$ wget http://www.thespermwhale.com/jaseweston/babi/tasks_1-20_v1-2.tar.gz\n'
              '$ mv tasks_1-20_v1-2.tar.gz ~/.keras/datasets/babi-tasks-v1-2.tar.gz')
        raise
    tar = tarfile.open(path)

    challenges = {
        # QA1 with 10,000 samples
        'single_supporting_fact_10k': 'tasks_1-20_v1-2/en-10k/qa1_single-supporting-fact_{}.txt',
        # QA2 with 10,000 samples
        'two_supporting_facts_10k': 'tasks_1-20_v1-2/en-10k/qa2_two-supporting-facts_{}.txt',
    }
    challenge_type = 'single_supporting_fact_10k'
    challenge = challenges[challenge_type]

    print('Extracting stories for the challenge:', challenge_type)
    return (tar.extractfile(challenge.format('train')),tar.extractfile(challenge.format('test')))


def custom_babl_dataset(train_file="train_data_entity_full.txt", test_file="test_data_entity_full.txt"):
    '''
    Loads your custom dataset (needs to be in BAbl format to be parsable)
    '''
    train_raw = open(train_file, 'r')
    test_raw = open(test_file, 'r')
    return (train_raw, test_raw)

def get_babl_data(datatype="custom"):
    '''
     Preprocess BAbl data
     '''
    if datatype=="default":
        ftrain, ftest = default_babl_dataset()
    else:
        ftrain, ftest = custom_babl_dataset()

    train_stories = get_stories(ftrain, only_supporting=False, max_length=5000, data_type="train", dataset="entity")
    test_stories = get_stories(ftest, only_supporting=False, max_length=5000, data_type="test", dataset="entity")

    vocab = set()
    for story, q, answer in train_stories + test_stories:
        vocab |= set(story + q + [answer])
    vocab = sorted(vocab)

    # Reserve 0 for masking via pad_sequences
    vocab_size = len(vocab) + 1
    story_maxlen = max(map(len, (x for x, _, _ in train_stories + test_stories)))
    query_maxlen = max(map(len, (x for _, x, _ in train_stories + test_stories)))

    print('-')
    print('Vocab size:', vocab_size, 'unique words')
    print('Story max length:', story_maxlen, 'words')
    print('Query max length:', query_maxlen, 'words')
    print('Number of training stories:', len(train_stories))
    print('Number of test stories:', len(test_stories))
    print('-')
    print('Here\'s what a "story" tuple looks like (input, query, answer):')
    print(train_stories[0])
    print('-')
    print('Vectorizing the word sequences...')

    word_idx = dict((c, i + 1) for i, c in enumerate(vocab))
    inputs_train, queries_train, answers_train = vectorize_stories(train_stories,
                                                                   word_idx,
                                                                   story_maxlen,
                                                                   query_maxlen)
    inputs_test, queries_test, answers_test = vectorize_stories(test_stories,
                                                                word_idx,
                                                                story_maxlen,
                                                                query_maxlen)

    print('-')
    print('inputs: integer tensor of shape (samples, max_length)')
    print('inputs_train shape:', inputs_train.shape)
    print('inputs_test shape:', inputs_test.shape)
    print('-')
    print('queries: integer tensor of shape (samples, max_length)')
    print('queries_train shape:', queries_train.shape)
    print('queries_test shape:', queries_test.shape)
    print('-')
    print('answers: binary (1 or 0) tensor of shape (samples, vocab_size)')
    print('answers_train shape:', answers_train.shape)
    print('answers_test shape:', answers_test.shape)
    print('-')
    print('Compiling...')

    return (inputs_train, queries_train, answers_train,inputs_test, queries_test, answers_test )