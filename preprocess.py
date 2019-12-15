import codecs
import copy
import csv
import gensim
import multiprocessing
import nltk
import os
import pandas as pd
import re
import sklearn
import xml.etree.ElementTree as etree
from bs4 import BeautifulSoup
# from collections import Counter
from collections import defaultdict
from gensim import corpora
from gensim import models
from gensim import similarities
from gensim.models.word2vec import Word2Vec
from gensim.test.utils import datapath, get_tmpfile
from html.parser import HTMLParser
from nltk import word_tokenize, sent_tokenize
from nltk.text import TextCollection
from nltk.tokenize import TreebankWordTokenizer
from scipy import spatial
from sklearn.feature_extraction.text import TfidfVectorizer

PATH_WIKI_XML = 'D:\\things'
FILENAME_WIKI = 'simplewiki-20191101-pages-meta-current.xml'
FILENAME_ARTICLES = 'articles.csv'
FILENAME_REDIRECT = 'articles_redirect.csv'
FILENAME_TEMPLATE = 'articles_template.csv'
FILENAME_FULL_ARTICLES = 'full_articles.csv'
FILENAME_CLEAN_ARTICLES = 'clean_articles.csv'
ENCODING = "utf-8"
NUM_TOPICS = 500
nltk.download('stopwords')
nltk.download('punkt')
stopwords = nltk.corpus.stopwords.words('english')
stopwords += '- -- " \' ? , . ! * ** *** ( ) = == === : ; \'\' ` `` [ ] & %'.split()
regex = re.compile("\[\[[\w\s]*\]\]")
nonword = re.compile(r'^\W*$')

class MLStripper(HTMLParser):
    def __init__(self):
        self.reset()
        self.strict = False
        self.convert_charrefs= True
        self.fed = []
    def handle_data(self, d):
        self.fed.append(d)
    def get_data(self):
        return ''.join(self.fed)

def clean_article(article):
    soup = BeautifulSoup(article)
    blob = remove_text_inside_brackets(soup.get_text())
    for result in re.findall("\[\[[\w\s',-.\?]*\]\]", blob):
        blob = blob.replace(result, result[2:-2])
    # TODO: handle the split("|")[1] as alternate names for NER, see if in list of articles, add to list of entities, associate with
    # main name via is-a dictionary?
    for result in re.findall("\[\[[(\w\|,'-.\?)*?\w\s]*\]\]", blob):
        blob = blob.replace(result, result[2:-2].split("|")[0])
    for result in re.findall("\[?\[\w*:.*\]", blob):
        blob = blob.replace(result, '')
    if re.search("=*?=\s?(R|r)eferences\s?=*?=", blob):
        blob = blob[:re.search("=*?=\s?(R|r)eferences\s?=*?=", blob).start()]
    return blob

def cos_sim(sklearn_corpus, ind1, ind2):
    return 1 - spatial.distance.cosine(sklearn_corpus[ind1].todense(), sklearn_corpus[ind2].todense())

def hms_string(sec_elapsed):
    h = int(sec_elapsed / (60 * 60))
    m = int((sec_elapsed % (60 * 60)) / 60)
    s = sec_elapsed % 60
    return "{}:{:>02}:{:>05.2f}".format(h, m, s)

def nltk_tfidf_vectorize(lists_of_tokens):
    texts = TextCollection(lists_of_tokens)
    for article in lists_of_tokens:
        yield {
            term: texts.tf_idf(term, article)
            for term in article
        }

def nltk_vectorize(article):
    features = defaultdict(int)
    for tok in article:
        features[tok] += 1
    return features

def remove_text_inside_brackets(text, brackets="{}"):
    count = [0] * (len(brackets) // 2) # count open/close brackets
    saved_chars = []
    for character in text:
        for i, b in enumerate(brackets):
            if character == b: # found bracket
                kind, is_close = divmod(i, 2)
                count[kind] += (-1)**is_close # `+1`: open, `-1`: close
                if count[kind] < 0: # unbalanced bracket
                    count[kind] = 0  # keep it
                else:  # found bracket to remove
                    break
        else: # character is not a [balanced] bracket
            if not any(count): # outside brackets
                saved_chars.append(character)
    return ''.join(saved_chars)

def strip_tags(html):
    s = MLStripper()
    s.feed(html)
    return s.get_data()

def strip_tag_name(t):
    # t = elem.tag
    idx = k = t.rfind("}")
    if idx != -1:
        t = t[idx + 1:]
    return t

def token_frequency_counter(my_tokens):
    frequency = defaultdict(int)
    for article in my_tokens:
        for tok in article:
            frequency[tok] += 1
    return frequency

def tokenize_for_word2vec(text, stopwords=stopwords, regex_pattern=nonword):
    corpus_tokens = []
    for article in text:
        article_sentences = sent_tokenize(str(article))
        for sentence in article_sentences:
            corpus_tokens.append([word for word in word_tokenize(sentence) if (word not in stopwords and not nonword.search(word))])
    return corpus_tokens

def tokenize_for_lda(article, tokenizer=TreebankWordTokenizer(), stopwords=stopwords, regex_pattern=nonword):
    article_tokens = [tok for tok in tokenizer.tokenize(article) if (tok.lower() not in stopwords and not regex_pattern.search(tok))]
    return article_tokens

def unmatched_bracket(text):
    """
    Returns true if there is an unmatched bracket
        this is a sentence {with a bracket } - false
        this is a sentence {with a bracket } and {this - true
    """
    for c in reversed(text):
        if c is "}":
            return False
        elif c is "{":
            return True

def dump2csv(pathWikiXML, pathArticles, pathArticlesRedirect, pathTemplateRedirect):
    # Todo: remove csv step and write to array, return df from this function
    totalCount = 0
    articleCount = 0
    redirectCount = 0
    templateCount = 0
    title = None
    with codecs.open(pathArticles, "w", ENCODING) as articlesFH, \
            codecs.open(pathArticlesRedirect, "w", ENCODING) as redirectFH, \
            codecs.open(pathTemplateRedirect, "w", ENCODING) as templateFH:
        articlesWriter = csv.writer(articlesFH, quoting=csv.QUOTE_MINIMAL)
        redirectWriter = csv.writer(redirectFH, quoting=csv.QUOTE_MINIMAL)
        templateWriter = csv.writer(templateFH, quoting=csv.QUOTE_MINIMAL)

        articlesWriter.writerow(['id', 'title', 'text'])
        redirectWriter.writerow(['id', 'title', 'text'])
        templateWriter.writerow(['id', 'title'])

        for event, elem in etree.iterparse(pathWikiXML, events=('start', 'end')):
            tname = strip_tag_name(elem.tag)

            if event == 'start':
                if tname == 'page':
                    title = ''
                    idnum = -1
                    redirect = ''
                    text = ''
                    inrevision = False
                    ns = 0
                elif tname == 'revision':
                    # Do not pick up on revision id's
                    inrevision = True
            else:
                if tname == 'title':
                    title = elem.text
                elif tname == 'id' and not inrevision:
                    idnum = int(elem.text)
                elif tname == 'redirect':
                    redirect = elem.attrib['title']
                elif tname == 'ns':
                    ns = int(elem.text)
                elif tname == 'text' and elem.text:
                    text += elem.text
                elif tname == 'page':
                    totalCount += 1

                    if ns == 10:
                        templateCount += 1
                        templateWriter.writerow([idnum, title])
                    elif len(redirect) > 0:
                        articleCount += 1
                        articlesWriter.writerow([idnum, title, text.replace(',', ',')])

                    else:
                        redirectCount += 1
                        redirectWriter.writerow([idnum, title, text.replace(',', ',')])


                    # if totalCount > 1 and (totalCount % 100000) == 0:
                    #     print("{:,}".format(totalCount))

                elem.clear()
    # elapsed_time = time.time() - start_time
    # print("Total pages: {:,}".format(totalCount))
    # print("Template pages: {:,}".format(templateCount))
    # print("Article pages: {:,}".format(articleCount))
    # print("Redirect pages: {:,}".format(redirectCount))
    # print("Elapsed time: {}".format(hms_string(elapsed_time)))

def main():
    pathWikiXML = os.path.join(PATH_WIKI_XML, FILENAME_WIKI)
    pathArticles = os.path.join(PATH_WIKI_XML, FILENAME_ARTICLES)
    pathArticlesRedirect = os.path.join(PATH_WIKI_XML, FILENAME_REDIRECT)
    pathTemplateRedirect = os.path.join(PATH_WIKI_XML, FILENAME_TEMPLATE)
    pathFullArticles = os.path.join(PATH_WIKI_XML, FILENAME_FULL_ARTICLES)
    pathCleanArticles = os.path.join(PATH_WIKI_XML, FILENAME_CLEAN_ARTICLES)

    dump2csv(pathWikiXML, pathArticles, pathArticlesRedirect, pathTemplateRedirect)

    dfArticlesRedirect = pd.read_csv(pathArticlesRedirect)
    dfArticlesRedirect['text'] = [str(t) for t in dfArticlesRedirect['text']]
    dfOnlyArticles = dfArticlesRedirect.loc[[t[:7] != 'User ta' for t in dfArticlesRedirect['title']]]
    dfOnlyArticles = dfOnlyArticles.loc[[t[:5] != 'User:' for t in dfOnlyArticles['title']]]
    dfOnlyArticles = dfOnlyArticles.loc[[t[:9] != 'Category:' for t in dfOnlyArticles['title']]]
    dfOnlyArticles = dfOnlyArticles.loc[[t[:5] != 'Talk:' for t in dfOnlyArticles['title']]]
    dfOnlyArticles = dfOnlyArticles.loc[[t[:10] != 'Wikipedia:' for t in dfOnlyArticles['title']]]
    dfOnlyArticles.to_csv(pathFullArticles)

    dfCleanArticles = copy.copy(dfOnlyArticles)
    dfCleanArticles['text'] = dfCleanArticles['text'].apply(lambda x: clean_article(x))
    dfCleanArticles['id'] = [t + 1 for t in range(len(dfCleanArticles['id']))] # reset IDs after removal
    dfCleanArticles.set_index('id')
    dfCleanArticles.to_csv(pathCleanArticles)

    corpus = pd.read_csv(pathCleanArticles, encoding=ENCODING)

    articles = [str(text) for text in corpus.text]
    titles = [str(title) for title in corpus.title]
    my_tokens = [tokenize_for_lda(str(art)) for art in corpus.text]
    frequency = token_frequency_counter(my_tokens)
    my_tokens = [[tok for tok in article if frequency[tok] > 5] for article in my_tokens]


    # SPACY
    # import en_core_web_lg
    # parser = en_core_web_lg.load()
    # doc = parser(australiaArticle)
    # with doc.retokenize() as retokenizer:
    #     for ent in doc.ents:
    #         retokenizer.merge(doc[ent.start:ent.end])
    # def tokenize(text):
    #     doc = parser(text)
    #     with doc.retokenize() as retokenizer:
    #         for ent in doc.ents:
    #             retokenizer.merge(doc[ent.start:ent.end], attrs={"LEMMA": ent.text})
    #     return [x for x in doc if (not x.is_punct and not nonword.search(str(x)) and str(x).lower() not in stopwords)]

    # SKLEARN
    # sklearn_tfidf = TfidfVectorizer(input=articles, encoding=ENCODING, decode_error='replace' '<UNK>',
                                    # stop_words=stopwords, min_df=0.00005)
    # sklearn_corpus = sklearn_tfidf.fit_transform(articles)

    # NLTK
    # nltk_tfidf_vectors = nltk_tfidf_vectorize(my_tokens)
    # nltk_vectors = map(nltk_vectorize, my_tokens)


    # GENSIM
    # lda/lsa
    dictionary = corpora.Dictionary(my_tokens)
    gensim_vectors = [dictionary.doc2bow(doc) for doc in my_tokens]
    gensim_tfidf = gensim.models.TfidfModel(dictionary=dictionary, normalize=True)
    gensim_tfidf_vectors = [gensim_tfidf[dictionary.doc2bow(article)] for article in my_tokens]
    dictionary.save('/tmp/simplewiki_mystopwords.dict')  # store the dictionary, for future reference
    dictionary.save_as_text('gensim_dictionary.txt', sort_by_word=True)
    gensim_tfidf.save('gensim_tfidf.pkl')
    lsi_model = models.LsiModel(gensim_tfidf_vectors, id2word=dictionary, num_topics=NUM_TOPICS)
    # corpus_lsi = lsi_model[gensim_tfidf_vectors]
    tmp_fname = get_tmpfile("simple_wiki_lsi_{}.model".format(NUM_TOPICS))
    lsi_model.save(tmp_fname)
    # loaded_model = LsiModel.load(tmp_fname)
    index_temp = get_tmpfile("index")
    index = similarities.Similarity(index_temp, gensim_vectors,
                                    num_features=len(dictionary.keys()))  # transform corpus to LSI space and index it
    lda_model = gensim.models.ldamodel.LdaModel(corpus=gensim_vectors, id2word=dictionary, num_topics=NUM_TOPICS,
                                                passes=1)
    tmp_fname = get_tmpfile("simple_wiki_lda_{}.model".format(NUM_TOPICS))
    lda_model.save(tmp_fname)

    # word2vec
    corpus_tokens = tokenize_for_word2vec(corpus.text)
    num_features = 400
    min_word_count = 5
    num_workers = multiprocessing.cpu_count()
    window_size = 6
    subsampling = 1e-3
    model = Word2Vec(corpus_tokens, workers=num_workers, size=num_features, min_count=min_word_count,
                     window=window_size, sample=subsampling)
    model_name = 'simple_wiki_word2vec_model'
    model.save(model_name)
    # this frees up memory by discarding unneeded data
    # but the model cannot be trained further after this
        # model.init_sims(replace=True)
    # Alternatively, try this one recommended by gensim docs:
        # word_vectors = model.wv
        # del model




if __name__== "__main__":
    main()

