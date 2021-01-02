# Complete_NLP Discussion:
Discuss all about natural language processing
# ***Natural Language Processing (NLP)***
Everything we express (either verbally or in written) carries huge amounts of information. The topic we choose, our tone, our selection of words, everything adds some type of information that can be interpreted and value extracted from it. In theory, we can understand and even predict human behaviour using that information.

Data generated from conversations, declarations or even tweets are examples of unstructured data. Unstructured data doesn’t fit neatly into the traditional row and column structure of relational databases, and represent the vast majority of data available in the actual world. It is messy and hard to manipulate.

# Natural Language Processing or NLP is a field of Artificial Intelligence that gives the machines the ability to read, understand and derive meaning from human languages.
# ***Tokenization:***
Is the process of segmenting running text into sentences and words. In essence, it’s the task of cutting a text into pieces called tokens, and at the same time throwing away certain characters, such as punctuation.

Tokenization can remove punctuation too, easing the path to a proper word segmentation but also triggering possible complications. In the case of periods that follow abbreviation (e.g. dr.), the period following that abbreviation should be considered as part of the same token and not be removed.

The tokenization process can be particularly problematic when dealing with biomedical text domains which contain lots of hyphens, parentheses, and other punctuation marks.
# ***Stop Words Removal:***
Includes getting rid of common language articles, pronouns and prepositions such as “and”, “the” or “to” in English. In this process some very common words that appear to provide little or no value to the NLP objective are filtered and excluded from the text to be processed, hence removing widespread and frequent terms that are not informative about the corresponding text.

Stop words can be safely ignored by carrying out a lookup in a pre-defined list of keywords, freeing up database space and improving processing time.

The thing is stop words removal can wipe out relevant information and modify the context in a given sentence. 
# **Stemming:**
Refers to the process of slicing the end or the beginning of words with the intention of removing affixes (lexical additions to the root of the word).

#** Affixes that are attached at the beginning of the word are called prefixes (e.g. “astro” in the word “astrobiology”) and the ones attached at the end of the word are called suffixes (e.g. “ful” in the word “helpful”**).


The problem is that affixes can create or expand new forms of the same word (called inflectional affixes), or even create new words themselves (called derivational affixes). In English, prefixes are always derivational (the affix creates a new word as in the example of the prefix “eco” in the word “ecosystem”), but suffixes can be derivational (the affix creates a new word as in the example of the suffix “ist” in the word “guitarist”) or inflectional (the affix creates a new form of word as in the example of the suffix “er” in the word “faster”).


Playing------ Play

News------- New (Wrong)
# ***Part_Of_speech Tagging:***

Basically, the goal of a POS tagger is to assign linguistic (mostly grammatical) information to sub-sentential units. Such units are called tokens and, most of the time, correspond to words and symbols (e.g. punctuation).


CC coordinating conjunction

CD cardinal digit

DT determiner

EX existential there (like: “there is” … think of it like “there exists”)

FW foreign word

IN preposition/subordinating conjunction

JJ adjective ‘big’

JJR adjective, comparative ‘bigger’

JJS adjective, superlative ‘biggest’

LS list marker 1)

MD modal could, will

NN noun, singular ‘desk’

NNS noun plural ‘desks’

NNP proper noun, singular ‘Harrison’

NNPS proper noun, plural ‘Americans’

PDT predeterminer ‘all the kids’

POS possessive ending parent‘s

PRP personal pronoun I, he, she
# ***Stemming Algorithm Examples:***

Two stemming algorithms I immediately came in contact with when I first started 


using stemming were the Porter stemmer and the Snowball stemmer from NLTK. While I won’t go into a lot of details about either, I will highlight a little bit about them so that you can know even more than I did when I first started using them.


*   ***Porter stemmer:*** This stemming algorithm is an older one. It’s from the 1980s and its main concern is removing the common endings to words so that they can be resolved to a common form. It’s not too complex and development on it is frozen. Typically, it’s a nice starting basic stemmer, but it’s not really advised to use it for any production/complex application. Instead, it has its place in research as a nice, basic stemming algorithm that can guarantee reproducibility. It also is a very gentle stemming algorithm when compared to others.
*   ***Snowball stemmer:*** This algorithm is also known as the Porter2 stemming algorithm. It is almost universally accepted as better than the Porter stemmer, even being acknowledged as such by the individual who created the Porter stemmer. That being said, it is also more aggressive than the Porter stemmer. A lot of the things added to the Snowball stemmer were because of issues noticed with the Porter stemmer. There is about a 5% difference in the way that Snowball stems versus Porter.


*   ***Lancaster stemmer: ***Just for fun, the Lancaster stemming algorithm is another algorithm that you can use. This one is the most aggressive stemming algorithm of the bunch. However, if you use the stemmer in NLTK, you can add your own custom rules to this algorithm very easily. It’s a good choice for that. One complaint around this stemming algorithm though is that it sometimes is overly aggressive and can really transform words into strange stems. Just make sure it does what you want it to before you go with this option!
# ***Lemmatization:***
We’ve talked about stemming, but what about the other side of things? How is lemmatization different? Well, if we think of stemming as just take a best guess of where to snip a word based on how it looks, lemmatization is a more calculated process. It involves resolving words to their dictionary form. In fact, a lemma of a word is its dictionary or canonical form!
Because lemmatization is more nuanced in this respect, it requires a little more to actually make work. For lemmatization to resolve a word to its lemma, it needs to know its part of speech. That requires extra computational linguistics power such as a part of speech tagger. This allows it to do better resolutions (like resolving is and are to “be”).
Another thing to note about lemmatization is that it’s often times harder to create a lemmatizer in a new language than it is a stemming algorithm. Because lemmatizers require a lot more knowledge about the structure of a language, it’s a much more intensive process than just trying to set up a heuristic stemming algorithm.
# ***Why is Lemmatization better than Stemming?***
Stemming algorithm works by cutting the suffix from the word. In a broader sense cuts either the beginning or end of the word.
On the contrary, Lemmatization is a more powerful operation, and it takes into consideration morphological analysis of the words. It returns the lemma which is the base form of all its inflectional forms. In-depth linguistic knowledge is required to create dictionaries and look for the proper form of the word. Stemming is a general operation while lemmatization is an intelligent operation where the proper form will be looked in the dictionary. Hence, lemmatization helps in forming better machine learning features.
# ***Word2Vec***
How can we build simple, scalable, fast to train models which can run over billions of words that will produce exceedingly good word representations? Let’s look into Word2Vec model to find answer to this.

Word2Vec is a group of models which helps derive relations between a word and its contextual words. Let’s look at two important models inside Word2Vec: Skip-grams and CBOW

In Skip-gram model, we take a centre word and a window of context (neighbor) words and we try to predict context words out to some window size for each centre word. So, our model is going to define a probability distribution i.e. probability of a word appearing in context given a centre word and we are going to choose our vector representations to maximize the probability.
# ***Continuous Bag of Words model (CBOW)***
In abstract terms, this is opposite of skip-gram. In CBOW, we try to predict centre word by summing vectors of surrounding words.
This was about converting words into vectors. But where does the “learning” happen? Essentially, we begin with small random initialization of word vectors. Our predictive model learns the vectors by minimizing the loss function. In Word2vec, this happens with a feed-forward neural network and optimization techniques such as Stochastic gradient descent. There are also count-based models which make a co-occurrence count matrix of words in our corpus; we have a large matrix with each row for the “words” and columns for the “context”. The number of “contexts” is of course large, since it is essentially combinatorial in size. To overcome this size issue, we apply SVD to the matrix. This reduces the dimensions of the matrix retaining maximum information.
# ***TF-IDF Vectorizer:***

TF-IDF stands for term frequency-inverse document frequency. TF-IDF weight is a statistical measure used to evaluate how important a word is to a document in a collection or corpus. The importance increases proportionally to the number of times a word appears in the document but is offset by the frequency of the word in the corpus.


*  ***Term Frequency (TF):*** is a scoring of the frequency of the word in the current document. Since every document is different in length, it is possible that a term would appear much more times in long documents than shorter ones. The term frequency is often divided by the document length to normalize.

![alt text](https://miro.medium.com/max/808/1*SUAeubfQGK_w0XZWQW6V1Q.png)

*   ***Inverse Document Frequency (IDF):*** is a scoring of how rare the word is across documents. IDF is a measure of how rare a term is. Rarer the term, more is the IDF score.


![alt text](https://miro.medium.com/max/822/1*T57j-UDzXizqG40FUfmkLw.png)

Thus

![alt text](https://miro.medium.com/max/215/1*YrgmAeG7KNRB4dQcGcsdyg.png)
