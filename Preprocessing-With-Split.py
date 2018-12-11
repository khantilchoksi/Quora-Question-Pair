# coding: utf-8
from nltk.stem import SnowballStemmer, WordNetLemmatizer
from collections import defaultdict
from nltk.corpus import stopwords
from string import punctuation
from nltk import pos_tag
import numpy as np
import pandas as pd
import nltk
import sys
import os
import re

stop_words = set(stopwords.words('english'))
tag_map = defaultdict(lambda: "n")
tag_map['J'], tag_map['V'], tag_map['R'] = "a", "v", "r"
stemmer = SnowballStemmer('english')
lemmatizer = WordNetLemmatizer()
replacements = [
    # Expanding contractions
    (r"\b([A-Za-z]+)'s\b", '\\1 is'), (r"\b([A-Za-z]+)'re\b", '\\1 are'),
    (r"\b([A-Za-z]+)'ve\b", '\\1 have'), (r"\b([A-Za-z]+)'ll\b", '\\1 will'),
    (r"\b([A-Za-z]+)n't\b", '\\1 not'),
    (" whats ", " what is "), (" whos ", "who is "), ("wheres ", "where is"),
    (" whens ", " when is "), (" hows ", " how is "), (" im ", " i am "),
    (" hes ", " he is "), (" shes ", " she is "), ("thats ", "that is "),
    ("theres ", "there is "), (" isnt ", " is not "), ("wasnt", "was not"),
    (" arent", " are not"), ("werent", "were not"), (" cant ", " can not "),
    ("cannot", "can not"), ("couldnt", "could not"), (" dont", " do not"),
    ("didnt", "did not"), ("shouldnt", "should not"), ("wouldnt", "would not"),
    ("doesnt", "does not"), ("havent", "have not"), ("hasnt", "has not"),
    ("hadnt", "had not"),

    # Spelling mistakes
    ("colour", "color"), (" centre ", " center "), (" grey ", " gray "), (" favourite ", " favorite "),
    (" travelling ", " traveling "), ("programing", "programming"), ("calender", "calendar"),
    ("intially", "initially"), ("quikly", "quickly"), ("imrovement", "improvement"),
    ("demonitization", "demonetization"), ("demonetisation", "demonetization"), ("demonetize", "demonetization"),
    ("canceled", 'canceled'), (" defence ", " defense "), ("programme", "program"),
    ("actived", "active"), (" tution", " tuition"), ("banglore", "bangalore"), ('behaviour', 'behavior'),
    ("bengaluru", "bangalore"), ('organisation', 'organization'), ('realise', 'realize'),
    ('accomodation', 'accommodation'), ('adress', 'address'), ('alot', 'a lot'), ('athiest', 'atheist'),
    ('beleive', 'believe'), ('enviroment', 'environment'), ('freind', 'friend'), ('goverment', 'government'),
    ('grammer', 'grammar'), ('hight', 'height'), ('independance', 'independence'), ('intresting', 'interesting'),
    ('lazer', 'laser'), ('lightening', 'lightning'), ('ninty',
                                                      'ninety'), ('noone', 'no one'), ('oppurtunity', 'opportunity'),
    ('recieve', 'receive'), ('reccomend', 'recommend'), ('seperate', 'separate'), ('thier', 'their'),
    ('vegeterian', 'vegetarian'), ('writen', 'written'), ('writting', 'writing'), ('truely', 'truly'),
    ('commitee', 'committee'), ('independant', 'independent'), ('foriegn', 'foreign'), ('happend', 'happened'),
    ('publically', 'publicly'), ('realy', "really"), ('tatoo',
                                                      'tattoo'), ('prefered', 'preferred'), ('technologyss', 'technology'),
    ('technologys', 'technology'), ('programr', 'programmer'), ('healtheir', 'healthier'), ('aadhaar', 'aadhar'),

    # Expanding abbreviations
    ('aadhar', 'indian identity card'),
    ("upsc", "union public service commission"), (" ugc ", " university grants commission "),
    (" sbi ", " state bank of india "), (" iit", " indian institute of technology"),
    (" iims ", " indian institute of management "), ("mca", "master of computer application"),
    ("mds", "master of dental surgery"), ("nba", "national basketball association"),
    (" iim", " indian institute of management "), ("mbbs", " bachelor of medicine bachelor of surgery "),
    ("bitcoin", " bit coin "), (" iisc ", " indian institute of science "),
    ('isro', "indian space research organization"),
    ('toefl', " test of english as a foreign language"), ("mtech", "master of engineering"),
    ("aiims", " all india institutes of medical sciences"), ("gmat", "graduate management admission test"),
    (" gre ", " graduate record examinations "), ("mnc", "multinational corporation"),
    ('tcs', 'tata consultancy service'), ('kvpy', 'indian scholarship'),
    ('ibps', 'institute of banking personnel selection'),
    ('irctc', 'indian railway catering and tourism corporation'),
    ('ielts', 'international english language testing system'),
    ("brexit", "british exit"), (" bba ", " bachelor of business administration "),
    (" mba ", " master of business administration "), (" obc", " other backward caste"),
    (" cs ", " computer science "), (" cse ", " computer science "),
    (' ece ', ' electronics and communication engineering '), (" btech", " bachelor of technology"),
    (" nra ", " national rifle association "),
    ("kms", " kilometers "), (r"\0rs ", " rupees "),
    (" uk ", " england "), (" u s ", " usa "),
    ("the us ", "usa"), (" americaa ", " america "),
    ("e-mail", "email"), (" e mail", "email"), (" 9 11 ", "911"), (" b g ", " bg "),
    ('₹', ' rupee '), (' txt ', " text "), ("cgpa", "grade point average"), (" J K ", " JK "),

    (" imessage ", " message application "), (' wechat ', " chat application"),
    (" ios", " operating system"), (" iPhone ", " phone "), (" iphone ", " phone "),
    (" i phone ", " phone "), ("watsapp", "whatsapp"), (" OS ", " operating system "), ("Wi-Fi", "wifi"),

    (" bestfriend", " best friend"), (" bf ", " boy friend "), (" gf ", " girl friend "),
    (" boyfriend", " boy friend "), (" girlfriend", " girl friend"),

    ("upvote", "up vote"), (" downvotes ", " up votes "),

    # Years
    (" 1990", " year nineteen ninety "), (" 1991",
                                          " year nineteen ninety one "), (" 1992", " year nineteen ninety two"),
    (" 2001", " year two thousand one"), (" 2002", " year two thousand two"), (" 2003", " year two thousand three"),
    (" 2010", " year two thousand ten"), (" 2011", " year two thousand eleven"), (" 2012", " year two thousand twelve"),
    (" 2013", " year two thousand thirteen"), (" 2014 ", " year two thousand fourteen"),
    (" 2015", " year two thousand fifteen"), (" 2016",
                                              " year two thousand sixteen"), (" 2017", " year two thousand seventeen"),
    (" 2018", " year two thousand eighteen"), (" 2019",
                                               " year two thousand nineteen"), (" 2020", " year two thousand twenty"),
    (" 2021", " year two thousand twenty one"),

    # Meaning of most commonly occuring non-vocabulary terms
    ('quorans', 'quora users'), ('quora', 'question and answer website'),
    ('instagram', 'photo and video sharing social network'),
    ('whatsapp', 'messenger'), ('snapchat', 'multimedia messaging app'),
    ('paytm', 'indian ecommerce payment system'), ('redmi',
                                                   'phone company'), ('flipkart', 'indian electronic commerce company'),
    ('jio', 'indian mobile network operator'), ('spotify',
                                                          'music streaming platform'), ('snapdeal', 'indian ecommerce company'),
    ('accenture', 'global management consulting and professional services firm'),
    ('narendra', 'indian prime minister'), ('goswami',
                                            'indian television anchor'), ('arnab', 'indian television anchor'),
    ('airbnb', 'online marketplace and hospitality service'),
    ('kejriwal', 'delhi chief minister'), ('arvind', 'delhi chief minister'), ('cambodia', 'southeast asian nation'),
    ('h1b', 'american work visa'), ('xiaomi', 'chinese electronics company'), (" cheque ", " bank draft"),
    ('zuckerberg', 'facebook founder'),

    # Numbers
    (" II ", " two "), (" III ", " three "), (" V ", " five "),
    (" 1st", " first"), (" 2nd", " second"), (" 3rd", " third"), (" 4th", " fourth"), (" 5th", " fifth"),
    (" 6th", " sixth"), (" 7th", " seventh"), (" 8th", " eigth"), (" 9th", " ninth"), (" 10th", " tenth"),
    ("11th", " eleventh"), (" 12th", " twelfth"),
    (" 00000", " 0"), (" 0000", " 0"), (" 000", " 0 "), (" 00", " 0"), (" 0 ", " zero "),
    (" 1 ", " one "), (" 01 ", " one "), (" 2 ", " two "), (" 3 ", " three "), (" 4 ", " four "), (" 5 ", " five "),
    (" 6 ", " six "), (" 7 ", " seven "), (" 8 ", " eight "), (" 9 ",
                                                               " nine "), (" 10 ", " ten "), (" 11 ", " eleven "),
    (' 12 ', ' twelve '), (' 13 ', ' thirteen '), (' 14 ',
                                                   ' fourteen '), (' 15 ', ' fifteen '), (' 16 ', ' sixteen '),
    (' 17 ', ' seventeen '), (' 18 ', ' eighteen '), (' 19 ', ' nineteen '), (" 20 ", " twenty "),
    (" 21 ", " twenty one "), (" 24 ", " twenty four "), (" 25 ", " twenty five "),
    (' 30 ', ' thirty '), (' 36 ', " thirty six "), (" 40 ", " forty "), (" 50 ", " fifty "), (" 60 ", " sixty "),
    (" 70 ", " seventy "), (" 80 ", " eighty "), (" 90 ", " ninety "), (" 99 ", " ninety nine "),
    (" 100 ", " hundred "), (" 200 ", " two hundred "), (" 250 ", " two fifty "), (" 500 ", " five hundred "),
    (" 1000 ", " thousand "), (" 2000 ", " two thousand "), ("10k", " ten thousand "), ("30k", " thirty thousand "),
    ("60k", " sixty thousand "),
    (r"\0s", "0"), (r"\s{2,}", " "), (r"[^A-Za-z0-9]", " "),
    ('\s+', ' '),  # replace multi space with one single space
]


def clean_text(text, remove_stop_words=False, lemmatize=True, stem_words=False):
    # Given a text as string
    # 1. Converts it into lower case
    # 2. Replaces the old string patterns with new ones
    # 3. Removes all kinds of punctuation
    # 4. Optionally, lemmatizes every word
    # 5. Optionally, removes stop words
    # 6. Optionally, converts every word to its rootword
    # 7. Returns the processed text as a string

    if type(text) != str:
        return str(text).lower()

    text = text.lower()

    # Replace old patterns with new
    for old, new in replacements:
        text = re.sub(old, new, text)

    text = text.lower()

    # Remove punctuation from text
    text = ''.join([c for c in text if c not in punctuation])

    text = text.split()

    # Lemmatize words
    if lemmatize:
        text = [lemmatizer.lemmatize(word, tag_map[tag[0]]) for word, tag in pos_tag(text)]

    # Remove stop words
    if remove_stop_words:
        text = [w for w in text if not w in stop_words]

    # Shorten words to their stems
    if stem_words:
        text = [stemmer.stem(word) for word in text]

    text = " ".join(text)

    text = text.lower()

    # Return the clean text as string
    return(text)


data = pd.read_csv('qq-train.csv').fillna("")
q1, q2 = data[['qid1', 'question1']], data[['qid2', 'question2']]
q1.columns = ['qid', 'question']
q2.columns = ['qid', 'question']
question_data = pd.concat((q1, q2), axis=0).fillna("").sort_values(by='qid').drop_duplicates('qid')
question_data['question'] = question_data['question'].apply(lambda x: clean_text(x, True, False, False))
print('Questions done.')
question_data.to_csv('questions.csv', index=False)
data[['qid1','qid2','is_duplicate']].to_csv('indexes.csv', index=False)
