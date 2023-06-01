import os
import torch
import pandas as pd
import numpy as np
import pickle
from tqdm.auto import tqdm
import re
import contractions

from datasets import Dataset

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import DataCollatorWithPadding

from torch.nn import functional as F
from torch.utils.data import DataLoader

def remove_URL(text):
    """
        Remove URLs from a sample string
    """
    return re.sub(r"https?://\S+|www\.\S+", "", text)


def remove_html(text):
    """
        Remove the html in sample text
    """
    html = re.compile(r"<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});")
    return re.sub(html, "", text)


def remove_non_ascii(text):
    """
        Remove non-ASCII characters
    """
    return re.sub(r'[^\x00-\x7f]', r'', text)


def remove_special_characters(text):
    """
        Remove special special characters, including symbols, emojis, and other graphic characters
    """
    emoji_pattern = re.compile(
        '['
        u'\U0001F600-\U0001F64F'  # emoticons
        u'\U0001F300-\U0001F5FF'  # symbols & pictographs
        u'\U0001F680-\U0001F6FF'  # transport & map symbols
        u'\U0001F1E0-\U0001F1FF'  # flags (iOS)
        u'\U00002702-\U000027B0'
        u'\U000024C2-\U0001F251'
        ']+',
        flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)


def remove_hashtag(text):
    return re.sub(r'#([^\s]+)', r'\1', text)  # hashtag symbol


def remove_punct(text):
    """
        Remove the punctuation
    """
    return re.sub(r'[]!"$%&\'()*+,/:;=#@?[\\^_`{|}~-]+', "", text)  # exclude a dot


def other_clean(text):
    """
        Other manual text cleaning techniques
    """
    # Typos, slang and other
    sample_typos_slang = {
        "w/e": "whatever",
        "usagov": "usa government",
        "recentlu": "recently",
        "ph0tos": "photos",
        "amirite": "am i right",
        "exp0sed": "exposed",
        "<3": "love",
        "luv": "love",
        "amageddon": "armageddon",
        "trfc": "traffic",
        "16yr": "16 year"
    }

    # Acronyms
    sample_acronyms = {
        "mh370": "malaysia airlines flight 370",
        "okwx": "oklahoma city weather",
        "arwx": "arkansas weather",
        "gawx": "georgia weather",
        "scwx": "south carolina weather",
        "cawx": "california weather",
        "tnwx": "tennessee weather",
        "azwx": "arizona weather",
        "alwx": "alabama weather",
        "usnwsgov": "united states national weather service",
        "2mw": "tomorrow"
    }

    # Some common abbreviations
    sample_abbr = {
        "$": " dollar ",
        "€": " euro ",
        "4ao": "for adults only",
        "a.m": "before midday",
        "a3": "anytime anywhere anyplace",
        "aamof": "as a matter of fact",
        "acct": "account",
        "adih": "another day in hell",
        "afaic": "as far as i am concerned",
        "afaict": "as far as i can tell",
        "afaik": "as far as i know",
        "afair": "as far as i remember",
        "afk": "away from keyboard",
        "app": "application",
        "approx": "approximately",
        "apps": "applications",
        "asap": "as soon as possible",
        "asl": "age, sex, location",
        "atk": "at the keyboard",
        "ave.": "avenue",
        "aymm": "are you my mother",
        "ayor": "at your own risk",
        "b&b": "bed and breakfast",
        "b+b": "bed and breakfast",
        "b.c": "before christ",
        "b2b": "business to business",
        "b2c": "business to customer",
        "b4": "before",
        "b4n": "bye for now",
        "b@u": "back at you",
        "bae": "before anyone else",
        "bak": "back at keyboard",
        "bbbg": "bye bye be good",
        "bbc": "british broadcasting corporation",
        "bbias": "be back in a second",
        "bbl": "be back later",
        "bbs": "be back soon",
        "be4": "before",
        "bfn": "bye for now",
        "blvd": "boulevard",
        "bout": "about",
        "brb": "be right back",
        "bros": "brothers",
        "brt": "be right there",
        "bsaaw": "big smile and a wink",
        "btw": "by the way",
        "bwl": "bursting with laughter",
        "c/o": "care of",
        "cet": "central european time",
        "cf": "compare",
        "cia": "central intelligence agency",
        "csl": "can not stop laughing",
        "cu": "see you",
        "cul8r": "see you later",
        "cv": "curriculum vitae",
        "cwot": "complete waste of time",
        "cya": "see you",
        "cyt": "see you tomorrow",
        "dae": "does anyone else",
        "dbmib": "do not bother me i am busy",
        "diy": "do it yourself",
        "dm": "direct message",
        "dwh": "during work hours",
        "e123": "easy as one two three",
        "eet": "eastern european time",
        "eg": "example",
        "embm": "early morning business meeting",
        "encl": "enclosed",
        "encl.": "enclosed",
        "etc": "and so on",
        "faq": "frequently asked questions",
        "fawc": "for anyone who cares",
        "fb": "facebook",
        "fc": "fingers crossed",
        "fig": "figure",
        "fimh": "forever in my heart",
        "ft.": "feet",
        "ft": "featuring",
        "ftl": "for the loss",
        "ftw": "for the win",
        "fwiw": "for what it is worth",
        "fyi": "for your information",
        "g9": "genius",
        "gahoy": "get a hold of yourself",
        "gal": "get a life",
        "gcse": "general certificate of secondary education",
        "gfn": "gone for now",
        "gg": "good game",
        "gl": "good luck",
        "glhf": "good luck have fun",
        "gmt": "greenwich mean time",
        "gmta": "great minds think alike",
        "gn": "good night",
        "g.o.a.t": "greatest of all time",
        "goat": "greatest of all time",
        "goi": "get over it",
        "gps": "global positioning system",
        "gr8": "great",
        "gratz": "congratulations",
        "gyal": "girl",
        "h&c": "hot and cold",
        "hp": "horsepower",
        "hr": "hour",
        "hrh": "his royal highness",
        "ht": "height",
        "ibrb": "i will be right back",
        "ic": "i see",
        "icq": "i seek you",
        "icymi": "in case you missed it",
        "idc": "i do not care",
        "idgadf": "i do not give a damn fuck",
        "idgaf": "i do not give a fuck",
        "idk": "i do not know",
        "ie": "that is",
        "i.e": "that is",
        "ifyp": "i feel your pain",
        "IG": "instagram",
        "iirc": "if i remember correctly",
        "ilu": "i love you",
        "ily": "i love you",
        "imho": "in my humble opinion",
        "imo": "in my opinion",
        "imu": "i miss you",
        "iow": "in other words",
        "irl": "in real life",
        "j4f": "just for fun",
        "jic": "just in case",
        "jk": "just kidding",
        "jsyk": "just so you know",
        "l8r": "later",
        "lb": "pound",
        "lbs": "pounds",
        "ldr": "long distance relationship",
        "lmao": "laugh my ass off",
        "lmfao": "laugh my fucking ass off",
        "lol": "laughing out loud",
        "ltd": "limited",
        "ltns": "long time no see",
        "m8": "mate",
        "mf": "motherfucker",
        "mfs": "motherfuckers",
        "mfw": "my face when",
        "mofo": "motherfucker",
        "mph": "miles per hour",
        "mr": "mister",
        "mrw": "my reaction when",
        "ms": "miss",
        "mte": "my thoughts exactly",
        "nagi": "not a good idea",
        "nbc": "national broadcasting company",
        "nbd": "not big deal",
        "nfs": "not for sale",
        "ngl": "not going to lie",
        "nhs": "national health service",
        "nrn": "no reply necessary",
        "nsfl": "not safe for life",
        "nsfw": "not safe for work",
        "nth": "nice to have",
        "nvr": "never",
        "nyc": "new york city",
        "oc": "original content",
        "og": "original",
        "ohp": "overhead projector",
        "oic": "oh i see",
        "omdb": "over my dead body",
        "omg": "oh my god",
        "omw": "on my way",
        "p.a": "per annum",
        "p.m": "after midday",
        "pm": "prime minister",
        "poc": "people of color",
        "pov": "point of view",
        "pp": "pages",
        "ppl": "people",
        "prw": "parents are watching",
        "ps": "postscript",
        "pt": "point",
        "ptb": "please text back",
        "pto": "please turn over",
        "qpsa": "what happens",  # "que pasa",
        "ratchet": "rude",
        "rbtl": "read between the lines",
        "rlrt": "real life retweet",
        "rofl": "rolling on the floor laughing",
        "roflol": "rolling on the floor laughing out loud",
        "rotflmao": "rolling on the floor laughing my ass off",
        "rt": "retweet",
        "ruok": "are you ok",
        "sfw": "safe for work",
        "sk8": "skate",
        "smh": "shake my head",
        "sq": "square",
        "srsly": "seriously",
        "ssdd": "same stuff different day",
        "tbh": "to be honest",
        "tbs": "tablespooful",
        "tbsp": "tablespooful",
        "tfw": "that feeling when",
        "thks": "thank you",
        "tho": "though",
        "thx": "thank you",
        "tia": "thanks in advance",
        "til": "today i learned",
        "tl;dr": "too long i did not read",
        "tldr": "too long i did not read",
        "tmb": "tweet me back",
        "tntl": "trying not to laugh",
        "ttyl": "talk to you later",
        "u.s.": 'usa',
        "u": "you",
        "u2": "you too",
        "u4e": "yours for ever",
        "utc": "coordinated universal time",
        "w/": "with",
        "w/o": "without",
        "w8": "wait",
        "wassup": "what is up",
        "wb": "welcome back",
        "wtf": "what the fuck",
        "wtg": "way to go",
        "wtpa": "where the party at",
        "wuf": "where are you from",
        "wuzup": "what is up",
        "wywh": "wish you were here",
        "yd": "yard",
        "ygtr": "you got that right",
        "ynk": "you never know",
        "zzz": "sleeping bored and tired"
    }

    sample_typos_slang_pattern = re.compile(
        r'(?<!\w)(' + '|'.join(re.escape(key) for key in sample_typos_slang.keys()) + r')(?!\w)')
    sample_acronyms_pattern = re.compile(
        r'(?<!\w)(' + '|'.join(re.escape(key) for key in sample_acronyms.keys()) + r')(?!\w)')
    sample_abbr_pattern = re.compile(r'(?<!\w)(' + '|'.join(re.escape(key) for key in sample_abbr.keys()) + r')(?!\w)')

    text = sample_typos_slang_pattern.sub(lambda x: sample_typos_slang[x.group()], text)
    text = sample_acronyms_pattern.sub(lambda x: sample_acronyms[x.group()], text)
    text = sample_abbr_pattern.sub(lambda x: sample_abbr[x.group()], text)

    return text


def transform_text(text):
    text = remove_URL(text)
    text = remove_html(text)
    text = remove_non_ascii(text)
    text = remove_special_characters(text)
    text = remove_hashtag(text)
    text = remove_punct(text)
    text = text.lower()
    text = other_clean(text)
    text = contractions.fix(text)
    text = text.lower()
    text = text.strip()
    return text


tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
tokenizer.pad_token_id = 105


def tokenize_function(data):
    return tokenizer(data['text'],
                     max_length=50,
                     truncation=True)


def make_prediction_en(input_text: str, transform_text=transform_text, tokenize_function=tokenize_function) -> str:
    model = pickle.load(open('model.pkl', 'rb'))

    labels_int2str = {0: 'фейк на тему развлечений', 1: 'правда на тему развлечений',
                      2: 'фейк на тему здоровья', 3: 'правда на тему здоровья',
                      4: 'фейк на тему политики', 5: 'правда на тему политики',
                      6: 'фейк на тему спорта', 7: 'правда на тему спорта'}

    input_text = transform_text(input_text)

    input_df = Dataset.from_pandas(pd.DataFrame({'text': [input_text],
                                                 'labels': [7]}))
    input_df = input_df.map(tokenize_function, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    input_df = input_df.remove_columns(["text"])
    input_df.set_format("torch")
    pred_dataloader = DataLoader(dataset=input_df, batch_size=1, collate_fn=data_collator)

    device = torch.device('cpu')
    model.to(device)

    model.eval()
    for batch in pred_dataloader:
        with torch.no_grad():
            outputs = model(**batch)
    logits = outputs.logits
    preds_proba = F.softmax(logits, dim=-1)
    predictions = torch.argmax(logits, dim=-1)
    print(preds_proba)
    return labels_int2str[int(predictions)]


def make_prediction_ru(input_text: str, transform_text=transform_text, tokenize_function=tokenize_function) -> str:
    model = pickle.load(open('model.pkl', 'rb'))

    tokenizer_for_translation = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-ru-en")
    model_for_translation = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-ru-en")

    input_text = tokenizer_for_translation(input_text, return_tensors='pt')
    generated_ids = model_for_translation.generate(**input_text)
    input_text = tokenizer_for_translation.batch_decode(generated_ids, skip_special_tokens=True)[0]

    labels_int2str = {0: 'фейк на тему развлечений', 1: 'правда на тему развлечений',
                      2: 'фейк на тему здоровья', 3: 'правда на тему здоровья',
                      4: 'фейк на тему политики', 5: 'правда на тему политики',
                      6: 'фейк на тему спорта', 7: 'правда на тему спорта'}

    input_text = transform_text(input_text)

    input_df = Dataset.from_pandas(pd.DataFrame({'text': [input_text],
                                                 'labels': [7]}))
    input_df = input_df.map(tokenize_function, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    input_df = input_df.remove_columns(["text"])
    input_df.set_format("torch")
    pred_dataloader = DataLoader(dataset=input_df, batch_size=1, collate_fn=data_collator)

    device = torch.device('cpu')
    model.to(device)

    model.eval()
    for batch in pred_dataloader:
        with torch.no_grad():
            outputs = model(**batch)
    logits = outputs.logits
    preds_proba = F.softmax(logits, dim=-1)
    predictions = torch.argmax(logits, dim=-1)

    return labels_int2str[int(predictions)]
