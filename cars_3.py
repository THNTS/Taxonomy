import string
from east.asts import base
from east import utils
import glob
import errno
import pandas as pd
import numpy as np
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from multiprocessing.dummy import Pool
import tqdm


def clear_text(text, lowerize=True):

    pat = re.compile(r'[^A-Za-z0-9 \-\n\r.,;!?А-Яа-я]+')
    cleared_text = re.sub(pat, ' ', text)

    if lowerize:
        cleared_text = cleared_text.lower()

    tokens = cleared_text.split()
    return tokens

def text_substrings(text, k):
    substrings_array = []
    for i in range(len(text) - k + 1):
        substring = " ".join(text[i : i + k])
        substrings_array.append(substring)
        i+=1
    return substrings_array

def calculate(text):
    new = []
    if (len(text)) != 0:
        ast = base.AST.get_ast(text)
        # Compute the relevance of a keyphrase to the text collection indexed by this AST.
        # The relevance score will always be in [0; 1]
        for j in range(len(leafs)):
            new.append(ast.score(leafs[j]))
    else:
        new =np.zeros(len(leafs))
    return new

def get_leafs():
    text = """двигатель мощный
    двигатель надежный
    двигатель не ест масло
    двигатель экономичный
    двигатель не мощный 
    двигатель не надежный
    двигатель ест масло
    двигатель не экономичный
    машина экономичная
    небольшой расход
    маленький расход
    машина не экономичная
    большой расход
    переключает быстро
    переключает медленно
    подвеска мягкая
    подвеска высокая
    подвеска жесткая
    подвеска низкая
    крепкий кузов
    кузов не гниет
    кузов не склонен к коррозии
    кузов гниет 
    кузов склонен к коррозии
    машина вместительная
    машина очень вместительная
    просторно
    много места
    машина не вместительная
    машина не очень вместительная
    не просторно
    немного места
    отличный обзор
    большие зеркала
    плохой обзор
    слабый обзор
    маленькие зеркала
    салон супер
    удобный салон
    печка жарит
    кондиционер морозит
    шумоизоляция на уровне
    классная шумоизоляция 
    замечательная шумоизоляция
    дорогу не слышно
    улицу не слышно
    как в танке
    маленький салон
    не удобный салон
    слабая шумоизоляция
    все слышно
    большой багажник
    маленький багажник
    отличное управление
    легкое управление
    неплохая управляемость
    сложное управление
    плохая управляемость
    ураганная динамика
    хорошая динамика
    динамичное авто
    машина очень быстрая 
    очень шустрая
    шустрая
    динамика супер
    динамики хватает
    динамики за глаза
    разгон быстрый
    слабая динамика
    авто медленное
    авто очень медленное
    динамики не хватает
    медленный разгон
    детали не дорогие
    дешевые детали
    дешевые запчасти
    недорогие запчасти
    дорогие детали
    недешевые детали
    бракованные запчасти
    дорогие запчасти
    недешевые запчасти
    недорогое обслуживание
    дешевое обслуживание 
    дорогое обслуживание
    недешевое обслуживание
    куплена в кредит
    куплена у дилера
    куплена у друга
    куплена у знакомого
    куплена у одноклубника
    недорогая страховка
    страховка дорогая
    маленький налог
    мизерный налог
    не большой налог
    налог большой
    огромный налог
    надежная
    очень надежная
    как часы
    ломкая
    не надежная 
    не очень надежная
    комфортная
    очень комфортная
    комфортабельная
    удобная
    очень удобная
    корабль
    плывет по дороге
    не комфортная
    не очень комфортная
    не комфортабельная
    неудобная
    не очень удобная"""

    leafs = text.split('\n')
    return leafs

if __name__ == '__main__':
    leafs = get_leafs()
    filenames = sorted(glob.glob('/Users/daniilbabin/DS/Taxonomy/35_reviews/*'))

    df = pd.DataFrame(0, index=np.arange(len(leafs)), columns=leafs)
    array = np.zeros(shape=(len(filenames), len(leafs)))
    scores = []
    texts = []

    for name in filenames:
        l = ''
        with open(name, encoding='utf-8') as f:
            i = 0
            for line in f:
                if i >= 8:
                    l = l + " " + line
                i += 1
        review = clear_text(l)
        texts.append(review)

    strings_collection = []

    for text in texts:
        strings_collection.append(text_substrings(text, 5))

    #print(strings_collection)
    # list = [18,19,20,21,27,44,77,80,99]
    #
    # for i in list:
    #     print(i)
    #     print("\n")
    #     print(strings_collection[i])
    #     print("\n")
    #     print(calculate(strings_collection[i]))
    #     print("\n")

    pool = Pool(1000)
    # for _ in tqdm.tqdm(pool.imap_unordered(calculate, strings_collection), total=len(strings_collection)):
    #     pass

    results = pool.map(calculate, strings_collection[20000:21000])

    # Close the pool and wait for the work to finish
    pool.close()
    pool.join()

    df = pd.DataFrame(results, index=np.arange(1000), columns=leafs)
    #df.to_excel("/Users/daniilbabin/DS/Taxonomy/true.xlsx", sheet_name='first_9999')
    df.to_csv("/Users/daniilbabin/DS/Taxonomy/true_21.csv")
