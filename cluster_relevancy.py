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
from numpy import genfromtxt


def get_cluster_filter_1():
    dict_1 = dict.fromkeys(get_leafs(), 0)
    dict_1["двигатель не мощный"] = 0.318
    dict_1["двигатель не надежный"] = 0.344
    dict_1["двигатель ест масло"] = 0.311
    dict_1["двигатель не экономичный"] = 0.282
    dict_1["подвеска жесткая"] = 0.174
    dict_1["машина не вместительная"] = 0.183
    dict_1["машина не очень вместительная"] = 0.177
    dict_1["немного места"] = 0.169
    dict_1["маленький багажник"] = 0.185
    dict_1["авто медленное"] = 0.171
    dict_1["авто очень медленное"] = 0.162
    dict_1["бракованные запчасти"] = 0.177
    dict_1["дорогие запчасти"] = 0.194
    dict_1["недешевые запчасти"] = 0.189
    dict_1["дорогое обслуживание"] = 0.286
    dict_1["недешевое обслуживание"] = 0.264
    dict_1["не комфортная"] = 0.232
    dict_1["не очень комфортная"] = 0.217
    keys = np.array(list(dict_1.values()))
    return keys


def get_cluster_filter_2():
    dict_2 = dict.fromkeys(get_leafs(), 0)
    dict_2["двигатель мощный"] = 0.352
    dict_2["двигатель надежный"] = 0.396
    dict_2["двигатель не ест масло"] = 0.311
    dict_2["двигатель экономичный"] = 0.306
    dict_2["маленький расход"] = 0.161
    dict_2["машина вместительная"] = 0.191
    dict_2["машина очень вместительная"] = 0.190
    dict_2["большой багажник"] = 0.164
    dict_2["дешевые запчасти"] = 0.196
    dict_2["недорогие запчасти"] = 0.179
    dict_2["недорогое обслуживание"] = 0.269
    dict_2["надежная"] = 0.182
    dict_2["очень надежная"] = 0.178
    dict_2["комфортная"] = 0.280
    dict_2["очень комфортная"] = 0.259
    dict_2["комфортабельная"] = 0.211
    keys = np.array(list(dict_2.values()))
    return keys


def get_cluster_filter_3():
    dict_3 = dict.fromkeys(get_leafs(), 0)
    dict_3["шумоизоляция на уровне"] = 0.218
    dict_3["замечательная шумоизоляция"] = 0.175
    dict_3["дорогие запчасти"] = 0.162
    dict_3["дорогое обслуживание"] = 0.245
    dict_3["недешевое обслуживание"] = 0.213
    dict_3["комфортная"] = 0.678
    dict_3["очень комфортная"] = 0.450
    dict_3["комфортабельная"] = 0.357
    keys = np.array(list(dict_3.values()))
    return keys


def get_cluster_filter_4():
    dict_4 = dict.fromkeys(get_leafs(), 0)
    dict_4["двигатель мощный"] = 0.470
    dict_4["двигатель надежный"] = 0.416
    dict_4["двигатель не ест масло"] = 0.342
    dict_4["двигатель экономичный"] = 0.382
    dict_4["шумоизоляция на уровне"] = 0.224
    dict_4["замечательная шумоизоляция"] = 0.239
    dict_4["дорогое обслуживание"] = 0.240
    dict_4["недешевое обслуживание"] = 0.210
    dict_4["не комфортная"] = 0.255
    dict_4["не очень комфортная"] = 0.216
    keys = np.array(list(dict_4.values()))
    return keys


def get_cluster_filter_5():
    dict_5 = dict.fromkeys(get_leafs(), 0)
    dict_5["двигатель не мощный"] = 0.437
    dict_5["двигатель не надежный"] = 0.392
    dict_5["двигатель ест масло"] = 0.416
    dict_5["двигатель не экономичный"] = 0.367
    dict_5["слабая шумоизоляция"] = 0.284
    dict_5["недорогое обслуживание"] = 0.284
    dict_5["очень комфортная"] = 0.154
    dict_5["комфортабельная"] = 0.169
    keys = np.array(list(dict_5.values()))
    return keys


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

    x = text.split('\n    ')
    #leafs = map(lambda x: x[4:], x)
    return x


if __name__ == '__main__':
    leafs = get_leafs()
    first_cluster = get_cluster_filter_1()
    second_cluster = get_cluster_filter_2()
    third_cluster = get_cluster_filter_3()
    fourth_cluster = get_cluster_filter_4()
    fifth_cluster = get_cluster_filter_5()

    my_data = genfromtxt('/Users/daniilbabin/DS/Taxonomy/final_2.csv', delimiter=',')
    my_data_2 = my_data[1:, 1:]

    first_cluster_dist = my_data_2.dot(first_cluster)
    second_cluster_dist = my_data_2.dot(second_cluster)
    third_cluster_dist = my_data_2.dot(third_cluster)
    fourth_cluster_dist = my_data_2.dot(fourth_cluster)
    fifth_cluster_dist = my_data_2.dot(fifth_cluster)

    filenames = sorted(glob.glob('/Users/daniilbabin/DS/Taxonomy/35_reviews/*'))
    filenames_array = np.array(filenames)

    data = {'car_model': filenames_array,
            'first_cluster': first_cluster_dist,
            'second_cluster': second_cluster_dist,
            'third_cluster': third_cluster_dist,
            'fourth_cluster': fourth_cluster_dist,
            'fifth_cluster': fifth_cluster_dist}

    df = pd.DataFrame(data, columns=['car_model', 'first_cluster', 'second_cluster', 'third_cluster',
                                     'fourth_cluster', 'fifth_cluster'])

    df.to_excel('/Users/daniilbabin/DS/Taxonomy/cluster_relevancy.xlsx', index=False)
