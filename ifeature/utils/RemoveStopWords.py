# -*- coding: utf-8 -*-
"""
__title__ = 'RemoveStopWords'
__author__ = 'JieYuan'
__mtime__ = '18-10-30'
"""
import re
import jieba


class RemoveStopWords(object):

    def __init__(self, stop_words_path):
        self._get_stop_words(stop_words_path)

    def get_tokens(self, sentence, pure=True, join=False):
        jieba.enable_parallel(5)
        if pure:
            pure_pattern = re.compile('[^0-9A-Za-z\u4e00-\u9fa5]+')
            ws = jieba.cut(pure_pattern.sub(' ', sentence.lower()))
        else:
            ws = jieba.cut(sentence.lower())

        ws = [w for w in ws if w not in self.stop_words.union([' ', '\t'])]

        return ' '.join(ws) if join else ws

    def _get_stop_words(self, path):
        print('Loading Stop Words ...')
        with open(path) as f:
            self.stop_words = {w.strip() for w in f.readlines()}
