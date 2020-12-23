"""
数据清理脚本
"""
import pandas as pd
from normalize import NormalizePipline

data = pd.read_excel("excel_data.xlsx")

pipline = NormalizePipline(data,
                           src_colomn="原文",
                           tgt_colomn="译文",
                           src_lang="zh",
                           tgt_lang="en")

# 丢弃数据
pipline.first_clean_rules(). \
        filter_too_long(). \
        filter_too_short(). \
        filter_3rdlang(). \
        filter_garbled(). \
        normalize_punc(). \
        align_end_punc()

# 获取丢弃的数据
pipline.rubbish
# 获取修改的数据
pipline.modified
# 获取可用的数据 （未修改数据+修改的数据，不包含丢弃的数据）
pipline.data
