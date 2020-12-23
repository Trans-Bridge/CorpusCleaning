"""
Helper class for clean multilingual data.
"""
import re
import logging
import pandas as pd

from tqdm import tqdm

__all__ = ["NormalizePipline"]

class NormalizePipline(object):

    ZHCN_PUNC_REPLACEMENT = [
        (u"(", u"（"),
        (u")", u"）"),
        (u"/", u"／"),
        (u",", u"，"),
        (u";", u"；"),
        (u":", u"："),
        (u"!", u"！"),
        (u"?", u"？")
    ]

    LATIN_PUNC_REPLACEMENT = [
        (u"’", u"'"),
        (u"：", u":"),
        (u"。", u"."),
        (u"；", u";"),
        (u"×", u"X"),
        (u"“", u'"'),
        (u"”", u'"'),
        (u"．", u"."),
        (u"～", u"~"),
        (u"）", u")"),
        (u"（", u"("),
        (u"＞", u">"),
        (u"＜", u"<"),
        (u"＝", u"="),
        (u"？", u"?"),
        (u"！", u"!"),
        (u"＋", u"+"),
        (u"－", u"-"),
        (u"［", u"["),
        (u"]", u"]"),
        (u"｛", u"{"),
        (u"｝", u"}"),
        (u"＆", u"&"),
        (u"＾", u"^"),
        (u"＄", u"$"),
        (u"＃", u"#"),
        (u"＠", u"@"),
        (u"＊", u"*"),
        (u"＼", u"\\"),
        (u"｜", u"|"),
        (u"／", u""),
        (u"/", u"/"),
        (u"×", u"X")
    ]

    # enzh - latin
    END_PUNC_MAPPING = {
        "；": ";",
        "：": ":",
        "！": "!",
        "？": "?",
        "。": ".",
        "，": ","
    }

    ROMAN = [
        "I",
        "II",
        "III",
        "IV",
        "V",
        "VI",
        "VII",
        "VIII",
        "IX",
        "X",
        "XI",
        "XII",
        "XIII",
        "XIV",
        "XV",
        "XVI",
        "XVII",
        "XVIII",
        "XIX",
        "XX",
        "XXI",
        "XXII",
        "XXIII",
        "XXIV",
        "XXV",
        "XXVI",
        "XXVII",
        "XXVIII",
        "XXIX",
        "XXX",
        "XXXI",
        "XXXII",
        "XXXIII",
        "XXXIV",
        "XXXV",
        "XXXVI",
        "XXXVII",
        "XXXVIII",
        "XXXIX",
        "XL",
        "XLI",
        "XLII",
        "XLIII",
        "XLIV",
        "XLV",
        "XLVI",
        "XLVII",
        "XLVIII",
        "XLIX",
        "L",
        "LI",
        "LII",
        "LIII",
        "LIV",
        "LVVVVV",
        "LVI",
        "LVII",
        "LVIII",
        "LIX",
        "LX"
    ]

    ROMAN_FULL_TO_HALF = [
        ("Ｌ", "L"),
        ("Ｘ", "X"),
        ("Ｖ", "V"),
        ("Ｉ", "I")
    ]

    ROMAN_PREFIX = [
        "Chapter",
        "Article",
        "Section",
        "Volume",
        "Book",
        "Part",
        "Phase",
        "Level",
        "Scheme",
        "Alternative",
        "Proposal",
        "Stage",
        "Line",
        "Figure",
        "Table",
        "Attachment",
        "Appendix",
        "Annex",
        "Figure",
        "Schedule",
        "Paragraph",
        "Item",
        "Subitem",
        "Edition",
        "Zone",
        "Team",
        "Grade",
        "Zones"
    ]

    THIRD_PARTY_LANG = open("assets/third_party_lang.txt").read()

    def __init__(self, data, src_colomn, tgt_colomn, src_lang, tgt_lang):
        """初始化

        Args:
            data (pd.Dataframe): pd.Dataframe对象，其中包含 src_colomn, tgt_colomn的指定列
            src_colomn (str): 包含平行语料原文的列
            tgt_colomn (str): 包含平行语料译文的列
            src_lang ([type]): 原文语言，zh为中文，en为英文
            tgt_lang ([type]): 译文语言，zh为中文，en为译文
        """
        if isinstance(data, str):
            self.data = pd.read_excel(data)
        else:
            self.data = data

        self.src_colomn = src_colomn
        self.tgt_colomn = tgt_colomn
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

        self.rubbish = pd.DataFrame()
        self.modified = pd.DataFrame()

        self.regx3rdlang = re.compile("[{}]".format(self.THIRD_PARTY_LANG))

    def deduplicate(self):
        """以原文为key值对语料进行去重
        """
        self.data.drop_duplicates(
            [self.src_colomn], inplace=True)
        return self

    def drop_non_text(self):
        """舍弃Dataframe中的非字符串类型数据
        """
        tqdm.pandas(desc="filter non text.")
        drop = self.data.progress_apply(lambda x: not isinstance(
            x[self.src_colomn], str) or not isinstance(x[self.tgt_colomn], str), axis=1)
        keep = ~drop
        keep_data = self.data[keep]
        drop_data = self.data[drop]
        drop_data["Drop Reason"] = "Non text"
        self.data = keep_data
        self.rubbish = pd.concat([self.rubbish, drop_data])
        return self

    def filter_3rdlang(self):
        """过滤包含除原文译文外第三方语言的语料
        """
        tqdm.pandas(desc="filter third party lang.")

        def func(x):
            if self.regx3rdlang.search(x[self.src_colomn]) or self.regx3rdlang.search(self.tgt_colomn):
                return True
            else:
                return False

        drop = self.data.progress_apply(func, axis=1)
        keep = ~drop
        keep_data = self.data[keep]
        drop_data = self.data[drop]
        drop_data["Drop Reason"] = "ThirdPartyLang"
        self.data = keep_data
        self.rubbish = pd.concat([self.rubbish, drop_data])
        return self

    def filter_garbled(self, garbled_dict="assets/Garbled.txt"):
        """过滤包含乱码的语料

        Args:
            garbled_dict (str, optional): 包含乱码词语的文件路径. Defaults to "Garbled.txt".
        """
        logging.info("Starting filter garbled.")
        garbled = open(garbled_dict).read().split()
        regx = re.compile("({})".format("|".join(garbled)))

        def func(x):
            if regx.findall(x[self.src_colomn]) or regx.findall(x[self.tgt_colomn]):
                return True
            else:
                return False

        tqdm.pandas(desc="filter garbled.")
        drop = self.data.progress_apply(func, axis=1)
        keep = ~drop
        keep_data = self.data[keep]
        drop_data = self.data[drop]
        drop_data["Drop Reason"] = "Garbled"
        self.data = keep_data
        self.rubbish = pd.concat([self.rubbish, drop_data])
        return self

    def replace_roman(self):
        """替换语料中的罗马字符为中文字符
        """
        mapping = {value: str(i+1) for i, value in enumerate(self.ROMAN)}
        mapping.update({value.lower(): str(i+1)
                        for i, value in enumerate(self.ROMAN)})
        roman = sorted(self.ROMAN)
        roman.reverse()

        roman_prefix = self.ROMAN_PREFIX + \
            [i.lower() for i in self.ROMAN_PREFIX]
        roman_prefix_lens = set([len(i) for i in roman_prefix])
        roman_prefix_group = []
        for l in roman_prefix_lens:
            prefixes = [i+" " for i in roman_prefix if len(i) == l]
            r = re.compile("(?<={})({})".format(
                "|".join(prefixes), "|".join(roman)))
            roman_prefix_group.append(r)

        roman.remove("XX")
        roman.remove("XXX")
        roman.remove("LX")

        roman_len1 = [i for i in roman if len(i) == 1]
        roman_len2 = [i for i in roman if len(i) > 1]
        roman_lower = [i.lower() for i in roman]
        roman_lower1 = [i for i in roman_lower if len(i) == 1]
        roman_lower2 = [i for i in roman_lower if len(i) > 1]

        regx1 = re.compile(
            "(?<![A-Z])({})(?![A-Z])".format("|".join(roman_len2)))
        # regx2 = re.compile(
        #     "(?<![A-Za-z0-9\"])({})(?! am| will|[A-Za-z0-9\"])".format("|".join(roman_len1)))
        regx3 = re.compile("(?<=[(（])({})(?=[)）])".format(
            "|".join(roman + roman_lower)))
        regx4 = re.compile("^({})(?=[. ])".format(
            "|".join(roman_len2 + roman_lower2 + roman_len1 + roman_lower1)))
        regx5 = re.compile("^({})".format("|".join(roman_lower2)))

        def replace_one(text):
            for ori, sub in self.ROMAN_FULL_TO_HALF:
                text = text.replace(ori, sub)
            for regx in [regx1, regx3, regx4, regx5] + roman_prefix_group:
                while True:
                    obj = regx.search(text)
                    if not obj:
                        break
                    start, end = obj.span()
                    text = text[:start] + mapping[text[start:end]] + text[end:]
            return text

        backup_src = []
        backup_tgt = []

        def func(x):
            src = x[self.src_colomn]
            tgt = x[self.tgt_colomn]

            ori_src, ori_tgt = src, tgt

            src = replace_one(src)
            tgt = replace_one(tgt)

            if ori_src != src or ori_tgt != tgt:

                backup_src.append(src)
                backup_tgt.append(tgt)
                return False
            else:
                return True
        tqdm.pandas(desc="filter roman.")
        keep = self.data.progress_apply(func, axis=1)
        modified = ~keep
        keep_data = self.data[keep]
        modified_data = self.data[modified]
        modified_data["modified reason"] = "RomanNumerals"
        modified_data[self.src_colomn] = backup_src
        modified_data[self.tgt_colomn] = backup_tgt
        self.data = keep_data
        self.modified = pd.concat([self.modified, modified_data])
        return self

    def normalize_punc(self):
        """对原文和译文的全角半角字符进行转换，中文半角->转全角，拉丁语系全角->半角
        """
        tqdm.pandas(desc="normalize punc")

        backup_src = []
        backup_tgt = []

        def func(x):
            src = x[self.src_colomn]
            tgt = x[self.tgt_colomn]

            ori_src, ori_tgt = src, tgt
            if self.src_lang == "zh":
                replacement = self.ZHCN_PUNC_REPLACEMENT
            else:
                replacement = self.LATIN_PUNC_REPLACEMENT
            for ori, sub in replacement:
                src = src.replace(ori, sub)

            if self.tgt_lang == "zh":
                replacement = self.ZHCN_PUNC_REPLACEMENT
            else:
                replacement = self.LATIN_PUNC_REPLACEMENT

            for ori, sub in replacement:
                tgt = tgt.replace(ori, sub)

            if ori_src != src or ori_tgt != tgt:
                backup_src.append(src)
                backup_tgt.append(tgt)
                return False
            else:
                return True

        keep = self.data.progress_apply(func, axis=1)
        modified = ~keep
        keep_data = self.data[keep]
        modified_data = self.data[modified]
        modified_data["modified reason"] = "Punctuation"
        modified_data[self.src_colomn] = backup_src
        modified_data[self.tgt_colomn] = backup_tgt
        self.data = pd.concat([keep_data, modified_data])
        self.modified = pd.concat([self.modified, modified_data])
        return self

    def align_brackets(self):
        """
        对齐括号，书名号。括号可以统计原文和译文书名号的数量，如果数量不对齐的可以扔掉。书名号检查中文部分书名号是否闭合。
        """
        # TODO
        tqdm.pandas(desc="align_brackets")
        regx_bracket = re.compile("[(（]")
        regx_anti_bracket = re.compile("[)）]")
        regx_book_title_mark = re.compile("《")
        regx_anti_book_title_mark = re.compile("》")

        def func(x):
            src = x[self.src_colomn]
            tgt = x[self.tgt_colomn]

            if len(regx_bracket.findall(src)) != len(regx_anti_bracket.findall(tgt)):
                return False

            if self.src_lang == "zh":
                if len(regx_book_title_mark.findall(src)) != len(regx_book_title_mark.findall(src)):
                    return False

            if self.tgt_lang == "zh":
                if len(regx_book_title_mark.findall(tgt)) != len(regx_book_title_mark.findall(tgt)):
                    return False

            return True

        drop = self.data.progress_apply(func, axis=1)
        keep = ~drop
        keep_data = self.data[keep]
        drop_data = self.data[drop]
        drop_data["Drop Reason"] = "BracketsAlignment"
        self.data = keep_data
        self.rubbish = pd.concat([self.rubbish, drop_data])
        return self

    def align_end_punc(self):
        """处理原文译文尾部标点符号不一致的问题
        """

        if self.src_lang == "zh" and self.tgt_lang != "zh":
            mapping = self.END_PUNC_MAPPING

        elif self.src_lang != "zh" and self.tgt_lang == "zh":
            mapping = {v: k for k, v in self.END_PUNC_MAPPING.items()}

        elif self.src_lang != "zh" and self.tgt_lang != "zh":
            mapping = {v: v for k, v in self.END_PUNC_MAPPING.items()}

        else:
            mapping = {k: k for k, v in self.END_PUNC_MAPPING.items()}

        reverse_mappping = {v: k for k, v in mapping.items()}

        backup_src = []
        backup_tgt = []

        puncs = [i[0] for i in self.LATIN_PUNC_REPLACEMENT] + [i[1]
                                                               for i in self.LATIN_PUNC_REPLACEMENT]

        def func(x):
            src = x[self.src_colomn].strip()
            tgt = x[self.tgt_colomn].strip()

            if len(src) == 0 or len(tgt) == 0:
                print(x["id"], "strip后长度为0")
                return True

            ori_src, ori_tgt = src, tgt
            if src[-1] in mapping and tgt[-1] not in reverse_mappping:
                if tgt[-1] not in puncs:
                    tgt += mapping[src[-1]]
                else:
                    tgt = tgt[:-1] + mapping[src[-1]]

            elif tgt[-1] in reverse_mappping and src[-1] not in mapping:
                if src[-1] not in puncs:
                    src += reverse_mappping[tgt[-1]]
                else:
                    src = src[:-1] + reverse_mappping[tgt[-1]]

            elif tgt[-1] in reverse_mappping and src[-1] in mapping:
                tgt = tgt[:-1] + mapping[src[-1]]

            # 中文最后为句号时的补丁
            if self.src_lang == "zh" and src[-1] == '.':
                src = src[:-1] + "。"
            if self.tgt_lang == "zh" and tgt[-1] == '.':
                tgt = tgt[:-1] + "。"

            if ori_src != src or ori_tgt != tgt:
                backup_src.append(src)
                backup_tgt.append(tgt)
                return False
            else:
                return True

        tqdm.pandas(desc="align end punc")
        keep = self.data.progress_apply(func, axis=1)
        modified = ~keep
        keep_data = self.data[keep]
        modified_data = self.data[modified]
        modified_data["modified reason"] = "EndPunctuation"
        modified_data[self.src_colomn] = backup_src
        modified_data[self.tgt_colomn] = backup_tgt
        self.data = pd.concat([keep_data, modified_data])
        self.modified = pd.concat([self.modified, modified_data])
        return self

    def filter_too_long(self, threshold=100, use_src_colomn=True):
        """过滤字符长度大于某阈值的语料

        Args:
            threshold (int, optional): 字符长度阈值. Defaults to 100.
            use_src_colomn (bool, optional): 是否以原文的长度作为基准。如果该值未False则以译文的长度为基准。Defaults to True.
        """
        tqdm.pandas(desc="filter too long")
        colomn = self.src_colomn if use_src_colomn else self.tgt_colomn
        drop = self.data.progress_apply(
            lambda x: len(x[colomn]) > threshold, axis=1)
        keep = ~drop
        keep_data = self.data[keep]
        drop_data = self.data[drop]
        drop_data["Drop Reason"] = "SentenceTooLong"
        self.data = keep_data
        self.rubbish = pd.concat([self.rubbish, drop_data])
        return self

    def filter_too_short(self, threshold=15, use_src_colomn=True):
        """过滤字符长度小于阈值的语料

        Args:
            threshold (int, optional): 字符长度阈值. Defaults to 15.
            use_src_colomn (bool, optional): 是否以原文的长度作为基准。如果该值未False则以译文的长度为基准. Defaults to True.
        """
        tqdm.pandas(desc="filter too short")
        colomn = self.src_colomn if use_src_colomn else self.tgt_colomn
        drop = self.data.progress_apply(
            lambda x: len(x[colomn]) < threshold, axis=1)
        keep = ~drop
        keep_data = self.data[keep]
        drop_data = self.data[drop]
        drop_data["Drop Reason"] = "SentenceTooShort"
        self.data = keep_data
        self.rubbish = pd.concat([self.rubbish, drop_data])
        return self


    def first_clean_rules(self):
        """
        第一次机清规则（历史遗留）
        """
        if self.tgt_colomn == "zh":
            src_column, tgt_column = self.tgt_colomn, self.src_colomn
        else:
            src_column, tgt_column = self.src_colomn, self.tgt_colomn

        re_han = re.compile("([\u4E00-\u9FD5]+)")
        re_en = re.compile("([A-Za-z])")
        re_blank = re.compile("\\s")

        def filter_func(line):

            src = line[src_column]
            tgt = line[tgt_column]

            # 原文译文相同
            if src == tgt:
                return True

            # 英文中有中文
            if re_han.search(tgt):
                return True

            # 译文长度大于原文，必然是没有对齐的平行语料
            if len(tgt) <= len(src):
                return True

            # 原文中英文字符数比中文字符数多
            if len(re_en.findall(src)) / len(src) > 0.5:
                return True

            # 空格占据英文或者中文字符的40%以上
            if len(re_blank.findall(tgt)) / len(tgt) > 0.4 or len(re_blank.findall(src)) / len(src) > 0.4:
                return True

            return False

        tqdm.pandas(desc="first clean rules.")
        drop = self.data.progress_apply(filter_func, axis=1)
        keep = ~drop
        keep_data = self.data[keep]
        drop_data = self.data[drop]
        drop_data["Drop Reason"] = "FisrtCleanRules"
        self.data = keep_data
        self.rubbish = pd.concat([self.rubbish, drop_data])
        return self