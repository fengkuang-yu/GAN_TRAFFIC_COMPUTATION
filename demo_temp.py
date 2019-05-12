# -*- coding: utf-8 -*-

"""
@Author  :   {Yu Yinghao}
@Software:   PyCharm
@File    :   demo_temp.py
@Time    :   2019/5/10 21:46
@Desc    :
"""


# class Solution:
#     def findLength(self, A, B) -> int:
#         length = 0
#         for i in range(len(A)):
#             if i + length >= len(A):
#                 return length
#             j = 0
#             while j < len(B):
#                 if  j + length >= len(B):
#                     break
#                 while A[i:i + length + 1] == B[j:j + length + 1] and i + length + 1 <= min(len(B), len(A)):
#                     length += 1
#                     j += length - 1
#                 j += 1
#         return length
class Solution:
    def intToRoman(self, num: int) -> str:
        from collections import OrderedDict
        num_char_dict = OrderedDict[(1000, "M"), (900, "CM"), (500, "D"), (400, "CD"),
                                    (100, "C"), (90, "XC"), (50, "L"), (40, "XL"),
                                    (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I"),]
        res = ''
        for cur_num in num_char_dict:
            temp = num // cur_num
            while temp:
                res += num_char_dict[cur_num]
                temp -= 1
        return res


res = Solution()



temp_1 = [0,0,0,0,0,0,1,0,0,0]
temp_2 = [0,0,0,0,0,0,0,1,0,0]
res.findLength(temp_1, temp_2)