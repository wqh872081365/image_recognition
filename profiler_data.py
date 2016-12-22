# -*- coding: utf-8 -*-

import pstats
import os


file_list = os.listdir('./work_prof')
# 可以得到时间
for file in file_list:
    p = pstats.Stats('work_prof/'+file)
    p.strip_dirs().sort_stats('cumtime').print_stats(10, 1.0, '.*')

# p.strip_dirs().sort_stats('cumtime').print_stats()

# 可视化
# gprof2dot -f pstats prof/mkm_run.prof | dot -Tpng -o prof/mkm_run.png

# KCacheGrind & pyprof2calltree
# pyprof2calltree -i prof/mkm_run.prof -k  # 转换格式并立即运行KCacheGrind

# 得到运行脚本代码
"""
file_list = os.listdir('./work_prof')
for file in file_list:
    name = file.split('.')[0]+'.png'
    print 'gprof2dot -f pstats work_prof/'+file+' | dot -Tpng -o work_prof_png/'+name
"""

# KCacheGrind & pyprof2calltree
# pyprof2calltree -i work_prof/uploadphoto_run.prof -k  # 转换格式并立即运行KCacheGrind
# pyprof2calltree -i work_prof/resetcode_run.prof -k  # 转换格式并立即运行KCacheGrind

