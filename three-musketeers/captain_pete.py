from board import Board #line:1
from agent import Agent #line:2
import random #line:3
import pickle #line:4
import numpy as np #line:5
import math #line:6
import multiprocessing as mp #line:7
from math import sqrt ,log #line:8

class CaptainPete (Agent):#line:10
    def __init__ (O0O0OO00O0O00OO0O ,player =2 ):#line:11
        super ().__init__ (player )#line:12
        O0O0OO00O0O00OO0O.player =player #line:13
        O0O0OO00O0O00OO0O.musketeers =O0O0OO00O0O00OO0O.player %2 +1 #line:14
        O0O0OO00O0O00OO0O.lIIllIlllIlllIlIll =IlIlIIIIlIlIIlIIIl (player ,O0O0OO00O0O00OO0O.musketeers )#line:15
        O0O0OO00O0O00OO0O.llllllllllllllI =IlIIlllIIlIIIllIIl ()#line:16
    def next_action (O0O00OOOOOO0O0O00 ,O00O00OOOO0OOO00O ):#line:18
        OO00O0000O0O0000O =random.choices (["A","B","C"],weights =[0.5 ,0.4 ,0.1 ],k =1 )[0 ]#line:23
        O00OOOO000O000OOO =None #line:25
        if OO00O0000O0O0000O =="A":#line:27
            O00OOOO000O000OOO ,_OOO0O0O0O00O0OOOO =O0O00OOOOOO0O0O00.lIIllIlllIlllIlIll.lIIllIlllIlllIlIll (O00O00OOOO0OOO00O ,O0O00OOOOOO0O0O00.player , 3 , float ('-inf'),float ('inf'))#line:28
        elif OO00O0000O0O0000O =="B":#line:29
            O00OOOO000O000OOO =O0O00OOOOOO0O0O00.llllllllllllllI.sch (O00O00OOOO0OOO00O )#line:30
        else :#line:31
            O0OOO0OO0OOOO0OOO =O00O00OOOO0OOO00O.get_possible_actions (O0O00OOOOOO0O0O00.player )#line:32
            O00OOOO000O000OOO =random.choice (O0OOO0OO0OOOO0OOO )#line:33
        return O00OOOO000O000OOO #line:35
    def heuristic_utility (O0OO0000O0O000O0O ,O00OOO00O0000000O :Board ):#line:37
        pass #line:38
class IIllIllllIIlIIIlIl :#line:40
    def __init__ (OO00O0OO0000OO0OO ,O0O00O00000O0O000 ,p =None ,a =None ):#line:41
        OO00O0OO0000OO0OO.s =O0O00O00000O0O000 #line:42
        OO00O0OO0000OO0OO.p =p #line:43
        OO00O0OO0000OO0OO.a =a #line:44
        OO00O0OO0000OO0OO.c =[]#line:45
        OO00O0OO0000OO0OO.v =0 #line:46
        OO00O0OO0000OO0OO.w =0 #line:47
        OO00O0OO0000OO0OO.ua =O0O00O00000O0O000.get_possible_actions (2 )#line:48
    def add_child (O00O0000OO0O000OO ,OOOO000O000000OOO ):#line:50
        O00O0000OO0O000OO.c.append (OOOO000O000000OOO )#line:51
    def ife (O000OO0O00OOOO000 ):#line:53
        return len (O000OO0O00OOOO000.ua )==0 #line:54
    def bcbcbc (OO0OOO0O00O0O0000 ,ew =9.4 ):#line:56
        OO0O0O0OOOO0OO000 =np.array ([OO0OO000O00O00OO0.v for OO0OO000O00O00OO0 in OO0OOO0O00O0O0000.c ])#line:57
        OOO000O0OOOOOO000 =np.array ([O0OO0OOOO0OO00O00.w for O0OO0OOOO0OO00O00 in OO0OOO0O00O0O0000.c ])#line:58
        O00O0OO00OOO00O0O =sqrt (OO0OOO0O00O0O0000.v )#line:59
        OOOOOOO000000000O =OOO000O0OOOOOO000 /(OO0O0O0OOOO0OO000 +1e-6 )+ew *np.sqrt (log (OO0OOO0O00O0O0000.v )/(OO0O0O0OOOO0OO000 +1 ))#line:61
        OO00000OO0O00OO0O =np.argmax (OOOOOOO000000000O )#line:62
        return OO0OOO0O00O0O0000.c [OO00000OO0O00OO0O ]#line:63
class IlIIlllIIlIIIllIIl :#line:65
    def __init__ (O000OO0000O000O00 ,max_iterations =100 ):#line:66
        O000OO0000O000O00.max_iterations =max_iterations #line:67
        O000OO0000O000O00.sc ={}#line:68
    def sch (OOOO0O00000OO0000 ,OO000O00O00O0O00O ):#line:70
        OO00O000OO0OO0000 =IIllIllllIIlIIIlIl (OO000O00O00O0O00O )#line:71
        for _O00000O0OOOO00O0O in range (OOOO0O00000OO0000.max_iterations ):#line:73
            O0OOOO0O00OOO0O0O =OOOO0O00000OO0000._sel (OO00O000OO0OO0000 )#line:74
            OO0OOOOO00O0O0OO0 =OOOO0O00000OO0000._sim (O0OOOO0O00OOO0O0O.s )#line:75
            OOOO0O00000OO0000._backpropagate (O0OOOO0O00OOO0O0O ,OO0OOOOO00O0O0OO0 )#line:76
        return OO00O000OO0OO0000.bcbcbc (0 ).a #line:78
    def _sel (OOO000O00O00OO000 ,O0OO00OO0O00OO0O0 ):#line:80
        while O0OO00OO0O00OO0O0.ife ()and O0OO00OO0O00OO0O0.c :#line:81
            O0OO00OO0O00OO0O0 =O0OO00OO0O00OO0O0.bcbcbc ()#line:82
        if not O0OO00OO0O00OO0O0.ife ():#line:84
            O0O0OO0000O000OO0 =O0OO00OO0O00OO0O0.ua.pop ()#line:85
            O0O00OOO00000O000 =O0OO00OO0O00OO0O0.s.clone ()#line:86
            O0O00OOO00000O000.play (2 ,O0O0OO0000O000OO0 )#line:87
            OO0O0OO0OOOOOO000 =IIllIllllIIlIIIlIl (O0O00OOO00000O000 ,p =O0OO00OO0O00OO0O0 ,a =O0O0OO0000O000OO0 )#line:88
            O0OO00OO0O00OO0O0.add_child (OO0O0OO0OOOOOO000 )#line:89
            return OO0O0OO0OOOOOO000 #line:90
        return O0OO00OO0O00OO0O0 #line:91
    def _sim (O0O000OOOO0O0OOOO ,O000O0000000OO000 ):#line:93
        OOO0OOO0OO000OOO0 =hash (O000O0000000OO000 )#line:94
        if OOO0OOO0OO000OOO0 in O0O000OOOO0O0OOOO.sc :#line:95
            return O0O000OOOO0O0OOOO.sc [OOO0OOO0OO000OOO0 ]#line:96
        O0O000O0O00OOOOOO =2 #line:98
        while True :#line:99
            OO0OO000OOO0OOOO0 ,OO00O00O000OO0OOO =O000O0000000OO000.is_end (O0O000O0O00OOOOOO )#line:100
            if OO0OO000OOO0OOOO0 :#line:101
                O0O000OOOO0O0OOOO.sc [OOO0OOO0OO000OOO0 ]=OO00O00O000OO0OOO #line:102
                return OO00O00O000OO0OOO #line:103
            OOOOO000O000OO0OO =O000O0000000OO000.get_possible_actions (O0O000O0O00OOOOOO )#line:105
            if not OOOOO000O000OO0OO :#line:106
                break #line:107
            OO0O00OO0000OOO0O =O0O000OOOO0O0OOOO._p (O000O0000000OO000 ,OOOOO000O000OO0OO )#line:109
            O000O0000000OO000.play (O0O000O0O00OOOOOO ,OO0O00OO0000OOO0O )#line:110
            O0O000O0O00OOOOOO =1 if O0O000O0O00OOOOOO ==2 else 2 #line:111
    def _p (O0OO0OOOO000OO000 ,OOO000O0O0OO00OOO ,O0O00OOO0OOOOOOOO ):#line:113
        O0000000OO00OO0O0 =None #line:114
        O000O0OO0OOO0OO00 =float ('inf')#line:115
        for OO0OO00O000O0O0O0 in O0O00OOO0OOOOOOOO :#line:117
            O000O0OOOO000OO00 =OOO000O0O0OO00OOO.clone ()#line:118
            O000O0OOOO000OO00.play (2 ,OO0OO00O000O0O0O0 )#line:119
            O0OO0OO000000OOOO =O000O0OOOO000OO00.find_musketeer_positions ()#line:121
            O000000000000OOO0 =0 #line:122
            O0OOO00O0OO000O0O =len (O0OO0OO000000OOOO )#line:123
            for O00OOO0OOOOOO0O0O in range (O0OOO00O0OO000O0O ):#line:125
                for O00OOO0O000O00O00 in range (O00OOO0OOOOOO0O0O +1 ,O0OOO00O0OO000O0O ):#line:126
                    O00O0O00000OOO0OO =abs (O0OO0OO000000OOOO [O00OOO0OOOOOO0O0O ][0 ]-O0OO0OO000000OOOO [O00OOO0O000O00O00 ][0 ])+abs (O0OO0OO000000OOOO [O00OOO0OOOOOO0O0O ][1 ]-O0OO0OO000000OOOO [O00OOO0O000O00O00 ][1 ])#line:128
                    O000000000000OOO0 +=O00O0O00000OOO0OO #line:129
            if O000000000000OOO0 <O000O0OO0OOO0OO00 :#line:131
                O000O0OO0OOO0OO00 =O000000000000OOO0 #line:132
                O0000000OO00OO0O0 =OO0OO00O000O0O0O0 #line:133
        return O0000000OO00OO0O0 #line:135
    def _backpropagate (OOO0OOO0OO0O000OO ,O000OO0O0O0OO0OOO ,O00000OO00O0OOOO0 ):#line:138
        while O000OO0O0O0OO0OOO is not None :#line:139
            O000OO0O0O0OO0OOO.v +=1 #line:140
            if O00000OO00O0OOOO0 ==2 :#line:141
                O000OO0O0O0OO0OOO.w +=1 #line:142
            O000OO0O0O0OO0OOO =O000OO0O0O0OO0OOO.p #line:143
class IlIlIIIIlIlIIlIIIl :#line:145
    def __init__ (OOO000000O0OOOOOO ,O0O00OO000O00O000 ,O000000O0O00OOO00 ):#line:146
        OOO000000O0OOOOOO.player =O0O00OO000O00O000 #line:147
        OOO000000O0OOOOOO.musketeers =O000000O0O00OOO00 #line:148
    def heuristic_utility (OO00O00OOO0OO0OOO ,O00O00OO00O00O0O0 :Board ):#line:150
        return 3 *OO00O00OOO0OO0OOO.evma (O00O00OO00O00O0O0 )+OO00O00OOO0OO0OOO.mpa (O00O00OO00O00O0O0 )+2 *OO00O00OOO0OO0OOO.evaluate_musketeers_proximity_to_center (O00O00OO00O00O0O0 )+OO00O00OOO0OO0OOO.evdtn (O00O00OO00O00O0O0 )#line:151
    def evdtn (O0OO0O00OOO00O0OO ,O00000O0OOO0OO000 ):#line:153
        OO0O0OOOOOOO0OOO0 =O00000O0OOO0OO000.find_musketeer_positions ()#line:154
        OO0000O00O0000O00 =O00000O0OOO0OO000.find_enemy_positions ()#line:155
        OO00O00OO00O00000 =0 #line:157
        for O00O0000000O0OOO0 in OO0O0OOOOOOO0OOO0 :#line:159
            for OOOO0O0OO0OOO0O0O in OO0000O00O0000O00 :#line:160
                OO00O00OO00O00000 +=1 /(abs (O00O0000000O0OOO0 [0 ]-OOOO0O0OO0OOO0O0O [0 ])+abs (O00O0000000O0OOO0 [1 ]-OOOO0O0OO0OOO0O0O [1 ])+1 )#line:161
        return -OO00O00OO00O00000 #line:163
    def evaluate_musketeers_proximity_to_center (OOOO0OOO0O0OO0O00 ,O0O000O0O00O0OO00 ):#line:165
        OOO0OOO0OO00O0O0O =(O0O000O0O00O0OO00.board_size [0 ]//2 ,O0O000O0O00O0OO00.board_size [0 ]//2 )#line:166
        def OOOO0OO0OOOOOO00O (OO0000O0O0OOOOO0O ):#line:168
            return abs (OO0000O0O0OOOOO0O [0 ]-OOO0OOO0OO00O0O0O [0 ])+abs (OO0000O0O0OOOOO0O [1 ]-OOO0OOO0OO00O0O0O [1 ])#line:169
        O0OO00OO00OO000O0 =0 #line:171
        O00OO0O00O000OOO0 =O0O000O0O00O0OO00.find_musketeer_positions ()#line:172
        for O0OOOO0O00OOO0000 in O00OO0O00O000OOO0 :#line:174
            O0OO00OO00OO000O0 +=OOOO0OO0OOOOOO00O (O0OOOO0O00OOO0000 )#line:175
        return -O0OO00OO00OO000O0 #line:177
    def mpa (O00000OO0O000O000 ,O00O0OOOO0OOOO0OO :Board ):#line:179
        return len (O00O0OOOO0OOOO0OO.get_possible_actions (O00000OO0O000O000.musketeers ))#line:180
    def epa (O0000OOO0OO000O0O ,O0OO00OO000O0OOOO :Board ):#line:182
        return len (O0OO00OO000O0OOOO.get_possible_actions (O0000OOO0OO000O0O.player ))#line:183
    def emc (OO0O00000O0OO0OOO ,O0OO0O0OO0OO000O0 ):#line:185
        O0O00OOO00OOOO00O =O0OO0O0OO0OO000O0.find_musketeer_positions ()#line:186
        OOOO0OOO00OO0OOO0 =0 #line:188
        O0000OOO00OOO0OOO =len (O0O00OOO00OOOO00O )#line:189
        for O0OO00O0O0OOOOO00 in range (O0000OOO00OOO0OOO ):#line:191
            for O00O0000OO0OOO00O in range (O0OO00O0O0OOOOO00 +1 ,O0000OOO00OOO0OOO ):#line:192
                OOO00OOO000OO0OO0 =abs (O0O00OOO00OOOO00O [O0OO00O0O0OOOOO00 ][0 ]-O0O00OOO00OOOO00O [O00O0000OO0OOO00O ][0 ])+abs (O0O00OOO00OOOO00O [O0OO00O0O0OOOOO00 ][1 ]-O0O00OOO00OOOO00O [O00O0000OO0OOO00O ][1 ])#line:194
                OOOO0OOO00OO0OOO0 +=1 /(OOO00OOO000OO0OO0 +1 )#line:195
        return OOOO0OOO00OO0OOO0 #line:197
    def evma (O0OO000O0000OO000 ,OO000OO0000OO0OO0 ):#line:199
        O000000000O0O000O =OO000OO0000OO0OO0.find_musketeer_positions ()#line:200
        OOOO00000OOOOOO0O =[O000O0OOO00OOOOO0 [0 ]for O000O0OOO00OOOOO0 in O000000000O0O000O ]#line:202
        OO00O0000O00O0OO0 =[O0000OO0O0OOO0OOO [1 ]for O0000OO0O0OOO0OOO in O000000000O0O000O ]#line:203
        O00O0O0O0O0O000OO =len (set (OO00O0000O00O0OO0 ))#line:205
        O00O0O00O00OOOOOO =len (set (OOOO00000OOOOOO0O ))#line:206
        if O00O0O0O0O0O000OO ==1 or O00O0O00O00OOOOOO ==1 :#line:208
            return 10 #line:209
        if O00O0O0O0O0O000OO ==2 or O00O0O00O00OOOOOO ==2 :#line:211
            return 5 #line:212
        return 0 #line:214
    def lIIllIlllIlllIlIll (OOO0O0O0OO00O0000 ,O0O0OOOOO00OO00OO :Board ,OO0000OOO0000000O :int ,OO0OOOO0OO00OO0OO :int ,O0OO0000O00OOO0OO :int ,OOOO00OOOO0OOO000 :int ):#line:216
        O0000OOOO0OO00O0O ,OOOO0000O00O0O000 =O0O0OOOOO00OO00OO.is_end (OO0000OOO0000000O )#line:217
        if O0000OOOO0OO00O0O :#line:218
            if OOOO0000O00O0O000 ==OOO0O0O0OO00O0000.player :#line:219
                return None ,1 #line:220
            elif OOOO0000O00O0O000 ==OOO0O0O0OO00O0000.player %2 +1 :#line:221
                return None ,-1 #line:222
            else :#line:223
                return None ,0 #line:224
        if OO0OOOO0OO00OO0OO ==0 :#line:226
            return None ,OOO0O0O0OO00O0000.heuristic_utility (O0O0OOOOO00OO00OO )#line:227
        OO0OO00OOO0OO00O0 =O0O0OOOOO00OO00OO.get_possible_actions (OOO0O0O0OO00O0000.player )#line:229
        O0OO0OO0O0O0O0O00 =[]#line:230
        for O0OOOOOO00O000O0O in OO0OO00OOO0OO00O0 :#line:231
            OOOO000O00OO0OO0O =O0O0OOOOO00OO00OO.clone ()#line:232
            OO000OOO00O0O0OOO =OOOO000O00OO0OO0O.play (OOO0O0O0OO00O0000.player ,O0OOOOOO00O000O0O )#line:233
            if not OO000OOO00O0O0OOO :#line:234
                raise Exception ("HFIHFIFIIFJIFJIJ")#line:235
            O0OO0OO0O0O0O0O00.append ((O0OOOOOO00O000O0O ,OOOO000O00OO0OO0O ))#line:236
        O000OOO0OO0OO00OO =None #line:238
        if OO0000OOO0000000O !=OOO0O0O0OO00O0000.player :#line:240
            OOOOO0OO00O0O000O =float ('inf')#line:241
            for O0OOOOOO00O000O0O ,O0O00OOOOOOO0O0OO in O0OO0OO0O0O0O0O00 :#line:242
                _O00OO00O0O00OOO00 ,O0OOO0O0000000000 =OOO0O0O0OO00O0000.lIIllIlllIlllIlIll (O0O00OOOOOOO0O0OO ,(OO0000OOO0000000O %2 )+1 ,OO0OOOO0OO00OO0OO -1 ,O0OO0000O00OOO0OO ,OOOO00OOOO0OOO000 )#line:243
                if O0OOO0O0000000000 <OOOOO0OO00O0O000O :#line:244
                    OOOOO0OO00O0O000O =O0OOO0O0000000000 #line:245
                    O000OOO0OO0OO00OO =O0OOOOOO00O000O0O #line:246
                OOOO00OOOO0OOO000 =min (OOOO00OOOO0OOO000 ,O0OOO0O0000000000 )#line:247
                if OOOO00OOOO0OOO000 <=O0OO0000O00OOO0OO :#line:248
                    break #line:249
            return O000OOO0OO0OO00OO ,OOOOO0OO00O0O000O #line:250
        else :#line:251
            O0O00OOOO00000O00 =float ('-inf')#line:252
            for O0OOOOOO00O000O0O ,O0O00OOOOOOO0O0OO in O0OO0OO0O0O0O0O00 :#line:253
                _O00OO00O0O00OOO00 ,O0OOO0O0000000000 =OOO0O0O0OO00O0000.lIIllIlllIlllIlIll (O0O00OOOOOOO0O0OO ,(OO0000OOO0000000O %2 )+1 ,OO0OOOO0OO00OO0OO -1 ,O0OO0000O00OOO0OO ,OOOO00OOOO0OOO000 )#line:254
                if O0OOO0O0000000000 >O0O00OOOO00000O00 :#line:255
                    O0O00OOOO00000O00 =O0OOO0O0000000000 #line:256
                    O000OOO0OO0OO00OO =O0OOOOOO00O000O0O #line:257
                O0OO0000O00OOO0OO =max (O0OO0000O00OOO0OO ,O0OOO0O0000000000 )#line:258
                if OOOO00OOOO0OOO000 <=O0OO0000O00OOO0OO :#line:259
                    break #line:260
            return O000OOO0OO0OO00OO ,O0O00OOOO00000O00 