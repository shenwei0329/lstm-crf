# -*- coding: utf-8 -*-
import json
import requests

REQUEST_URL = "http://10.111.30.195:8183/api/v1.0/lstm"
HEADER = {'Content-Type':'application/json; charset=utf-8'}

def ltp_service(text):
    requestDict = {'text':text}
    rsp = requests.post(REQUEST_URL, data=json.dumps(requestDict), headers=HEADER)
    if rsp.status_code == 201:
        return json.loads(rsp.text)['text']
 
# 程序入口函数
if __name__=="__main__":

    rt = ltp_service(u"在四川省，案由成都市局成华区分局侦查终结，以被告人吴一凡、邓大江、张奎、吴谦涉嫌故意伤害罪，经成都市成华区人民检察院于2014年1月3日移送本院审查起诉。签约仪式前，秦光荣、李纪恒、仇和等一同会见了参加签约的企业家。")
    print(u"人员:")
    for _v in rt['per']:
        print(u"\t{}".format(_v))
    print(u"地点:")
    for _v in rt['loc']:
        print(u"\t{}".format(_v))
    print(u"组织:")
    for _v in rt['org']:
        print(u"\t{}".format(_v))

#

