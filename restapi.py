# -*- coding: utf-8 -*-
import json
import requests

REQUEST_URL = "http://10.111.30.195:8181/api/v1.0/ltps"
HEADER = {'Content-Type':'application/json; charset=utf-8'}

def ltp_service(text, cmd=['seg']):
    requestDict = {'text':text, 'cmd':cmd}
    rsp = requests.post(REQUEST_URL, data=json.dumps(requestDict), headers=HEADER)
    if rsp.status_code == 201:
        rt =  json.loads(rsp.text)
        return rt['ltp']

"""
f = open("my.txt", "r", encoding='utf-8')
while True:
    line = f.readline()
    if len(line)==0:
        break
    _s = ltp_service(line)
    print(_s)
"""

#

