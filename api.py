# -*- coding:UTF-8 -*-
#

from flask import Flask, jsonify
from flask import request
import logging
import service

app = Flask(__name__)


@app.route('/api/v1.0/lstm', methods=['POST'])
def create_lstm():

    if not request.json or not 'text' in request.json:
        abort(400)

    # sentence = (request.json['text']).encode('utf-8')
    sentence = (request.json['text'])
    print(u'sentence: %s' % str(sentence))

    per, loc, org = service.do_lstm(str(sentence))

    return jsonify({'text': {'per': per, 'loc': loc, 'org': org}}), 201

if __name__ == '__main__':

    hdl = logging.FileHandler('api.log', encoding='UTF-8')
    hdl.setLevel(logging.DEBUG)
    app.logger.addHandler(hdl)

    app.run(host="0.0.0.0", port=8183, debug=False)


#
# Eof
#
