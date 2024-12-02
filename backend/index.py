from flask import Flask, request, make_response, Response, jsonify
import json
import time
from chat import answer
from flask_restful import Api

import datetime

app = Flask(__name__)
api = Api(app)
    
@app.route('/api/ask', methods=['POST'])
def get_prompt_answer():
    data = json.loads(request.data)
    prompt = data.get("prompt")
    # history = data.get("history")
    print(prompt)

    if prompt == "":
        return make_response("Empty prompt", 400)

    try:
        response, data, extra = answer(prompt)
    except:
        return make_response("AI resources unavailable", 404)

                
    return jsonify({"response": response}), 200


if __name__ == '__main__':
    app.run(
        host='0.0.0.0', port=5000, debug=True)
