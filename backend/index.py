from flask import Flask, request, make_response, Response
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
        output = answer(prompt)
    except:
        return make_response("AI resources unavailable", 404)

    def generate():
        for chunk in output:
            if chunk.choices and chunk.choices[0].delta.content:
                delta_content = chunk.choices[0].delta.content
                yield delta_content
                
    return Response(generate(), mimetype='text/event-stream') 


if __name__ == '__main__':
    app.run(
        host='0.0.0.0', port=5000, debug=True)