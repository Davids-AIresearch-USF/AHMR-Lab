import requests
import json

def ask_ollama(url,data):
    """
    url [str]: url fo the query
    data [dict]: data dict with model name, prompt, ...
    """
    data_json = json.dumps(data)
    try:
        response = requests.post(url, data=data_json, headers={'Content-Type': 'application/json'})

        if response.status_code == 200:
            response_dict = response.json()
            return response_dict  
        
    # else:
    #     print(f"Request failed with status code {response.status_code}")
    except requests.RequestException as e:
        return f"[Error interacting with model: {e}]"



def get_output(model,msg,format=None):

    url = 'http://localhost:11434/api/chat'

    data = {
        "model": model,
        "messages": msg,
        "stream": False,
    }

    if format:
        data['format'] = format

    answer = ask_ollama(url,data)
    try:
        answer = answer['message']['content']
        return answer
    except:
        return(answer)
