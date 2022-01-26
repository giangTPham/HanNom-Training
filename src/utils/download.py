import requests

url = 'https://www.kaggle.com/dataset/89092c51429b20acec29171ddf6d30915dafabe6c79dd7dc2de58903f49683c6/download'

r = requests.get(url, allow_redirects=True)
name = r.headers.get('content-type')
open(name, 'wb').write(r.content)
kaggle datasets download -d banggiangle/chinesefont