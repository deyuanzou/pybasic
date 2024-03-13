import os

import basic

while True:
    text = input('basic > ')
    if text.startswith('$'):
        os.system(text.lstrip('$'))
    else:
        result, error = basic.run(text)
        if error:
            print(error.as_string())
        else:
            print(result)
